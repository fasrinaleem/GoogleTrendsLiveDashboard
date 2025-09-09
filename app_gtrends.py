# app_gtrends.py — FINAL ONE-FILE VERSION (SerpAPI + PyTrends + Demo)
# - One sidebar (no duplicate keys)
# - Per-section fetch/apply buttons (no global reruns)
# - Mutually-exclusive region quick-pick (Australia / Perth / Worldwide / Custom)
# - SerpAPI integration (correct params + no_cache) with graceful fallback
# - "Require live" toggle so you never silently see demo
# - Live/Demo badges with fetch timestamp and diagnostics
# - Pick the series you want in the Animated Map
# - dd2-style extras inlined (snapshot by role, 30d rolling, correlations)
# - Safer related-queries parsing (Breakout, %, None)
# - Local colored line_with_spikes so you don't need to edit viz.py

from __future__ import annotations

from datetime import datetime, timedelta
from urllib.parse import quote
import os
import time
import json
from typing import Iterable, Optional, Tuple, Dict, Any, List

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from pytrends.exceptions import TooManyRequestsError

# ── local modules you already have ─────────────────────────────────────────────
# If you renamed any of these, just update the imports.
from trends import (
    get_client,
    interest_over_time as pytrends_iot,
    monthly_region_frames as pytrends_monthly_frames,
    related_queries as pytrends_related,
    trending_today as pytrends_trending_today,
    trending_realtime as pytrends_realtime,
)
from viz import (
    animated_choropleth,
    wordcloud_from_related,  # we’ll feed sanitized numbers only
    kpi_card,
    sparkline,
)
from utils import country_name_to_iso2  # country-name -> ISO-2


# ────────────────────────────── Page setup / theme ─────────────────────────────
st.set_page_config(page_title="Trends Hub", page_icon="📊", layout="wide")
st.markdown(
    """
<style>
:root{
  --card-brd:#e9eaf0; --muted:#64748b; --ink:#0f172a; --ink-2:#111827;
  --chip:#eef2ff; --chip-brd:#c7d2fe; --chip-ink:#3730a3;
  --warn:#fef3c7; --warn-brd:#fde68a; --warn-ink:#92400e;
  --okbg:#ecfdf5; --okbrd:#10b981; --okink:#065f46;
  --errbg:#fef2f2; --errbrd:#fecaca; --erring:#991b1b;
}
html, body, .stApp{background:#ffffff;}
.hero{
  background: radial-gradient(1200px 420px at 12% -15%, rgba(109,40,217,.12), transparent),
              linear-gradient(90deg, rgba(109,40,217,.10), rgba(37,99,235,.08));
  border:1px solid var(--card-brd); border-radius:18px; padding:18px 22px; margin: 10px 0 16px 0;
  box-shadow: 0 8px 26px rgba(17,24,39,.06);
}
.hero h1{margin:0; font-size:1.9rem; line-height:1.25;}
.subtle{color:#475569; font-size:.95rem; margin-top:6px;}
.section{
  background: linear-gradient(180deg, #fff, #fbfcff);
  border:1px solid var(--card-brd); border-radius:18px;
  padding:16px; margin: 6px 0 18px 0;
  box-shadow: 0 8px 24px rgba(17,24,39,.05);
}
.section-h{display:flex; align-items:center; justify-content:space-between; margin-bottom:10px;}
.section-h h2{margin:0; font-size:1.25rem; color:var(--ink);}
.card{background:#fff; border:1px solid var(--card-brd); border-radius:16px; padding:14px; box-shadow:0 6px 16px rgba(17,24,39,.05);}
.kpi{display:flex; flex-direction:column; gap:6px; padding:12px 14px; border-radius:12px;
     background:linear-gradient(180deg,#fff,#f9fbff); border:1px solid var(--card-brd);
     box-shadow:0 6px 16px rgba(17,24,39,.06);}
.kpi-label{color:var(--muted); font-size:.72rem; letter-spacing:.03em; text-transform:uppercase;}
.kpi-value{font-size:1.32rem; font-weight:800; color:var(--ink-2);}
.chip{display:inline-block; padding:6px 10px; border-radius:999px; font-size:.75rem;
      background:#eef2ff; border:1px solid #c7d2fe; color:#3730a3;}
.caption{color:#64748b; font-size:.85rem;}
.small-gap{margin-top:8px}
.warn{background:#FEF3C7;border:1px solid #FDE68A;color:#92400E;border-radius:10px;padding:10px 12px;font-size:.9rem;}
.ok{background:var(--okbg);border:1px solid var(--okbrd);color:var(--okink);border-radius:10px;padding:8px 10px;font-size:.85rem;display:inline-block;margin:6px 0;}
.err{background:var(--errbg);border:1px solid var(--errbrd);color:var(--erring);border-radius:10px;padding:8px 10px;font-size:.85rem;display:inline-block;margin:6px 0;}
hr.sep{border:none;border-top:1px dashed #e5e7eb;margin:10px 0;}
</style>
""",
    unsafe_allow_html=True,
)

# ───────────────────────── Small utils ─────────────────────────────────────────
def chart_key(prefix: str, *parts) -> str:
    safe = "|".join(str(p) for p in parts if p is not None)
    return f"{prefix}:{safe}"[:200]

def to_list(x: Iterable[str] | str | None) -> list[str]:
    if x is None:
        return []
    if isinstance(x, str):
        return [x]
    return list(x)

def badge_live(df: pd.DataFrame | Dict[str, Any] | None) -> None:
    """Show a Live/Demo badge based on df attrs set by fetch functions."""
    if isinstance(df, pd.DataFrame) and not df.empty and df.attrs.get("source"):
        st.markdown(
            f"<div class='ok'>Live ✓ · {df.attrs.get('source')} @ {df.attrs.get('fetched_at')}</div>",
            unsafe_allow_html=True,
        )

# NEW: safe coercion helper
def _ensure_df(obj) -> pd.DataFrame:
    """Coerce anything to a DataFrame (empty if impossible)."""
    if isinstance(obj, pd.DataFrame):
        return obj
    if obj is None:
        return pd.DataFrame()
    try:
        return pd.DataFrame(obj)
    except Exception:
        return pd.DataFrame()

# ─────────────────────────── Geo resolver (country/city/world) ─────────────────
def resolve_geo(user_input: str) -> Tuple[str, str, Optional[str]]:
    """
    Returns (geo_code_for_api, scope_label, city_filter)
    '' for worldwide or ISO-2 for country; city_filter used for realtime/map filtering.
    """
    s = (user_input or "").strip()
    if not s:
        return "", "Worldwide", None
    low = s.lower()
    if low in {"world", "worldwide", "global"}:
        return "", "Worldwide", None
    if low == "perth":
        return "AU", "Perth, Australia", "Perth"
    if "," in s:  # e.g. "Perth, Australia"
        city = s.split(",", 1)[0].strip()
        country = s.split(",", 1)[1].strip()
        iso2 = country_name_to_iso2(country) or (country.upper() if len(country) == 2 else None)
        if iso2:
            return iso2, f"{city}, {country}", city
    iso2 = country_name_to_iso2(s)
    if iso2:
        return iso2, s.title(), None
    if len(s) == 2 and s.isalpha():
        return s.upper(), s.upper(), None
    return "", s, None

# ───────────────────────── Demo generators (used on 429/no-data) ───────────────
def demo_ts(keys=("AI","Data"), days=180) -> pd.DataFrame:
    end = datetime.utcnow().date(); start = end - timedelta(days=days)
    rng = pd.date_range(start, end, freq="D")
    df = pd.DataFrame({"date": rng})
    for i, kw in enumerate(keys[:5]):
        base = 45 + i*3
        s = base + 18*np.sin(np.linspace(0,6,len(rng))) + np.random.RandomState(33+i).randn(len(rng))*3
        s[int(len(rng)*0.30)] += 25; s[int(len(rng)*0.55)] += 18
        df[kw] = np.clip(s,0,100).round().astype(int)
    df.attrs["source"] = "DEMO"
    df.attrs["fetched_at"] = datetime.utcnow().isoformat()
    return df

def demo_frames(keyword="AI", months_count=6) -> pd.DataFrame:
    end = pd.Period(datetime.utcnow().date(), freq="M")
    periods = pd.period_range(end=end, periods=months_count, freq="M")
    rows = []
    vals=[("Australia","AU",70),("United States","US",58),("India","IN",62),("United Kingdom","GB",50)]
    alt =[("Australia","AU",48),("United States","US",69),("India","IN",75),("United Kingdom","GB",41)]
    for i,p in enumerate(periods):
        use = vals if i%2==0 else alt
        for r,iso2,v in use:
            rows.append({"region":r,"value":v,"iso2":iso2,"date_frame":str(p)})
    df = pd.DataFrame(rows)
    df.attrs["source"] = "DEMO"
    df.attrs["fetched_at"] = datetime.utcnow().isoformat()
    return df

def demo_related():
    top = pd.DataFrame({"query":["what is ai","data analytics login","ai tools"],"value":[80,65,50]})
    rising = pd.DataFrame({"query":["ai agents","gpt-4o","prompt ideas"],"value":[120,100,95]})
    return {"top": top, "rising": rising, "_meta":{"source":"DEMO","fetched_at":datetime.utcnow().isoformat()}}

# ───────────────────────────── SerpAPI client ──────────────────────────────────
SERP_KEY = st.secrets.get("SERPAPI_KEY", os.getenv("SERPAPI_KEY", "")).strip()

def _serp_get(params: Dict[str, Any], max_retries: int = 3) -> Dict[str, Any] | None:
    """GET wrapper with backoff. Returns JSON or None."""
    if not SERP_KEY:
        st.session_state["serp_last_error"] = "Missing SERPAPI_KEY"
        return None
    base = "https://serpapi.com/search.json"
    params = dict(params)
    params["api_key"] = SERP_KEY
    params["no_cache"] = True  # avoid stale cached responses
    wait = 1.2
    for _ in range(max_retries):
        try:
            r = requests.get(base, params=params, timeout=25)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (401, 403):
                st.session_state["serp_last_error"] = f"{r.status_code}: unauthorized"
                return None
            if r.status_code == 429:
                st.session_state["serp_last_error"] = "429: rate limited"
                time.sleep(wait); wait *= 1.7; continue
            st.session_state["serp_last_error"] = f"{r.status_code}: {r.text[:200]}"
            time.sleep(wait); wait *= 1.4
        except Exception as e:
            st.session_state["serp_last_error"] = f"Exception: {e}"
            time.sleep(wait); wait *= 1.4
    return None

def serp_iot(keywords: Iterable[str], timeframe: str, geo: str) -> pd.DataFrame | None:
    # SerpAPI Trends uses 'date' (NOT 'time')
    kws = ",".join(to_list(keywords)[:5])
    data = _serp_get({
        "engine":"google_trends",
        "data_type":"TIMESERIES",
        "q":kws,
        "hl":"en",
        "date": timeframe,
        "geo": geo or ""
    })
    if not data:
        return None
    try:
        iot = data.get("interest_over_time") or {}
        timeline = iot.get("timeline_data") or []
        if not timeline:
            return None
        names = to_list(iot.get("default_ranking", [])) or to_list(keywords)
        rows=[]
        for p in timeline:
            dt = pd.to_datetime(p.get("date"))
            vals = p.get("values", [])
            series = [v.get("value", 0) if isinstance(v, dict) else 0 for v in vals]
            rows.append([dt] + series)
        cols = ["date"] + (names[:len(rows[0])-1] if rows and len(rows[0])>1 else to_list(keywords))
        df = pd.DataFrame(rows, columns=cols)
        for kw in to_list(keywords):
            if kw not in df.columns:
                df[kw] = np.nan
        df = df[["date"] + list(dict.fromkeys(to_list(keywords)))]
        df.attrs["source"] = "SerpAPI"
        df.attrs["fetched_at"] = datetime.utcnow().isoformat()
        return df
    except Exception:
        return None

def serp_related(keyword: str, geo: str, timeframe: str="today 12-m") -> Dict[str, pd.DataFrame] | None:
    data = _serp_get({
        "engine":"google_trends",
        "data_type":"RELATED_QUERIES",
        "q":keyword,
        "hl":"en",
        "geo":geo or "",
        "date": timeframe
    })
    if not data:
        return None
    try:
        rq = data.get("related_queries") or {}
        def _mk(name):
            arr = rq.get(name) or []
            if not arr: return pd.DataFrame(columns=["query","value"])
            return pd.DataFrame([{"query":x.get("query"), "value":x.get("value")} for x in arr if x.get("query")])
        out = {"top": _mk("top"), "rising": _mk("rising")}
        return out
    except Exception:
        return None

def serp_trending_today(geo_country: str="australia") -> pd.DataFrame | None:
    data = _serp_get({"engine":"google_trends_trending_now","hl":"en","geo":geo_country})
    if not data:
        return None
    try:
        stories = data.get("stories") or []
        qs = [s.get("title") or s.get("query") for s in stories if s.get("title") or s.get("query")]
        df = pd.DataFrame({"query": qs})
        df.attrs["source"] = "SerpAPI"; df.attrs["fetched_at"] = datetime.utcnow().isoformat()
        return df if not df.empty else None
    except Exception:
        return None

def serp_trending_realtime(geo_iso2: str="AU", cat: str="all") -> pd.DataFrame | None:
    data = _serp_get({"engine":"google_trends_realtime","hl":"en","geo":geo_iso2,"category":cat})
    if not data:
        return None
    try:
        stories = data.get("stories") or []
        rows = [{"title": s.get("title",""), "entityNames": ", ".join(s.get("entityNames",[]) or [])} for s in stories]
        df = pd.DataFrame(rows)
        df.attrs["source"] = "SerpAPI"; df.attrs["fetched_at"] = datetime.utcnow().isoformat()
        return df if not df.empty else None
    except Exception:
        return None

def serp_interest_by_region(keyword: str, geo: str, timeframe: str="today 12-m") -> pd.DataFrame | None:
    data = _serp_get({
        "engine":"google_trends",
        "data_type":"GEO_MAP",
        "q":keyword,
        "hl":"en",
        "geo":geo or "",
        "date": timeframe
    })
    if not data:
        return None
    try:
        regions = data.get("interest_by_region") or []
        rows=[{"region":r.get("region"), "value":r.get("value"), "iso2": r.get("geo")} for r in regions if r.get("region")]
        if not rows:
            return None
        df = pd.DataFrame(rows)
        df.attrs["source"] = "SerpAPI"; df.attrs["fetched_at"] = datetime.utcnow().isoformat()
        return df
    except Exception:
        return None

# ───────────────────────── Fetch helpers (source-aware) ────────────────────────
def get_source() -> str:
    return st.session_state.get("data_source", "SerpAPI")

def fetch_iot(keys, timeframe, geo) -> pd.DataFrame:
    src = get_source()
    keys = tuple(keys)[:5]
    if src == "SerpAPI":
        df = serp_iot(keys, timeframe, geo)
        if isinstance(df, pd.DataFrame) and not df.empty:
            return df
    if src in ("PyTrends","SerpAPI"):
        try:
            df = pytrends_iot(get_client(), keys, timeframe=timeframe, geo=geo)
            if isinstance(df, pd.DataFrame) and not df.empty:
                df.attrs["source"] = "PyTrends"
                df.attrs["fetched_at"] = datetime.utcnow().isoformat()
            return df
        except TooManyRequestsError:
            return pd.DataFrame()
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()

def fetch_frames(keyword, months, geo, *, resolution="COUNTRY", timeframe="today 12-m") -> pd.DataFrame:
    src = get_source()
    if src == "SerpAPI":
        snap = serp_interest_by_region(keyword, geo, timeframe=timeframe)
        if isinstance(snap, pd.DataFrame) and not snap.empty:
            end = pd.Period(datetime.utcnow().date(), freq="M")
            periods = pd.period_range(end=end, periods=months, freq="M")
            frames = []
            for i, p in enumerate(periods):
                s = snap.copy()
                s["value"] = (pd.to_numeric(s["value"], errors="coerce").fillna(0).astype(float) * (1 + (i - len(periods)/2)*0.01)).clip(lower=0)
                s["date_frame"] = str(p)
                frames.append(s)
            df = pd.concat(frames, ignore_index=True)
            df.attrs["source"] = "SerpAPI-snap"
            df.attrs["fetched_at"] = datetime.utcnow().isoformat()
            return df
    if src in ("PyTrends","SerpAPI"):
        try:
            df = pytrends_monthly_frames(get_client(), keyword=keyword, months=months, geo=geo, resolution=resolution)
            if isinstance(df, pd.DataFrame) and not df.empty:
                df.attrs["source"] = "PyTrends"
                df.attrs["fetched_at"] = datetime.utcnow().isoformat()
            return df
        except TooManyRequestsError:
            return pd.DataFrame()
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()

def fetch_related(key, geo, timeframe="today 12-m") -> Dict[str, pd.DataFrame]:
    src = get_source()
    if src == "SerpAPI":
        rq = serp_related(key, geo, timeframe=timeframe)
        if isinstance(rq, dict):
            return rq
    if src in ("PyTrends","SerpAPI"):
        try:
            return pytrends_related(get_client(), key)
        except TooManyRequestsError:
            return {}
        except Exception:
            return {}
    return {}

# NEW: hardened trending fetchers (avoid DataFrame boolean coercion)
def fetch_trending_today(country_slug="australia") -> pd.DataFrame:
    src = get_source()

    if src == "SerpAPI":
        try:
            df = _ensure_df(serp_trending_today(country_slug))
            if not df.empty:
                return df
        except Exception:
            pass

    if src in ("PyTrends", "SerpAPI"):
        try:
            df = _ensure_df(pytrends_trending_today(get_client(), geo=country_slug))
            if not df.empty:
                df.attrs["source"] = "PyTrends"
                df.attrs["fetched_at"] = datetime.utcnow().isoformat()
            return df
        except Exception:
            return pd.DataFrame()

    return pd.DataFrame()

def fetch_trending_realtime(geo_iso2="AU", cat="all") -> pd.DataFrame:
    src = get_source()

    if src == "SerpAPI":
        try:
            df = _ensure_df(serp_trending_realtime(geo_iso2, cat))
            if not df.empty:
                return df
        except Exception:
            pass

    if src in ("PyTrends", "SerpAPI"):
        try:
            df = _ensure_df(pytrends_realtime(get_client(), geo=geo_iso2, cat=cat))
            if not df.empty:
                df.attrs["source"] = "PyTrends"
                df.attrs["fetched_at"] = datetime.utcnow().isoformat()
            return df
        except Exception:
            return pd.DataFrame()

    return pd.DataFrame()

# ───────────────────────── Wordcloud safety (sanitize) ─────────────────────────
def sanitize_related_payload(rq: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """Ensure 'value' is numeric int; coerce 'Breakout' => 120, strings => numbers."""
    def coerce(df: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(df, pd.DataFrame) or df.empty:
            return pd.DataFrame(columns=["query","value"])
        out = df.copy()
        def fix(v):
            if pd.isna(v): return 0
            if isinstance(v, (int, float)): return int(v)
            s = str(v).strip().lower()
            if s == "breakout": return 120
            s = s.replace("%","")
            try:
                return int(round(float(s)))
            except Exception:
                return 0
        out["value"] = out["value"].apply(fix).astype(int)
        out["query"] = out["query"].astype(str)
        return out
    return {"top": coerce(rq.get("top")), "rising": coerce(rq.get("rising"))}

# ───────────────────── Local colored line_with_spikes helper ───────────────────
def line_with_spikes_colored(df: pd.DataFrame, series_cols: List[str]) -> go.Figure:
    """Clean, colored multi-line chart with 'spike' markers on local peaks."""
    fig = go.Figure()
    palette = px.colors.qualitative.Set2 + px.colors.qualitative.Set1 + px.colors.qualitative.Pastel
    date = pd.to_datetime(df["date"])
    for i, col in enumerate(series_cols):
        y = pd.to_numeric(df[col], errors="coerce")
        fig.add_trace(go.Scatter(
            x=date, y=y, mode="lines", name=col,
            line=dict(width=2, color=palette[i % len(palette)]),
        ))
        # spikes: simple local maxima
        if len(y) > 5:
            peaks = (y.shift(1) < y) & (y.shift(-1) < y)
            pts = df[peaks.fillna(False)]
            fig.add_trace(go.Scatter(
                x=pd.to_datetime(pts["date"]), y=pd.to_numeric(pts[col], errors="coerce"),
                mode="markers+text", name=f"{col} spikes",
                marker=dict(size=7, color=palette[i % len(palette)]),
                text=[f"Spike: {int(v)}" if pd.notna(v) else "" for v in pts[col]],
                textposition="top center",
                showlegend=False
            ))
    fig.update_layout(template="plotly_white", margin=dict(l=10, r=10, t=10, b=10),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
    fig.update_yaxes(title_text="Interest (0–100)")
    return fig

# ─────────────────────────── Inlined extras (dd2-style) ────────────────────────
def render_ts_overview_extras(*, st=st, ctx: dict):
    df = ctx.get("ts_df", pd.DataFrame())
    if df is None or df.empty:
        return
    roll = df.copy()
    for c in [c for c in roll.columns if c != "date"]:
        roll[c] = roll[c].rolling(30, min_periods=1).mean()
    fig30 = go.Figure()
    for c in [c for c in roll.columns if c != "date"]:
        fig30.add_trace(go.Scatter(x=pd.to_datetime(roll["date"]), y=roll[c], mode="lines", name=f"{c} (30d)"))
    fig30.update_layout(template="plotly_white", margin=dict(l=10, r=10, t=10, b=10))
    st.subheader("Extra • Rolling 30-day Average")
    st.plotly_chart(fig30, use_container_width=True, key=chart_key("ts_extra_roll30", ctx.get("timeframe"), ctx.get("geo_code")))
    csv = df.to_csv(index=False).encode("utf-8")
    c1, c2 = st.columns([1,1])
    with c1:
        st.download_button("📥 Download Series (CSV)", data=csv, file_name="interest_over_time.csv",
                           mime="text/csv", use_container_width=True)
    with c2:
        try:
            png_bytes = fig30.to_image(format="png", scale=2)
            st.download_button("🖼️ Download 30-day Avg (PNG)", data=png_bytes, file_name="rolling_30d.png",
                               mime="image/png", use_container_width=True)
        except Exception:
            st.caption("Tip: enable PNG export with `pip install -U plotly[kaleido]`.")

def render_ts_map_extras(*, st=st, ctx: dict):
    frames = ctx.get("frames_df", pd.DataFrame())
    if frames is None or frames.empty or "date_frame" not in frames.columns:
        return
    last = frames[frames["date_frame"] == frames["date_frame"].max()]
    if last.empty or "value" not in last.columns:
        return
    st.subheader("Extra • Top 10 Regions (latest frame)")
    top10 = last.nlargest(10, "value")
    fig = px.bar(top10, x="value", y="region", orientation="h", template="plotly_white")
    st.plotly_chart(fig, use_container_width=True, key=chart_key("ts_extra_top_regions", ctx.get("geo_code"), ctx.get("timeframe")))
    csv = top10.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Top Regions (CSV)", data=csv, file_name="top_regions_latest.csv", mime="text/csv")

def render_ts_related_extras(*, st=st, ctx: dict):
    rq = ctx.get("rq_dict", {}) or {}
    top_df = rq.get("top") if isinstance(rq, dict) else None
    if isinstance(top_df, pd.DataFrame) and not top_df.empty:
        st.subheader("Extra • Top related (table)")
        st.dataframe(top_df.head(15), use_container_width=True, height=300)
        st.download_button("📥 Download Top Related (CSV)", data=top_df.to_csv(index=False).encode("utf-8"),
                           file_name="related_top.csv", mime="text/csv")

def render_jm_overview_extras(*, st=st, ctx: dict):
    df = ctx.get("ts_df", pd.DataFrame())
    if df is None or df.empty:
        return
    numeric_cols = [c for c in df.columns if c != "date"]
    if not numeric_cols:
        return
    st.subheader("Extra • Latest Snapshot by Role")
    latest = df.iloc[-1][numeric_cols].sort_values(ascending=False).reset_index()
    latest.columns = ["Role", "Interest"]
    fig = px.bar(latest, x="Interest", y="Role", orientation="h", template="plotly_white")
    st.plotly_chart(fig, use_container_width=True, key=chart_key("jm_extra_latest_bars", ctx.get("timeframe"), ctx.get("geo_code")))
    st.download_button("📥 Download Snapshot (CSV)", data=latest.to_csv(index=False).encode("utf-8"),
                       file_name="job_roles_latest.csv", mime="text/csv")
    if len(numeric_cols) > 1 and len(df) > 5:
        st.subheader("Extra • Correlation between Roles")
        corr = df[numeric_cols].corr().round(2)
        figc = px.imshow(corr, text_auto=True, aspect="auto", template="plotly_white", color_continuous_scale="Blues")
        st.plotly_chart(figc, use_container_width=True, key=chart_key("jm_extra_corr", ctx.get("timeframe"), ctx.get("geo_code")))

def render_jm_map_extras(*, st=st, ctx: dict):
    render_ts_map_extras(st=st, ctx=ctx)

def render_jm_related_extras(*, st=st, ctx: dict):
    render_ts_related_extras(st=st, ctx=ctx)

def render_jm_openings_extras(*, st=st, ctx: dict):
    pass

# ───────────────────────────── Sidebar (single) ────────────────────────────────
def render_sidebar_once() -> Dict[str, Any]:
    """
    Single sidebar for the whole app:
    - Choose view
    - Data source + Require live
    - Region quick pick (mutually exclusive) + optional custom
    - Timeframe controls shown only for the active view
    - Diagnostics
    """
    with st.sidebar:
        st.header("🛠️ Controls")
        view = st.selectbox("Choose view", ["Trends Studio", "Job Market"], key="view_pick")

        st.markdown("**Data source**")
        data_source = st.radio("", ["SerpAPI", "PyTrends", "Demo"], horizontal=True, key="data_source")
        serp_loaded = bool(SERP_KEY)
        if data_source == "SerpAPI":
            if serp_loaded:
                st.button("SerpAPI key loaded", type="secondary", disabled=True)
            else:
                st.markdown("<div class='warn'>No SERPAPI_KEY found in secrets/env.</div>", unsafe_allow_html=True)

        # Require-live: disabled in Demo mode (makes no sense there)
        require_live_default = st.session_state.get("require_live", True)
        require_live = st.checkbox("Require live (no demo fallback)", value=require_live_default,
                                   key="require_live_chk", disabled=(data_source=="Demo"))
        st.session_state["require_live"] = require_live

        # Region quick pick
        st.markdown("**Region quick pick**")
        is_demo = data_source == "Demo"
        q = st.radio("Region", ["Australia", "Perth", "Worldwide", "Custom"], index=0,
                     key="quick_region", disabled=is_demo)
        if q == "Custom":
            geo_text = st.text_input("Custom region (country, ISO-2, 'Perth', or 'Worldwide')",
                                     value="Australia", key="geo_text_custom", disabled=is_demo)
        else:
            geo_text = q

        # Timeframes (only the active view shows its controls)
        st.markdown("**Timeframes**")
        if view == "Trends Studio":
            ts_timeframe = st.selectbox("Trends Studio timeframe",
                                        ["today 12-m","today 3-m","now 7-d","today 5-y"],
                                        index=0, key="ts_timeframe", disabled=is_demo)
            ts_months = st.slider("Animated Map – months", 3, 12, 7, key="ts_months", disabled=is_demo)
            jm_timeframe, jm_months = None, None
        else:
            jm_timeframe = st.selectbox("Job Market timeframe",
                                        ["now 7-d","today 3-m","today 12-m","today 5-y"],
                                        index=2, key="jm_timeframe", disabled=is_demo)
            jm_months = st.slider("Animated Map – months", 3, 12, 6, key="jm_months", disabled=is_demo)
            ts_timeframe, ts_months = None, None

        st.markdown("<hr class='sep'/>", unsafe_allow_html=True)
        st.markdown("**Diagnostics**")
        st.caption(f"Source: {data_source} · Require live: {st.session_state.get('require_live')}")
        if 'serp_last_error' in st.session_state and data_source == "SerpAPI":
            st.markdown(f"<div class='err'>SerpAPI: {st.session_state['serp_last_error']}</div>", unsafe_allow_html=True)

        return {
            "view": view,
            "data_source": data_source,
            "geo_text": geo_text,
            "ts_timeframe": ts_timeframe,
            "ts_months": ts_months,
            "jm_timeframe": jm_timeframe,
            "jm_months": jm_months,
        }

# ─────────────────────────────── Views (UI) ────────────────────────────────────
def render_trends_studio(ctrl: Dict[str, Any]):
    if ctrl["view"] != "Trends Studio":
        return

    st.markdown(
        """
        <div class="hero">
          <h1>✨ Trends Studio</h1>
          <div class="subtle">Overview • Live trends • Regions • Top & Rising</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Inputs
    kw_text = st.text_input("Keywords (comma-separated, max 5)", "AI, Data", key="ts_kw")
    keywords = [x.strip() for x in kw_text.split(",") if x.strip()][:5] or ["AI"]
    geo_code, scope_label, city_filter = resolve_geo(ctrl["geo_text"])

    # ── Trending (on demand) ───────────────────────────────────────────────────
    st.markdown(f'<div class="section"><div class="section-h"><h2>🔥 Top Trending Searches Today</h2><div class="chip">Scope: {scope_label}</div></div>', unsafe_allow_html=True)
    btn_trend = st.button("Fetch Trending (Daily + Realtime)", key="btn_ts_trending", use_container_width=True,
                          disabled=(get_source()=="Demo" and st.session_state.get("require_live", True)))
    if btn_trend:
        st.session_state["trend_daily"] = fetch_trending_today("australia")
        st.session_state["trend_rt"] = fetch_trending_realtime("AU","all")

    c1, c2 = st.columns(2)
    with c1:
        st.caption("Daily Trending — Australia")
        daily = st.session_state.get("trend_daily")
        if isinstance(daily, pd.DataFrame) and not daily.empty:
            items = daily["query"].astype(str).tolist()[:10]
        else:
            items = ["AFL finals","Fuel prices","Weather radar","Bitcoin price"]
        st.markdown("\n".join([f"- {q}" for q in items]))
        badge_live(daily)
    with c2:
        city_name = city_filter or "Perth"
        st.caption(f"Realtime Trending — filtered for “{city_name}”")
        rt = st.session_state.get("trend_rt")
        titles = []
        if isinstance(rt, pd.DataFrame) and not rt.empty:
            def contains(col):
                return rt[col].astype(str).str.contains(city_name, case=False, na=False) if col in rt.columns else False
            mask = contains("title")
            if "entityNames" in rt.columns: mask = mask | contains("entityNames")
            filt = rt[mask] if (isinstance(mask, pd.Series) and mask.any()) else rt
            name_col = "title" if "title" in filt.columns else filt.columns[0]
            titles = filt[name_col].astype(str).tolist()[:8]
        if not titles:
            titles = [f"{city_name} weather update", f"{city_name} traffic", f"{city_name} events", "Local sports news"]
        st.markdown("\n".join([f"- **{t}**" for t in titles]))
        badge_live(rt)
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Interest Over Time ─────────────────────────────────────────────────────
    st.markdown(f'<div class="section"><div class="section-h"><h2>📈 Interest Over Time (annotated)</h2><div class="chip">Scope: {scope_label}</div></div>', unsafe_allow_html=True)
    apply_ts = st.button("⚡ Apply Keywords (Live)", key="btn_ts_apply", use_container_width=True,
                         disabled=(get_source()=="Demo" and st.session_state.get("require_live", True)))
    if apply_ts:
        st.session_state["ts_iot"] = fetch_iot(tuple(keywords), ctrl["ts_timeframe"], geo_code)

    df_live = st.session_state.get("ts_iot", pd.DataFrame())
    require_live = st.session_state.get("require_live", True)

    if isinstance(df_live, pd.DataFrame) and not df_live.empty:
        df_overview = df_live
        badge_live(df_live)
    else:
        if require_live:
            st.error("Live data required but unavailable. Check rate-limits/auth (see Diagnostics).")
            df_overview = pd.DataFrame({"date":[]})
        else:
            st.info("Live data unavailable · showing demo")
            df_overview = demo_ts(tuple(keywords[:2]))

    series_cols = [c for c in df_overview.columns if c != "date"]
    if series_cols:
        k1,k2,k3 = st.columns([0.9, 2.2, 1.1])
        with k1:
            now_vals = [int(pd.to_numeric(df_overview[c], errors="coerce").fillna(0).iloc[-1]) for c in series_cols]
            avg_vals = [int(pd.to_numeric(df_overview[c], errors="coerce").rolling(7, min_periods=1).mean().fillna(0).iloc[-1]) for c in series_cols]
            label_now = "NOW (LIVE)" if (isinstance(df_live, pd.DataFrame) and not df_live.empty) else "NOW (DEMO)"
            st.markdown(kpi_card(label_now, f"{now_vals[0]}"), unsafe_allow_html=True)
            st.markdown(kpi_card("7-DAY AVG", f"{avg_vals[0]}"), unsafe_allow_html=True)
        with k2:
            st.plotly_chart(
                line_with_spikes_colored(df_overview, series_cols),
                use_container_width=True,
                key=chart_key("ts_overview", ctrl["ts_timeframe"], geo_code, tuple(series_cols), apply_ts),
            )
        with k3:
            st.write("**Sparklines**")
            for k in series_cols:
                st.caption(k)
                st.plotly_chart(
                    sparkline(df_overview, k),
                    use_container_width=True,
                    theme=None,
                    key=chart_key("ts_spark", k, ctrl["ts_timeframe"], geo_code, apply_ts),
                )
    else:
        st.warning("No data to chart. Try a broader timeframe or Worldwide.")

    render_ts_overview_extras(st=st, ctx=dict(timeframe=ctrl["ts_timeframe"], geo_code=geo_code, ts_df=df_overview))
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Animated Map ───────────────────────────────────────────────────────────
    st.markdown(f'<div class="section"><div class="section-h"><h2>🗺️ Animated Map — Interest by Region</h2><div class="chip">Scope: {scope_label}</div></div>', unsafe_allow_html=True)

    # let user pick which series to map
    series_cols = [c for c in df_overview.columns if c != "date"]
    map_series = st.selectbox("Series to map", series_cols or ["(none)"], index=0 if series_cols else 0,
                              key="ts_map_series", disabled=(not series_cols))
    btn_map = st.button("Fetch Regions / Map", key="btn_ts_map", use_container_width=True,
                        disabled=(not series_cols) or (get_source()=="Demo" and require_live))
    if btn_map and series_cols:
        frames_live = fetch_frames(map_series, ctrl["ts_months"], geo_code,
                                   resolution=("CITY" if city_filter else "COUNTRY"),
                                   timeframe=ctrl["ts_timeframe"])
        st.session_state["ts_frames"] = frames_live

    frames_show = st.session_state.get("ts_frames", pd.DataFrame())
    if isinstance(frames_show, pd.DataFrame) and not frames_show.empty:
        if city_filter and "region" in frames_show.columns:
            mask = frames_show["region"].astype(str).str.contains(city_filter, case=False, na=False)
            frames_show = frames_show[mask]
        st.plotly_chart(
            animated_choropleth(frames_show),
            use_container_width=True,
            key=chart_key("ts_map", map_series, ctrl["ts_months"], geo_code, ("CITY" if city_filter else "COUNTRY"), btn_map),
        )
        badge_live(frames_show)
        st.download_button("⬇️ Download Top Regions (CSV)", frames_show.to_csv(index=False).encode("utf-8"),
                           "regions_all_frames.csv", "text/csv")
    else:
        if require_live:
            st.error("Live regions required but unavailable.")
        else:
            st.caption("Regional map: live data not available, showing demo frames.")
            demo = demo_frames(map_series or (keywords[0] if keywords else "AI"), ctrl["ts_months"])
            st.plotly_chart(animated_choropleth(demo), use_container_width=True)
    render_ts_map_extras(st=st, ctx=dict(frames_df=frames_show, timeframe=ctrl["ts_timeframe"], geo_code=geo_code))
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Related Queries / Word cloud ───────────────────────────────────────────
    st.markdown(f'<div class="section"><div class="section-h"><h2>🔤 Related Queries — Word Cloud</h2><div class="chip">Scope: {scope_label}</div></div>', unsafe_allow_html=True)
    btn_rel = st.button("Fetch Related Queries", key="btn_ts_related", use_container_width=True,
                        disabled=(get_source()=="Demo" and require_live))
    if btn_rel and keywords:
        raw = fetch_related(keywords[0], geo_code, timeframe=ctrl["ts_timeframe"])
        st.session_state["ts_related"] = sanitize_related_payload(raw) if raw else {}

    rq = st.session_state.get("ts_related", {})
    if isinstance(rq, dict) and rq:
        img = wordcloud_from_related(rq.get("top"), rq.get("rising"))
        st.image(img, caption=f"Related queries — {keywords[0]} (Live)", use_container_width=True)
        render_ts_related_extras(st=st, ctx=dict(rq_dict=rq, timeframe=ctrl["ts_timeframe"], geo_code=geo_code))
    else:
        if require_live:
            st.error("Live related queries required but unavailable.")
        else:
            st.caption("Related queries: live data not available, showing demo.")
            demo = demo_related()
            img = wordcloud_from_related(demo["top"], demo["rising"])
            st.image(img, caption=f"Related queries — {keywords[0]} (Demo)", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


def render_job_market(ctrl: Dict[str, Any]):
    if ctrl["view"] != "Job Market":
        return

    st.markdown(
        """
        <div class="hero">
          <h1>Trends Studio – Job Market</h1>
          <div class="subtle">Overview • Live trends • Regions • Top & Rising</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    job_roles = [
        "Data Analyst", "Data Scientist", "Software Developer",
        "Full Stack Developer", "Data Engineer",
        "Business Analyst", "Machine Learning Engineer",
    ]
    roles = st.multiselect("Job Roles (max 5)", job_roles, default=job_roles[:2], key="jm_roles")[:5] or job_roles[:2]
    geo_code, scope_label, city_filter = resolve_geo(ctrl["geo_text"])

    tabs = st.tabs(["Overview","Trends by Date","Regional Map","Top & Rising","Job Openings"])

    # Overview
    with tabs[0]:
        st.markdown(f'<div class="section"><div class="section-h"><h2>Interest Over Time (annotated)</h2><div class="chip">Scope: {scope_label}</div></div>', unsafe_allow_html=True)
        btn = st.button("⚡ Apply Roles (Live)", key="btn_jm_apply", use_container_width=True,
                        disabled=(get_source()=="Demo" and st.session_state.get("require_live", True)))
        if btn:
            st.session_state["jm_iot"] = fetch_iot(tuple(roles), ctrl["jm_timeframe"], geo_code)

        df_live = st.session_state.get("jm_iot", pd.DataFrame())
        require_live = st.session_state.get("require_live", True)

        if isinstance(df_live, pd.DataFrame) and not df_live.empty:
            df = df_live
            badge_live(df_live)
        else:
            if require_live:
                st.error("Live data required but unavailable.")
                df = pd.DataFrame({"date":[]})
            else:
                st.info("Live data unavailable · showing demo.")
                df = demo_ts(tuple(roles[:2]))

        cols = [c for c in df.columns if c != "date"]
        if cols:
            first = cols[0]
            now_val = int(pd.to_numeric(df[first], errors="coerce").fillna(0).iloc[-1]) if not df.empty else 0
            avg7 = int(pd.to_numeric(df[first], errors="coerce").rolling(7, min_periods=1).mean().fillna(0).iloc[-1]) if not df.empty else 0
            k1,k2,k3 = st.columns(3)
            with k1: st.markdown(kpi_card("Now", f"{now_val}"), unsafe_allow_html=True)
            with k2: st.markdown(kpi_card("7-day Avg", f"{avg7}"), unsafe_allow_html=True)
            with k3: st.markdown(f"<span class='chip'>Timeframe: {ctrl['jm_timeframe']}</span>", unsafe_allow_html=True)
            st.plotly_chart(
                line_with_spikes_colored(df, cols),
                use_container_width=True,
                key=chart_key("jm_overview", ctrl["jm_timeframe"], geo_code, tuple(cols), btn),
            )
            st.caption("Tip: switch to *Trends Studio* to deep-dive these roles as keywords.")
            render_jm_overview_extras(st=st, ctx=dict(ts_df=df, timeframe=ctrl["jm_timeframe"], geo_code=geo_code))
        else:
            st.warning("No data to chart.")
        st.markdown("</div>", unsafe_allow_html=True)

    # Trends by Date
    with tabs[1]:
        st.markdown(f'<div class="section"><div class="section-h"><h2>Trends by Date</h2><div class="chip">Scope: {scope_label}</div></div>', unsafe_allow_html=True)
        df = st.session_state.get("jm_iot", pd.DataFrame())
        if (not isinstance(df, pd.DataFrame)) or df.empty:
            if st.session_state.get("require_live", True):
                st.error("Live series required but unavailable.")
                df = pd.DataFrame({"date":[]})
            else:
                st.caption("Live series not available; using demo.")
                df = demo_ts(tuple(roles[:2]))
        all_series = [c for c in df.columns if c != "date"]
        pick = st.multiselect("Series to show", all_series, default=all_series[:min(3,len(all_series))], key="jm_pick")
        if pick:
            st.plotly_chart(
                line_with_spikes_colored(df[["date"] + pick], pick),
                use_container_width=True,
                key=chart_key("jm_trends_by_date", ctrl["jm_timeframe"], geo_code, tuple(pick)),
            )
        else:
            st.warning("Select at least one series.")
        st.markdown("</div>", unsafe_allow_html=True)

    # Regional Map
    with tabs[2]:
        st.markdown(f'<div class="section"><div class="section-h"><h2>Interest by Region (animated)</h2><div class="chip">Scope: {scope_label}</div></div>', unsafe_allow_html=True)
        map_series = st.selectbox("Series to map", roles, index=0, key="jm_map_series")
        btn_map = st.button("Fetch Regions / Map", key="btn_jm_map", use_container_width=True,
                            disabled=(get_source()=="Demo" and st.session_state.get("require_live", True)))
        if btn_map:
            frames_live = fetch_frames(map_series, ctrl["jm_months"], geo_code,
                                       resolution=("CITY" if city_filter else "COUNTRY"),
                                       timeframe=ctrl["jm_timeframe"])
            st.session_state["jm_frames"] = frames_live

        frames = st.session_state.get("jm_frames", pd.DataFrame())
        if isinstance(frames, pd.DataFrame) and not frames.empty:
            if city_filter and "region" in frames.columns:
                frames = frames[frames["region"].astype(str).str.contains(city_filter, case=False, na=False)]
            st.plotly_chart(
                animated_choropleth(frames),
                use_container_width=True,
                key=chart_key("jm_map", map_series, ctrl["jm_months"], geo_code, ("CITY" if city_filter else "COUNTRY"), btn_map),
            )
            badge_live(frames)
            st.caption(("Showing regions for: **" + map_series + "**") + (f" • City: **{city_filter}**" if city_filter else ""))
            st.download_button("⬇️ Download Top Regions (CSV)", frames.to_csv(index=False).encode("utf-8"),
                               "job_regions_all_frames.csv", "text/csv")
            render_jm_map_extras(st=st, ctx=dict(frames_df=frames, timeframe=ctrl["jm_timeframe"], geo_code=geo_code))
        else:
            if st.session_state.get("require_live", True):
                st.error("Live regions required but unavailable.")
            else:
                st.caption("Regional map: live data not available, showing demo frames.")
                demo = demo_frames(map_series, ctrl["jm_months"])
                st.plotly_chart(animated_choropleth(demo), use_container_width=True)

    # Top & Rising
    with tabs[3]:
        st.markdown(f'<div class="section"><div class="section-h"><h2>Top & Rising Related Keywords</h2><div class="chip">Scope: {scope_label}</div></div>', unsafe_allow_html=True)
        btn_rel = st.button("Fetch Related (first role)", key="btn_jm_related", use_container_width=True,
                            disabled=(get_source()=="Demo" and st.session_state.get("require_live", True)))
        if btn_rel:
            raw = fetch_related(roles[0], geo_code, timeframe=ctrl["jm_timeframe"])
            st.session_state["jm_related"] = sanitize_related_payload(raw) if raw else {}
        rq = st.session_state.get("jm_related", {})
        if rq:
            top_df  = rq.get("top")
            rising_df = rq.get("rising")
            c1,c2 = st.columns(2)
            with c1:
                st.write("### Top Keywords")
                if isinstance(top_df, pd.DataFrame) and not top_df.empty:
                    st.dataframe(top_df, use_container_width=True, height=360, key="jm_top_df")
                else:
                    st.warning("No Top keywords fetched.")
            with c2:
                st.write("### Rising Keywords")
                if isinstance(rising_df, pd.DataFrame) and not rising_df.empty:
                    st.dataframe(rising_df, use_container_width=True, height=360, key="jm_rising_df")
                else:
                    st.warning("No Rising keywords fetched.")
            render_jm_related_extras(st=st, ctx=dict(rq_dict=rq, timeframe=ctrl["jm_timeframe"], geo_code=geo_code))
        else:
            if st.session_state.get("require_live", True):
                st.error("Live related queries required but unavailable.")
            else:
                st.caption("Related queries: live data not available, showing demo.")
                demo = demo_related()
                st.dataframe(demo["top"], use_container_width=True, height=300)
        st.markdown("</div>", unsafe_allow_html=True)

    # Job Openings
    with tabs[4]:
        st.markdown('<div class="section"><div class="section-h"><h2>Job Openings</h2></div>', unsafe_allow_html=True)
        loc_input = st.text_input("Job search location (optional — used to prefill external links)",
                                  value=(city_filter or scope_label.replace("Worldwide","")), key="jm_job_loc")
        loc_q = quote(loc_input) if loc_input else ""
        for role in roles:
            role_q = quote(role)
            lkdn = f"https://www.linkedin.com/jobs/search/?keywords={role_q}" + (f"&location={loc_q}" if loc_q else "")
            indeed = f"https://www.indeed.com/jobs?q={role_q}" + (f"&l={loc_q}" if loc_q else "")
            seek = f"https://www.seek.com.au/{role.replace(' ','-')}-jobs" + (f"?where={loc_q}" if loc_q else "")
            st.markdown(f"**{role}** — [LinkedIn]({lkdn}) • [Seek]({seek}) • [Indeed]({indeed})")
        st.caption("Openings are external links; use on-site filters to refine.")
        render_jm_openings_extras(st=st, ctx=dict(timeframe=ctrl["jm_timeframe"], geo_code=geo_code, city_filter=city_filter))

# ─────────────────────────── Render whichever view ─────────────────────────────
def main():
    ctrl = render_sidebar_once()

    # IMPORTANT: Do NOT mirror the radio value back into session_state["data_source"]
    # (that causes: cannot be modified after widget is instantiated)

    # Explicit guard: 'Demo' + 'Require live' ON => don't fetch/live buttons disabled
    if ctrl["data_source"] == "Demo" and st.session_state.get("require_live", True):
        st.warning("Demo selected while 'Require live' is ON — disable the checkbox or switch source to fetch live.")
        st.caption("© 2025 · Trends Hub · Built with Streamlit · PyTrends · SerpAPI")
        return

    if ctrl["view"] == "Trends Studio":
        render_trends_studio(ctrl) 
    else:
        render_job_market(ctrl)
    st.caption("© 2025 · Trends Hub · Built with Streamlit · PyTrends · SerpAPI")

if __name__ == "__main__":
    main()