# app_gtrends.py — ONE FILE VERSION (SerpAPI + PyTrends + Demo)
# =============================================================================
# GOALS
# -----
# - One sidebar rendered ONCE (prevents duplicate widget-key / session_state errors)
# - Mutually-exclusive region quick pick (Australia / Perth / Worldwide / Custom)
# - Per-section "Fetch" buttons (no global rerun button)
# - Uses SerpAPI when selected, falls back to PyTrends, then Demo
# - Clear diagnostics why live fetch failed (status pill + details expander)
# - dd2-style extras: rolling 30d, snapshot-by-role, correlation heatmap
# - Colored time-series lines (per your request)
# - No deprecated use_column_width (uses use_container_width=True or width='stretch')
#
# REQUIREMENTS
# ------------
# pip install streamlit plotly requests pytrends numpy pandas
#
# SERPAPI KEY
# -----------
# .streamlit/secrets.toml     -> SERPAPI_KEY = "YOUR_REAL_KEY"
# (or) ENV var SERPAPI_KEY    -> export SERPAPI_KEY=YOUR_REAL_KEY
#
# FILES YOU ALREADY HAVE
# ----------------------
# - trends.py  : get_client(), interest_over_time(), monthly_region_frames(), related_queries(), trending_today(), trending_realtime()
# - viz.py     : line_with_spikes(), animated_choropleth(), wordcloud_from_related(), kpi_card(), sparkline()
# - utils.py   : country_name_to_iso2()
#
# NOTES
# -----
# - The app loads DEMO by default until you click the per-section Fetch buttons.
# - Choose data source in the sidebar: SerpAPI / PyTrends / Demo.
# - If SerpAPI is selected but the key is missing/invalid, sidebar shows a warning.
# =============================================================================

from __future__ import annotations

# ───────────────────────────────── Imports ─────────────────────────────────────
from datetime import datetime, timedelta
from urllib.parse import quote
from typing import Iterable, Optional, Tuple, Dict, Any

import os
import time
import json
import math

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from pytrends.exceptions import TooManyRequestsError

# ── local modules (your originals) ─────────────────────────────────────────────
from trends import (
    get_client,
    interest_over_time as pytrends_iot,
    monthly_region_frames as pytrends_monthly_frames,
    related_queries as pytrends_related,
    trending_today as pytrends_trending_today,
    trending_realtime as pytrends_realtime,
)
from viz import (
    line_with_spikes,         # kept (we'll also offer a colored variant below)
    animated_choropleth,
    wordcloud_from_related,
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
  --ok:#ecfdf5; --ok-brd:#a7f3d0; --ok-ink:#065f46;
  --bad:#fef2f2; --bad-brd:#fecaca; --bad-ink:#991b1b;
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
.warn{background:var(--warn);border:1px solid var(--warn-brd);color:var(--warn-ink);
      border-radius:10px;padding:10px 12px;font-size:.9rem;}
.bad{background:var(--bad);border:1px solid var(--bad-brd);color:var(--bad-ink);
      border-radius:10px;padding:10px 12px;font-size:.9rem;}
.ok{background:var(--ok);border:1px solid var(--ok-brd);color:var(--ok-ink);
      border-radius:10px;padding:10px 12px;font-size:.9rem;}
.pill{display:inline-block;padding:4px 10px;border-radius:999px;border:1px solid var(--card-brd);
     background:#fff;font-size:.8rem}
hr.soft{border:none;border-top:1px solid var(--card-brd);margin:8px 0 12px 0}
</style>
""",
    unsafe_allow_html=True,
)

# ───────────────────────── Small utils ─────────────────────────────────────────
def chart_key(prefix: str, *parts) -> str:
    """Short, stable keys for Plotly components to avoid re-mount churn."""
    safe = "|".join(str(p) for p in parts if p is not None)
    return f"{prefix}:{safe}"[:200]

def to_list(x: Iterable[str] | str) -> list[str]:
    if x is None:
        return []
    if isinstance(x, str):
        return [x]
    return list(x)

def qualitative_palette(n: int) -> list[str]:
    """Return a qualitative color list with at least n colors."""
    palettes = [
        px.colors.qualitative.Plotly,
        px.colors.qualitative.Safe,
        px.colors.qualitative.Set2,
        px.colors.qualitative.Set3,
        px.colors.qualitative.Pastel,
        px.colors.qualitative.Bold,
        px.colors.qualitative.Prism,
    ]
    out = []
    for pal in palettes:
        out.extend(pal)
        if len(out) >= n:
            break
    # ensure uniqueness order
    uniq = []
    for c in out:
        if c not in uniq:
            uniq.append(c)
    return uniq[:max(1, n)]

# ─────────────────────────── Geo resolver (country/city/world) ─────────────────
def resolve_geo(user_input: str) -> Tuple[str, str, Optional[str]]:
    """
    Returns (geo_code_for_api, scope_label, city_filter)
      - '' for worldwide
      - 'ISO-2' code for country
      - city_filter is used only for realtime/map filtering text contains
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
    return pd.DataFrame(rows)

def demo_related():
    top = pd.DataFrame({"query":["what is ai","data analytics login","ai tools"],"value":[80,65,50]})
    rising = pd.DataFrame({"query":["ai agents","gpt-4o","prompt ideas"],"value":[120,100,95]})
    return {"top": top, "rising": rising}

# ───────────────────────────── SerpAPI client ──────────────────────────────────
SERP_KEY = st.secrets.get("SERPAPI_KEY", os.getenv("SERPAPI_KEY", "")).strip()

def _serp_get(params: Dict[str, Any], max_retries: int = 3) -> Dict[str, Any] | None:
    """GET wrapper with backoff. Returns JSON or None."""
    if not SERP_KEY:
        return None
    base = "https://serpapi.com/search.json"
    params = dict(params)
    params["api_key"] = SERP_KEY
    wait = 1.2
    for _ in range(max_retries):
        try:
            r = requests.get(base, params=params, timeout=20)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (401, 403):
                st.session_state["serp_last_error"] = f"{r.status_code}: unauthorized"
                return None
            if r.status_code == 429:
                st.session_state["serp_last_error"] = "429: rate-limit"
                time.sleep(wait); wait *= 1.7; continue
            st.session_state["serp_last_error"] = f"{r.status_code}: {r.text[:180]}"
            time.sleep(wait); wait *= 1.4
        except Exception as e:
            st.session_state["serp_last_error"] = f"Exception: {e}"
            time.sleep(wait); wait *= 1.4
    return None

def serp_iot(keywords: Iterable[str], timeframe: str, geo: str) -> pd.DataFrame | None:
    """
    SerpAPI: engine=google_trends, data_type=TIMESERIES
    """
    kws = ",".join(to_list(keywords)[:5])
    data = _serp_get({
        "engine":"google_trends","data_type":"TIMESERIES",
        "q":kws,"hl":"en","time":timeframe,"geo":geo
    })
    if not data:
        return None
    try:
        iot = data.get("interest_over_time") or {}
        timeline = iot.get("timeline_data") or []
        if not timeline:
            return None
        # column names: prefer query names from 'default_ranking' else user-supplied keywords
        names = to_list(iot.get("default_ranking", [])) or to_list(keywords)
        rows = []
        for p in timeline:
            dt = pd.to_datetime(p.get("date"))
            vals = p.get("values", [])
            series = []
            for v in vals:
                if isinstance(v, dict):
                    series.append(v.get("value", 0))
                else:
                    series.append(0)
            rows.append([dt] + series)
        cols = ["date"] + names[:len(rows[0])-1] if rows and len(rows[0])>1 else ["date"] + to_list(keywords)
        df = pd.DataFrame(rows, columns=cols)
        # ensure all requested columns exist
        for kw in to_list(keywords):
            if kw not in df.columns:
                df[kw] = np.nan
        df = df[["date"] + list(dict.fromkeys(to_list(keywords)))]
        return df
    except Exception:
        return None

def serp_related(keyword: str, geo: str) -> Dict[str, pd.DataFrame] | None:
    data = _serp_get({"engine":"google_trends","data_type":"RELATED_QUERIES","q":keyword,"hl":"en","geo":geo})
    if not data: return None
    try:
        rq = data.get("related_queries", {}) or {}
        def _mk(name):
            arr = rq.get(name) or []
            if not arr: return pd.DataFrame()
            return pd.DataFrame([{"query":x.get("query"), "value":x.get("value")} for x in arr if x.get("query")])
        return {"top": _mk("top"), "rising": _mk("rising")}
    except Exception:
        return None

def serp_trending_today(geo_country: str="australia") -> pd.DataFrame | None:
    data = _serp_get({"engine":"google_trends_trending_now","hl":"en","geo":geo_country})
    if not data: return None
    try:
        stories = data.get("stories") or []
        qs = []
        for s in stories:
            q = s.get("title") or s.get("query")
            if q: qs.append(q)
        return pd.DataFrame({"query": qs}) if qs else None
    except Exception:
        return None

def serp_trending_realtime(geo_iso2: str="AU", cat: str="all") -> pd.DataFrame | None:
    data = _serp_get({"engine":"google_trends_realtime","hl":"en","geo":geo_iso2,"category":cat})
    if not data: return None
    try:
        stories = data.get("stories") or []
        rows = [{"title": s.get("title",""), "entityNames": ", ".join(s.get("entityNames",[]) or [])} for s in stories]
        return pd.DataFrame(rows) if rows else None
    except Exception:
        return None

def serp_interest_by_region(keyword: str, geo: str) -> pd.DataFrame | None:
    data = _serp_get({"engine":"google_trends","data_type":"GEO_MAP","q":keyword,"hl":"en","geo":geo})
    if not data: return None
    try:
        regions = data.get("interest_by_region") or []
        rows = [{"region":r.get("region"), "value":r.get("value"), "iso2": r.get("geo")} for r in regions if r.get("region")]
        return pd.DataFrame(rows) if rows else None
    except Exception:
        return None

def serp_healthcheck() -> dict:
    """Cheap probe to show why SerpAPI might fail for the user."""
    if not SERP_KEY:
        return {"ok": False, "reason": "SERPAPI_KEY not found in secrets or env."}
    data = _serp_get({
        "engine":"google_trends","data_type":"TIMESERIES",
        "q":"AI","hl":"en","time":"now 7-d","geo":""
    }, max_retries=1)
    if data:
        return {"ok": True, "reason": "SerpAPI responded."}
    return {"ok": False, "reason": st.session_state.get("serp_last_error","No response")}

# ───────────────────────── Source-aware fetch helpers ──────────────────────────
def get_source() -> str:
    # Read-only; the widget sets this key, we do not assign to it after creation.
    return st.session_state.get("data_source", "Demo")

def fetch_iot(keys, timeframe, geo) -> pd.DataFrame:
    src = get_source()
    keys = tuple(keys)[:5]
    if src == "SerpAPI":
        df = serp_iot(keys, timeframe, geo)
        if isinstance(df, pd.DataFrame) and not df.empty:
            return df
    if src in ("PyTrends","SerpAPI"):
        try:
            return pytrends_iot(get_client(), keys, timeframe=timeframe, geo=geo)
        except TooManyRequestsError:
            return pd.DataFrame()
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()

def fetch_frames(keyword, months, geo, *, resolution="COUNTRY") -> pd.DataFrame:
    src = get_source()
    if src == "SerpAPI":
        # SerpAPI is snapshot only => synthesize monthly frames for animation
        snap = serp_interest_by_region(keyword, geo)
        if isinstance(snap, pd.DataFrame) and not snap.empty:
            end = pd.Period(datetime.utcnow().date(), freq="M")
            periods = pd.period_range(end=end, periods=months, freq="M")
            frames=[]
            for i,p in enumerate(periods):
                s = snap.copy()
                s["value"] = (s["value"].astype(float) * (1 + (i - len(periods)/2)*0.01)).clip(lower=0)
                s["date_frame"] = str(p)
                frames.append(s)
            return pd.concat(frames, ignore_index=True)
    if src in ("PyTrends","SerpAPI"):
        try:
            return pytrends_monthly_frames(get_client(), keyword=keyword, months=months, geo=geo, resolution=resolution)
        except TooManyRequestsError:
            return pd.DataFrame()
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()

def fetch_related(key, geo) -> Dict[str, pd.DataFrame]:
    src = get_source()
    if src == "SerpAPI":
        rq = serp_related(key, geo)
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

def fetch_trending_today(country_slug="australia") -> pd.DataFrame:
    src = get_source()
    if src == "SerpAPI":
        df = serp_trending_today(country_slug)
        if isinstance(df, pd.DataFrame) and not df.empty:
            return df
    if src in ("PyTrends","SerpAPI"):
        try:
            return pytrends_trending_today(get_client(), geo=country_slug)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()

def fetch_trending_realtime(geo_iso2="AU", cat="all") -> pd.DataFrame:
    src = get_source()
    if src == "SerpAPI":
        df = serp_trending_realtime(geo_iso2, cat)
        if isinstance(df, pd.DataFrame) and not df.empty:
            return df
    if src in ("PyTrends","SerpAPI"):
        try:
            return pytrends_realtime(get_client(), geo=geo_iso2, cat=cat)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()

# ─────────────────────── Colored time-series helper plot ───────────────────────
def colored_line(df: pd.DataFrame, series_cols: list[str], title: str | None = None) -> go.Figure:
    """
    Custom colored variant of your line_with_spikes. Keeps your original available.
    """
    fig = go.Figure()
    palette = qualitative_palette(len(series_cols))
    for i, col in enumerate(series_cols):
        fig.add_trace(go.Scatter(
            x=df["date"], y=df[col],
            mode="lines+markers",
            name=col,
            line=dict(width=2, color=palette[i % len(palette)]),
            marker=dict(size=4),
            hovertemplate="%{y}<extra>"+col+"</extra>"
        ))
    if title:
        fig.update_layout(title=title)
    fig.update_layout(template="plotly_white", margin=dict(l=10,r=10,t=30,b=10), legend=dict(orientation="h", y=1.1))
    return fig

# ─────────────────────────── Inlined “dd2” EXTRAS ─────────────────────────────
def render_ts_overview_extras(*, st=st, ctx: dict):
    df = ctx.get("ts_df", pd.DataFrame())
    if df is None or df.empty:
        return
    roll = df.copy()
    for c in [c for c in roll.columns if c != "date"]:
        roll[c] = roll[c].rolling(30, min_periods=1).mean()
    fig30 = go.Figure()
    for c in [c for c in roll.columns if c != "date"]:
        fig30.add_trace(go.Scatter(x=roll["date"], y=roll[c], mode="lines", name=f"{c} (30d)"))
    fig30.update_layout(template="plotly_white", margin=dict(l=10, r=10, t=10, b=10))
    st.subheader("Extra • Rolling 30-day Average")
    st.plotly_chart(fig30, use_container_width=True, key=chart_key("ts_extra_roll30", ctx.get("timeframe"), ctx.get("geo_code")))
    csv = df.to_csv(index=False).encode("utf-8")
    c1, c2 = st.columns([1,1])
    with c1:
        st.download_button("📥 Download Series (CSV)", data=csv, file_name="interest_over_time.csv", mime="text/csv", use_container_width=True)
    with c2:
        try:
            png_bytes = fig30.to_image(format="png", scale=2)
            st.download_button("🖼️ Download 30-day Avg (PNG)", data=png_bytes, file_name="rolling_30d.png", mime="image/png", use_container_width=True)
        except Exception:
            st.caption("Tip: to enable PNG export, install `pip install -U plotly[kaleido]`.")

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
    # Placeholder to extend safely without scraping
    pass

# ───────────────────────── Sidebar (render ONCE) ───────────────────────────────
def effective_geo_text(quick: str, custom_text: str) -> str:
    if quick == "Australia": return "Australia"
    if quick == "Perth":     return "Perth"
    if quick == "Worldwide": return "Worldwide"
    return custom_text or "Australia"

def render_sidebar_once() -> dict:
    """
    Builds the sidebar ONCE. Returns a dict with all controls used by views.
    • We DO NOT assign to widget-backed st.session_state keys after creation.
    • No duplicate keys across multiple sidebars.
    """
    with st.sidebar:
        st.markdown("### 🔧 Controls")
        view = st.selectbox("Choose view", ["Trends Studio", "Job Market"], key="view_pick")

        # Data source picker (read-only for code; we won't override it)
        data_source = st.radio("Data source", ["SerpAPI", "PyTrends", "Demo"], horizontal=True, key="data_source")
        if data_source == "SerpAPI":
            if SERP_KEY:
                st.markdown("<span class='pill'>SerpAPI key loaded</span>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='bad'>No SERPAPI_KEY found in secrets or env.<br/>Select PyTrends or Demo.</div>", unsafe_allow_html=True)

        st.markdown("#### Region quick pick")
        quick = st.radio("Region", ["Australia","Perth","Worldwide","Custom"], horizontal=False, key="quick_region")
        # We do NOT force the text field to change; we just ignore it unless 'Custom'
        geo_text = st.text_input(
            "Custom region (country, ISO-2, 'Perth', or 'Worldwide')",
            value=st.session_state.get("custom_geo_text","Australia"),
            key="custom_geo_text",
            disabled=(quick != "Custom")
        )

        st.markdown("#### Timeframes")
        ts_timeframe = st.selectbox("Trends Studio timeframe", ["today 12-m","today 3-m","now 7-d","today 5-y"], index=0, key="ts_timeframe")
        jm_timeframe = st.selectbox("Job Market timeframe", ["now 7-d","today 3-m","today 12-m","today 5-y"], index=2, key="jm_timeframe")

        st.markdown("#### Animated Map frames")
        ts_months = st.slider("Trends Studio months", 3, 12, 7, key="ts_months")
        jm_months = st.slider("Job Market months", 3, 12, 6, key="jm_months")

        # Diagnostics
        with st.expander("🔍 Data source diagnostics"):
            st.write("Selected source:", data_source)
            if data_source == "SerpAPI":
                hc = serp_healthcheck()
                if hc["ok"]:
                    st.markdown("<div class='ok'>SerpAPI OK.</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='warn'>SerpAPI problem: {hc['reason']}</div>", unsafe_allow_html=True)
            st.write("Tip: PyTrends can rate-limit; if so, try SerpAPI or Demo temporarily.")

        return dict(
            view=view,
            data_source=data_source,
            quick=quick,
            geo_text=geo_text,
            ts_timeframe=ts_timeframe,
            jm_timeframe=jm_timeframe,
            ts_months=ts_months,
            jm_months=jm_months,
        )

# ─────────────────────────────── Views (UI) ────────────────────────────────────
def render_trends_studio(ctrl: dict):
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

    # Keyword entry + Apply
    default_kw = st.session_state.get("applied_keywords", "AI, Data")
    kw_text = st.text_input("Keywords (comma-separated, max 5)", default_kw, key="ts_kw")
    colA, colB = st.columns([1,1])
    with colA:
        if st.button("Apply Keywords", key="btn_apply_kw"):
            st.session_state["applied_keywords"] = kw_text
            st.toast("Keywords applied. Click fetch buttons in sections to get live data.", icon="✅")
    with colB:
        st.caption("Applied: " + (st.session_state.get("applied_keywords","AI, Data")))

    keywords = [x.strip() for x in st.session_state.get("applied_keywords", kw_text).split(",") if x.strip()][:5] or ["AI"]
    eff_geo_text = effective_geo_text(ctrl["quick"], ctrl["geo_text"])
    geo_code, scope_label, city_filter = resolve_geo(eff_geo_text)

    # ── Trending section ───────────────────────────────────────────────────────
    st.markdown(f'<div class="section"><div class="section-h"><h2>🔥 Top Trending Searches Today</h2><div class="chip">Scope: {scope_label}</div></div>', unsafe_allow_html=True)
    btn_trend = st.button("Fetch Trending (Daily + Realtime)", key="btn_ts_trending", use_container_width=True)

    if btn_trend:
        st.session_state["trend_daily"] = fetch_trending_today("australia")
        st.session_state["trend_rt"] = fetch_trending_realtime("AU","all")

    c1, c2 = st.columns(2)
    with c1:
        st.caption("Daily Trending — Australia")
        daily = st.session_state.get("trend_daily")
        items = (
            daily["query"].astype(str).tolist()[:10]
            if (isinstance(daily, pd.DataFrame) and not daily.empty and "query" in daily.columns)
            else ["AFL finals","Fuel prices","Weather radar","Bitcoin price"]
        )
        st.markdown("\n".join([f"- {q}" for q in items]))
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
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Interest Over Time ─────────────────────────────────────────────────────
    st.markdown(f'<div class="section"><div class="section-h"><h2>📈 Interest Over Time</h2><div class="chip">Scope: {scope_label}</div></div>', unsafe_allow_html=True)
    col1, col2 = st.columns([1,3])
    with col1:
        btn_ts = st.button("⚡ Fetch Time Series (Live)", key="btn_ts_iot", use_container_width=True)
        st.caption("Powered by: " + get_source())
    with col2:
        st.caption(f"Timeframe: **{ctrl['ts_timeframe']}** • Geo: **{eff_geo_text}**")

    if btn_ts:
        st.session_state["ts_iot"] = fetch_iot(tuple(keywords), ctrl["ts_timeframe"], geo_code)

    df_live = st.session_state.get("ts_iot", pd.DataFrame())
    df_overview = df_live if (isinstance(df_live, pd.DataFrame) and not df_live.empty) else demo_ts(tuple(keywords[:2]))
    series_cols = [c for c in df_overview.columns if c != "date"]

    k1,k2,k3 = st.columns([0.9, 2.2, 1.1])
    with k1:
        now_vals = [int(df_overview[c].iloc[-1]) for c in series_cols]
        avg_vals = [int(df_overview[c].rolling(7, min_periods=1).mean().iloc[-1]) for c in series_cols]
        label_now = "NOW (LIVE)" if (isinstance(df_live, pd.DataFrame) and not df_live.empty) else "NOW (DEMO)"
        st.markdown(kpi_card(label_now, f"{now_vals[0]}"), unsafe_allow_html=True)
        st.markdown(kpi_card("7-DAY AVG", f"{avg_vals[0]}"), unsafe_allow_html=True)
    with k2:
        # colored variant, as requested
        st.plotly_chart(
            colored_line(df_overview, series_cols, title=None),
            use_container_width=True,
            key=chart_key("ts_overview_colored", ctrl["ts_timeframe"], geo_code, tuple(series_cols), btn_ts),
        )
    with k3:
        st.write("**Sparklines**")
        for k in series_cols:
            st.caption(k)
            st.plotly_chart(
                sparkline(df_overview, k),
                use_container_width=True,
                theme=None,
                key=chart_key("ts_spark", k, ctrl["ts_timeframe"], geo_code, btn_ts),
            )

    render_ts_overview_extras(st=st, ctx=dict(timeframe=ctrl["ts_timeframe"], geo_code=geo_code, ts_df=df_overview))
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Animated Map ───────────────────────────────────────────────────────────
    st.markdown(f'<div class="section"><div class="section-h"><h2>🗺️ Animated Map — Interest by Region</h2><div class="chip">Scope: {scope_label}</div></div>', unsafe_allow_html=True)
    btn_map = st.button("Fetch Regions / Map", key="btn_ts_map", use_container_width=True)
    if btn_map:
        frames_live = fetch_frames(keywords[0], ctrl["ts_months"], geo_code, resolution=("CITY" if city_filter else "COUNTRY"))
        st.session_state["ts_frames"] = frames_live

    frames_show = st.session_state.get("ts_frames", pd.DataFrame())
    if frames_show is None or frames_show.empty:
        st.caption("Regional map: live data not available, showing demo frames.")
        frames_show = demo_frames(keywords[0], ctrl["ts_months"])
    if isinstance(frames_show, pd.DataFrame) and city_filter:
        mask = frames_show["region"].astype(str).str.contains(city_filter, case=False, na=False)
        frames_show = frames_show[mask]
    st.plotly_chart(
        animated_choropleth(frames_show),
        use_container_width=True,
        key=chart_key("ts_map", keywords[0], ctrl["ts_months"], geo_code, ("CITY" if city_filter else "COUNTRY"), btn_map),
    )
    render_ts_map_extras(st=st, ctx=dict(frames_df=frames_show, timeframe=ctrl["ts_timeframe"], geo_code=geo_code))
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Related Queries / Word cloud ───────────────────────────────────────────
    st.markdown(f'<div class="section"><div class="section-h"><h2>🔤 Related Queries — Word Cloud</h2><div class="chip">Scope: {scope_label}</div></div>', unsafe_allow_html=True)
    btn_rel = st.button("Fetch Related Queries", key="btn_ts_related", use_container_width=True)
    if btn_rel:
        st.session_state["ts_related"] = fetch_related(keywords[0], geo_code)

    rq = st.session_state.get("ts_related", {})
    if not rq:
        st.caption("Related queries: live data not available, showing demo.")
        rq = demo_related()
    img = wordcloud_from_related(rq.get("top"), rq.get("rising"))
    src = "Live" if isinstance(st.session_state.get("ts_iot"), pd.DataFrame) and not st.session_state["ts_iot"].empty else "Demo"
    st.image(img, caption=f"Related queries — {keywords[0]} ({src})", use_container_width=True)
    render_ts_related_extras(st=st, ctx=dict(rq_dict=rq, timeframe=ctrl["ts_timeframe"], geo_code=geo_code))
    st.markdown("</div>", unsafe_allow_html=True)

def render_job_market(ctrl: dict):
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

    # Roles entry + Apply
    default_roles = st.session_state.get("applied_roles", ["Data Analyst","Data Scientist"])
    all_roles = [
        "Data Analyst", "Data Scientist", "Software Developer",
        "Full Stack Developer", "Data Engineer",
        "Business Analyst", "Machine Learning Engineer",
    ]
    roles = st.multiselect("Job Roles (max 5)", all_roles, default=default_roles, key="jm_roles")[:5] or all_roles[:2]
    colA, colB = st.columns([1,1])
    with colA:
        if st.button("Apply Roles", key="btn_apply_roles"):
            st.session_state["applied_roles"] = roles
            st.toast("Roles applied. Click fetch buttons in sections to get live data.", icon="✅")
    with colB:
        st.caption("Applied: " + ", ".join(st.session_state.get("applied_roles", roles)))

    eff_roles = st.session_state.get("applied_roles", roles)
    eff_geo_text = effective_geo_text(ctrl["quick"], ctrl["geo_text"])
    geo_code, scope_label, city_filter = resolve_geo(eff_geo_text)

    tabs = st.tabs(["Overview","Trends by Date","Regional Map","Top & Rising","Job Openings"])

    # Overview
    with tabs[0]:
        st.markdown(f'<div class="section"><div class="section-h"><h2>Interest Over Time</h2><div class="chip">Scope: {scope_label}</div></div>', unsafe_allow_html=True)
        col1, col2 = st.columns([1,3])
        with col1:
            btn = st.button("⚡ Fetch Job Role Time Series", key="btn_jm_iot", use_container_width=True)
            st.caption("Powered by: " + get_source())
        with col2:
            st.caption(f"Timeframe: **{ctrl['jm_timeframe']}** • Geo: **{eff_geo_text}**")

        if btn:
            st.session_state["jm_iot"] = fetch_iot(tuple(eff_roles), ctrl["jm_timeframe"], geo_code)

        df_live = st.session_state.get("jm_iot", pd.DataFrame())
        df = df_live if (isinstance(df_live, pd.DataFrame) and not df_live.empty) else demo_ts(tuple(eff_roles[:2]))

        cols = [c for c in df.columns if c != "date"]
        first = cols[0]
        now_val = int(df[first].iloc[-1]); avg7 = int(df[first].rolling(7, min_periods=1).mean().iloc[-1])
        k1,k2,k3 = st.columns(3)
        with k1: st.markdown(kpi_card("Now", f"{now_val}"), unsafe_allow_html=True)
        with k2: st.markdown(kpi_card("7-day Avg", f"{avg7}"), unsafe_allow_html=True)
        with k3: st.markdown(f"<span class='chip'>Timeframe: {ctrl['jm_timeframe']}</span>", unsafe_allow_html=True)

        st.plotly_chart(
            colored_line(df, cols, title=None),
            use_container_width=True,
            key=chart_key("jm_overview_colored", ctrl["jm_timeframe"], geo_code, tuple(cols), btn),
        )
        st.caption("Tip: switch to *Trends Studio* to deep-dive these roles as keywords.")
        render_jm_overview_extras(st=st, ctx=dict(ts_df=df, timeframe=ctrl["jm_timeframe"], geo_code=geo_code))
        st.markdown("</div>", unsafe_allow_html=True)

    # Trends by Date
    with tabs[1]:
        st.markdown(f'<div class="section"><div class="section-h"><h2>Trends by Date</h2><div class="chip">Scope: {scope_label}</div></div>', unsafe_allow_html=True)
        df_live = st.session_state.get("jm_iot", pd.DataFrame())
        df = df_live if (isinstance(df_live, pd.DataFrame) and not df_live.empty) else demo_ts(tuple(eff_roles[:2]))
        all_series = [c for c in df.columns if c != "date"]
        pick = st.multiselect("Series to show", all_series, default=all_series[:min(3,len(all_series))], key="jm_pick")
        if pick:
            st.plotly_chart(
                colored_line(df[["date"] + pick], pick, title=None),
                use_container_width=True,
                key=chart_key("jm_trends_by_date_colored", ctrl["jm_timeframe"], geo_code, tuple(pick)),
            )
        else:
            st.warning("Select at least one series.")
        st.markdown("</div>", unsafe_allow_html=True)

    # Regional Map
    with tabs[2]:
        st.markdown(f'<div class="section"><div class="section-h"><h2>Interest by Region (animated)</h2><div class="chip">Scope: {scope_label}</div></div>', unsafe_allow_html=True)
        btn_map = st.button("Fetch Regions / Map (first role)", key="btn_jm_map", use_container_width=True)
        if btn_map:
            frames_live = fetch_frames(eff_roles[0], ctrl["jm_months"], geo_code, resolution=("CITY" if city_filter else "COUNTRY"))
            st.session_state["jm_frames"] = frames_live

        frames = st.session_state.get("jm_frames", pd.DataFrame())
        if frames is None or frames.empty:
            st.caption("Regional map: live data not available, showing demo frames.")
            frames = demo_frames(eff_roles[0], ctrl["jm_months"])
        if isinstance(frames, pd.DataFrame) and city_filter:
            mask = frames["region"].astype(str).str.contains(city_filter, case=False, na=False)
            frames = frames[mask]
        st.plotly_chart(
            animated_choropleth(frames),
            use_container_width=True,
            key=chart_key("jm_map", eff_roles[0], ctrl["jm_months"], geo_code, ("CITY" if city_filter else "COUNTRY"), btn_map),
        )
        st.caption(("Showing regions for: **" + eff_roles[0] + "**") + (f" • City: **{city_filter}**" if city_filter else ""))
        render_jm_map_extras(st=st, ctx=dict(frames_df=frames, timeframe=ctrl["jm_timeframe"], geo_code=geo_code))

    # Top & Rising
    with tabs[3]:
        st.markdown(f'<div class="section"><div class="section-h"><h2>Top & Rising Related Keywords</h2><div class="chip">Scope: {scope_label}</div></div>', unsafe_allow_html=True)
        btn_rel = st.button("Fetch Related (first role)", key="btn_jm_related", use_container_width=True)
        if btn_rel:
            st.session_state["jm_related"] = fetch_related(eff_roles[0], geo_code)
        rq = st.session_state.get("jm_related", {})
        if not rq:
            st.caption("Related queries: live data not available, showing demo.")
            rq = demo_related()
        top_df  = rq.get("top") if isinstance(rq, dict) else None
        rising_df = rq.get("rising") if isinstance(rq, dict) else None
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
        st.markdown("</div>", unsafe_allow_html=True)

    # Job Openings
    with tabs[4]:
        st.markdown('<div class="section"><div class="section-h"><h2>Job Openings</h2></div>', unsafe_allow_html=True)
        loc_input = st.text_input("Job search location (optional — used to prefill external links)",
                                  value=(city_filter or scope_label.replace("Worldwide","")), key="jm_job_loc")
        loc_q = quote(loc_input) if loc_input else ""
        for role in eff_roles:
            role_q = quote(role)
            lkdn = f"https://www.linkedin.com/jobs/search/?keywords={role_q}" + (f"&location={loc_q}" if loc_q else "")
            indeed = f"https://www.indeed.com/jobs?q={role_q}" + (f"&l={loc_q}" if loc_q else "")
            seek = f"https://www.seek.com.au/{role.replace(' ','-')}-jobs" + (f"?where={loc_q}" if loc_q else "")
            st.markdown(f"**{role}** — [LinkedIn]({lkdn}) • [Seek]({seek}) • [Indeed]({indeed})")
        st.caption("Openings are external links; use on-site filters to refine.")
        render_jm_openings_extras(st=st, ctx=dict(timeframe=ctrl["jm_timeframe"], geo_code=geo_code, city_filter=city_filter))

# ─────────────────────────── Render whichever view ─────────────────────────────
def main():
    # Build sidebar ONCE
    ctrl = render_sidebar_once()
    # Render both views; each returns early if not selected
    render_trends_studio(ctrl)
    render_job_market(ctrl)
    st.caption("© 2025 · Trends Hub · Built with Streamlit · PyTrends · SerpAPI")

if __name__ == "__main__":
    main()
