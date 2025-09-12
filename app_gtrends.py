from __future__ import annotations

import time
from datetime import datetime, timedelta
from typing import Iterable, Tuple, Dict, Any, List, Optional
from urllib.parse import quote

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from wordcloud import WordCloud
from pytrends.request import TrendReq
from pytrends.exceptions import TooManyRequestsError

# ───────────────────── Page & theme ─────────────────────
st.set_page_config(page_title="Trends Hub", page_icon="📊", layout="wide")
st.markdown(
    """
<style>
:root{--card-brd:#e9eaf0;--muted:#64748b;--ink:#0f172a;}
html,body,.stApp{background:#fff}
.hero{background:radial-gradient(1200px 420px at 12% -15%, rgba(109,40,217,.12),transparent),
linear-gradient(90deg, rgba(109,40,217,.10), rgba(37,99,235,.08));
border:1px solid var(--card-brd);border-radius:18px;padding:18px 22px;margin:10px 0 16px;box-shadow:0 8px 26px rgba(17,24,39,.06)}
.hero h1{margin:0;font-size:1.9rem}
.subtle{color:#475569;font-size:.95rem;margin-top:6px}
.section{background:linear-gradient(180deg,#fff,#fbfcff);border:1px solid var(--card-brd);border-radius:18px;padding:16px;margin:6px 0 18px;box-shadow:0 8px 24px rgba(17,24,39,.05)}
.section-h{display:flex;align-items:center;justify-content:space-between;margin-bottom:10px}
.section-h h2{margin:0;font-size:1.25rem;color:var(--ink)}
.chip{display:inline-block;padding:6px 10px;border-radius:999px;font-size:.75rem;background:#eef2ff;border:1px solid #c7d2fe;color:#3730a3}
.kpi{display:flex;flex-direction:column;gap:6px;padding:12px 14px;border-radius:12px;background:linear-gradient(180deg,#fff,#f9fbff);border:1px solid var(--card-brd)}
.kpi-label{color:#64748b;font-size:.72rem;text-transform:uppercase}
.kpi-value{font-size:1.32rem;font-weight:800;color:#111827}
.badge-ok{background:#ecfdf5;border:1px solid #10b981;color:#065f46;border-radius:10px;padding:6px 10px;font-size:.8rem;display:inline-block;margin-top:6px}
.err{background:#fef2f2;border:1px solid #fecaca;color:#991b1b;border-radius:10px;padding:8px 10px;font-size:.85rem;display:inline-block;margin:6px 0}
</style>
""",
    unsafe_allow_html=True,
)

# ───────────────────── Small utils ─────────────────────
def chart_key(prefix: str, *parts) -> str:
    return (prefix + ":" + "|".join(str(p) for p in parts if p is not None))[:200]

def kpi_card(label: str, value: str):
    return f"<div class='kpi'><div class='kpi-label'>{label}</div><div class='kpi-value'>{value}</div></div>"

def to_list(x: Iterable[str] | str | None) -> list[str]:
    if x is None: return []
    if isinstance(x, str): return [x]
    return list(x)

def country_name_to_iso2(name: str | None) -> Optional[str]:
    if not name: return None
    m = {"australia":"AU","united states":"US","usa":"US","united kingdom":"GB","uk":"GB","india":"IN",
         "canada":"CA","singapore":"SG","new zealand":"NZ","germany":"DE","france":"FR","japan":"JP","brazil":"BR"}
    return m.get(name.strip().lower())

def resolve_geo(user_input: str) -> Tuple[str, str, Optional[str]]:
    s = (user_input or "").strip()
    if not s or s.lower() in {"world","worldwide","global"}: return "", "Worldwide", None
    if s.lower()=="perth": return "AU","Perth, Australia","Perth"
    if "," in s:
        city,country = s.split(",",1)
        iso = country_name_to_iso2(country) or (country.strip().upper() if len(country.strip())==2 else None)
        return (iso or ""), f"{city.strip()}, {country.strip()}", city.strip()
    iso = country_name_to_iso2(s) or (s.upper() if len(s)==2 else "")
    return (iso or ""), (s.title() if iso else s), None

# ───────────────────── Fallback data (instant visuals) ─────────────────────
def fb_ts(keys=("AI","Data"), days=365) -> pd.DataFrame:
    end = datetime.utcnow().date(); start = end - timedelta(days=days)
    rng = pd.date_range(start, end, freq="D")
    df = pd.DataFrame({"date": rng})
    for i, kw in enumerate(keys[:5]):
        base = 45 + i*3
        s = base + 18*np.sin(np.linspace(0,6,len(rng))) + np.random.RandomState(33+i).randn(len(rng))*3
        s[int(len(rng)*0.30)] += 25; s[int(len(rng)*0.55)] += 18
        df[kw] = np.clip(s,0,100).round().astype(int)
    return df

def fb_frames(keyword="AI", months_count=6) -> pd.DataFrame:
    end = pd.Period(datetime.utcnow().date(), freq="M")
    periods = pd.period_range(end=end, periods=months_count, freq="M")
    rows = []
    vals=[("Australia","AU",70),("United States","US",58),("India","IN",62),("United Kingdom","GB",50),("Canada","CA",49),("Germany","DE",43)]
    alt =[("Australia","AU",48),("United States","US",69),("India","IN",75),("United Kingdom","GB",41),("Canada","CA",55),("Germany","DE",38)]
    for i,p in enumerate(periods):
        use = vals if i%2==0 else alt
        for r,iso2,v in use:
            rows.append({"region":r,"value":v,"iso2":iso2,"date_frame":str(p)})
    return pd.DataFrame(rows)

def fb_related_for_roles(roles: List[str]) -> Dict[str, pd.DataFrame]:
    # lightweight synthetic related terms per role
    base = {
        "Data Analyst":["excel","sql","power bi","tableau","dashboard","etl","analytics"],
        "Data Scientist":["python","machine learning","pandas","statistics","modeling","jupyter"],
        "Software Developer":["javascript","react","node","git","docker","api","frontend","backend"],
        "Full Stack Developer":["react","node","postgres","devops","graphql","aws","nextjs"],
        "Data Engineer":["spark","airflow","kafka","dbt","warehouse","pipeline","azure"],
        "Business Analyst":["requirements","process","stakeholder","documentation","agile","jira"],
        "Machine Learning Engineer":["mlops","pytorch","tensorflow","feature store","deployment","huggingface"],
    }
    rows=[]
    for r in roles[:5]:
        for i,term in enumerate(base.get(r, [])):
            rows.append([f"{term}", 50 + (len(r)%7)*5 + (i%5)*3])
    top = pd.DataFrame(rows, columns=["query","value"]).groupby("query", as_index=False)["value"].sum().sort_values("value", ascending=False)
    rising = top.copy(); rising["value"] = (rising["value"]*1.3).round().astype(int)
    return {"top": top.head(40), "rising": rising.head(40)}

def fb_trending_daily():
    return pd.DataFrame({"query":["Weather radar","AFL finals","Fuel prices","Bitcoin price","Taylor Swift"]})

def fb_trending_rt():
    return pd.DataFrame({"title":["Perth weather update","Perth traffic","Local sports news","Concerts in Perth"]})

# ───────────────────── PyTrends live (only on button clicks) ─────────────────────
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18 Safari/605.1.15"

@st.cache_resource(show_spinner=False)
def get_client() -> TrendReq:
    return TrendReq(
        hl="en-US",
        tz=480,
        timeout=(10, 25),
        retries=3,
        backoff_factor=0.7,
        requests_args={"headers": {"User-Agent": UA}},
    )

def _payload(py: TrendReq, kw: List[str], timeframe: str, geo: str, slow: bool) -> bool:
    backoff = 1.0 if slow else 0.6
    tries = 6 if slow else 4
    for _ in range(tries):
        try:
            py.build_payload(kw_list=kw, timeframe=timeframe, geo=geo)
            return True
        except TooManyRequestsError:
            time.sleep(backoff); backoff *= 1.6
        except Exception:
            time.sleep(backoff); backoff *= 1.2
    return False

@st.cache_data(show_spinner=False, ttl=90)
def live_iot(keys: Tuple[str, ...], timeframe: str, geo: str, slow: bool) -> pd.DataFrame:
    try:
        py = get_client()
        if not _payload(py, list(keys)[:5], timeframe, geo, slow): return pd.DataFrame()
        df = py.interest_over_time()
        if df is None or df.empty: return pd.DataFrame()
        df = df.reset_index().rename(columns={"date":"date"})
        if "isPartial" in df.columns: df = df.drop(columns=["isPartial"])
        keep = ["date"] + [c for c in df.columns if c in list(keys)]
        return df[keep]
    except Exception:
        return pd.DataFrame()

@st.cache_data(show_spinner=False, ttl=90)
def live_frames(keyword: str, months: int, geo: str, resolution="COUNTRY", slow: bool=True) -> pd.DataFrame:
    try:
        py = get_client()
        out = []
        now = pd.Timestamp.utcnow().to_period("M")
        months = min(months, 3) if slow else months
        for i in range(months, 0, -1):
            end = (now - i + 1)
            tf = f"{end.start_time.date()} {end.end_time.date()}"
            if not _payload(py, [keyword], tf, geo, slow): continue
            df = py.interest_by_region(resolution=resolution, inc_low_vol=True, inc_geo_code=True)
            if df is None or df.empty: continue
            df = df.reset_index()
            if "geoCode" in df.columns: df = df.rename(columns={"geoCode":"iso2"})
            if "geoName" in df.columns: df = df.rename(columns={"geoName":"region"})
            if keyword in df.columns: df = df.rename(columns={keyword:"value"})
            if "value" not in df.columns:
                num_cols = df.select_dtypes(include=[np.number]).columns
                df["value"] = df[num_cols[0]] if len(num_cols) else 0
            df["value"] = pd.to_numeric(df["value"], errors="coerce").fillna(0)
            df["date_frame"] = str(end)
            out.append(df[["region","value"] + (["iso2"] if "iso2" in df.columns else []) + ["date_frame"]])
            time.sleep(0.25 if slow else 0.1)
        if not out: return pd.DataFrame()
        return pd.concat(out, ignore_index=True)
    except Exception:
        return pd.DataFrame()

@st.cache_data(show_spinner=False, ttl=90)
def live_related(keyword: str, geo: str, slow: bool) -> Dict[str, pd.DataFrame]:
    try:
        py = get_client()
        if not _payload(py, [keyword], "today 12-m", geo, slow): return {}
        rq = py.related_queries() or {}
        slot = rq.get(keyword) if isinstance(rq, dict) else None
        if not isinstance(slot, dict): return {}
        top = slot.get("top"); rising = slot.get("rising")
        return {
            "top": top if isinstance(top, pd.DataFrame) else pd.DataFrame(columns=["query","value"]),
            "rising": rising if isinstance(rising, pd.DataFrame) else pd.DataFrame(columns=["query","value"]),
        }
    except Exception:
        return {}

def _sanitize_related_df(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame(columns=["query","value"])
    out = df.copy()
    def fix(v):
        if pd.isna(v): return 0
        if isinstance(v,(int,float)): return int(v)
        s = str(v).strip().lower()
        if s=="breakout": return 120
        s = s.replace("%","")
        try: return int(round(float(s)))
        except: return 0
    out["value"] = out["value"].apply(fix).astype(int)
    out["query"] = out["query"].astype(str)
    return out

@st.cache_data(show_spinner=False, ttl=90)
def live_related_multi(roles: Tuple[str, ...], geo: str, slow: bool) -> Dict[str, pd.DataFrame]:
    # aggregate related queries for several roles
    agg_top: Dict[str,int] = {}
    agg_rise: Dict[str,int] = {}
    for r in roles[:5]:
        slot = live_related(r, geo, slow=slow)
        if not slot:
            time.sleep(0.15)
            continue
        top = _sanitize_related_df(slot.get("top", pd.DataFrame()))
        rising = _sanitize_related_df(slot.get("rising", pd.DataFrame()))
        for _,row in top.iterrows():
            q = row["query"].strip()
            if q: agg_top[q] = agg_top.get(q,0) + int(row["value"])
        for _,row in rising.iterrows():
            q = row["query"].strip()
            if q: agg_rise[q] = agg_rise.get(q,0) + int(row["value"])
        time.sleep(0.15 if slow else 0.05)
    top_df = pd.DataFrame([{"query":k,"value":v} for k,v in agg_top.items()]).sort_values("value", ascending=False)
    rising_df = pd.DataFrame([{"query":k,"value":v} for k,v in agg_rise.items()]).sort_values("value", ascending=False)
    return {"top": top_df.head(200), "rising": rising_df.head(200)}

# NEW: live trending daily & realtime used in section_trending
@st.cache_data(show_spinner=False, ttl=120)
def live_trending_daily(geo_name: str = "australia", slow: bool = True) -> pd.DataFrame:
    try:
        py = get_client()
        df = py.trending_searches(pn=geo_name)
        if df is not None and not df.empty:
            df = df.copy()
            df.columns = ["query"]
            return df
    except TooManyRequestsError:
        time.sleep(1.0 if slow else 0.3)
    except Exception:
        pass
    return pd.DataFrame()

@st.cache_data(show_spinner=False, ttl=120)
def live_trending_rt(geo_code: str = "AU", cat: str = "all", slow: bool = True) -> pd.DataFrame:
    try:
        py = get_client()
        df = py.realtime_trending_searches(pn=geo_code, cat=cat)
        if df is not None and not df.empty:
            if "title" not in df.columns:
                df = df.rename(columns={df.columns[0]: "title"})
            return df
    except TooManyRequestsError:
        time.sleep(1.0 if slow else 0.3)
    except Exception:
        pass
    return pd.DataFrame()

# ───────────────────── Correlation helpers (keywords ↔ roles) ─────────────────────
def _batch(iterable: List[str], n: int = 5):
    for i in range(0, len(iterable), n):
        yield iterable[i:i+n]

@st.cache_data(show_spinner=False, ttl=120)
def fetch_keyword_series(keywords: Tuple[str, ...], timeframe: str, geo: str, slow: bool) -> pd.DataFrame:
    """Fetch IoT series for up to 5 keywords (PyTrends max per payload)."""
    return live_iot(tuple(keywords), timeframe, geo, slow=slow)

@st.cache_data(show_spinner=True, ttl=180)
def compute_keyword_role_correlations(
    roles: Tuple[str, ...],
    related_df: pd.DataFrame,
    timeframe: str,
    geo: str,
    slow: bool,
    max_keywords: int = 20
) -> pd.DataFrame:
    """Compute Pearson correlation between related keywords' IoT series and each role's IoT series."""
    if related_df is None or related_df.empty:
        return pd.DataFrame()

    roles = tuple([r for r in roles if isinstance(r, str) and r.strip()])[:5]
    roles_iot = fetch_keyword_series(roles, timeframe, geo, slow)
    if roles_iot is None or roles_iot.empty or "date" not in roles_iot.columns:
        return pd.DataFrame()

    rel_kw = (
        related_df["query"]
        .astype(str).str.strip().replace("", np.nan).dropna().drop_duplicates().tolist()
    )[:max_keywords]
    if not rel_kw:
        return pd.DataFrame()

    # Fetch keyword IoT in batches of 5
    kw_series: Dict[str, pd.Series] = {}
    for chunk in _batch(rel_kw, 5):
        dfk = fetch_keyword_series(tuple(chunk), timeframe, geo, slow)
        if dfk is None or dfk.empty or "date" not in dfk.columns:
            continue
        dfk = dfk.copy()
        dfk["date"] = pd.to_datetime(dfk["date"])
        for c in [c for c in dfk.columns if c != "date"]:
            kw_series[c] = dfk.set_index("date")[c].astype(float)
        time.sleep(0.15 if slow else 0.05)

    if not kw_series:
        return pd.DataFrame()

    roles_df = roles_iot.copy()
    roles_df["date"] = pd.to_datetime(roles_df["date"])
    roles_df = roles_df.drop_duplicates(subset=["date"]).sort_values("date").set_index("date")
    for r in roles:
        if r in roles_df.columns:
            roles_df[r] = pd.to_numeric(roles_df[r], errors="coerce").astype(float)

    out_rows = []
    for q, s_kw in kw_series.items():
        joint = roles_df.join(s_kw.rename("kw"), how="inner").dropna(how="any")
        if joint.empty:
            row = {"query": q}
            row.update({f"corr_{r}": np.nan for r in roles})
            out_rows.append(row)
            continue
        row = {"query": q}
        for r in roles:
            if r in joint.columns:
                try:
                    row[f"corr_{r}"] = float(joint["kw"].corr(joint[r]))
                except Exception:
                    row[f"corr_{r}"] = np.nan
            else:
                row[f"corr_{r}"] = np.nan
        out_rows.append(row)

    corr_df = pd.DataFrame(out_rows)

    def _best(row):
        vals = {r: row.get(f"corr_{r}") for r in roles}
        vals = {k:v for k,v in vals.items() if pd.notna(v)}
        if not vals:
            return ""
        best_role = max(vals, key=lambda k: vals[k])
        return f"{best_role} ({vals[best_role]:.2f})"
    corr_df["best_match"] = corr_df.apply(_best, axis=1)

    sort_cols = [f"corr_{r}" for r in roles]
    corr_df["_max_corr"] = corr_df[sort_cols].max(axis=1, skipna=True)
    corr_df = corr_df.sort_values("_max_corr", ascending=False).drop(columns="_max_corr")
    return corr_df

def render_correlation_heatmap(corr_df: pd.DataFrame, roles: Tuple[str, ...], title: str):
    """Renders a keyword x role correlation heatmap."""
    try:
        cols = [f"corr_{r}" for r in roles if f"corr_{r}" in corr_df.columns]
        if not cols or corr_df.empty:
            st.info("No correlation values available to render.")
            return
        mat = corr_df[["query"] + cols].set_index("query")
        fig = px.imshow(
            mat,
            aspect="auto",
            color_continuous_scale="RdBu",
            zmin=-1.0, zmax=1.0,
            labels=dict(color="Pearson r"),
            title=None,
        )
        fig.update_layout(template="plotly_white", margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.caption(f"Heatmap skipped: {e}")

# ───────────────────── Viz helpers ─────────────────────
def get_line_palette(name: str) -> List[str]:
    q = px.colors.qualitative
    palettes = {
        "Vivid": q.Bold + q.Set1 + q.Vivid,
        "Bright": q.Plotly + q.Bold,
        "Pastel": q.Pastel + q.Set3,
        "D3": q.D3,
        "G10": q.G10,
        "Dark24": q.Dark24,
    }
    return palettes.get(name, q.Set2 + q.Set1 + q.Pastel)

def line_with_spikes_colored(df: pd.DataFrame, series_cols: List[str], palette_name: str) -> go.Figure:
    fig = go.Figure()
    palette = get_line_palette(palette_name)
    date = pd.to_datetime(df["date"])
    for i, col in enumerate(series_cols):
        y = pd.to_numeric(df[col], errors="coerce")
        fig.add_trace(go.Scatter(x=date, y=y, mode="lines", name=col,
                                 line=dict(width=2, color=palette[i % len(palette)])))
    fig.update_layout(template="plotly_white", margin=dict(l=10,r=10,t=10,b=10),
                      legend=dict(orientation="h", y=1.02, x=0))
    fig.update_yaxes(title_text="Interest (0–100)")
    return fig

def animated_choropleth(frames: pd.DataFrame, scale: str) -> go.Figure:
    df = frames.copy()
    df["value"] = pd.to_numeric(df.get("value", 0), errors="coerce").fillna(0).clip(lower=0)
    if "date_frame" in df.columns:
        df["date_frame"] = df["date_frame"].astype(str)
        df = df.sort_values("date_frame")

    loccol = "region" if "region" in df.columns else ("iso3" if "iso3" in df.columns else None)
    locmode = "country names" if loccol == "region" else ("ISO-3" if loccol == "iso3" else None)
    if loccol is None:
        loccol, locmode = "region", "country names"

    vmax = float(df["value"].max()) if "value" in df.columns else 100.0
    vmax = max(50.0, vmax)

    fig = px.choropleth(
        df,
        locations=loccol,
        locationmode=locmode,
        color="value",
        hover_name=("region" if "region" in df.columns else loccol),
        animation_frame=("date_frame" if "date_frame" in df.columns else None),
        color_continuous_scale=scale,
        range_color=(0, vmax),
        scope="world",
        projection="natural earth",
    )
    fig.update_layout(template="plotly_white", margin=dict(l=10, r=10, t=10, b=10))
    return fig

def wordcloud_from_related(top_df: pd.DataFrame | None, rising_df: pd.DataFrame | None, max_words: int, colormap: str):
    def bag(df):
        if df is None or df.empty: return {}
        t = df.copy()
        t["value"] = pd.to_numeric(t["value"], errors="coerce").fillna(0).astype(int)
        t["query"] = t["query"].astype(str)
        d = {}
        for _,r in t.iterrows():
            q = r["query"].strip()
            if q: d[q] = d.get(q,0) + int(r["value"])
        return d
    b = bag(top_df); rb = bag(rising_df)
    for k,v in rb.items(): b[k] = b.get(k,0) + int(v*0.25)  # small weight from rising
    if not b: b = {"no data":1}
    return WordCloud(width=1200, height=460, background_color="white",
                     max_words=int(max_words), colormap=colormap).generate_from_frequencies(b).to_image()

# ───────────────────── Sidebar (now includes color + cloud controls) ─────────────────────
def sidebar() -> Dict[str, Any]:
    with st.sidebar:
        st.header("🛠️ Controls")
        view = st.selectbox("Choose view", ["Trends Studio","Job Market"], key="view")

        st.markdown("**Region**")
        q = st.radio("Region", ["Australia","Perth","Worldwide","Custom"], index=0, key="region")
        geo_text = st.text_input("Custom (country/ISO-2 or Perth)", value="Australia", key="geo_text") if q=="Custom" else q

        st.markdown("**Timeframes**")
        if view=="Trends Studio":
            ts_timeframe = st.selectbox("Trends timeframe", ["today 12-m","today 3-m","now 7-d","today 5-y"], index=0, key="ts_tf")
            ts_months = st.slider("Animated Map – months", 3, 12, 7, key="ts_months")
            jm_timeframe, jm_months = None, None
        else:
            jm_timeframe = st.selectbox("Job Market timeframe", ["now 7-d","today 3-m","today 12-m","today 5-y"], index=2, key="jm_tf")
            jm_months = st.slider("Animated Map – months", 3, 12, 6, key="jm_months")
            ts_timeframe, ts_months = None, None

        st.markdown("**Visual options**")
        line_theme = st.selectbox("Line palette", ["Vivid","Bright","Pastel","D3","G10","Dark24"], index=0, key="line_theme")
        map_scale = st.selectbox("Map color scale", 
                                 ["Turbo","Viridis","Plasma","Cividis","Magma","Inferno","Ice","Mint","Teal","Sunset","Portland","Picnic","Jet","Blues"],
                                 index=0, key="map_scale")
        cloud_words = st.slider("Word cloud: max words", min_value=80, max_value=400, value=250, step=10, key="wc_max")
        cloud_map = st.selectbox("Word cloud colormap", 
                                 ["tab20","viridis","plasma","inferno","magma","cividis","turbo","prism","Paired","Set2","Dark2","tab10"],
                                 index=0, key="wc_cmap")

        slow = st.checkbox("🐢 Slow mode (avoid 429)", value=True)
        st.button("🔄 Force refresh caches", on_click=lambda: [st.cache_data.clear()])

        return {"view":view, "geo_text":geo_text, "ts_timeframe":ts_timeframe, "ts_months":ts_months,
                "jm_timeframe":jm_timeframe, "jm_months":jm_months, "slow":slow,
                "line_theme":line_theme, "map_scale":map_scale, "wc_max":cloud_words, "wc_cmap":cloud_map}

# ───────────────────── Sections (fallback first + Fetch Live buttons) ─────────────────────
def section_trending(scope_label: str, city_filter: Optional[str], slow: bool):
    st.markdown(f'<div class="section"><div class="section-h"><h2>🔥 Top Trending Searches Today</h2><div class="chip">Scope: {scope_label}</div></div>', unsafe_allow_html=True)
    c1,c2 = st.columns(2)
    with c1:
        st.caption("Daily Trending — Australia")
        sample = fb_trending_daily()
        st.markdown("\n".join([f"- {q}" for q in sample["query"].astype(str).tolist()]))
        if st.button("Fetch Live (Daily)", key="btn_trend_daily"):
            live = live_trending_daily("australia", slow=slow)
            if not live.empty:
                st.markdown("---")
                st.markdown("\n".join([f"- {q}" for q in live["query"].astype(str).tolist()[:10]]))
                st.markdown("<span class='badge-ok'>Live ✓</span>", unsafe_allow_html=True)
    with c2:
        city = city_filter or "Perth"
        st.caption(f"Realtime Trending — filtered for “{city}”")
        sample = fb_trending_rt()
        st.markdown("\n".join([f"- **{t}**" for t in sample["title"].astype(str).tolist()]))
        if st.button("Fetch Live (Realtime)", key="btn_trend_rt"):
            live = live_trending_rt("AU","all", slow=slow)
            if not live.empty:
                name = "title" if "title" in live.columns else live.columns[0]
                mask = live[name].astype(str).str.contains(city, case=False, na=False)
                show = live[mask] if mask.any() else live
                st.markdown("---")
                st.markdown("\n".join([f"- **{t}**" for t in show[name].astype(str).tolist()[:8]]))
                st.markdown("<span class='badge-ok'>Live ✓</span>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

def section_iot(title: str, keywords: List[str], timeframe: str, geo_code: str, slow: bool, line_theme: str):
    st.markdown(f'<div class="section"><div class="section-h"><h2>{title}</h2><div class="chip">Timeframe: {timeframe}</div></div>', unsafe_allow_html=True)
    df = fb_ts(tuple(keywords[:5]))
    cols = [c for c in df.columns if c!="date"]
    k1,k2,k3 = st.columns([0.9,2.2,1.1])
    with k1:
        now_vals = [int(df[c].iloc[-1]) for c in cols]
        avg_vals = [int(df[c].rolling(7, min_periods=1).mean().iloc[-1]) for c in cols]
        st.markdown(kpi_card("Now", f"{now_vals[0]}"), unsafe_allow_html=True)
        st.markdown(kpi_card("7-day Avg", f"{avg_vals[0]}"), unsafe_allow_html=True)
    with k2:
        st.plotly_chart(line_with_spikes_colored(df, cols, line_theme), use_container_width=True,
                        key=chart_key("iot_sample", timeframe, geo_code, tuple(cols), line_theme))
    with k3:
        if st.button("Fetch Live (Series)", key=chart_key("btn_iot", timeframe, geo_code, tuple(cols))):
            live = live_iot(tuple(keywords), timeframe, geo_code, slow=slow)
            if not live.empty:
                st.plotly_chart(line_with_spikes_colored(live, [c for c in live.columns if c!="date"], line_theme),
                                use_container_width=True,
                                key=chart_key("iot_live", timeframe, geo_code, tuple(live.columns), line_theme))
                st.markdown("<span class='badge-ok'>Live ✓</span>", unsafe_allow_html=True)
                st.download_button("📥 Download Series (CSV)", live.to_csv(index=False).encode("utf-8"),
                                   file_name="interest_over_time.csv", mime="text/csv")
    st.markdown("</div>", unsafe_allow_html=True)

def section_map(series_name: str, months: int, geo_code: str, city_filter: Optional[str], slow: bool, map_scale: str):
    st.markdown(f'<div class="section"><div class="section-h"><h2>🗺️ Animated Map — Interest by Region</h2><div class="chip">Series: {series_name}</div></div>', unsafe_allow_html=True)
    demo = fb_frames(series_name, months)
    st.plotly_chart(animated_choropleth(demo, map_scale), use_container_width=True,
                    key=chart_key("map_sample", series_name, months, geo_code, map_scale))
    if st.button("Fetch Live (Regions)", key=chart_key("btn_map", series_name, months, geo_code)):
        frames = live_frames(series_name, months, geo_code, resolution=("CITY" if city_filter else "COUNTRY"),
                             slow=slow)
        if isinstance(frames, pd.DataFrame) and not frames.empty:
            show = frames
            if city_filter and "region" in show.columns:
                show = show[show["region"].astype(str).str.contains(city_filter, case=False, na=False)]
            st.plotly_chart(animated_choropleth(show, map_scale), use_container_width=True,
                            key=chart_key("map_live", series_name, months, geo_code, map_scale))
            st.markdown("<span class='badge-ok'>Live ✓</span>", unsafe_allow_html=True)
            st.download_button("⬇️ Download Top Regions (CSV)", show.to_csv(index=False).encode("utf-8"),
                               "regions_all_frames.csv", "text/csv")
    st.markdown("</div>", unsafe_allow_html=True)

def section_related(keyword: str, geo_code: str, slow: bool, wc_max: int, wc_cmap: str):
    st.markdown(f'<div class="section"><div class="section-h"><h2>🔤 Related Queries — Word Cloud</h2><div class="chip">Keyword: {keyword}</div></div>', unsafe_allow_html=True)
    # fallback
    demo_top = pd.DataFrame({"query":["what is ai","data analytics","ai tools","gpt tutorial","vector db","langchain","openai"],"value":[80,65,50,40,35,32,30]})
    demo_ris = pd.DataFrame({"query":["ai agents","gpt-4o","prompt ideas","streamlit ai","llama","rag"],"value":[120,100,95,60,55,52]})
    img = wordcloud_from_related(demo_top, demo_ris, wc_max, wc_cmap)
    st.image(img, caption=f"Related queries — {keyword}", use_container_width=True)

    if st.button("Fetch Live (Related)", key=chart_key("btn_related", keyword, geo_code)):
        rq = live_related(keyword, geo_code, slow=slow)
        if rq:
            top = _sanitize_related_df(rq.get("top", pd.DataFrame()))
            rising = _sanitize_related_df(rq.get("rising", pd.DataFrame()))
            img = wordcloud_from_related(top, rising, wc_max, wc_cmap)
            st.image(img, caption=f"Related queries — {keyword}", use_container_width=True)
            st.markdown("<span class='badge-ok'>Live ✓</span>", unsafe_allow_html=True)
            if not top.empty or not rising.empty:
                c1,c2 = st.columns(2)
                with c1:
                    st.subheader("Top keywords")
                    st.dataframe(top.head(100), use_container_width=True, height=360)
                with c2:
                    st.subheader("Rising keywords")
                    st.dataframe(rising.head(100), use_container_width=True, height=360)
    st.markdown("</div>", unsafe_allow_html=True)

# ───────────────────── Views ─────────────────────
def trends_studio(ctrl: Dict[str, Any]):
    st.markdown("""<div class="hero"><h1>✨ Trends Studio</h1>
    <div class="subtle">Loads instantly • Press “Fetch Live” in any section</div></div>""", unsafe_allow_html=True)
    kw_text = st.text_input("Keywords (comma-separated, max 5)", "AI, Data", key="ts_kw")
    keywords = [x.strip() for x in kw_text.split(",") if x.strip()][:5] or ["AI"]
    kw_for_related = st.selectbox("Word cloud keyword", options=keywords, index=0, key="kw_wc_pick")

    geo_code, scope_label, city_filter = resolve_geo(ctrl["geo_text"])
    section_trending(scope_label, city_filter, slow=ctrl["slow"])
    section_iot("📈 Interest Over Time (annotated)", keywords, ctrl["ts_timeframe"], geo_code, slow=ctrl["slow"], line_theme=ctrl["line_theme"])
    series_choice = st.selectbox("Series for map", options=keywords, index=0)
    section_map(series_choice, ctrl["ts_months"], geo_code, city_filter, slow=ctrl["slow"], map_scale=ctrl["map_scale"])
    section_related(kw_for_related, geo_code, slow=ctrl["slow"], wc_max=ctrl["wc_max"], wc_cmap=ctrl["wc_cmap"])

def job_market(ctrl: Dict[str, Any]):
    st.markdown("""<div class="hero"><h1>Trends Studio – Job Market</h1>
    <div class="subtle">Loads instantly • Press “Fetch Live” per tab</div></div>""", unsafe_allow_html=True)

    job_roles = ["Data Analyst","Data Scientist","Software Developer","Full Stack Developer","Data Engineer","Business Analyst","Machine Learning Engineer"]
    roles = st.multiselect("Job Roles (max 5)", job_roles, default=job_roles[:2], key="jm_roles")[:5] or job_roles[:2]
    geo_code, scope_label, city_filter = resolve_geo(ctrl["geo_text"])

    tabs = st.tabs(["Overview","Trends by Date","Regional Map","Top & Rising","Job Openings"])

    # Overview
    with tabs[0]:
        st.markdown(f'<div class="section"><div class="section-h"><h2>Interest Over Time (annotated)</h2><div class="chip">Scope: {scope_label}</div></div>', unsafe_allow_html=True)
        df = fb_ts(tuple(roles[:5]))
        cols = [c for c in df.columns if c!="date"]
        first = cols[0]
        now_val = int(df[first].iloc[-1]) if not df.empty else 0
        avg7 = int(df[first].rolling(7, min_periods=1).mean().iloc[-1]) if not df.empty else 0
        k1,k2,k3 = st.columns(3)
        with k1: st.markdown(kpi_card("Now", f"{now_val}"), unsafe_allow_html=True)
        with k2: st.markdown(kpi_card("7-day Avg", f"{avg7}"), unsafe_allow_html=True)
        with k3: st.markdown(f"<span class='chip'>Timeframe: {ctrl['jm_timeframe']}</span>", unsafe_allow_html=True)
        st.plotly_chart(line_with_spikes_colored(df, cols, ctrl["line_theme"]), use_container_width=True,
                        key=chart_key("jm_overview_sample", ctrl["jm_timeframe"], geo_code, tuple(cols), ctrl["line_theme"]))
        if st.button("Fetch Live (Overview series)", key="btn_jm_overview_live"):
            live = live_iot(tuple(roles), ctrl["jm_timeframe"], geo_code, slow=ctrl["slow"])
            if not live.empty:
                st.plotly_chart(line_with_spikes_colored(live, [c for c in live.columns if c!="date"], ctrl["line_theme"]),
                                use_container_width=True,
                                key=chart_key("jm_overview_live", ctrl["jm_timeframe"], geo_code, ctrl["line_theme"]))
                st.markdown("<span class='badge-ok'>Live ✓</span>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Overview related insights (word cloud + tables) for SELECTED ROLES (combined)
        st.markdown(f'<div class="section"><div class="section-h"><h2>Related Insights (combined roles)</h2><div class="chip">{", ".join(roles)}</div></div>', unsafe_allow_html=True)
        demo = fb_related_for_roles(roles)
        img = wordcloud_from_related(demo["top"], demo["rising"], ctrl["wc_max"], ctrl["wc_cmap"])
        st.image(img, caption="Related queries — combined roles", use_container_width=True)
        if st.button("Fetch Live (Related for roles)", key="btn_jm_overview_related_live"):
            agg = live_related_multi(tuple(roles), geo_code, slow=ctrl["slow"])
            if agg:
                img = wordcloud_from_related(agg["top"], agg["rising"], ctrl["wc_max"], ctrl["wc_cmap"])
                st.image(img, caption="Related queries — combined roles", use_container_width=True)
                st.markdown("<span class='badge-ok'>Live ✓</span>", unsafe_allow_html=True)
                c1,c2 = st.columns(2)
                with c1:
                    st.write("### Top (combined)")
                    st.dataframe(agg["top"].head(150), use_container_width=True, height=360)
                with c2:
                    st.write("### Rising (combined)")
                    st.dataframe(agg["rising"].head(150), use_container_width=True, height=360)

    # Trends by Date
    with tabs[1]:
        st.markdown(f'<div class="section"><div class="section-h"><h2>Trends by Date</h2><div class="chip">Scope: {scope_label}</div></div>', unsafe_allow_html=True)
        df2 = fb_ts(tuple(roles[:5]))
        all_series = [c for c in df2.columns if c != "date"]
        pick = st.multiselect("Series to show", all_series, default=all_series[:min(3,len(all_series))], key="jm_pick")
        if pick:
            st.plotly_chart(line_with_spikes_colored(df2[["date"] + pick], pick, ctrl["line_theme"]),
                            use_container_width=True,
                            key=chart_key("jm_trends_by_date_sample", ctrl["jm_timeframe"], geo_code, tuple(pick), ctrl["line_theme"]))
        if st.button("Fetch Live (Selected series)", key="btn_jm_trend_live"):
            live = live_iot(tuple(roles), ctrl["jm_timeframe"], geo_code, slow=ctrl["slow"])
            if not live.empty and pick:
                kept = ["date"] + [c for c in pick if c in live.columns]
                st.plotly_chart(line_with_spikes_colored(live[kept], [c for c in pick if c in live.columns], ctrl["line_theme"]),
                                use_container_width=True,
                                key=chart_key("jm_trends_by_date_live", ctrl["jm_timeframe"], geo_code, tuple(pick), ctrl["line_theme"]))
                st.markdown("<span class='badge-ok'>Live ✓</span>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Regional Map
    with tabs[2]:
        st.markdown(f'<div class="section"><div class="section-h"><h2>Interest by Region (animated)</h2><div class="chip">Scope: {scope_label}</div></div>', unsafe_allow_html=True)
        map_series = st.selectbox("Series to map", roles, index=0, key="jm_map_series")
        demo_frames = fb_frames(map_series, ctrl["jm_months"])
        st.plotly_chart(animated_choropleth(demo_frames, ctrl["map_scale"]), use_container_width=True,
                        key=chart_key("jm_map_sample", map_series, ctrl["jm_months"], geo_code, ctrl["map_scale"]))
        if st.button("Fetch Live (Regions)", key="btn_jm_map_live"):
            frames = live_frames(map_series, ctrl["jm_months"], geo_code,
                                 resolution=("CITY" if city_filter else "COUNTRY"),
                                 slow=ctrl["slow"])
            if isinstance(frames, pd.DataFrame) and not frames.empty:
                show = frames
                if city_filter and "region" in frames.columns:
                    show = frames[frames["region"].astype(str).str.contains(city_filter, case=False, na=False)]
                st.plotly_chart(animated_choropleth(show, ctrl["map_scale"]), use_container_width=True,
                                key=chart_key("jm_map_live", map_series, ctrl["jm_months"], geo_code, ctrl["map_scale"]))
                st.markdown("<span class='badge-ok'>Live ✓</span>", unsafe_allow_html=True)
                st.download_button("⬇️ Download Top Regions (CSV)", show.to_csv(index=False).encode("utf-8"),
                                   "job_regions_all_frames.csv", "text/csv")
        st.markdown("</div>", unsafe_allow_html=True)

    # Top & Rising (with correlations)
    with tabs[3]:
        st.markdown(f'<div class="section"><div class="section-h"><h2>Top & Rising Related Keywords</h2><div class="chip">Scope: {scope_label}</div></div>', unsafe_allow_html=True)

        target = st.selectbox("Source", options=["All selected roles (combined)"] + roles, index=0, key="jm_rel_source")
        mode = st.radio("Dataset", ["Top", "Rising"], horizontal=True, index=0, key="jm_rel_mode")
        max_kw = st.slider("Max keywords for correlation", 5, 40, 20, 5, key="jm_rel_maxkw")

        # Fallback
        if target == "All selected roles (combined)":
            demo = fb_related_for_roles(roles)
        else:
            demo = fb_related_for_roles([target])
        wc = wordcloud_from_related(demo["top"], demo["rising"], ctrl["wc_max"], ctrl["wc_cmap"])
        st.image(wc, caption=f"Related — {target}", use_container_width=True)

        # Live + correlations
        if st.button("Fetch Live (Top/Rising + Correlations)", key="btn_jm_related_live_tab"):
            if target == "All selected roles (combined)":
                rq = live_related_multi(tuple(roles), geo_code, slow=ctrl["slow"])
            else:
                rqo = live_related(target, geo_code, slow=ctrl["slow"])
                if rqo:
                    rqo = {"top": _sanitize_related_df(rqo.get("top", pd.DataFrame())),
                           "rising": _sanitize_related_df(rqo.get("rising", pd.DataFrame()))}
                rq = rqo

            if rq:
                # Word cloud
                img = wordcloud_from_related(rq.get("top"), rq.get("rising"), ctrl["wc_max"], ctrl["wc_cmap"])
                st.image(img, caption=f"Related — {target}", use_container_width=True)
                st.markdown("<span class='badge-ok'>Live ✓</span>", unsafe_allow_html=True)

                # Which set to analyze
                src_df = rq["top"] if mode == "Top" else rq["rising"]
                src_df = _sanitize_related_df(src_df)

                with st.spinner("Computing keyword↔role correlations..."):
                    corr_df = compute_keyword_role_correlations(
                        roles=tuple(roles),
                        related_df=src_df,
                        timeframe=ctrl["jm_timeframe"],
                        geo=geo_code,
                        slow=ctrl["slow"],
                        max_keywords=int(max_kw),
                    )

                if corr_df is not None and not corr_df.empty:
                    st.subheader("Correlation heatmap (keyword vs role)")
                    render_correlation_heatmap(corr_df, tuple(roles), title="Keyword↔Role correlations")

                    st.subheader("Correlation table")
                    role_cols = [f"corr_{r}" for r in roles if f"corr_{r}" in corr_df.columns]
                    disp = corr_df[["query", "best_match"] + role_cols].copy()
                    st.dataframe(disp, use_container_width=True, height=520)
                    st.download_button(
                        "⬇️ Download correlations (CSV)",
                        disp.to_csv(index=False).encode("utf-8"),
                        file_name=f"correlations_{mode.lower()}_{scope_label.replace(' ','_')}.csv",
                        mime="text/csv",
                    )
                else:
                    st.info("No correlations could be computed (try fewer roles/keywords, shorter timeframe, or enable Slow mode).")
            else:
                st.info("No live related keywords received. Try a different role set, timeframe, or region.")
        st.markdown("</div>", unsafe_allow_html=True)

    # Job Openings
    with tabs[4]:
        st.markdown('<div class="section"><div class="section-h"><h2>Job Openings</h2></div>', unsafe_allow_html=True)
        loc_input = st.text_input("Job search location (optional — used to prefill external links)",
                                  value=(city_filter or scope_label.replace("Worldwide","")), key="jm_job_loc")
        loc_q = quote(loc_input) if loc_input else ""
        for role in roles:
            role_q = quote(role)
            lkdn  = f"https://www.linkedin.com/jobs/search/?keywords={role_q}" + (f"&location={loc_q}" if loc_q else "")
            indeed= f"https://www.indeed.com/jobs?q={role_q}" + (f"&l={loc_q}" if loc_q else "")
            seek  = f"https://www.seek.com.au/{role.replace(' ','-')}-jobs" + (f"?where={loc_q}" if loc_q else "")
            st.markdown(f"**{role}** — [LinkedIn]({lkdn}) • [Seek]({seek}) • [Indeed]({indeed})")
        st.caption("Openings are external links; use on-site filters to refine.")
        st.markdown("</div>", unsafe_allow_html=True)

# ───────────────────── Main ─────────────────────
def main():
    ctrl = sidebar()
    if ctrl["view"]=="Trends Studio":
        trends_studio(ctrl)
    else:
        job_market(ctrl)
    st.caption("© 2025 · Trends Hub · PyTrends • Per-section “Fetch Live” • Color controls • Bigger wordclouds • Correlations")

if __name__ == "__main__":
    main()
