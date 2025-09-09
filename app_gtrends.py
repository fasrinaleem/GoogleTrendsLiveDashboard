# app_gtrends.py — ONE FILE VERSION
# One sidebar; 429-safe; country names + Perth + worldwide; dd2 extras inlined.

from datetime import datetime, timedelta
from urllib.parse import quote
from io import BytesIO

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from pytrends.exceptions import TooManyRequestsError

from trends import (
    get_client,
    interest_over_time,
    monthly_region_frames,
    related_queries,
    trending_today,
    trending_realtime,
)
from viz import (
    line_with_spikes,
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
</style>
""",
    unsafe_allow_html=True,
)

# small util
def chart_key(prefix: str, *parts) -> str:
    safe = "|".join(str(p) for p in parts if p is not None)
    return f"{prefix}:{safe}"[:200]


# ─────────────────────────── Geo resolver (country/city/world) ─────────────────
def resolve_geo(user_input: str):
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


# ─────────────────────────── Cache wrappers (gentle TTL) ───────────────────────
@st.cache_data(ttl=900, show_spinner=False)
def cached_interest_over_time(keys, tf, g):
    try:
        return interest_over_time(get_client(), keys, timeframe=tf, geo=g)
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=1800, show_spinner=False)
def cached_monthly_region_frames(key, months_, g, resolution="COUNTRY"):
    try:
        return monthly_region_frames(get_client(), keyword=key, months=months_, geo=g, resolution=resolution)
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=600, show_spinner=False)
def cached_trending_today():
    try:
        return trending_today(get_client(), geo="australia")
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=600, show_spinner=False)
def cached_trending_realtime():
    try:
        return trending_realtime(get_client(), geo="AU", cat="all")
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=900, show_spinner=False)
def cached_related_queries(key):
    try:
        return related_queries(get_client(), key)
    except Exception:
        return {}


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


# ───────────────────────── 429-SAFE Live fetch helpers ─────────────────────────
def fetch_iot(keys, tf, g, live=True, force=False):
    if not live:
        return pd.DataFrame()
    try:
        if force:
            return interest_over_time(get_client(), keys, timeframe=tf, geo=g)
        return cached_interest_over_time(keys, tf, g)
    except TooManyRequestsError:
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()

def fetch_frames(key, months, g, live=True, force=False, resolution="COUNTRY"):
    if not live:
        return pd.DataFrame()
    try:
        if force:
            return monthly_region_frames(get_client(), keyword=key, months=months, geo=g, resolution=resolution)
        return cached_monthly_region_frames(key, months, g, resolution=resolution)
    except TooManyRequestsError:
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()

def fetch_related(key, live=True, force=False):
    if not live:
        return {}
    try:
        if force:
            return related_queries(get_client(), key)
        return cached_related_queries(key)
    except TooManyRequestsError:
        return {}
    except Exception:
        return {}


# ─────────────────────────── Inlined “dd2.py” EXTRAS ───────────────────────────
# These used to live in dd2.py; now they’re local functions you can edit freely.
def render_ts_overview_extras(*, st=st, ctx: dict):
    """Extras under Trends Studio > Overview: rolling 30d average + exports."""
    df = ctx.get("ts_df", pd.DataFrame())
    if df is None or df.empty:
        return

    # Rolling 30-day average line chart
    roll = df.copy()
    for c in [c for c in roll.columns if c != "date"]:
        roll[c] = roll[c].rolling(30, min_periods=1).mean()

    fig30 = go.Figure()
    for c in [c for c in roll.columns if c != "date"]:
        fig30.add_trace(go.Scatter(x=roll["date"], y=roll[c], mode="lines", name=f"{c} (30d)"))
    fig30.update_layout(template="plotly_white", margin=dict(l=10, r=10, t=10, b=10))

    st.subheader("Extra • Rolling 30-day Average")
    st.plotly_chart(fig30, use_container_width=True, key=chart_key("ts_extra_roll30", ctx.get("timeframe"), ctx.get("geo_code")))

    # Export buttons (CSV + PNG if kaleido available)
    csv = df.to_csv(index=False).encode("utf-8")
    col1, col2 = st.columns([1,1])
    with col1:
        st.download_button("📥 Download Series (CSV)", data=csv, file_name="interest_over_time.csv", mime="text/csv", use_container_width=True)
    with col2:
        try:
            # PNG export (optional): requires plotly[kaleido]
            png_bytes = fig30.to_image(format="png", scale=2)
            st.download_button("🖼️ Download 30-day Avg (PNG)", data=png_bytes, file_name="rolling_30d.png", mime="image/png", use_container_width=True)
        except Exception:
            st.caption("Tip: to enable PNG export, install `pip install -U plotly[kaleido]`.")

def render_ts_map_extras(*, st=st, ctx: dict):
    """Extras under Trends Studio > Map: Top 10 regions from latest frame + CSV export."""
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
    """Extras under Trends Studio > Related: show Top table + CSV/PNG export of wordcloud."""
    rq = ctx.get("rq_dict", {}) or {}
    top_df = rq.get("top") if isinstance(rq, dict) else None
    rising_df = rq.get("rising") if isinstance(rq, dict) else None

    if isinstance(top_df, pd.DataFrame) and not top_df.empty:
        st.subheader("Extra • Top related (table)")
        st.dataframe(top_df.head(15), use_container_width=True, height=300)
        st.download_button("📥 Download Top Related (CSV)", data=top_df.to_csv(index=False).encode("utf-8"),
                           file_name="related_top.csv", mime="text/csv")

def render_jm_overview_extras(*, st=st, ctx: dict):
    """Extras under Job Market > Overview: latest snapshot bars + correlation heatmap."""
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
    """Extras under Job Market > Map: same as Trends (Top 10 regions)."""
    render_ts_map_extras(st=st, ctx=ctx)

def render_jm_related_extras(*, st=st, ctx: dict):
    """Extras under Job Market > Related: mirror of Trends."""
    render_ts_related_extras(st=st, ctx=ctx)

def render_jm_openings_extras(*, st=st, ctx: dict):
    """Placeholder for custom extras under Job Openings (e.g., scrape or API)."""
    # Keep lightweight to avoid violating site ToS. Links are provided in core UI.
    pass


# ─────────────────────────────── Views (UI) ────────────────────────────────────
def render_trends_studio():
    st.markdown(
        """
        <div class="hero">
          <h1>✨ Trends Studio</h1>
          <div class="subtle">Overview • Live trends • Regions • Top & Rising</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Sidebar
    with st.sidebar:
        st.subheader("Trends Studio – Controls")
        kw_text   = st.text_input("Keywords (comma-separated, max 5)", "AI, Data", key="ts_kw")
        timeframe = st.selectbox("Timeframe", ["today 12-m","today 3-m","now 7-d","today 5-y"], index=0, key="ts_timeframe")
        geo_text  = st.text_input("Region (country, ISO-2, city like 'Perth', or 'Worldwide')", "Australia", key="ts_geo_text")
        qp1, qp2, qp3 = st.columns(3)
        with qp1:
            if st.button("Australia", key="ts_qp_au"): st.session_state.ts_geo_text = "Australia"
        with qp2:
            if st.button("Perth", key="ts_qp_perth"): st.session_state.ts_geo_text = "Perth"
        with qp3:
            if st.button("Worldwide", key="ts_qp_world"): st.session_state.ts_geo_text = "Worldwide"
        months    = st.slider("Animated Map – months", 3, 12, 7, key="ts_months")
        st.markdown("---")
        live_api  = st.checkbox("Live API calls (disable to use Demo)", value=True, key="ts_live_api")
        colA, colB = st.columns([1,1])
        with colA:
            refresh = st.button("🔄 Refresh All", use_container_width=True, key="ts_refresh")
        with colB:
            auto = st.checkbox("Auto refresh each minute", value=False, key="ts_auto")

    if st.session_state.get("ts_auto"):
        st.markdown("""<script>setTimeout(function(){window.location.reload()},60000);</script>""", unsafe_allow_html=True)

    # Resolve geo & keywords
    geo_code, scope_label, city_filter = resolve_geo(st.session_state.get("ts_geo_text", geo_text))
    keywords = [x.strip() for x in kw_text.split(",") if x.strip()][:5] or ["AI"]

    # Trending
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown(f'<div class="section-h"><h2>🔥 Top Trending Searches Today</h2><div class="chip">Scope: {scope_label}</div></div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Fetch trending now", key="ts_trending_btn", use_container_width=True, disabled=not live_api):
            st.session_state.trend_daily = cached_trending_today()
            st.session_state.trend_rt = cached_trending_realtime()
            st.toast("Trending updated", icon="✅")
        daily = st.session_state.get("trend_daily")
        st.caption("Daily Trending — Australia")
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
            if "articles" in rt.columns:    mask = mask | contains("articles")
            filt = rt[mask] if (isinstance(mask, pd.Series) and mask.any()) else rt
            name_col = "title" if "title" in filt.columns else filt.columns[0]
            titles = filt[name_col].astype(str).tolist()[:8]
        if not titles:
            titles = [f"{city_name} weather update", f"{city_name} traffic", f"{city_name} events", "Local sports news"]
        st.markdown("\n".join([f"- **{t}**" for t in titles]))
    st.markdown('</div>', unsafe_allow_html=True)

    # Overview (line + KPIs + sparklines)
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown(f'<div class="section-h"><h2>📈 Interest Over Time (annotated)</h2><div class="chip">Scope: {scope_label}</div></div>', unsafe_allow_html=True)
    df_live = fetch_iot(tuple(keywords), timeframe, geo_code, live=live_api, force=refresh)
    if (refresh or live_api) and (df_live is None or df_live.empty):
        st.info("⚠️ Live data unavailable (rate limit or no data). Showing demo series.", icon="⚠️")
    df_overview = df_live if (isinstance(df_live, pd.DataFrame) and not df_live.empty) else demo_ts(tuple(keywords))
    series_cols = [c for c in df_overview.columns if c != "date"]

    k1,k2,k3 = st.columns([0.9, 2.2, 1.1])
    with k1:
        now_vals = [int(df_overview[c].iloc[-1]) for c in series_cols]
        avg_vals = [int(df_overview[c].rolling(7, min_periods=1).mean().iloc[-1]) for c in series_cols]
        label_now = "NOW (LIVE)" if (isinstance(df_live, pd.DataFrame) and not df_live.empty) else "NOW (DEMO)"
        st.markdown(kpi_card(label_now, f"{now_vals[0]}"), unsafe_allow_html=True)
        st.markdown(kpi_card("7-DAY AVG", f"{avg_vals[0]}"), unsafe_allow_html=True)
    with k2:
        st.plotly_chart(
            line_with_spikes(df_overview, series_cols),
            use_container_width=True,
            key=chart_key("ts_overview", timeframe, geo_code, tuple(series_cols), refresh),
        )
    with k3:
        st.write("**Sparklines**")
        for k in series_cols:
            st.caption(k)
            st.plotly_chart(
                sparkline(df_overview, k),
                use_container_width=True,
                theme=None,
                key=chart_key("ts_spark", k, timeframe, geo_code, refresh),
            )

    # Inlined extras (formerly dd2)
    render_ts_overview_extras(st=st, ctx=dict(
        timeframe=timeframe, geo_code=geo_code, ts_df=df_overview
    ))
    st.markdown('</div>', unsafe_allow_html=True)

    # Animated map
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown(f'<div class="section-h"><h2>🗺️ Animated Map — Interest by Region</h2><div class="chip">Scope: {scope_label}</div></div>', unsafe_allow_html=True)
    resolution = "CITY" if city_filter else "COUNTRY"
    frames_live = fetch_frames(keywords[0], st.session_state.get("ts_months", 7), geo_code, live=live_api, force=refresh, resolution=resolution)
    if (refresh or live_api) and (frames_live is None or frames_live.empty):
        st.caption("ℹ️ Regional map: live data not available, showing demo frames.")
    frames_show = frames_live if (isinstance(frames_live, pd.DataFrame) and not frames_live.empty) else demo_frames(keywords[0], st.session_state.get("ts_months", 7))
    if isinstance(frames_show, pd.DataFrame) and city_filter:
        mask = frames_show["region"].astype(str).str.contains(city_filter, case=False, na=False)
        frames_show = frames_show[mask]
    st.plotly_chart(
        animated_choropleth(frames_show),
        use_container_width=True,
        key=chart_key("ts_map", keywords[0], st.session_state.get("ts_months", 7), geo_code, resolution, refresh),
    )
    render_ts_map_extras(st=st, ctx=dict(
        frames_df=frames_show, timeframe=timeframe, geo_code=geo_code
    ))
    st.markdown('</div>', unsafe_allow_html=True)

    # Related → Word cloud
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown(f'<div class="section-h"><h2>🔤 Related Queries — Word Cloud</h2><div class="chip">Scope: {scope_label}</div></div>', unsafe_allow_html=True)
    rq = fetch_related(keywords[0], live=live_api, force=refresh)
    if (refresh or live_api) and (not rq):
        st.caption("ℹ️ Related queries: live data not available, showing demo.")
    if not rq: rq = demo_related()
    img = wordcloud_from_related(rq.get("top"), rq.get("rising"))
    src = "Live" if df_live is not None and not df_live.empty else "Demo"
    st.image(img, caption=f"Related queries — {keywords[0]} ({src})", use_column_width=True)
    render_ts_related_extras(st=st, ctx=dict(
        rq_dict=rq, timeframe=timeframe, geo_code=geo_code
    ))
    st.markdown('</div>', unsafe_allow_html=True)


def render_job_market():
    st.markdown(
        """
        <div class="hero">
          <h1>Trends Studio – Job Market</h1>
          <div class="subtle">Overview • Live trends • Regions • Top & Rising</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.subheader("Job Market – Controls")
        job_roles = [
            "Data Analyst", "Data Scientist", "Software Developer",
            "Full Stack Developer", "Data Engineer",
            "Business Analyst", "Machine Learning Engineer",
        ]
        roles = st.multiselect("Job Roles (max 5)", job_roles, default=job_roles[:2], key="jm_roles")[:5] or job_roles[:2]
        timeframe = st.selectbox("Timeframe", ["now 7-d","today 3-m","today 12-m","today 5-y"], index=2, key="jm_timeframe")

        geo_text = st.text_input("Region (country, ISO-2, city like 'Perth', or 'Worldwide')", "Australia", key="jm_geo_text")
        qp1, qp2, qp3 = st.columns(3)
        with qp1:
            if st.button("Australia", key="jm_qp_au"): st.session_state.jm_geo_text = "Australia"
        with qp2:
            if st.button("Perth", key="jm_qp_perth"): st.session_state.jm_geo_text = "Perth"
        with qp3:
            if st.button("Worldwide", key="jm_qp_world"): st.session_state.jm_geo_text = "Worldwide"

        months = st.slider("Animated Map – months", 3, 12, 6, key="jm_months")
        st.markdown("---")
        live_api = st.checkbox("Live API calls", value=True, key="jm_live_api")
        refresh  = st.button("🔄 Refresh All", key="jm_refresh", use_container_width=True)

    geo_code, scope_label, city_filter = resolve_geo(st.session_state.get("jm_geo_text", geo_text))

    tabs = st.tabs(["Overview","Trends by Date","Regional Map","Top & Rising","Job Openings"])

    # Overview
    with tabs[0]:
        st.markdown(f'<div class="section"><div class="section-h"><h2>Interest Over Time (annotated)</h2><div class="chip">Scope: {scope_label}</div></div>', unsafe_allow_html=True)
        df_live = fetch_iot(tuple(roles), timeframe, geo_code, live=live_api, force=refresh)
        if (refresh or live_api) and (df_live is None or df_live.empty):
            st.info("⚠️ Live data unavailable (rate limit or no data). Showing demo series.", icon="⚠️")
        df = df_live if (isinstance(df_live, pd.DataFrame) and not df_live.empty) else demo_ts(tuple(roles))
        cols = [c for c in df.columns if c != "date"]
        first = cols[0]
        now_val = int(df[first].iloc[-1]); avg7 = int(df[first].rolling(7, min_periods=1).mean().iloc[-1])
        k1,k2,k3 = st.columns(3)
        with k1: st.markdown(kpi_card("Now", f"{now_val}"), unsafe_allow_html=True)
        with k2: st.markdown(kpi_card("7-day Avg", f"{avg7}"), unsafe_allow_html=True)
        with k3: st.markdown(f"<span class='chip'>Timeframe: {timeframe}</span>", unsafe_allow_html=True)
        st.plotly_chart(
            line_with_spikes(df, cols),
            use_container_width=True,
            key=chart_key("jm_overview", timeframe, geo_code, tuple(cols), refresh),
        )
        st.caption("Tip: switch to *Trends Studio* to deep-dive these roles as keywords.")
        render_jm_overview_extras(st=st, ctx=dict(
            ts_df=df, timeframe=timeframe, geo_code=geo_code
        ))
        st.markdown("</div>", unsafe_allow_html=True)

    # Trends by Date
    with tabs[1]:
        st.markdown(f'<div class="section"><div class="section-h"><h2>Trends by Date</h2><div class="chip">Scope: {scope_label}</div></div>', unsafe_allow_html=True)
        df_live = fetch_iot(tuple(roles), timeframe, geo_code, live=live_api, force=refresh)
        if (refresh or live_api) and (df_live is None or df_live.empty):
            st.caption("ℹ️ Live series not available; using demo.")
        df = df_live if (isinstance(df_live, pd.DataFrame) and not df_live.empty) else demo_ts(tuple(roles))
        all_series = [c for c in df.columns if c != "date"]
        pick = st.multiselect("Series to show", all_series, default=all_series[:min(3,len(all_series))], key="jm_pick")
        if pick:
            st.plotly_chart(
                line_with_spikes(df[["date"] + pick], pick),
                use_container_width=True,
                key=chart_key("jm_trends_by_date", timeframe, geo_code, tuple(pick), refresh),
            )
        else:
            st.warning("Select at least one series.")
        st.markdown("</div>", unsafe_allow_html=True)

    # Regional Map
    with tabs[2]:
        st.markdown(f'<div class="section"><div class="section-h"><h2>Interest by Region (animated)</h2><div class="chip">Scope: {scope_label}</div></div>', unsafe_allow_html=True)
        resolution = "CITY" if city_filter else "COUNTRY"
        frames_live = fetch_frames(roles[0], months, geo_code, live=live_api, force=refresh, resolution=resolution)
        if (refresh or live_api) and (frames_live is None or frames_live.empty):
            st.caption("ℹ️ Regional map: live data not available, showing demo frames.")
        frames = frames_live if (isinstance(frames_live, pd.DataFrame) and not frames_live.empty) else demo_frames(roles[0], months)
        if isinstance(frames, pd.DataFrame) and city_filter:
            mask = frames["region"].astype(str).str.contains(city_filter, case=False, na=False)
            frames = frames[mask]
        st.plotly_chart(
            animated_choropleth(frames),
            use_container_width=True,
            key=chart_key("jm_map", roles[0], months, geo_code, resolution, refresh),
        )
        st.caption(("Showing regions for: **" + roles[0] + "**") + (f" • City: **{city_filter}**" if city_filter else ""))
        render_jm_map_extras(st=st, ctx=dict(
            frames_df=frames, timeframe=timeframe, geo_code=geo_code
        ))

    # Top & Rising
    with tabs[3]:
        st.markdown(f'<div class="section"><div class="section-h"><h2>Top & Rising Related Keywords</h2><div class="chip">Scope: {scope_label}</div></div>', unsafe_allow_html=True)
        rq = fetch_related(roles[0], live=live_api, force=refresh)
        if (refresh or live_api) and (not rq):
            st.caption("ℹ️ Related queries: live data not available, showing demo.")
        if not rq: rq = demo_related()
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
        render_jm_openings_extras(st=st, ctx=dict(
            timeframe=timeframe, geo_code=geo_code, city_filter=city_filter
        ))


# ─────────────────────────── One sidebar at a time ────────────────────────────
with st.sidebar:
    view = st.selectbox("Choose view", ["Trends Studio", "Job Market"], key="view_pick")

if view == "Trends Studio":
    render_trends_studio()
else:
    render_job_market()

st.caption("© 2025 · Trends Hub · Built with Streamlit + pytrends")
