# app.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pytrends.exceptions import TooManyRequestsError

from trends import get_client, interest_over_time, monthly_region_frames, related_queries
from viz import line_with_spikes, animated_choropleth, wordcloud_from_related, kpi_card, sparkline
from trends import (
    get_client, interest_over_time, monthly_region_frames, related_queries,
    trending_today, trending_realtime
)

# Create session_state alias
ss = st.session_state
# Initialize keys if not already set
if "trend_nonce" not in ss: ss.trend_nonce = 0
if "trend_time" not in ss: ss.trend_time = None
if "trend_daily" not in ss: ss.trend_daily = None
if "trend_rt" not in ss: ss.trend_rt = None


# ---------- Page + styles ----------
st.set_page_config(page_title="Trends Studio", layout="wide")

st.markdown("""
<style>
/* Light gradient hero */
.hero {
  background: radial-gradient(1200px 400px at 20% -10%, rgba(109,40,217,0.10), transparent),
              linear-gradient(90deg, rgba(109,40,217,0.10), rgba(37,99,235,0.08));
  border-radius: 18px; padding: 18px 22px; margin-bottom: 14px;
  border: 1px solid #e9eaf0;
}
.hero h1 { margin: 0; font-size: 1.9rem; line-height: 1.25; }
.subtle { color: #475569; font-size: 0.95rem; margin-top: 4px; }

/* Cards */
.card {
  background: #ffffff;
  border: 1px solid #e9eaf0;
  border-radius: 16px;
  padding: 16px;
  box-shadow: 0 6px 20px rgba(17, 24, 39, 0.06);
}
.card h3 { margin-top: 0; }

/* KPI pills */
.kpi {
  display:flex; flex-direction:column; gap:6px; padding:12px 14px; border-radius:12px;
  background: linear-gradient(180deg, #ffffff, #fafbff);
  border:1px solid #e9eaf0;
  box-shadow: 0 4px 16px rgba(17,24,39,0.05);
}
.kpi-label { color:#64748b; font-size:0.75rem; letter-spacing:.03em; text-transform:uppercase;}
.kpi-value { font-size:1.25rem; font-weight:800; color:#111827;}
.kpi-delta { margin-left:8px; font-size:.85rem; color:#16a34a; }

/* Chips */
.chip { display:inline-block; padding:4px 10px; border-radius:999px; font-size:0.75rem;
  background: #eef2ff; border:1px solid #c7d2fe; color:#3730a3; }
.chip.warn { background:#fef3c7; border-color:#fde68a; color:#92400e; }

/* Section titles */
.section-title { font-size:1.05rem; color:#0f172a; margin-bottom:8px; font-weight:700; }

.card ul, .card li { margin: 0; padding-left: 0; }
.card li { list-style: none; margin-bottom: 6px; }
.card li::before { content: "• "; color: #6D28D9; font-weight: 700; margin-right: 4px; }

</style>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="hero"><h1>✨ Trends Studio</h1>'
    '<div class="subtle">Instant overview • Live on demand • Annotated lines • Animated map • Word cloud</div></div>',
    unsafe_allow_html=True
)
# ---- Trending caches ----
@st.cache_data(ttl=600, show_spinner=False)
def cached_trending_today(geo_name="australia", _nonce: int = 0):
    pytrends = get_client()
    return trending_today(pytrends, geo=geo_name)

@st.cache_data(ttl=600, show_spinner=False)
def cached_trending_realtime(country_code="AU", cat="all", _nonce: int = 0):
    pytrends = get_client()
    return trending_realtime(pytrends, geo=country_code, cat=cat)

# ---------- Caches for live ----------
@st.cache_data(ttl=900, show_spinner=False)
def live_ts(keywords, timeframe, geo):
    pytrends = get_client()
    return interest_over_time(pytrends, keywords, timeframe=timeframe, geo=geo)

@st.cache_data(ttl=900, show_spinner=False)
def live_frames(keyword, months):
    pytrends = get_client()
    return monthly_region_frames(pytrends, keyword=keyword, months=months, geo="")

@st.cache_data(ttl=900, show_spinner=False)
def live_related(keyword, timeframe, geo):
    pytrends = get_client()
    pytrends.build_payload([keyword], timeframe=timeframe, geo=geo)
    return related_queries(pytrends, keyword)
    
# ========= 🔥 Trending Now (button + visible refresh) =========
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("### 🔥 Top Trending Searches Today")

btn_col, chip_col = st.columns([1, 5])
with btn_col:
    if st.button("🔄 Fetch live trending", key="btn_trending"):
        ss.trend_nonce += 1  # bump nonce to force cache miss
        # pull fresh
        ss.trend_daily = cached_trending_today("australia", ss.trend_nonce)
        ss.trend_rt    = cached_trending_realtime("AU", "all", ss.trend_nonce)
        from datetime import datetime
        ss.trend_time = datetime.now().strftime("%H:%M")
        st.rerun()

with chip_col:
    if ss.trend_time:
        st.markdown(f"<span class='chip'>Live @ {ss.trend_time}</span>", unsafe_allow_html=True)
    else:
        st.markdown("<span class='chip warn'>Demo until you fetch live</span>", unsafe_allow_html=True)

colA, colB = st.columns(2)

with colA:
    st.caption("Daily Trending — Australia")
    daily = ss.trend_daily
    if daily is None or daily.empty or "query" not in daily.columns:
        # fallback list
        items = ["AFL finals", "Fuel prices", "Weather radar", "Bitcoin price"]
    else:
        items = daily["query"].astype(str).tolist()[:10]
    for q in items:
        st.markdown(f"- {q}")

with colB:
    st.caption("Realtime Trending — Perth (AU)*")
    rt = ss.trend_rt
    titles = []
    if rt is not None and not rt.empty:
        # Perth filter (title, entityNames, articles)
        def contains(col):
            return rt[col].astype(str).str.contains("Perth", case=False, na=False) if col in rt.columns else False
        mask = contains("title")
        if "entityNames" in rt.columns: mask = mask | contains("entityNames")
        if "articles" in rt.columns:    mask = mask | contains("articles")
        filt = rt[mask] if mask.any() else rt
        title_col = "title" if "title" in filt.columns else filt.columns[0]
        titles = filt[title_col].astype(str).tolist()[:8]

    if not titles:
        titles = ["Perth weather update", "Perth traffic", "Perth events this weekend", "Optus Stadium news"]

    for t in titles:
        st.markdown(f"- **{t}**")
    st.caption("*Realtime is provided at country-level by Google; filtered to items mentioning ‘Perth’.")
st.markdown("</div>", unsafe_allow_html=True)
# ========= end Trending Now =========

# ---------- Sidebar controls ----------
with st.sidebar:
    st.subheader("Controls")
    kw_text = st.text_input("Keywords", "AI, Data Analytics")
    timeframe = st.selectbox("Timeframe", ["today 12-m","today 3-m","now 7-d","today 5-y"])
    geo = st.text_input("Region (ISO-2, blank = worldwide)", "")
    months = st.slider("Animated Map months", 3, 12, 6)
    st.markdown("---")
    preview_only = st.checkbox("Preview mode (no live calls)", value=True,
                               help="Keeps everything snappy; fetch live data only when you click a button.")

def parse_keywords(s: str): 
    return [x.strip() for x in s.split(",") if x.strip()][:5]

keywords = parse_keywords(kw_text) or ["AI"]

# ---------- Demo data for instant render ----------
def demo_ts(keywords=("AI","Data Analytics"), days=120):
    end = datetime.utcnow().date(); start = end - timedelta(days=days)
    rng = pd.date_range(start, end, freq="D")
    df = pd.DataFrame({"date": rng})
    for i, kw in enumerate(keywords):
        base = 45 + i*4
        s = base + 18*np.sin(np.linspace(0, 6, len(rng))) + np.random.RandomState(33+i).randn(len(rng))*3
        s[int(len(rng)*0.30)] += 28; s[int(len(rng)*0.55)] += 18
        df[kw] = np.clip(s, 0, 100).round().astype(int)
    return df

def demo_frames(keyword="AI"):
    return pd.DataFrame([
        {"region":"Australia","value":70,"iso2":"AU","date_frame":"T1"},
        {"region":"United States","value":58,"iso2":"US","date_frame":"T1"},
        {"region":"India","value":62,"iso2":"IN","date_frame":"T1"},
        {"region":"United Kingdom","value":50,"iso2":"GB","date_frame":"T1"},
        {"region":"Australia","value":48,"iso2":"AU","date_frame":"T2"},
        {"region":"United States","value":69,"iso2":"US","date_frame":"T2"},
        {"region":"India","value":75,"iso2":"IN","date_frame":"T2"},
        {"region":"United Kingdom","value":41,"iso2":"GB","date_frame":"T2"},
    ])

def demo_related():
    top = pd.DataFrame({"query":["what is ai","Data Analytics login","ai tools"],"value":[80,65,50]})
    rising = pd.DataFrame({"query":["ai agents","gpt-4o","prompt ideas"],"value":[120,100,95]})
    return {"top": top, "rising": rising}

# Keep last good live results in session
ss = st.session_state
ss.setdefault("ts_last", None)
ss.setdefault("frames_last", None)
ss.setdefault("rq_last", None)

# ---------- Initialize session keys for trending today ----------
ss = st.session_state
ss.setdefault("trend_nonce", 0)
ss.setdefault("trend_time", None)
ss.setdefault("trend_daily", None)
ss.setdefault("trend_rt", None)

# ---------- Overview row (KPIs + sparklines) ----------
ov = st.container()
with ov:
    colK, colS = st.columns([1, 3])
    with colK:
        demo = demo_ts(tuple(keywords[:2]))
        last_vals = [int(demo[k].iloc[-1]) for k in demo.columns if k!="date"]
        avg_vals  = [int(demo[k].rolling(7, min_periods=1).mean().iloc[-1]) for k in demo.columns if k!="date"]
        st.markdown(kpi_card("Now (demo)", f"{last_vals[0]}"), unsafe_allow_html=True)
        st.markdown(kpi_card("7-day avg", f"{avg_vals[0]}"), unsafe_allow_html=True)
        if ss["ts_last"] is not None:
            st.markdown("<span class='chip'>Live data loaded</span>", unsafe_allow_html=True)
        else:
            st.markdown("<span class='chip warn'>Using demo until you fetch live</span>", unsafe_allow_html=True)
    with colS:
        st.markdown("<div class='section-title'>Interest Over Time (Annotated Spikes)</div>", unsafe_allow_html=True)
        st.plotly_chart(
            line_with_spikes(demo, [c for c in demo.columns if c!='date']),
            use_container_width=True, key="overview_demo"
        )

# ---------- Tabs ----------
tab1, tab2, tab3 = st.tabs(["📈 Trends", "🗺️ Map", "🔤 Related"])

# -- Tab 1: Trends ---------------------------------------------------------
with tab1:
    c1, c2 = st.columns([3, 1])
    with c1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.write("**Live (on-demand)**")
        if preview_only:
            st.info("Preview mode is on. Uncheck in the sidebar to enable live fetch.")
        else:
            if st.button("Fetch live time series", use_container_width=True, key="btn_ts"):
                with st.spinner("Pulling latest from Google Trends…"):
                    try:
                        df = live_ts(keywords, timeframe, geo)
                        if not df.empty:
                            ss["ts_last"] = df
                            st.toast("Live time series updated", icon="✅")
                        else:
                            st.toast("No live data for those settings. Showing demo.", icon="ℹ️")
                    except TooManyRequestsError:
                        if ss["ts_last"] is not None:
                            st.toast("Rate limited – showing last live data", icon="⚠️")
                        else:
                            st.toast("Rate limited – showing demo", icon="⚠️")
                st.rerun()

        df_show = ss["ts_last"] if ss["ts_last"] is not None else demo
        lbl = "Live" if ss["ts_last"] is not None else "Demo"
        st.caption(f"Showing: {lbl}")
        st.plotly_chart(
            line_with_spikes(df_show, keywords[: len([c for c in df_show.columns if c!='date'])]),
            use_container_width=True, key="trends_live"
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.write("**Sparklines**")
        base_df = ss["ts_last"] if ss["ts_last"] is not None else demo
        for k in [c for c in base_df.columns if c != "date"]:
            st.caption(k)
            st.plotly_chart(sparkline(base_df, k), use_container_width=True, theme=None, key=f"spark_{k}")
        st.markdown("</div>", unsafe_allow_html=True)

# -- Tab 2: Map ------------------------------------------------------------
with tab2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    colL, colR = st.columns([3,1])
    with colR:
        st.write("**Live (on-demand)**")
        if preview_only:
            st.info("Preview mode is on. Uncheck in the sidebar to enable live fetch.")
        else:
            if st.button("Fetch live map", use_container_width=True, key="btn_map"):
                with st.spinner("Building monthly frames…"):
                    try:
                        frames = live_frames(keywords[0], months)
                        if not frames.empty:
                            ss["frames_last"] = frames
                            st.toast("Live map updated", icon="✅")
                        else:
                            st.toast("No regional data. Showing demo.", icon="ℹ️")
                    except TooManyRequestsError:
                        if ss["frames_last"] is not None:
                            st.toast("Rate limited – showing last live map", icon="⚠️")
                        else:
                            st.toast("Rate limited – showing demo map", icon="⚠️")
                st.rerun()
    with colL:
        frames_show = ss["frames_last"] if ss["frames_last"] is not None else demo_frames(keywords[0])
        st.caption("Animated regional interest")
        st.plotly_chart(animated_choropleth(frames_show), use_container_width=True, key="map")
    st.markdown("</div>", unsafe_allow_html=True)

# -- Tab 3: Related --------------------------------------------------------
with tab3:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    colL, colR = st.columns([3,1])
    with colR:
        st.write("**Live (on-demand)**")
        if preview_only:
            st.info("Preview mode is on. Uncheck in the sidebar to enable live fetch.")
        else:
            if st.button("Fetch live related", use_container_width=True, key="btn_related"):
                with st.spinner("Fetching related queries…"):
                    try:
                        rq = live_related(keywords[0], timeframe, geo)
                        if rq:
                            ss["rq_last"] = rq
                            st.toast("Live related queries updated", icon="✅")
                        else:
                            st.toast("No related queries – showing demo.", icon="ℹ️")
                    except TooManyRequestsError:
                        if ss["rq_last"] is not None:
                            st.toast("Rate limited – showing last live related queries", icon="⚠️")
                        else:
                            st.toast("Rate limited – showing demo word cloud", icon="⚠️")
                st.rerun()
    with colL:
        use_rq = ss["rq_last"] if ss["rq_last"] is not None else demo_related()
        img = wordcloud_from_related(use_rq.get("top"), use_rq.get("rising"))
        st.image(img, caption=f"Related queries – {keywords[0]}", use_column_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
