# app.py
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st
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

# ────────────────────────── Page + base styles ──────────────────────────
st.set_page_config(page_title="Trends Studio", layout="wide")

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
.section-h{ display:flex; align-items:center; justify-content:space-between; margin-bottom:10px;}
.section-h h2{margin:0; font-size:1.25rem; color:var(--ink);}
.kpi{display:flex; flex-direction:column; gap:6px; padding:12px 14px; border-radius:12px;
     background:linear-gradient(180deg,#fff,#f9fbff); border:1px solid var(--card-brd);
     box-shadow:0 6px 16px rgba(17,24,39,.06);}
.kpi-label{color:var(--muted); font-size:.72rem; letter-spacing:.03em; text-transform:uppercase;}
.kpi-value{font-size:1.32rem; font-weight:800; color:var(--ink-2);}
.chip{display:inline-block; padding:6px 10px; border-radius:999px; font-size:.75rem;
      background:var(--chip); border:1px solid var(--chip-brd); color:var(--chip-ink);}
.chip.warn{background:var(--warn); border-color:var(--warn-brd); color:var(--warn-ink);}
.caption{color:var(--muted); font-size:.85rem;}
.small-gap{margin-top:8px}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="hero"><h1>✨ Trends Studio</h1>'
    '<div class="subtle">Instant overview • Live on demand • Annotated lines • Animated map • Word cloud</div></div>',
    unsafe_allow_html=True,
)

# ────────────────────────── Session keys ──────────────────────────
ss = st.session_state
ss.setdefault("ts_last", None)
ss.setdefault("frames_last", None)
ss.setdefault("rq_last", None)
ss.setdefault("trend_nonce", 0)
ss.setdefault("trend_time", None)
ss.setdefault("trend_daily", None)
ss.setdefault("trend_rt", None)
ss.setdefault("controls_sig", None)
ss.setdefault("last_auto_ts", 0.0)
ss.setdefault("last_auto_map", 0.0)
ss.setdefault("last_auto_rq", 0.0)

# ────────────────────────── Sidebar controls ──────────────────────────
with st.sidebar:
    st.subheader("Controls")
    kw_text = st.text_input("Keywords", "AI, Data")
    timeframe = st.selectbox("Timeframe", ["today 12-m", "today 3-m", "now 7-d", "today 5-y"])
    geo = st.text_input("Region (ISO-2, blank = worldwide)", "")
    months = st.slider("Animated Map – months", 3, 12, 7, help="Number of monthly frames for the map animation.")
    st.markdown("---")
    live_api = st.checkbox("Live API calls (disable to use Demo)", value=False)
    live_ticker = st.checkbox("Live ticker (auto refresh every 60s)", value=True, help="Only when Live API calls are ON")

def parse_keywords(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()][:5]

keywords = parse_keywords(kw_text) or ["AI"]

# Reset stale live data when controls change
def controls_fingerprint() -> str:
    return f"{','.join(keywords)}|{timeframe}|{geo}|{months}"

sig = controls_fingerprint()
if ss.controls_sig is None:
    ss.controls_sig = sig
elif ss.controls_sig != sig:
    ss.controls_sig = sig
    ss.ts_last = None
    ss.frames_last = None
    ss.rq_last = None

# Optional gentle page auto-reload (JS fallback; Streamlit removed experimental_autorefresh)
if live_api and live_ticker:
    st.markdown(
        "<script>setTimeout(() => { window.location.reload(); }, 60000);</script>",
        unsafe_allow_html=True,
    )

# ────────────────────────── Demo generators (instant UI) ──────────────────────────
def demo_ts(keys=("AI", "Data"), days=180) -> pd.DataFrame:
    end = datetime.utcnow().date()
    start = end - timedelta(days=days)
    rng = pd.date_range(start, end, freq="D")
    df = pd.DataFrame({"date": rng})
    for i, kw in enumerate(keys):
        base = 45 + i * 3
        s = base + 18 * np.sin(np.linspace(0, 6, len(rng))) + np.random.RandomState(33 + i).randn(len(rng)) * 3
        s[int(len(rng) * 0.30)] += 25
        s[int(len(rng) * 0.55)] += 18
        df[kw] = np.clip(s, 0, 100).round().astype(int)
    return df

def demo_frames(keyword="AI", months_count: int = 6) -> pd.DataFrame:
    # produce simple, readable month frames
    end = datetime.utcnow().date().replace(day=1)
    frames = []
    for m in range(months_count, 0, -1):
        label = (end - pd.DateOffset(months=m-1)).strftime("%b %Y")
        frames += [
            {"region": "Australia", "value": 50 + (m % 5) * 4, "iso2": "AU", "date_frame": label},
            {"region": "United States", "value": 45 + (m % 6) * 3, "iso2": "US", "date_frame": label},
            {"region": "India", "value": 48 + (m % 4) * 6, "iso2": "IN", "date_frame": label},
            {"region": "United Kingdom", "value": 40 + (m % 3) * 5, "iso2": "GB", "date_frame": label},
        ]
    return pd.DataFrame(frames)

def demo_related():
    top = pd.DataFrame({"query": ["what is ai", "data analytics login", "ai tools"], "value": [80, 65, 50]})
    rising = pd.DataFrame({"query": ["ai agents", "gpt-4o", "prompt ideas"], "value": [120, 100, 95]})
    return {"top": top, "rising": rising}

# ────────────────────────── Section 0: Trending Today ──────────────────────────
st.markdown('<div class="section">', unsafe_allow_html=True)
st.markdown('<div class="section-h"><h2>🔥 Top Trending Searches Today</h2></div>', unsafe_allow_html=True)
left, right = st.columns([1, 1])

with left:
    if st.button("🔄 Fetch live trending", key="btn_trending"):
        ss.trend_nonce += 1
        try:
            py = get_client()
            ss.trend_daily = trending_today(py, geo="australia")
            ss.trend_rt = trending_realtime(py, geo="AU", cat="all")
            ss.trend_time = datetime.now().strftime("%H:%M")
            st.toast("Trending updated", icon="✅")
        except Exception:
            st.toast("Couldn’t refresh trending. Showing fallback.", icon="⚠️")

    st.caption("Daily Trending — Australia")
    daily = ss.trend_daily
    items = (
        daily["query"].astype(str).tolist()[:10]
        if (daily is not None and not daily.empty and "query" in daily.columns)
        else ["AFL finals", "Fuel prices", "Weather radar", "Bitcoin price"]
    )
    st.markdown("\n".join([f"- {q}" for q in items]))

with right:
    st.caption("Realtime Trending — Perth (AU)*")
    rt = ss.trend_rt
    titles = []
    if rt is not None and not rt.empty:
        def contains(col):
            return rt[col].astype(str).str.contains("Perth", case=False, na=False) if col in rt.columns else False
        mask = contains("title")
        if "entityNames" in rt.columns:
            mask = mask | contains("entityNames")
        if "articles" in rt.columns:
            mask = mask | contains("articles")
        filt = rt[mask] if (isinstance(mask, pd.Series) and mask.any()) else rt
        name_col = "title" if "title" in filt.columns else filt.columns[0]
        titles = filt[name_col].astype(str).tolist()[:8]
    if not titles:
        titles = ["Perth weather update", "Perth traffic", "Perth events this weekend", "Optus Stadium news"]
    st.markdown("\n".join([f"- **{t}**" for t in titles]))
    st.caption("※ Realtime is provided at country level by Google; filtered to items mentioning “Perth”.")
st.markdown("</div>", unsafe_allow_html=True)

# ────────────────────────── Section 1: Overview (KPIs + Line) ──────────────────────────
st.markdown('<div class="section">', unsafe_allow_html=True)
st.markdown('<div class="section-h"><h2>📈 Interest Over Time (Annotated)</h2></div>', unsafe_allow_html=True)

kpi_col, chart_col, spark_col = st.columns([0.9, 2.2, 1.1])

df_overview = ss.ts_last if ss.ts_last is not None else demo_ts(tuple(keywords[:2]))
series_cols = [c for c in df_overview.columns if c != "date"]

with kpi_col:
    now_vals = [int(df_overview[c].iloc[-1]) for c in series_cols]
    avg_vals = [int(df_overview[c].rolling(7, min_periods=1).mean().iloc[-1]) for c in series_cols]
    label_now = "NOW (LIVE)" if ss.ts_last is not None else "NOW (DEMO)"
    st.markdown(kpi_card(label_now, f"{now_vals[0]}"), unsafe_allow_html=True)
    st.markdown(kpi_card("7-DAY AVG", f"{avg_vals[0]}"), unsafe_allow_html=True)
    if ss.ts_last is None:
        st.markdown('<span class="chip warn small-gap">Using demo until you fetch live</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="chip small-gap">Live loaded</span>', unsafe_allow_html=True)

with chart_col:
    st.plotly_chart(line_with_spikes(df_overview, series_cols), use_container_width=True, key="overview_chart")

with spark_col:
    st.write("**Sparklines**")
    base_df = ss.ts_last if ss.ts_last is not None else df_overview
    for k in [c for c in base_df.columns if c != "date"]:
        st.caption(k)
        st.plotly_chart(sparkline(base_df, k), use_container_width=True, theme=None, key=f"spark_{k}")

st.divider()
live_l, live_r = st.columns([1.7, 1])
with live_l:
    st.write("**Live (on-demand)**")
    if not live_api:
        st.info("Live API calls are OFF (using Demo). Turn them on in the sidebar.")
    else:
        if st.button("⚡ Fetch live time series", key="btn_ts", use_container_width=True):
            with st.spinner("Pulling latest from Google Trends…"):
                try:
                    df = interest_over_time(get_client(), keywords, timeframe=timeframe, geo=geo)
                    if not df.empty:
                        ss.ts_last = df
                        st.toast("Live time series updated", icon="✅")
                    else:
                        st.toast("No live data for those settings. Showing demo.", icon="ℹ️")
                except TooManyRequestsError:
                    st.toast("Rate limited – try again later.", icon="⚠️")
            st.rerun()
        # Auto-pull every 60s when ticker is enabled
        if live_ticker and time.time() - ss.last_auto_ts > 60:
            try:
                df = interest_over_time(get_client(), keywords, timeframe=timeframe, geo=geo)
                if not df.empty:
                    ss.ts_last = df
                ss.last_auto_ts = time.time()
            except Exception:
                pass

with live_r:
    st.write("")

st.markdown("</div>", unsafe_allow_html=True)

# ────────────────────────── Section 2: Animated Map ──────────────────────────
st.markdown('<div class="section">', unsafe_allow_html=True)
st.markdown('<div class="section-h"><h2>🗺️ Animated Map — Interest by Region</h2></div>', unsafe_allow_html=True)

map_l, map_r = st.columns([3, 1])
with map_l:
    frames_show = ss.frames_last if ss.frames_last is not None else demo_frames(keywords[0], months)
    st.plotly_chart(animated_choropleth(frames_show), use_container_width=True, key="map_fig")
with map_r:
    st.write("**Live (on-demand)**")
    if not live_api:
        st.info("Live API calls are OFF (using Demo). Turn them on in the sidebar.")
    else:
        if st.button("🧭 Fetch live map", key="btn_map", use_container_width=True):
            with st.spinner("Building monthly frames…"):
                try:
                    frames = monthly_region_frames(get_client(), keyword=keywords[0], months=months, geo="")
                    if not frames.empty:
                        ss.frames_last = frames
                        st.toast("Live map updated", icon="✅")
                    else:
                        st.toast("No regional data. Showing demo.", icon="ℹ️")
                except TooManyRequestsError:
                    st.toast("Rate limited – try again later.", icon="⚠️")
            st.rerun()
        if live_ticker and time.time() - ss.last_auto_map > 300:  # refresh map every 5 min
            try:
                frames = monthly_region_frames(get_client(), keyword=keywords[0], months=months, geo="")
                if not frames.empty:
                    ss.frames_last = frames
                ss.last_auto_map = time.time()
            except Exception:
                pass
st.markdown("</div>", unsafe_allow_html=True)

# ────────────────────────── Section 3: Related → Word Cloud ──────────────────────────
st.markdown('<div class="section">', unsafe_allow_html=True)
st.markdown('<div class="section-h"><h2>🔤 Related Queries — Word Cloud</h2></div>', unsafe_allow_html=True)

rel_l, rel_r = st.columns([3, 1])
with rel_l:
    use_rq = ss.rq_last if ss.rq_last is not None else demo_related()
    img = wordcloud_from_related(use_rq.get("top"), use_rq.get("rising"))
    st.image(img, caption=f"Related queries — {keywords[0]} (first keyword)", use_column_width=True)
with rel_r:
    st.write("**Live (on-demand)**")
    if not live_api:
        st.info("Live API calls are OFF (using Demo). Turn them on in the sidebar.")
    else:
        if st.button("🔎 Fetch live related", key="btn_related", use_container_width=True):
            with st.spinner("Fetching related queries…"):
                try:
                    rq = related_queries(get_client(), keywords[0])
                    if rq:
                        ss.rq_last = rq
                        st.toast("Live related queries updated", icon="✅")
                    else:
                        st.toast("No related queries – showing demo.", icon="ℹ️")
                except TooManyRequestsError:
                    st.toast("Rate limited – try again later.", icon="⚠️")
            st.rerun()
        if live_ticker and time.time() - ss.last_auto_rq > 180:  # refresh related every 3 min
            try:
                rq = related_queries(get_client(), keywords[0])
                if rq:
                    ss.rq_last = rq
                ss.last_auto_rq = time.time()
            except Exception:
                pass
st.markdown("</div>", unsafe_allow_html=True)

# Footer hint (demo/live status)
mode_live = any(x is not None for x in (ss.ts_last, ss.frames_last, ss.rq_last))
st.caption(f"Showing: {'Live' if mode_live else 'Demo'}")
