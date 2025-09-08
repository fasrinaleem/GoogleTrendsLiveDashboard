# app.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pytrends.exceptions import TooManyRequestsError

from trends import get_client, interest_over_time, monthly_region_frames, related_queries
from viz import line_with_spikes, animated_choropleth, wordcloud_from_related

st.set_page_config(page_title="Google Trends – Interactive Dashboard", layout="wide")
st.title("📈 Google Trends – Interactive Dashboard")
st.caption("Fresh overview (instant) • Live sections on-demand • Annotated lines • Animated map • Word cloud")

# ------------------ Demo sample (instant) ------------------
def demo_sample(keywords=("AI", "ChatGPT"), days=90):
    end = datetime.utcnow().date()
    start = end - timedelta(days=days)
    rng = pd.date_range(start, end, freq="D")
    df = pd.DataFrame({"date": rng})
    for i, kw in enumerate(keywords):
        base = 40 + i * 5
        s = (base
             + 20 * np.sin(np.linspace(0, 6, len(rng)))
             + np.random.RandomState(42+i).randn(len(rng)) * 4)
        # clip to 0..100 and add spikes
        s[int(len(rng)//3)] += 35
        s[int(len(rng)//2)] += 25
        df[kw] = np.clip(s, 0, 100).round(0).astype(int)
    return df

def demo_map_frames(keyword="AI"):
    # two simple frames with a few countries
    return pd.DataFrame(
        [
            # frame 1
            {"region": "Australia", "value": 70, "iso2": "AU", "date_frame": "Frame 1"},
            {"region": "United States", "value": 55, "iso2": "US", "date_frame": "Frame 1"},
            {"region": "India", "value": 60, "iso2": "IN", "date_frame": "Frame 1"},
            {"region": "United Kingdom", "value": 50, "iso2": "GB", "date_frame": "Frame 1"},
            # frame 2
            {"region": "Australia", "value": 50, "iso2": "AU", "date_frame": "Frame 2"},
            {"region": "United States", "value": 65, "iso2": "US", "date_frame": "Frame 2"},
            {"region": "India", "value": 75, "iso2": "IN", "date_frame": "Frame 2"},
            {"region": "United Kingdom", "value": 40, "iso2": "GB", "date_frame": "Frame 2"},
        ]
    )

def demo_related():
    top = pd.DataFrame({"query": ["what is ai", "chatgpt login", "ai tools"], "value": [80, 65, 50]})
    rising = pd.DataFrame({"query": ["ai agents", "gpt-4o", "prompt engineering"], "value": [120, 100, 95]})
    return {"top": top, "rising": rising}

# ------------------ Cache live calls ------------------
@st.cache_data(ttl=900, show_spinner=False)
def cached_interest_over_time(keywords, timeframe, geo):
    pytrends = get_client()
    return interest_over_time(pytrends, keywords, timeframe=timeframe, geo=geo)

@st.cache_data(ttl=900, show_spinner=False)
def cached_monthly_frames(keyword, months):
    pytrends = get_client()
    return monthly_region_frames(pytrends, keyword=keyword, months=months, geo="")

@st.cache_data(ttl=900, show_spinner=False)
def cached_related_queries(keyword, timeframe, geo):
    pytrends = get_client()
    pytrends.build_payload([keyword], timeframe=timeframe, geo=geo)
    return related_queries(pytrends, keyword)

# ------------------ Controls ------------------
with st.sidebar:
    st.header("Controls")
    keywords_text = st.text_input("Keywords (comma-separated, up to 5)", value="AI, ChatGPT")
    timeframe = st.selectbox("Timeframe", ["today 12-m", "today 3-m", "now 7-d", "today 5-y"])
    geo = st.text_input("Region (ISO-2 or blank for worldwide)", value="")
    months = st.slider("Animated Map: Months of history (live)", 3, 12, 6, 1)

    st.markdown("---")
    preview_only = st.checkbox("Preview mode (no live calls)", value=True)
    st.caption("Preview mode shows an instant overview without hitting Google. Uncheck to fetch live data, section by section.")

def parse_keywords(s: str):
    return [x.strip() for x in s.split(",") if x.strip()][:5]

keywords = parse_keywords(keywords_text)
if not keywords:
    st.warning("Please enter at least one keyword.")
    st.stop()

# ------------------ Section 1: Time series ------------------
st.subheader("1) Interest Over Time (Annotated)")
colA, colB = st.columns([1, 1])

with colA:
    st.write("**Fresh Overview (instant)**")
    df_demo = demo_sample(tuple(keywords[:2]))  # small instant sample
    st.plotly_chart(line_with_spikes(df_demo, df_demo.columns[1:].tolist()), use_container_width=True)

with colB:
    st.write("**Live (on-demand)**")
    if preview_only:
        st.info("Live fetch is disabled (Preview mode). Uncheck the box in sidebar to fetch live data.")
    else:
        if st.button("Fetch live time series"):
            try:
                df_ts = cached_interest_over_time(keywords, timeframe, geo)
                if df_ts.empty:
                    st.info("No live data returned. Try different timeframe/region/keywords.")
                else:
                    st.plotly_chart(line_with_spikes(df_ts, keywords), use_container_width=True)
                    st.download_button("Download live CSV", df_ts.to_csv(index=False), "interest_over_time.csv")
            except TooManyRequestsError:
                st.error("Google is rate-limiting. Please wait 1–2 minutes and try again.")

# ------------------ Section 2: Animated map ------------------
st.subheader("2) Animated Map – Interest by Country")
with st.expander("Show map"):
    col1, col2 = st.columns([1,1])

    with col1:
        st.write("**Fresh Overview (instant)**")
        frames_demo = demo_map_frames(keyword=keywords[0])
        st.plotly_chart(animated_choropleth(frames_demo, title=f"Animated Interest – {keywords[0]} (demo)"),
                        use_container_width=True)

    with col2:
        st.write("**Live (on-demand)**")
        if preview_only:
            st.info("Live fetch is disabled (Preview mode). Uncheck to enable.")
        else:
            if st.button("Fetch live map"):
                try:
                    frames_df = cached_monthly_frames(keywords[0], months)
                    if frames_df.empty:
                        st.info("No regional data for this keyword/months.")
                    else:
                        st.plotly_chart(animated_choropleth(frames_df, title=f"Animated Interest – {keywords[0]}"),
                                        use_container_width=True)
                        st.download_button("Download map frames CSV", frames_df.to_csv(index=False), "map_frames.csv")
                except TooManyRequestsError:
                    st.warning("Map skipped due to rate limits. Try again later.")

# ------------------ Section 3: Word cloud ------------------
st.subheader("3) Word Cloud – Related Queries")
with st.expander("Show word cloud"):
    col3, col4 = st.columns([1,1])

    with col3:
        st.write("**Fresh Overview (instant)**")
        demo_rq = demo_related()
        img_buf = wordcloud_from_related(demo_rq["top"], demo_rq["rising"])
        st.image(img_buf, caption=f"Related Queries (demo) – {keywords[0]}", use_column_width=True)

    with col4:
        st.write("**Live (on-demand)**")
        if preview_only:
            st.info("Live fetch is disabled (Preview mode). Uncheck to enable.")
        else:
            if st.button("Fetch live related queries"):
                try:
                    rq = cached_related_queries(keywords[0], timeframe, geo)
                    img_live = wordcloud_from_related(rq.get("top"), rq.get("rising"))
                    st.image(img_live, caption=f"Related Queries (live) – {keywords[0]}", use_column_width=True)
                except TooManyRequestsError:
                    st.info("Related queries were rate-limited. Try again later.")
