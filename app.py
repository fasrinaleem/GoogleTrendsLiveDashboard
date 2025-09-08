import streamlit as st
from pytrends.exceptions import TooManyRequestsError
from trends import get_client, interest_over_time, monthly_region_frames, related_queries
from viz import line_with_spikes, animated_choropleth, wordcloud_from_related

st.set_page_config(page_title="Google Trends Dashboard", layout="wide")
st.title("📈 Google Trends – Interactive Dashboard")
st.caption("Annotated trend lines • Animated map • Word cloud of related queries")

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

with st.sidebar:
    st.header("Controls")
    keywords_text = st.text_input("Keywords (comma-separated, up to 5)", value="AI, Bitcoin")
    timeframe = st.selectbox("Timeframe", ["today 12-m", "today 3-m", "now 7-d", "today 5-y"])
    geo = st.text_input("Region (ISO-2 code or leave blank for Worldwide)", value="")
    months = st.slider("Animated Map: Months of history", min_value=3, max_value=12, value=6, step=1)
    run = st.button("Run")

def parse_keywords(s: str):
    return [x.strip() for x in s.split(",") if x.strip()][:5]

if run:
    keywords = parse_keywords(keywords_text)
    if not keywords:
        st.warning("Please enter at least one keyword.")
        st.stop()

    st.subheader("1) Interest Over Time (Annotated)")
    try:
        df_ts = cached_interest_over_time(keywords, timeframe, geo)
        if df_ts.empty:
            st.info("No data returned. Try a different timeframe/region/keywords.")
        else:
            st.plotly_chart(line_with_spikes(df_ts, keywords), use_container_width=True)
            st.download_button("Download time series CSV", df_ts.to_csv(index=False), "interest_over_time.csv")
    except TooManyRequestsError:
        st.error("Google is rate-limiting. Please wait 1–2 minutes and try again.")
        st.stop()

    st.subheader("2) Animated Map – Interest by Country")
    try:
        frames_df = cached_monthly_frames(keywords[0], months)
        if frames_df.empty:
            st.info("No regional data returned. Try different keyword or months.")
        else:
            st.plotly_chart(animated_choropleth(frames_df, title=f"Animated Interest – {keywords[0]}"), use_container_width=True)
    except TooManyRequestsError:
        st.warning("Map skipped due to rate limits. Try later.")

    st.subheader("3) Word Cloud – Related Queries")
    try:
        rq = cached_related_queries(keywords[0], timeframe, geo)
        img_buf = wordcloud_from_related(rq.get("top"), rq.get("rising"))
        st.image(img_buf, caption=f"Related Queries – {keywords[0]}", use_column_width=True)
    except TooManyRequestsError:
        st.info("Related queries were rate-limited. Re-run later.")
else:
    st.info("Enter your keywords in the sidebar and click **Run**.")
