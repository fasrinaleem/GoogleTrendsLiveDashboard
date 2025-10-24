# views/trends_studio.py
import streamlit as st
from services.trends_client import fetch_trending_daily, fetch_trending_realtime, fetch_iot, fetch_region_frames, fetch_related, sanitize_related_df
from utils.geo import resolve_geo
from components.charts import line_with_spikes, animated_choropleth, wordcloud_from_related

def render():
    st.markdown("<div class='hero'><h1>✨ Trends Studio</h1></div>", unsafe_allow_html=True)

    kw_text = st.text_input("Keywords (comma-separated, max 5)", "AI, Data")
    keywords = [x.strip() for x in kw_text.split(",") if x.strip()][:5] or ["AI"]

    region_pick = st.selectbox("Region", ["Australia","Perth","Worldwide","Custom"], index=0)
    geo_text = st.text_input("Custom (country/ISO-2 or Perth)", "Australia") if region_pick=="Custom" else region_pick
    geo_code, scope_label, city_filter = resolve_geo(geo_text)

    st.subheader("🔥 Trending")
    c1,c2 = st.columns(2)
    with c1:
        daily = fetch_trending_daily("australia")
        if not daily.empty:
            st.write(daily.head(10))
        else: st.info("No daily trending data.")
    with c2:
        realtime = fetch_trending_realtime("AU","all")
        if not realtime.empty:
            st.write(realtime.head(10))
        else: st.info("No realtime trending data.")

    st.subheader("📈 Interest Over Time")
    tf = st.selectbox("Timeframe", ["today 12-m","today 3-m","now 7-d","today 5-y"], index=0)
    iot = fetch_iot(tuple(keywords), tf, geo_code)
    if not iot.empty:
        st.plotly_chart(line_with_spikes(iot, [c for c in iot.columns if c!='date']), use_container_width=True)
    else:
        st.info("No IoT data for this selection.")

    st.subheader("🗺️ Regional Interest")
    series = st.selectbox("Series for map", keywords)
    frames = fetch_region_frames(series, 6, geo_code, resolution=("CITY" if city_filter else "COUNTRY"))
    if not frames.empty:
        st.plotly_chart(animated_choropleth(frames), use_container_width=True)
    else: st.info("No region data.")

    st.subheader("🔤 Related Queries")
    kw_wc = st.selectbox("Word cloud keyword", options=keywords)
    rq = fetch_related(kw_wc, geo_code)
    top = sanitize_related_df(rq.get("top"))
    rising = sanitize_related_df(rq.get("rising"))
    st.image(wordcloud_from_related(top, rising), caption=f"Related — {kw_wc}", use_container_width=True)
