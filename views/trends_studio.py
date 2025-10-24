# views/trends_studio.py
import streamlit as st
from services.trends_client import (
    fetch_trending_daily, 
    fetch_trending_realtime, 
    fetch_iot, 
    fetch_region_frames, 
    fetch_related, 
    sanitize_related_df,
    get_health
)
from utils.geo import resolve_geo
from components.charts import line_with_spikes, animated_choropleth, wordcloud_from_related
import pandas as pd

def render():
    st.markdown("<div class='hero'><h1>✨ Trends Studio</h1></div>", unsafe_allow_html=True)

    # Debug status
    with st.expander("🔍 Debug Status"):
        st.json(get_health())
        st.caption("⚠️ If everything says 'empty', your IP is rate-limited. Use 'Fetch Live' buttons below sparingly.")

    # Input controls
    kw_text = st.text_input("Keywords (comma-separated, max 5)", "AI, Data")
    keywords = [x.strip() for x in kw_text.split(",") if x.strip()][:5] or ["AI"]

    region_pick = st.selectbox("Region", ["Australia", "Perth", "Worldwide", "Custom"], index=0)
    geo_text = st.text_input("Custom (country/ISO-2 or Perth)", "Australia") if region_pick == "Custom" else region_pick
    geo_code, scope_label, city_filter = resolve_geo(geo_text)

    st.info(f"🌍 Scope: **{scope_label}** | Geo Code: `{geo_code or 'worldwide'}`")

    # ====================
    # 1. TRENDING SECTION
    # ====================
    st.markdown("---")
    st.subheader("🔥 Trending Searches")
    
    col1, col2 = st.columns([3, 1])
    with col2:
        fetch_trending = st.button("🔴 Fetch Live Trending", use_container_width=True)

    if fetch_trending:
        with st.spinner("Fetching trending data..."):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Daily Trending**")
                daily = fetch_trending_daily(geo_code.lower() if geo_code else "australia")
                if not daily.empty:
                    st.dataframe(daily.head(15), use_container_width=True, height=400)
                else:
                    st.warning("No daily trending data.")
            
            with c2:
                st.markdown("**Realtime Trending**")
                realtime = fetch_trending_realtime(geo_code or "AU", "all")
                if not realtime.empty:
                    st.dataframe(realtime.head(15), use_container_width=True, height=400)
                else:
                    st.warning("No realtime trending data.")
    else:
        st.info("👆 Click **Fetch Live Trending** to load current trending searches")

    # ====================
    # 2. INTEREST OVER TIME
    # ====================
    st.markdown("---")
    st.subheader("📈 Interest Over Time")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        tf = st.selectbox("Timeframe", ["now 7-d", "today 3-m", "today 12-m", "today 5-y"], index=2)
    with col2:
        fetch_iot_btn = st.button("🔴 Fetch Live IoT", use_container_width=True)

    if fetch_iot_btn:
        with st.spinner(f"Fetching Interest Over Time for {keywords}..."):
            iot = fetch_iot(tuple(keywords), tf, geo_code)
            if not iot.empty:
                cols = [c for c in iot.columns if c != 'date']
                st.plotly_chart(line_with_spikes(iot, cols), use_container_width=True)
                
                # Download button
                csv = iot.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "⬇️ Download IoT Data (CSV)",
                    csv,
                    f"iot_{'-'.join(keywords[:3])}_{tf.replace(' ', '_')}.csv",
                    mime="text/csv"
                )
                st.markdown("<span class='badge-ok'>✓ Live data loaded</span>", unsafe_allow_html=True)
            else:
                st.error("❌ No IoT data returned. Try:\n- Fewer keywords\n- Different timeframe\n- Different region")
    else:
        st.info(f"👆 Click **Fetch Live IoT** to load trends for: **{', '.join(keywords)}**")

    # ====================
    # 3. REGIONAL INTEREST
    # ====================
    st.markdown("---")
    st.subheader("🗺️ Regional Interest (Animated)")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        series = st.selectbox("Series for map", keywords, key="map_series")
    with col2:
        months = st.slider("Months", 3, 12, 6)
    with col3:
        fetch_map = st.button("🔴 Fetch Live Map", use_container_width=True)

    if fetch_map:
        with st.spinner(f"Building animated map for '{series}' ({months} months)..."):
            frames = fetch_region_frames(
                series, 
                months, 
                geo_code, 
                resolution=("CITY" if city_filter else "COUNTRY")
            )
            if not frames.empty:
                st.plotly_chart(animated_choropleth(frames), use_container_width=True)
                
                # Download
                csv = frames.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "⬇️ Download Regional Data (CSV)",
                    csv,
                    f"regions_{series.replace(' ', '_')}_{months}m.csv",
                    mime="text/csv"
                )
                st.markdown("<span class='badge-ok'>✓ Live map loaded</span>", unsafe_allow_html=True)
            else:
                st.error("❌ No regional data. Try:\n- Fewer months\n- Different keyword\n- Broader region (Worldwide)")
    else:
        st.info(f"👆 Click **Fetch Live Map** to load regional data for: **{series}**")

    # ====================
    # 4. RELATED QUERIES
    # ====================
    st.markdown("---")
    st.subheader("🔤 Related Queries & Word Cloud")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        kw_wc = st.selectbox("Keyword for word cloud", options=keywords, key="wc_keyword")
    with col2:
        fetch_related_btn = st.button("🔴 Fetch Live Related", use_container_width=True)

    if fetch_related_btn:
        with st.spinner(f"Fetching related queries for '{kw_wc}'..."):
            rq = fetch_related(kw_wc, geo_code)
            top = sanitize_related_df(rq.get("top"))
            rising = sanitize_related_df(rq.get("rising"))
            
            if not top.empty or not rising.empty:
                # Word cloud
                st.image(
                    wordcloud_from_related(top, rising), 
                    caption=f"Related Queries — {kw_wc}",
                    use_container_width=True
                )
                
                # Tables in expander
                with st.expander("📊 Show Data Tables"):
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown("**Top Queries**")
                        st.dataframe(top.head(50), use_container_width=True, height=400)
                    with c2:
                        st.markdown("**Rising Queries**")
                        st.dataframe(rising.head(50), use_container_width=True, height=400)
                
                # Download
                combined = pd.DataFrame({
                    "top_query": top["query"].tolist() + [""] * max(0, len(rising) - len(top)),
                    "top_value": top["value"].tolist() + [""] * max(0, len(rising) - len(top)),
                    "rising_query": rising["query"].tolist() + [""] * max(0, len(top) - len(rising)),
                    "rising_value": rising["value"].tolist() + [""] * max(0, len(top) - len(rising)),
                })
                csv = combined.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "⬇️ Download Related Queries (CSV)",
                    csv,
                    f"related_{kw_wc.replace(' ', '_')}.csv",
                    mime="text/csv"
                )
                st.markdown("<span class='badge-ok'>✓ Live data loaded</span>", unsafe_allow_html=True)
            else:
                st.error("❌ No related queries found. Keyword might be too niche or rate-limited.")
    else:
        st.info(f"👆 Click **Fetch Live Related** to load queries for: **{kw_wc}**")

    # Footer tips
    st.markdown("---")
    st.markdown("""
    ### 💡 Tips to Avoid Rate Limits
    - **Wait 3-5 seconds** between clicking fetch buttons
    - Use **broader regions** (Worldwide > Country > City)
    - Try **shorter timeframes** (7-d instead of 5-y)
    - **Reduce keywords** (2-3 instead of 5)
    - If blocked: **wait 1 hour** or add proxies in `.streamlit/secrets.toml`
    """)