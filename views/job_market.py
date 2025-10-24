# views/job_market.py
from __future__ import annotations

import time
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from utils.geo import resolve_geo
from components.charts import (
    line_with_spikes,
    animated_choropleth,
    wordcloud_from_related,
)
from services.trends_client import (
    fetch_iot,
    fetch_region_frames,
    fetch_related,
    sanitize_related_df,
    get_health,
)


# ---------- Small helpers ----------
def _batch(items: List[str], n: int = 5):
    for i in range(0, len(items), n):
        yield items[i : i + n]


def _render_heatmap(corr_df: pd.DataFrame, roles: Tuple[str, ...]):
    """Compact heatmap for correlations (keyword x role)."""
    cols = [f"corr_{r}" for r in roles if f"corr_{r}" in corr_df.columns]
    if not cols or corr_df.empty:
        st.info("No correlation values available to render.")
        return
    mat = corr_df[["query"] + cols].set_index("query")
    fig = px.imshow(
        mat,
        aspect="auto",
        color_continuous_scale="RdBu",
        zmin=-1.0,
        zmax=1.0,
        labels=dict(color="Pearson r"),
        title=None,
    )
    fig.update_layout(template="plotly_white", margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)


def _compute_keyword_role_correlations(
    roles: Tuple[str, ...],
    related_df: pd.DataFrame,
    timeframe: str,
    geo: str,
    max_keywords: int = 20,
) -> pd.DataFrame:
    """
    Compute Pearson correlation between each related keyword's IoT series
    and every selected role's IoT series.
    """
    if related_df is None or related_df.empty:
        return pd.DataFrame()

    roles = tuple([r for r in roles if isinstance(r, str) and r.strip()])[:5]
    roles_iot = fetch_iot(roles, timeframe, geo)
    if roles_iot is None or roles_iot.empty or "date" not in roles_iot.columns:
        return pd.DataFrame()

    rel_kw = (
        related_df["query"]
        .astype(str)
        .str.strip()
        .replace("", np.nan)
        .dropna()
        .drop_duplicates()
        .tolist()
    )[:max_keywords]
    if not rel_kw:
        return pd.DataFrame()

    # Fetch keyword IoT in batches (PyTrends limit 5 per payload)
    kw_series: Dict[str, pd.Series] = {}
    for chunk in _batch(rel_kw, 5):
        dfk = fetch_iot(tuple(chunk), timeframe, geo)
        if dfk is None or dfk.empty or "date" not in dfk.columns:
            continue
        dfk = dfk.copy()
        dfk["date"] = pd.to_datetime(dfk["date"])
        for c in [c for c in dfk.columns if c != "date"]:
            kw_series[c] = dfk.set_index("date")[c].astype(float)
        time.sleep(0.08)  # gentle pacing

    if not kw_series:
        return pd.DataFrame()

    roles_df = roles_iot.copy()
    roles_df["date"] = pd.to_datetime(roles_df["date"])
    roles_df = (
        roles_df.drop_duplicates(subset=["date"])
        .sort_values("date")
        .set_index("date")
    )
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
            try:
                row[f"corr_{r}"] = float(joint["kw"].corr(joint[r])) if r in joint.columns else np.nan
            except Exception:
                row[f"corr_{r}"] = np.nan
        out_rows.append(row)

    corr_df = pd.DataFrame(out_rows)

    def _best(row):
        vals = {r: row.get(f"corr_{r}") for r in roles}
        vals = {k: v for k, v in vals.items() if pd.notna(v)}
        if not vals:
            return ""
        best_role = max(vals, key=lambda k: vals[k])
        return f"{best_role} ({vals[best_role]:.2f})"

    corr_df["best_match"] = corr_df.apply(_best, axis=1)
    sort_cols = [f"corr_{r}" for r in roles]
    corr_df["_max_corr"] = corr_df[sort_cols].max(axis=1, skipna=True)
    corr_df = corr_df.sort_values("_max_corr", ascending=False).drop(columns="_max_corr")
    return corr_df


# ---------- Main view ----------
def render():
    st.markdown(
        """<div class="hero"><h1>💼 Job Market</h1>
        <div class="subtle">Roles • Trends • Regions • Related • Correlations</div></div>""",
        unsafe_allow_html=True,
    )

    # Debug status
    with st.expander("🔍 Debug Status"):
        st.json(get_health())
        st.caption("⚠️ If everything says 'empty', your IP is rate-limited. Use 'Fetch Live' buttons sparingly.")

    # Configuration
    roles_all = [
        "Data Analyst",
        "Data Scientist",
        "Software Developer",
        "Full Stack Developer",
        "Data Engineer",
        "Business Analyst",
        "Machine Learning Engineer",
    ]
    roles = (
        st.multiselect("Job Roles (max 5)", roles_all, default=roles_all[:2])[:5]
        or roles_all[:2]
    )

    # Region / timeframe
    region_pick = st.selectbox(
        "Region", ["Australia", "Perth", "Worldwide", "Custom"], index=0
    )
    geo_text = (
        st.text_input("Custom (country/ISO-2 or Perth)", value="Australia")
        if region_pick == "Custom"
        else region_pick
    )
    geo_code, scope_label, city_filter = resolve_geo(geo_text)
    
    tf = st.selectbox(
        "Timeframe", ["now 7-d", "today 3-m", "today 12-m", "today 5-y"], index=2
    )

    st.info(f"🌍 Scope: **{scope_label}** | Selected Roles: **{', '.join(roles)}**")

    tabs = st.tabs([
        "📊 Overview", 
        "📈 Trends by Date", 
        "🗺️ Regional Map", 
        "🔤 Top & Rising", 
        "🔗 Keyword↔Role Correlations"
    ])

    # =========================
    # TAB 1: OVERVIEW
    # =========================
    with tabs[0]:
        st.markdown(
            f"<div class='section'><div class='section-h'><h2>Interest Over Time</h2><div class='chip'>{scope_label}</div></div>",
            unsafe_allow_html=True,
        )
        
        fetch_overview = st.button("🔴 Fetch Live Overview", use_container_width=True, key="fetch_overview")
        
        if fetch_overview:
            with st.spinner(f"Fetching IoT for {len(roles)} roles..."):
                live = fetch_iot(tuple(roles), tf, geo_code)
                if not live.empty:
                    cols = [c for c in live.columns if c != "date"]
                    st.plotly_chart(
                        line_with_spikes(live, cols),
                        use_container_width=True,
                    )
                    
                    # Download
                    csv = live.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "⬇️ Download Overview Data (CSV)",
                        csv,
                        f"overview_{'-'.join(roles[:2])}_{tf.replace(' ', '_')}.csv",
                        mime="text/csv"
                    )
                    st.markdown("<span class='badge-ok'>✓ Live data loaded</span>", unsafe_allow_html=True)
                else:
                    st.error("❌ No series returned. Try:\n- Fewer roles\n- Different timeframe\n- Broader region")
        else:
            st.info(f"👆 Click **Fetch Live Overview** to load trends for: **{', '.join(roles)}**")
        
        st.markdown("</div>", unsafe_allow_html=True)

    # =========================
    # TAB 2: TRENDS BY DATE
    # =========================
    with tabs[1]:
        st.markdown(
            f"<div class='section'><div class='section-h'><h2>Trends by Date (Filtered)</h2><div class='chip'>{scope_label}</div></div>",
            unsafe_allow_html=True,
        )
        
        fetch_trends = st.button("🔴 Fetch Live Trends", use_container_width=True, key="fetch_trends")
        
        if fetch_trends:
            with st.spinner("Fetching trends data..."):
                live2 = fetch_iot(tuple(roles), tf, geo_code)
                if not live2.empty:
                    all_series = [c for c in live2.columns if c != "date"]
                    pick = st.multiselect(
                        "Select roles to display",
                        all_series,
                        default=all_series[: min(3, len(all_series))],
                    )
                    if pick:
                        kept = ["date"] + [c for c in pick if c in live2.columns]
                        st.plotly_chart(
                            line_with_spikes(live2[kept], pick),
                            use_container_width=True,
                        )
                        
                        # Download
                        csv = live2[kept].to_csv(index=False).encode('utf-8')
                        st.download_button(
                            "⬇️ Download Filtered Trends (CSV)",
                            csv,
                            f"trends_filtered_{'-'.join(pick[:2])}.csv",
                            mime="text/csv"
                        )
                        st.markdown("<span class='badge-ok'>✓ Live data loaded</span>", unsafe_allow_html=True)
                else:
                    st.error("❌ No series to display. Try fewer roles or different settings.")
        else:
            st.info("👆 Click **Fetch Live Trends** to load data, then select roles to visualize")
        
        st.markdown("</div>", unsafe_allow_html=True)

    # =========================
    # TAB 3: REGIONAL MAP
    # =========================
    with tabs[2]:
        st.markdown(
            f"<div class='section'><div class='section-h'><h2>Interest by Region (Animated)</h2><div class='chip'>{scope_label}</div></div>",
            unsafe_allow_html=True,
        )
        
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            series_for_map = st.selectbox("Role to map", roles, index=0, key="map_role")
        with col2:
            months = st.slider("Animated map — months", 3, 12, 6)
        with col3:
            fetch_map = st.button("🔴 Fetch Map", use_container_width=True, key="fetch_map")
        
        if fetch_map:
            with st.spinner(f"Building {months}-month animated map for '{series_for_map}'..."):
                frames = fetch_region_frames(
                    series_for_map,
                    months,
                    geo_code,
                    resolution=("CITY" if city_filter else "COUNTRY"),
                )
                if isinstance(frames, pd.DataFrame) and not frames.empty:
                    st.plotly_chart(
                        animated_choropleth(frames, "Regional interest"),
                        use_container_width=True,
                    )
                    
                    # Download
                    csv = frames.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "⬇️ Download Regional Data (CSV)",
                        csv,
                        f"job_regions_{series_for_map.replace(' ', '_')}_{months}m.csv",
                        mime="text/csv"
                    )
                    st.markdown("<span class='badge-ok'>✓ Live map loaded</span>", unsafe_allow_html=True)
                else:
                    st.error("❌ No region data. Try:\n- Fewer months (3-6)\n- Different role\n- Broader region")
        else:
            st.info(f"👆 Click **Fetch Map** to load regional data for: **{series_for_map}**")
        
        st.markdown("</div>", unsafe_allow_html=True)

    # =========================
    # TAB 4: TOP & RISING
    # =========================
    with tabs[3]:
        st.markdown(
            f"<div class='section'><div class='section-h'><h2>Top & Rising Related Keywords</h2><div class='chip'>{scope_label}</div></div>",
            unsafe_allow_html=True,
        )
        
        col1, col2 = st.columns([3, 1])
        with col1:
            source = st.selectbox(
                "Source role", 
                options=roles + ["All selected roles (combined)"], 
                index=0,
                key="related_source"
            )
        with col2:
            fetch_related_btn = st.button("🔴 Fetch Related", use_container_width=True, key="fetch_related")

        if fetch_related_btn:
            with st.spinner(f"Fetching related keywords for '{source}'..."):
                if source == "All selected roles (combined)":
                    agg_top, agg_ris = {}, {}
                    progress_bar = st.progress(0)
                    for idx, r in enumerate(roles):
                        rq = fetch_related(r, geo_code)
                        top_df = sanitize_related_df(rq.get("top"))
                        ris_df = sanitize_related_df(rq.get("rising"))
                        for _, row in top_df.iterrows():
                            agg_top[row["query"]] = agg_top.get(row["query"], 0) + int(row["value"])
                        for _, row in ris_df.iterrows():
                            agg_ris[row["query"]] = agg_ris.get(row["query"], 0) + int(row["value"])
                        progress_bar.progress((idx + 1) / len(roles))
                        time.sleep(1)  # Rate limit protection
                    progress_bar.empty()
                    
                    top = (
                        pd.DataFrame([{"query": k, "value": v} for k, v in agg_top.items()])
                        .sort_values("value", ascending=False)
                        if agg_top else pd.DataFrame(columns=["query", "value"])
                    )
                    rising = (
                        pd.DataFrame([{"query": k, "value": v} for k, v in agg_ris.items()])
                        .sort_values("value", ascending=False)
                        if agg_ris else pd.DataFrame(columns=["query", "value"])
                    )
                else:
                    rq = fetch_related(source, geo_code)
                    top = sanitize_related_df(rq.get("top"))
                    rising = sanitize_related_df(rq.get("rising"))

                if not top.empty or not rising.empty:
                    # Word cloud
                    st.image(
                        wordcloud_from_related(top, rising),
                        caption=f"Related Keywords — {source}",
                        use_container_width=True,
                    )

                    # Tables
                    with st.expander("📊 Show Data Tables"):
                        c1, c2 = st.columns(2)
                        with c1:
                            st.markdown("**Top Queries**")
                            st.dataframe(top.head(150), use_container_width=True, height=360)
                        with c2:
                            st.markdown("**Rising Queries**")
                            st.dataframe(rising.head(150), use_container_width=True, height=360)
                    
                    # Download
                    combined = pd.DataFrame({
                        "top_query": top["query"].tolist() + [""] * max(0, len(rising) - len(top)),
                        "top_value": top["value"].tolist() + [""] * max(0, len(rising) - len(top)),
                        "rising_query": rising["query"].tolist() + [""] * max(0, len(top) - len(rising)),
                        "rising_value": rising["value"].tolist() + [""] * max(0, len(top) - len(rising)),
                    })
                    csv = combined.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "⬇️ Download Related Keywords (CSV)",
                        csv,
                        f"related_{source.replace(' ', '_')}.csv",
                        mime="text/csv"
                    )
                    st.markdown("<span class='badge-ok'>✓ Live data loaded</span>", unsafe_allow_html=True)
                else:
                    st.error("❌ No related queries found. Try a different role or broader region.")
        else:
            st.info(f"👆 Click **Fetch Related** to load keywords for: **{source}**")
        
        st.markdown("</div>", unsafe_allow_html=True)

    # =========================
    # TAB 5: CORRELATIONS
    # =========================
    with tabs[4]:
        st.markdown(
            f"<div class='section'><div class='section-h'><h2>Keyword ↔ Role Correlations</h2><div class='chip'>{scope_label}</div></div>",
            unsafe_allow_html=True,
        )
        
        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
        with col1:
            base_role = st.selectbox("Role source for keywords", roles, index=0, key="corr_role")
        with col2:
            dataset_choice = st.radio(
                "Use keywords from", ["Top", "Rising"], horizontal=True, index=0, key="corr_type"
            )
        with col3:
            max_kw = st.slider("Max keywords", 5, 40, 20, 5, key="corr_max")
        with col4:
            compute_corr = st.button("🔴 Compute", use_container_width=True, key="compute_corr")

        if compute_corr:
            with st.spinner("Step 1/3: Fetching related keywords..."):
                rq = fetch_related(base_role, geo_code)
                src_df = sanitize_related_df(
                    rq.get("top" if dataset_choice == "Top" else "rising")
                )

            if src_df is not None and not src_df.empty:
                with st.spinner("Step 2/3: Fetching IoT for keywords (this may take 30-60s)..."):
                    corr_df = _compute_keyword_role_correlations(
                        roles=tuple(roles),
                        related_df=src_df,
                        timeframe=tf,
                        geo=geo_code,
                        max_keywords=int(max_kw),
                    )

                if corr_df is not None and not corr_df.empty:
                    st.success("Step 3/3: Done! ✓")
                    
                    # Heatmap
                    st.subheader("Correlation Heatmap")
                    _render_heatmap(corr_df, tuple(roles))
                    
                    # Table
                    st.subheader("Correlation Table")
                    role_cols = [f"corr_{r}" for r in roles if f"corr_{r}" in corr_df.columns]
                    disp = corr_df[["query", "best_match"] + role_cols].copy()
                    st.dataframe(disp, use_container_width=True, height=520)
                    
                    # Download
                    csv = disp.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "⬇️ Download Correlations (CSV)",
                        csv,
                        file_name=f"correlations_{dataset_choice.lower()}_{scope_label.replace(' ','_')}.csv",
                        mime="text/csv",
                    )
                    st.markdown("<span class='badge-ok'>✓ Correlations computed</span>", unsafe_allow_html=True)
                else:
                    st.error(
                        "❌ Could not compute correlations. Try:\n"
                        "- Fewer keywords (5-10)\n"
                        "- Shorter timeframe (today 3-m)\n"
                        "- Different base role"
                    )
            else:
                st.error("❌ No keywords found for correlation. Try a different role.")
        else:
            st.info(f"👆 Click **Compute** to calculate correlations between **{base_role}** keywords and all selected roles")
            st.warning("⚠️ This operation makes many API calls and may take 30-60 seconds. Use sparingly!")
        
        st.markdown("</div>", unsafe_allow_html=True)

    # =========================
    # JOB OPENINGS SECTION
    # =========================
    st.markdown("---")
    st.subheader("🔗 Job Openings Quick Links")
    
    selected_role = st.selectbox("Select role to search", roles, key="job_search_role")
    search_query = selected_role.replace(" ", "%20")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.link_button(
            "🔵 LinkedIn Jobs",
            f"https://www.linkedin.com/jobs/search/?keywords={search_query}&location=Australia",
            use_container_width=True
        )
    with col2:
        st.link_button(
            "🟡 Seek.com.au",
            f"https://www.seek.com.au/{search_query}-jobs",
            use_container_width=True
        )
    with col3:
        st.link_button(
            "🔴 Indeed Australia",
            f"https://au.indeed.com/jobs?q={search_query}",
            use_container_width=True
        )

    # Footer tips
    st.markdown("---")
    st.markdown("""
    ### 💡 Tips for Job Market Analysis
    - **Start with Overview** to get general trends
    - **Wait 5-10 seconds** between fetch operations
    - **Correlations are resource-intensive** - use last after other tabs
    - For combined data: expect 1-2 minutes processing time
    - If rate-limited: wait 1 hour or add proxies to `.streamlit/secrets.toml`
    """)