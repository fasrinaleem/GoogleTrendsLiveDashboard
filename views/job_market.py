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


@st.cache_data(show_spinner=True, ttl=180)
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

    tabs = st.tabs(
        ["Overview", "Trends by Date", "Regional Map", "Top & Rising", "Keyword↔Role Correlations"]
    )

    # --- Overview ---
    with tabs[0]:
        st.markdown(
            f"<div class='section'><div class='section-h'><h2>Interest Over Time</h2><div class='chip'>{scope_label}</div></div>",
            unsafe_allow_html=True,
        )
        live = fetch_iot(tuple(roles), tf, geo_code)
        if not live.empty:
            cols = [c for c in live.columns if c != "date"]
            st.plotly_chart(
                line_with_spikes(live, cols),
                use_container_width=True,
            )
            st.markdown("<span class='badge-ok'>Live ✓</span>", unsafe_allow_html=True)
        else:
            st.info("No series returned. Try a different timeframe/region or fewer roles.")
        st.markdown("</div>", unsafe_allow_html=True)

    # --- Trends by Date (subset) ---
    with tabs[1]:
        st.markdown(
            f"<div class='section'><div class='section-h'><h2>Trends by Date</h2><div class='chip'>{scope_label}</div></div>",
            unsafe_allow_html=True,
        )
        live2 = fetch_iot(tuple(roles), tf, geo_code)
        if not live2.empty:
            all_series = [c for c in live2.columns if c != "date"]
            pick = st.multiselect(
                "Series to show",
                all_series,
                default=all_series[: min(3, len(all_series))],
            )
            if pick:
                kept = ["date"] + [c for c in pick if c in live2.columns]
                st.plotly_chart(
                    line_with_spikes(live2[kept], pick),
                    use_container_width=True,
                )
                st.markdown("<span class='badge-ok'>Live ✓</span>", unsafe_allow_html=True)
        else:
            st.info("No series to display.")
        st.markdown("</div>", unsafe_allow_html=True)

    # --- Regional Map ---
    with tabs[2]:
        st.markdown(
            f"<div class='section'><div class='section-h'><h2>Interest by Region (animated)</h2><div class='chip'>{scope_label}</div></div>",
            unsafe_allow_html=True,
        )
        series_for_map = st.selectbox("Series to map", roles, index=0)
        months = st.slider("Animated map — months", 3, 12, 6)
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
            st.download_button(
                "⬇️ Regions CSV",
                frames.to_csv(index=False).encode("utf-8"),
                "job_regions_all_frames.csv",
            )
            st.markdown("<span class='badge-ok'>Live ✓</span>", unsafe_allow_html=True)
        else:
            st.info("No region data returned. Try fewer months or another role.")
        st.markdown("</div>", unsafe_allow_html=True)

    # --- Top & Rising ---
    with tabs[3]:
        st.markdown(
            f"<div class='section'><div class='section-h'><h2>Top & Rising Related Keywords</h2><div class='chip'>{scope_label}</div></div>",
            unsafe_allow_html=True,
        )
        source = st.selectbox(
            "Source", options=roles + ["All selected roles (combined)"], index=0
        )

        if source == "All selected roles (combined)":
            agg_top, agg_ris = {}, {}
            for r in roles:
                rq = fetch_related(r, geo_code)
                top_df = sanitize_related_df(rq.get("top"))
                ris_df = sanitize_related_df(rq.get("rising"))
                for _, row in top_df.iterrows():
                    agg_top[row["query"]] = agg_top.get(row["query"], 0) + int(row["value"])
                for _, row in ris_df.iterrows():
                    agg_ris[row["query"]] = agg_ris.get(row["query"], 0) + int(row["value"])
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

        st.image(
            wordcloud_from_related(top, rising),
            caption=f"Related — {source}",
            use_container_width=True,
        )

        with st.expander("Show tables"):
            c1, c2 = st.columns(2)
            with c1:
                st.write("### Top")
                st.dataframe(top.head(150), use_container_width=True, height=360)
            with c2:
                st.write("### Rising")
                st.dataframe(rising.head(150), use_container_width=True, height=360)
        st.markdown("<span class='badge-ok'>Live ✓</span>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # --- Keyword ↔ Role Correlations ---
    with tabs[4]:
        st.markdown(
            f"<div class='section'><div class='section-h'><h2>Keyword ↔ Role Correlations</h2><div class='chip'>{scope_label}</div></div>",
            unsafe_allow_html=True,
        )
        base_role = st.selectbox("Role source for keywords", roles, index=0)
        dataset_choice = st.radio(
            "Use keywords from", ["Top", "Rising"], horizontal=True, index=0
        )
        max_kw = st.slider("Max keywords", 5, 40, 20, 5)

        rq = fetch_related(base_role, geo_code)
        src_df = sanitize_related_df(
            rq.get("top" if dataset_choice == "Top" else "rising")
        )

        with st.spinner("Computing correlations..."):
            corr_df = _compute_keyword_role_correlations(
                roles=tuple(roles),
                related_df=src_df,
                timeframe=tf,
                geo=geo_code,
                max_keywords=int(max_kw),
            )

        if corr_df is not None and not corr_df.empty:
            st.subheader("Heatmap")
            _render_heatmap(corr_df, tuple(roles))
            st.subheader("Table")
            role_cols = [f"corr_{r}" for r in roles if f"corr_{r}" in corr_df.columns]
            disp = corr_df[["query", "best_match"] + role_cols].copy()
            st.dataframe(disp, use_container_width=True, height=520)
            st.download_button(
                "⬇️ Download correlations (CSV)",
                disp.to_csv(index=False).encode("utf-8"),
                file_name=f"correlations_{dataset_choice.lower()}_{scope_label.replace(' ','_')}.csv",
                mime="text/csv",
            )
            st.markdown("<span class='badge-ok'>Live ✓</span>", unsafe_allow_html=True)
        else:
            st.info(
                "No correlations could be computed. Try fewer roles/keywords, a shorter timeframe, or a broader region."
            )
        st.markdown("</div>", unsafe_allow_html=True)
