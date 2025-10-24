# app_gtrends.py
from __future__ import annotations

import time
from typing import Dict, Any, List, Optional, Tuple
from urllib.parse import quote

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ----- Local modules (make sure these files exist) -----
from viz import kpi_card, line_with_spikes, animated_choropleth, wordcloud_from_related
from trends import (
    get_client as t_get_client,
    interest_over_time as t_iot,
    monthly_region_frames as t_frames,
    related_queries as t_related,
    trending_today as t_trend_today,
    trending_realtime as t_trend_rt,
)

# ───────────────────── Page & theme ─────────────────────
st.set_page_config(page_title="Trends Hub", page_icon="📊", layout="wide")

# ───────────────────── Load CSS file ─────────────────────
def load_css(path: str = "styles.css"):
    try:
        with open(path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning(f"Could not find {path}. Make sure it exists next to this script.")

load_css("styles.css")

# ───────────────────── Small utils ─────────────────────
def chart_key(prefix: str, *parts) -> str:
    return (prefix + ":" + "|".join(str(p) for p in parts if p is not None))[:200]

def to_list(x) -> list[str]:
    if x is None:
        return []
    if isinstance(x, str):
        return [x]
    return list(x)

def country_name_to_iso2(name: str | None) -> Optional[str]:
    if not name:
        return None
    m = {"australia":"AU","united states":"US","usa":"US","united kingdom":"GB","uk":"GB","india":"IN",
         "canada":"CA","singapore":"SG","new zealand":"NZ","germany":"DE","france":"FR","japan":"JP","brazil":"BR"}
    return m.get(name.strip().lower())

def resolve_geo(user_input: str) -> Tuple[str, str, Optional[str]]:
    s = (user_input or "").strip()
    if not s or s.lower() in {"world","worldwide","global"}:
        return "", "Worldwide", None
    if s.lower() == "perth":
        return "AU", "Perth, Australia", "Perth"
    if "," in s:
        city, country = s.split(",", 1)
        iso = country_name_to_iso2(country) or (country.strip().upper() if len(country.strip()) == 2 else None)
        return (iso or ""), f"{city.strip()}, {country.strip()}", city.strip()
    iso = country_name_to_iso2(s) or (s.upper() if len(s) == 2 else "")
    return (iso or ""), (s.title() if iso else s), None

def _sanitize_related_df(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame(columns=["query","value"])
    out = df.copy()

    def fix(v):
        if pd.isna(v):
            return 0
        if isinstance(v, (int, float)):
            return int(v)
        s = str(v).strip().lower()
        if s == "breakout":
            return 120
        s = s.replace("%", "")
        try:
            return int(round(float(s)))
        except Exception:
            return 0

    out["value"] = out["value"].apply(fix).astype(int)
    out["query"] = out["query"].astype(str)
    return out

# ───────────────────── Cached wrappers (live-first) ─────────────────────
@st.cache_resource(show_spinner=False)
def get_client():
    # Perth timezone (UTC+8 → 480 minutes)
    return t_get_client(hl="en-US", tz=480)

@st.cache_data(show_spinner=True, ttl=180)
def fetch_iot(keys: Tuple[str, ...], timeframe: str, geo: str) -> pd.DataFrame:
    try:
        py = get_client()
        return t_iot(py, list(keys[:5]), timeframe=timeframe, geo=geo)
    except Exception:
        return pd.DataFrame()

@st.cache_data(show_spinner=True, ttl=180)
def fetch_region_frames(keyword: str, months: int, geo: str, resolution="COUNTRY") -> pd.DataFrame:
    try:
        py = get_client()
        return t_frames(py, keyword=keyword, months=int(months), geo=geo, resolution=resolution)
    except Exception:
        return pd.DataFrame()

@st.cache_data(show_spinner=True, ttl=180)
def fetch_related(keyword: str, geo: str) -> Dict[str, pd.DataFrame]:
    try:
        py = get_client()
        slot = t_related(py, keyword)
        top = slot.get("top") if isinstance(slot, dict) else None
        ris = slot.get("rising") if isinstance(slot, dict) else None
        return {
            "top": top if isinstance(top, pd.DataFrame) else pd.DataFrame(columns=["query","value"]),
            "rising": ris if isinstance(ris, pd.DataFrame) else pd.DataFrame(columns=["query","value"]),
        }
    except Exception:
        return {"top": pd.DataFrame(columns=["query","value"]), "rising": pd.DataFrame(columns=["query","value"])}

@st.cache_data(show_spinner=False, ttl=180)
def fetch_trending_daily(geo_name: str="australia") -> pd.DataFrame:
    try:
        py = get_client()
        return t_trend_today(py, geo=geo_name) or pd.DataFrame()
    except Exception:
        return pd.DataFrame()

@st.cache_data(show_spinner=False, ttl=180)
def fetch_trending_realtime(geo_code: str="AU", cat: str="all") -> pd.DataFrame:
    try:
        py = get_client()
        return t_trend_rt(py, geo=geo_code, cat=cat) or pd.DataFrame()
    except Exception:
        return pd.DataFrame()

# ───────────────────── Correlations (keywords ↔ roles) ─────────────────────
def _batch(items: List[str], n: int = 5):
    for i in range(0, len(items), n):
        yield items[i:i+n]

@st.cache_data(show_spinner=True, ttl=180)
def compute_keyword_role_correlations(
    roles: Tuple[str, ...],
    related_df: pd.DataFrame,
    timeframe: str,
    geo: str,
    max_keywords: int = 20
) -> pd.DataFrame:
    """Compute Pearson correlation between related keywords' IoT series and each role's IoT series."""
    if related_df is None or related_df.empty:
        return pd.DataFrame()

    roles = tuple([r for r in roles if isinstance(r, str) and r.strip()])[:5]
    roles_iot = fetch_iot(roles, timeframe, geo)
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
        dfk = fetch_iot(tuple(chunk), timeframe, geo)
        if dfk is None or dfk.empty or "date" not in dfk.columns:
            continue
        dfk = dfk.copy()
        dfk["date"] = pd.to_datetime(dfk["date"])
        for c in [c for c in dfk.columns if c != "date"]:
            kw_series[c] = dfk.set_index("date")[c].astype(float)
        time.sleep(0.08)

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

def render_correlation_heatmap(corr_df: pd.DataFrame, roles: Tuple[str, ...]):
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

# ───────────────────── Sidebar ─────────────────────
def sidebar() -> Dict[str, Any]:
    with st.sidebar:
        st.header("🛠️ Controls")

        st.markdown("**View**")
        view = st.selectbox("Choose view", ["Trends Studio", "Job Market"], key="view")

        st.markdown("**Region**")
        q = st.radio("Region", ["Australia","Perth","Worldwide","Custom"], index=0, key="region")
        geo_text = st.text_input("Custom (country/ISO-2 or Perth)", value="Australia", key="geo_text") if q=="Custom" else q

        st.markdown("**Timeframes**")
        if view == "Trends Studio":
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

        st.button("🔄 Force refresh caches", on_click=lambda: [st.cache_data.clear(), st.cache_resource.clear()])

        return {"view":view, "geo_text":geo_text, "ts_timeframe":ts_timeframe, "ts_months":ts_months,
                "jm_timeframe":jm_timeframe, "jm_months":jm_months,
                "line_theme":line_theme, "map_scale":map_scale, "wc_max":cloud_words, "wc_cmap":cloud_map}

# ───────────────────── Sections (live-first) ─────────────────────
def section_front_page():
    st.markdown(
        """
        <div class="section">
          <div class="section-h"><h2>Project Overview</h2></div>
          <p><strong>Case:</strong> Real-time job market interest via Google Trends.</p>
          <p><strong>Problem:</strong> Hiring and learners need timely signals for roles and skills; static reports lag.</p>
          <p><strong>Goal:</strong> Provide an interactive dashboard showing time trends, regional demand, and related skills.</p>
          <p><strong>Audience:</strong> Students, job seekers, career advisors, and hiring teams.</p>
          <p><strong>Dataset:</strong> Google Trends (public, near real-time): Interest Over Time, Interest by Region, Trending, and Related Queries.</p>
          <p><strong>Analyses:</strong> Multi-keyword series, animated region maps, related-keyword clouds, and keyword↔role correlation heatmaps.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

def section_trending(scope_label: str, city_filter: Optional[str]):
    st.markdown(f"<div class='section'><div class='section-h'><h2>🔥 Top Trending Searches Today</h2><div class='chip'>Scope: {scope_label}</div></div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.caption("Daily Trending — Australia (live)")
        live = fetch_trending_daily("australia")
        if not live.empty:
            st.markdown("\n".join(f"- {q}" for q in live["query"].astype(str).head(10)))
            st.markdown("<span class='badge-ok'>Live ✓</span>", unsafe_allow_html=True)
        else:
            st.info("No daily trending data returned. Try again.")
        if st.button("Retry (Daily)"):
            st.cache_data.clear()
    with c2:
        city = city_filter or "Perth"
        st.caption(f"Realtime Trending — filtered for “{city}” (live)")
        live_rt = fetch_trending_realtime("AU", "all")
        if not live_rt.empty:
            name = "title" if "title" in live_rt.columns else live_rt.columns[0]
            mask = live_rt[name].astype(str).str.contains(city, case=False, na=False)
            show = live_rt[mask] if mask.any() else live_rt
            st.markdown("\n".join(f"- **{t}**" for t in show[name].astype(str).head(8)))
            st.markdown("<span class='badge-ok'>Live ✓</span>", unsafe_allow_html=True)
        else:
            st.info("No realtime results now. Try again.")
        if st.button("Retry (Realtime)"):
            st.cache_data.clear()
    st.markdown("</div>", unsafe_allow_html=True)

def section_iot(title: str, keywords: List[str], timeframe: str, geo_code: str):
    st.markdown(f"<div class='section'><div class='section-h'><h2>{title}</h2><div class='chip'>Timeframe: {timeframe}</div></div>", unsafe_allow_html=True)
    live = fetch_iot(tuple(keywords), timeframe, geo_code)
    if not live.empty:
        cols = [c for c in live.columns if c != "date"]
        k1, k2 = st.columns([1, 4])
        with k1:
            now_vals = [int(pd.to_numeric(live[c], errors='coerce').fillna(0).iloc[-1]) for c in cols]
            avg_vals = [int(pd.to_numeric(live[c], errors='coerce').rolling(7, min_periods=1).mean().fillna(0).iloc[-1]) for c in cols]
            st.markdown(kpi_card("Now", f"{now_vals[0]}"), unsafe_allow_html=True)
            st.markdown(kpi_card("7-day Avg", f"{avg_vals[0]}"), unsafe_allow_html=True)
        with k2:
            st.plotly_chart(line_with_spikes(live, cols), use_container_width=True,
                            key=chart_key("iot_live", timeframe, geo_code, tuple(cols)))
        st.download_button("📥 Download Series (CSV)", live.to_csv(index=False).encode("utf-8"),
                           file_name="interest_over_time.csv", mime="text/csv")
        st.markdown("<span class='badge-ok'>Live ✓</span>", unsafe_allow_html=True)
    else:
        st.info("No interest-over-time series returned. Try a different timeframe/region or fewer keywords.")
        if st.button("Retry (Series)"): st.cache_data.clear()
    st.markdown("</div>", unsafe_allow_html=True)

def section_map(series_name: str, months: int, geo_code: str, city_filter: Optional[str], map_scale: str):
    st.markdown(f"<div class='section'><div class='section-h'><h2>🗺️ Animated Map — Interest by Region</h2><div class='chip'>Series: {series_name}</div></div>", unsafe_allow_html=True)
    frames = fetch_region_frames(series_name, months, geo_code,
                                 resolution=("CITY" if city_filter else "COUNTRY"))
    if isinstance(frames, pd.DataFrame) and not frames.empty:
        show = frames
        if city_filter and "region" in show.columns:
            show = show[show["region"].astype(str).str.contains(city_filter, case=False, na=False)]
        fig = animated_choropleth(show, title="Regional interest")
        # swap color scale if requested
        try:
            fig.update_traces(zmin=0)  # keep range sane
            fig.update_layout(coloraxis_colorscale=map_scale)
        except Exception:
            pass
        st.plotly_chart(fig, use_container_width=True,
                        key=chart_key("map_live", series_name, months, geo_code, map_scale))
        st.download_button("⬇️ Download Regions (CSV)", show.to_csv(index=False).encode("utf-8"),
                           "regions_all_frames.csv", "text/csv")
        st.markdown("<span class='badge-ok'>Live ✓</span>", unsafe_allow_html=True)
    else:
        st.info("No region data returned. Try fewer months, a broader region, or a different keyword.")
        if st.button("Retry (Regions)"): st.cache_data.clear()
    st.markdown("</div>", unsafe_allow_html=True)

def section_related(keyword: str, geo_code: str, wc_max: int, wc_cmap: str):
    st.markdown(f"<div class='section'><div class='section-h'><h2>🔤 Related Queries — Word Cloud</h2><div class='chip'>Keyword: {keyword}</div></div>", unsafe_allow_html=True)
    rq = fetch_related(keyword, geo_code)
    top = _sanitize_related_df(rq.get("top", pd.DataFrame()))
    rising = _sanitize_related_df(rq.get("rising", pd.DataFrame()))
    img_buf = wordcloud_from_related(top, rising)
    st.image(img_buf, caption=f"Related queries — {keyword}", use_container_width=True)
    if not top.empty or not rising.empty:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Top keywords")
            st.dataframe(top.head(100), use_container_width=True, height=360)
        with c2:
            st.subheader("Rising keywords")
            st.dataframe(rising.head(100), use_container_width=True, height=360)
        st.markdown("<span class='badge-ok'>Live ✓</span>", unsafe_allow_html=True)
    else:
        st.info("No related queries right now. Try a broader keyword or another region.")
        if st.button("Retry (Related)"): st.cache_data.clear()
    st.markdown("</div>", unsafe_allow_html=True)

# ───────────────────── Views ─────────────────────
def trends_studio(ctrl: Dict[str, Any]):
    section_front_page()
    st.markdown("""<div class="hero"><h1>✨ Trends Studio</h1>
    <div class="subtle">Live data • Caching enabled • Retry buttons if rate-limited</div></div>""", unsafe_allow_html=True)

    kw_text = st.text_input("Keywords (comma-separated, max 5)", "AI, Data", key="ts_kw")
    keywords = [x.strip() for x in kw_text.split(",") if x.strip()][:5] or ["AI"]
    kw_for_related = st.selectbox("Word cloud keyword", options=keywords, index=0, key="kw_wc_pick")

    geo_code, scope_label, city_filter = resolve_geo(ctrl["geo_text"])
    section_trending(scope_label, city_filter)
    section_iot("📈 Interest Over Time (live)", keywords, ctrl["ts_timeframe"], geo_code)
    series_choice = st.selectbox("Series for map", options=keywords, index=0)
    section_map(series_choice, ctrl["ts_months"], geo_code, city_filter, map_scale=ctrl["map_scale"])
    section_related(kw_for_related, geo_code, wc_max=ctrl["wc_max"], wc_cmap=ctrl["wc_cmap"])

def job_market(ctrl: Dict[str, Any]):
    st.markdown("""<div class="hero"><h1>Trends Studio – Job Market</h1>
    <div class="subtle">Select roles • Live series • Regions • Related keywords • Correlations</div></div>""", unsafe_allow_html=True)

    job_roles = ["Data Analyst","Data Scientist","Software Developer","Full Stack Developer","Data Engineer","Business Analyst","Machine Learning Engineer"]
    roles = st.multiselect("Job Roles (max 5)", job_roles, default=job_roles[:2], key="jm_roles")[:5] or job_roles[:2]
    geo_code, scope_label, city_filter = resolve_geo(ctrl["geo_text"])

    tabs = st.tabs(["Overview","Trends by Date","Regional Map","Top & Rising","Keyword↔Role Correlations"])

    # Overview
    with tabs[0]:
        st.markdown(f"<div class='section'><div class='section-h'><h2>Interest Over Time (live)</h2><div class='chip'>Scope: {scope_label}</div></div>", unsafe_allow_html=True)
        live = fetch_iot(tuple(roles), ctrl["jm_timeframe"], geo_code)
        if not live.empty:
            cols = [c for c in live.columns if c != "date"]
            k1, k2 = st.columns([1, 4])
            with k1:
                first = cols[0]
                now_val = int(pd.to_numeric(live[first], errors='coerce').fillna(0).iloc[-1])
                avg7 = int(pd.to_numeric(live[first], errors='coerce').rolling(7, min_periods=1).mean().fillna(0).iloc[-1])
                st.markdown(kpi_card("Now", f"{now_val}"), unsafe_allow_html=True)
                st.markdown(kpi_card("7-day Avg", f"{avg7}"), unsafe_allow_html=True)
                st.markdown(f"<span class='chip'>Timeframe: {ctrl['jm_timeframe']}</span>", unsafe_allow_html=True)
            with k2:
                st.plotly_chart(line_with_spikes(live, cols), use_container_width=True,
                                key=chart_key("jm_overview_live", ctrl["jm_timeframe"], geo_code, tuple(cols)))
            st.markdown("<span class='badge-ok'>Live ✓</span>", unsafe_allow_html=True)
        else:
            st.info("No series returned. Try a different timeframe/region or fewer roles.")
            if st.button("Retry (Overview)"): st.cache_data.clear()
        st.markdown("</div>", unsafe_allow_html=True)

    # Trends by Date (subset of roles)
    with tabs[1]:
        st.markdown(f"<div class='section'><div class='section-h'><h2>Trends by Date (live)</h2><div class='chip'>Scope: {scope_label}</div></div>", unsafe_allow_html=True)
        live2 = fetch_iot(tuple(roles), ctrl["jm_timeframe"], geo_code)
        if not live2.empty:
            all_series = [c for c in live2.columns if c != "date"]
            pick = st.multiselect("Series to show", all_series, default=all_series[:min(3, len(all_series))], key="jm_pick")
            if pick:
                kept = ["date"] + [c for c in pick if c in live2.columns]
                st.plotly_chart(line_with_spikes(live2[kept], pick), use_container_width=True,
                                key=chart_key("jm_trends_by_date_live", ctrl["jm_timeframe"], geo_code, tuple(pick)))
                st.markdown("<span class='badge-ok'>Live ✓</span>", unsafe_allow_html=True)
        else:
            st.info("No series to display.")
            if st.button("Retry (Trends by Date)"): st.cache_data.clear()
        st.markdown("</div>", unsafe_allow_html=True)

    # Regional Map
    with tabs[2]:
        st.markdown(f"<div class='section'><div class='section-h'><h2>Interest by Region (animated, live)</h2><div class='chip'>Scope: {scope_label}</div></div>", unsafe_allow_html=True)
        map_series = st.selectbox("Series to map", roles, index=0, key="jm_map_series")
        frames = fetch_region_frames(map_series, ctrl["jm_months"], geo_code,
                                     resolution=("CITY" if city_filter else "COUNTRY"))
        if isinstance(frames, pd.DataFrame) and not frames.empty:
            show = frames
            if city_filter and "region" in frames.columns:
                show = frames[frames["region"].astype(str).str.contains(city_filter, case=False, na=False)]
            fig = animated_choropleth(show, title="Regional interest")
            try:
                fig.update_layout(coloraxis_colorscale=ctrl["map_scale"])
            except Exception:
                pass
            st.plotly_chart(fig, use_container_width=True,
                            key=chart_key("jm_map_live", map_series, ctrl["jm_months"], geo_code, ctrl["map_scale"]))
            st.download_button("⬇️ Download Regions (CSV)", show.to_csv(index=False).encode("utf-8"),
                               "job_regions_all_frames.csv", "text/csv")
            st.markdown("<span class='badge-ok'>Live ✓</span>", unsafe_allow_html=True)
        else:
            st.info("No region data returned. Try fewer months or another role.")
            if st.button("Retry (Regions – Job Market)"): st.cache_data.clear()
        st.markdown("</div>", unsafe_allow_html=True)

    # Top & Rising
    with tabs[3]:
        st.markdown(f"<div class='section'><div class='section-h'><h2>Top & Rising Related Keywords (live)</h2><div class='chip'>Scope: {scope_label}</div></div>", unsafe_allow_html=True)
        target = st.selectbox("Source", options=roles + ["All selected roles (combined)"], index=0, key="jm_rel_source")
        mode = st.radio("Dataset", ["Top", "Rising"], horizontal=True, index=0, key="jm_rel_mode")

        # Live related for a single role or combined: fetch per-role and aggregate
        if target == "All selected roles (combined)":
            # Aggregate by summing values across roles
            agg_top, agg_ris = {}, {}
            for r in roles:
                rq = fetch_related(r, geo_code)
                top = _sanitize_related_df(rq.get("top", pd.DataFrame()))
                ris = _sanitize_related_df(rq.get("rising", pd.DataFrame()))
                for _, row in top.iterrows():
                    agg_top[row["query"]] = agg_top.get(row["query"], 0) + int(row["value"])
                for _, row in ris.iterrows():
                    agg_ris[row["query"]] = agg_ris.get(row["query"], 0) + int(row["value"])
            top_df = pd.DataFrame([{"query":k, "value":v} for k,v in agg_top.items()]).sort_values("value", ascending=False)
            ris_df = pd.DataFrame([{"query":k, "value":v} for k,v in agg_ris.items()]).sort_values("value", ascending=False)
        else:
            rq = fetch_related(target, geo_code)
            top_df = _sanitize_related_df(rq.get("top", pd.DataFrame()))
            ris_df = _sanitize_related_df(rq.get("rising", pd.DataFrame()))

        img_buf = wordcloud_from_related(top_df, ris_df)
        st.image(img_buf, caption=f"Related — {target}", use_container_width=True)
        st.markdown("<span class='badge-ok'>Live ✓</span>", unsafe_allow_html=True)

        with st.expander("Show tables"):
            c1, c2 = st.columns(2)
            with c1:
                st.write("### Top")
                st.dataframe(top_df.head(150), use_container_width=True, height=360)
            with c2:
                st.write("### Rising")
                st.dataframe(ris_df.head(150), use_container_width=True, height=360)

    # Correlations tab
    with tabs[4]:
        st.markdown(f"<div class='section'><div class='section-h'><h2>Keyword ↔ Role Correlations</h2><div class='chip'>Scope: {scope_label}</div></div>", unsafe_allow_html=True)
        # Use Top or Rising from the first role by default
        base_role = st.selectbox("Role source for keywords", roles, index=0, key="jm_corr_role")
        dataset_choice = st.radio("Use keywords from", ["Top","Rising"], index=0, horizontal=True, key="jm_corr_kind")
        max_kw = st.slider("Max keywords", 5, 40, 20, 5, key="jm_corr_maxkw")

        rq = fetch_related(base_role, geo_code)
        src_df = _sanitize_related_df(rq.get("top" if dataset_choice=="Top" else "rising", pd.DataFrame()))

        with st.spinner("Computing correlations..."):
            corr_df = compute_keyword_role_correlations(
                roles=tuple(roles),
                related_df=src_df,
                timeframe=ctrl["jm_timeframe"],
                geo=geo_code,
                max_keywords=int(max_kw),
            )

        if corr_df is not None and not corr_df.empty:
            st.subheader("Heatmap")
            render_correlation_heatmap(corr_df, tuple(roles))
            st.subheader("Table")
            role_cols = [f"corr_{r}" for r in roles if f"corr_{r}" in corr_df.columns]
            disp = corr_df[["query","best_match"] + role_cols].copy()
            st.dataframe(disp, use_container_width=True, height=520)
            st.download_button("⬇️ Download correlations (CSV)",
                               disp.to_csv(index=False).encode("utf-8"),
                               file_name=f"correlations_{dataset_choice.lower()}_{scope_label.replace(' ','_')}.csv",
                               mime="text/csv")
            st.markdown("<span class='badge-ok'>Live ✓</span>", unsafe_allow_html=True)
        else:
            st.info("No correlations could be computed. Try fewer roles/keywords, or a different timeframe/region.")

# ───────────────────── Main ─────────────────────
def main():
    ctrl = sidebar()
    if ctrl["view"] == "Trends Studio":
        trends_studio(ctrl)
    else:
        job_market(ctrl)
    st.caption("© 2025 · Trends Hub · Live-first • Streamlit cache • Retry buttons • Correlations")

if __name__ == "__main__":
    main()