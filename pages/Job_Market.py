# pages/Job_Market.py
import time
import urllib.parse
import datetime as dt
import pandas as pd
import streamlit as st
import plotly.express as px
from pytrends.request import TrendReq


# Optional word cloud
try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except Exception:
    WORDCLOUD_AVAILABLE = False

# -----------------------
# Page config
# -----------------------
st.set_page_config(page_title="Job Market", page_icon="💼", layout="wide")
st.title("💼 Job Market")
st.caption("Each section has its own controls. Results persist after you click other buttons.")

# -----------------------
# Helpers
# -----------------------
DEFAULT_ROLES = [
    "Software Engineer",
    "Data Analyst",
    "Data Scientist",
    "Machine Learning Engineer",
    "DevOps Engineer",
    "Cloud Engineer",
    "Cybersecurity Analyst",
    "Business Analyst",
    "Product Manager",
    "UI UX Designer",
    "Generative AI Engineer",
]

@st.cache_data(show_spinner=False)
def _build_pytrends(keywords, geo, timeframe, tz=480):
    pt = TrendReq(hl="en-US", tz=tz)
    pt.build_payload(
        kw_list=keywords,
        geo=(geo or "").strip().upper(),
        timeframe=timeframe,
    )
    return pt

@st.cache_data(show_spinner=False)
def get_interest_over_time(keywords, geo, timeframe):
    pt = _build_pytrends(keywords, geo, timeframe)
    df = pt.interest_over_time()
    if df is not None and not df.empty and "isPartial" in df.columns:
        df = df.drop(columns=["isPartial"])
    return df

def clean_keywords(raw: str):
    kws = [k.strip() for k in (raw or "").split(",") if k.strip()]
    if len(kws) > 5:
        st.warning(f"Using only first 5 keywords: {', '.join(kws[:5])}")
        kws = kws[:5]
    return kws

def url_encode(s: str) -> str:
    return urllib.parse.quote_plus(s)

def job_links(job_title: str, region_label: str, geo_code: str):
    li_base = "https://www.linkedin.com/jobs/search/"
    indeed_base = "https://www.indeed.com/jobs"
    seek_base = "https://www.seek.com.au/jobs"

    if geo_code == "AU":
        indeed_base = "https://au.indeed.com/jobs"
        seek_base = "https://www.seek.com.au/jobs"
    elif geo_code == "NZ":
        indeed_base = "https://nz.indeed.com/jobs"
        seek_base = "https://www.seek.co.nz/jobs"
    elif geo_code == "GB":
        indeed_base = "https://uk.indeed.com/jobs"
    elif geo_code == "IN":
        indeed_base = "https://in.indeed.com/jobs"
    elif geo_code == "CA":
        indeed_base = "https://ca.indeed.com/jobs"

    q = url_encode(job_title.strip())
    loc = url_encode(region_label.strip()) if region_label else ""

    linkedin_url = f"{li_base}?keywords={q}" + (f"&location={loc}" if loc else "")
    seek_url = f"{seek_base}?keywords={q}" + (f"&where={loc}" if loc else "")
    indeed_url = f"{indeed_base}?q={q}" + (f"&l={loc}" if loc else "")

    return linkedin_url, seek_url, indeed_url

# ---------- Related queries + Word Cloud helpers ----------
@st.cache_data(show_spinner=False)
def fetch_related_queries(keyword: str, geo: str, timeframe: str = "today 12-m"):
    try:
        pt = _build_pytrends([keyword], geo, timeframe)
        rq = pt.related_queries()
        if rq:
            return rq
    except Exception:
        pass
    try:
        pt = _build_pytrends([keyword], "", timeframe)
        rq = pt.related_queries()
        return rq or {}
    except Exception:
        return {}

def _freq_from_top_and_rising(related_dict: dict, anchor: str) -> dict:
    freq = {}
    if not isinstance(related_dict, dict):
        return freq
    parts = related_dict.get(anchor) if anchor in related_dict else (list(related_dict.values())[0] if related_dict else None)
    if not isinstance(parts, dict):
        return freq
    top = parts.get("top")
    if isinstance(top, pd.DataFrame) and not top.empty:
        for _, r in top.iterrows():
            q = str(r.get("query") or "").strip()
            v = float(r.get("value") or 0)
            if q:
                freq[q] = freq.get(q, 0) + v
    rising = parts.get("rising")
    if isinstance(rising, pd.DataFrame) and not rising.empty:
        for _, r in rising.iterrows():
            q = str(r.get("query") or "").strip()
            v = float(r.get("value") or 0)
            if q:
                freq[q] = freq.get(q, 0) + 0.8 * v
    if freq:
        m = max(freq.values())
        if m > 0:
            for k in list(freq.keys()):
                freq[k] = freq[k] / m * 100.0
    return freq

def build_wordcloud_image(freq_map: dict):
    if not WORDCLOUD_AVAILABLE:
        return None, "Package `wordcloud` not installed. Install with: pip install wordcloud"
    if not freq_map:
        return None, "No related queries yet."
    wc = WordCloud(width=1400, height=500, background_color="white", colormap="viridis")
    img = wc.generate_from_frequencies(freq_map).to_image()
    return img, None

# ---------- Worldwide animated map helpers ----------
def _month_windows(n_months: int = 6):
    end = dt.date.today().replace(day=1)
    for i in range(n_months, 0, -1):
        start = (end - pd.DateOffset(months=i)).date()
        stop  = (end - pd.DateOffset(months=i-1)).date()
        yield start.strftime("%Y-%m"), start.isoformat(), stop.isoformat()

@st.cache_data(show_spinner=False)
def fetch_worldwide_ibr_timeseries(keyword: str, n_months: int = 6) -> pd.DataFrame:
    rows = []
    try:
        pt = _build_pytrends([keyword], "", "today 12-m")
    except Exception:
        return pd.DataFrame()

    for label, start, stop in _month_windows(n_months):
        timeframe = f"{start} {stop}"
        try:
            pt.build_payload([keyword], geo="", timeframe=timeframe)
            ibr = pt.interest_by_region(resolution="country", inc_low_vol=True)
        except Exception:
            ibr = pd.DataFrame()
        time.sleep(0.35)
        if isinstance(ibr, pd.DataFrame) and not ibr.empty:
            ibr = ibr.reset_index()
            country_col = ibr.columns[0]
            val_col = keyword if keyword in ibr.columns else ibr.columns[-1]
            for _, r in ibr.iterrows():
                rows.append({"period": label, "location": r[country_col], "value": float(r[val_col] or 0)})
    return pd.DataFrame(rows)

# -----------------------
# Salary data (AUD) helpers
# -----------------------
SALARY_RANGES_AUD = {
    "Data Analyst":       {"Entry": 70000,  "Median": 95000,  "Senior": 120000, "Top 10%": 150000},
    "Data Scientist":     {"Entry": 95000,  "Median": 135000, "Senior": 165000, "Top 10%": 200000},
    "Software Engineer":  {"Entry": 90000,  "Median": 125000, "Senior": 160000, "Top 10%": 195000},
    "Machine Learning Engineer": {"Entry": 110000, "Median": 150000, "Senior": 185000, "Top 10%": 230000},
    "DevOps Engineer":    {"Entry": 100000, "Median": 135000, "Senior": 170000, "Top 10%": 205000},
    "Cloud Engineer":     {"Entry": 105000, "Median": 140000, "Senior": 175000, "Top 10%": 210000},
    "Cybersecurity Analyst": {"Entry": 95000, "Median": 125000, "Senior": 160000, "Top 10%": 195000},
    "Business Analyst":   {"Entry": 80000,  "Median": 110000, "Senior": 140000, "Top 10%": 170000},
    "Product Manager":    {"Entry": 120000, "Median": 155000, "Senior": 190000, "Top 10%": 230000},
    "UI UX Designer":     {"Entry": 85000,  "Median": 110000, "Senior": 140000, "Top 10%": 170000},
    "Generative AI Engineer": {"Entry": 140000, "Median": 190000, "Senior": 240000, "Top 10%": 300000},
}

def get_salary_range_aud(role: str) -> pd.DataFrame:
    row = SALARY_RANGES_AUD.get(role, SALARY_RANGES_AUD["Software Engineer"])
    df = pd.DataFrame(
        [{"Role": role, **row}],
        columns=["Role", "Entry", "Median", "Senior", "Top 10%"]
    )
    return df

# -----------------------
# Session-state (persistence)
# -----------------------
st.session_state.setdefault("jm_iot_df", pd.DataFrame())
st.session_state.setdefault("jm_wc_img", None)
st.session_state.setdefault("jm_wc_freq", {})
st.session_state.setdefault("jm_world_ts_df", pd.DataFrame())
st.session_state.setdefault("jm_related_top_df", pd.DataFrame())
st.session_state.setdefault("jm_related_rising_df", pd.DataFrame())
st.session_state.setdefault("jm_courses_list", [])
st.session_state.setdefault("jm_salary_df", pd.DataFrame())  # <-- NEW

# ========================
# SECTION 1: Interest Over Time  (own controls; role as dropdown)
# ========================
st.subheader("Interest Over Time")
st.write("Shows the relative Google search interest for the selected role over time.")
with st.container(border=True):
    c1, c2 = st.columns([2, 2])
    with c1:
        role_iot = st.selectbox("Job role / title", options=DEFAULT_ROLES, index=1, key="iot_role")
        synonyms_hint = st.text_input(
            "Optional: synonyms (comma-separated) to compare",
            placeholder="e.g., Business Intelligence Analyst, Reporting Analyst",
            key="iot_synonyms"
        )
    with c2:
        geo_iot = st.selectbox("Geo (ISO-2)", ["", "AU", "US", "GB", "CA", "IN", "DE", "NZ", "JP", "FR"], index=1, key="iot_geo")
        timeframe_iot = st.selectbox("Timeframe", ["now 7-d", "today 1-m", "today 3-m", "today 12-m", "today 5-y", "all"], index=2, key="iot_timeframe")

    keywords_iot = [role_iot] + (clean_keywords(synonyms_hint) if synonyms_hint else [])

    fetch_ts = st.button("Fetch Time Series", type="primary", key="btn_iot_fetch")

    if fetch_ts:
        with st.spinner("Fetching time series…"):
            iot = get_interest_over_time(keywords_iot, geo_iot, timeframe_iot)
        if iot is None or iot.empty:
            st.info("No time-series data. Try a broader timeframe or simpler role title.")
        else:
            st.session_state["jm_iot_df"] = iot

    if not st.session_state["jm_iot_df"].empty:
        iot = st.session_state["jm_iot_df"]
        fig = px.line(
            iot.reset_index(),
            x="date",
            y=[c for c in iot.columns if c != "isPartial"],
            labels={"value": "Interest", "date": "Date"},
            title=None
        )
        st.plotly_chart(fig, use_container_width=True)
        with st.expander("Show raw data"):
            st.dataframe(iot)
        st.download_button(
            "Download CSV",
            data=iot.to_csv().encode("utf-8"),
            file_name=f"interest_over_time_{role_iot.replace(' ','_')}.csv",
            mime="text/csv",
            key="dl_iot"
        )

st.divider()

# =========================
# SECTION 2: Salary Range — Annual (AUD)
# =========================
st.subheader("Salary Range — Annual (AUD)")
with st.container(border=True):
    s1, s2 = st.columns([2, 2])
    with s1:
        role_salary = st.selectbox("Role", options=DEFAULT_ROLES, index=1, key="salary_role")
    with s2:
        st.caption("Currency: AUD")

    btn_salary = st.button("Fetch Salary Range", key="btn_salary")
    if btn_salary:
        st.session_state["jm_salary_df"] = get_salary_range_aud(role_salary)

    if not st.session_state["jm_salary_df"].empty:
        df_sal = st.session_state["jm_salary_df"]
        st.dataframe(df_sal, use_container_width=True)
        # Nice bar for the ranges (exclude 'Role' column)
        sal_long = df_sal.melt(id_vars=["Role"], var_name="Band", value_name="AUD")
        fig_sal = px.bar(
            sal_long,
            x="Band",
            y="AUD",
            color="Band",
            text="AUD",
            title=f"Annual Salary Bands — {df_sal.iloc[0]['Role']} (AUD)"
        )
        fig_sal.update_traces(texttemplate="$%{text:,}", textposition="outside")
        fig_sal.update_layout(yaxis_title="AUD", showlegend=False)
        st.plotly_chart(fig_sal, use_container_width=True)

st.divider()

# =========================
# SECTION 3: Word Cloud — Related Searches
# =========================
st.subheader("Word Cloud — Related Searches")
with st.container(border=True):
    wc1, wc2, wc3 = st.columns([2, 1, 1])
    with wc1:
        role_wc = st.selectbox("Role for related queries", options=DEFAULT_ROLES, index=1, key="wc_role")
    with wc2:
        geo_wc = st.selectbox("Geo", ["", "AU", "US", "GB", "CA", "IN", "DE", "NZ", "JP", "FR"], index=1, key="wc_geo")
    with wc3:
        timeframe_wc = st.selectbox("Timeframe", ["today 12-m", "today 3-m", "now 7-d", "today 5-y", "all"], index=0, key="wc_timeframe")

    build_wc = st.button("Generate Word Cloud", key="btn_wc")

    if build_wc:
        with st.spinner("Fetching related queries…"):
            rq = fetch_related_queries(role_wc, geo_wc, timeframe_wc)
            freq = _freq_from_top_and_rising(rq, role_wc)
            img, err = build_wordcloud_image(freq)
        st.session_state["jm_wc_freq"] = freq
        st.session_state["jm_wc_img"] = img
        if err:
            st.info(err)

    if st.session_state["jm_wc_img"] is not None:
        st.image(st.session_state["jm_wc_img"], use_column_width=True, caption=f"Related searches for “{role_wc}”")

st.divider()

# =========================
# SECTION 4: Worldwide Popularity (Animated by Month)
# =========================
st.subheader("Worldwide Popularity (Animated by Month)")
with st.container(border=True):
    map_cols = st.columns([2, 1])
    with map_cols[0]:
        role_map = st.selectbox("Role to map worldwide", options=DEFAULT_ROLES, index=1, key="map_role")
    with map_cols[1]:
        months = st.slider("Animated months", min_value=3, max_value=12, value=6, step=1, key="map_months")

    build_map = st.button("Build Worldwide Map", key="btn_worldmap")

    if build_map:
        with st.spinner("Building worldwide monthly map…"):
            ts = fetch_worldwide_ibr_timeseries(role_map, months)
        st.session_state["jm_world_ts_df"] = ts

    ts_df = st.session_state["jm_world_ts_df"]
    if not ts_df.empty:
        vmax = max(1, ts_df["value"].max())
        ts_df = ts_df.copy()
        ts_df["_size"] = ts_df["value"] / vmax * 30 + 4
        fig = px.scatter_geo(
            ts_df,
            locations="location",
            locationmode="country names",
            animation_frame="period",
            color="value",
            size="_size",
            projection="natural earth",
            color_continuous_scale="Ice",
            title=f"Worldwide Popularity of {role_map}",
        )
        st.plotly_chart(fig, use_container_width=True)

st.divider()

# =========================
# SECTION 5: Top & Rising Keywords (Related to the Role)
# =========================
st.subheader("Top & Rising Keywords (Related to the Role)")
with st.container(border=True):
    tr1, tr2, tr3 = st.columns([2, 1, 1])
    with tr1:
        role_tr = st.selectbox("Role for related keywords", options=DEFAULT_ROLES, index=1, key="tr_role")
    with tr2:
        geo_tr = st.selectbox("Geo", ["", "AU", "US", "GB", "CA", "IN", "DE", "NZ", "JP", "FR"], index=1, key="tr_geo")
    with tr3:
        timeframe_tr = st.selectbox("Timeframe", ["today 12-m", "today 3-m", "now 7-d", "today 5-y", "all"], index=0, key="tr_timeframe")

    fetch_tr = st.button("Fetch Top & Rising", key="btn_top_rising")

    if fetch_tr:
        with st.spinner("Fetching Top & Rising related queries…"):
            rq = fetch_related_queries(role_tr, geo_tr, timeframe_tr)

        top_df = pd.DataFrame()
        rising_df = pd.DataFrame()
        parts = rq.get(role_tr) if isinstance(rq, dict) else None
        if isinstance(parts, dict):
            top_df = parts.get("top") if isinstance(parts.get("top"), pd.DataFrame) else pd.DataFrame()
            rising_df = parts.get("rising") if isinstance(parts.get("rising"), pd.DataFrame) else pd.DataFrame()

        st.session_state["jm_related_top_df"] = top_df
        st.session_state["jm_related_rising_df"] = rising_df

    if (not st.session_state["jm_related_top_df"].empty) or (not st.session_state["jm_related_rising_df"].empty):
        tab_top, tab_rising = st.tabs(["🏆 Top", "📈 Rising"])

        with tab_top:
            df = st.session_state["jm_related_top_df"]
            if df.empty:
                st.info("No Top queries available.")
            else:
                st.dataframe(df, use_container_width=True)
                st.download_button(
                    "Download Top CSV",
                    data=df.to_csv(index=False).encode("utf-8"),
                    file_name=f"related_top_{role_tr.replace(' ','_')}.csv",
                    mime="text/csv",
                    key="dl_top"
                )

        with tab_rising:
            df = st.session_state["jm_related_rising_df"]
            if df.empty:
                st.info("No Rising queries available.")
            else:
                st.dataframe(df, use_container_width=True)
                st.download_button(
                    "Download Rising CSV",
                    data=df.to_csv(index=False).encode("utf-8"),
                    file_name=f"related_rising_{role_tr.replace(' ','_')}.csv",
                    mime="text/csv",
                    key="dl_rising"
                )

st.divider()

# =========================
# SECTION 6: Course Finder — role dropdown (comes BEFORE Job Openings)
# =========================
st.subheader("Course Finder")
with st.container(border=True):
    cf1, cf2, cf3 = st.columns([2, 2, 2])
    with cf1:
        role_cf = st.selectbox("Role / topic", options=DEFAULT_ROLES, index=1, key="cf_role")
    with cf2:
        extra_kw = st.text_input("Extra keywords (optional)", placeholder="e.g., SQL, Tableau, Portfolio", key="cf_extra")
    with cf3:
        level_cf = st.selectbox("Level", ["All", "Beginner", "Intermediate", "Advanced"], index=0, key="cf_level")

    def _build_course_links(role: str, level: str, extra: str):
        base_query = role
        if level and level != "All":
            base_query += f" {level}"
        if extra.strip():
            base_query += f" {extra.strip()}"
        q = url_encode(base_query)

        return [
            {"provider": "Coursera", "url": f"https://www.coursera.org/search?query={q}"},
            {"provider": "edX", "url": f"https://www.edx.org/search?q={q}"},
            {"provider": "Udemy", "url": f"https://www.udemy.com/courses/search/?q={q}"},
            {"provider": "LinkedIn Learning", "url": f"https://www.linkedin.com/learning/search?keywords={q}"},
            {"provider": "YouTube", "url": f"https://www.youtube.com/results?search_query={q}+course"},
            {"provider": "Google (all providers)", "url": f"https://www.google.com/search?q={q}+course"},
        ]

    find_courses = st.button("Find Courses", key="btn_courses")

    if find_courses:
        st.session_state["jm_courses_list"] = _build_course_links(role_cf, level_cf, extra_kw)

    if st.session_state["jm_courses_list"]:
        links = st.session_state["jm_courses_list"]
        cols = st.columns(min(3, len(links)))
        for i, link in enumerate(links):
            with cols[i % len(cols)]:
                st.link_button(f"Open {link['provider']}", link["url"])
        with st.expander("Show course search links"):
            st.dataframe(pd.DataFrame(links))

st.divider()

# =========================
# SECTION 7 (LAST): Job Openings — Quick Links
# =========================
st.subheader("Job Openings — Quick Links")
with st.container(border=True):
    jl1, jl2, jl3 = st.columns([2, 2, 1])
    with jl1:
        job_title = st.selectbox("Job title", options=DEFAULT_ROLES, index=1, key="jl_title")
    with jl2:
        job_region = st.text_input("Location (optional)", placeholder="e.g., Perth WA, Australia", key="jl_loc")
    with jl3:
        geo_links = st.selectbox("Job board geo", ["", "AU", "US", "GB", "CA", "IN", "NZ"], index=1, key="jl_geo")

    li_url, seek_url, indeed_url = job_links(job_title, job_region, geo_links or "")

    b1, b2, b3 = st.columns(3)
    with b1:
        st.link_button("Open LinkedIn Jobs", li_url)
    with b2:
        st.link_button("Open SEEK", seek_url)
    with b3:
        st.link_button("Open Indeed", indeed_url)

st.caption("Tip: Refine the job title (e.g., “Junior Data Analyst”, “Graduate Software Engineer”) and add a city/state for more relevant results.")