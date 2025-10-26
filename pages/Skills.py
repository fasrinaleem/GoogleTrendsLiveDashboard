# pages/Skills.py
# Skill Trends + Interest by Region + Animated Worldwide Bubble Map + Word Cloud + Job Role Matcher
# Buttons live inside their own sections; results persist via session_state.

import time
import random
import datetime as dt
import pandas as pd
import streamlit as st
import plotly.express as px
from pytrends.request import TrendReq

# Optional word cloud (we handle nicely if missing)
try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except Exception:
    WORDCLOUD_AVAILABLE = False

st.set_page_config(page_title="Skill Popularity", page_icon="📈", layout="wide")
st.title("📊 Skill Popularity Tracker")

# -------------------- Inputs --------------------
skill = st.text_input("Enter a skill name:", value="Python")
geo = st.text_input("Enter region code (e.g., AU, US or leave blank for global):", value="AU").upper().strip()

st.divider()

# -------------------- Session defaults (persistence) --------------------
st.session_state.setdefault("iot_df", pd.DataFrame())
st.session_state.setdefault("iot_plot_cols", [])
st.session_state.setdefault("ibr_df", pd.DataFrame())
st.session_state.setdefault("ibr_value_col", None)
st.session_state.setdefault("ibr_geo", "")
st.session_state.setdefault("ibr_ts_df", pd.DataFrame())   # animated map
st.session_state.setdefault("roles_df", pd.DataFrame())    # job matcher
st.session_state.setdefault("wc_img", None)                # word cloud image
st.session_state.setdefault("wc_freq", {})                 # word cloud freq map

# -------------------- PyTrends Client --------------------
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
    "Mozilla/5.0 (X11; Linux x86_64)",
]

def trend_client():
    return TrendReq(
        hl="en-US",
        tz=360,
        retries=2,
        backoff_factor=0.3,
        requests_args={"headers": {"User-Agent": random.choice(USER_AGENTS)}},
    )

def get_iot(keyword: str, region: str, timeframe: str = "today 12-m") -> pd.DataFrame:
    pt = trend_client()
    try:
        pt.build_payload([keyword], geo=region or "", timeframe=timeframe)
        df = pt.interest_over_time()
        if df is None or df.empty:
            raise ValueError
    except Exception:
        pt.build_payload([keyword], geo="", timeframe=timeframe)
        df = pt.interest_over_time()
    if isinstance(df, pd.DataFrame):
        df = df.drop(columns=["isPartial"], errors="ignore")
    return df if isinstance(df, pd.DataFrame) else pd.DataFrame()

def get_ibr(keyword: str, region: str, timeframe: str = "today 12-m") -> pd.DataFrame:
    pt = trend_client()
    try:
        pt.build_payload([keyword], geo=region or "", timeframe=timeframe)
        ibr = pt.interest_by_region(resolution="region", inc_low_vol=True)
        if ibr is None or ibr.empty:
            raise ValueError
    except Exception:
        pt.build_payload([keyword], geo="", timeframe=timeframe)
        ibr = pt.interest_by_region(resolution="region", inc_low_vol=True)
    return ibr if isinstance(ibr, pd.DataFrame) else pd.DataFrame()

# -------- Worldwide monthly time-series for animated map --------
def month_windows(n_months: int = 6):
    end = dt.date.today().replace(day=1)
    for i in range(n_months, 0, -1):
        start = (end - pd.DateOffset(months=i)).date()
        stop  = (end - pd.DateOffset(months=i-1)).date()
        yield start.strftime("%Y-%m"), start.isoformat(), stop.isoformat()

def fetch_worldwide_ibr_timeseries(keyword: str, n_months: int = 6) -> pd.DataFrame:
    rows = []
    pt = trend_client()
    for label, start, stop in month_windows(n_months):
        timeframe = f"{start} {stop}"
        try:
            pt.build_payload([keyword], geo="", timeframe=timeframe)  # force WORLDWIDE
            ibr = pt.interest_by_region(resolution="country", inc_low_vol=True)
        except Exception:
            ibr = pd.DataFrame()
        time.sleep(0.4)  # be polite
        if isinstance(ibr, pd.DataFrame) and not ibr.empty:
            ibr = ibr.reset_index()
            country_col = ibr.columns[0]
            val_col = keyword if keyword in ibr.columns else ibr.columns[-1]
            for _, r in ibr.iterrows():
                rows.append({"period": label, "location": r[country_col], "value": float(r[val_col] or 0)})
    return pd.DataFrame(rows)

# ===================================================================
# Interest Over Time
# ===================================================================
st.subheader("Interest Over Time (last 12 months)")
if st.button("Fetch Interest Over Time", key="btn_iot"):
    if not skill.strip():
        st.warning("Enter a skill first ⚠")
    else:
        df = get_iot(skill, geo)
        if df.empty:
            st.error("No trend data. Try worldwide.")
        else:
            cols = [c for c in df.columns if skill.lower() in c.lower()]
            st.session_state["iot_df"] = df
            st.session_state["iot_plot_cols"] = cols if cols else df.columns.tolist()

if not st.session_state["iot_df"].empty:
    df = st.session_state["iot_df"]
    st.line_chart(df[st.session_state["iot_plot_cols"]])
    with st.expander("Show raw time-series data"):
        st.dataframe(df)

st.divider()

# ===================================================================
# Interest by Region
# ===================================================================
st.subheader("Interest by Region")
if st.button("Fetch Interest by Region", key="btn_ibr"):
    if not skill.strip():
        st.warning("Enter a skill first ⚠")
    else:
        ibr = get_ibr(skill, geo)
        if ibr.empty:
            st.error("No regional data.")
        else:
            col = skill if skill in ibr.columns else ibr.columns[-1]
            st.session_state["ibr_df"] = ibr
            st.session_state["ibr_value_col"] = col
            st.session_state["ibr_geo"] = geo

if not st.session_state["ibr_df"].empty:
    ibr = st.session_state["ibr_df"]
    col = st.session_state["ibr_value_col"]
    st.plotly_chart(px.bar(ibr.sort_values(by=col, ascending=False), y=col), use_container_width=True)

st.divider()

# ===================================================================
# Worldwide Animated Bubble Map
# ===================================================================
st.subheader("Worldwide Animated Bubble Map by Month")
if st.button("Build Animated Bubble Map", key="btn_animap"):
    if not skill.strip():
        st.warning("Enter a skill first ⚠")
    else:
        ts = fetch_worldwide_ibr_timeseries(skill, 6)
        if ts.empty:
            st.error("Could not generate map.")
        else:
            st.session_state["ibr_ts_df"] = ts

ts_df = st.session_state["ibr_ts_df"]
if not ts_df.empty:
    vmax = max(1, ts_df["value"].max())
    ts_df["_size"] = ts_df["value"] / vmax * 30 + 4
    fig = px.scatter_geo(
        ts_df,
        locations="location",
        locationmode="country names",
        animation_frame="period",
        color="value",
        size="_size",
        projection="natural earth",
        color_continuous_scale="Turbo",
        title=f"Worldwide Popularity of {skill}",
    )
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ===================================================================
# Word Cloud — Related Queries (Top + Rising)
# ===================================================================
st.subheader("Word Cloud — Related Queries")

def fetch_related_queries(keyword: str, region: str, timeframe: str = "today 12-m"):
    """Lightweight fetch of related queries with fallback to worldwide."""
    pt = trend_client()
    try:
        pt.build_payload([keyword], geo=region or "", timeframe=timeframe)
        rq = pt.related_queries()
        if not rq:
            raise ValueError
    except Exception:
        pt.build_payload([keyword], geo="", timeframe=timeframe)
        rq = pt.related_queries()
    return rq or {}

def build_wc_frequencies(related_dict: dict) -> dict:
    """Combine Top (weight=1.0) + Rising (weight=0.8) into a frequency map."""
    freq = {}
    if not isinstance(related_dict, dict):
        return freq
    for kw, parts in related_dict.items():
        if not isinstance(parts, dict):
            continue
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
    # normalize a bit for nicer shapes
    if freq:
        m = max(freq.values())
        if m > 0:
            for k in list(freq.keys()):
                freq[k] = freq[k] / m * 100.0
    return freq

def render_wordcloud(freq_map: dict):
    if not WORDCLOUD_AVAILABLE:
        st.error("Package `wordcloud` not installed. Install with: `pip install wordcloud`")
        return None
    if not freq_map:
        st.warning("No related queries available yet. Try 'Fetch Interest by Region' first or another timeframe.")
        return None
    wc = WordCloud(width=1400, height=500, background_color="white", colormap="viridis")
    img = wc.generate_from_frequencies(freq_map).to_image()
    return img

# Button under section (persists results)
if st.button("Generate Word Cloud", key="btn_wc"):
    if not skill.strip():
        st.warning("Enter a skill first ⚠")
    else:
        with st.spinner("Fetching related queries…"):
            rq = fetch_related_queries(skill, geo)
            freq = build_wc_frequencies(rq)
            img = render_wordcloud(freq)
        st.session_state["wc_freq"] = freq
        st.session_state["wc_img"] = img

# Always render persisted cloud if exists
if st.session_state["wc_img"] is not None:
    st.image(st.session_state["wc_img"], use_column_width=True, caption="Related Searches — Word Cloud")

st.divider()

# ===================================================================
# Job Role Matcher
# ===================================================================
st.subheader("Suggested Job Roles (based on your skill)")

ROLE_KNOWLEDGE = [
    {"role": "Data Analyst",
     "keywords": ["excel", "sql", "python", "tableau", "power bi", "statistics"],
     "paths": ["Data Scientist", "Business Analyst"]},
    {"role": "Data Scientist",
     "keywords": ["python", "pandas", "numpy", "ml", "machine learning", "tensorflow", "pytorch", "statistics"],
     "paths": ["ML Engineer", "MLOps Engineer"]},
    {"role": "Backend Developer",
     "keywords": ["python", "django", "flask", "node", "java", "spring", "golang", "api"],
     "paths": ["Software Engineer", "Cloud Engineer"]},
    {"role": "Frontend Developer",
     "keywords": ["javascript", "react", "vue", "angular", "css", "html", "typescript"],
     "paths": ["Full-Stack Developer", "UI Engineer"]},
    {"role": "Cloud Engineer",
     "keywords": ["aws", "azure", "gcp", "kubernetes", "docker", "terraform", "devops"],
     "paths": ["SRE", "Platform Engineer"]},
    {"role": "Business Analyst",
     "keywords": ["excel", "power bi", "tableau", "sql", "requirements"],
     "paths": ["Product Analyst", "Product Manager"]},
    {"role": "Cyber Security Analyst",
     "keywords": ["security", "siem", "splunk", "network", "threat"],
     "paths": ["Security Engineer", "GRC Analyst"]},
    {"role": "UX/UI Designer",
     "keywords": ["figma", "adobe", "ux", "ui", "prototyping"],
     "paths": ["Product Designer", "Design Lead"]},
    {"role": "Mobile Developer",
     "keywords": ["swift", "kotlin", "react native", "flutter", "android", "ios"],
     "paths": ["Mobile Tech Lead", "Full-Stack Developer"]},
]

def score_roles(skill_text: str) -> pd.DataFrame:
    text = (skill_text or "").lower().strip()
    rows = []
    for item in ROLE_KNOWLEDGE:
        hits = [k for k in item["keywords"] if k in text]
        score = min(100, len(hits) * 20)  # simple weighting
        if hits:
            rows.append({
                "Role": item["role"],
                "Match Score": score,
                "Keywords Hit": ", ".join(hits),
                "Career Growth": ", ".join(item["paths"]),
            })
    return pd.DataFrame(rows).sort_values("Match Score", ascending=False)

if st.button("Find Matching Job Roles", key="btn_roles"):
    st.session_state["roles_df"] = score_roles(skill)

if not st.session_state["roles_df"].empty:
    st.dataframe(st.session_state["roles_df"], use_container_width=True)
    # 👇 NEW: short explanation under the table
    st.markdown(
        """
        **About Match Score:**  
        The score is a quick relevance signal based on how many of your typed skill keywords
        appear in each role’s core toolset. Each keyword match contributes **~20 points**
        (capped at **100**). It helps you compare which roles most closely align with your current
        skills. Try adding tools (e.g., *Python pandas SQL*) to refine the matches.
        """,
        help="Simple heuristic: score = 20 × (# of matched keywords), capped at 100."
    )

st.caption("Tip: Add tools (e.g., 'Python pandas SQL') for richer matches.")
st.caption("✅ All sections persist; data fetched live via PyTrends. Word cloud built from Related Queries (Top + Rising).")