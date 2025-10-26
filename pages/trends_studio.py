# views/trends_studio.py
# Trends Studio — single-button generator; no sidebar controls; results persist.

import time
import random
import pandas as pd
import streamlit as st
from pytrends.request import TrendReq

# ----------------------
# Page config
# ----------------------
st.set_page_config(page_title="Trends Studio", page_icon="📈", layout="wide")
st.title("Trends Studio")
st.caption("All controls live at the top. One button builds every section. Results persist.")

# Make buttons Google-blue
st.markdown(
    """
    <style>
      .stButton > button, .stLinkButton > button, .stDownloadButton > button {
        background: #2F56F7; color: #fff; border: 0; border-radius: 10px;
        padding: 10px 18px; font-weight: 600;
      }
      .stButton > button:hover { background: #2646cc; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------
# Session defaults (persistence)
# ----------------------
st.session_state.setdefault("ts_daily_df", pd.DataFrame())         # Trending Keywords (Top 10)
st.session_state.setdefault("ts_topics_list", [])                   # Trending Topics (Job Market)
st.session_state.setdefault("ts_world_news_df", pd.DataFrame())     # World Trending News (Job Market)
st.session_state.setdefault("ts_salary_df", pd.DataFrame())         # Highest Paying Jobs Summary
st.session_state.setdefault("ts_role", "Data Analyst")              # default role
st.session_state.setdefault("ts_geo", "AU")                         # default geo

# ----------------------
# Helpers
# ----------------------
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
    "Mozilla/5.0 (X11; Linux x86_64)",
]

JOB_ROLES = [
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

# Map ISO-2 → pytrends 'pn' parameter names where available
PN_MAP = {
    "AU": "australia",
    "US": "united_states",
    "GB": "united_kingdom",
    "CA": "canada",
    "IN": "india",
    "NZ": "new_zealand",
    "DE": "germany",
    "FR": "france",
    "JP": "japan",
}

def trend_client():
    # Keep request time <5s using connect/read timeouts; rotate UA
    return TrendReq(
        hl="en-US",
        tz=600,
        retries=1,
        backoff_factor=0.2,
        requests_args={
            "headers": {"User-Agent": random.choice(USER_AGENTS)},
            "timeout": (2.5, 2.0),  # (connect, read) — aim <5s total
        },
    )

# ----------------------
# Dummy data builders (role-aware)
# ----------------------
def _dummy_trending_keywords(role: str) -> pd.DataFrame:
    bank = {
        "Data Analyst": ["power bi", "sql queries", "tableau", "etl pipeline", "excel shortcuts",
                         "kpi dashboard", "python pandas", "data cleaning", "looker studio", "a/b testing"],
        "Data Scientist": ["llm fine tuning", "rag systems", "mlops", "model drift", "vector db",
                           "feature store", "xgboost", "time series", "shap values", "prompt engineering"],
        "Software Engineer": ["next.js", "typescript", "kubernetes", "microservices", "system design",
                              "grpc", "clean architecture", "ci/cd", "terraform", "serverless"],
    }
    base = bank.get(role, bank["Software Engineer"])
    return pd.DataFrame({
        "Keyword": [k.title() for k in base],
        "Score": [round(random.uniform(60, 100), 1) for _ in base],
    })

def _dummy_topics_for_role(role: str):
    topic_bank = {
        "Data Analyst": ["AI-assisted BI", "Metrics Layer", "Self-serve Dashboards",
                         "Excel↔Python Workflows", "Data Quality Ops", "Storytelling with Data"],
        "Data Scientist": ["RAG Evaluation", "Responsible AI", "Realtime Inference",
                           "Feature Engineering 2.0", "GPU Cost Control", "Synthetic Data"],
        "Software Engineer": ["Edge Computing", "Observability", "API-first Design",
                              "Cloud Cost Optimization", "Contract Testing", "Platform Engineering"],
    }
    topics = topic_bank.get(role, topic_bank["Software Engineer"])
    random.shuffle(topics)
    return topics[:6]

def _dummy_world_news_for_role(role: str) -> pd.DataFrame:
    regions = ["US", "EU", "APAC", "UK", "India"]
    verbs = ["surges", "booms", "accelerates", "hiring uptick", "investment rises"]
    domains = {
        "Data Analyst": "analytics & BI",
        "Data Scientist": "AI & ML",
        "Software Engineer": "software & cloud",
    }
    domain = domains.get(role, "tech")
    rows = []
    for r in regions:
        v = random.choice(verbs)
        rows.append({
            "Region": r,
            "Headline": f"{r} {domain} demand {v} as firms double down on {role.lower()} initiatives",
        })
    return pd.DataFrame(rows)

def _dummy_salary(role: str) -> pd.DataFrame:
    table_aud = {
        "Data Analyst": {"US": 145000, "Australia": 115000, "UK": 100000, "Germany": 105000, "India": 45000},
        "Data Scientist": {"US": 210000, "Australia": 165000, "UK": 145000, "Germany": 150000, "India": 90000},
        "Software Engineer": {"US": 200000, "Australia": 155000, "UK": 135000, "Germany": 140000, "India": 80000},
        "Machine Learning Engineer": {"US": 230000, "Australia": 180000, "UK": 160000, "Germany": 160000, "India": 100000},
        "DevOps Engineer": {"US": 190000, "Australia": 150000, "UK": 130000, "Germany": 135000, "India": 75000},
        "Cloud Engineer": {"US": 205000, "Australia": 165000, "UK": 145000, "Germany": 150000, "India": 82000},
        "Cybersecurity Analyst": {"US": 185000, "Australia": 145000, "UK": 125000, "Germany": 130000, "India": 70000},
        "Business Analyst": {"US": 150000, "Australia": 125000, "UK": 110000, "Germany": 115000, "India": 60000},
        "Product Manager": {"US": 225000, "Australia": 175000, "UK": 155000, "Germany": 155000, "India": 110000},
        "UI UX Designer": {"US": 160000, "Australia": 130000, "UK": 115000, "Germany": 120000, "India": 65000},
        "Generative AI Engineer": {"US": 300000, "Australia": 230000, "UK": 200000, "Germany": 195000, "India": 140000},
    }
    rows = [{"Country": c, "Avg Salary (AUD)": v} for c, v in table_aud.get(role, table_aud["Software Engineer"]).items()]
    return pd.DataFrame(rows).sort_values("Avg Salary (AUD)", ascending=False)

# ----------------------
# Live fetcher (best-effort with quick timeouts)
# ----------------------
def fetch_trending_daily(geo_iso: str) -> pd.DataFrame:
    """Top daily trending searches for a country. Returns up to 20; we'll trim to 10."""
    pn = PN_MAP.get((geo_iso or "").upper(), None)
    if not pn:
        return pd.DataFrame()
    t0 = time.time()
    try:
        pt = trend_client()
        df = pt.trending_searches(pn=pn)  # one column with trending queries
        if isinstance(df, pd.DataFrame) and not df.empty:
            df.columns = ["Keyword"]
            df["Score"] = [round(100 - i*3 + random.uniform(-2, 2), 1) for i in range(len(df))]
            out = df.head(10).copy()
        else:
            out = pd.DataFrame()
    except Exception:
        out = pd.DataFrame()
    # if slow or empty, trigger dummy by returning empty
    if (time.time() - t0) > 5 or out.empty:
        return pd.DataFrame()
    return out

# ----------------------
# Top controls (inline)
# ----------------------
c1, c2 = st.columns([2, 1])
with c1:
    st.session_state["ts_role"] = st.selectbox(
        "Job Market Role (influences topics, news, salaries)",
        JOB_ROLES,
        index=JOB_ROLES.index(st.session_state.get("ts_role", "Data Analyst"))
    )
with c2:
    st.session_state["ts_geo"] = st.selectbox(
        "Geo (ISO-2) for Trending Keywords",
        ["AU", "US", "GB", "CA", "IN", "NZ", "DE", "FR", "JP"],
        index=["AU", "US", "GB", "CA", "IN", "NZ", "DE", "FR", "JP"].index(st.session_state.get("ts_geo", "AU"))
    )

st.divider()

# =========================
# Single action: Build all sections
# =========================
st.subheader("Build Everything")
build_all = st.button("Build All Sections", type="primary", key="btn_all")
if build_all:
    # 1) Trending Keywords (Top 10)
    live = fetch_trending_daily(st.session_state["ts_geo"])
    st.session_state["ts_daily_df"] = live if not live.empty else _dummy_trending_keywords(st.session_state["ts_role"])
    # 2) Trending Topics (Job Market)
    st.session_state["ts_topics_list"] = _dummy_topics_for_role(st.session_state["ts_role"])
    # 3) World Trending News (Job Market)
    st.session_state["ts_world_news_df"] = _dummy_world_news_for_role(st.session_state["ts_role"])
    # 4) Highest Paying Jobs Summary
    st.session_state["ts_salary_df"] = _dummy_salary(st.session_state["ts_role"])
    st.success("All sections generated.")

st.divider()

# =========================
# Trending Keywords (Top 10)
# =========================
st.subheader("Trending Keywords (Top 10)")
df_daily = st.session_state["ts_daily_df"]
if not df_daily.empty:
    st.dataframe(df_daily.reset_index(drop=True), use_container_width=True)
else:
    st.info("Click **Build All Sections** to generate.")

st.divider()

# =========================
# Trending Topics (Job Market)
# =========================
st.subheader("Trending Topics (Job Market)")
topics = st.session_state["ts_topics_list"]
if topics:
    st.markdown("\n".join([f"- {t}" for t in topics]))
else:
    st.info("Click **Build All Sections** to generate.")

st.divider()

# =========================
# World Trending News (Job Market)
# =========================
st.subheader("World Trending News (Job Market)")
wn = st.session_state["ts_world_news_df"]
if not wn.empty:
    cols = [c for c in wn.columns if c.lower() != "source"]  # hide source column if present
    st.dataframe(wn[cols], use_container_width=True)
else:
    st.info("Click **Build All Sections** to generate.")

st.divider()

# =========================
# Highest Paying Jobs — Summary (AUD)
# =========================
st.subheader("Highest Paying Jobs — Summary (AUD)")
sal = st.session_state["ts_salary_df"]
if not sal.empty:
    st.dataframe(sal, use_container_width=True)
else:
    st.info("Click **Build All Sections** to generate.")