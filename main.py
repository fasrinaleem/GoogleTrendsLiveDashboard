# main.py
import streamlit as st
from pathlib import Path

# ---------- Page Config ----------
st.set_page_config(page_title="Google Trends Hub", page_icon="📊", layout="wide")

# ---------- Styles (white theme + animations) ----------
def apply_google_style():
    st.markdown("""
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700;900&display=swap" rel="stylesheet">
    <style>
      :root {
        --accent: #1A73E8;
        --bg: #ffffff;
        --text: #111827;
        --muted: #6b7280;
        --card: #ffffff;
        --ring: rgba(26,115,232,0.18);
      }
      html, body, [class*="css"] { font-family: 'Roboto', sans-serif; background: var(--bg); color: var(--text); }
      /* Buttons */
      .stButton > button, .stDownloadButton > button, .stLinkButton > button {
        background: var(--accent); color: #fff; border-radius: 10px;
        padding: 10px 18px; border: 0; font-weight: 600; box-shadow: 0 4px 12px var(--ring);
        transition: transform .08s ease, box-shadow .2s ease, background .2s ease;
      }
      .stButton > button:hover, .stLinkButton > button:hover { background: #1669C1; transform: translateY(-1px); box-shadow: 0 8px 20px var(--ring); }
      /* Hero */
      .hero {
        border-radius: 18px;
        padding: 28px 28px 22px 28px;
        background: linear-gradient(120deg, #E8F1FF 0%, #F5FBFF 55%, #FFFFFF 100%);
        position: relative; overflow: hidden; border: 1px solid #eef2ff;
      }
      .orb { position:absolute; width:220px; height:220px; border-radius:50%;
             background: radial-gradient(closest-corner, rgba(26,115,232,.12), rgba(26,115,232,0));
             animation: float 8s ease-in-out infinite; filter: blur(1px); }
      .orb.o1 { top:-60px; right:-40px; animation-delay: .0s; }
      .orb.o2 { bottom:-80px; left:-40px; animation-delay: 1.2s; }
      @keyframes float {
        0%,100% { transform: translateY(0) translateX(0) scale(1); }
        50%     { transform: translateY(-8px) translateX(6px) scale(1.03); }
      }
      .hero h1 { margin: 0 0 6px 0; font-weight: 900; letter-spacing:.2px; }
      .pill {
        display:inline-flex; gap:8px; align-items:center; padding:6px 10px;
        background:#ecf3ff; color:#1A73E8; border-radius:999px; font-weight:600; font-size:12px;
      }
      .muted { color: var(--muted); }
      /* Feature cards */
      .grid { display:grid; gap:16px; grid-template-columns: repeat(auto-fit, minmax(290px, 1fr)); }
      .card {
        background: var(--card); border: 1px solid #eef2ff; border-radius: 14px; padding: 18px;
        transition: transform .08s ease, box-shadow .2s ease;
        box-shadow: 0 0 0 rgba(0,0,0,0);
      }
      .card:hover { transform: translateY(-2px); box-shadow: 0 10px 24px rgba(17,24,39,.06); }
      .card h3 { margin-top: 4px; margin-bottom: 8px; }
      .card ul { margin: 0 0 4px 20px; }
      .kicker { font-size:13px; font-weight:700; letter-spacing:.25px; color:#2563eb; text-transform:uppercase; }
      .sep { margin: 18px 0; height:1px; background: #f3f4f6; }
    </style>
    """, unsafe_allow_html=True)

apply_google_style()

# ---------- Helper to resolve page paths ----------
def resolve_page(*candidates: str) -> str | None:
    for c in candidates:
        if Path(c).is_file():
            return c
    return None

# Try both “numbered” and “plain” filenames
TRENDS_PAGE = resolve_page("pages/1_Trends_Studio.py", "pages/Trends_Studio.py", "pages/Trends_Studio.py")
SKILLS_PAGE  = resolve_page("pages/2_Skills.py", "pages/Skills.py", "pages/skills.py")
JOB_PAGE     = resolve_page("pages/3_Job_Market.py", "pages/Job_Market.py", "pages/job_market.py")

# ---------- Hero ----------
st.markdown(
    """
    <div class="hero">
      <div class="orb o1"></div>
      <div class="orb o2"></div>
      <div class="pill">📊 Google Trends Hub</div>
      <h1>Discover trends, skills & job insights — fast.</h1>
      <p class="muted">
        A lightweight suite built on Google Trends (via PyTrends) with smart, realistic fallbacks.
        Explore trending keywords and topics, visualize interest over time & region, generate skill insights,
        and jump straight to job or course searches.
      </p>
    </div>
    """, unsafe_allow_html=True
)

st.markdown("<div class='sep'></div>", unsafe_allow_html=True)

# ---------- Quick Start ----------
col_l, col_r = st.columns([2,1])
with col_l:
    st.markdown("#### 🚀 Quick Start")
    st.write(
        "- Open a page below.\n"
        "- Use the button in each page (or section) to fetch live trends.\n"
        "- If an API call is slow/restricted, we auto-fill **realistic demo data** so the flow never breaks."
    )
with col_r:
    st.markdown("#### 🔧 Tech")
    st.write("Streamlit • PyTrends • Plotly • WordCloud (optional)")

st.markdown("<div class='sep'></div>", unsafe_allow_html=True)

# ---------- Page Cards ----------
st.markdown("### Pages & Features")

# Cards grid
st.markdown("<div class='grid'>", unsafe_allow_html=True)

# Card 1 — Trends Studio
with st.container():
    st.markdown(
        """
        <div class="card">
          <div class="kicker">Trends Studio</div>
          <h3>📈 Explore Live Trends</h3>
          <ul>
            <li>One-click: trending keywords (Top 10), job-market topics, global news & salary summary</li>
            <li>Role & Geo selectors (no sidebar clutter)</li>
            <li>Auto dummy data if API is slow (>5s)</li>
          </ul>
        </div>
        """, unsafe_allow_html=True
    )
    if TRENDS_PAGE:
        st.page_link(TRENDS_PAGE, label="Open Trends Studio", icon="📈", use_container_width=True)
    else:
        st.info("Trends Studio page not found in /pages.")

# Card 2 — Skills
with st.container():
    st.markdown(
        """
        <div class="card">
          <div class="kicker">Skills</div>
          <h3>🧠 Skill Popularity & Roles</h3>
          <ul>
            <li>Interest over time & by region + animated world map</li>
            <li>Related queries word cloud</li>
            <li>Job Role matcher with transparent Match Score</li>
          </ul>
        </div>
        """, unsafe_allow_html=True
    )
    if SKILLS_PAGE:
        st.page_link(SKILLS_PAGE, label="Open Skills", icon="🧠", use_container_width=True)
    else:
        st.info("Skills page not found in /pages.")

# Card 3 — Job Market
with st.container():
    st.markdown(
        """
        <div class="card">
          <div class="kicker">Job Market</div>
          <h3>💼 Roles, Trends & Openings</h3>
          <ul>
            <li>Role interest time-series and worldwide popularity</li>
            <li>Top & Rising related keywords + word cloud</li>
            <li>Salary ranges (AUD), course finder & job-board quick links</li>
          </ul>
        </div>
        """, unsafe_allow_html=True
    )
    if JOB_PAGE:
        st.page_link(JOB_PAGE, label="Open Job Market", icon="💼", use_container_width=True)
    else:
        st.info("Job Market page not found in /pages.")

st.markdown("</div>", unsafe_allow_html=True)  # end grid

st.markdown("<div class='sep'></div>", unsafe_allow_html=True)

# ---------- Footer ----------
st.caption("Tip: Use the **sidebar → Pages** menu to jump between tools. All results persist while you explore.")