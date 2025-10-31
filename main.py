# ===============================================================
# 📊 Google Trends Live Dashboard – Main Landing Page
# ===============================================================

import streamlit as st
from PIL import Image
# ---------------------------------------------------------------
# Page Configuration (Title, Icon, Layout)
# ---------------------------------------------------------------
st.set_page_config(
    page_title="Google Trends Live Dashboard | Job & Skill Insights",
    page_icon="🌐",
    layout="wide"
)

# ---------------------------------------------------------------
# Apply Global CSS
# ---------------------------------------------------------------
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Hide the default Streamlit sidebar on this main page
hide_sidebar = """
    <style>
        section[data-testid="stSidebar"] {display: none;}
        div[data-testid="stAppViewContainer"] {
            margin-left: 0;
            padding-left: 2rem;
            padding-right: 2rem;
        }
    </style>
"""
st.markdown(hide_sidebar, unsafe_allow_html=True)

# ---------------------------------------------------------------
# Header Section
# ---------------------------------------------------------------
st.title("Google Trends Live Dashboard")
st.markdown("#### Discover Real-Time Job, Skill & Market Insights")

st.markdown("---")

# ---------------------------------------------------------------
# 🎓 Group Project Summary (As required by Assessment)
# ---------------------------------------------------------------
st.subheader("Project Introduction")

st.write("""
The **Google Trends Live Dashboard** is an interactive data visualisation system designed 
to analyse global search trends related to **job roles, skills, and career markets**.  
It enables users to explore real-time interest patterns, compare roles, 
and identify emerging opportunities across industries and regions.
""")

# ---------------------------------------------------------------
# Problem Definition, Goal, Narrative, Audience
# ---------------------------------------------------------------
col1, col2 = st.columns(2)
with col1:
    st.markdown("### Problem Definition")
    st.write("""
Traditional job market reports are static and quickly outdated. 
There is a growing need for a **live, data-driven system** that captures the 
**real-time evolution of skills and job demand** to guide students, job seekers, 
and educators in their decision-making.
""")

with col2:
    st.markdown("### Goal of the Dashboard")
    st.write("""
To provide an interactive, easy-to-use platform that visualises 
Google search trends related to **jobs and skills**, helping users 
understand what roles and technologies are in demand worldwide.
""")

st.markdown("---")

# ---------------------------------------------------------------
# Narrative & Target Audience
# ---------------------------------------------------------------
st.markdown("### Narrative & Target Audience")
st.write("""
The dashboard tells the **story of a changing digital job landscape**.  
Users can interactively explore how interest in various careers and technical skills 
shifts over time, by region, and across countries.  

**Target Audience:**  
Students, job seekers, data analysts, career advisors, 
policy makers, and educators seeking labour market intelligence.
""")

st.markdown("---")

# ---------------------------------------------------------------
# Dataset Overview & Analysis Summary
# ---------------------------------------------------------------
st.subheader("Dataset Overview & Analysis Summary")

st.write("""
**Data Source:** [Google Trends API (PyTrends)](https://pypi.org/project/pytrends/)  
Data is collected in real-time from Google Search Index values, 
representing the relative popularity of search terms (0–100 scale).  

**Analysis Performed:**
- **Exploratory Data Analysis (EDA):** Examined patterns in job and skill trends.  
- **Time Series Analysis:** Visualised changes in search interest over months.  
- **Regional & Global Mapping:** Identified geographic hotspots of demand.  
- **Keyword Discovery:** Extracted top & rising related queries for each term.  
- **Hypothesis:**  
  “Search popularity for technology-related jobs and skills correlates 
  with real-world hiring trends and salary growth.”
""")

st.markdown("---")

# ---------------------------------------------------------------
# 🔍 Navigation Buttons (Interactive Entry Points)
# ---------------------------------------------------------------
st.subheader("Explore the Dashboard")

st.write("Choose a section to begin your exploration:")

colA, colB, colC = st.columns(3)


with colA:
    if st.button("🧠 Skills Analytics"):
        st.switch_page("pages/1_Skills.py")

with colB:
    if st.button("💼 Job Market Insights"):
        st.switch_page("pages/2_Job_Market.py")

with colC:
    if st.button("🎯 Trends Studio"):
        st.switch_page("pages/3_Trends_Studio.py")

st.markdown("---")

# ---------------------------------------------------------------
# Optional Footer Section
# ---------------------------------------------------------------
st.caption("Developed by Group Members – Google Trends Live Dashboard (2025)")
st.caption("Powered by Streamlit • Data Source: Google Trends API (PyTrends)")