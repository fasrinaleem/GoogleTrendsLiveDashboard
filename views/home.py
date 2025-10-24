# views/home.py
import streamlit as st

def render():
    st.markdown("""
    <div class="hero"><h1>Trends Hub – Real-Time Google Trends Dashboard</h1>
    <div class="subtle">Interactive • Live data • Exportable insights</div></div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="section">
      <div class="section-h"><h2>Project Overview</h2></div>
      <p><strong>Case:</strong> Real-time interest signals for job roles & skills.</p>
      <p><strong>Goal:</strong> Show dynamic job-related trends, regions, and related queries.</p>
      <p><strong>Dataset:</strong> Google Trends (public, live, global).</p>
      <p><strong>Analyses:</strong> Time-series, regional heatmaps, word clouds, keyword↔role correlations.</p>
      <p><strong>Tools:</strong> Streamlit • Plotly • PyTrends • WordCloud.</p>
    </div>
    """, unsafe_allow_html=True)

    st.info("Use the left sidebar to explore ✨ Trends Studio or 💼 Job Market dashboards.")
