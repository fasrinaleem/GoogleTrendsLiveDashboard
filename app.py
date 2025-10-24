# app.py
import streamlit as st
from views import home, trends_studio, job_market

st.set_page_config(page_title="Trends Hub", page_icon="📊", layout="wide")

def _load_css(path: str = "styles.css"):
    try:
        st.markdown(f"<style>{open(path,'r',encoding='utf-8').read()}</style>", unsafe_allow_html=True)
    except Exception:
        pass

_load_css()

PAGES = {
    "🏠 Home": home.render,
    "✨ Trends Studio": trends_studio.render,
    "💼 Job Market": job_market.render,
}

def main():
    with st.sidebar:
        st.header("🧭 Navigation")
        page = st.radio("Go to", list(PAGES.keys()), index=0)
        st.button("🔄 Clear caches", on_click=lambda: [st.cache_data.clear(), st.cache_resource.clear()])
    PAGES[page]()

if __name__ == "__main__":
    main()
