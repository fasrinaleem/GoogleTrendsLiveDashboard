# Trends Hub (PyTrends-only) — README

A modern Streamlit app for exploring Google Trends with **live, on-demand fetches**, **annotated time-series**, **animated regional maps**, and **word-cloud insights**—plus a **Job-Market** view that aggregates related skills across roles. It’s designed to load instantly using lightweight fallbacks, and only call the Google Trends endpoint when you explicitly click **Fetch Live**.

## Demo
https://drive.google.com/file/d/1WJGgwG27d0DRhM6I9-HHzX6uPMSEU0Jf/view?usp=sharing

---

## ✨ Highlights

- **PyTrends-only** (no SerpAPI or other providers).
- **Per-section “Fetch Live” buttons** → no surprise reruns.
- **Two views**:
  - **Trends Studio**: free-form keywords, interest over time, map, related queries.
  - **Job Market**: role-based trends, combined related skill cloud, openings links.
- **Color controls**: choose line palettes & map color scales.
- **Bigger word clouds** with adjustable colormap & max words.
- **Safe fallbacks** ensure the UI renders even if you hit API limits.
- **Throttling, caching, and retry logic** to reduce 429s.
- **Download buttons** for CSV exports (series & regions).
- **Animated visuals** using **Plotly** (maps, heatmaps, time-series).

---

## 🧱 Project Structure

```
.
├── app_gtrends.py     # Main Streamlit app (UI + sections + live calls + fallbacks)
├── trends.py          # Thin wrapper around PyTrends with retry + throttling
├── utils.py           # ISO helpers + spike detection utilities
└── viz.py             # Plotly charts + wordcloud renderer
```

---

## 🔧 Requirements

- **Python**: 3.9 – 3.12
- **Packages**:
  - `streamlit` → web framework
  - `pandas`, `numpy` → data handling
  - `plotly` → charts + **animations**
  - `pytrends` → Google Trends client
  - `wordcloud` → word cloud visualizations
  - `tenacity` → retry logic
  - `pycountry` → country code lookups

### Install

```bash
# recommend using a venv
python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate

pip install -U pip
pip install streamlit pandas numpy plotly pytrends wordcloud tenacity pycountry
```

---

## ▶️ Run the App

```bash
streamlit run app_gtrends.py
```

Open the URL printed in your terminal (usually `http://localhost:8501/`).

---

## 🖥️ How the App Works

1. The app renders instantly using **fallback demo data** (synthetic but realistic).
2. You choose options **in the sidebar**: view, region, timeframe, colors, wordcloud settings.
3. In each section you can click **“Fetch Live”** to retrieve fresh data from Google Trends via **PyTrends**.
4. Responses are **cached** briefly (90s) to avoid duplicate calls during quick interactions.
5. **Slow mode** spacing reduces the risk of HTTP 429 (rate limiting).
6. **Plotly animations** are used for:
   - **Choropleth maps** (`px.choropleth` with `animation_frame`)
   - **Heatmaps** (`px.imshow` with color scales)
   - **Time-series** line charts with spikes.

---

## 🌏 Regions & Geo Resolution

- The function `resolve_geo()` normalizes your input:
  - `"Worldwide"` → `geo=""`
  - `"Australia"` → `geo="AU"`
  - `"Perth"` → returns `(geo="AU", city_filter="Perth")`
  - `"Custom"` → country name or ISO-2 (e.g., `US`, `India`, `DE`)
- **Animated maps** use:
  - `COUNTRY` resolution by default
  - `CITY` resolution if a `city_filter` is present (when using “Perth”)

> Note: City resolution depends on PyTrends/Google Trends availability and can be sparse.

---

## 🧩 Views & Sections

### 1) Trends Studio

- **Top Trending Searches Today**  
  - Daily + Realtime trending (with optional live fetch)
- **Interest Over Time (annotated)**  
  - IoT series with KPI cards + spike markers  
- **Animated Map — Interest by Region**  
  - **Plotly animated choropleth** with fallback + live fetch
- **Related Queries — Word Cloud**  
  - Word cloud + tables (Top & Rising keywords)

### 2) Job Market

- **Overview** → IoT for selected roles + related queries cloud
- **Trends by Date** → role time-series
- **Interest by Region (animated)** → regional animated map per role
- **Top & Rising (with correlations)** →  
  - Word cloud of related keywords  
  - Correlation **heatmap** (Plotly) between related keywords and selected roles  
  - Correlation table with CSV download
- **Job Openings** → Quick links to LinkedIn, Seek, Indeed

---

## 🧮 Visualization Layer

- **Plotly** → line charts, animated maps, heatmaps
- **Wordcloud** → keyword clouds
- **Custom HTML/CSS** → KPI cards

---

## 🚀 Deployment

### Streamlit Cloud
Push repo → deploy → set Python version + requirements.

### Docker
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir streamlit pandas numpy plotly pytrends wordcloud tenacity pycountry
EXPOSE 8501
CMD ["streamlit", "run", "app_gtrends.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

Build & run:
```bash
docker build -t trends-hub .
docker run -p 8501:8501 trends-hub
```

---

## 📜 License

MIT License (example)  
© 2025 · Trends Hub

---

## 🙌 Credits

- Built with **Streamlit** + **Plotly**
- Data via **PyTrends**
- Country codes via **pycountry**
- Word clouds via **wordcloud**

---

## 🧭 Quick Start Cheatsheet

1. `streamlit run app_gtrends.py`
2. Pick **Trends Studio** or **Job Market** in sidebar
3. Adjust **Region**, **Timeframe**, **Colors**
4. Enter keywords/roles
5. Click **Fetch Live** in each section
6. Download CSVs when needed
