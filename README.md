# Trends Hub (PyTrends-only) — README

A modern Streamlit app for exploring Google Trends with **live, on-demand fetches**, **annotated time-series**, **animated regional maps**, and **word-cloud insights**—plus a **Job-Market** view that aggregates related skills across roles. It’s designed to load instantly using lightweight fallbacks, and only call the Google Trends endpoint when you explicitly click **Fetch Live**.

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
- **Packages** (key ones):
  - `streamlit`, `pandas`, `numpy`
  - `plotly`
  - `pytrends`
  - `wordcloud`
  - `tenacity`
  - `pycountry`

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

---

## 🧭 Sidebar Controls

- **View**
  - `Trends Studio` or `Job Market`
- **Region**
  - Quick picks: `Australia`, `Perth`, `Worldwide`, or `Custom`
  - `Custom` accepts country names/ISO-2 (e.g., `US`, `United States`) or the word `Perth`
- **Timeframe**
  - `Trends timeframe` (Studio): `today 12-m`, `today 3-m`, `now 7-d`, `today 5-y`
  - `Job Market timeframe` (Job Market): similar options
- **Visual options**
  - **Line palette**: `Vivid`, `Bright`, `Pastel`, `D3`, `G10`, `Dark24`
  - **Map color scale**: multiple Plotly scales (`Turbo`, `Viridis`, `Plasma`, …)
  - **Word cloud**: `max words` + `colormap`
- **Slow mode**
  - Adds gentle delays and increases retry backoff to avoid 429s.
- **Force refresh caches**
  - Clears `@st.cache_data` memoization.

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
  - **Daily (Australia)**: fallback list with button `Fetch Live (Daily)`
  - **Realtime**: fallback list (filtered by city if available) with `Fetch Live (Realtime)`
- **Interest Over Time (annotated)**
  - Synthetic series on load; `Fetch Live (Series)` pulls **PyTrends `interest_over_time`**.
  - KPI cards show latest value and 7-day average for the first series.
  - **Annotated spikes** (client-side) in chart come from z-score detection.
- **Animated Map — Interest by Region**
  - Fallback animated frames (`fb_frames`) on load.
  - `Fetch Live (Regions)` builds **monthly frames** using **`interest_by_region`** for each month window.
  - Optional CSV download.
- **Related Queries — Word Cloud**
  - Fallback top & rising keywords rendered as a **WordCloud**.
  - `Fetch Live (Related)` uses **`related_queries`** for the selected keyword, normalizes “Breakout” and “%” values, and renders the cloud + tables.

**Inputs**
- `Keywords (comma-separated, max 5)` → used for IoT & Map series selection.
- `Word cloud keyword` → selects which keyword to query for related terms.

---

### 2) Job Market

- **Overview**
  - Select up to 5 **roles** (e.g., Data Analyst, Data Scientist, …).
  - Shows IoT line chart + KPI cards (fallback first; `Fetch Live (Overview series)` to pull PyTrends).
  - **Related Insights (combined roles)**: builds a word cloud aggregating related queries across all selected roles (`Fetch Live (Related for roles)` to query live for each role and merge counts).
- **Trends by Date**
  - Choose which series to plot (subset of selected roles); `Fetch Live (Selected series)` pulls live IoT then re-renders chosen subset.
- **Interest by Region (animated)**
  - Fallback frames on load; `Fetch Live (Regions)` to pull monthly `interest_by_region` frames for the chosen role.
  - Download CSV for the concatenated frames.
- **Top & Rising**
  - Word cloud per role, or combined across all selected roles.
  - `Fetch Live (Top & Rising)` to see live tables & cloud.
- **Job Openings**
  - Convenience links to LinkedIn, Seek, Indeed for each selected role, with optional location prefill.

---

## 🧠 Live Data Layer (PyTrends)

All live calls are wrapped to be **gentle** to Google:

- **`@st.cache_resource`** memoizes the `TrendReq` client.
- **`@st.cache_data(ttl=90)`** memoizes data calls briefly (90 seconds).
- **`slow`** checkbox increases backoff between calls (and limits months for maps).
- **`TooManyRequestsError`** is caught and retried with exponential backoff.

### Key Live Functions (in `app_gtrends.py`)

- `live_iot(keys, timeframe, geo, slow)` → `interest_over_time()` (cleaned to drop `isPartial`)
- `live_frames(keyword, months, geo, resolution, slow)` → loops monthly windows, calls `interest_by_region`
- `live_related(keyword, geo, slow)` → `related_queries()` for one term
- `live_related_multi(roles, geo, slow)` → aggregates top/rising across multiple roles

> **Normalization of “Related” values**  
> Values can be numbers, `"Breakout"`, or percentages (`"45%"`).  
> `_sanitize_related_df()` converts all to integers; `"Breakout"` is treated as `120`.

---

## 📈 Visualization Layer

- **Line charts** with spikes (Plotly)
- **Animated choropleths** (Plotly Express)
- **Word clouds** (wordcloud library)
- **KPI cards** (HTML/CSS styled)

---

## 🧪 Fallbacks for Instant UI

- `fb_ts()` → synthetic IoT series
- `fb_frames()` → animated regional frames
- `fb_related_for_roles()` → synthetic related queries
- `fb_trending_daily()` / `fb_trending_rt()` → small trending lists

---

## 🔒 Caching Strategy

- `@st.cache_resource` → PyTrends session
- `@st.cache_data(ttl=90)` → live calls
- **Force refresh button** clears caches

---

## 🛡️ Rate-Limiting & Reliability

- **Slow mode** → gentler timing
- **Retries** → exponential backoff
- **Fallbacks** → always show something

---

## 🧮 Utilities (`utils.py`)

- `zscore_spikes(series, window, z, min_gap)` → finds trend spikes
- `country_name_to_iso2()` / `iso2_to_iso3()` → ISO helpers
- `add_iso_codes()` → attach iso2 + iso3 codes for maps

---

## 🧯 Troubleshooting

- **Blank live results** → enable **Slow mode**, reduce keywords, shorten timeframe, clear cache
- **429 TooManyRequestsError** → wait a few mins, keep Slow mode on
- **City empty** → not always available
- **Word cloud “no data”** → pick more popular keyword
- **SSL/Proxy errors** → only if using `trends.py` proxy pool

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
