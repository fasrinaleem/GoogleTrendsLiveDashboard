# 📊 Google Trends Live Dashboard

A powerful multi-page **Streamlit** web application that visualizes **real-time and historical Google Trends data** for job roles, skills, and global market analysis.

This project helps students, professionals, and researchers explore **job market trends**, **skill popularity**, **salary ranges**, **career growth paths**, and **global search interest** — all within a single, interactive, and visually appealing dashboard.

---

## 🌟 Overview

This system was rebuilt from the ground up with a **new modular architecture**, modern UI, and more features.  
Users can now analyze any skill or job role using real-world Google search data and receive instant insights across time, region, and industry.

### 🔑 Highlights
- Real-time and cached **PyTrends API** integration.
- Dynamic **interest over time** and **regional analytics**.
- **Animated world maps**, **word clouds**, and **career recommendations**.
- Built-in modules for **Skills**, **Job Market**, and **Trends Studio**.
- Professional white-themed UI with **Google Material-inspired styling**.

---

## 🧭 Navigation & Pages

### 🏠 **Main Page**
The first landing page users see.

**Features:**
- Elegant white-themed landing UI.
- Animated introduction banner.
- Quick access buttons for:
  - 📈 Trends Studio  
  - 🧠 Skills  
  - 💼 Job Market
- Project overview and purpose summary.

---

### 🧠 **Skills Page**
_Analyze how skills evolve in popularity and where they’re most in demand._

**Key Functions:**
- 📈 **Interest Over Time (12 months)** – visual line graph.  
- 🗺️ **Interest by Region** – bar chart by state or country.  
- 🌍 **Worldwide Animated Map** – bubble map showing global trends.  
- ☁️ **Word Cloud** – generated from top & rising related queries.  
- 💼 **Job Role Matcher** – suggests roles and career paths based on keywords.

**Example Use:**  
Search “Python” → See Australian state trends, global map, word cloud, and matching roles like Data Scientist or Backend Developer.

---

### 💼 **Job Market Page**
_Compare job roles, salaries, and related keywords over time._

**Key Functions:**
- 📊 **Interest Over Time** – multi-role comparison (e.g., Data Analyst vs Business Analyst).  
- 💰 **Salary Range (AUD)** – bar chart for Entry, Median, Senior, and Top 10% levels.  
- 🌎 **Worldwide Popularity (Animated Map)** – monthly visualization of interest growth.  
- 🔝 **Top & Rising Keywords** – tables and visuals for role-related trending queries.  
- ☁️ **Word Cloud** – based on related searches.  
- 🎓 **Course Finder** – opens relevant learning platforms (Coursera, edX, Udemy, LinkedIn).  
- 🔗 **Job Openings Links** – connect directly to SEEK, LinkedIn Jobs, or Indeed.

---

### 🌐 **Trends Studio**
_Global job trends, market topics, and highest paying roles overview._

**Key Functions:**
- 🔟 **Trending Keywords** – Top 10 most searched career-related terms.  
- 📰 **Trending Topics** – world job market and tech industry themes.  
- 🌍 **World Trending News** – region-wise professional trend highlights.  
- 💸 **Highest Paying Jobs Summary (AUD)** – top-earning countries and roles.

---

## 🧩 Project Architecture

GOOGLETRENDSLIVEDASHBOARD/
│
├── .streamlit/                 # Streamlit configuration
├── pages/                      # Multi-page app structure
│   ├── Job_Market.py           # Job analytics and market trends
│   ├── Skills.py               # Skills analysis and role insights
│   └── Trends_Studio.py        # Macro view of trends and global insights
│
├── Main.py                     # Landing page and navigation
├── style.css                   # Unified white design & animations
├── requirements.txt             # Dependencies
└── .gitignore / LICENSE         # Repo essentials


---

## 🧠 Data Flow & Logic

1. **User Input:** Skill or Job Role + Region  
2. **PyTrends API:** Fetches real-time Google Trends data  
3. **Fallback Handling:** If API limit exceeded → cached data loads automatically  
4. **Processing:** Pandas transforms and normalizes data  
5. **Visualization:** Rendered via Plotly, Streamlit, and WordCloud  
6. **Session State:** Caches results to maintain persistence across navigation  

---

## ⚙️ Installation Guide

### Step 1 — Clone the Repository
```bash
git clone https://github.com/yourusername/GoogleTrendsLiveDashboard.git
cd GoogleTrendsLiveDashboard

Step 2 — Create Virtual Environment (Optional)
python -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows

Step 3 — Install Requirements
pip install -r requirements.txt

Step 4 — Launch Streamlit App
streamlit run Main.py

Then open http://localhost:8501 in your browser.

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-------------|
| **Frontend** | Streamlit + CSS |
| **Backend** | Python 3.11+ |
| **Data Source** | Google Trends (PyTrends API) |
| **Visuals** | Plotly, WordCloud |
| **Design** | Google Fonts (Roboto), Custom CSS |
| **Hosting** | Streamlit Cloud / Localhost |

## 📈 Improvements from Previous Version

| Area | Previous Version | Current Version |
|------|------------------|-----------------|
| **Architecture** | Single-page, mixed logic | Modular 3-page design (Skills, Job Market, Trends Studio) |
| **Design** | Dark theme, minimal layout | Professional white theme via CSS styling |
| **Data Handling** | Static data only | Live API + cached fallback mechanism |
| **Performance** | Redundant reloads | Optimized via Streamlit session state |
| **Features** | Keyword trends only | Salary, courses, jobs, world map, and word cloud added |
| **User Navigation** | Sidebar links only | Clean landing with page-link buttons |
| **Visualizations** | Line chart only | Bar charts, word clouds, maps, animated visuals |
| **Scalability** | Monolithic | Modular & easily expandable |

## 📸 Screenshots

| Module | Preview |
|--------|----------|
| **Main** | <img width="1000" alt="Main Page" src="https://github.com/user-attachments/assets/6711e8be-cf40-4caa-9978-a7cbd4e97848" /> |
| **Skills** | <img width="1000" alt="Skills Page" src="https://github.com/user-attachments/assets/e5652a46-1589-4337-b3a7-661dd6239a7b" /> |
| **Job Market** | <img width="1000" alt="Job Market Page" src="https://github.com/user-attachments/assets/11b2f681-1c24-41af-af4e-355a760f1629" /> |
| **Trends Studio** | <img width="1000" alt="Trends Studio Page" src="https://github.com/user-attachments/assets/e643cf3a-928e-40ed-81ab-704a11238cfb" /> |

🔐 Data Handling Notes
	•	Uses PyTrends wrapper to communicate with Google Trends.
	•	Includes API retry logic and polite request delays.
	•	Fallback datasets simulate real data when API rate-limit is hit.
	•	All sections use session persistence, preventing unnecessary API calls.


🧰 Dependencies
	•	streamlit
	•	pandas
	•	plotly
	•	pytrends
	•	wordcloud
	•	pillow
	•	numpy

Install via:
pip install -r requirements.txt


This project is licensed under the MIT License.
You’re free to use, modify, and distribute it with attribution.


💡 Acknowledgements
	•	Google Trends API (PyTrends)
	•	Streamlit
	•	Plotly Express
	•	WordCloud Python Library

🧩 Summary

The Google Trends Live Dashboard transforms raw Google search data into actionable insights.
From skill interest analytics to job salary comparisons and global market intelligence — this project provides a modern, data-driven view of the world’s career landscape in real time.
