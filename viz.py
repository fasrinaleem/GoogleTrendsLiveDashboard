import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud, STOPWORDS
from io import BytesIO
from utils import zscore_spikes, add_iso_codes

COLORWAY = ["#6D28D9", "#2563EB", "#DC2626", "#059669", "#EA580C", "#D946EF"]

def kpi_card(label: str, value: str, delta: str | None = None):
    """Return a small HTML card."""
    delta_html = f"<span class='kpi-delta'>{delta}</span>" if delta else ""
    return f"""
    <div class="kpi">
      <div class="kpi-label">{label}</div>
      <div class="kpi-value">{value}{delta_html}</div>
    </div>
    """

def line_with_spikes(df: pd.DataFrame, keywords, max_labels=5):
    fig = go.Figure()
    for kw in keywords:
        if kw not in df.columns: continue
        fig.add_trace(go.Scatter(x=df["date"], y=df[kw], mode="lines", name=kw))
        spikes = zscore_spikes(df[kw], window=7, z=1.8, min_gap=5)
        spikes = spikes.sort_values("z", ascending=False).head(max_labels).sort_values("index")
        for _, r in spikes.iterrows():
            i = int(r["index"])
            fig.add_trace(go.Scatter(
                x=[df["date"].iloc[i]], y=[df[kw].iloc[i]],
                mode="markers+text", text=[f"Spike: {int(r['value'])}"],
                textposition="top center", showlegend=False, marker=dict(size=8)
            ))
    fig.update_layout(
        template="plotly_white",
        colorway=COLORWAY,
        title=None,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title=None,
        yaxis_title="Interest (0–100)",
        hovermode="x unified",
    )
    return fig

def sparkline(df: pd.DataFrame, col: str):
    sp = go.Figure(go.Scatter(x=df["date"], y=df[col], mode="lines", line=dict(width=2), showlegend=False))
    sp.update_layout(template="plotly_white")
    sp.update_layout(margin=dict(l=0, r=0, t=0, b=0), xaxis=dict(visible=False), yaxis=dict(visible=False), height=60)
    return sp

def animated_choropleth(frames_df: pd.DataFrame, title="Regional interest"):
    if frames_df.empty: return go.Figure()
    frames_df = add_iso_codes(frames_df, "region")
    df = frames_df.dropna(subset=["iso3"]).copy()
    df["value"] = df["value"].astype(float)
    fig = px.choropleth(df, locations="iso3", color="value", hover_name="region", animation_frame="date_frame",
                        color_continuous_scale="Turbo", projection="natural earth", title=None,)
    fig.update_layout(template="plotly_white", margin=dict(l=10, r=10, t=10, b=10))
    return fig

def wordcloud_from_related(top_df, rising_df):
    text_weights = {}
    for d in [top_df, rising_df]:
        if d is not None and not d.empty:
            for _, r in d.iterrows():
                text_weights[r["query"]] = text_weights.get(r["query"], 0) + int(r["value"])
    wc = WordCloud(width=1000, height=400, background_color="white",
                   stopwords=set(STOPWORDS), collocations=False)
    wc.generate_from_frequencies(text_weights or {"no data": 1})
    buf = BytesIO(); wc.to_image().save(buf, format="PNG"); buf.seek(0)
    return buf
