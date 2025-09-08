import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud, STOPWORDS
from io import BytesIO
from utils import zscore_spikes, add_iso_codes

def line_with_spikes(df: pd.DataFrame, keywords, max_labels=6):
    fig = go.Figure()
    for kw in keywords:
        if kw not in df.columns: continue
        fig.add_trace(go.Scatter(x=df["date"], y=df[kw], mode="lines", name=kw))
        spikes = zscore_spikes(df[kw], window=7, z=1.8, min_gap=5)
        spikes = spikes.sort_values("z", ascending=False).head(max_labels).sort_values("index")
        for _, row in spikes.iterrows():
            i = int(row["index"])
            fig.add_trace(go.Scatter(
                x=[df["date"].iloc[i]], y=[df[kw].iloc[i]],
                mode="markers+text",
                text=[f"Spike: {int(row['value'])}"],
                textposition="top center", showlegend=False
            ))
    fig.update_layout(title="Interest Over Time (Annotated Spikes)", xaxis_title="Date", yaxis_title="Interest (0–100)")
    return fig

def animated_choropleth(frames_df: pd.DataFrame, title="Animated Interest by Country"):
    if frames_df.empty: return go.Figure()
    frames_df = add_iso_codes(frames_df, "region")
    df = frames_df.dropna(subset=["iso3"]).copy()
    df["value"] = df["value"].astype(float)
    fig = px.choropleth(df, locations="iso3", color="value", hover_name="region",
                        animation_frame="date_frame", color_continuous_scale="Viridis",
                        projection="natural earth", title=title)
    fig.update_layout(coloraxis_colorbar=dict(title="Interest"))
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
    buf = BytesIO()
    wc.to_image().save(buf, format="PNG")
    buf.seek(0)
    return buf
