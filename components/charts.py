# components/charts.py
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
from wordcloud import WordCloud, STOPWORDS
from utils.series import zscore_spikes
from utils.geo import add_iso_codes

COLORWAY = ["#6D28D9","#2563EB","#DC2626","#059669","#EA580C","#D946EF"]

def line_with_spikes(df: pd.DataFrame, keywords):
    fig = go.Figure()
    for kw in keywords:
        if kw not in df.columns: continue
        fig.add_trace(go.Scatter(x=df["date"], y=df[kw], mode="lines", name=kw))
        spikes = zscore_spikes(df[kw])
        for _, r in spikes.iterrows():
            fig.add_trace(go.Scatter(
                x=[df["date"].iloc[int(r["index"])]],
                y=[r["value"]],
                mode="markers+text",
                text=[f"{int(r['value'])}"],
                textposition="top center",
                marker=dict(size=6),
                showlegend=False,
            ))
    fig.update_layout(template="plotly_white", colorway=COLORWAY,
                      margin=dict(l=10,r=10,t=10,b=10), hovermode="x unified",
                      yaxis_title="Interest (0–100)")
    return fig

def animated_choropleth(df: pd.DataFrame, title="Regional Interest"):
    if df.empty: return go.Figure()
    df = add_iso_codes(df)
    fig = px.choropleth(df, locations="iso3", color="value",
                        hover_name="region", animation_frame="date_frame",
                        color_continuous_scale="Turbo", title=None,
                        projection="natural earth")
    fig.update_layout(template="plotly_white", margin=dict(l=10,r=10,t=10,b=10))
    return fig

def wordcloud_from_related(top_df, rising_df):
    text_weights = {}
    for d in [top_df, rising_df]:
        if d is not None and not d.empty:
            for _, r in d.iterrows():
                text_weights[str(r["query"])] = text_weights.get(str(r["query"]),0)+int(r["value"])
    wc = WordCloud(width=1400,height=520,background_color="white",
                   stopwords=set(STOPWORDS),collocations=False)
    wc.generate_from_frequencies(text_weights or {"no data":1})
    buf = BytesIO(); wc.to_image().save(buf, format="PNG"); buf.seek(0)
    return buf
