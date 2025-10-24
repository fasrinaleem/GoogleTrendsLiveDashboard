# utils/series.py
import numpy as np
import pandas as pd

def zscore_spikes(series: pd.Series, window=7, z=1.8, min_gap=5):
    s = series.astype(float).copy()
    roll_mean = s.rolling(window=window, center=True, min_periods=1).mean()
    roll_std = s.rolling(window=window, center=True, min_periods=1).std(ddof=0).replace(0, np.nan)
    zs = (s - roll_mean) / roll_std
    spikes_idx, last_i = [], -999
    for i, val in enumerate(zs):
        if pd.notna(val) and val >= z and (i - last_i) >= min_gap:
            spikes_idx.append(i); last_i = i
    return pd.DataFrame({"index": spikes_idx,
                         "value": s.iloc[spikes_idx].values,
                         "z": [zs.iloc[i] for i in spikes_idx]})
