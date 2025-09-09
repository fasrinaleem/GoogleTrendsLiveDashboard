# utils.py
import numpy as np
import pandas as pd
import pycountry

def zscore_spikes(series: pd.Series, window: int = 7, z: float = 1.8, min_gap: int = 5) -> pd.DataFrame:
    s = series.astype(float).copy()
    roll_mean = s.rolling(window=window, min_periods=1, center=True).mean()
    roll_std  = s.rolling(window=window, min_periods=1, center=True).std(ddof=0).replace(0, np.nan)
    zs = (s - roll_mean) / roll_std
    spikes_idx, last_i = [], -999
    for i, val in enumerate(zs):
        if pd.notna(val) and val >= z and (i - last_i) >= min_gap:
            spikes_idx.append(i); last_i = i
    return pd.DataFrame({"index": spikes_idx,
                         "value": s.iloc[spikes_idx].values,
                         "z": [zs.iloc[i] for i in spikes_idx]})

def country_name_to_iso2(name: str):
    try:
        return pycountry.countries.lookup(name).alpha_2
    except Exception:
        return None

def iso2_to_iso3(iso2: str):
    try:
        return pycountry.countries.get(alpha_2=iso2).alpha_3
    except Exception:
        return None

def add_iso_codes(df: pd.DataFrame, country_col: str = "region") -> pd.DataFrame:
    df = df.copy()
    df["iso2"] = df[country_col].apply(country_name_to_iso2)
    df["iso3"] = df["iso2"].apply(lambda x: iso2_to_iso3(x) if isinstance(x, str) else None)
    return df
