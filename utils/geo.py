# utils/geo.py
from typing import Optional, Tuple
import pycountry
import pandas as pd

def country_name_to_iso2(name: str | None) -> Optional[str]:
    m = {"australia":"AU","united states":"US","usa":"US","united kingdom":"GB","uk":"GB",
         "india":"IN","canada":"CA","singapore":"SG","new zealand":"NZ",
         "germany":"DE","france":"FR","japan":"JP","brazil":"BR"}
    return m.get((name or "").strip().lower())

def add_iso_codes(df: pd.DataFrame) -> pd.DataFrame:
    if "region" not in df.columns: return df
    df = df.copy()
    try:
        df["iso2"] = df["region"].apply(country_name_to_iso2)
        df["iso3"] = df["iso2"].apply(lambda x: pycountry.countries.get(alpha_2=x).alpha_3 if x else None)
    except Exception: pass
    return df

def resolve_geo(user_input: str) -> Tuple[str, str, Optional[str]]:
    s = (user_input or "").strip()
    if not s or s.lower() in {"world","worldwide","global"}: return "", "Worldwide", None
    if s.lower()=="perth": return "AU","Perth, Australia","Perth"
    if "," in s:
        city,country=s.split(",",1)
        iso=country_name_to_iso2(country) or (country.strip().upper() if len(country.strip())==2 else None)
        return (iso or ""), f"{city.strip()}, {country.strip()}", city.strip()
    iso=country_name_to_iso2(s) or (s.upper() if len(s)==2 else "")
    return (iso or ""), (s.title() if iso else s), None
