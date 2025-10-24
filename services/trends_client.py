# services/trends_client.py
import time, random, datetime as dt
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception_type
from pytrends.request import TrendReq
from pytrends.exceptions import TooManyRequestsError
import requests

BASE_SLEEP_SEC = 4.0
JITTER_SEC = 2.0
_session = None

def _sleep(): time.sleep(BASE_SLEEP_SEC + random.uniform(0, JITTER_SEC))

def get_client(hl="en-US", tz=480):
    global _session
    if _session: return _session
    _session = TrendReq(
        hl=hl, tz=tz, timeout=(10,30), retries=0, backoff_factor=0,
        requests_args={"headers": {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X) Safari/605.1.15"}}
    )
    return _session

def sanitize_related_df(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame(columns=["query","value"])
    df = df.copy()
    df["value"] = pd.to_numeric(df["value"], errors="coerce").fillna(0).astype(int)
    df["query"] = df["query"].astype(str)
    return df

@retry(stop=stop_after_attempt(3), wait=wait_random_exponential(multiplier=2, max=60),
       retry=retry_if_exception_type(TooManyRequestsError), reraise=True)
def fetch_iot(keys, timeframe, geo):
    py = get_client(); py.build_payload(keys, timeframe=timeframe, geo=geo)
    df = py.interest_over_time(); _sleep()
    if df is None or df.empty: return pd.DataFrame()
    return df.drop(columns=["isPartial"], errors="ignore").reset_index()

def fetch_region_frames(keyword, months, geo, resolution="COUNTRY"):
    py = get_client(); frames = []
    for m in range(months,0,-1):
        end = dt.date.today()-dt.timedelta(days=30*(m-1))
        start = end - dt.timedelta(days=30)
        tf = f"{start} {end}"
        py.build_payload([keyword], timeframe=tf, geo=geo)
        reg = py.interest_by_region(resolution=resolution, inc_low_vol=True, inc_geo_code=True)
        if reg is None or reg.empty: continue
        key_col = keyword if keyword in reg.columns else reg.select_dtypes("number").columns[0]
        df = reg.reset_index().rename(columns={"geoName":"region", key_col:"value"})
        df["date_frame"]=str(end)[:7]; frames.append(df[["region","value","date_frame"]])
        _sleep()
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

def fetch_related(keyword, geo):
    py = get_client(); py.build_payload([keyword], timeframe="today 12-m", geo=geo)
    rq = py.related_queries(); _sleep()
    slot = rq.get(keyword, {}) if isinstance(rq, dict) else {}
    return {"top": slot.get("top", pd.DataFrame()), "rising": slot.get("rising", pd.DataFrame())}

def fetch_trending_daily(geo="australia"):
    py = get_client()
    try:
        df = py.trending_searches(pn=geo); _sleep()
        if df is not None and not df.empty: df.columns=["query"]; return df
    except Exception: pass
    return pd.DataFrame()

def fetch_trending_realtime(geo="AU", cat="all"):
    py = get_client()
    try:
        df = py.realtime_trending_searches(pn=geo, cat=cat); _sleep()
        if df is not None and not df.empty:
            if "title" not in df.columns: df=df.rename(columns={df.columns[0]:"title"})
            return df
    except Exception: pass
    return pd.DataFrame()
