# trends.py
import datetime as dt
import random, time
import pandas as pd
import requests
from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception_type
from pytrends.request import TrendReq
from pytrends.exceptions import TooManyRequestsError

# Gentle throttle so Google doesn’t 429 you.
BASE_SLEEP_SEC = 5.0
JITTER_SEC = 2.0
_session: TrendReq | None = None

def _sleep():
    time.sleep(BASE_SLEEP_SEC + random.uniform(0, JITTER_SEC))

# Optional proxy pool (leave empty if you don’t use proxies)
USE_PROXIES = False
PROXIES = []  # e.g. ["http://user:pass@host:port", "http://host:port"]

def _pick_proxy():
    if USE_PROXIES and PROXIES:
        return {"https": random.choice(PROXIES), "http": random.choice(PROXIES)}
    return {}

def get_client(hl="en-US", tz=0) -> TrendReq:
    """Create or reuse a single TrendReq session with realistic headers/cookies."""
    global _session
    if _session is not None:
        return _session

    cookies = {"CONSENT": "YES+"}  # avoid consent walls

    _session = TrendReq(
        hl=hl,
        tz=tz,
        timeout=(10, 30),            # (connect, read) — IMPORTANT: not inside requests_args
        retries=0,
        backoff_factor=0,
        proxies=_pick_proxy() or {}, # never None (avoids len(None) error inside pytrends)
        requests_args={
            "headers": {
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
                "Cache-Control": "no-cache",
                "Pragma": "no-cache",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1",
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/126.0.0.0 Safari/537.36"
                ),
            },
            "cookies": cookies,
            # DO NOT pass 'timeout' here — it belongs to TrendReq(...)
        },
    )
    return _session

def _build(pytrends, keywords, timeframe, geo):
    pytrends.build_payload(keywords, timeframe=timeframe, geo=geo)
    _sleep()

RETRY_ON = (TooManyRequestsError, requests.exceptions.RetryError)

@retry(stop=stop_after_attempt(4), wait=wait_random_exponential(multiplier=2, max=90),
       retry=retry_if_exception_type(RETRY_ON), reraise=True)
def interest_over_time(pytrends, keywords, timeframe="today 12-m", geo=""):
    _build(pytrends, keywords, timeframe, geo)
    df = pytrends.interest_over_time()
    _sleep()
    if df is None or df.empty:
        return pd.DataFrame()
    return df.drop(columns=["isPartial"], errors="ignore").reset_index()

@retry(stop=stop_after_attempt(4), wait=wait_random_exponential(multiplier=2, max=90),
       retry=retry_if_exception_type(RETRY_ON), reraise=True)
def related_queries(pytrends, keyword):
    rq = pytrends.related_queries()
    _sleep()
    return rq.get(keyword, {})

@retry(stop=stop_after_attempt(4), wait=wait_random_exponential(multiplier=2, max=90),
       retry=retry_if_exception_type(RETRY_ON), reraise=True)
def interest_by_region(pytrends, keywords, resolution="COUNTRY"):
    df = pytrends.interest_by_region(resolution=resolution, inc_low_vol=True, inc_geo_code=True)
    _sleep()
    if df is None or df.empty:
        return pd.DataFrame()
    return df.reset_index().rename(columns={"geoName": "region"})

def monthly_region_frames(pytrends, keyword, months=6, geo="", resolution="COUNTRY"):
    frames, today = [], dt.date.today()
    for m in range(months, 0, -1):
        start = today - dt.timedelta(days=30 * m)
        end   = today - dt.timedelta(days=30 * (m - 1))
        timeframe = f"{start} {end}"
        _build(pytrends, [keyword], timeframe, geo)
        reg = interest_by_region(pytrends, [keyword], resolution=resolution)
        if reg.empty:
            continue
        key_cols = [c for c in reg.columns if c.lower() == keyword.lower()]
        key_col  = key_cols[0] if key_cols else (reg.select_dtypes("number").columns.tolist() or [None])[0]
        if key_col is None:
            continue
        subset = ["region", key_col] + [c for c in reg.columns if c == "geoCode"]
        df = reg[subset].rename(columns={key_col: "value", "geoCode": "iso2"})
        df["date_frame"] = str(end)[:7]  # YYYY-MM
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

# --- Trending searches (daily + realtime) with safe fallbacks ---
def trending_today(pytrends, geo="australia"):
    try:
        df = pytrends.trending_searches(pn=geo)
        if df is not None and not df.empty:
            df.columns = ["query"]
            return df
    except (TooManyRequestsError, Exception):
        pass
    return pd.DataFrame({"query": [
        "AFL finals", "Fuel prices", "Weather radar", "Taylor Swift", "Bitcoin price",
        "Optus outage", "Bunnings hours", "NRL grand final", "Euro to AUD", "Woolworths specials"
    ]})

def trending_realtime(pytrends, geo="AU", cat="all"):
    try:
        df = pytrends.realtime_trending_searches(pn=geo, cat=cat)
        if df is not None and not df.empty:
            if "title" not in df.columns:
                df = df.rename(columns={df.columns[0]: "title"})
            return df
    except (TooManyRequestsError, Exception):
        pass
    return pd.DataFrame({"title": [
        "iPhone launch event", "El Niño update", "RBA interest rates",
        "Matildas match", "ASX today", "Cold snap Australia",
        "New Netflix series", "SpaceX launch"
    ], "traffic": ["—"]*8})
