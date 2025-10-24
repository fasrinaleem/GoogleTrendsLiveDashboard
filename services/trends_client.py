# services/trends_client.py
import os, time, random, datetime as dt
from typing import Iterable, Dict, Tuple
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception_type
from pytrends.request import TrendReq
from pytrends.exceptions import TooManyRequestsError
import requests
import streamlit as st

# ---- Config via .streamlit/secrets.toml (optional) ---------------------------
# Example:
# [trends]
# extra_sleep = 1.0
# use_proxies = false
# proxies = ["http://host:port","http://user:pass@host:port"]

_secrets = st.secrets.get("trends", {}) if hasattr(st, "secrets") else {}
_EXTRA_SLEEP = float(_secrets.get("extra_sleep", 1.0))     # add on top of base
_USE_PROXIES = bool(_secrets.get("use_proxies", False))
_PROXIES = list(_secrets.get("proxies", []))

# ---- Global throttle (keeps us under the radar) ------------------------------
_BASE_SLEEP = 1.2 + _EXTRA_SLEEP     # default ~2.2s if extra_sleep=1.0
_JITTER = (0.3, 0.9)                 # random jitter
_session: TrendReq | None = None

LAST_STATUS: Dict[str, str] = {}     # basic diagnostics for the UI

def _throttle(extra: float = 0.0):
    time.sleep(_BASE_SLEEP + random.uniform(*_JITTER) + extra)

def _pick_proxy() -> Dict[str, str]:
    if _USE_PROXIES and _PROXIES:
        p = random.choice(_PROXIES)
        return {"http": p, "https": p}
    return {}

def get_client(hl="en-US", tz=480) -> TrendReq:
    """Singleton TrendReq session with realistic headers + consent cookie."""
    global _session
    if _session is not None:
        return _session

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/126.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    }
    cookies = {"CONSENT": "YES+"}  # skip consent wall

    _session = TrendReq(
        hl=hl, tz=tz,
        timeout=(12, 40),
        retries=0,
        backoff_factor=0,
        proxies=_pick_proxy(),
        requests_args={"headers": headers, "cookies": cookies},
    )
    LAST_STATUS["client"] = "ready"
    return _session

# ---- Small helpers -----------------------------------------------------------
def _safe_df(df) -> pd.DataFrame:
    return df if isinstance(df, pd.DataFrame) else pd.DataFrame()

def _try(func, default):
    try:
        return func()
    except (TooManyRequestsError, requests.exceptions.RetryError, Exception) as e:
        LAST_STATUS["last_error"] = f"{type(e).__name__}"
        return default

def sanitize_related_df(df: pd.DataFrame) -> pd.DataFrame:
    """Public helper so views can sanitize DataFrames consistently."""
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame(columns=["query", "value"])
    out = df.copy()
    out["value"] = pd.to_numeric(out["value"], errors="coerce").fillna(0).astype(int)
    out["query"] = out["query"].astype(str)
    return out

# ---- Interest Over Time ------------------------------------------------------
@retry(
    stop=stop_after_attempt(4),
    wait=wait_random_exponential(multiplier=1.8, max=40),
    retry=retry_if_exception_type((TooManyRequestsError, requests.exceptions.RetryError)),
    reraise=True,
)
def _iot_once(keys, timeframe, geo) -> pd.DataFrame:
    py = get_client()
    py.build_payload(list(keys)[:5], timeframe=timeframe, geo=geo)
    _throttle(0.3)
    df = py.interest_over_time()
    _throttle(0.3)
    return _safe_df(df)

def fetch_iot(keys: Iterable[str], timeframe: str, geo: str) -> pd.DataFrame:
    """IoT with smart fallbacks: try (geo,timeframe) → try (geo,"today 3-m") → try ("",timeframe)."""
    try:
        df = _iot_once(tuple(keys), timeframe, geo)
        if df is not None and not df.empty:
            LAST_STATUS["iot"] = f"ok geo='{geo}' tf='{timeframe}'"
            return df.drop(columns=["isPartial"], errors="ignore").reset_index()

        # Fallback 1: keep geo, shorten timeframe to today 3-m
        if timeframe != "today 3-m":
            df2 = _iot_once(tuple(keys), "today 3-m", geo)
            if df2 is not None and not df2.empty:
                LAST_STATUS["iot"] = f"fallback ok geo='{geo}' tf='today 3-m'"
                return df2.drop(columns=["isPartial"], errors="ignore").reset_index()

        # Fallback 2: worldwide with original timeframe
        if geo:
            df3 = _iot_once(tuple(keys), timeframe, "")
            if df3 is not None and not df3.empty:
                LAST_STATUS["iot"] = f"fallback ok geo='' tf='{timeframe}'"
                return df3.drop(columns=["isPartial"], errors="ignore").reset_index()

        LAST_STATUS["iot"] = "empty"
        return pd.DataFrame()
    except Exception as e:
        LAST_STATUS["iot"] = f"error {type(e).__name__}"
        return pd.DataFrame()

# ---- Interest by Region (monthly frames) -------------------------------------
def fetch_region_frames(keyword: str, months: int, geo: str, resolution: str = "COUNTRY") -> pd.DataFrame:
    """Build monthly frames; if empty with geo, retry worldwide."""
    def _frames(_geo: str) -> pd.DataFrame:
        py = get_client()
        frames = []
        today = dt.date.today()
        months_clamped = int(max(1, min(12, months)))
        for m in range(months_clamped, 0, -1):
            end = today - dt.timedelta(days=30 * (m - 1))
            start = end - dt.timedelta(days=30)
            timeframe = f"{start} {end}"
            _try(lambda: py.build_payload([keyword], timeframe=timeframe, geo=_geo), False)
            _throttle(0.3)
            reg = _try(lambda: py.interest_by_region(resolution=resolution, inc_low_vol=True, inc_geo_code=True),
                       pd.DataFrame())
            _throttle(0.3)
            if reg is None or reg.empty:
                continue
            key_col = keyword if keyword in reg.columns else (reg.select_dtypes("number").columns.tolist() or [None])[0]
            if key_col is None:
                continue
            df = reg.reset_index().rename(columns={"geoName": "region", key_col: "value", "geoCode": "iso2"})
            df["date_frame"] = str(end)[:7]
            frames.append(df[["region", "value", "date_frame"] + (["iso2"] if "iso2" in df.columns else [])])
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    first = _frames(geo)
    if not first.empty:
        LAST_STATUS["frames"] = f"ok geo='{geo}'"
        return first
    second = _frames("")
    LAST_STATUS["frames"] = "fallback-worldwide" if not second.empty else "empty"
    return second

# ---- Related Queries ---------------------------------------------------------
def fetch_related(keyword: str, geo: str) -> Dict[str, pd.DataFrame]:
    """Fetch related; if empty for geo, retry worldwide."""
    def _rel(_geo: str) -> Dict[str, pd.DataFrame]:
        py = get_client()
        _try(lambda: py.build_payload([keyword], timeframe="today 12-m", geo=_geo), False)
        _throttle(0.3)
        rq = _try(lambda: py.related_queries(), {}) or {}
        _throttle(0.3)
        slot = rq.get(keyword, {}) if isinstance(rq, dict) else {}
        return {
            "top": sanitize_related_df(slot.get("top", pd.DataFrame())),
            "rising": sanitize_related_df(slot.get("rising", pd.DataFrame())),
        }
    out = _rel(geo)
    if not out["top"].empty or not out["rising"].empty:
        LAST_STATUS["related"] = f"ok geo='{geo}'"
        return out
    out2 = _rel("")
    LAST_STATUS["related"] = "fallback-worldwide" if (not out2["top"].empty or not out2["rising"].empty) else "empty"
    return out2

# ---- Trending (daily + realtime) ---------------------------------------------
def fetch_trending_daily(geo: str = "australia") -> pd.DataFrame:
    py = get_client()
    df = _try(lambda: py.trending_searches(pn=geo), pd.DataFrame())
    _throttle(0.2)
    if isinstance(df, pd.DataFrame) and not df.empty:
        df = df.copy(); df.columns = ["query"]
        LAST_STATUS["trending_daily"] = "ok"
        return df
    LAST_STATUS["trending_daily"] = "empty"
    return pd.DataFrame()

def fetch_trending_realtime(geo: str = "AU", cat: str = "all") -> pd.DataFrame:
    py = get_client()
    df = _try(lambda: py.realtime_trending_searches(pn=geo, cat=cat), pd.DataFrame())
    _throttle(0.2)
    if isinstance(df, pd.DataFrame) and not df.empty:
        if "title" not in df.columns: df = df.rename(columns={df.columns[0]: "title"})
        LAST_STATUS["trending_rt"] = "ok"
        return df
    LAST_STATUS["trending_rt"] = "empty"
    return pd.DataFrame()

# ---- Health check for UI -----------------------------------------------------
def get_health() -> Dict[str, str]:
    """Return last statuses to display in an expander for debugging."""
    return dict(LAST_STATUS)
