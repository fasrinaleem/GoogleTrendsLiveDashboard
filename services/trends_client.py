# services/trends_client.py  (FINAL)
from __future__ import annotations

import json
import random
import re
import time
import datetime as dt
from typing import Dict, Iterable
from urllib.parse import urlencode

import pandas as pd
import requests
import requests as _rq
import streamlit as st
from pytrends.exceptions import TooManyRequestsError, ResponseError
from pytrends.request import TrendReq
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception_type,
)

# ───────────────────────────── Config (optional via .streamlit/secrets.toml) ─────────────────────────────
# [trends]
# extra_sleep = 1.0
# use_proxies = false
# proxies = ["http://host:port","http://user:pass@host:port"]

_secrets = st.secrets.get("trends", {}) if hasattr(st, "secrets") else {}
_EXTRA_SLEEP = float(_secrets.get("extra_sleep", 1.0))
_USE_PROXIES = bool(_secrets.get("use_proxies", False))
_raw_proxies = _secrets.get("proxies", [])
_PROXIES: list[str] = [
    str(p).strip()
    for p in (_raw_proxies if isinstance(_raw_proxies, (list, tuple)) else [])
    if str(p).strip()
]

# ─────────────────────────────────────────── Globals & Debug ────────────────────────────────────────────
_BASE_SLEEP = 1.2 + _EXTRA_SLEEP
_JITTER = (0.3, 0.9)
_session: TrendReq | None = None

LAST_STATUS: Dict[str, str] = {}  # shown in the UI debug expander

def _throttle(extra: float = 0.0):
    time.sleep(_BASE_SLEEP + random.uniform(*_JITTER) + extra)

def _touch_debug(label: str, value: str):
    LAST_STATUS[label] = value
    LAST_STATUS["ts"] = dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"

# ───────────────────────────────────────────── HTTP helpers ─────────────────────────────────────────────
_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)

def _http_session() -> _rq.Session:
    s = _rq.Session()
    s.headers.update(
        {
            "User-Agent": _UA,
            "Accept": "*/*",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://trends.google.com/",
            "Connection": "keep-alive",
        }
    )
    # Avoid consent wall
    s.cookies.set("CONSENT", "YES+")
    return s

def _strip_xssi(text: str) -> str:
    # Google prefixes many JSON responses with )]}' to prevent XSSI
    return text[5:] if text.startswith(")]}',") else text

# ───────────────────────────────────────────── PyTrends client ──────────────────────────────────────────
def get_client(hl: str = "en-US", tz: int = 480) -> TrendReq:
    """
    Singleton TrendReq with realistic headers, no duplicate 'cookies' arg,
    and proxy handling that won't trip pytrends internals.
    """
    global _session
    if _session is not None:
        return _session

    headers = {
        "User-Agent": _UA,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    }

    args = {
        "hl": hl,
        "tz": tz,
        "timeout": (12, 40),
        "retries": 0,
        "backoff_factor": 0,
        # IMPORTANT: do NOT pass "cookies" here; pytrends manages cookies internally
        "requests_args": {"headers": headers},
        # pytrends expects a list; give empty list if not using proxies
        "proxies": list(_PROXIES) if (_USE_PROXIES and len(_PROXIES) > 0) else [],
    }

    _session = TrendReq(**args)

    # Force consent cookie and normalise pytrends proxy attributes
    try:
        _session.cookies.set("CONSENT", "YES+")
        if not isinstance(_session.proxies, list):
            _session.proxies = []
        if getattr(_session, "proxy_index", None) is None:
            _session.proxy_index = 0
    except Exception:
        _session.proxies = []

    LAST_STATUS["client"] = "ready"
    return _session

# ───────────────────────────────────────────── Utilities ────────────────────────────────────────────────
def _safe_df(df) -> pd.DataFrame:
    return df if isinstance(df, pd.DataFrame) else pd.DataFrame()

def _try(func, default):
    import traceback

    try:
        return func()
    except (TooManyRequestsError, ResponseError, requests.exceptions.RetryError, Exception) as e:
        LAST_STATUS["last_error"] = f"{type(e).__name__}: {e}"
        tb = traceback.format_exc().strip().splitlines()[-1]
        LAST_STATUS["last_error_trace"] = tb
        return default

def sanitize_related_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise 'related queries' to exactly ['query','value'] regardless of source schema.
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame(columns=["query", "value"])

    out = df.copy()

    # query column
    qcol = None
    for cand in ["query", "title", "topic_title", "keyword"]:
        if cand in out.columns:
            qcol = cand
            break
    if qcol is None:
        nonnum = out.select_dtypes(exclude=["number"]).columns.tolist()
        qcol = nonnum[0] if nonnum else out.columns[0]
    out["query"] = out[qcol].astype(str)

    # value column
    vcol = None
    for cand in ["value", "score", "traffic"]:
        if cand in out.columns:
            vcol = cand
            break
    if vcol is None:
        nums = out.select_dtypes(include=["number"]).columns.tolist()
        out["value"] = (
            pd.to_numeric(out[nums[0]], errors="coerce").fillna(0).astype(int) if nums else 0
        )
    else:
        out["value"] = pd.to_numeric(out[vcol], errors="coerce").fillna(0).astype(int)

    return out[["query", "value"]]

# ───────────────────────────────────────────── Interest over time ───────────────────────────────────────
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
    try:
        df = _iot_once(tuple(keys), timeframe, geo)
        if df is not None and not df.empty:
            LAST_STATUS["iot"] = f"ok geo='{geo}' tf='{timeframe}'"
            return df.drop(columns=["isPartial"], errors="ignore").reset_index()

        if timeframe != "today 3-m":
            df2 = _iot_once(tuple(keys), "today 3-m", geo)
            if df2 is not None and not df2.empty:
                LAST_STATUS["iot"] = "fallback ok tf='today 3-m'"
                return df2.drop(columns=["isPartial"], errors="ignore").reset_index()

        if geo:
            df3 = _iot_once(tuple(keys), timeframe, "")
            if df3 is not None and not df3.empty:
                LAST_STATUS["iot"] = f"fallback ok geo='' tf='{timeframe}'"
                return df3.drop(columns=["isPartial"], errors="ignore").reset_index()

        LAST_STATUS["iot"] = "empty"
        return pd.DataFrame()
    except Exception as e:
        LAST_STATUS["iot"] = f"error {type(e).__name__}: {e}"
        return pd.DataFrame()

# ───────────────────────────────────────────── Interest by region (monthly frames) ──────────────────────
def fetch_region_frames(keyword: str, months: int, geo: str, resolution: str = "COUNTRY") -> pd.DataFrame:
    def _frames(_geo: str) -> pd.DataFrame:
        py = get_client()
        frames = []
        today = dt.date.today()
        months_clamped = int(max(1, min(12, months)))
        for m in range(months_clamped, 0, -1):
            end = today - dt.timedelta(days=30 * (m - 1))
            start = end - dt.timedelta(days=30)
            _try(lambda: py.build_payload([keyword], timeframe=f"{start} {end}", geo=_geo), False)
            _throttle(0.3)
            reg = _try(
                lambda: py.interest_by_region(
                    resolution=resolution, inc_low_vol=True, inc_geo_code=True
                ),
                pd.DataFrame(),
            )
            _throttle(0.3)
            if reg is None or reg.empty:
                continue
            key_col = (
                keyword
                if keyword in reg.columns
                else (reg.select_dtypes("number").columns.tolist() or [None])[0]
            )
            if key_col is None:
                continue
            df = reg.reset_index().rename(
                columns={"geoName": "region", key_col: "value", "geoCode": "iso2"}
            )
            df["date_frame"] = str(end)[:7]
            frames.append(
                df[["region", "value", "date_frame"] + (["iso2"] if "iso2" in df.columns else [])]
            )
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    first = _frames(geo)
    if not first.empty:
        LAST_STATUS["frames"] = f"ok geo='{geo}'"
        return first
    second = _frames("")
    LAST_STATUS["frames"] = "fallback-worldwide" if not second.empty else "empty"
    return second

# ───────────────────────────────────────────── Related queries ──────────────────────────────────────────
def fetch_related(keyword: str, geo: str) -> Dict[str, pd.DataFrame]:
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
    LAST_STATUS["related"] = (
        "fallback-worldwide" if (not out2["top"].empty or not out2["rising"].empty) else "empty"
    )
    return out2

# ───────────────────────────────────────────── Trending (daily & realtime) ──────────────────────────────
# Mapping between daily slugs (pytrends) and ISO-2 (HTTP)
_DAILY_ISO_BY_SLUG = {
    "united_states": "US",
    "australia": "AU",
    "india": "IN",
    "united_kingdom": "GB",
    "canada": "CA",
    "singapore": "SG",
    "new_zealand": "NZ",
}
_DAILY_PN_BY_ISO = {v: k for k, v in _DAILY_ISO_BY_SLUG.items()}

def _norm_daily_pn(geo: str | None) -> list[str]:
    """Candidates for pytrends daily (slugs)."""
    g = (geo or "").strip()
    if len(g) > 2:  # name typed
        base = [g.lower().replace(" ", "_")]
    else:  # ISO-2
        base = [_DAILY_PN_BY_ISO.get(g.upper(), "united_states")]
    out: list[str] = []
    for x in base + ["united_states", "australia", "india"]:
        if x not in out:
            out.append(x)
    return out

def _daily_slug_to_iso2(slug_or_iso: str) -> str:
    s = (slug_or_iso or "").strip()
    return s.upper() if len(s) == 2 else _DAILY_ISO_BY_SLUG.get(s.lower(), "US")

def _norm_rt_pn(geo: str | None) -> list[str]:
    """Candidates for realtime (ISO-2)."""
    g = (geo or "").strip()
    if len(g) == 2:
        base = [g.upper()]
    else:
        m = {
            "australia": "AU",
            "united states": "US",
            "usa": "US",
            "india": "IN",
            "united kingdom": "GB",
            "uk": "GB",
            "canada": "CA",
            "singapore": "SG",
        }
        base = [m.get(g.lower(), "US")]
    out: list[str] = []
    for x in base + ["US", "AU", "IN"]:
        if x not in out:
            out.append(x)
    return out

# ---- Daily fallback: JSON → HTML page scrape ----
def _daily_via_http(pn_slug_or_iso: str, hl="en-US", tz=0) -> pd.DataFrame:
    """
    Daily trending fallback:
      1) JSON API: /trends/api/dailytrends?geo=US
      2) HTML page: /trends/trendingsearches/daily?geo=US  (extract embedded JSON)
    Returns DataFrame with 'query' column.
    """
    s = _http_session()
    geo_iso = _daily_slug_to_iso2(pn_slug_or_iso)

    # JSON API
    url_json = "https://trends.google.com/trends/api/dailytrends"
    params_json = {"hl": hl, "tz": tz, "geo": geo_iso}
    full_json = f"{url_json}?{urlencode(params_json)}"
    _touch_debug("trending_daily_http_url", full_json)
    r = s.get(full_json, timeout=20)
    _touch_debug("trending_daily_http_status", str(r.status_code))
    if r.status_code == 200:
        try:
            data = json.loads(_strip_xssi(r.text))
            days = data.get("default", {}).get("trendingSearchesDays", [])
            rows = []
            for d in days:
                for item in d.get("trendingSearches", []):
                    title = item.get("title", {}).get("query")
                    if title:
                        rows.append({"query": title})
            if rows:
                return pd.DataFrame(rows)
        except Exception:
            pass

    # HTML page with embedded JSON
    url_html = "https://trends.google.com/trends/trendingsearches/daily"
    params_html = {"hl": hl, "geo": geo_iso}
    full_html = f"{url_html}?{urlencode(params_html)}"
    _touch_debug("trending_daily_html_url", full_html)
    r2 = s.get(full_html, timeout=20)
    _touch_debug("trending_daily_html_status", str(r2.status_code))
    if r2.status_code != 200:
        return pd.DataFrame()

    # The page contains: var json = {...};
    m = re.search(r"var\s+json\s*=\s*(\{.*?\});", r2.text, re.S)
    if not m:
        return pd.DataFrame()
    try:
        data = json.loads(m.group(1))
        days = data.get("trendingSearchesDays", [])
        rows = []
        for d in days:
            for item in d.get("trendingSearches", []):
                title = item.get("title", {}).get("query")
                if title:
                    rows.append({"query": title})
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame()

# ---- Realtime fallback: JSON → HTML page scrape ----
def _realtime_via_http(pn_iso2: str, cat="all", hl="en-US", tz=0) -> pd.DataFrame:
    """
    Realtime trending fallback:
      1) JSON API: /trends/api/realtimetrends?geo=US&cat=all
      2) HTML page: /trends/trendingsearches/realtime?geo=US&category=all  (extract embedded JSON)
    Returns DataFrame with 'title' column.
    """
    s = _http_session()

    # JSON API
    url_json = "https://trends.google.com/trends/api/realtimetrends"
    params_json = {
        "hl": hl,
        "tz": tz,
        "cat": cat,
        "fi": 0,
        "fs": 0,
        "geo": pn_iso2,
        "ri": 300,
        "rs": 20,
        "sort": 0,
    }
    full_json = f"{url_json}?{urlencode(params_json)}"
    _touch_debug("trending_rt_http_url", full_json)
    r = s.get(full_json, timeout=20)
    _touch_debug("trending_rt_http_status", str(r.status_code))
    if r.status_code == 200:
        try:
            data = json.loads(_strip_xssi(r.text))
            stories = data.get("storySummaries", {}).get("trendingStories", [])
            rows = []
            for it in stories:
                title = it.get("title") or (it.get("entityNames", [None])[0])
                if title:
                    rows.append({"title": title})
            if rows:
                return pd.DataFrame(rows)
        except Exception:
            pass

    # HTML page with embedded JSON ("trendingStories" array)
    url_html = "https://trends.google.com/trends/trendingsearches/realtime"
    params_html = {"hl": hl, "tz": tz, "category": cat, "geo": pn_iso2}
    full_html = f"{url_html}?{urlencode(params_html)}"
    _touch_debug("trending_rt_html_url", full_html)
    r2 = s.get(full_html, timeout=20)
    _touch_debug("trending_rt_html_status", str(r2.status_code))
    if r2.status_code != 200:
        return pd.DataFrame()

    m = re.search(r'(?s)("trendingStories"\s*:\s*\[.*?\])', r2.text)
    if not m:
        return pd.DataFrame()
    try:
        data = json.loads("{" + m.group(1) + "}")
        stories = data.get("trendingStories", [])
        rows = []
        for it in stories:
            title = it.get("title") or (it.get("entityNames", [None])[0])
            if title:
                rows.append({"title": title})
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame()

# ---- Public fetchers (pytrends first, then HTTP fallbacks) -------------------
def fetch_trending_daily(geo: str = "australia") -> pd.DataFrame:
    py = get_client()
    for pn in _norm_daily_pn(geo):  # pytrends wants slugs
        df = _try(lambda: py.trending_searches(pn=pn), pd.DataFrame())
        _throttle(0.2)
        if isinstance(df, pd.DataFrame) and not df.empty:
            df = df.copy()
            df.columns = ["query"]
            LAST_STATUS["trending_daily"] = f"ok pn='{pn}' (pytrends)"
            return df

        # HTTP fallback (converts slug → ISO-2 internally)
        df_http = _daily_via_http(pn)
        if isinstance(df_http, pd.DataFrame) and not df_http.empty:
            LAST_STATUS["trending_daily"] = f"ok http geo='{_daily_slug_to_iso2(pn)}'"
            return df_http

    LAST_STATUS["trending_daily"] = "empty"
    return pd.DataFrame()

def fetch_trending_realtime(geo: str = "AU", cat: str = "all") -> pd.DataFrame:
    py = get_client()
    for pn in _norm_rt_pn(geo):  # realtime wants ISO-2
        df = _try(lambda: py.realtime_trending_searches(pn=pn, cat=cat), pd.DataFrame())
        _throttle(0.2)
        if isinstance(df, pd.DataFrame) and not df.empty:
            if "title" not in df.columns:
                df = df.rename(columns={df.columns[0]: "title"})
            LAST_STATUS["trending_rt"] = f"ok pn='{pn}' (pytrends)"
            return df

        df_http = _realtime_via_http(pn, cat=cat)
        if isinstance(df_http, pd.DataFrame) and not df_http.empty:
            LAST_STATUS["trending_rt"] = f"ok pn='{pn}' (http)"
            return df_http

    LAST_STATUS["trending_rt"] = "empty"
    return pd.DataFrame()

# ───────────────────────────────────────────── Health for UI ────────────────────────────────────────────
def get_health() -> Dict[str, str]:
    """Return last statuses (shown in the UI's Debug box)."""
    return dict(LAST_STATUS)
