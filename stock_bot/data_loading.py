"""Helpers to load OHLCV data from Polygon.io into pandas DataFrames.

Public functions:
  - get_polygon_data(ticker, start_date, end_date, api_key, ...)
  - load_tickers(tickers, start_date, end_date, api_key, ...)

The functions return a DataFrame with columns:
  ['timestamp', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume']

This module is intentionally small and dependency-free (requests + pandas).
"""
from __future__ import annotations

from typing import List, Optional, Union
import time
from datetime import date, datetime

import os
import pandas as pd
import requests

DEFAULT_COLUMNS = ["timestamp", "Ticker", "Open", "High", "Low", "Close", "Volume"]


def _to_iso(d: Union[str, date, datetime]) -> str:
    """Convert a date-like object or ISO string to YYYY-MM-DD string.

    Accepts strings already in YYYY-MM-DD format, datetime.date or datetime.
    """
    if isinstance(d, str):
        return d
    if isinstance(d, datetime):
        return d.date().isoformat()
    if isinstance(d, date):
        return d.isoformat()
    raise ValueError("start/end must be str, date or datetime")


def get_polygon_data(
    ticker: str,
    start_date: Union[str, date, datetime],
    end_date: Union[str, date, datetime],
    timespan: str = "day",
    adjusted: bool = True,
    limit: int = 5000,
    max_retries: int = 3,
    timeout: int = 10,
) -> pd.DataFrame:
    """Fetch OHLCV aggregates for `ticker` from Polygon.io and return a tidy DataFrame.

    Parameters
    - ticker: symbol, e.g. 'NVDA'
    - start_date, end_date: date-like (YYYY-MM-DD or datetime/date)
    - api_key: Polygon API key (required)
    - timespan: 'day'|'minute' etc. (kept as 'day' by default)
    - adjusted: whether to request adjusted prices

    Returns a DataFrame with columns: timestamp, Ticker, Open, High, Low, Close, Volume.
    If no data is returned, returns an empty DataFrame with the expected columns.
    """
    api_key = os.environ.get("POLYGON_API_KEY")
    if not api_key:
        raise ValueError("Environment variable POLYGON_API_KEY is required to fetch data from Polygon.io")

    start = _to_iso(start_date)
    end = _to_iso(end_date)

    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/{timespan}/{start}/{end}"
    params = {
        "adjusted": str(adjusted).lower(),
        "sort": "asc",
        "limit": limit,
        "apiKey": api_key,
    }

    last_exc: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            resp.raise_for_status()
            payload = resp.json()
            results = payload.get("results", [])

            if not results:
                # Return empty DataFrame with expected schema
                return pd.DataFrame(columns=DEFAULT_COLUMNS)

            df = pd.DataFrame(results)
            # polygon returns timestamps in ms in the 't' column
            if "t" in df.columns:
                df["timestamp"] = pd.to_datetime(df["t"], unit="ms")
            else:
                # fallback to index-based timestamp if present
                df["timestamp"] = pd.to_datetime(df.get("timestamp", pd.Series(dtype="datetime64[ns]")))

            df = df.rename(columns={"o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"})
            df["Ticker"] = ticker

            # Keep only the canonical columns and preserve order
            keep = [c for c in DEFAULT_COLUMNS if c in df.columns]
            return df[keep].reset_index(drop=True)

        except Exception as exc:  # network / JSON / 4xx/5xx
            last_exc = exc
            # simple backoff for retries
            if attempt < max_retries:
                time.sleep(2 ** attempt * 0.5)
                continue
            # re-raise the last exception after exhausting retries
            raise


def load_tickers(
    tickers: List[str],
    start_date: Union[str, date, datetime],
    end_date: Union[str, date, datetime],
    timespan: str = "day",
) -> pd.DataFrame:
    """Fetch data for multiple tickers and concatenate into a single DataFrame.

    Returns an empty DataFrame with the standard columns if no data is found.
    """
    frames: List[pd.DataFrame] = []
    for t in tickers:
        try:
            df = get_polygon_data(t, start_date, end_date, timespan=timespan)
        except Exception:
            # do not stop the whole load if one ticker fails; just skip it
            df = pd.DataFrame()
        if not df.empty:
            frames.append(df)

    if not frames:
        return pd.DataFrame(columns=DEFAULT_COLUMNS)

    data = pd.concat(frames, ignore_index=True)
    return data


__all__ = ["get_polygon_data", "load_tickers"]


def fetch_all(tickers: List[str], start_date: Union[str, date, datetime], end_date: Union[str, date, datetime], timespan: str = "day") -> pd.DataFrame:
    """Fetch data for multiple tickers and concatenate into a single DataFrame.

    This is a thin wrapper around `load_tickers` kept for API compatibility with
    the earlier `Simulator.fetch_all` method. Returns an empty DataFrame with
    the standard columns if no data is found.
    """
    return load_tickers(tickers, start_date, end_date, timespan=timespan)
