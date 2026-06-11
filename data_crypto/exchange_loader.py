from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd


BINANCE_SPOT_KLINES_URL = "https://api.binance.com/api/v3/klines"
COINGECKO_MARKETS_URL = "https://api.coingecko.com/api/v3/coins/markets"
BINANCE_LIMIT = 1000

INTERVAL_TO_MS = {
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "2h": 7_200_000,
    "4h": 14_400_000,
    "6h": 21_600_000,
    "8h": 28_800_000,
    "12h": 43_200_000,
    "1d": 86_400_000,
    "3d": 259_200_000,
    "1w": 604_800_000,
}


@dataclass(frozen=True)
class BinanceDatasetMetadata:
    exchange: str
    interval: str
    start: str
    end: str
    symbol_count: int
    row_count: int
    benchmark_symbol: str
    generated_at_utc: str


def fetch_binance_klines(
    symbol: str,
    *,
    interval: str = "1d",
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    pause_seconds: float = 0.15,
    timeout_seconds: float = 30.0,
) -> pd.DataFrame:
    if interval not in INTERVAL_TO_MS:
        raise ValueError(f"Unsupported Binance interval: {interval}")

    start_ts = _to_milliseconds(start)
    end_ts = _to_milliseconds(end)
    if start_ts >= end_ts:
        raise ValueError("start must be earlier than end.")

    rows: list[list[Any]] = []
    cursor = start_ts
    step_ms = INTERVAL_TO_MS[interval]

    while cursor < end_ts:
        params = {
            "symbol": str(symbol).upper(),
            "interval": interval,
            "startTime": int(cursor),
            "endTime": int(end_ts),
            "limit": BINANCE_LIMIT,
        }
        batch = _binance_get_json(BINANCE_SPOT_KLINES_URL, params=params, timeout_seconds=timeout_seconds)
        if not isinstance(batch, list):
            raise ValueError(f"Unexpected Binance kline response for {symbol}: {type(batch)!r}")
        if not batch:
            break

        rows.extend(batch)
        last_open_time = int(batch[-1][0])
        next_cursor = last_open_time + step_ms
        if next_cursor <= cursor:
            break
        cursor = next_cursor
        time.sleep(pause_seconds)

        if len(batch) < BINANCE_LIMIT:
            break

    if not rows:
        raise ValueError(f"No Binance kline rows returned for {symbol}.")
    return _normalize_binance_klines(symbol, rows)


def fetch_binance_panel(
    universe: list[dict[str, str]],
    *,
    interval: str,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    pause_seconds: float = 0.15,
    timeout_seconds: float = 30.0,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for record in universe:
        symbol = str(record["symbol"]).upper()
        name = str(record.get("name", symbol))
        frame = fetch_binance_klines(
            symbol,
            interval=interval,
            start=start,
            end=end,
            pause_seconds=pause_seconds,
            timeout_seconds=timeout_seconds,
        )
        frame["name"] = name
        frames.append(frame)
    panel = pd.concat(frames, ignore_index=True)
    panel = panel.sort_values(["symbol", "date"], kind="mergesort").reset_index(drop=True)
    return panel


def save_binance_crypto_dataset(
    *,
    universe: list[dict[str, str]],
    output_dir: str | Path,
    interval: str,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    benchmark_symbol: str = "BTCUSDT",
    pause_seconds: float = 0.15,
    timeout_seconds: float = 30.0,
) -> dict[str, Path]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)

    enriched_universe = enrich_universe_with_market_cap_weights(universe, timeout_seconds=timeout_seconds)
    panel = fetch_binance_panel(
        enriched_universe,
        interval=interval,
        start=start,
        end=end,
        pause_seconds=pause_seconds,
        timeout_seconds=timeout_seconds,
    )
    universe_df = pd.DataFrame(enriched_universe)
    benchmark_frame = panel.loc[panel["symbol"].astype(str).str.upper() == str(benchmark_symbol).upper(), ["date", "close", "name"]].copy()
    if benchmark_frame.empty:
        raise ValueError(f"Benchmark symbol {benchmark_symbol} was not found in the Binance universe data.")
    benchmark_frame["name"] = str(benchmark_symbol).upper()
    equal_weight_benchmark = build_equal_weight_benchmark_from_panel(panel, benchmark_name="crypto30_equal_weight")
    market_cap_benchmark = build_market_cap_benchmark_from_panel(panel, enriched_universe, benchmark_name="crypto30_market_cap_weighted")

    panel_path = root / "panel.csv"
    universe_path = root / "universe.csv"
    benchmark_path = root / "benchmark.csv"
    equal_weight_path = root / "equal_weight_benchmark.csv"
    market_cap_path = root / "market_cap_benchmark.csv"
    metadata_path = root / "metadata.json"

    panel.to_csv(panel_path, index=False)
    universe_df.to_csv(universe_path, index=False)
    benchmark_frame.to_csv(benchmark_path, index=False)
    equal_weight_benchmark.to_csv(equal_weight_path, index=False)
    market_cap_benchmark.to_csv(market_cap_path, index=False)

    metadata = BinanceDatasetMetadata(
        exchange="binance_spot",
        interval=str(interval),
        start=str(pd.Timestamp(start)),
        end=str(pd.Timestamp(end)),
        symbol_count=len(universe),
        row_count=len(panel),
        benchmark_symbol=str(benchmark_symbol).upper(),
        generated_at_utc=str(pd.Timestamp.utcnow()),
    )
    metadata_path.write_text(json.dumps(asdict(metadata), ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "panel_csv": panel_path,
        "universe_csv": universe_path,
        "benchmark_csv": benchmark_path,
        "equal_weight_benchmark_csv": equal_weight_path,
        "market_cap_benchmark_csv": market_cap_path,
        "metadata_json": metadata_path,
    }


def _binance_get_json(
    url: str,
    *,
    params: dict[str, Any],
    timeout_seconds: float,
) -> Any:
    query = urllib.parse.urlencode(params)
    request_url = f"{url}?{query}"
    request = urllib.request.Request(
        request_url,
        headers={
            "User-Agent": "QuantSystemsCryptoLoader/1.0",
            "Accept": "application/json",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        payload = response.read().decode("utf-8")
    return json.loads(payload)


def _normalize_binance_klines(symbol: str, rows: list[list[Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(
        rows,
        columns=[
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_volume",
            "trade_count",
            "taker_buy_base_volume",
            "taker_buy_quote_volume",
            "ignore",
        ],
    )
    frame["date"] = pd.to_datetime(frame["open_time"], unit="ms", utc=True).dt.tz_localize(None)
    frame["symbol"] = str(symbol).upper()
    for column in ("open", "high", "low", "close", "volume", "quote_volume", "taker_buy_base_volume", "taker_buy_quote_volume"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["trade_count"] = pd.to_numeric(frame["trade_count"], errors="coerce")
    return frame[
        [
            "date",
            "symbol",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "quote_volume",
            "trade_count",
            "taker_buy_base_volume",
            "taker_buy_quote_volume",
        ]
    ].sort_values("date", kind="mergesort").reset_index(drop=True)


def enrich_universe_with_market_cap_weights(
    universe: list[dict[str, str]],
    *,
    timeout_seconds: float,
) -> list[dict[str, str]]:
    ids = [str(record.get("coingecko_id", "")).strip() for record in universe if record.get("coingecko_id")]
    if not ids:
        return [dict(record) for record in universe]
    payload = _coingecko_get_json(
        COINGECKO_MARKETS_URL,
        params={
            "vs_currency": "usd",
            "ids": ",".join(ids),
            "order": "market_cap_desc",
            "per_page": max(len(ids), 50),
            "page": 1,
            "sparkline": "false",
        },
        timeout_seconds=timeout_seconds,
    )
    market_cap_map = {
        str(item.get("id")): float(item.get("market_cap") or 0.0)
        for item in payload
        if item.get("id")
    }
    total_market_cap = sum(max(value, 0.0) for value in market_cap_map.values())
    enriched: list[dict[str, str]] = []
    for record in universe:
        updated = dict(record)
        cg_id = str(record.get("coingecko_id", "")).strip()
        market_cap = float(market_cap_map.get(cg_id, 0.0))
        updated["market_cap_usd"] = market_cap
        updated["market_cap_weight"] = 0.0 if total_market_cap <= 0.0 else market_cap / total_market_cap
        enriched.append(updated)
    return enriched


def build_equal_weight_benchmark_from_panel(panel: pd.DataFrame, benchmark_name: str) -> pd.DataFrame:
    ordered = panel.copy()
    ordered["date"] = pd.to_datetime(ordered["date"], utc=False)
    ordered = ordered.sort_values(["symbol", "date"], kind="mergesort").reset_index(drop=True)
    ordered["daily_return"] = ordered.groupby("symbol", sort=False)["close"].pct_change(fill_method=None)
    mean_return = ordered.groupby("date", sort=False)["daily_return"].mean().fillna(0.0)
    close = 100.0 * (1.0 + mean_return).cumprod()
    return pd.DataFrame({"date": mean_return.index, "close": close.to_numpy(dtype=float), "name": benchmark_name})


def build_market_cap_benchmark_from_panel(
    panel: pd.DataFrame,
    universe: list[dict[str, str]],
    benchmark_name: str,
) -> pd.DataFrame:
    weights = {
        str(record["symbol"]).upper(): float(record.get("market_cap_weight", 0.0))
        for record in universe
    }
    weight_total = sum(max(weight, 0.0) for weight in weights.values())
    if weight_total <= 0.0:
        return build_equal_weight_benchmark_from_panel(panel, benchmark_name)
    normalized = {symbol: max(weight, 0.0) / weight_total for symbol, weight in weights.items()}

    ordered = panel.copy()
    ordered["date"] = pd.to_datetime(ordered["date"], utc=False)
    ordered = ordered.sort_values(["symbol", "date"], kind="mergesort").reset_index(drop=True)
    ordered["daily_return"] = ordered.groupby("symbol", sort=False)["close"].pct_change(fill_method=None)
    ordered["cap_weight"] = ordered["symbol"].astype(str).str.upper().map(normalized).fillna(0.0)
    weighted_return = ordered.groupby("date", sort=False).apply(
        lambda day: float((day["daily_return"].fillna(0.0) * day["cap_weight"]).sum())
    )
    close = 100.0 * (1.0 + weighted_return).cumprod()
    return pd.DataFrame({"date": weighted_return.index, "close": close.to_numpy(dtype=float), "name": benchmark_name})


def _coingecko_get_json(
    url: str,
    *,
    params: dict[str, Any],
    timeout_seconds: float,
) -> Any:
    query = urllib.parse.urlencode(params)
    request_url = f"{url}?{query}"
    request = urllib.request.Request(
        request_url,
        headers={
            "User-Agent": "QuantSystemsCryptoLoader/1.0",
            "Accept": "application/json",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        payload = response.read().decode("utf-8")
    return json.loads(payload)


def _to_milliseconds(value: str | pd.Timestamp) -> int:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return int(timestamp.timestamp() * 1000)
