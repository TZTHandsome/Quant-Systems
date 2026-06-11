from __future__ import annotations

import pandas as pd


def to_backtesting_format(df: pd.DataFrame) -> pd.DataFrame:
    formatted = df.copy()
    formatted["date"] = pd.to_datetime(formatted["date"], utc=False)
    formatted = formatted.set_index("date").sort_index()

    rename_map = {
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
        "fast_ma": "FastMA",
        "slow_ma": "SlowMA",
        "momentum": "Momentum",
        "volatility": "Volatility",
        "trend_strength": "TrendStrength",
        "volume_zscore": "VolumeZScore",
    }
    existing = {source: target for source, target in rename_map.items() if source in formatted.columns}
    return formatted.rename(columns=existing)
