from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .config import RegimeConfig


@dataclass
class RegimeSnapshot:
    date: pd.Timestamp
    trend_score: float
    vol_score: float
    trend_state: str
    vol_state: str
    regime: str


class RegimeDetector:
    def __init__(
        self,
        alpha: float = 0.1,
        threshold: float = 0.2,
        trend_window: int = 20,
        volatility_window: int = 20,
        zscore_window: int = 252,
    ) -> None:
        self.alpha = float(alpha)
        self.threshold = float(threshold)
        self.trend_window = int(trend_window)
        self.volatility_window = int(volatility_window)
        self.zscore_window = int(zscore_window)
        self.history_: pd.Series = pd.Series(dtype=float)
        self.state_history_: pd.DataFrame = pd.DataFrame(
            columns=["trend_z", "vol_z", "trend_score", "vol_score", "trend_state", "vol_state", "regime"]
        )
        self._current_snapshot: RegimeSnapshot | None = None

    def fit(self, historical_data: pd.Series | pd.DataFrame) -> RegimeDetector:
        returns = _normalize_returns(historical_data)
        self.history_ = returns.astype(float).copy()
        features = self._compute_feature_frame(self.history_)
        self.state_history_ = self._smooth_and_classify(features)
        if not self.state_history_.empty:
            last_date = pd.Timestamp(self.state_history_.index[-1])
            last_row = self.state_history_.iloc[-1]
            self._current_snapshot = RegimeSnapshot(
                date=last_date,
                trend_score=float(last_row["trend_score"]),
                vol_score=float(last_row["vol_score"]),
                trend_state=str(last_row["trend_state"]),
                vol_state=str(last_row["vol_state"]),
                regime=str(last_row["regime"]),
            )
        return self

    def update(self, new_data_point: pd.Series | dict[str, Any] | tuple[Any, float]) -> str:
        if self.history_.empty:
            raise ValueError("RegimeDetector.update() requires fit() to be called first.")

        date, value = _normalize_single_point(new_data_point)
        self.history_.loc[pd.Timestamp(date)] = float(value)
        self.history_ = self.history_.sort_index()

        lookback = self.zscore_window + max(self.trend_window, self.volatility_window) + 5
        recent_history = self.history_.iloc[-lookback:]
        recent_features = self._compute_feature_frame(recent_history)
        if recent_features.empty:
            return self.get_current_regime()

        latest_features = recent_features.iloc[-1]
        previous_snapshot = self._current_snapshot
        previous_trend_score = previous_snapshot.trend_score if previous_snapshot is not None else 0.0
        previous_vol_score = previous_snapshot.vol_score if previous_snapshot is not None else 0.0

        trend_score = (self.alpha * float(latest_features["trend_z"])) + ((1.0 - self.alpha) * previous_trend_score)
        vol_score = (self.alpha * float(latest_features["vol_z"])) + ((1.0 - self.alpha) * previous_vol_score)

        trend_state = _apply_hysteresis(
            score=trend_score,
            threshold=self.threshold,
            previous_state=previous_snapshot.trend_state if previous_snapshot is not None else "bull",
            positive_label="bull",
            negative_label="bear",
        )
        vol_state = _apply_hysteresis(
            score=vol_score,
            threshold=self.threshold,
            previous_state=previous_snapshot.vol_state if previous_snapshot is not None else "low_vol",
            positive_label="high_vol",
            negative_label="low_vol",
        )
        regime = f"{trend_state}_{vol_state}"

        snapshot = RegimeSnapshot(
            date=pd.Timestamp(recent_features.index[-1]),
            trend_score=float(trend_score),
            vol_score=float(vol_score),
            trend_state=trend_state,
            vol_state=vol_state,
            regime=regime,
        )
        self._current_snapshot = snapshot
        self.state_history_.loc[snapshot.date, :] = {
            "trend_z": float(latest_features["trend_z"]),
            "vol_z": float(latest_features["vol_z"]),
            "trend_score": snapshot.trend_score,
            "vol_score": snapshot.vol_score,
            "trend_state": snapshot.trend_state,
            "vol_state": snapshot.vol_state,
            "regime": snapshot.regime,
        }
        return regime

    def get_current_regime(self) -> str:
        if self._current_snapshot is None:
            return "bull_low_vol"
        return self._current_snapshot.regime

    def get_state_history(self) -> pd.DataFrame:
        if self.state_history_.empty:
            return self.state_history_.copy()
        history = self.state_history_.copy()
        history.index.name = "date"
        return history.reset_index()

    def _compute_feature_frame(self, returns: pd.Series) -> pd.DataFrame:
        clean = returns.astype(float).sort_index()
        trend_signal = (1.0 + clean).rolling(self.trend_window, min_periods=self.trend_window).apply(
            np.prod,
            raw=True,
        ) - 1.0
        volatility_signal = clean.rolling(
            self.volatility_window,
            min_periods=self.volatility_window,
        ).std()

        min_zscore_periods = min(self.zscore_window, max(60, self.zscore_window // 4))
        trend_mean = trend_signal.rolling(self.zscore_window, min_periods=min_zscore_periods).mean()
        trend_std = trend_signal.rolling(self.zscore_window, min_periods=min_zscore_periods).std()
        vol_mean = volatility_signal.rolling(self.zscore_window, min_periods=min_zscore_periods).mean()
        vol_std = volatility_signal.rolling(self.zscore_window, min_periods=min_zscore_periods).std()

        trend_z = ((trend_signal - trend_mean) / trend_std.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
        vol_z = ((volatility_signal - vol_mean) / vol_std.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
        features = pd.DataFrame(
            {
                "trend_20d": trend_signal,
                "vol_20d": volatility_signal,
                "trend_z": trend_z.fillna(0.0),
                "vol_z": vol_z.fillna(0.0),
            }
        )
        return features

    def _smooth_and_classify(self, features: pd.DataFrame) -> pd.DataFrame:
        if features.empty:
            return pd.DataFrame(
                columns=["trend_z", "vol_z", "trend_score", "vol_score", "trend_state", "vol_state", "regime"]
            )

        rows: list[dict[str, Any]] = []
        previous_trend_score = 0.0
        previous_vol_score = 0.0
        previous_trend_state = "bull"
        previous_vol_state = "low_vol"

        for date, row in features.iterrows():
            trend_score = (self.alpha * float(row["trend_z"])) + ((1.0 - self.alpha) * previous_trend_score)
            vol_score = (self.alpha * float(row["vol_z"])) + ((1.0 - self.alpha) * previous_vol_score)
            trend_state = _apply_hysteresis(
                score=trend_score,
                threshold=self.threshold,
                previous_state=previous_trend_state,
                positive_label="bull",
                negative_label="bear",
            )
            vol_state = _apply_hysteresis(
                score=vol_score,
                threshold=self.threshold,
                previous_state=previous_vol_state,
                positive_label="high_vol",
                negative_label="low_vol",
            )
            rows.append(
                {
                    "date": pd.Timestamp(date),
                    "trend_z": float(row["trend_z"]),
                    "vol_z": float(row["vol_z"]),
                    "trend_score": float(trend_score),
                    "vol_score": float(vol_score),
                    "trend_state": trend_state,
                    "vol_state": vol_state,
                    "regime": f"{trend_state}_{vol_state}",
                }
            )
            previous_trend_score = trend_score
            previous_vol_score = vol_score
            previous_trend_state = trend_state
            previous_vol_state = vol_state

        result = pd.DataFrame(rows).set_index("date")
        return result


def compute_market_proxy_returns(panel: pd.DataFrame, return_column: str = "daily_return") -> pd.Series:
    frame = panel.copy()
    frame["date"] = pd.to_datetime(frame["date"], utc=False)
    if return_column in frame.columns:
        returns = pd.to_numeric(frame[return_column], errors="coerce")
    else:
        close = pd.to_numeric(frame["close"], errors="coerce")
        returns = (
            pd.DataFrame(
                {
                    "symbol": frame["symbol"].astype(str),
                    "date": frame["date"],
                    "close": close,
                }
            )
            .sort_values(["symbol", "date"], kind="mergesort")
            .groupby("symbol", sort=False)["close"]
            .pct_change(fill_method=None)
        )
    grouped = pd.DataFrame({"date": frame["date"], "return": returns.astype(float)})
    market_returns = grouped.groupby("date", sort=False)["return"].mean().fillna(0.0)
    market_returns.index = pd.to_datetime(market_returns.index, utc=False)
    return market_returns.astype(float)


def build_regime_frame(panel: pd.DataFrame, config: RegimeConfig) -> pd.DataFrame:
    if not config.enabled or panel.empty:
        return pd.DataFrame(columns=["date", "trend_score", "vol_score", "trend_state", "vol_state", "regime"])
    market_returns = compute_market_proxy_returns(panel, return_column=config.return_column)
    detector = RegimeDetector(
        alpha=config.alpha,
        threshold=config.threshold,
        trend_window=config.trend_window,
        volatility_window=config.volatility_window,
        zscore_window=config.zscore_window,
    ).fit(market_returns)
    return detector.get_state_history()


def filter_regime_frame_by_dates(regime_frame: pd.DataFrame, dates: pd.Series | list[Any]) -> pd.DataFrame:
    if regime_frame.empty:
        return regime_frame.copy()
    date_index = pd.to_datetime(pd.Series(dates).dropna().unique(), utc=False)
    filtered = regime_frame.copy()
    filtered["date"] = pd.to_datetime(filtered["date"], utc=False)
    return filtered.loc[filtered["date"].isin(date_index)].copy().reset_index(drop=True)


def summarize_regime_frame(regime_frame: pd.DataFrame) -> dict[str, Any]:
    if regime_frame.empty:
        return {
            "days": 0,
            "dominant_regime": None,
            "switch_count": 0,
            "switch_rate": 0.0,
            "avg_trend_score": 0.0,
            "avg_vol_score": 0.0,
            "bull_fraction": 0.0,
            "bear_fraction": 0.0,
            "high_vol_fraction": 0.0,
            "low_vol_fraction": 0.0,
            "regime_counts": {},
            "regime_fractions": {},
        }

    frame = regime_frame.copy()
    frame["date"] = pd.to_datetime(frame["date"], utc=False)
    regimes = frame["regime"].astype(str)
    regime_counts = regimes.value_counts(dropna=False).to_dict()
    total = int(len(frame))
    switch_count = int((regimes != regimes.shift(1)).sum() - 1) if total > 1 else 0
    switch_count = max(switch_count, 0)
    trend_states = frame["trend_state"].astype(str)
    vol_states = frame["vol_state"].astype(str)
    return {
        "days": total,
        "dominant_regime": str(regimes.mode().iloc[0]) if not regimes.mode().empty else None,
        "switch_count": switch_count,
        "switch_rate": float(switch_count / max(total - 1, 1)),
        "avg_trend_score": float(pd.to_numeric(frame["trend_score"], errors="coerce").fillna(0.0).mean()),
        "avg_vol_score": float(pd.to_numeric(frame["vol_score"], errors="coerce").fillna(0.0).mean()),
        "bull_fraction": float((trend_states == "bull").mean()),
        "bear_fraction": float((trend_states == "bear").mean()),
        "high_vol_fraction": float((vol_states == "high_vol").mean()),
        "low_vol_fraction": float((vol_states == "low_vol").mean()),
        "regime_counts": {str(key): int(value) for key, value in regime_counts.items()},
        "regime_fractions": {str(key): float(value / total) for key, value in regime_counts.items()},
    }


def _normalize_returns(historical_data: pd.Series | pd.DataFrame) -> pd.Series:
    if isinstance(historical_data, pd.Series):
        returns = historical_data.copy()
        returns.index = pd.to_datetime(returns.index, utc=False)
        return returns.sort_index().astype(float)

    frame = historical_data.copy()
    if "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"], utc=False)
        frame = frame.sort_values("date")
        if "return" in frame.columns:
            return pd.Series(frame["return"].astype(float).to_numpy(), index=frame["date"])
        if "returns" in frame.columns:
            return pd.Series(frame["returns"].astype(float).to_numpy(), index=frame["date"])
    if frame.shape[1] == 1:
        series = frame.iloc[:, 0].astype(float).copy()
        series.index = pd.to_datetime(frame.index, utc=False)
        return series.sort_index()
    raise ValueError("historical_data must be a Series or a DataFrame with date/return columns.")


def _normalize_single_point(new_data_point: pd.Series | dict[str, Any] | tuple[Any, float]) -> tuple[pd.Timestamp, float]:
    if isinstance(new_data_point, tuple) and len(new_data_point) == 2:
        return pd.Timestamp(new_data_point[0]), float(new_data_point[1])
    if isinstance(new_data_point, dict):
        return pd.Timestamp(new_data_point["date"]), float(new_data_point["return"])
    if isinstance(new_data_point, pd.Series):
        if "date" in new_data_point.index and "return" in new_data_point.index:
            return pd.Timestamp(new_data_point["date"]), float(new_data_point["return"])
        if len(new_data_point) == 1:
            return pd.Timestamp(new_data_point.index[0]), float(new_data_point.iloc[0])
    raise ValueError("new_data_point must provide date and return information.")


def _apply_hysteresis(
    *,
    score: float,
    threshold: float,
    previous_state: str,
    positive_label: str,
    negative_label: str,
) -> str:
    if score > threshold:
        return positive_label
    if score < -threshold:
        return negative_label
    return previous_state
