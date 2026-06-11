from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd
from backtesting import Strategy as BacktestingPyStrategy


class BaseStrategy(ABC):
    @abstractmethod
    def generate_signal(self, data: pd.DataFrame) -> int:
        """
        Input: historical data up to the current bar.
        Output: signal in {-1, 0, 1}.
        """

    def generate_signal_series(self, data: pd.DataFrame) -> pd.Series:
        signals: list[int] = []
        for end_idx in range(len(data)):
            history = data.iloc[: end_idx + 1]
            signals.append(int(self.generate_signal(history)))
        return pd.Series(signals, index=data.index, dtype="float64")


def build_backtestingpy_adapter(order_size: float):
    normalized_size = _normalize_order_size(order_size)

    class StrategyAdapter(BacktestingPyStrategy):
        def init(self) -> None:
            return

        def next(self) -> None:
            signal = float(self.data.Signal[-1])

            if signal > 0 and not self.position:
                self.buy(size=normalized_size)
            elif signal < 0 and not self.position:
                self.sell(size=normalized_size)
            elif signal == 0 and self.position:
                self.position.close()

    return StrategyAdapter


def _normalize_order_size(risk_fraction: float) -> float:
    if risk_fraction <= 0:
        raise ValueError("risk_fraction must be positive.")

    if risk_fraction <= 1.0:
        return min(risk_fraction, 0.9999)

    return float(int(risk_fraction))
