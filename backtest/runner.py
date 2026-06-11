from __future__ import annotations

import pandas as pd

from backtest.engine import BacktestEngine
from backtest.strategy import BaseStrategy


def run_backtest_pipeline(
    data: pd.DataFrame,
    strategy: BaseStrategy,
    config: dict[str, float],
    output_dir: str | None = None,
) -> tuple[pd.DataFrame, dict[str, float], dict[str, object]]:
    engine = BacktestEngine(
        initial_capital=config["initial_capital"],
        transaction_cost_bps=config["transaction_cost_bps"],
        slippage_bps=config["slippage_bps"],
        risk_fraction=config["risk_fraction"],
    )
    return engine.run(data, strategy, output_dir=output_dir)


def run_cross_sectional_backtest_pipeline(
    panel: pd.DataFrame,
    strategy: object,
    config: dict[str, float],
    output_dir: str | None = None,
) -> tuple[pd.DataFrame, dict[str, float], dict[str, object]]:
    engine = BacktestEngine(
        initial_capital=config["initial_capital"],
        transaction_cost_bps=config["transaction_cost_bps"],
        slippage_bps=config["slippage_bps"],
        risk_fraction=config["risk_fraction"],
    )
    return engine.run_cross_sectional(panel, strategy, output_dir=output_dir)
