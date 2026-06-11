from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from alpha_mining.config import AlphaMiningConfig, SelectedFactor
from alpha_mining.portfolio_construction import (
    combine_factor_columns,
    compute_signal_volatility,
    latest_weights_from_snapshot,
)
from alpha_mining.regime import build_regime_frame
from alpha_mining.registry import FactorRegistry


@dataclass
class PaperBroker:
    cash: float = 1.0
    positions: dict[str, float] = field(default_factory=dict)
    trade_log: list[dict[str, Any]] = field(default_factory=list)
    equity_curve: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class PaperTradingResult:
    equity_curve: pd.DataFrame
    daily_metrics: pd.DataFrame
    trades: pd.DataFrame
    positions: pd.DataFrame
    broker: PaperBroker


@dataclass
class PaperRuntimeInputs:
    score_series: pd.Series
    volatility_series: pd.Series
    regime_by_date: pd.Series | None
    index_by_date: dict[pd.Timestamp, pd.Index]
    open_by_date: dict[pd.Timestamp, dict[str, float]]
    close_by_date: dict[pd.Timestamp, dict[str, float]]
    name_by_date: dict[pd.Timestamp, dict[str, str]]


def split_crypto_research_and_paper_panel(
    panel: pd.DataFrame,
    paper_trading_bars: int = 90,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = panel.copy()
    ordered["date"] = pd.to_datetime(ordered["date"], utc=False)
    ordered = ordered.sort_values(["date", "symbol"], kind="mergesort").reset_index(drop=True)
    unique_dates = sorted(ordered["date"].dropna().unique())
    if len(unique_dates) <= paper_trading_bars:
        raise ValueError("Not enough bars to split research and paper periods.")

    paper_dates = unique_dates[-paper_trading_bars:]
    paper_start = pd.Timestamp(paper_dates[0])
    research_panel = ordered.loc[ordered["date"] < paper_start].copy().reset_index(drop=True)
    paper_panel = ordered.loc[ordered["date"].isin(paper_dates)].copy().reset_index(drop=True)
    if research_panel.empty or paper_panel.empty:
        raise ValueError("Failed to build non-empty crypto research/paper panels.")
    return research_panel, paper_panel


def run_crypto_paper_trading(
    panel: pd.DataFrame,
    config: AlphaMiningConfig,
    initial_cash: float = 1.0,
    output_dir: str | Path = "outputs",
    paper_trading_bars: int = 90,
) -> PaperTradingResult:
    frozen_factors, metadata = load_frozen_factors(config)
    if not frozen_factors:
        raise ValueError("No frozen factors found. Crypto paper trading must load frozen factors only.")
    return run_crypto_paper_trading_with_factors(
        panel=panel,
        config=config,
        selected_factors=frozen_factors,
        initial_cash=initial_cash,
        output_dir=output_dir,
        paper_trading_bars=paper_trading_bars,
        training_metadata=metadata,
    )


def run_crypto_paper_trading_with_factors(
    panel: pd.DataFrame,
    config: AlphaMiningConfig,
    selected_factors: list[SelectedFactor],
    initial_cash: float = 1.0,
    output_dir: str | Path = "outputs",
    paper_trading_bars: int = 90,
    training_metadata: dict[str, Any] | None = None,
    paper_start_date: str | pd.Timestamp | None = None,
    paper_end_date: str | pd.Timestamp | None = None,
    progress_callback: Any | None = None,
) -> PaperTradingResult:
    if not selected_factors:
        raise ValueError("Crypto paper trading requires at least one selected factor.")

    ordered = panel.copy()
    ordered["date"] = pd.to_datetime(ordered["date"], utc=False)
    ordered = ordered.sort_values(["date", "symbol"], kind="mergesort").reset_index(drop=True)

    paper_dates = _resolve_paper_dates(
        ordered,
        paper_trading_bars=paper_trading_bars,
        paper_start_date=paper_start_date,
        paper_end_date=paper_end_date,
    )
    first_paper_date = pd.Timestamp(paper_dates[0])
    if progress_callback is not None:
        progress_callback(
            stage="prepare",
            current=0,
            total=max(len(paper_dates) - 1, 1),
            message=f"Preparing paper window {paper_dates[0]} -> {paper_dates[-1]}",
        )

    metadata = training_metadata or {}
    if metadata:
        training_end = pd.to_datetime(metadata.get("data_range", {}).get("date_max"), utc=False, errors="coerce")
        if pd.notna(training_end) and training_end >= first_paper_date:
            raise ValueError(
                "Frozen factor training range overlaps the crypto paper window. "
                f"Training end date {training_end} must be earlier than paper start date {first_paper_date}."
            )

    if progress_callback is not None:
        progress_callback(
            stage="prepare",
            current=0,
            total=max(len(paper_dates) - 1, 1),
            message="Precomputing factor values, volatility, and regime states",
        )
    runtime_inputs = _prepare_paper_runtime_inputs(ordered, selected_factors, config)

    broker = PaperBroker(cash=float(initial_cash))
    current_weights: dict[str, float] = {}
    equity = float(initial_cash)
    daily_rows: list[dict[str, Any]] = []
    position_rows: list[dict[str, Any]] = []

    for index in range(len(paper_dates) - 1):
        signal_date = pd.Timestamp(paper_dates[index])
        next_date = pd.Timestamp(paper_dates[index + 1])
        if progress_callback is not None:
            progress_callback(
                stage="paper_trading",
                current=index,
                total=max(len(paper_dates) - 1, 1),
                message=f"Processing {signal_date.date()} -> {next_date.date()}",
            )
        next_open = runtime_inputs.open_by_date.get(next_date)
        next_close = runtime_inputs.close_by_date.get(next_date)
        symbol_names = runtime_inputs.name_by_date.get(next_date, {})
        if not next_open or not next_close:
            continue

        target_weights = _compute_signal_weights(
            ordered=ordered,
            signal_date=signal_date,
            runtime_inputs=runtime_inputs,
            config=config,
            previous_weights=current_weights,
        )
        if not target_weights:
            continue

        equity_before = float(equity)
        new_weights, trades, turnover, cost = _rebalance_weights(
            signal_date=signal_date,
            execution_date=next_date,
            current_weights=current_weights,
            target_weights=target_weights,
            transaction_cost_bps=config.evaluation.transaction_cost_bps,
            slippage_bps=config.evaluation.slippage_bps,
            turnover_limit=config.portfolio.turnover_limit,
            symbol_names=symbol_names,
        )
        if not new_weights:
            continue

        gross_return = _compute_weighted_open_to_close_return(new_weights, next_open, next_close)
        net_return = gross_return - cost
        gross_pnl = equity_before * gross_return
        cost_amount = equity_before * cost
        net_pnl = equity_before * net_return
        equity *= 1.0 + net_return

        current_weights = new_weights
        broker.positions = dict(current_weights)
        broker.trade_log.extend(trades)
        broker.cash = equity
        broker.equity_curve.append({"date": str(next_date), "equity": float(equity)})
        daily_rows.append(
            {
                "signal_date": str(signal_date),
                "execution_date": str(next_date),
                "date": str(next_date),
                "equity_before": float(equity_before),
                "equity_after": float(equity),
                "gross_pnl": float(gross_pnl),
                "cost_amount": float(cost_amount),
                "net_pnl": float(net_pnl),
                "return": float(net_return),
                "gross_return": float(gross_return),
                "turnover": float(turnover),
                "cost": float(cost),
                "num_positions": int(len(current_weights)),
            }
        )
        position_rows.extend(
            _build_position_rows(
                signal_date=signal_date,
                execution_date=next_date,
                weights=current_weights,
                entry_open=next_open,
                next_close=next_close,
                symbol_names=symbol_names,
                equity_before=equity_before,
            )
        )

    trades_df = pd.DataFrame(broker.trade_log)
    equity_df = pd.DataFrame(broker.equity_curve)
    daily_metrics_df = pd.DataFrame(daily_rows)
    positions_df = pd.DataFrame(position_rows)
    if progress_callback is not None:
        progress_callback(
            stage="finalize",
            current=max(len(paper_dates) - 1, 1),
            total=max(len(paper_dates) - 1, 1),
            message="Saving paper trading artifacts",
        )
    _save_logs(output_dir, trades_df, equity_df, daily_metrics_df, positions_df)
    if progress_callback is not None:
        progress_callback(
            stage="completed",
            current=max(len(paper_dates) - 1, 1),
            total=max(len(paper_dates) - 1, 1),
            message="Paper trading complete",
        )
    return PaperTradingResult(
        equity_curve=equity_df,
        daily_metrics=daily_metrics_df,
        trades=trades_df,
        positions=positions_df,
        broker=broker,
    )


def _resolve_paper_dates(
    ordered: pd.DataFrame,
    *,
    paper_trading_bars: int,
    paper_start_date: str | pd.Timestamp | None,
    paper_end_date: str | pd.Timestamp | None,
) -> list[pd.Timestamp]:
    unique_dates = sorted(pd.to_datetime(ordered["date"], utc=False).dropna().unique())
    if paper_start_date is None and paper_end_date is None:
        if len(unique_dates) <= paper_trading_bars:
            raise ValueError("Not enough bars to split research and paper periods.")
        return [pd.Timestamp(date) for date in unique_dates[-paper_trading_bars:]]

    start = pd.Timestamp(paper_start_date) if paper_start_date is not None else pd.Timestamp(unique_dates[-paper_trading_bars])
    end = pd.Timestamp(paper_end_date) if paper_end_date is not None else pd.Timestamp(unique_dates[-1])
    resolved = [pd.Timestamp(date) for date in unique_dates if pd.Timestamp(date) >= start and pd.Timestamp(date) <= end]
    if len(resolved) < 2:
        raise ValueError("Paper trading window must contain at least 2 bars.")
    return resolved


def load_frozen_factors(config: AlphaMiningConfig) -> tuple[list[SelectedFactor], dict[str, Any]]:
    registry = FactorRegistry(config.registry_dir())
    return registry.load(config), registry.load_metadata(config)


def _compute_signal_weights(
    ordered: pd.DataFrame,
    signal_date: pd.Timestamp,
    runtime_inputs: PaperRuntimeInputs,
    config: AlphaMiningConfig,
    previous_weights: dict[str, float],
) -> dict[str, float]:
    latest_index = runtime_inputs.index_by_date.get(pd.Timestamp(signal_date))
    if latest_index is None or len(latest_index) == 0:
        return {}

    return latest_weights_from_snapshot(
        symbols=ordered.loc[latest_index, "symbol"].astype(str),
        score=runtime_inputs.score_series.loc[latest_index].astype(float),
        volatility=runtime_inputs.volatility_series.loc[latest_index].astype(float),
        latest_date=signal_date,
        weighting_scheme=config.portfolio.weighting_scheme,
        long_quantile=config.evaluation.long_quantile,
        short_quantile=config.evaluation.short_quantile,
        position_limit=config.portfolio.position_limit,
        gross_leverage=config.portfolio.gross_leverage,
        signal_clip=config.portfolio.signal_clip,
        smoothing=config.portfolio.smoothing,
        market_neutral=config.portfolio.market_neutral,
        regime_by_date=runtime_inputs.regime_by_date,
        benchmark_follow_enabled=config.portfolio.benchmark_follow_enabled,
        benchmark_follow_btc_symbol=config.portfolio.benchmark_follow_btc_symbol,
        benchmark_follow_btc_weight=config.portfolio.benchmark_follow_btc_weight,
        regime_benchmark_blend=dict(config.portfolio.regime_benchmark_blend),
        previous_weights=previous_weights,
    )


def _prepare_paper_runtime_inputs(
    ordered: pd.DataFrame,
    frozen_factors: list[SelectedFactor],
    config: AlphaMiningConfig,
) -> PaperRuntimeInputs:
    factor_columns: dict[str, pd.Series] = {}
    for factor in frozen_factors:
        values = factor.node.evaluate(ordered).astype(float) * float(factor.direction)
        factor_columns[factor.expression] = values

    regime_by_date = None
    if config.regime.enabled:
        regime_frame = build_regime_frame(ordered, config.regime)
        if not regime_frame.empty:
            regime_by_date = regime_frame.set_index("date")["regime"]
            regime_by_date.index = pd.to_datetime(regime_by_date.index, utc=False)

    score_series = combine_factor_columns(
        factor_columns=factor_columns,
        selected_factors=frozen_factors,
        weight_scheme=config.portfolio.factor_weight_scheme,
        dates=ordered["date"],
        regime_by_date=regime_by_date,
        regime_config=config.regime,
    )
    volatility_series = compute_signal_volatility(ordered, window=config.portfolio.signal_vol_window)

    index_by_date: dict[pd.Timestamp, pd.Index] = {}
    open_by_date: dict[pd.Timestamp, dict[str, float]] = {}
    close_by_date: dict[pd.Timestamp, dict[str, float]] = {}
    name_by_date: dict[pd.Timestamp, dict[str, str]] = {}
    for date_value, day_index in ordered.groupby("date", sort=False).groups.items():
        timestamp = pd.Timestamp(date_value)
        day_slice = ordered.loc[day_index]
        index_by_date[timestamp] = pd.Index(day_index)
        open_by_date[timestamp] = dict(zip(day_slice["symbol"].astype(str), day_slice["open"].astype(float), strict=False))
        close_by_date[timestamp] = dict(zip(day_slice["symbol"].astype(str), day_slice["close"].astype(float), strict=False))
        if "name" in day_slice.columns:
            name_by_date[timestamp] = dict(zip(day_slice["symbol"].astype(str), day_slice["name"].astype(str), strict=False))
        else:
            name_by_date[timestamp] = {}

    return PaperRuntimeInputs(
        score_series=score_series,
        volatility_series=volatility_series,
        regime_by_date=regime_by_date,
        index_by_date=index_by_date,
        open_by_date=open_by_date,
        close_by_date=close_by_date,
        name_by_date=name_by_date,
    )


def _rebalance_weights(
    signal_date: pd.Timestamp,
    execution_date: pd.Timestamp,
    current_weights: dict[str, float],
    target_weights: dict[str, float],
    transaction_cost_bps: float,
    slippage_bps: float,
    turnover_limit: float,
    symbol_names: dict[str, str],
) -> tuple[dict[str, float], list[dict[str, Any]], float, float]:
    combined_symbols = sorted(set(current_weights) | set(target_weights))
    raw_turnover = sum(abs(target_weights.get(symbol, 0.0) - current_weights.get(symbol, 0.0)) for symbol in combined_symbols)
    turnover = min(float(raw_turnover), float(turnover_limit))
    scale = 1.0 if raw_turnover <= 0.0 else min(1.0, float(turnover_limit) / float(raw_turnover))

    new_weights: dict[str, float] = {}
    trade_rows: list[dict[str, Any]] = []
    for symbol in combined_symbols:
        old_weight = float(current_weights.get(symbol, 0.0))
        target_weight = float(target_weights.get(symbol, 0.0))
        adjusted_weight = old_weight + ((target_weight - old_weight) * scale)
        if abs(adjusted_weight) > 0.0:
            new_weights[symbol] = adjusted_weight
        delta_weight = adjusted_weight - old_weight
        if abs(delta_weight) <= 0.0:
            continue
        trade_rows.append(
            {
                "signal_date": str(signal_date),
                "execution_date": str(execution_date),
                "date": str(execution_date),
                "symbol": symbol,
                "name": symbol_names.get(symbol, symbol),
                "side": "buy" if delta_weight > 0 else "sell",
                "weight_before": old_weight,
                "target_weight": target_weight,
                "weight_after": adjusted_weight,
                "weight_delta": delta_weight,
            }
        )
    total_cost_bps = float(transaction_cost_bps) + float(slippage_bps)
    cost = turnover * (total_cost_bps / 10000.0)
    return new_weights, trade_rows, turnover, cost


def _compute_weighted_open_to_close_return(
    weights: dict[str, float],
    next_open: dict[str, float],
    next_close: dict[str, float],
) -> float:
    gross_return = 0.0
    for symbol, weight in weights.items():
        entry = float(next_open.get(symbol, float("nan")))
        exit_ = float(next_close.get(symbol, float("nan")))
        if entry <= 0.0 or exit_ <= 0.0:
            continue
        gross_return += float(weight) * ((exit_ / entry) - 1.0)
    return gross_return


def _build_position_rows(
    signal_date: pd.Timestamp,
    execution_date: pd.Timestamp,
    weights: dict[str, float],
    entry_open: dict[str, float],
    next_close: dict[str, float],
    symbol_names: dict[str, str],
    equity_before: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for symbol, weight in sorted(weights.items()):
        entry = float(entry_open.get(symbol, float("nan")))
        exit_ = float(next_close.get(symbol, float("nan")))
        if entry <= 0.0 or exit_ <= 0.0:
            continue
        notional = float(abs(weight) * equity_before)
        units = notional / entry if entry > 0 else 0.0
        gross_pnl = float(weight) * equity_before * ((exit_ / entry) - 1.0)
        rows.append(
            {
                "signal_date": str(signal_date),
                "execution_date": str(execution_date),
                "symbol": symbol,
                "name": symbol_names.get(symbol, symbol),
                "weight": float(weight),
                "entry_open": entry,
                "exit_close": exit_,
                "units_est": units,
                "gross_pnl": gross_pnl,
            }
        )
    return rows


def _save_logs(
    output_dir: str | Path,
    trades_df: pd.DataFrame,
    equity_df: pd.DataFrame,
    daily_metrics_df: pd.DataFrame,
    positions_df: pd.DataFrame,
) -> None:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    trades_df.to_csv(root / "paper_trades.csv", index=False)
    equity_df.to_csv(root / "paper_equity.csv", index=False)
    daily_metrics_df.to_csv(root / "paper_daily_metrics.csv", index=False)
    positions_df.to_csv(root / "paper_positions.csv", index=False)
