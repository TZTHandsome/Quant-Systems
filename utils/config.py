"""Central configuration for the baseline trading system."""

DEFAULT_CONFIG = {
    "initial_capital": 100000.0,
    "transaction_cost_bps": 10.0,
    "slippage_bps": 5.0,
    "fast_ma_window": 20,
    "slow_ma_window": 50,
    "momentum_window": 10,
    "volatility_window": 20,
    "risk_fraction": 1.0,
}
