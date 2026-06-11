from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from alpha_mining.run_crypto_workflow import CRYPTO_UNIVERSE_PRESETS
from data_crypto.exchange_loader import save_binance_crypto_dataset
from data_crypto.loader import load_crypto_universe_csv


DEFAULT_OUTPUT_DIR = Path("crypto_data") / "binance_crypto30_daily"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch Binance spot history for a crypto universe and save reusable local CSV files.")
    parser.add_argument("--universe-preset", default="crypto30", choices=sorted(CRYPTO_UNIVERSE_PRESETS.keys()), help="Named crypto universe preset.")
    parser.add_argument("--universe-csv", default=None, help="Optional universe CSV with columns symbol,name.")
    parser.add_argument("--interval", default="1d", help="Binance kline interval, for example 1d, 4h, 1h.")
    parser.add_argument("--start", default="2023-01-01", help="Inclusive UTC start date/time.")
    parser.add_argument("--end", default=str(pd.Timestamp.utcnow().floor("D")), help="Exclusive UTC end date/time.")
    parser.add_argument("--benchmark-symbol", default="BTCUSDT", help="Symbol to save as benchmark.csv.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory to write panel.csv, universe.csv, benchmark.csv.")
    parser.add_argument("--pause-seconds", type=float, default=0.15, help="Pause between Binance requests.")
    parser.add_argument("--timeout-seconds", type=float, default=30.0, help="HTTP timeout per Binance request.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.universe_csv:
        universe = load_crypto_universe_csv(args.universe_csv)
    else:
        universe = [dict(record) for record in CRYPTO_UNIVERSE_PRESETS[str(args.universe_preset)]]

    outputs = save_binance_crypto_dataset(
        universe=universe,
        output_dir=args.output_dir,
        interval=args.interval,
        start=args.start,
        end=args.end,
        benchmark_symbol=args.benchmark_symbol,
        pause_seconds=args.pause_seconds,
        timeout_seconds=args.timeout_seconds,
    )
    print("Binance crypto dataset saved.")
    for label, path in outputs.items():
        print(f"{label}: {Path(path).resolve()}")


if __name__ == "__main__":
    main()
