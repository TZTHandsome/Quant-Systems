from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


def save_price_line_chart(
    df: pd.DataFrame,
    output_path: str | Path,
    title: str,
) -> None:
    chart_df = df.copy()
    chart_df["date"] = pd.to_datetime(chart_df["date"], utc=False)
    chart_df = chart_df.sort_values("date")

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(chart_df["date"], chart_df["close"], color="#1f77b4", linewidth=2.0)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Close Price")
    ax.grid(True, alpha=0.3)

    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
