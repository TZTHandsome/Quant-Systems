from __future__ import annotations

from itertools import combinations
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def compute_pbo(
    strategy_returns: dict[str, pd.DataFrame | pd.Series],
    *,
    n_groups: int = 6,
    test_group_count: int = 2,
    embargo_groups: int = 0,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    matrix = _build_return_matrix(strategy_returns)
    if matrix.empty or matrix.shape[1] < 2:
        return _empty_summary(n_groups, test_group_count, embargo_groups), pd.DataFrame(), pd.DataFrame()

    grouped = _assign_groups(matrix.index, n_groups=n_groups)
    group_scores = _compute_group_scores(matrix, grouped)
    split_rows: list[dict[str, Any]] = []

    group_ids = sorted(group_scores["group"].dropna().astype(int).unique().tolist())
    if len(group_ids) < max(test_group_count + 1, 3):
        return _empty_summary(n_groups, test_group_count, embargo_groups), group_scores, pd.DataFrame()

    score_columns = [column for column in group_scores.columns if column not in {"group", "date_start", "date_end", "rows"}]
    score_table = group_scores.set_index("group")[score_columns].astype(float)
    for test_groups in combinations(group_ids, test_group_count):
        train_groups = _resolve_train_groups(group_ids, list(test_groups), embargo_groups)
        if not train_groups:
            continue
        insample_scores = score_table.loc[train_groups].mean(axis=0)
        oos_scores = score_table.loc[list(test_groups)].mean(axis=0)
        if insample_scores.empty or oos_scores.empty:
            continue
        best_strategy = str(insample_scores.idxmax())
        best_is = float(insample_scores.loc[best_strategy])
        best_oos = float(oos_scores.loc[best_strategy])
        oos_percentile = float(oos_scores.rank(pct=True, method="average").loc[best_strategy])
        split_rows.append(
            {
                "test_groups": ",".join(str(item) for item in test_groups),
                "train_groups": ",".join(str(item) for item in train_groups),
                "best_strategy": best_strategy,
                "best_is_score": best_is,
                "best_oos_score": best_oos,
                "oos_percentile": oos_percentile,
                "overfit_flag": 1 if oos_percentile < 0.5 else 0,
            }
        )

    split_df = pd.DataFrame(split_rows)
    if split_df.empty:
        return _empty_summary(n_groups, test_group_count, embargo_groups), group_scores, split_df

    summary = {
        "n_strategies": int(len(score_columns)),
        "n_groups": int(n_groups),
        "test_group_count": int(test_group_count),
        "embargo_groups": int(embargo_groups),
        "n_splits": int(len(split_df)),
        "pbo": float(split_df["overfit_flag"].mean()),
        "mean_oos_percentile": float(split_df["oos_percentile"].mean()),
        "median_oos_percentile": float(split_df["oos_percentile"].median()),
        "mean_best_is_score": float(split_df["best_is_score"].mean()),
        "mean_best_oos_score": float(split_df["best_oos_score"].mean()),
    }
    return summary, group_scores, split_df


def save_pbo_outputs(
    output_dir: str | Path,
    summary: dict[str, Any],
    group_scores: pd.DataFrame,
    split_details: pd.DataFrame,
) -> None:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    (root / "pbo_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if not group_scores.empty:
        group_scores.to_csv(root / "pbo_strategy_group_scores.csv", index=False)
    if not split_details.empty:
        split_details.to_csv(root / "pbo_split_details.csv", index=False)


def _build_return_matrix(strategy_returns: dict[str, pd.DataFrame | pd.Series]) -> pd.DataFrame:
    columns: list[pd.Series] = []
    for name, payload in strategy_returns.items():
        if isinstance(payload, pd.Series):
            series = payload.copy()
            series.index = pd.to_datetime(series.index, utc=False)
            columns.append(series.rename(str(name)).astype(float))
            continue
        frame = payload.copy()
        if "date" not in frame.columns:
            continue
        return_column = "net_return" if "net_return" in frame.columns else "return"
        if return_column not in frame.columns:
            continue
        series = pd.Series(
            pd.to_numeric(frame[return_column], errors="coerce").to_numpy(),
            index=pd.to_datetime(frame["date"], utc=False),
            name=str(name),
        )
        columns.append(series.astype(float))
    if not columns:
        return pd.DataFrame()
    return pd.concat(columns, axis=1).sort_index().fillna(0.0)


def _assign_groups(index: pd.Index, *, n_groups: int) -> pd.DataFrame:
    dates = pd.to_datetime(pd.Index(index).unique(), utc=False).sort_values()
    split_dates = np.array_split(dates.to_numpy(), min(n_groups, len(dates)))
    rows: list[dict[str, Any]] = []
    for group_id, values in enumerate(split_dates):
        if len(values) == 0:
            continue
        rows.append(
            {
                "group": int(group_id),
                "date_start": pd.Timestamp(values[0]),
                "date_end": pd.Timestamp(values[-1]),
                "dates": {pd.Timestamp(item) for item in values},
            }
        )
    return pd.DataFrame(rows)


def _compute_group_scores(matrix: pd.DataFrame, grouped: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, group in grouped.iterrows():
        dates = group["dates"]
        subset = matrix.loc[matrix.index.isin(dates)].copy()
        row: dict[str, Any] = {
            "group": int(group["group"]),
            "date_start": group["date_start"],
            "date_end": group["date_end"],
            "rows": int(len(subset)),
        }
        for column in subset.columns:
            row[column] = _sharpe(subset[column])
        rows.append(row)
    return pd.DataFrame(rows)


def _sharpe(series: pd.Series) -> float:
    clean = pd.to_numeric(series, errors="coerce").dropna().astype(float)
    if clean.empty:
        return 0.0
    std = float(clean.std())
    if std <= 1e-12:
        return 0.0
    return float(clean.mean() / std * np.sqrt(252.0))


def _resolve_train_groups(group_ids: list[int], test_groups: list[int], embargo_groups: int) -> list[int]:
    blocked = set(test_groups)
    if embargo_groups > 0:
        for group in test_groups:
            for offset in range(1, embargo_groups + 1):
                blocked.add(group - offset)
                blocked.add(group + offset)
    return [group for group in group_ids if group not in blocked]


def _empty_summary(n_groups: int, test_group_count: int, embargo_groups: int) -> dict[str, Any]:
    return {
        "n_strategies": 0,
        "n_groups": int(n_groups),
        "test_group_count": int(test_group_count),
        "embargo_groups": int(embargo_groups),
        "n_splits": 0,
        "pbo": 0.0,
        "mean_oos_percentile": 0.0,
        "median_oos_percentile": 0.0,
        "mean_best_is_score": 0.0,
        "mean_best_oos_score": 0.0,
    }
