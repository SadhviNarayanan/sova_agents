import argparse
import importlib
import json
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from model import (
    HybridConfig,
    TEMPORAL_TARGETS,
    add_time_series_features,
    apply_percentile_scaler,
    preprocess_dataset,
)


def compute_temporal_raw(df: pd.DataFrame) -> np.ndarray:
    components: List[np.ndarray] = []
    for target in TEMPORAL_TARGETS:
        pred_col = f"{target}_roll14_mean"
        if pred_col not in df.columns:
            grouped = df.groupby("user_id")[target]
            preds = grouped.transform(lambda s: s.shift(1).rolling(14, min_periods=5).mean())
        else:
            preds = df[pred_col]
        preds = preds.fillna(df.groupby("user_id")[target].transform("median")).fillna(df[target].median())
        residual = np.abs(df[target].to_numpy(dtype=float) - preds.to_numpy(dtype=float))
        scale = df.groupby("user_id")[target].transform("std").replace(0, np.nan).fillna(1.0)
        components.append(residual / scale.to_numpy(dtype=float))
    return np.vstack(components).mean(axis=0)


def compute_population_raw(df: pd.DataFrame, population_model: Dict[str, Any]) -> np.ndarray:
    feature_cols = list(dict.fromkeys(population_model["feature_columns"]))
    cohort_cols = population_model["cohort_columns"]
    cohort_stats = population_model["cohort_stats"]
    global_median = pd.Series(population_model["global_median"])
    global_mad = pd.Series(population_model["global_mad"]).replace(0, 1e-6)

    dfx = df.copy()
    dfx["age_bucket"] = (dfx["age"] // 10) * 10
    for col in feature_cols:
        if col not in dfx.columns:
            dfx[col] = global_median.get(col, 0.0)
        dfx[col] = dfx[col].replace([np.inf, -np.inf], np.nan).fillna(global_median.get(col, 0.0))

    pop_raw = np.zeros(len(dfx), dtype=float)
    for cohort_key, grp in dfx.groupby(cohort_cols, dropna=False):
        idx = grp.index
        key = str(cohort_key)
        if key in cohort_stats:
            med = pd.Series(cohort_stats[key]["median"]).reindex(feature_cols)
            mad = pd.Series(cohort_stats[key]["mad"]).reindex(feature_cols).replace(0, 1e-6)
        else:
            med = global_median.reindex(feature_cols)
            mad = global_mad.reindex(feature_cols)
        med = med.fillna(global_median.reindex(feature_cols))
        mad = mad.fillna(global_mad.reindex(feature_cols)).replace(0, 1e-6)
        z = (dfx.loc[idx, feature_cols] - med).abs() / mad
        pop_raw[idx] = z.mean(axis=1).to_numpy(dtype=float)
    return pop_raw


def to_severity_level(hybrid_score: np.ndarray, temporal_pct: np.ndarray, population_pct: np.ndarray) -> np.ndarray:
    """
    Map model risk to operational severity levels:
    0 = benign, record only
    1 = low concern
    2 = moderate concern
    3 = high concern
    4 = critical (call 911 now)
    """
    levels = np.zeros_like(hybrid_score, dtype=int)

    levels[hybrid_score >= 0.60] = 1
    levels[hybrid_score >= 0.80] = 2
    levels[hybrid_score >= 0.93] = 3
    levels[hybrid_score >= 0.985] = 4

    # Escalate directly to level 4 when both towers are extreme.
    extreme_towers = (temporal_pct >= 0.98) & (population_pct >= 0.98)
    levels[extreme_towers] = 4
    return levels


def _fallback_rule_based_level(snapshot: Dict[str, Any]) -> int:
    """
    Heuristic fallback used when the model bundle or required fields are missing.
    Returns a level in [0, 4], matching heartbeat escalation routing.
    """
    recovery = float(snapshot.get("recovery_score", 50) or 50)
    hrv = float(snapshot.get("hrv", 0) or 0)
    hrv_baseline = float(snapshot.get("hrv_baseline", hrv or 1) or (hrv or 1))
    rhr = float(snapshot.get("resting_heart_rate", 0) or 0)
    rhr_baseline = float(snapshot.get("rhr_baseline", rhr) or rhr)
    respiratory_rate = float(snapshot.get("respiratory_rate", 0) or 0)
    skin_temp_dev = float(snapshot.get("skin_temp_deviation", 0) or 0)
    sleep_perf = float(snapshot.get("sleep_performance", 75) or 75)
    day_strain = float(snapshot.get("day_strain", 10) or 10)

    points = 0
    if recovery < 20:
        points += 3
    elif recovery < 35:
        points += 2
    elif recovery < 50:
        points += 1

    if hrv_baseline > 0:
        hrv_ratio = hrv / hrv_baseline
        if hrv_ratio < 0.55:
            points += 3
        elif hrv_ratio < 0.70:
            points += 2
        elif hrv_ratio < 0.85:
            points += 1

    rhr_delta = rhr - rhr_baseline
    if rhr_delta >= 20:
        points += 3
    elif rhr_delta >= 12:
        points += 2
    elif rhr_delta >= 7:
        points += 1

    if respiratory_rate >= 20:
        points += 1
    if abs(skin_temp_dev) >= 1.2:
        points += 2
    elif abs(skin_temp_dev) >= 0.8:
        points += 1
    if sleep_perf < 55:
        points += 1
    if day_strain >= 18:
        points += 1

    if recovery < 20 and rhr_delta >= 20:
        return 4

    if points >= 9:
        return 4
    if points >= 6:
        return 3
    if points >= 4:
        return 2
    if points >= 2:
        return 1
    return 0


def infer_anomaly_level(
    snapshot: Dict[str, Any],
    history_rows: List[Dict[str, Any]] | None = None,
    model_path: Path | None = None,
) -> int:
    """
    In-process anomaly scoring API for claw.
    Input: latest WHOOP-like snapshot + optional same-user history rows.
    Output: integer anomaly level in [0, 4].
    """
    history_rows = history_rows or []
    if model_path is None:
        model_path = Path(__file__).resolve().parent.parent / "Anomaly_Detection_mod" / "artifacts" / "hybrid_detector_model.pkl"

    user_id = str(snapshot.get("user_id", "unknown"))
    rows = [row for row in history_rows if str(row.get("user_id", user_id)) == user_id]
    rows.append(snapshot)
    df = pd.DataFrame(rows)

    if "date" not in df.columns:
        return _fallback_rule_based_level(snapshot)
    if "user_id" not in df.columns:
        df["user_id"] = user_id

    try:
        with model_path.open("rb") as f:
            bundle = pickle.load(f)
        cfg = HybridConfig(**bundle["config"])
        dfx = preprocess_dataset(df, cfg)
        dfx, _, _, _ = add_time_series_features(dfx)

        temporal_raw = compute_temporal_raw(dfx)
        population_raw = compute_population_raw(dfx, bundle["population_model"])

        temporal_pct = apply_percentile_scaler(temporal_raw, bundle["temporal_scaler"])
        population_pct = apply_percentile_scaler(population_raw, bundle["population_scaler"])
        hybrid_score = cfg.temporal_weight * temporal_pct + cfg.population_weight * population_pct
        severity_level = to_severity_level(hybrid_score, temporal_pct, population_pct)
        return int(severity_level[-1])
    except Exception:
        return _fallback_rule_based_level(snapshot)


def run_inference(
    csv_path: Path,
    model_path: Path,
    output_dir: Path,
    predictions_name: str,
    user_id: str,
    output_daily_scores: bool,
) -> Dict[str, Any]:
    with model_path.open("rb") as f:
        bundle = pickle.load(f)

    cfg = HybridConfig(**bundle["config"])
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(csv_path)
    raw = raw[raw["user_id"] == user_id].copy()
    if raw.empty:
        raise ValueError(f"No rows found for user_id={user_id}")

    df = preprocess_dataset(raw, cfg)
    df, _, _, _ = add_time_series_features(df)

    temporal_raw = compute_temporal_raw(df)
    population_raw = compute_population_raw(df, bundle["population_model"])

    temporal_pct = apply_percentile_scaler(temporal_raw, bundle["temporal_scaler"])
    population_pct = apply_percentile_scaler(population_raw, bundle["population_scaler"])
    hybrid_score = cfg.temporal_weight * temporal_pct + cfg.population_weight * population_pct
    severity_level = to_severity_level(hybrid_score, temporal_pct, population_pct)

    df["temporal_raw"] = temporal_raw
    df["population_raw"] = population_raw
    df["temporal_pct"] = temporal_pct
    df["population_pct"] = population_pct
    df["hybrid_score"] = hybrid_score
    df["severity_level"] = severity_level

    predictions_path = None
    if output_daily_scores:
        predictions_path = output_dir / predictions_name
        df[["user_id", "date", "severity_level"]].to_csv(predictions_path, index=False)

    latest_row = df.sort_values("date").iloc[-1]

    metrics = {
        "user_id": user_id,
        "rows_scored": int(len(df)),
        "level_0": int((df["severity_level"] == 0).sum()),
        "level_1": int((df["severity_level"] == 1).sum()),
        "level_2": int((df["severity_level"] == 2).sum()),
        "level_3": int((df["severity_level"] == 3).sum()),
        "level_4": int((df["severity_level"] == 4).sum()),
    }
    metrics_path = output_dir / "inference_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return {
        "model_path": str(model_path),
        "predictions_path": str(predictions_path) if predictions_path is not None else None,
        "metrics_path": str(metrics_path),
        "final_user_score": {
            "user_id": user_id,
            "date": str(pd.to_datetime(latest_row["date"]).date()),
            "severity_level": int(latest_row["severity_level"]),
        },
        "metrics": metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Inference-only scoring for hybrid wearable anomaly model.")
    parser.add_argument("--csv_path", required=True, help="Path to input CSV to score")
    parser.add_argument(
        "--model_path",
        default="./artifacts/hybrid_detector_model.pkl",
        help="Path to trained model bundle (.pkl)",
    )
    parser.add_argument(
        "--output_dir",
        default="./artifacts",
        help="Directory to save inference outputs",
    )
    parser.add_argument(
        "--predictions_name",
        default="hybrid_inference_predictions.csv",
        help="Output CSV filename for per-day scored rows",
    )
    parser.add_argument(
        "--user_id",
        required=True,
        help="Single user_id to score",
    )
    parser.add_argument(
        "--output_daily_scores",
        action="store_true",
        help="If set, also write per-day severity levels for this user",
    )
    args = parser.parse_args()

    result = run_inference(
        csv_path=Path(args.csv_path),
        model_path=Path(args.model_path),
        output_dir=Path(args.output_dir),
        predictions_name=args.predictions_name,
        user_id=args.user_id,
        output_daily_scores=args.output_daily_scores,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
