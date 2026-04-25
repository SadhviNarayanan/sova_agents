import argparse
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


BASE_FEATURES = [
    "hrv",
    "resting_heart_rate",
    "respiratory_rate",
    "skin_temp_deviation",
    "recovery_score",
    "sleep_hours",
    "sleep_efficiency",
    "day_strain",
]

CONTEXT_FEATURES = ["age", "gender", "fitness_level", "day_of_week", "workout_completed"]

TEMPORAL_TARGETS = ["hrv", "resting_heart_rate", "respiratory_rate", "skin_temp_deviation"]
ROLLING_WINDOWS = (7, 14)

RISK_DIRECTIONS = {
    "hrv": -1.0,
    "resting_heart_rate": +1.0,
    "respiratory_rate": +1.0,
    "skin_temp_deviation": +1.0,
    "recovery_score": -1.0,
    "sleep_hours": -1.0,
    "sleep_efficiency": -1.0,
    "day_strain": +1.0,
}


@dataclass
class HybridConfig:
    min_history_days: int = 14
    clip_quantile: float = 0.01
    temporal_weight: float = 0.6
    population_weight: float = 0.4
    high_anomaly_quantile: float = 0.95
    critical_anomaly_quantile: float = 0.99
    random_state: int = 42


def safe_rank_percentile(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(1, len(values) + 1)
    return ranks / max(len(values), 1)


def fit_percentile_scaler(values: np.ndarray) -> Dict[str, np.ndarray]:
    grid = np.linspace(0.0, 1.0, 1001)
    quantiles = np.quantile(values, grid)
    return {"grid": grid, "quantiles": quantiles}


def apply_percentile_scaler(values: np.ndarray, scaler: Dict[str, np.ndarray]) -> np.ndarray:
    return np.interp(values, scaler["quantiles"], scaler["grid"], left=0.0, right=1.0)


def preprocess_dataset(df: pd.DataFrame, cfg: HybridConfig) -> pd.DataFrame:
    dfx = df.copy()
    dfx["date"] = pd.to_datetime(dfx["date"])
    dfx = dfx.sort_values(["user_id", "date"]).reset_index(drop=True)

    required_cols = BASE_FEATURES + CONTEXT_FEATURES + ["user_id", "date"]
    missing = [col for col in required_cols if col not in dfx.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    dfx["gender"] = dfx["gender"].astype(str).str.lower()
    dfx["fitness_level"] = dfx["fitness_level"].astype(str).str.lower()
    dfx["day_of_week"] = dfx["day_of_week"].astype(str).str.lower()
    dfx["workout_completed"] = dfx["workout_completed"].astype(int)

    numeric_cols = sorted(set(BASE_FEATURES + ["age", "hrv_baseline", "rhr_baseline"]))
    numeric_cols = [c for c in numeric_cols if c in dfx.columns]
    for col in numeric_cols:
        dfx[col] = pd.to_numeric(dfx[col], errors="coerce")

    for col in numeric_cols:
        lower = dfx[col].quantile(cfg.clip_quantile)
        upper = dfx[col].quantile(1.0 - cfg.clip_quantile)
        dfx[col] = dfx[col].clip(lower=lower, upper=upper)

    dfx[numeric_cols] = dfx.groupby("user_id")[numeric_cols].transform(lambda g: g.ffill().bfill())
    medians = dfx[numeric_cols].median()
    dfx[numeric_cols] = dfx[numeric_cols].fillna(medians)

    if "hrv_baseline" in dfx.columns:
        dfx["hrv_to_baseline"] = dfx["hrv"] / dfx["hrv_baseline"].replace(0, np.nan)
    else:
        dfx["hrv_to_baseline"] = np.nan
    if "rhr_baseline" in dfx.columns:
        dfx["rhr_minus_baseline"] = dfx["resting_heart_rate"] - dfx["rhr_baseline"]
    else:
        dfx["rhr_minus_baseline"] = np.nan

    dfx["hrv_to_baseline"] = dfx["hrv_to_baseline"].replace([np.inf, -np.inf], np.nan).fillna(1.0)
    dfx["rhr_minus_baseline"] = dfx["rhr_minus_baseline"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return dfx


def add_time_series_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str], List[str]]:
    dfx = df.copy()
    temporal_lag_features: List[str] = []
    temporal_residual_features: List[str] = []

    for feat in BASE_FEATURES:
        dfx[f"{feat}_lag1"] = dfx.groupby("user_id")[feat].shift(1)
        dfx[f"{feat}_delta1"] = dfx[feat] - dfx[f"{feat}_lag1"]
        temporal_lag_features.extend([f"{feat}_lag1", f"{feat}_delta1"])

        for w in ROLLING_WINDOWS:
            mu_col = f"{feat}_roll{w}_mean"
            sd_col = f"{feat}_roll{w}_std"
            z_col = f"{feat}_roll{w}_z"
            grouped = dfx.groupby("user_id")[feat]
            dfx[mu_col] = grouped.transform(lambda s: s.shift(1).rolling(w, min_periods=3).mean())
            dfx[sd_col] = grouped.transform(lambda s: s.shift(1).rolling(w, min_periods=3).std())
            dfx[sd_col] = dfx[sd_col].replace(0, np.nan)
            dfx[z_col] = (dfx[feat] - dfx[mu_col]) / dfx[sd_col]
            temporal_residual_features.extend([mu_col, sd_col, z_col])

    dfx["history_days"] = dfx.groupby("user_id").cumcount()
    temporal_features = temporal_lag_features + temporal_residual_features + [
        "hrv_to_baseline",
        "rhr_minus_baseline",
        "history_days",
        "workout_completed",
    ]

    pop_features = BASE_FEATURES + [
        "age",
        "workout_completed",
        "hrv_to_baseline",
        "rhr_minus_baseline",
        "activity_duration_min",
        "activity_strain",
        "sleep_efficiency",
        "day_strain",
    ]
    pop_features = [c for c in pop_features if c in dfx.columns]
    pop_features = list(dict.fromkeys(pop_features))

    cat_cols = ["gender", "fitness_level", "day_of_week", "primary_sport", "activity_type", "workout_time_of_day"]
    cat_cols = [c for c in cat_cols if c in dfx.columns]

    for col in temporal_features + pop_features:
        if col in dfx.columns:
            dfx[col] = dfx[col].replace([np.inf, -np.inf], np.nan)

    return dfx, temporal_features, pop_features, cat_cols


def fit_temporal_models(df: pd.DataFrame, cfg: HybridConfig) -> Tuple[Dict[str, Any], np.ndarray]:
    models: Dict[str, Any] = {"method": "rolling_forecast_residual", "window": 14}
    dfx = df.copy()

    train_mask = dfx["history_days"] >= cfg.min_history_days
    if train_mask.sum() == 0:
        raise ValueError("Not enough history for temporal training. Increase data or lower min_history_days.")

    temporal_components = []
    for target in TEMPORAL_TARGETS:
        pred_col = f"{target}_roll14_mean"
        if pred_col not in dfx.columns:
            grouped = dfx.groupby("user_id")[target]
            dfx[pred_col] = grouped.transform(lambda s: s.shift(1).rolling(14, min_periods=5).mean())

        preds = dfx[pred_col].fillna(dfx.groupby("user_id")[target].transform("median")).fillna(
            dfx[target].median()
        )
        residual = np.abs(dfx[target].to_numpy(dtype=float) - preds.to_numpy(dtype=float))
        scale = dfx.groupby("user_id")[target].transform("std").replace(0, np.nan).fillna(1.0)
        temporal_components.append(residual / scale.to_numpy(dtype=float))

    temporal_raw = np.vstack(temporal_components).mean(axis=0)
    return models, temporal_raw


def fit_population_model(
    df: pd.DataFrame, pop_features: List[str], cat_cols: List[str], cfg: HybridConfig
) -> Tuple[Dict[str, Any], np.ndarray, List[str]]:
    dfx = df.copy()
    dfx["age_bucket"] = (dfx["age"] // 10) * 10
    cohort_cols = ["age_bucket", "gender", "fitness_level"]

    for col in pop_features:
        dfx[col] = dfx[col].replace([np.inf, -np.inf], np.nan)
        dfx[col] = dfx[col].fillna(dfx[col].median())

    cohort_stats: Dict[str, Any] = {}
    pop_raw = np.zeros(len(dfx), dtype=float)

    global_medians = dfx[pop_features].median()
    global_mads = (dfx[pop_features] - global_medians).abs().median().replace(0, 1e-6)

    for cohort_key, grp in dfx.groupby(cohort_cols, dropna=False):
        idx = grp.index
        if len(grp) < 30:
            med = global_medians
            mad = global_mads
        else:
            med = grp[pop_features].median()
            mad = (grp[pop_features] - med).abs().median().replace(0, 1e-6)

        z = (dfx.loc[idx, pop_features] - med).abs() / mad
        pop_raw[idx] = z.mean(axis=1).to_numpy(dtype=float)
        cohort_stats[str(cohort_key)] = {
            "median": med.to_dict(),
            "mad": mad.to_dict(),
            "count": int(len(grp)),
        }

    population_model = {
        "method": "cohort_robust_zscore",
        "cohort_columns": cohort_cols,
        "feature_columns": pop_features,
        "global_median": global_medians.to_dict(),
        "global_mad": global_mads.to_dict(),
        "cohort_stats": cohort_stats,
    }
    return population_model, pop_raw, pop_features


def assign_anomaly_labels(
    hybrid_score: np.ndarray, temporal_pct: np.ndarray, pop_pct: np.ndarray, cfg: HybridConfig
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    high_mask = hybrid_score >= cfg.high_anomaly_quantile
    critical_mask = (hybrid_score >= cfg.critical_anomaly_quantile) | (
        (temporal_pct >= 0.98) & (pop_pct >= 0.98)
    )

    labels = np.full(hybrid_score.shape, "normal", dtype=object)
    labels[high_mask] = "high"
    labels[critical_mask] = "critical"
    return labels, high_mask, critical_mask


def explain_triggered_signals(row: pd.Series) -> List[str]:
    triggered: List[str] = []
    for feat, direction in RISK_DIRECTIONS.items():
        z_col = f"{feat}_roll7_z"
        if z_col not in row or pd.isna(row[z_col]):
            continue
        z = float(row[z_col])
        if direction > 0 and z >= 1.5:
            triggered.append(feat)
        if direction < 0 and z <= -1.5:
            triggered.append(feat)
    return sorted(set(triggered))


def build_firebase_record(row: pd.Series) -> Dict[str, Any]:
    return {
        "user_id": row["user_id"],
        "date": str(pd.to_datetime(row["date"]).date()),
        "anomaly_detected": bool(row["anomaly_label"] in {"high", "critical"}),
        "severity": round(float(row["hybrid_score"]), 4),
        "anomaly_label": row["anomaly_label"],
        "triggered_signals": row["triggered_signals"],
        "hybrid_components": {
            "temporal_percentile": round(float(row["temporal_pct"]), 4),
            "population_percentile": round(float(row["population_pct"]), 4),
            "temporal_raw": round(float(row["temporal_raw"]), 4),
            "population_raw": round(float(row["population_raw"]), 4),
        },
        "current_vitals": {
            f: round(float(row[f]), 4) if f in row and pd.notna(row[f]) else None for f in BASE_FEATURES
        },
        "routing_hint": "urgent_review" if row["anomaly_label"] == "critical" else "routine_review",
        "reason": "hybrid_threshold_crossed" if row["anomaly_label"] != "normal" else "within_band",
    }


def train_and_export(
    csv_path: Path, output_dir: Path, json_name: str = "firebase_import.json"
) -> Dict[str, Any]:
    cfg = HybridConfig()
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(csv_path)
    df = preprocess_dataset(raw, cfg)
    df, temporal_features, pop_features, cat_cols = add_time_series_features(df)

    temporal_models, temporal_raw = fit_temporal_models(df, cfg)
    population_model, population_raw, pop_model_cols = fit_population_model(df, pop_features, cat_cols, cfg)

    temporal_scaler = fit_percentile_scaler(temporal_raw)
    population_scaler = fit_percentile_scaler(population_raw)
    temporal_pct = apply_percentile_scaler(temporal_raw, temporal_scaler)
    population_pct = apply_percentile_scaler(population_raw, population_scaler)

    hybrid_score = cfg.temporal_weight * temporal_pct + cfg.population_weight * population_pct
    anomaly_label, high_mask, critical_mask = assign_anomaly_labels(
        hybrid_score, temporal_pct, population_pct, cfg
    )

    df["temporal_raw"] = temporal_raw
    df["population_raw"] = population_raw
    df["temporal_pct"] = temporal_pct
    df["population_pct"] = population_pct
    df["hybrid_score"] = hybrid_score
    df["anomaly_label"] = anomaly_label
    df["triggered_signals"] = df.apply(explain_triggered_signals, axis=1)

    model_bundle = {
        "config": cfg.__dict__,
        "base_features": BASE_FEATURES,
        "context_features": CONTEXT_FEATURES,
        "risk_directions": RISK_DIRECTIONS,
        "temporal_features": temporal_features,
        "temporal_targets": TEMPORAL_TARGETS,
        "temporal_models": temporal_models,
        "population_model": population_model,
        "population_model_columns": pop_model_cols,
        "cat_columns": cat_cols,
        "temporal_scaler": temporal_scaler,
        "population_scaler": population_scaler,
    }

    model_path = output_dir / "hybrid_detector_model.pkl"
    with model_path.open("wb") as f:
        pickle.dump(model_bundle, f)

    predictions_path = output_dir / "hybrid_predictions.csv"
    df[
        [
            "user_id",
            "date",
            "hybrid_score",
            "anomaly_label",
            "temporal_pct",
            "population_pct",
            "temporal_raw",
            "population_raw",
            "triggered_signals",
        ]
    ].to_csv(predictions_path, index=False)

    metrics = {
        "rows": int(len(df)),
        "unique_users": int(df["user_id"].nunique()),
        "high_alerts": int(high_mask.sum()),
        "critical_alerts": int(critical_mask.sum()),
        "any_alerts": int((df["anomaly_label"] != "normal").sum()),
        "alert_rate": float((df["anomaly_label"] != "normal").mean()),
        "temporal_weight": cfg.temporal_weight,
        "population_weight": cfg.population_weight,
    }

    metrics_path = output_dir / "training_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    records = {str(i): build_firebase_record(row) for i, (_, row) in enumerate(df.iterrows())}
    firebase_path = output_dir / json_name
    with firebase_path.open("w", encoding="utf-8") as f:
        json.dump({"anomaly_events": records}, f)

    return {
        "model_path": str(model_path),
        "metrics_path": str(metrics_path),
        "predictions_path": str(predictions_path),
        "firebase_json_path": str(firebase_path),
        "metrics": metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train hybrid anomaly detector for WHOOP-style data.")
    parser.add_argument(
        "--csv_path",
        default="./whoop_fitness_dataset_100k.csv",
        help="Path to source CSV file",
    )
    parser.add_argument(
        "--output_dir",
        default="./artifacts",
        help="Directory to save model/metrics/predictions/firebase JSON",
    )
    parser.add_argument(
        "--json_name",
        default="firebase_import.json",
        help="Output Firebase import JSON filename",
    )
    args = parser.parse_args()

    result = train_and_export(Path(args.csv_path), Path(args.output_dir), args.json_name)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
