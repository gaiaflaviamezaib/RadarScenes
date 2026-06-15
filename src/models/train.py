"""Train supervised radar object classifiers.

Experiment A – Point-Level Baseline
    Data: radar_points_features.csv (dynamic classes only).
    Models: Random Forest only.
    Split: Strict sequence-level GroupShuffleSplit.

Experiment B – Track-Level Aggregation
    Data: radar_track_features.csv (all classes).
    Models: XGBoost, Random Forest, Logistic Regression, Gaussian NB.
    Split: Strict sequence-level GroupShuffleSplit.
"""

from pathlib import Path
import sys

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config  # noqa: E402
from src.utils import balanced_sample, ensure_output_dirs, load_feature_table, load_track_table  # noqa: E402


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------

def split_by_sequence(df: pd.DataFrame, features: list[str]):
    """Strict sequence-level split.  No random-shuffle fallback."""
    X = df[features]
    y = df["label_id"].astype(int)
    groups = df["sequence_id"]

    splitter = GroupShuffleSplit(
        n_splits=50, test_size=0.2, random_state=config.RANDOM_SEED
    )
    for train_idx, test_idx in splitter.split(X, y, groups):
        test_classes = set(y.iloc[test_idx].unique())
        train_classes = set(y.iloc[train_idx].unique())
        if test_classes.issubset(train_classes):
            n_train_seq = groups.iloc[train_idx].nunique()
            n_test_seq = groups.iloc[test_idx].nunique()
            print(
                f"  Sequence-level split: {n_train_seq} train / "
                f"{n_test_seq} test sequences."
            )
            return (
                X.iloc[train_idx], X.iloc[test_idx],
                y.iloc[train_idx], y.iloc[test_idx],
                df.iloc[test_idx],
            )

    raise RuntimeError(
        "Could not find a sequence-level split where all test classes "
        "appear in training.  Are there enough sequences?"
    )


# ---------------------------------------------------------------------------
# Model catalogue
# ---------------------------------------------------------------------------

def _scaled(model):
    """Wrap a model in an impute → scale → model pipeline."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", model),
    ])


POINT_MODELS = {
    "random_forest": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(
            n_estimators=150,
            class_weight="balanced_subsample",
            n_jobs=1,
            random_state=config.RANDOM_SEED,
        )),
    ]),
}


def _track_models() -> dict:
    """Build the Track-Level model catalogue (XGBoost loaded lazily)."""
    specs: dict = {
        "random_forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(
                n_estimators=150,
                class_weight="balanced_subsample",
                n_jobs=1,
                random_state=config.RANDOM_SEED,
            )),
        ]),
        "logistic_regression": _scaled(
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=config.RANDOM_SEED,
            )
        ),
        "gaussian_nb": _scaled(GaussianNB()),
    }
    try:
        from xgboost import XGBClassifier
        specs["xgboost"] = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", XGBClassifier(
                n_estimators=160,
                max_depth=5,
                learning_rate=0.08,
                subsample=0.9,
                colsample_bytree=0.9,
                tree_method="hist",
                eval_metric="mlogloss",
                n_jobs=1,
                random_state=config.RANDOM_SEED,
            )),
        ])
    except ImportError:
        print("  XGBoost is not installed — skipping.")
    return specs


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def run_experiment(
    experiment_name: str,
    df: pd.DataFrame,
    features: list[str],
    models: dict,
    sample_rows: int | None = None,
) -> pd.DataFrame:
    """Train and evaluate every model in *models* on *df*."""
    print(f"\n{'='*60}")
    print(f"  {experiment_name}")
    print(f"{'='*60}")

    if sample_rows and len(df) > sample_rows:
        df = balanced_sample(df, max_rows=sample_rows)

    X_train, X_test, y_train, y_test, test_meta = split_by_sequence(df, features)

    rows = []
    for name, model in models.items():
        train_X, train_y = X_train, y_train
        print(f"  Training {name} on {len(train_X):,} samples …")

        # XGBoost needs contiguous 0..K-1 labels.
        if name == "xgboost":
            from sklearn.preprocessing import LabelEncoder
            enc = LabelEncoder()
            train_y_enc = enc.fit_transform(train_y)
            model.fit(train_X, train_y_enc)
            pred = enc.inverse_transform(model.predict(X_test).astype(int))
            joblib.dump(
                {"model": model, "label_encoder": enc},
                config.OUTPUT_ROOT / "models" / f"{experiment_name}_{name}.joblib",
            )
        else:
            model.fit(train_X, train_y)
            pred = model.predict(X_test)
            joblib.dump(
                model,
                config.OUTPUT_ROOT / "models" / f"{experiment_name}_{name}.joblib",
            )

        acc = accuracy_score(y_test, pred)
        f1 = f1_score(y_test, pred, average="macro", zero_division=0)
        rows.append({
            "experiment": experiment_name,
            "model": name,
            "n_train": len(train_X),
            "n_test": len(X_test),
            "accuracy": acc,
            "macro_f1": f1,
        })
        print(f"    → accuracy {acc:.4f}  |  macro-F1 {f1:.4f}")

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ensure_output_dirs()

    all_metrics = []

    # ── Experiment A: Point-Level Baseline (dynamic only, Random Forest) ──
    print("\nLoading point-level data (dynamic only) …")
    point_df = load_feature_table(sample_rows=72_000, dynamic_only=True)
    point_df = point_df.dropna(subset=["label_id"])
    metrics_a = run_experiment(
        experiment_name="experiment_A_point_level",
        df=point_df,
        features=config.RADAR_FEATURES,
        models=POINT_MODELS,
        sample_rows=60_000,
    )
    all_metrics.append(metrics_a)

    # ── Experiment B: Track-Level Aggregation (all models) ────────────────
    print("\nLoading track-level data …")
    track_df = load_track_table(dynamic_only=False)
    metrics_b = run_experiment(
        experiment_name="experiment_B_track_level",
        df=track_df,
        features=config.TRACK_FEATURES,
        models=_track_models(),
    )
    all_metrics.append(metrics_b)

    # ── Summary ───────────────────────────────────────────────────────────
    summary = pd.concat(all_metrics, ignore_index=True)
    summary_path = config.OUTPUT_ROOT / "tables" / "training_summary_metrics.csv"
    summary.to_csv(summary_path, index=False)

    print(f"\n{'='*60}")
    print("  Final Summary — sorted by Macro F1")
    print(f"{'='*60}")
    print(summary.sort_values("macro_f1", ascending=False).to_string(index=False))
    print(f"\nSaved → {summary_path}")


if __name__ == "__main__":
    main()
