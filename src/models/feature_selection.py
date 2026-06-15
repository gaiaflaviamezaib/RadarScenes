"""Evaluate radar feature usefulness without brute-forcing every feature subset."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config  # noqa: E402
from src.utils import apply_plot_style, ensure_output_dirs, load_feature_table  # noqa: E402


def prepare(dynamic_only: bool) -> tuple[pd.DataFrame, pd.Series]:
    df = load_feature_table(sample_rows=150_000, dynamic_only=dynamic_only).dropna(subset=["label_id"])
    X = df[config.RADAR_FEATURES]
    y = df["label_id"].astype(int)
    print("Class counts:")
    print(y.value_counts().sort_index())
    return X, y


def save_importance_plot(table: pd.DataFrame, value_col: str, title: str, filename: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ordered = table.sort_values(value_col)
    ax.barh(ordered["feature"], ordered[value_col], color="#4e79a7")
    ax.set_title(title)
    ax.set_xlabel(value_col)
    fig.tight_layout()
    fig.savefig(config.OUTPUT_ROOT / "figures" / "feature_importance" / filename)
    plt.close(fig)


def save_correlation_pairs(X: pd.DataFrame, suffix: str, threshold: float = 0.92) -> None:
    numeric = X.select_dtypes(include=[np.number]).copy()
    if numeric.empty:
        return
    corr = numeric.corr(method="spearman").abs()
    pairs = []
    cols = list(corr.columns)
    for i, left in enumerate(cols):
        for right in cols[i + 1:]:
            value = float(corr.loc[left, right])
            if value >= threshold:
                pairs.append({"feature_left": left, "feature_right": right, "spearman_abs_corr": value, "experiment": suffix})
    out = pd.DataFrame(pairs).sort_values("spearman_abs_corr", ascending=False) if pairs else pd.DataFrame(
        columns=["feature_left", "feature_right", "spearman_abs_corr", "experiment"]
    )
    out.to_csv(config.OUTPUT_ROOT / "tables" / f"feature_correlation_pairs{suffix}.csv", index=False)


def run_experiment(dynamic_only: bool, suffix: str) -> None:
    X, y = prepare(dynamic_only)
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.25, random_state=config.RANDOM_SEED)
    save_correlation_pairs(X_train, suffix)

    # Testing every feature subset grows as 2^n. With 10 radar features this is
    # already 1024 subsets, and more engineered features would quickly become
    # inefficient, so we use established feature-selection and importance tools.
    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train)
    mi = SelectKBest(mutual_info_classif, k="all").fit(X_train_imp, y_train)
    mi_table = pd.DataFrame({"feature": config.RADAR_FEATURES, "mutual_info": mi.scores_, "experiment": suffix})
    mi_table.to_csv(config.OUTPUT_ROOT / "tables" / f"feature_importance_mutual_info{suffix}.csv", index=False)
    save_importance_plot(mi_table, "mutual_info", f"Mutual information feature ranking {suffix}", f"mutual_info{suffix}.png")

    logreg = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=config.RANDOM_SEED),
    )
    logreg.fit(X_train, y_train)
    coefs = np.abs(logreg.named_steps["logisticregression"].coef_).mean(axis=0)
    lr_table = pd.DataFrame({"feature": config.RADAR_FEATURES, "mean_abs_coefficient": coefs, "experiment": suffix})
    lr_table.to_csv(config.OUTPUT_ROOT / "tables" / f"feature_importance_logistic_regression{suffix}.csv", index=False)
    save_importance_plot(lr_table, "mean_abs_coefficient", f"Logistic regression coefficient ranking {suffix}", f"logistic_regression{suffix}.png")

    rf = make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestClassifier(n_estimators=120, class_weight="balanced_subsample", n_jobs=1, random_state=config.RANDOM_SEED),
    )
    rf.fit(X_train, y_train)
    rf_importance = rf.named_steps["randomforestclassifier"].feature_importances_
    rf_table = pd.DataFrame({"feature": config.RADAR_FEATURES, "random_forest_importance": rf_importance, "experiment": suffix})
    rf_table.to_csv(config.OUTPUT_ROOT / "tables" / f"feature_importance_random_forest{suffix}.csv", index=False)
    save_importance_plot(rf_table, "random_forest_importance", f"Random Forest feature importance {suffix}", f"random_forest{suffix}.png")

    perm = permutation_importance(rf, X_test, y_test, n_repeats=4, random_state=config.RANDOM_SEED, n_jobs=1)
    perm_table = pd.DataFrame({"feature": config.RADAR_FEATURES, "permutation_importance": perm.importances_mean, "experiment": suffix})
    perm_table.to_csv(config.OUTPUT_ROOT / "tables" / f"feature_importance_permutation{suffix}.csv", index=False)
    save_importance_plot(perm_table, "permutation_importance", f"Permutation importance {suffix}", f"permutation{suffix}.png")


def main() -> None:
    ensure_output_dirs()
    apply_plot_style()
    run_experiment(dynamic_only=False, suffix="_all_classes")
    run_experiment(dynamic_only=True, suffix="_dynamic_only")

    # Convenience combined files using the all-class experiment names requested
    # in the assignment.
    for name in ["mutual_info", "logistic_regression", "random_forest"]:
        src = config.OUTPUT_ROOT / "tables" / f"feature_importance_{name}_all_classes.csv"
        dst = config.OUTPUT_ROOT / "tables" / f"feature_importance_{name}.csv"
        if src.exists():
            pd.read_csv(src).to_csv(dst, index=False)


if __name__ == "__main__":
    main()
