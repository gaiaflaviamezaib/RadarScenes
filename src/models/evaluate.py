"""Evaluate trained model predictions with imbalance-aware metrics."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, accuracy_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config  # noqa: E402
from src.utils import apply_plot_style, ensure_output_dirs  # noqa: E402


def metric_row(group: pd.DataFrame) -> dict:
    y_true = group["label_id"].astype(int)
    y_pred = group["predicted_label_id"].astype(int)
    # Metrics follow the lecture one-vs-rest notation for each class c:
    # precision_c = TP_c / (TP_c + FP_c), recall_c = TP_c / (TP_c + FN_c),
    # specificity_c = TN_c / (TN_c + FP_c), and
    # F1_c = 2 precision_c recall_c / (precision_c + recall_c).
    # sklearn returns macro averages as (1 / C) * sum_c metric_c.
    macro = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    weighted = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    return {
        "experiment": group["experiment"].iloc[0],
        "model": group["model"].iloc[0],
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": macro[0],
        "macro_recall": macro[1],
        "macro_f1": macro[2],
        "weighted_precision": weighted[0],
        "weighted_recall": weighted[1],
        "weighted_f1": weighted[2],
    }


def save_confusion(group: pd.DataFrame) -> None:
    exp, model = group["experiment"].iloc[0], group["model"].iloc[0]
    labels = sorted(set(group["label_id"].astype(int)) | set(group["predicted_label_id"].astype(int)))
    names = [config.LABEL_MAP.get(i, str(i)) for i in labels]
    cm = confusion_matrix(group["label_id"].astype(int), group["predicted_label_id"].astype(int), labels=labels)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, cmap="Blues", xticklabels=names, yticklabels=names, ax=ax, annot=False)
    ax.set_title(f"Confusion matrix: {exp} / {model}")
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.tick_params(axis="x", rotation=60)
    fig.tight_layout()
    fig.savefig(config.OUTPUT_ROOT / "figures" / "confusion_matrices" / f"{exp}_{model}_confusion_matrix.png")
    plt.close(fig)

    report = classification_report(group["label_id"].astype(int), group["predicted_label_id"].astype(int),
                                   labels=labels, target_names=names, zero_division=0)
    (config.OUTPUT_ROOT / "tables" / "classification_reports" / f"{exp}_{model}.txt").write_text(report, encoding="utf-8")

    report_dict = classification_report(group["label_id"].astype(int), group["predicted_label_id"].astype(int),
                                        labels=labels, target_names=names, zero_division=0, output_dict=True)
    per_class = pd.DataFrame(report_dict).T.loc[names]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(per_class.index, per_class["f1-score"], color="#59a14f")
    ax.set_title(f"Per-class F1: {exp} / {model}")
    ax.set_ylabel("F1 score")
    ax.tick_params(axis="x", rotation=60)
    fig.tight_layout()
    fig.savefig(config.OUTPUT_ROOT / "figures" / "confusion_matrices" / f"{exp}_{model}_per_class_f1.png")
    plt.close(fig)


def save_model_comparison(metrics: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    labels = metrics["experiment"] + "\n" + metrics["model"]
    ax.bar(labels, metrics["macro_f1"], color="#4e79a7", label="macro F1")
    ax.plot(labels, metrics["accuracy"], color="#e15759", marker="o", label="accuracy")
    ax.set_title("Model comparison: macro F1 emphasized for imbalanced classes")
    ax.set_ylabel("score")
    ax.tick_params(axis="x", rotation=75)
    ax.legend()
    fig.tight_layout()
    fig.savefig(config.OUTPUT_ROOT / "figures" / "confusion_matrices" / "model_comparison_bar_chart.png")
    plt.close(fig)


def main() -> None:
    ensure_output_dirs()
    apply_plot_style()
    pred_path = config.OUTPUT_ROOT / "tables" / "model_predictions.csv"
    if not pred_path.exists():
        raise FileNotFoundError("Run 06_train_models.py before evaluation.")
    predictions = pd.read_csv(pred_path)
    metrics = []
    for (_, _), group in predictions.groupby(["experiment", "model"]):
        metrics.append(metric_row(group))
        save_confusion(group)
    metrics_df = pd.DataFrame(metrics).sort_values("macro_f1", ascending=False)
    metrics_df.to_csv(config.OUTPUT_ROOT / "tables" / "model_comparison_metrics.csv", index=False)
    save_model_comparison(metrics_df)
    print("Because RadarScenes is imbalanced, use macro F1 and per-class recall/F1 alongside accuracy.")
    print(metrics_df.head())


if __name__ == "__main__":
    main()
