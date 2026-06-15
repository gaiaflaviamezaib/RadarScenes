"""Create presentation-ready exploratory plots for RadarScenes."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config  # noqa: E402
from src.utils import (  # noqa: E402
    apply_plot_style,
    balanced_sample,
    ensure_output_dirs,
    label_legend_handles,
    load_feature_table,
    scatter_by_category,
    sensor_legend_handles,
)


def save_label_scatter(df: pd.DataFrame, x: str, y: str, title: str, filename: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 7))
    scatter_by_category(ax, df, x, y, "label_id", config.LABEL_COLORS, config.LABEL_MAP, s=6, alpha=0.65)
    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.legend(handles=label_legend_handles(df["label_id"].unique()), fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(config.OUTPUT_ROOT / "figures" / "dataset_visualizations" / filename)
    plt.close(fig)


def save_sensor_scatter(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 7))
    scatter_by_category(ax, df, "x_cc", "y_cc", "sensor_id", config.SENSOR_COLORS, None, s=6, alpha=0.65)
    ax.set_title("2D radar point cloud by sensor_id (balanced sample)")
    ax.set_xlabel("x_cc (m)")
    ax.set_ylabel("y_cc (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(handles=sensor_legend_handles(df["sensor_id"].unique()), fontsize=9)
    fig.tight_layout()
    fig.savefig(config.OUTPUT_ROOT / "figures" / "dataset_visualizations" / "B_point_cloud_by_sensor_id.png")
    plt.close(fig)


def save_bar_counts(df: pd.DataFrame) -> None:
    label_counts = df["label_id"].map(config.LABEL_MAP).value_counts()
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(label_counts.index, label_counts.values, color="#4e79a7")
    ax.set_title("Class imbalance in sampled radar detections")
    ax.set_ylabel("detections")
    ax.tick_params(axis="x", rotation=60)
    fig.tight_layout()
    fig.savefig(config.OUTPUT_ROOT / "figures" / "dataset_visualizations" / "F_class_imbalance.png")
    plt.close(fig)

    sensor_counts = df["sensor_id"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([f"sensor {i}" for i in sensor_counts.index], sensor_counts.values,
           color=[config.SENSOR_COLORS.get(int(i), "#333333") for i in sensor_counts.index])
    ax.set_title("Sensor distribution in sampled radar detections")
    ax.set_ylabel("detections")
    fig.tight_layout()
    fig.savefig(config.OUTPUT_ROOT / "figures" / "dataset_visualizations" / "G_sensor_distribution.png")
    plt.close(fig)

    seq_counts = df["sequence_id"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(seq_counts.index, seq_counts.values, color="#59a14f")
    ax.set_title("Per-sequence sample count")
    ax.set_ylabel("sampled detections")
    ax.tick_params(axis="x", rotation=75)
    fig.tight_layout()
    fig.savefig(config.OUTPUT_ROOT / "figures" / "dataset_visualizations" / "H_per_sequence_sample_count.png")
    plt.close(fig)


def save_track_plot() -> None:
    track_path = config.OUTPUT_ROOT / "tables" / "radar_track_features.csv"
    if not track_path.exists():
        return
    tracks = pd.read_csv(track_path)
    if tracks.empty:
        return
    tracks = balanced_sample(tracks, max_rows=50_000)
    fig, ax = plt.subplots(figsize=(10, 7))
    scatter_by_category(ax, tracks, "mean_range", "mean_rcs", "label_id", config.LABEL_COLORS, config.LABEL_MAP, s=12, alpha=0.7)
    ax.set_title("Track-level mean RCS vs mean range by label_id")
    ax.set_xlabel("mean range_sc (m)")
    ax.set_ylabel("mean RCS (dBsm)")
    ax.legend(handles=label_legend_handles(tracks["label_id"].unique()), fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(config.OUTPUT_ROOT / "figures" / "dataset_visualizations" / "I_track_mean_rcs_vs_range.png")
    plt.close(fig)


def save_pca(df: pd.DataFrame) -> None:
    X = df[config.RADAR_FEATURES]
    pipeline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), PCA(n_components=2, random_state=config.RANDOM_SEED))
    coords = pipeline.fit_transform(X)
    pca_df = df[["label_id"]].copy()
    pca_df["PC1"] = coords[:, 0]
    pca_df["PC2"] = coords[:, 1]
    save_label_scatter(pca_df, "PC1", "PC2", "PCA 2D projection colored by label_id (balanced sample)", "J_pca_by_label_id.png")


def save_3d_feature_view(df: pd.DataFrame) -> None:
    """Save a 3D feature-space view, not a physical 3D reconstruction."""
    small = balanced_sample(df, max_rows=18_000)
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")
    for label_id, group in small.groupby("label_id"):
        ax.scatter(
            group["x_cc"],
            group["y_cc"],
            group["vr_compensated"],
            s=4,
            alpha=0.45,
            color=config.LABEL_COLORS.get(int(label_id), "#333333"),
            label=f"{int(label_id)}: {config.LABEL_MAP.get(int(label_id), 'unknown')}",
        )
    ax.set_title("3D feature-space radar view: x_cc, y_cc, vr_compensated")
    ax.set_xlabel("x_cc (m)")
    ax.set_ylabel("y_cc (m)")
    ax.set_zlabel("vr_compensated (m/s)")
    ax.view_init(elev=25, azim=-55)
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.tight_layout()
    fig.savefig(config.OUTPUT_ROOT / "figures" / "dataset_visualizations" / "K_3d_feature_space_by_label_id.png")
    plt.close(fig)


def main() -> None:
    ensure_output_dirs()
    apply_plot_style()
    df = load_feature_table(sample_rows=120_000)
    df = df.dropna(subset=["label_id"])
    df["label_id"] = df["label_id"].astype(int)
    sampled = balanced_sample(df, max_rows=80_000)
    save_label_scatter(sampled, "x_cc", "y_cc", "2D radar point cloud by label_id (balanced sample)", "A_point_cloud_by_label_id.png")
    save_sensor_scatter(sampled)
    save_label_scatter(sampled, "range_sc", "rcs", "RCS vs range by label_id (balanced sample)", "C_rcs_vs_range_by_label_id.png")
    save_label_scatter(sampled, "range_sc", "vr_compensated", "Velocity vs range by label_id (balanced sample)", "D_velocity_vs_range_by_label_id.png")
    save_label_scatter(sampled, "range_sc", "azimuth_sc", "Azimuth vs range by label_id (balanced sample)", "E_azimuth_vs_range_by_label_id.png")
    save_bar_counts(sampled)
    save_track_plot()
    save_pca(sampled)
    save_3d_feature_view(sampled)
    metadata = "Plots use balanced sampling when the full point table is large. Categorical variables use discrete colors and legends."
    (config.OUTPUT_ROOT / "figures" / "dataset_visualizations" / "plot_metadata.txt").write_text(metadata, encoding="utf-8")


if __name__ == "__main__":
    main()
