"""Scan RadarScenes sequences and summarize dataset coverage."""

import logging
from pathlib import Path
import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config  # noqa: E402
from src.utils import apply_plot_style, count_camera_images, discover_sequences, ensure_output_dirs, save_bar, timestamp_to_seconds  # noqa: E402

log = logging.getLogger(__name__)


def summarize_sequence(sequence_path: Path) -> tuple[dict, dict, dict] | None:
    """Extract per-sequence statistics from the HDF5 radar file.

    Returns None when the file is unreadable so the caller can skip it
    entirely, avoiding poisoned zeroes in downstream aggregations.
    """
    sequence_id = sequence_path.name
    h5_path = sequence_path / "radar_data.h5"
    row = {
        "sequence_id": sequence_id,
        "num_radar_detections": 0,
        "num_odometry_samples": 0,
        "num_camera_images": count_camera_images(sequence_path),
        "radar_timestamp_min": pd.NA,
        "radar_timestamp_max": pd.NA,
        "duration_seconds": 0.0,
        "scenes_json_exists": (sequence_path / "scenes.json").exists(),
        "camera_folder_exists": (sequence_path / "camera").exists(),
    }
    # Kept as separate dicts keyed by raw integer IDs to avoid a sparse
    # matrix explosion in the main overview table.
    label_counts: dict[int, int] = {}
    sensor_counts: dict[int, int] = {}

    try:
        with h5py.File(h5_path, "r") as h5:
            if "radar_data" in h5:
                radar = h5["radar_data"]
                row["num_radar_detections"] = len(radar)
                if len(radar):
                    timestamps = radar["timestamp"][:]
                    row["radar_timestamp_min"] = int(timestamps.min())
                    row["radar_timestamp_max"] = int(timestamps.max())
                    row["duration_seconds"] = timestamp_to_seconds(timestamps.max() - timestamps.min())

                    ids, counts = np.unique(radar["label_id"][:], return_counts=True)
                    label_counts = dict(zip(ids.astype(int), counts.astype(int)))

                    ids, counts = np.unique(radar["sensor_id"][:], return_counts=True)
                    sensor_counts = dict(zip(ids.astype(int), counts.astype(int)))

            if "odometry" in h5:
                row["num_odometry_samples"] = len(h5["odometry"])
    except OSError as exc:
        log.warning("Skipping %s: %s", sequence_id, exc)
        return None

    return row, label_counts, sensor_counts


def make_plots(overview: pd.DataFrame, label_df: pd.DataFrame, sensor_df: pd.DataFrame) -> None:
    out = config.OUTPUT_ROOT / "figures" / "dataset_overview"
    save_bar(overview, "sequence_id", "num_radar_detections", "Radar detections per sequence", out / "radar_points_per_sequence.png")
    save_bar(overview, "sequence_id", "duration_seconds", "Duration per sequence", out / "duration_per_sequence.png", "#59a14f")
    save_bar(overview, "sequence_id", "num_camera_images", "Camera images per sequence", out / "camera_images_per_sequence.png", "#f28e2b")

    # Map raw label IDs to human-readable names and palette colours at plot time.
    label_totals = label_df.drop(columns=["sequence_id"], errors="ignore").sum().sort_index()
    fig, ax = plt.subplots(figsize=(12, 5))
    names = [config.LABEL_MAP.get(int(col), f"unknown_{col}") for col in label_totals.index]
    colors = [config.LABEL_COLORS.get(int(col), "#333333") for col in label_totals.index]
    ax.bar(names, label_totals.values, color=colors)
    ax.set_title("Label distribution across processed sequences")
    ax.set_ylabel("radar detections")
    ax.tick_params(axis="x", rotation=70)
    fig.tight_layout()
    fig.savefig(out / "label_distribution_all_sequences.png")
    plt.close(fig)

    sensor_totals = sensor_df.drop(columns=["sequence_id"], errors="ignore").sum().sort_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = [config.SENSOR_COLORS.get(int(col), "#333333") for col in sensor_totals.index]
    ax.bar([f"sensor_{col}" for col in sensor_totals.index], sensor_totals.values, color=colors)
    ax.set_title("Sensor distribution across processed sequences")
    ax.set_ylabel("radar detections")
    fig.tight_layout()
    fig.savefig(out / "sensor_distribution_all_sequences.png")
    plt.close(fig)


def write_summary(overview: pd.DataFrame) -> None:
    lines = [
        f"Processed sequences: {len(overview)}",
        f"Total radar detections: {int(overview['num_radar_detections'].sum()):,}",
        f"Total camera images: {int(overview['num_camera_images'].sum()):,}",
        f"Total odometry samples: {int(overview['num_odometry_samples'].sum()):,}",
        f"Distinct sequence durations: {overview['duration_seconds'].nunique(dropna=True)}",
        f"Distinct detection counts: {overview['num_radar_detections'].nunique(dropna=True)}",
    ]
    (config.OUTPUT_ROOT / "tables" / "dataset_overview_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    ensure_output_dirs()
    apply_plot_style()
    sequences = discover_sequences()
    rows, label_rows, sensor_rows = [], [], []
    for seq in sequences:
        print(f"Summarizing {seq.name}")
        result = summarize_sequence(seq)
        if result is None:
            continue
        row, label_counts, sensor_counts = result
        rows.append(row)
        label_rows.append({"sequence_id": seq.name, **label_counts})
        sensor_rows.append({"sequence_id": seq.name, **sensor_counts})

    overview = pd.DataFrame(rows)
    label_df = pd.DataFrame(label_rows).fillna(0)
    sensor_df = pd.DataFrame(sensor_rows).fillna(0)
    table_dir = config.OUTPUT_ROOT / "tables"
    overview.to_csv(table_dir / "dataset_overview.csv", index=False)
    label_df.to_csv(table_dir / "label_distribution_by_sequence.csv", index=False)
    sensor_df.to_csv(table_dir / "sensor_distribution_by_sequence.csv", index=False)
    make_plots(overview, label_df, sensor_df)
    write_summary(overview)


if __name__ == "__main__":
    main()
