"""Timestamp-link radar detections, odometry samples, camera frames, and scenes."""

from pathlib import Path
import sys

import h5py
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config  # noqa: E402
from src.utils import (  # noqa: E402
    add_label_names,
    apply_plot_style,
    camera_timestamp_from_name,
    discover_sequences,
    ensure_output_dirs,
    label_legend_handles,
    list_camera_images,
    load_scenes,
    structured_slice_to_frame,
    timestamp_to_seconds,
)


def nearest_value(sorted_values: np.ndarray, target: int) -> int | None:
    if sorted_values.size == 0:
        return None
    idx = int(np.searchsorted(sorted_values, target))
    candidates = []
    if idx < sorted_values.size:
        candidates.append(sorted_values[idx])
    if idx > 0:
        candidates.append(sorted_values[idx - 1])
    return int(min(candidates, key=lambda x: abs(int(x) - target)))


def radar_points_near(h5_path: Path, timestamp: int, window_us: int = 100_000) -> pd.DataFrame:
    with h5py.File(h5_path, "r") as h5:
        radar = h5["radar_data"]
        ts = radar["timestamp"][:]
        mask_idx = np.where((ts >= timestamp - window_us) & (ts <= timestamp + window_us))[0]
        if mask_idx.size == 0:
            return pd.DataFrame()
        df = pd.DataFrame.from_records(radar[mask_idx])
    for col in ["uuid", "track_id"]:
        if col in df.columns:
            df[col] = df[col].str.decode("utf-8", errors="ignore").str.strip("\x00")
    return add_label_names(df)


def link_sequence(sequence_path: Path, max_frames: int = 5) -> tuple[list[dict], list[tuple[Path, pd.DataFrame, dict]]]:
    h5_path = sequence_path / "radar_data.h5"
    cameras = [(p, camera_timestamp_from_name(p)) for p in list_camera_images(sequence_path)]
    cameras = [(p, ts) for p, ts in cameras if ts is not None]
    scenes = load_scenes(sequence_path)
    visual_examples = []
    rows = []
    if not cameras:
        return rows, visual_examples

    with h5py.File(h5_path, "r") as h5:
        radar_ts = np.sort(h5["radar_data"]["timestamp"][:])
        odom_ts = np.sort(h5["odometry"]["timestamp"][:]) if "odometry" in h5 else np.array([], dtype=np.int64)

    chosen = np.linspace(0, len(cameras) - 1, min(max_frames, len(cameras)), dtype=int)
    for camera_idx in chosen:
        camera_path, camera_ts = cameras[int(camera_idx)]
        radar_ts_match = nearest_value(radar_ts, camera_ts)
        odom_ts_match = nearest_value(odom_ts, camera_ts)
        if radar_ts_match is None:
            continue
        near = radar_points_near(h5_path, radar_ts_match)
        label_counts = near["label_name"].value_counts().to_dict() if not near.empty else {}
        row = {
            "sequence_id": sequence_path.name,
            "radar_timestamp": radar_ts_match,
            "nearest_odometry_timestamp": odom_ts_match,
            "nearest_camera_timestamp": camera_ts,
            "camera_filename": camera_path.name,
            "radar_to_camera_time_difference": timestamp_to_seconds(radar_ts_match - camera_ts),
            "radar_to_odometry_time_difference": timestamp_to_seconds(radar_ts_match - odom_ts_match) if odom_ts_match is not None else pd.NA,
            "number_of_radar_points_near_frame": len(near),
            "label_distribution_near_frame": label_counts,
            "scene_first_timestamp": scenes.get("first_timestamp"),
            "scene_last_timestamp": scenes.get("last_timestamp"),
        }
        rows.append(row)
        if len(visual_examples) < 1 and not near.empty:
            visual_examples.append((camera_path, near, row))
    return rows, visual_examples


def save_visual_example(camera_path: Path, radar_df: pd.DataFrame, row: dict, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    try:
        axes[0].imshow(mpimg.imread(camera_path))
        axes[0].axis("off")
    except Exception as exc:
        axes[0].text(0.5, 0.5, f"Could not load image:\n{exc}", ha="center", va="center")
        axes[0].axis("off")
    axes[0].set_title("Camera frame")

    for label_id, group in radar_df.groupby("label_id"):
        axes[1].scatter(group["x_cc"], group["y_cc"], s=12, alpha=0.75,
                        color=config.LABEL_COLORS.get(int(label_id), "#333333"),
                        label=config.LABEL_MAP.get(int(label_id), "unknown"))
    axes[1].set_aspect("equal", adjustable="box")
    axes[1].set_xlabel("x_cc (m)")
    axes[1].set_ylabel("y_cc (m)")
    axes[1].set_title("Timestamp-aligned radar point cloud")
    axes[1].legend(handles=label_legend_handles(radar_df["label_id"].unique()), fontsize=8, loc="best")
    fig.suptitle(
        f"{row['sequence_id']} | {row['camera_filename']} | radar-camera dt={row['radar_to_camera_time_difference']:.3f}s"
    )
    fig.tight_layout()
    fig.savefig(out_dir / f"{row['sequence_id']}_{camera_path.stem}_camera_radar_side_by_side.png")
    plt.close(fig)


def main() -> None:
    ensure_output_dirs()
    apply_plot_style()
    sequences = discover_sequences()
    all_rows = []
    visual_examples = []
    for seq in sequences:
        print(f"Linking timestamps in {seq.name}")
        rows, examples = link_sequence(seq)
        all_rows.extend(rows)
        if len(visual_examples) < 3:
            visual_examples.extend(examples[: max(0, 3 - len(visual_examples))])

    out_table = config.OUTPUT_ROOT / "tables" / "radar_odometry_camera_links.csv"
    pd.DataFrame(all_rows).to_csv(out_table, index=False)
    out_dir = config.OUTPUT_ROOT / "figures" / "camera_radar_side_by_side"
    for camera_path, radar_df, row in visual_examples[:3]:
        save_visual_example(camera_path, radar_df, row, out_dir)
    note = (
        "These figures are timestamp-aligned camera and radar views. They do not project radar detections onto "
        "the camera image because a full calibrated camera projection is not implemented here."
    )
    (out_dir / "projection_note.txt").write_text(note, encoding="utf-8")
    print(f"Saved {out_table}")


if __name__ == "__main__":
    main()
