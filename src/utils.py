"""Shared helpers for the RadarScenes reproducible pipeline."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Iterable

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402


def ensure_output_dirs() -> None:
    for relative in [
        "figures",
        "figures/dataset_overview",
        "figures/dataset_visualizations",
        "figures/feature_importance",
        "figures/confusion_matrices",
        "figures/camera_radar_side_by_side",
        "tables",
        "tables/classification_reports",
        "models",
        "presentation",
    ]:
        (config.OUTPUT_ROOT / relative).mkdir(parents=True, exist_ok=True)


def apply_plot_style() -> None:
    plt.rcParams.update(config.PLOT_STYLE)


def sequence_sort_key(path: Path) -> tuple[int, str]:
    match = re.search(r"(\d+)$", path.name)
    return (int(match.group(1)) if match else 10**9, path.name)


def discover_sequences(data_root: Path | None = None, max_sequences: int | None = None) -> list[Path]:
    root = Path(data_root or config.DATA_ROOT)
    if not root.exists():
        raise FileNotFoundError(f"DATA_ROOT does not exist: {root}")
    candidates = [p for p in root.iterdir() if p.is_dir() and (p / "radar_data.h5").exists()]
    candidates = sorted(candidates, key=sequence_sort_key)
    limit = config.MAX_SEQUENCES if max_sequences is None else max_sequences
    return candidates[:limit]


def decode_value(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore").strip("\x00")
    return value


def structured_slice_to_frame(dataset, start: int = 0, stop: int | None = None, columns: Iterable[str] | None = None) -> pd.DataFrame:
    arr = dataset[start:stop]
    df = pd.DataFrame.from_records(arr, columns=columns)
    for col in df.columns:
        if df[col].dtype == object or str(df[col].dtype).startswith("|S"):
            df[col] = df[col].map(decode_value)
    return df


def read_h5_table(h5_path: Path, key: str, columns: Iterable[str] | None = None) -> pd.DataFrame:
    with h5py.File(h5_path, "r") as h5:
        if key not in h5:
            return pd.DataFrame()
        return structured_slice_to_frame(h5[key], columns=columns)


def iter_radar_chunks(h5_path: Path, chunk_size: int = 250_000, columns: Iterable[str] | None = None):
    with h5py.File(h5_path, "r") as h5:
        if "radar_data" not in h5:
            return
        data = h5["radar_data"]
        total = len(data)
        for start in range(0, total, chunk_size):
            yield structured_slice_to_frame(data, start, min(start + chunk_size, total), columns=columns)


def timestamp_to_seconds(delta) -> float:
    """RadarScenes timestamps are integer microseconds in the provided files."""
    return float(delta) / 1_000_000.0


def add_label_names(df: pd.DataFrame) -> pd.DataFrame:
    if "label_id" in df.columns:
        df["label_id"] = pd.to_numeric(df["label_id"], errors="coerce").astype("Int64")
        df["label_name"] = df["label_id"].map(config.LABEL_MAP).fillna("unknown")
    return df


def engineer_radar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer geometric and magnitude-based radar features."""

    out = df.copy()
    numeric_cols = [
        "range_sc",
        "azimuth_sc",
        "rcs",
        "vr",
        "vr_compensated",
        "x_cc",
        "y_cc",
        "x_seq",
        "y_seq",
    ]
    for col in numeric_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    if "range_sc" in out.columns:
        out["radius_cc"] = np.sqrt(np.square(out.get("x_cc", 0.0)) + np.square(out.get("y_cc", 0.0)))
        out["rcs_per_range"] = out["rcs"] / out["range_sc"].replace(0, np.nan)
    else:
        out["radius_cc"] = np.nan
        out["rcs_per_range"] = np.nan

    if "vr" in out.columns:
        out["abs_vr"] = out["vr"].abs()
    else:
        out["abs_vr"] = np.nan

    if "vr_compensated" in out.columns:
        out["abs_vr_compensated"] = out["vr_compensated"].abs()
    else:
        out["abs_vr_compensated"] = np.nan

    if "azimuth_sc" in out.columns:
        out["azimuth_sin"] = np.sin(out["azimuth_sc"])
        out["azimuth_cos"] = np.cos(out["azimuth_sc"])
    else:
        out["azimuth_sin"] = np.nan
        out["azimuth_cos"] = np.nan

    if "x_cc" in out.columns and "y_cc" in out.columns:
        out["xy_product"] = out["x_cc"] * out["y_cc"]
    else:
        out["xy_product"] = np.nan

    return out


def clean_track_ids(df: pd.DataFrame) -> pd.DataFrame:
    if "track_id" in df.columns:
        s = df["track_id"].astype(str).str.strip()
        # Pandas string missing values can appear as "nan" or "<NA>" once cast.
        df["track_id_clean"] = s.mask(s.isin(["", "nan", "<NA>", "None", "0"]), "untracked")
    return df


def count_camera_images(sequence_path: Path) -> int:
    camera_dir = sequence_path / "camera"
    if not camera_dir.exists():
        return 0
    return sum(1 for p in camera_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})


def list_camera_images(sequence_path: Path) -> list[Path]:
    camera_dir = sequence_path / "camera"
    if not camera_dir.exists():
        return []
    return sorted(
        [p for p in camera_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}],
        key=lambda p: camera_timestamp_from_name(p) or 0,
    )


def camera_timestamp_from_name(path: Path) -> int | None:
    matches = re.findall(r"\d+", path.stem)
    if not matches:
        return None
    return int(max(matches, key=len))


def load_scenes(sequence_path: Path) -> dict:
    scene_path = sequence_path / "scenes.json"
    if not scene_path.exists():
        return {}
    try:
        with scene_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return {}


def label_legend_handles(label_ids: Iterable[int] | None = None) -> list[Line2D]:
    ids = sorted(config.LABEL_MAP if label_ids is None else {int(x) for x in label_ids if pd.notna(x)})
    return [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=config.LABEL_COLORS.get(i, "#333333"),
               label=f"{i}: {config.LABEL_MAP.get(i, 'unknown')}", markersize=7)
        for i in ids
    ]


def sensor_legend_handles(sensor_ids: Iterable[int]) -> list[Line2D]:
    ids = sorted({int(x) for x in sensor_ids if pd.notna(x)})
    return [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=config.SENSOR_COLORS.get(i, "#333333"),
               label=f"sensor {i}", markersize=7)
        for i in ids
    ]


def scatter_by_category(ax, df: pd.DataFrame, x: str, y: str, category: str, colors: dict, names: dict | None = None,
                        s: float = 5, alpha: float = 0.7) -> None:
    for value in sorted(df[category].dropna().unique()):
        sub = df[df[category] == value]
        label = names.get(int(value), str(value)) if names else str(value)
        ax.scatter(sub[x], sub[y], s=s, alpha=alpha, color=colors.get(int(value), "#333333"),
                   label=f"{int(value)}: {label}" if names else label, linewidths=0)


def balanced_sample(df: pd.DataFrame, label_col: str = "label_id", max_rows: int = 120_000,
                    random_state: int = config.RANDOM_SEED) -> pd.DataFrame:
    if len(df) <= max_rows or label_col not in df:
        return df.copy()
    per_class = max(1, max_rows // max(1, df[label_col].nunique()))
    parts = []
    for _, group in df.groupby(label_col, dropna=False):
        parts.append(group.sample(min(len(group), per_class), random_state=random_state))
    sampled = pd.concat(parts, ignore_index=True)
    if len(sampled) > max_rows:
        sampled = sampled.sample(max_rows, random_state=random_state)
    return sampled


def save_bar(df: pd.DataFrame, x: str, y: str, title: str, out_path: Path, color: str = "#4e79a7") -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(df[x].astype(str), df[y], color=color)
    ax.set_title(title)
    ax.set_xlabel(x.replace("_", " "))
    ax.set_ylabel(y.replace("_", " "))
    ax.tick_params(axis="x", rotation=75)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def load_feature_table(sample_rows: int | None = None, dynamic_only: bool = False) -> pd.DataFrame:
    path = config.OUTPUT_ROOT / "tables" / "radar_points_features.csv"
    if not path.exists():
        raise FileNotFoundError(f"Build features first: {path}")
    if sample_rows is None:
        df = pd.read_csv(path)
        if dynamic_only:
            df = df[df["label_id"] != 11]
        return df

    # Reservoir-like per-class sampling from chunks. This avoids loading the
    # full 50-sequence point CSV into memory for visualizations and models.
    per_class_cap = max(1, sample_rows // len(config.LABEL_MAP))
    buckets: dict[int, pd.DataFrame] = {}
    dtype_hints = {"uuid": "string", "track_id": "string", "label_name": "string"}
    for chunk in pd.read_csv(path, chunksize=500_000, dtype=dtype_hints, low_memory=False):
        if dynamic_only:
            chunk = chunk[chunk["label_id"] != 11]
        for label_id, group in chunk.groupby("label_id", dropna=True):
            label_id = int(label_id)
            combined = pd.concat([buckets.get(label_id, pd.DataFrame()), group], ignore_index=True)
            if len(combined) > per_class_cap:
                combined = combined.sample(per_class_cap, random_state=config.RANDOM_SEED)
            buckets[label_id] = combined
    if not buckets:
        return pd.DataFrame()
    df = pd.concat(buckets.values(), ignore_index=True)
    if len(df) > sample_rows:
        df = df.sample(sample_rows, random_state=config.RANDOM_SEED)
    return df


def load_track_table(dynamic_only: bool = False) -> pd.DataFrame:
    """Load the track-level aggregated feature table."""
    path = config.OUTPUT_ROOT / "tables" / "radar_track_features.csv"
    if not path.exists():
        raise FileNotFoundError(f"Build features first: {path}")
    df = pd.read_csv(path)
    if dynamic_only:
        df = df[df["label_id"] != 11]
    return df.dropna(subset=["label_id"])
