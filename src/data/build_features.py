"""Build point-level and track-level feature tables from up to 50 sequences."""

from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config  # noqa: E402
from src.utils import add_label_names, clean_track_ids, discover_sequences, ensure_output_dirs, engineer_radar_features, iter_radar_chunks, timestamp_to_seconds  # noqa: E402

# Temporary directory for per-sequence Parquet files (fast intermediate format).
_PARQUET_CACHE = config.OUTPUT_ROOT / "tables" / "_seq_cache"


def _extract_one_sequence(seq: Path, chunk_size: int = 250_000) -> Path:
    """Extract features for a single sequence and save as a Parquet file."""
    out_path = _PARQUET_CACHE / f"{seq.name}.parquet"
    if out_path.exists():
        return out_path  # Already processed — skip.
    parts = []
    for chunk in iter_radar_chunks(seq / "radar_data.h5", chunk_size=chunk_size):
        chunk.insert(0, "sequence_id", seq.name)
        chunk = add_label_names(chunk)
        chunk = engineer_radar_features(chunk)
        for col in config.POINT_COLUMNS:
            if col not in chunk.columns:
                chunk[col] = pd.NA
        out_cols = config.POINT_COLUMNS + [c for c in config.RADAR_ENGINEERED_FEATURES if c in chunk.columns]
        parts.append(chunk[out_cols])
    if parts:
        pd.concat(parts, ignore_index=True).to_parquet(out_path, index=False)
    return out_path


def append_point_features(sequences: list[Path], output_csv: Path, chunk_size: int = 250_000) -> None:
    """Extract features for all sequences and write a single combined CSV."""
    _PARQUET_CACHE.mkdir(parents=True, exist_ok=True)
    parquet_files = []
    for seq in sequences:
        print(f"Extracting point features from {seq.name}")
        parquet_files.append(_extract_one_sequence(seq, chunk_size=chunk_size))

    # Merge all per-sequence Parquet files into one CSV in a single pass.
    print("Merging all sequences into CSV...")
    first_write = True
    for pf in parquet_files:
        df = pd.read_parquet(pf)
        df.to_csv(output_csv, mode="w" if first_write else "a", header=first_write, index=False)
        first_write = False


def build_track_features(point_csv: Path, output_csv: Path, chunk_size: int = 500_000) -> None:
    partials = []
    usecols = [
        "sequence_id", "timestamp", "sensor_id", "range_sc", "azimuth_sc", "rcs", "vr",
        "vr_compensated", "x_cc", "y_cc", "x_seq", "y_seq", "radius_cc", "abs_vr", "abs_vr_compensated",
        "azimuth_sin", "azimuth_cos", "xy_product", "rcs_per_range", "track_id", "label_id", "label_name",
    ]
    dtype_hints = {"uuid": "string", "track_id": "string", "label_name": "string"}
    for chunk in pd.read_csv(point_csv, usecols=usecols, chunksize=chunk_size, dtype=dtype_hints, low_memory=False):
        chunk = clean_track_ids(chunk)
        chunk = chunk[chunk["track_id_clean"] != "untracked"]
        if chunk.empty:
            continue
        grouped = chunk.groupby(["sequence_id", "track_id_clean", "label_id", "label_name"], dropna=False)
        part = grouped.agg(
            num_points=("range_sc", "size"),
            mean_range=("range_sc", "mean"),
            std_range=("range_sc", "std"),
            mean_rcs=("rcs", "mean"),
            std_rcs=("rcs", "std"),
            mean_vr=("vr", "mean"),
            std_vr=("vr", "std"),
            mean_vr_compensated=("vr_compensated", "mean"),
            std_vr_compensated=("vr_compensated", "std"),
            mean_azimuth=("azimuth_sc", "mean"),
            std_azimuth=("azimuth_sc", "std"),
            mean_x_cc=("x_cc", "mean"),
            mean_y_cc=("y_cc", "mean"),
            mean_x_seq=("x_seq", "mean"),
            mean_y_seq=("y_seq", "mean"),
            std_x_cc=("x_cc", "std"),
            std_y_cc=("y_cc", "std"),
            mean_radius_cc=("radius_cc", "mean"),
            std_radius_cc=("radius_cc", "std"),
            mean_abs_vr=("abs_vr", "mean"),
            mean_abs_vr_compensated=("abs_vr_compensated", "mean"),
            mean_azimuth_sin=("azimuth_sin", "mean"),
            mean_azimuth_cos=("azimuth_cos", "mean"),
            mean_xy_product=("xy_product", "mean"),
            mean_rcs_per_range=("rcs_per_range", "mean"),
            number_of_sensors_seen=("sensor_id", "nunique"),
            min_timestamp=("timestamp", "min"),
            max_timestamp=("timestamp", "max"),
            min_x_cc=("x_cc", "min"),
            max_x_cc=("x_cc", "max"),
            min_y_cc=("y_cc", "min"),
            max_y_cc=("y_cc", "max"),
            min_radius_cc=("radius_cc", "min"),
            max_radius_cc=("radius_cc", "max"),
        ).reset_index()
        partials.append(part)

    if not partials:
        pd.DataFrame().to_csv(output_csv, index=False)
        return

    all_parts = pd.concat(partials, ignore_index=True)
    grouped = all_parts.groupby(["sequence_id", "track_id_clean", "label_id", "label_name"], dropna=False)
    tracks = grouped.agg(
        num_points=("num_points", "sum"),
        mean_range=("mean_range", "mean"),
        std_range=("std_range", "mean"),
        mean_rcs=("mean_rcs", "mean"),
        std_rcs=("std_rcs", "mean"),
        mean_vr=("mean_vr", "mean"),
        std_vr=("std_vr", "mean"),
        mean_vr_compensated=("mean_vr_compensated", "mean"),
        std_vr_compensated=("std_vr_compensated", "mean"),
        mean_azimuth=("mean_azimuth", "mean"),
        std_azimuth=("std_azimuth", "mean"),
        mean_x_cc=("mean_x_cc", "mean"),
        mean_y_cc=("mean_y_cc", "mean"),
        mean_x_seq=("mean_x_seq", "mean"),
        mean_y_seq=("mean_y_seq", "mean"),
        std_x_cc=("std_x_cc", "mean"),
        std_y_cc=("std_y_cc", "mean"),
        mean_radius_cc=("mean_radius_cc", "mean"),
        std_radius_cc=("std_radius_cc", "mean"),
        mean_abs_vr=("mean_abs_vr", "mean"),
        mean_abs_vr_compensated=("mean_abs_vr_compensated", "mean"),
        mean_azimuth_sin=("mean_azimuth_sin", "mean"),
        mean_azimuth_cos=("mean_azimuth_cos", "mean"),
        mean_xy_product=("mean_xy_product", "mean"),
        mean_rcs_per_range=("mean_rcs_per_range", "mean"),
        number_of_sensors_seen=("number_of_sensors_seen", "max"),
        min_timestamp=("min_timestamp", "min"),
        max_timestamp=("max_timestamp", "max"),
        min_x_cc=("min_x_cc", "min"),
        max_x_cc=("max_x_cc", "max"),
        min_y_cc=("min_y_cc", "min"),
        max_y_cc=("max_y_cc", "max"),
        min_radius_cc=("min_radius_cc", "min"),
        max_radius_cc=("max_radius_cc", "max"),
    ).reset_index().rename(columns={"track_id_clean": "track_id"})
    tracks["duration_seconds"] = (tracks["max_timestamp"] - tracks["min_timestamp"]).map(timestamp_to_seconds)
    tracks["x_span"] = tracks["max_x_cc"] - tracks["min_x_cc"]
    tracks["y_span"] = tracks["max_y_cc"] - tracks["min_y_cc"]
    tracks["radius_span"] = tracks["max_radius_cc"] - tracks["min_radius_cc"]
    tracks["point_density"] = tracks["num_points"] / tracks["duration_seconds"].replace(0, pd.NA)
    tracks["points_per_sensor"] = tracks["num_points"] / tracks["number_of_sensors_seen"].replace(0, pd.NA)
    tracks.drop(columns=["min_timestamp", "max_timestamp"]).to_csv(output_csv, index=False)


def main() -> None:
    ensure_output_dirs()
    sequences = discover_sequences()
    point_csv = config.OUTPUT_ROOT / "tables" / "radar_points_features.csv"
    track_csv = config.OUTPUT_ROOT / "tables" / "radar_track_features.csv"
    append_point_features(sequences, point_csv)
    print("Aggregating track features")
    build_track_features(point_csv, track_csv)
    print(f"Saved {point_csv}")
    print(f"Saved {track_csv}")


if __name__ == "__main__":
    main()
