"""Project-wide configuration for the RadarScenes classification pipeline."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# The downloaded RadarScenes archive in this workspace is expected to contain
# sequence folders such as data/data/sequence_1/radar_data.h5.
DATA_ROOT = PROJECT_ROOT / "data" / "data"
OUTPUT_ROOT = PROJECT_ROOT / "outputs"

MAX_SEQUENCES = 50
RANDOM_SEED = 42

LABEL_MAP = {
    0: "passenger car",
    1: "large vehicle",
    2: "truck",
    3: "bus",
    4: "train",
    5: "bicycle",
    6: "motorized two-wheeler",
    7: "pedestrian",
    8: "pedestrian group",
    9: "animal",
    10: "other dynamic object",
    11: "static environment",
}

RADAR_BASE_FEATURES = [
    "range_sc",
    "azimuth_sc",
    "rcs",
    "vr",
    "vr_compensated",
    "x_cc",
    "y_cc",
    "x_seq",
    "y_seq",
    "sensor_id",
]

# Derived from the raw radar measurements. These are intentionally simple and
# physics-aware rather than opaque transformations.
RADAR_ENGINEERED_FEATURES = [
    "radius_cc",
    "abs_vr",
    "abs_vr_compensated",
    "azimuth_sin",
    "azimuth_cos",
    "xy_product",
    "rcs_per_range",
]

RADAR_FEATURES = RADAR_BASE_FEATURES + RADAR_ENGINEERED_FEATURES

# Aggregated track-level features produced by build_features.py.
# These summarise hundreds of per-point measurements into one row per
# physical object, enabling the Track-Level experiment described in the report.
TRACK_FEATURES = [
    "num_points",
    "mean_range",
    "std_range",
    "mean_rcs",
    "std_rcs",
    "mean_vr",
    "std_vr",
    "mean_vr_compensated",
    "std_vr_compensated",
    "mean_azimuth",
    "std_azimuth",
    "mean_x_cc",
    "mean_y_cc",
    "std_x_cc",
    "std_y_cc",
    "mean_radius_cc",
    "std_radius_cc",
    "mean_abs_vr",
    "mean_abs_vr_compensated",
    "mean_azimuth_sin",
    "mean_azimuth_cos",
    "mean_xy_product",
    "mean_rcs_per_range",
    "number_of_sensors_seen",
    "x_span",
    "y_span",
    "radius_span",
    "duration_seconds",
    "point_density",
    "points_per_sensor",
]

POINT_COLUMNS = [
    "sequence_id",
    "timestamp",
    "sensor_id",
    "range_sc",
    "azimuth_sc",
    "rcs",
    "vr",
    "vr_compensated",
    "x_cc",
    "y_cc",
    "x_seq",
    "y_seq",
    "uuid",
    "track_id",
    "label_id",
    "label_name",
]

PLOT_STYLE = {
    "figure.figsize": (10, 6),
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 10,
}

# Discrete qualitative palettes. These are intentionally dictionaries rather
# than continuous colormaps so class and sensor IDs always use legend entries.
LABEL_COLORS = {
    0: "#1f77b4",
    1: "#ff7f0e",
    2: "#2ca02c",
    3: "#d62728",
    4: "#9467bd",
    5: "#8c564b",
    6: "#e377c2",
    7: "#7f7f7f",
    8: "#bcbd22",
    9: "#17becf",
    10: "#aec7e8",
    11: "#ffbb78",
}

SENSOR_COLORS = {
    0: "#4e79a7",
    1: "#f28e2b",
    2: "#e15759",
    3: "#76b7b2",
}
