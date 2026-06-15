# RadarScenes Classification Project

Semantic classification of automotive radar detections using the [RadarScenes](https://radar-scenes.com) dataset.  
The pipeline extracts point-level and track-level features from raw radar sequences and benchmarks several supervised classifiers (XGBoost, Random Forest, Logistic Regression, Gaussian NB) with strict sequence-level splitting to prevent data leakage.

## Repository Structure

```text
config.py                    # Paths, label maps, hyper-parameters
requirements.txt             # Python dependencies
src/
  data/
    build_features.py        # Point → Track feature extraction
    overview.py              # Dataset statistics and plots
    link_sensors.py          # Radar–camera–odometry timestamp alignment
  models/
    train.py                 # Experiment A (point-level) & B (track-level)
  utils.py                   # Shared helpers
outputs/
  tables/                    # Generated CSVs (metrics, overview)
  figures/                   # Generated plots
  models/                    # Serialised trained models (.joblib)
  website_react/             # Interactive React dashboard
```

## Data Setup

> **The dataset is not included in this repository due to its size (~15 GB).**

1. Download the RadarScenes dataset from the official website: <https://radar-scenes.com>
2. Extract the sequences into the `data/data/` directory so that each sequence folder sits directly under it:

```text
data/data/
  sequence_1/
    radar_data.h5
    scenes.json
    camera/
  sequence_2/
  ...
  sequence_50/
```

By default, the pipeline processes all sequences found in `data/data/`.  
You can adjust the maximum count in `config.py` (`MAX_SEQUENCES`).

## Setup & Execution

**Prerequisites (macOS only):** XGBoost requires the OpenMP runtime:
```bash
brew install libomp
```

**Step 1 — Create a virtual environment and install dependencies:**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Step 2 — Run the pipeline scripts in order:**
```bash
python src/data/overview.py          # Dataset statistics
python src/data/build_features.py    # Feature extraction (point + track)
python src/models/train.py           # Model training & evaluation
```

## Output

| Directory          | Contents                                           |
|--------------------|----------------------------------------------------|
| `outputs/tables/`  | CSV metrics (`training_summary_metrics.csv`, etc.) |
| `outputs/models/`  | Serialised classifiers (`.joblib`)                 |
| `outputs/figures/`  | Plots (label distribution, sensor coverage, etc.) |
| `outputs/website_react/` | Interactive React dashboard — run with `npm run dev` |
# RadarScenes
