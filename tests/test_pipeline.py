"""Tests on small synthetic tables, so they run without the RadarScenes dataset.

Run from the repository root with: python -m pytest
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.link_sensors import nearest_value
from src.models.train import split_by_sequence
from src.utils import (
    balanced_sample,
    camera_timestamp_from_name,
    clean_track_ids,
    discover_sequences,
    engineer_radar_features,
    sequence_sort_key,
)


def make_points(n_sequences=20, rows_per_sequence=30, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for seq in range(1, n_sequences + 1):
        for i in range(rows_per_sequence):
            rows.append({
                "sequence_id": f"sequence_{seq}",
                "label_id": i % 4,
                "f1": rng.normal(),
                "f2": rng.normal(),
            })
    return pd.DataFrame(rows)


def test_split_keeps_each_sequence_on_one_side():
    df = make_points()
    _, _, _, _, test_df = split_by_sequence(df, ["f1", "f2"])
    test_seqs = set(test_df["sequence_id"])
    train_seqs = set(df["sequence_id"]) - test_seqs

    assert test_seqs and train_seqs
    assert test_seqs.isdisjoint(train_seqs)
    # Every row of a test sequence ends up in the test set.
    assert len(test_df) == df["sequence_id"].isin(test_seqs).sum()


def test_split_test_classes_are_seen_in_training():
    df = make_points()
    # Class 9 appears in a single sequence, so that sequence must not be held out.
    df.loc[df["sequence_id"] == "sequence_3", "label_id"] = 9
    _, _, y_train, y_test, _ = split_by_sequence(df, ["f1", "f2"])
    assert set(y_test) <= set(y_train)


def test_split_is_reproducible():
    df = make_points()
    first = split_by_sequence(df, ["f1", "f2"])[4]
    second = split_by_sequence(df, ["f1", "f2"])[4]
    assert first.index.equals(second.index)


def test_engineer_radar_features():
    df = pd.DataFrame({
        "range_sc": [5.0, 0.0],
        "azimuth_sc": [0.0, np.pi / 2],
        "rcs": [10.0, 3.0],
        "vr": [-2.0, 1.0],
        "vr_compensated": [-0.5, 0.0],
        "x_cc": [3.0, 1.0],
        "y_cc": [4.0, 0.0],
    })
    out = engineer_radar_features(df)

    np.testing.assert_allclose(out["radius_cc"], [5.0, 1.0])
    assert out.loc[0, "rcs_per_range"] == pytest.approx(2.0)
    assert np.isnan(out.loc[1, "rcs_per_range"])  # zero range is not divided by
    np.testing.assert_allclose(out["abs_vr"], [2.0, 1.0])
    np.testing.assert_allclose(out["azimuth_sin"], [0.0, 1.0], atol=1e-12)
    np.testing.assert_allclose(out["azimuth_cos"], [1.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(out["xy_product"], [12.0, 0.0])
    assert "radius_cc" not in df.columns  # input is not modified


def test_sequences_sort_numerically():
    names = ["sequence_10", "sequence_2", "sequence_1"]
    ordered = sorted((Path(n) for n in names), key=sequence_sort_key)
    assert [p.name for p in ordered] == ["sequence_1", "sequence_2", "sequence_10"]


def test_discover_sequences(tmp_path):
    for n in [1, 2, 10]:
        seq = tmp_path / f"sequence_{n}"
        seq.mkdir()
        (seq / "radar_data.h5").touch()
    (tmp_path / "sequence_3").mkdir()  # no radar file, ignored

    found = discover_sequences(tmp_path, max_sequences=2)
    assert [p.name for p in found] == ["sequence_1", "sequence_2"]

    with pytest.raises(FileNotFoundError):
        discover_sequences(tmp_path / "missing")


def test_clean_track_ids():
    df = pd.DataFrame({"track_id": ["abc", "", "0", None, "7"]})
    out = clean_track_ids(df)
    assert list(out["track_id_clean"]) == ["abc", "untracked", "untracked", "untracked", "7"]


def test_camera_timestamp_from_name():
    assert camera_timestamp_from_name(Path("camera/156862647501.jpg")) == 156862647501
    assert camera_timestamp_from_name(Path("img_3_156862647501.jpg")) == 156862647501
    assert camera_timestamp_from_name(Path("camera.jpg")) is None


def test_nearest_value():
    values = np.array([100, 200, 300])
    assert nearest_value(values, 190) == 200
    assert nearest_value(values, 10) == 100
    assert nearest_value(values, 1000) == 300
    assert nearest_value(np.array([]), 5) is None


def test_balanced_sample_caps_each_class():
    df = pd.DataFrame({"label_id": [0] * 900 + [1] * 90 + [2] * 10})
    out = balanced_sample(df, max_rows=300)

    counts = out["label_id"].value_counts()
    assert len(out) <= 300
    assert counts[0] == 100  # 300 rows / 3 classes
    assert counts[1] == 90 and counts[2] == 10  # small classes kept whole
