# Final RadarScenes Project Report

## Executive Summary

This project builds a reproducible machine-learning pipeline for the RadarScenes dataset. The final submission uses 50 RadarScenes sequences, extracts radar detections from `radar_data.h5`, links them with sequence metadata, engineers point-level and track-level features, evaluates supervised classifiers with sequence-aware splits, and explains the results using feature importance, PCA, confidence intervals, and repeated robustness checks.

The central scientific finding is that object-level context matters. Classifying one radar detection at a time is difficult because a single radar return is sparse and ambiguous. Aggregating detections that share the same tracked-object identity gives the model motion consistency, spatial structure, sensor coverage, density, and persistence over time. This substantially improves performance.

## Final Headline Results

| Task | Model | Accuracy | Macro F1 | Confidence interval method |
|---|---|---:|---:|---|
| Point-level dynamic-only benchmark | Random Forest | 0.3714 [0.1118, 0.5158] | 0.3172 [0.1492, 0.3966] | sequence-bootstrap on saved test predictions |
| Point-level repeated robustness | Random Forest | 0.3162 [0.2462, 0.3863] | 0.2141 [0.1732, 0.2550] | repeated sequence-level splits, `n=10` |
| Track-level repeated benchmark | Random Forest | 0.7583 [0.7214, 0.7952] | 0.5211 [0.4545, 0.5877] | repeated sequence-level splits, `n=10` |
| Track-level repeated benchmark | XGBoost | 0.7757 [0.7349, 0.8165] | 0.5370 [0.4405, 0.6334] | repeated sequence-level splits, `n=10` |

The point-level result is the canonical full dynamic-only point benchmark. The repeated point-level result is a lightweight robustness estimate on a sampled/balanced subset. The track-level results use the full track feature table with 2,175 aggregated rows and sequence-level repeated splitting.

## Data Artifacts

The final project keeps full source-derived tables separate from lightweight previews.

| Artifact | Path | Rows | Purpose |
|---|---|---:|---|
| Full point-level table | `radarscenes_project/outputs/tables/radar_points_features.csv` | 28,749,044 | Main full detection-level feature table |
| Full track-level table | `radarscenes_project/outputs/tables/radar_track_features.csv` | 2,175 | Main object-level aggregated feature table |
| Sampled Excel preview | `radarscenes_project/outputs/tables/radar_points_sample_for_excel_and_3d.xlsx` | sampled | Human inspection and 3D website preview |
| Submission preview copy | `FINAL_SUBMISSION/supporting_outputs/radar_points_sample_for_excel_and_3d.xlsx` | sampled | Portable final-submission preview |
| Track benchmark tables | `radarscenes_project/outputs/tables/advanced_analysis/track_level_repeated_split_summary.csv` | summary | Track-level repeated evaluation |
| Feature importance table | `radarscenes_project/outputs/tables/advanced_analysis/track_level_xgboost_feature_importance_repeated.csv` | summary | Track-level XGBoost interpretation |

The sampled Excel files are not used as the authoritative training data. They exist so the professor can inspect representative radar rows without opening a 28.7-million-row CSV.

## Dataset Structure

RadarScenes contains 158 available sequences. This project processes 50 sequences by default to keep the pipeline computationally feasible while still moving beyond the original Sequence 102 prototype.

For the 50-sequence subset:

| Quantity | Count |
|---|---:|
| Sequences used | 50 |
| Scenes | 227,089 |
| Point-level radar detections | 28,749,044 |
| Tracked detections | 1,037,232 |
| Untracked detections | 27,711,812 |
| Track-level rows | 2,175 |
| Camera images | 10,971 |
| Odometry samples | 438,809 |

### Sequence, Scene, Detection, UUID, Track

| Concept | Meaning in this project |
|---|---|
| Sequence | One continuous RadarScenes driving recording folder, for example `sequence_102` |
| Scene | A metadata record in `scenes.json` tied to a timestamp and sensor event |
| Detection / point | One row in `radar_data.h5/radar_data`; one radar return |
| UUID | Unique identifier for one detection |
| Track ID | Dynamic-object identity over time, when available |
| Track aggregation | Grouping detections from the same tracked object and computing summary features |
| Machine-learning row | Either one point-level detection row or one track-level aggregated object row |

One radar detection is the same as one radar point in this project. One scene is not the same as one detection: a scene points to a block of detections recorded for a sensor-time event.

### Sensors and Timing

RadarScenes uses four automotive radar sensors. Their mounting positions and orientations are defined in `data/data/sensors.json`.

| Sensor | ID | x position | y position | yaw |
|---|---:|---:|---:|---:|
| `radar_1` | 1 | 3.663 | -0.873 | -1.48418552 |
| `radar_2` | 2 | 3.86 | -0.7 | -0.436185662 |
| `radar_3` | 3 | 3.86 | 0.7 | 0.436 |
| `radar_4` | 4 | 3.663 | 0.873 | 1.484 |

The sensors are interleaved rather than perfectly simultaneous. In the processed subset, the median all-scene timestamp gap is about 0.0155 seconds, while the median per-sensor timestamp gap is about 0.074 seconds, corresponding to roughly 13.5 Hz per sensor.

## Raw and Engineered Features

Each point-level row contains physical radar measurements and coordinates:

| Feature | Physical meaning | Why it may help |
|---|---|---|
| `range_sc` | radial distance in sensor coordinates | objects and noise vary with distance |
| `azimuth_sc` | bearing angle in sensor coordinates | object position relative to sensor |
| `rcs` | radar cross-section / reflectivity | material, shape, orientation, and size evidence |
| `vr` | measured radial velocity | movement toward/away from sensor |
| `vr_compensated` | radial velocity after ego-motion compensation | closer to object motion, not vehicle motion |
| `x_cc`, `y_cc` | position in car coordinates | spatial context around the ego vehicle |
| `x_seq`, `y_seq` | position in global sequence coordinates | trajectory/location context |
| `sensor_id` | radar sensor identity | viewing angle and coverage information |

The track-level table aggregates detections sharing the same object identity. It uses:

```text
sequence_id + track_id_clean + label_id + label_name
```

and computes object-level statistics such as means, standard deviations, min/max values, spatial spans, duration, sensor coverage, point density, and motion summaries.

Important interpretation: `num_points` is not pure object size. It mixes object size, observation duration, radar visibility, range, reflectivity, aspect angle, sensor coverage, and tracking behavior.

## Feature Meaning and Radar Physics

Radar Cross Section (`rcs`) is provided by the dataset. It is not recalculated from scratch. Conceptually, RCS measures how strongly an object reflects radar energy back to the sensor. A simplified monostatic radar equation is:

```text
P_r = (P_t G_t G_r lambda^2 sigma) / ((4 pi)^3 R^4 L)
```

where `P_r` is received power, `P_t` is transmitted power, `G_t` and `G_r` are antenna gains, `lambda` is wavelength, `sigma` is radar cross section, `R` is range, and `L` represents losses.

This equation explains why RCS is useful but not dominant. Reflectivity changes with range, orientation, material, occlusion, and multipath. The same object can look different from different angles, and different objects can produce similar RCS values.

The dataset also provides Cartesian coordinates. The conceptual polar-to-Cartesian conversion is:

```text
x = range_sc * cos(azimuth_sc)
y = range_sc * sin(azimuth_sc)
```

The project uses the supplied `x_cc`, `y_cc`, `x_seq`, and `y_seq` rather than recomputing them.

## Methods

### Supervised Learning Problem

The point-level classification problem is:

```text
Given x_i = [range, azimuth, rcs, velocity, coordinates, sensor_id],
predict y_i = label_id.
```

Two point-level experiments were run:

| Experiment | Classes |
|---|---|
| Experiment A | all classes, including static environment |
| Experiment B | dynamic-only classes, excluding `label_id = 11` |

The track-level classification problem uses aggregated object rows instead of individual detections.

### Models Actually Trained

The project trained and evaluated:

| Model | Status | Role |
|---|---|---|
| Logistic Regression | implemented | interpretable linear baseline |
| Gaussian Naive Bayes | implemented | course-aligned probabilistic baseline |
| KNN | implemented | non-parametric local baseline |
| Linear SVM | implemented | margin-based baseline |
| RBF SVM | implemented but skipped in fast/default runs | expensive non-linear baseline |
| Random Forest | implemented | strongest point-level baseline |
| XGBoost | implemented | strongest track-level model |
| MLPClassifier | implemented | neural-network baseline |
| CNN | exploratory | rasterized radar tracks, not final benchmark |

Gaussian Naive Bayes is explained through maximum likelihood estimation: for each class, the model estimates class-wise Gaussian means and variances from the training data. Method of moments is mostly a theoretical connection here; the practical baseline uses the standard Gaussian Naive Bayes fitting procedure.

### Evaluation Metrics

Accuracy measures the fraction of correct predictions:

```text
accuracy = correct predictions / all predictions
```

For class `k`:

```text
precision_k = TP_k / (TP_k + FP_k)
recall_k = TP_k / (TP_k + FN_k)
F1_k = 2 * precision_k * recall_k / (precision_k + recall_k)
```

Macro F1 averages F1 equally across classes:

```text
macro_F1 = mean(F1_k over classes)
```

Macro F1 is emphasized because RadarScenes is class-imbalanced. Accuracy can look acceptable while rare classes are poorly recognized.

### Split Strategy and Leakage Control

Final evaluations use sequence-aware splitting whenever possible. This matters because detections from the same sequence are correlated. If detections from the same sequence appeared in both train and test, performance could be inflated by shared context.

For repeated robustness and track-level evaluation, the project uses sequence-level splitting with disjoint train/test sequences. This means the model is tested on sequences it did not see during training.

## Point-Level Results

The best point-level dynamic-only benchmark was Random Forest:

| Metric | Value |
|---|---:|
| Accuracy | 0.3714 |
| Macro precision | 0.2933 |
| Macro recall | 0.4084 |
| Macro F1 | 0.3172 |
| Weighted F1 | 0.3045 |

XGBoost was also evaluated at point level:

| Experiment | Accuracy | Macro F1 |
|---|---:|---:|
| All classes | 0.1443 | 0.0940 |
| Dynamic-only | 0.2289 | 0.1709 |

In the point-level task, XGBoost did not outperform Random Forest. This is not a contradiction: gradient boosting can be strong, but here the individual detections are sparse and class-overlapping. Random Forest remained the strongest point-level baseline among the tested models.

## Confidence Intervals and Robustness

For the final point-level Random Forest benchmark, 95% confidence intervals were computed from saved predictions using sequence-level bootstrap resampling. This is more conservative than point-level bootstrap because detections inside a sequence are correlated.

| Metric | Point estimate | 95% CI |
|---|---:|---:|
| Accuracy | 0.3714 | [0.1118, 0.5158] |
| Macro precision | 0.2933 | [0.1667, 0.3795] |
| Macro recall | 0.4084 | [0.3006, 0.4612] |
| Macro F1 | 0.3172 | [0.1492, 0.3966] |

A 95% confidence interval gives a plausible range for the metric under repeated sampling of held-out sequences. It prevents overclaiming a single split result as an exact universal value.

### Repeated Point-Level Robustness

A lightweight repeated robustness study was run with:

- dynamic-only point-level task
- same 50-sequence project subset
- sequence-level `GroupShuffleSplit`
- 10 repetitions with seeds 42 through 51
- Logistic Regression, Random Forest, XGBoost, KNN, and Gaussian Naive Bayes
- sampling for speed: `sample_rows=60,000`, `max_balanced_rows=50,000`

Because this run is sampled, it is a robustness estimate, not a replacement for the canonical full point-level benchmark.

| Model | Accuracy mean [95% CI] | Macro F1 mean [95% CI] | Macro F1 std |
|---|---:|---:|---:|
| Random Forest | 0.3162 [0.2462, 0.3863] | 0.2141 [0.1732, 0.2550] | 0.0572 |
| Logistic Regression | 0.3368 [0.2343, 0.4393] | 0.2102 [0.1519, 0.2684] | 0.0814 |
| KNN | 0.3251 [0.2397, 0.4105] | 0.2088 [0.1521, 0.2656] | 0.0793 |
| Gaussian Naive Bayes | 0.3216 [0.1877, 0.4556] | 0.2072 [0.1404, 0.2741] | 0.0934 |
| XGBoost | 0.2938 [0.2299, 0.3578] | 0.2032 [0.1664, 0.2401] | 0.0515 |

Interpretation: point-level classification is sensitive to which sequences are held out. Random Forest remains strongest by macro F1, but model rankings are not perfectly stable across repeated splits.

## PCA and Feature Structure

PCA was used for interpretation, not as the final classifier.

| Cumulative variance target | Components needed |
|---|---:|
| 80% | 7 |
| 90% | 9 |
| 95% | 10 |
| 99% | 14 |

PCA shows that radar features are partially redundant. The first principal component is dominated by range, position, and reflectivity structure, including `radius_cc`, `range_sc`, `x_cc`, and `rcs`. The second component is driven more by angular and velocity behavior, including `azimuth_sc`, trigonometric azimuth features, `abs_vr`, `vr`, and `vr_compensated`.

The conclusion is not that features should simply be discarded. Instead, PCA confirms the physical grouping of the feature space: geometry, motion, angle, and reflectivity are related families of information.

## Feature Importance

Point-level feature analysis used mutual information, logistic regression coefficients, Random Forest importance, and permutation importance. These methods measure complementary ideas:

| Method | What it measures |
|---|---|
| Mutual information | non-linear dependence between feature and label |
| Logistic regression coefficients | scaled linear discriminative weight |
| Random Forest importance | impurity reduction from tree splits |
| Permutation importance | performance drop after shuffling one feature |

The main interpretation is that radar classification is not explained by a single variable. Motion, position, density, sensor coverage, and reflectivity work together. Agreement across methods is stronger evidence than any single ranking.

## Point-Level vs Track-Level Classification

Track-level classification is the strongest scientific result of the project.

Point-level classification asks the model to classify one instantaneous radar return. This is difficult because a single detection may contain only one noisy measurement of range, angle, reflectivity, and radial velocity. Many classes overlap at this level.

Track-level classification groups detections belonging to the same tracked dynamic object. The model then sees an object profile over time: how long the object persists, how it moves, how many sensors see it, how compact or spread out it is, and how its radar returns vary.

| Setting | Model | Accuracy | Macro F1 |
|---|---|---:|---:|
| Point-level canonical benchmark | Random Forest | 0.3714 [0.1118, 0.5158] | 0.3172 [0.1492, 0.3966] |
| Point-level repeated robustness | Random Forest | 0.3162 [0.2462, 0.3863] | 0.2141 [0.1732, 0.2550] |
| Track-level repeated benchmark | Random Forest | 0.7583 [0.7214, 0.7952] | 0.5211 [0.4545, 0.5877] |
| Track-level repeated benchmark | XGBoost | 0.7757 [0.7349, 0.8165] | 0.5370 [0.4405, 0.6334] |

### Why Track-Level Works Better

Aggregating radar detections into tracked objects substantially improves classification performance because the model can exploit:

- temporal behavior and persistence
- ego-motion-compensated velocity statistics
- motion consistency across detections
- spatial envelope and geometry
- sensor coverage
- point density and compactness
- variation in RCS rather than a single RCS value

This is why track-level XGBoost and Random Forest outperform the point-level benchmark.

### Track-Level XGBoost Feature Importance

The best track-level model was XGBoost. Its top repeated-split feature importances were:

| Rank | Feature | Interpretation |
|---:|---|---|
| 1 | `mean_abs_vr_compensated` | object motion after removing ego-vehicle motion |
| 2 | `max_x_cc` | forward spatial extent relative to ego vehicle |
| 3 | `number_of_sensors_seen` | multi-sensor visibility and viewpoint coverage |
| 4 | `point_density` | compactness/persistence of returns |
| 5 | `mean_xy_product` | coupled spatial geometry around the ego vehicle |

Motion dominates because autonomous-driving classes often differ by behavior. A pedestrian, bicycle, car, and truck may overlap in reflectivity at one instant, but their compensated motion patterns and spatial footprints are more informative over time.

Geometry matters because object classes occupy different spatial envelopes and viewpoints around the vehicle. Sensor coverage matters because an object seen by multiple radars provides a richer and more reliable signature. RCS remains useful, especially through variability such as `std_rcs`, but it is not dominant because reflectivity depends strongly on material, aspect angle, distance, occlusion, and multipath.

### Leakage Audit

The track-level improvement appears genuine:

- train/test splitting is sequence-level, so train and test sequences are disjoint
- no raw `track_id` string is used as a model feature
- label columns are excluded from training features
- `num_points`, duration, density, spread, and sensor counts are derived from detections, not labels
- no object track appears in both train and test when sequences are disjoint

The aggregation key includes `sequence_id`, `track_id_clean`, `label_id`, and `label_name` to build a clean table, but target label fields are not used as predictors.

## Sequence 102 Prototype vs Final Generalization

Sequence 102 was used historically as a development environment. It helped debug loading, visualization, feature extraction, and early modeling. High prototype results from that phase should not be treated as final benchmark results.

The final project expands to 50 sequences and reports multi-sequence, sequence-aware evaluation. Scores are lower but more defensible because the task is harder and closer to generalization.

## Website and Presentation

The final website is located at:

```text
FINAL_SUBMISSION/project_website/index.html
```

It includes:

- dataset overview
- radar feature explanations
- camera/radar/odometry timestamp linking
- final point-level and track-level results
- PCA and feature-importance interpretation
- confidence interval explanation
- interactive 3D radar feature-space controls
- links to the final report, deck, speaker script, and supporting outputs

The final presentation deck is:

```text
FINAL_SUBMISSION/RadarScenes_Final_Presentation_CLEAN_FINAL_WITH_SCENE_COMPARISON_ALL_MODELS_COURSE_CONTEXT_UPDATED_AUDIT.pptx
```

The final speaker script is:

```text
FINAL_SUBMISSION/RadarScenes_Final_Presentation_FINAL_SCRIPT_COMPREHENSIVE.docx
```

## Limitations

This is a rigorous radar-feature baseline, not a complete production perception system.

Main limitations:

- only 50 of 158 available RadarScenes sequences are processed in the final pipeline
- point-level classification remains difficult because individual radar detections are sparse
- rare classes remain hard because of class imbalance
- camera images are timestamp-aligned for context but not used as classifier input
- no calibrated radar-camera projection is claimed
- CNN work is exploratory and uses rasterized radar tracks, not camera images
- hyperparameters are reasonable baselines, not exhaustive optimized searches
- `sensor_id` is treated simply in baseline models; richer categorical encoding could be tested

## Professor Defense Answers

| Question | Defensible answer |
|---|---|
| Why only 50 sequences? | The full dataset has 158 sequences, but 50 sequences are enough to demonstrate a general pipeline while keeping runtime feasible. |
| Why Sequence 102 first? | It was a prototype/debugging environment, not the final benchmark. |
| Why are point-level scores low? | One radar point is sparse and ambiguous; many classes overlap at a single-detection level. |
| Why does track-level classification improve? | It adds temporal behavior, motion consistency, spatial structure, sensor coverage, and persistence. |
| Why does `track_id` not guarantee perfect classification? | It groups detections but does not make classes physically separable; tracks can be short, noisy, or ambiguous. |
| Why is motion more informative than RCS? | Compensated motion reflects object behavior, while RCS changes with material, angle, range, and occlusion. |
| Why use macro F1? | Macro F1 weights classes equally and is more honest under class imbalance. |
| Was CNN validated as final? | No. CNN is exploratory and not the final benchmark. |

## Final Conclusion

The project follows a clear scientific progression:

1. Start with individual RadarScenes radar detections.
2. Build point-level classifiers using physical radar features.
3. Analyze feature importance, redundancy, and PCA structure.
4. Quantify uncertainty with confidence intervals and repeated sequence-level splits.
5. Aggregate detections into tracked objects.
6. Show that object-level context substantially improves classification.
7. Interpret the improvement through motion, geometry, density, sensor coverage, and radar physics.

The strongest final conclusion is that radar perception improves when sparse detections are connected into tracked objects. The best model is not just better numerically; it is better for a physically meaningful reason: it can use motion and spatial context rather than relying on a single radar return.
