# ML pipeline architecture

## Overview

Binary classifier to distinguish CRY (cosmic ray) events from CE mix (conversion electron + backgrounds) using CRV coincidence features. The veto decision is event-level: an event is vetoed if ANY coincidence scores above threshold.

## Pipeline

```
AssembleDataset  -->  Train  -->  Validate
(assemble.py)       (train.py)   (validate.py)
                        ^
                        |
                    Optimise
                  (optimise.py)
```

## Modules

### AssembleDataset (`src/ml/assemble.py`)

Loads CRY and CE mix data, labels them (CRY=1, CE mix=0), combines, shuffles, and splits.

- **Input**: Processed data via `LoadML` (awkward arrays from ntuples)
- **Splitting**: `GroupShuffleSplit` (80/20) grouped by `subrun_event` so all coincidences from the same event stay together
- **Output**: Dictionary with `X_train`, `X_test`, `y_train`, `y_test`, `metadata_train`, `metadata_test`, `df_full`
- **Features dropped**: `d0`, `tanDip`, `maxr`, `mom_mag`, `PEs_per_hit`, `t0`, `timeStart`, `timeEnd`
- **Methods**:
  - `assemble_dataset()` -- returns data dict
  - `draw_features()` -- 2x4 grid of CRY vs CE mix feature distributions
  - `draw_cuts()` -- cut summary plots
  - `check_dT_window_results()` -- baseline dT window veto performance
  - `get_cut_flows()` -- cut flow tables

### Train (`src/ml/train.py`)

Trains a model on pre-split data. Supports XGBoost (default), sklearn, and Keras.

- **Input**: Data dict from `assemble_dataset()`
- **Scaling**: Optional `StandardScaler` (default on), fit on train only
- **Output**: Results dict with `model`, `scaler`, `feature_names`, `X_test`, `y_test`, `y_pred`, `y_proba`, `metadata_test`, etc.
- **Methods**:
  - `train(tag, **hyperparams)` -- single train/test run
  - `train_cv(tag, n_folds=5, min_eff=0.999, **hyperparams)` -- K-fold CV for robust threshold estimation, then trains final model on original split. Returns results dict with `cv_threshold` and `cv_metrics` attached.

### Validate (`src/ml/validate.py`)

Validates a trained model. All methods operate on the test set unless otherwise noted.

- **Input**: Results dict from `Train.train()` or `Train.train_cv()`
- **Key design**: Event-level grouping throughout. Score distributions, threshold finding, and physics plots all use the max score per event, consistent with how the veto is actually applied.
- **Methods**:
  - `find_threshold(min_eff=0.999)` -- event-level threshold scan. Finds highest threshold maintaining target veto efficiency. Returns threshold + scan arrays.
  - `plot_threshold_cv(fold_scans, cv_threshold)` -- overlays per-fold threshold scans with mean curves and CV-averaged threshold
  - `plot_score_distribution(threshold)` -- event-level max score histograms (full range + zoomed to 2x threshold)
  - `plot_roc()` -- ROC curve
  - `plot_feature_importance()` -- supports tree-based (`feature_importances_`) and linear (`coef_`) models
  - `get_events_by_threshold(threshold, above=False)` -- returns DataFrame of all coincidences from events whose max score is above/below threshold
  - `plot_physics_by_score(threshold, above=False)` -- feature distributions for events above/below threshold
  - `money_table(X, y, metadata, threshold)` -- event-level comparison of ML veto vs dT window cut
  - `roc_auc()`, `confusion_matrix()`, `classification_report()`, `print_summary()`

### Optimise (`src/ml/optimise.py`)

Grid search over hyperparameters. Objective: minimise deadtime subject to veto efficiency >= 99.9%.

- **Input**: Data dict from `assemble_dataset()`
- **Methods**:
  - `grid_search(param_grid)` -- single train/test split per combo. Plots threshold overlay for winning combo only.
  - `grid_search_cv(param_grid, n_folds=5)` -- K-fold CV per combo. Reports mean +/- std for all metrics.
  - `get_summary()` -- results as DataFrame sorted by deadtime
  - `save_summary()` -- write to CSV

## Workflow

Currently implemented in seperate notebooks.

### 1. Assemble

```python
asm = AssembleDataset(run=run)
asm.draw_features()
data = asm.assemble_dataset()
```

### 2. Hyperparameter selection (optional)

```python
opt = Optimise(data, run=run, min_efficiency=0.999)
best = opt.grid_search_cv(param_grid, n_folds=5)
opt.save_summary()
best_hp = best["hyperparams"]
```

### 3. Train with CV threshold

```python
trainer = Train(data, run=run)
results = trainer.train_cv(tag="xgb_final", n_folds=5, **best_hp)
threshold = results["cv_threshold"]
```

`train_cv` runs K-fold CV to get a robust threshold estimate, then trains the final model on the original train/test split. Use `cv_threshold` for deployment, not the single-split threshold.

### 4. Analyse

```python
ana = Validate(results, run=run)
ana.roc_auc()
ana.plot_roc()
ana.plot_feature_importance()
ana.plot_score_distribution(threshold)
ana.plot_physics_by_score(threshold, above=False)
ana.plot_physics_by_score(threshold, above=True)

# Full comparison table
tables = ana.money_table(data["df_full"], data["df_full"]["label"],
                         data["df_full"][["event", "subrun"]], threshold=threshold)
display(tables["metrics"])
display(tables["confusion"])

# Inspect missed events
df_missed = ana.get_events_by_threshold(threshold, above=False)
display(df_missed[df_missed["label"] == 1])
```

## Key Design Decisions

### Event-level veto
An event is vetoed if ANY coincidence scores above threshold. This means:
- **Threshold finding**: groups by (subrun, event), takes max score per event, scans threshold against that
- **Score distributions**: plot max score per event, not per coincidence
- **Physics distributions**: select events by max score, then plot all coincidences from those events
- **Train/test splitting**: `GroupShuffleSplit` / `GroupKFold` keeps all coincidences from the same event together

### CV threshold estimation
A single train/test split gives a noisy threshold because the 99.9% veto efficiency boundary depends on very few CRY events (~4 out of ~4000). K-fold CV averages the threshold across folds. The performance metrics (deadtime, veto efficiency) are stable across folds; it is the threshold value itself that varies because the efficiency curve is flat over a wide range.

### Output paths
- **Images**: `output/images/ml/{run}/{tag}/`
- **Results** (CSVs, models): `output/ml/{run}/results/{tag}/`
- **Optimisation summary**: `output/ml/{run}/results/optimisation_summary.csv`
