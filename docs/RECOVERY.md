# Recovering an interrupted training run

A full comparison trains 9 model families × 10 stocks × 2 horizons with expanding-window cross-validation. In one long process, TensorFlow memory can build up until the operating system kills the run. `recover_prd.py` resumes the experiment by running **one stock/horizon/model job per fresh Python process**, so memory is released after each model.

Each deep job keeps the original five expanding folds, final training, architecture, batch size, seed and early stopping. Isolating jobs prevents memory build-up across jobs. It cannot guarantee that a single job fits in every machine's memory.

## Usage

Make sure no other training command is running, then:

```bash
python -m unittest discover -s tests/prd -p test_recovery.py -v
python -u recover_prd.py
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--sources` | `reports/prd/prd-classical-2 reports/prd/prd-deep-full` | Earlier run folders whose completed work is reused |
| `--run-name` | `prd-recovered-full` | Combined output folder under `reports/prd/` |
| `--config` | `config.yaml` | Must match the configuration of the source runs |

## How it works

- Completed classical predictions are reused after verification.
- Deep models already saved in an interrupted run are reloaded **without refitting**. Their predictions are regenerated and checked against the original recorded metrics. A mismatch stops the run so it can be investigated.
- A saved model is reusable only if its weights, scalers, metadata, all three metric rows, and full cross-validation records and loss curves are present. Otherwise only that model is retrained.
- Before combining results, the runner checks data hashes, configuration, `src/prd` source hashes and package versions against the source runs.
- The final model for each stock and horizon is selected on validation MAE only. Origins, target dates and observed prices must match exactly across models.

Console messages:

| Message | Meaning |
| --- | --- |
| `Checking saved predictions` | Verifying an existing completed result |
| `Recovering saved model without fitting` | Reloading weights and regenerating predictions |
| `Training in a fresh process` | Fitting one missing model; long quiet periods are normal |
| `Already saved` | A verified checkpoint from an earlier recovery attempt |

## Output

Results go to `reports/prd/prd-recovered-full/`. Working child runs are named `prd-recovered-full-work-*`. Source run folders are never modified. The combined folder has the standard layout (`predictions.csv`, `metrics.csv`, `selected_models.csv`, CV scores, models, manifest), so the `eda`, `portfolio`, `sentiment` and `report` commands and the dashboard work on it directly.

## If it is interrupted again

Run the same command again with the same run name and sources. Verified checkpoints are skipped. An unfinished job restarts from its beginning, because training does not resume mid-epoch. Corrupted checkpoints or changed settings stop the run instead of being overwritten. A lock file prevents two recoveries from running at the same time.

> **Note:** Source hashes are part of the consistency check. Editing files in `src/prd/` after the source runs were produced will make the runner refuse to combine them. In that case, retrain the source runs with the current code.
