# Running the regression workflow

This guide walks through the full regression pipeline, from raw data to the dashboard. All commands run from the repository root.

## 1. Environment

The reference environment uses Python 3.11.

```bash
conda create -n stock-ai python=3.11
conda activate stock-ai
pip install -r requirements-prd.txt
```

| File | Use |
| --- | --- |
| `requirements-prd.txt` | Regression workflow: data, classical and deep models, EDA, portfolio, dashboard |
| `requirements-news.txt` | PyTorch and Transformers, needed only to rescore articles with FinBERT |
| `requirements.txt` | Both of the above |

TensorFlow is required for the deep models and XGBoost for its regressor. VADER uses its bundled lexicon. Every run records its exact installed packages in `environment.lock.txt`.

## 2. Data

The pipeline expects `data/raw/prices.parquet` and `data/raw/macro.parquet`. To download a fresh price and macro snapshot (yfinance and FRED) and prepare it in one step:

```bash
python -m src.prd rebuild
```

If you already have the raw files, run only the preparation step:

```bash
python -m src.prd prepare
```

Preparation writes `data/processed/prd/` from the unscaled raw data. It prints date ranges, feature counts, split boundaries and file hashes, and writes `data_quality_report.md` there. If it reports a long price gap or an invalid OHLC row, inspect the raw observation instead of removing the check.

## 3. First checkpoint

Run the tests and a single baseline before any long experiment:

```bash
python -m unittest discover -s tests/prd -v
python -m src.prd train --models naive --tickers AAPL --horizons 1 --run-name baseline-check
```

The baseline writes `reports/prd/baseline-check/metrics.csv` and `predictions.csv`. Every predicted price must equal that row's current adjusted price.

Run names are unique and are never overwritten. To repeat a check, use a new name such as `baseline-check-2`.

## 4. Model comparison

Classical families only:

```bash
python -m src.prd train --models linear random_forest xgboost svr --run-name prd-classical
```

Smaller deep-model check:

```bash
python -m src.prd train --models lstm gru bilstm transformer --tickers AAPL --horizons 1 --run-name deep-check
```

Full comparison (random walk and all eight model families, for 10 stocks and both horizons):

```bash
python -m src.prd train --run-name prd-full
```

Expanding-window deep evaluation takes substantial CPU time. If the process runs out of memory, use the [recovery runner](RECOVERY.md), which trains one model per fresh process.

All models use the same 60-session eligibility for scoring. Validation is split into model selection with early stopping, and interval calibration. Final models are not refitted on calibration or test data. Do not combine test scores from incompatible runs.

## 5. Analysis, reporting and dashboard

```bash
python -m src.prd eda       --run-dir reports/prd/prd-full
python -m src.prd portfolio --run-dir reports/prd/prd-full
python -m src.prd vader
python -m src.prd sentiment --run-dir reports/prd/prd-full
python -m src.prd report    --run-dir reports/prd/prd-full
streamlit run dashboard/prd_app.py
```

If you omit `--run-dir`, the latest completed run is used (recorded in `reports/prd/latest.json`). Portfolio analysis needs forecasts for every configured stock and defaults to horizon 1.

`python -m src.prd all --run-name prd-all` runs training, EDA, portfolio and recommendations, any available sentiment comparisons, and the draft report in one command. Run `prepare` first, and run `vader` first if you need the VADER comparison.

### Sentiment inputs

Sentiment reuses `data/processed/clean_news.parquet`, `daily_sentiment.parquet` and `news_coverage.parquet`. Only sessions with an audited `complete=True` coverage record may be filled as zero-news. Otherwise the experiment uses observed-news windows and states that restriction. If coverage is insufficient, the comparison is recorded as *pending* with a reason. A pending comparison is neither a positive nor a negative result. See [NEWS_COLLECTION.md](NEWS_COLLECTION.md) for extending coverage.

## 6. Run outputs

Each run folder under `reports/prd/<run-name>/` contains:

| Output | Contents |
| --- | --- |
| `manifest.json` | Configuration, code and data hashes, model coverage, test-exposure status |
| `metrics.csv` | Errors for each stock, horizon, model and split |
| `macro_average_metrics.csv` | The same metrics averaged across stocks |
| `cv_metrics.csv` | Expanding-window fold results and tuning candidates |
| `selected_models.csv` | Model chosen on validation for each stock and horizon |
| `models/` | Regressors, Keras models, scalers, settings and loss curves |
| `predictions.csv` | Origin and target dates, predicted and actual prices, intervals |
| `math_checks.json` | Gradient-descent and covariance cross-checks |
| `eda/` | Training-only statistics, figures and a review notebook |
| `sentiment/` | Paired comparisons and coverage status |
| `portfolio/` | Frontier, weights, risk, recommendations, execution history, hit rates |
| `RESULTS_DRAFT.md` | Results-backed draft report |

Keep failed baseline comparisons in any write-up. The historical test period has already been inspected during development, and current-vintage macro revisions remain a limitation.
