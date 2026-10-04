# Implementation status

This table separates what is implemented in code from what still needs empirical review or further work.

| Area | Implemented | Open items |
| --- | --- | --- |
| Data quality | NYSE alignment, limited gap filling, conflict and outlier logs, OHLC assertions, quality report | Review of flagged observations; eight-year per-asset coverage check |
| Features | Adjusted-price indicators, stochastic oscillator and OBV, market breadth, calendar and macro features | Verified point-in-time macro release vintages |
| Targets and baseline | 1- and 5-session adjusted log returns, price reconstruction, exact dates, random walk | — |
| Validation | Chronological blocks, purging, within-block windows, training-only scaling, calibration split | Genuinely unseen evaluation period |
| Classical models | Linear/Ridge, Random Forest, XGBoost, SVR with time-series tuning | — |
| Deep models | LSTM, GRU, BiLSTM, Transformer encoder with Huber loss, dropout and early stopping | — |
| Maths | Manual gradient descent, covariance and risk functions with cross-checks | — |
| EDA | ADF, tail and normality tests, seasonality/STL, correlations, figures, review notebook | Written interpretation |
| Sentiment | FinBERT reuse, VADER, paired regression A/B, momentum and volume features, missingness tracking | Complete news coverage audit, so that the paired comparisons can run |
| Portfolio | Forecast/historical blend, Ledoit–Wolf shrinkage, weight cap, frontier, Monte Carlo check, backtests, bootstrap intervals | — |
| Recommendations | Explainable scores, thresholds, rebalance band, next-open execution, costs, hit rates | Evaluation with real holdings (none assumed) |
| Dashboard | Seven panels, cached outputs and controls | — |
| Reproducibility | Rebuild and train commands, pinned dependencies, saved hashes and package snapshots | News ingestion remains a separate, credential-dependent workflow |
| Written report | Results-grounded Markdown draft generator | Reviewed final report |

## Methodological decisions

- **Horizons.** Five sessions is the multi-day horizon, alongside the next-day forecast.
- **Baseline direction.** The random walk predicts no change. Under strict sign matching, its directional accuracy is the observed unchanged-price fraction, not an assumed 50%.
- **Timing.** Closing features produce a forecast after close *t*, and trades execute at the next open. Forecast error and tradable open-to-open performance are measured separately.
- **BiLSTM.** The backward pass stays inside the historical input window.
- **MAPE.** Computed on positive reconstructed prices, not near-zero signed returns. Scores are reported per asset before macro averages.
- **Test exposure.** The historical test period was inspected during development. A new run name or model does not undo that exposure.
- **Macro data.** Observation lags do not remove later revisions to macro data.
- **News coverage.** News filenames do not prove that API collection was complete.
