# Testing

The test suite uses hand-computable sequences, deterministic synthetic prices, temporary folders and mocked HTTP responses. It needs no market data, network access or API keys. These are engineering tests, not investment results.

## Running

```bash
python -m unittest discover -s tests/prd -v
python -m unittest discover -s tests/news_collection -p test_collect_news_prd.py -v
```

TensorFlow and XGBoost tests run only when those packages are installed. A skipped test means unverified, not passed.

## Latest results

Python 3.11, TensorFlow and XGBoost installed:

| Suite | Tests | Result |
| --- | ---: | --- |
| `tests/prd` | 26 | All passed, none skipped |
| `tests/news_collection` | 9 | All passed |

## Coverage

**Leakage and validation contract** (`test_contract.py`)
- Targets are shifted exactly once; windows and targets never cross split boundaries.
- Changing future prices does not change earlier training examples or fitted scalers.
- Engineered features do not use future prices or future macro releases. A macro value released at the weekend survives as-of alignment.
- Long price gaps are rejected.
- Direction is measured relative to the forecast origin.
- Manual gradient descent and covariance match library references.
- Drawdown includes the initial wealth.

**End-to-end** (`test_integration.py`)
- A small classical experiment (Linear/Ridge, Random Forest, SVR) writes aligned predictions, selects models on validation, and generates a results draft.
- Forecasts, optimisation, portfolio returns, recommendations and risk outputs share execution dates.
- Execution at the next open does not capture the overnight move before entry. Fixed holdings are not rebalanced daily.
- Minimum variance matches the analytic equal-variance solution.
- Unknown news is never filled as no-news, and missing sentiment stays visible in recommendations.
- All four deep models fit, save and reload; XGBoost fits and predicts.

**Recovery runner** (`test_recovery.py`)
- Classical predictions are restored without refitting; missing predictions are rebuilt from saved models.
- Changed configuration and shifted scoring dates are rejected.
- Checkpoint corruption is detected, and restarts reuse verified checkpoints, including after a simulated worker kill (`exit -9`).
- Selection is based on validation only, even when the test winner differs.
- Transformer saved-model recovery works.

**News collector** (`test_collect_news_prd.py`)
- Interval subtraction, legacy-file preservation and the alternating stock plan work as described.
- Errors are handled with API-key redaction, including network errors.
- Empty responses, 1,000-item cap splitting, and stopping on an unresolved one-day cap are handled.
- Restarts verify checksums, and the request budget persists between runs.
- Dates are validated.
