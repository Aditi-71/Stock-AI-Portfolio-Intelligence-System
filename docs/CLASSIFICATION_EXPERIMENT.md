> Historical classification experiment supplied with the original source. Its reported numbers have not been recomputed by the new PRD regression workflow.

# Stock AI & Portfolio Intelligence System

An end-to-end quantitative finance and machine learning project that combines:

* Stock and macroeconomic data ingestion
* Technical feature engineering
* Classical machine learning
* Deep learning with LSTM and GRU
* Walk-forward validation
* Trading strategy backtesting
* FinBERT-based financial news sentiment analysis
* Portfolio optimization
* Risk analytics
* Interactive Streamlit dashboard

The project focuses on realistic time-series evaluation and leakage-aware experimentation rather than maximizing headline accuracy.

---

## Project Overview

The system analyzes a universe of 10 U.S. equities:

* AAPL
* MSFT
* NVDA
* JPM
* XOM
* JNJ
* PG
* KO
* CAT
* HD

Benchmark:

* S&P 500 (`^GSPC`)

Primary forecasting task:

> Predict whether AAPL's next-day adjusted-close return will be positive or negative.

Historical data covers approximately:

**2015-01-02 to 2026-08-27**

---

# Key Results

## Machine Learning

The following models were evaluated using chronological walk-forward validation:

| Model               | Mean Walk-Forward ROC-AUC |
| ------------------- | ------------------------: |
| Random Forest       |                **0.5266** |
| XGBoost             |                    0.5220 |
| Gradient Boosting   |                    0.5159 |
| GRU                 |                    0.5157 |
| Logistic Regression |                    0.5123 |
| LSTM                |                    0.4985 |

### Final forecasting model

**Random Forest**

Mean walk-forward results:

* Accuracy: **52.99%**
* F1: **0.6022**
* ROC-AUC: **0.5266**

The signal is modest and close to the always-up baseline, so the model should not be interpreted as a strong or reliable market predictor.

---

# Trading Backtest

The Random Forest probabilities were converted into a long/cash trading strategy.

Validation was used to select the probability threshold, after which the threshold was locked before final test evaluation.

### Random Forest Strategy

* Cumulative return: **43.27%**
* Annualized return: **22.87%**
* Annualized volatility: **26.83%**
* Sharpe ratio: **0.90**
* Maximum drawdown: **-33.10%**
* Market exposure: **76.82%**

### AAPL Buy and Hold

* Cumulative return: **37.84%**
* Annualized return: **20.18%**
* Annualized volatility: **29.81%**
* Sharpe ratio: **0.76**
* Maximum drawdown: **-33.36%**

A transaction cost of 10 basis points was applied when entering or exiting positions.

---

# Financial News Sentiment

Historical financial news was processed using **ProsusAI/FinBERT**.

The NLP pipeline includes:

1. Historical news ingestion
2. Duplicate removal
3. NYSE trading-session alignment
4. After-hours news mapping to the next trading session
5. FinBERT inference
6. Daily sentiment aggregation
7. Walk-forward A/B testing

Approximately **18,800 AAPL articles** were scored.

FinBERT produces positive, negative, and neutral probabilities.

A continuous sentiment score was calculated as:

```text
sentiment_score = positive_probability - negative_probability
```

Daily sentiment features included:

* Mean sentiment
* Sentiment standard deviation
* Positive ratio
* Negative ratio
* Neutral ratio
* Sentiment balance
* Article count
* Log article count
* Has-news indicator

---

## Does Sentiment Improve Prediction?

A Random Forest with sentiment features was compared against the same model without sentiment on identical walk-forward folds.

| Model                                | Mean ROC-AUC |
| ------------------------------------ | -----------: |
| Random Forest without sentiment      |   **0.5261** |
| Random Forest with FinBERT sentiment |       0.5181 |

Difference:

**-0.0080 ROC-AUC**

Adding FinBERT sentiment did not improve prediction.

The sentiment pipeline is therefore retained as an NLP research component but excluded from the final forecasting model.

This result demonstrates that adding additional data does not automatically improve predictive performance.

---

# Portfolio Optimization

Three portfolio strategies were evaluated:

1. Equal Weight
2. Minimum Variance
3. Maximum Sharpe

Optimization used historical data only through:

**2024-11-21**

The resulting portfolio weights were then locked and evaluated out-of-sample from:

**2024-11-22 to 2026-08-26**

This prevents the portfolio optimizer from using future information.

---

## Out-of-Sample Portfolio Results

| Strategy         | Cumulative Return | Annualized Volatility |   Sharpe |  Sortino | Max Drawdown |
| ---------------- | ----------------: | --------------------: | -------: | -------: | -----------: |
| Minimum Variance |        **42.26%** |                13.70% | **1.54** | **2.35** |   **-9.14%** |
| Equal Weight     |            41.18% |            **13.53%** |     1.53 |     2.34 |      -15.94% |
| Maximum Sharpe   |            35.34% |                19.31% |     0.99 |     1.49 |      -21.48% |
| S&P 500          |            29.03% |                16.65% |     0.96 |     1.41 |      -18.90% |

---

## Minimum-Variance Portfolio

The minimum-variance portfolio used a **Ledoit-Wolf shrinkage covariance matrix**.

Initial optimized weights:

| Stock | Weight |
| ----- | -----: |
| JNJ   | 31.44% |
| KO    | 26.89% |
| PG    | 22.93% |
| XOM   |  7.31% |
| HD    |  5.72% |
| AAPL  |  3.01% |
| CAT   |  2.30% |
| MSFT  |  0.41% |
| NVDA  |  0.00% |
| JPM   |  0.00% |

Estimated training variance was reduced by approximately **34%** relative to equal weight.

Out-of-sample, the strategy produced the smallest maximum drawdown and strongest risk-adjusted performance.

---

## Maximum-Sharpe Portfolio

The maximum-Sharpe optimizer used historical mean returns and a Ledoit-Wolf covariance estimate.

A maximum allocation of 30% per asset was imposed to limit concentration.

The optimizer assigned the maximum allowed allocation to NVDA because of its exceptionally high historical return.

However, this portfolio did not generalize well out-of-sample:

* Training estimated Sharpe: **1.45**
* Test Sharpe: **0.99**

This illustrates a common portfolio optimization problem:

> Expected-return estimates are highly unstable and can lead to overfit allocations.

---

# Risk Analysis

Portfolio risk metrics include:

* Annualized volatility
* Sharpe ratio
* Sortino ratio
* Maximum drawdown
* Historical Value at Risk
* Conditional Value at Risk
* Beta relative to the S&P 500
* Worst daily return
* Best daily return

### 95% Tail Risk

| Strategy         |   VaR 95% |  CVaR 95% | Beta |
| ---------------- | --------: | --------: | ---: |
| Minimum Variance |     1.23% | **1.69%** | 0.13 |
| Equal Weight     | **1.21%** |     1.81% | 0.72 |
| Maximum Sharpe   |     1.81% |     2.71% | 1.04 |
| S&P 500          |     1.57% |     2.37% | 1.00 |

---

# Feature Engineering

The model uses approximately 241 engineered features.

Feature groups include:

## Price and Return Features

* 1-day log return
* 5-day log return
* 10-day log return
* 21-day log return
* Lagged returns

## Trend Features

* SMA10
* SMA20
* SMA50
* SMA200
* EMA20
* MACD
* MACD signal

Moving averages are normalized relative to adjusted price.

## Momentum Features

* RSI14
* ROC10

## Volatility Features

* 20-day volatility
* 60-day volatility
* ATR14
* Bollinger Band width
* Bollinger Band position

## Volume Features

* Volume z-score
* Volume change

## Market Features

* S&P 500 return
* VIX level
* VIX change

## Macroeconomic Features

* 10-year Treasury yield
* 3-month Treasury yield
* Yield spread
* CPI
* Unemployment rate

CPI and unemployment data were conservatively lagged to reduce release-timing leakage.

## Calendar Features

* Day of week
* Month
* Turn-of-month indicator

---

# Models

The project evaluates:

### Classical Machine Learning

* Logistic Regression
* Random Forest
* Gradient Boosting
* XGBoost

### Deep Learning

* LSTM
* GRU

Sequence models use 60-day input windows.

---

# Walk-Forward Validation

Time-series data cannot be randomly shuffled without risking future information leakage.

Therefore, the project uses chronological validation with expanding training windows.

Example structure:

```text
Past data       Future validation
──────────────▶ ─────────────────▶

Fold 1
Train → Validate

Fold 2
Larger Train → Validate

Fold 3
Larger Train → Validate
```

This provides a more realistic estimate of how the model may behave on unseen future periods.

---

# Data Pipeline

```text
Market / Macro Data
        ↓
Cleaning
        ↓
Feature Engineering
        ↓
Target Construction
        ↓
Chronological Split
        ↓
Scaling
        ↓
ML / Deep Learning
        ↓
Walk-Forward Validation
        ↓
Model Selection
        ↓
Trading Backtest
```

News pipeline:

```text
Financial News
        ↓
Cleaning
        ↓
Trading Session Alignment
        ↓
FinBERT
        ↓
Daily Sentiment Features
        ↓
Walk-Forward A/B Test
```

Portfolio pipeline:

```text
Adjusted Prices
        ↓
Daily Returns
        ↓
Training Covariance / Returns
        ↓
Portfolio Optimization
        ↓
Locked Weights
        ↓
Out-of-Sample Evaluation
        ↓
Risk Analytics
```

---

# Interactive Dashboard

The project includes a Streamlit dashboard with four sections:

### Overview

* Project summary
* Model comparison
* Sentiment experiment
* Portfolio results

### ML Forecasting

* Walk-forward model comparison
* Final Random Forest metrics
* Prediction probability distribution
* Confusion matrix
* Sentiment A/B test

### News Sentiment

* FinBERT label distribution
* Sentiment score distribution
* Rolling sentiment
* News volume
* Sentiment experiment results

### Portfolio & Risk

* Cumulative growth comparison
* Portfolio performance table
* Sharpe and Sortino ratios
* Drawdowns
* Portfolio weights
* VaR and CVaR
* Market beta

---

# Project Structure

```text
stock-ai-system/
│
├── dashboard/
│   ├── Overview.py
│   │
│   └── pages/
│       ├── 1_ML_Forecasting.py
│       ├── 2_News_Sentiment.py
│       └── 3_Portfolio_Risk.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── src/
│   ├── ingest.py
│   ├── clean.py
│   ├── features.py
│   ├── target.py
│   ├── split.py
│   ├── prepare.py
│   ├── scale.py
│   │
│   ├── train_baseline.py
│   ├── train_random_forest.py
│   ├── train_gradient_boosting.py
│   ├── train_xgboost.py
│   ├── train_lstm.py
│   ├── train_gru.py
│   │
│   ├── walk_forward_classical.py
│   ├── walk_forward_lstm.py
│   ├── walk_forward_gru.py
│   │
│   ├── ingest_news.py
│   ├── clean_news.py
│   ├── score_sentiment.py
│   ├── aggregate_sentiment.py
│   ├── compare_sentiment_rf.py
│   │
│   ├── backtest_random_forest_test.py
│   ├── prepare_portfolio_returns.py
│   ├── backtest_equal_weight_portfolio.py
│   ├── backtest_min_variance_portfolio.py
│   ├── backtest_max_sharpe_portfolio.py
│   └── portfolio_risk_report.py
│
├── config.yaml
├── requirements.txt
├── .gitignore
└── README.md
```

---

# Installation

Clone the repository:

```bash
git clone <repository-url>
cd stock-ai-system-prd
```

Create and activate a Python environment.

Example using Conda:

```bash
conda create -n stock-ai python=3.11
conda activate stock-ai
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Alpha Vantage API Key

Historical financial news ingestion requires an Alpha Vantage API key.

Set it as an environment variable:

```bash
export ALPHAVANTAGE_API_KEY="your-api-key"
```

Do not hard-code API keys into sourc
