import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from test_contract import fixture

from src.prd.data import blocks_for
from src.prd.experiment import run_experiment
from src.prd.models_ml import candidates
from src.prd.portfolio import (
    efficient_frontier,
    fixed_holdings_returns,
    optimise,
    random_portfolios,
    run_portfolio,
)
from src.prd.recommend import execute_recommendations, recommendation_scores
from src.prd.report import write_report
from src.prd.sentiment import filter_paired_blocks, sentiment_features


class IntegrationTests(unittest.TestCase):
    def test_portfolio_and_recommendation_outputs_share_execution_dates(self):
        x, p, c = fixture()
        c["regression"].update(weight_cap=0.5, mc_portfolios=200, bootstrap_samples=10)
        clean = pd.concat(
            {
                "Open": p * 0.999,
                "High": p * 1.02,
                "Low": p * 0.98,
                "Close": p,
                "Adj Close": p,
                "Volume": p * 0 + 1000,
            },
            axis=1,
        )
        with tempfile.TemporaryDirectory() as directory:
            c["project_root"] = directory
            output = run_experiment(
                c,
                ["naive"],
                horizons=[1],
                run_name="portfolio-test",
                inputs=(x, p, {"test_status": "synthetic_test_fixture"}),
                progress=lambda _: None,
            )
            with (
                patch("src.prd.portfolio.load_prepared", return_value=(x, p, {})),
                patch("pandas.read_parquet", return_value=clean),
            ):
                path = run_portfolio(c, output)
            r = pd.read_csv(path / "returns.csv")
            self.assertEqual(
                set(r.columns) - {"Date"},
                {
                    "Equal weight",
                    "Minimum variance",
                    "Maximum Sharpe",
                    "Benchmark",
                    "Recommendations",
                },
            )
            self.assertTrue(np.isfinite(r.drop(columns="Date")).all().all())
            meta = json.loads((path / "metadata.json").read_text())
            self.assertTrue(meta["mc_minimum_variance_check_passed"])
            self.assertGreater(
                pd.Timestamp(meta["first_return_date"]), pd.Timestamp(meta["entry_open"])
            )
            self.assertTrue((path / "recommendation_metadata.json").exists())

    def test_classical_experiment_writes_aligned_results_and_report(self):
        x, p, config = fixture()

        def small_grid(name):
            if name == "random_forest":
                return [{"n_estimators": 8, "max_depth": 3, "min_samples_leaf": 2}]
            return candidates(name)

        with tempfile.TemporaryDirectory() as directory:
            config["project_root"] = directory
            with patch("src.prd.models_ml.candidates", side_effect=small_grid):
                output = run_experiment(
                    config,
                    ["naive", "linear", "random_forest", "svr"],
                    ["AAPL"],
                    [1, 5],
                    "integration",
                    inputs=(x, p, {"test_status": "synthetic_test_fixture"}),
                    progress=lambda _: None,
                )
            frame = pd.read_csv(output / "predictions.csv")
            self.assertEqual(set(frame.model), {"naive", "linear", "random_forest", "svr"})
            for _, group in frame.groupby(["horizon", "split"]):
                count = group.groupby("origin_date").model.nunique()
                self.assertTrue(count.eq(4).all())
            naive = frame[frame.model == "naive"]
            np.testing.assert_allclose(naive.current_price, naive.predicted_price)
            meta = json.loads((output / "manifest.json").read_text())
            self.assertEqual(meta["status"], "completed")
            self.assertFalse(meta["full_model_coverage"])
            report = write_report(config, output)
            self.assertIn("synthetic_test_fixture", report.read_text())
            with self.assertRaises(FileExistsError):
                run_experiment(
                    config,
                    ["naive"],
                    ["AAPL"],
                    [1],
                    "integration",
                    inputs=(x, p, {}),
                    progress=lambda _: None,
                )

    def test_minimum_variance_matches_analytic_equal_variance_case(self):
        mu = np.array([0.08, 0.10, 0.12, 0.14])
        covariance = np.eye(4) * 0.04
        w = optimise(mu, covariance, cap=0.4, objective="variance")
        np.testing.assert_allclose(w, np.repeat(0.25, 4), atol=1e-6)
        frontier = efficient_frontier(mu, covariance, cap=0.4, points=8)
        self.assertTrue(np.isfinite(frontier.to_numpy()).all())
        cloud = random_portfolios(mu, covariance, cap=0.4, count=200, seed=3)
        self.assertGreaterEqual(cloud.volatility.min(), 0.1 - 1e-8)
        with self.assertRaises(ValueError):
            optimise(mu, covariance, cap=0.2)

    def test_fixed_holdings_are_not_daily_rebalanced(self):
        returns = np.array([[1.0, 0.0], [-0.5, 0.0]])
        output = fixed_holdings_returns(returns, [0.5, 0.5])
        self.assertAlmostEqual(output[0], 0.5)
        self.assertAlmostEqual(output[1], -1 / 3)
        self.assertAlmostEqual(np.prod(1 + output), 1.0)

    def test_execution_does_not_capture_the_overnight_move_before_entry(self):
        dates = pd.bdate_range("2024-01-01", periods=4)
        opens = pd.DataFrame({"A": [100.0, 200.0, 220.0, 242.0]}, index=dates)
        signals = pd.DataFrame({"A": ["BUY", "HOLD"]}, index=dates[:2])
        strategy, hits = execute_recommendations(
            opens, signals, [1.0], dates[0], cost_bps=0, cap=1.0, band=0
        )
        np.testing.assert_allclose(strategy["return"], [0.1, 0.1])
        self.assertEqual(strategy.index[0], dates[2])
        self.assertTrue(hits.hit.all())
        charged, _ = execute_recommendations(
            opens, signals, [1.0], dates[0], cost_bps=10, cap=1.0, band=0
        )
        self.assertAlmostEqual(charged["return"].iloc[0], 1.1 / 1.001 - 1)

    def test_unknown_news_is_not_filled_as_no_news(self):
        x, p, config = fixture()
        dates = x.index
        daily = pd.DataFrame(
            {"session_date": dates[5:], "ticker": "AAPL", "sentiment_mean": 0.2, "article_count": 3}
        )
        legacy = pd.DataFrame({"session_date": dates, "ticker": "AAPL", "downloaded": True})
        s = sentiment_features(daily, legacy, dates, "AAPL")
        self.assertTrue(s.news_sentiment.iloc[:5].isna().all())
        coverage = legacy.assign(complete=True)
        complete = sentiment_features(daily, coverage, dates, "AAPL")
        self.assertTrue(complete.news_sentiment.iloc[:5].eq(0).all())
        base = blocks_for(x, p, "AAPL", 5, config)
        augmented = blocks_for(x.join(s), p, "AAPL", 5, config)
        pairs = filter_paired_blocks(base, augmented)
        for name in pairs[0]:
            np.testing.assert_array_equal(pairs[0][name].origins, pairs[1][name].origins)
            self.assertTrue(np.isfinite(pairs[1][name].sequences()).all())

    def test_missing_sentiment_is_visible_in_recommendations(self):
        result = recommendation_scores(
            [0.01], [0.02], 0.01, [np.nan], 1, [0.5, 0.2, 0.3], [-0.1, 0.1]
        )
        self.assertTrue(result.sentiment_missing.iloc[0])
        self.assertTrue(np.isnan(result.sentiment_subscore.iloc[0]))


class OptionalRuntimeTests(unittest.TestCase):
    @unittest.skipUnless(
        importlib.util.find_spec("xgboost"), "XGBoost is not installed in this runtime"
    )
    def test_xgboost_fit_predict(self):
        from src.prd.models_ml import build_classical

        x, p, c = fixture()
        block = blocks_for(x, p, "AAPL", 1, c)["train"]
        model = build_classical(
            "xgboost",
            {"n_estimators": 5, "max_depth": 2, "learning_rate": 0.1, "subsample": 1.0},
            c,
        ).fit(block.X, block.y)
        self.assertTrue(np.isfinite(model.predict(block.X[:5])).all())

    @unittest.skipUnless(
        importlib.util.find_spec("tensorflow"), "TensorFlow is not installed in this runtime"
    )
    def test_all_four_deep_models_fit_and_reload(self):
        from src.prd.models_dl import backend, fit_deep

        x, p, c = fixture()
        c["regression"].update(epochs=1, patience=1, batch_size=32)
        blocks = blocks_for(x, p, "AAPL", 1, c)
        for name in ("lstm", "gru", "bilstm", "transformer"):
            model = fit_deep(name, blocks["train"], blocks["validation"], c)
            self.assertTrue(np.isfinite(model.predict_block(blocks["test"])).all())
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory)
                model.save(path)
                tf, _ = backend()
                loaded = tf.keras.models.load_model(path / "model.keras")
                values = blocks["test"].sequences(model.x_scaler)
                np.testing.assert_allclose(
                    model.model.predict(values, verbose=0),
                    loaded.predict(values, verbose=0),
                    atol=1e-6,
                )


if __name__ == "__main__":
    unittest.main()
