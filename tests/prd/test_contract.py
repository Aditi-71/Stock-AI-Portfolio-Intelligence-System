import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.prd.config import load_config
from src.prd.data import blocks_for, regression_targets, walk_forward_blocks
from src.prd.maths import covariance, gradient_descent_linear, risk_metrics
from src.prd.metrics import reconstruct_prices, regression_metrics
from src.prd.models_ml import build_classical
from src.prd.prepare import align_macro, clean_prices, engineer_features


def fixture(n=320):
    rng = np.random.default_rng(17)
    dates = pd.bdate_range("2017-01-02", periods=n)
    log_returns = rng.normal(0.0002, 0.012, (n, 4))
    prices = pd.DataFrame(
        100 * np.exp(np.cumsum(log_returns, axis=0)),
        index=dates,
        columns=["AAPL", "MSFT", "NVDA", "^GSPC"],
    )
    features = pd.DataFrame(
        {
            "AAPL_ret_1d": log_returns[:, 0],
            "MSFT_ret_1d": log_returns[:, 1],
            "NVDA_ret_1d": log_returns[:, 2],
            "lag_aapl": np.r_[0.0, log_returns[:-1, 0]],
        },
        index=dates,
    )
    config = load_config(Path(__file__).resolve().parents[2] / "config.yaml")
    config.update(universe=["AAPL", "MSFT", "NVDA"], sequence_length=6)
    config["regression"].update(horizons=[1, 5], cv_splits=2, n_jobs=1)
    return features, prices, config


class ContractTests(unittest.TestCase):
    def test_target_is_shifted_exactly_once(self):
        dates = pd.bdate_range("2024-01-01", periods=8)
        p = pd.DataFrame({"AAPL": [10.0, 11.0, 13.0, 12.0, 14.0, 17.0, 16.0, 19.0]}, index=dates)
        t = regression_targets(p, "AAPL", 3)
        self.assertEqual(t.actual_price.iloc[1], 14.0)
        self.assertEqual(t.target_date.iloc[1], dates[4])
        self.assertAlmostEqual(t.target_return.iloc[1], np.log(14 / 11))
        self.assertTrue(t.target_date.iloc[-3:].isna().all())

    def test_windows_and_targets_do_not_cross_splits(self):
        x, p, c = fixture()
        for h in (1, 5):
            blocks = blocks_for(x, p, "AAPL", h, c)
            for b in blocks.values():
                self.assertGreaterEqual(b.origins.min() - b.length + 1, b.lo)
                self.assertLess(b.origins.max() + h, b.hi)
                np.testing.assert_allclose(b.sequences()[0, -1], b.X[0])
            t = regression_targets(p, "AAPL", h)
            for _, train, val in walk_forward_blocks(x, t, h, c):
                self.assertLess(train.rows.target_date.max(), x.index[val.lo])

    def test_future_mutation_does_not_change_training_examples_or_scalers(self):
        x, p, c = fixture()
        train = blocks_for(x, p, "AAPL", 5, c)["train"]
        x2, p2 = x.copy(), p.copy()
        x2.iloc[train.hi :] *= 1000
        p2.iloc[train.hi :] *= 4
        other = blocks_for(x2, p2, "AAPL", 5, c)["train"]
        np.testing.assert_array_equal(train.X, other.X)
        np.testing.assert_array_equal(train.y, other.y)
        first = build_classical("linear", {"alpha": 1.0}, c).fit(train.X, train.y)
        second = build_classical("linear", {"alpha": 1.0}, c).fit(other.X, other.y)
        np.testing.assert_allclose(first.regressor_[0].mean_, second.regressor_[0].mean_)
        np.testing.assert_allclose(first.transformer_.mean_, train.y.mean())

    def test_direction_is_relative_to_forecast_origin(self):
        result = regression_metrics([110, 90, 100], [100, 100, 100], [100, 100, 100])
        self.assertAlmostEqual(result["directional_accuracy_pct"], 100 / 3)
        self.assertEqual(result["unchanged_forecast_pct"], 100)
        self.assertAlmostEqual(result["mae"], 20 / 3)
        np.testing.assert_allclose(
            reconstruct_prices(np.array([100.0, 50.0]), np.log([1.1, 0.9])), [110, 45]
        )

    def test_macro_weekend_release_survives_asof_alignment(self):
        dates = pd.DatetimeIndex(["2024-01-05", "2024-01-08", "2024-01-09"])
        raw = pd.DataFrame({"CPI": [3.0, 4.0]}, index=pd.to_datetime(["2024-01-01", "2024-01-07"]))
        aligned = align_macro(raw, dates, {"CPI": 0})
        np.testing.assert_array_equal(aligned.CPI, [3.0, 4.0, 4.0])

    def test_engineered_features_do_not_use_future_prices_or_macro(self):
        _, prices, config = fixture()
        prices["^VIX"] = 20.0 + np.sin(np.arange(len(prices)) / 13.0)
        raw = pd.concat(
            {
                "Open": prices,
                "High": prices * 1.01,
                "Low": prices * 0.99,
                "Close": prices,
                "Adj Close": prices,
                "Volume": prices * 0 + 1000,
            },
            axis=1,
        )
        macro = pd.DataFrame(
            {"DGS10": 4.0, "DGS3MO": 3.0, "CPIAUCSL": 250.0, "UNRATE": 5.0}, index=prices.index
        )
        features = engineer_features(raw, macro, config["universe"], config["benchmark"])
        self.assertTrue(np.isfinite(features.iloc[200:]).all().all())
        changed_raw, changed_macro = raw.copy(), macro.copy()
        changed_raw.iloc[250:] *= 3
        changed_macro.iloc[250:] += 10
        other = engineer_features(
            changed_raw, changed_macro, config["universe"], config["benchmark"]
        )
        pd.testing.assert_frame_equal(features.iloc[:250], other.iloc[:250])

    def test_price_quality_refuses_long_gaps(self):
        dates = pd.bdate_range("2024-01-01", periods=9)
        fields = {
            ("Open", "A"): 10.0,
            ("High", "A"): 11.0,
            ("Low", "A"): 9.0,
            ("Close", "A"): 10.0,
            ("Adj Close", "A"): 10.0,
            ("Volume", "A"): 100.0,
        }
        raw = pd.DataFrame(fields, index=dates)
        raw.iloc[3, :] = np.nan
        cleaned, report = clean_prices(raw, dates, ["A"], 1)
        self.assertFalse(cleaned.isna().any().any())
        self.assertEqual(report["filled_rows"], 1)
        raw.iloc[4, :] = np.nan
        with self.assertRaisesRegex(ValueError, "gap exceeds"):
            clean_prices(raw, dates, ["A"], 1)

    def test_gradient_descent_and_covariance_match_references(self):
        rng = np.random.default_rng(10)
        x = rng.normal(size=(200, 3))
        y = 0.2 + x @ np.array([0.3, -0.2, 0.1])
        result = gradient_descent_linear(x, y)
        self.assertTrue(result["converged"])
        np.testing.assert_allclose(result["coef"], [0.3, -0.2, 0.1], atol=1e-8)
        self.assertAlmostEqual(result["intercept"], 0.2)
        np.testing.assert_allclose(covariance(x), np.cov(x, rowvar=False, ddof=1))

    def test_drawdown_includes_initial_wealth(self):
        r = risk_metrics([-0.1, 0.0, 0.0], annual_rf=0)
        self.assertAlmostEqual(r["max_drawdown"], -0.1)

    def test_all_classical_models_use_regression_targets(self):
        x, p, c = fixture()
        train = blocks_for(x, p, "AAPL", 1, c)["train"]
        for name, params in [
            ("linear", {"alpha": 0.0}),
            ("random_forest", {"n_estimators": 8, "max_depth": 3, "min_samples_leaf": 2}),
            ("svr", {"C": 0.1, "epsilon": 0.1, "kernel": "rbf", "gamma": "scale"}),
        ]:
            estimator = build_classical(name, params, c).fit(train.X, train.y)
            pred = estimator.predict(train.X[:7])
            self.assertEqual(pred.shape, (7,))
            self.assertTrue(np.isfinite(pred).all())


if __name__ == "__main__":
    unittest.main()
