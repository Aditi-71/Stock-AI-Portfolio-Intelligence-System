"""Recovery tests use deterministic synthetic data, never market results."""

import argparse
import copy
import importlib.util
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import yaml
from test_contract import fixture

import recover_prd as recovery
from src.prd.config import sha256
from src.prd.data import blocks_for
from src.prd.experiment import run_experiment
from src.prd.metrics import predictions_frame


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        x, p, c = fixture()
        c.update(project_root=str(self.root), universe=["AAPL"])
        c["regression"]["horizons"] = [1]
        self.config, self.inputs = c, (x, p, {})
        self.source = run_experiment(
            c, ["linear"], ["AAPL"], [1], "source", inputs=self.inputs, progress=lambda _: None
        )
        self.expected = recovery.identity(c, {})

    def test_restore_model_recreates_missing_predictions_without_fitting(self):
        original = pd.read_csv(self.source / "predictions.csv")
        model_path = self.source / "models/AAPL_h1/linear/model.joblib"
        digest = sha256(model_path)
        (self.source / "predictions.csv").rename(self.source / "original_predictions.csv")
        dest = self.root / "checkpoint"
        with patch(
            "sklearn.linear_model.LinearRegression.fit",
            side_effect=AssertionError("Must not refit"),
        ):
            recovery.checkpoint(
                self.source,
                dest,
                "AAPL",
                1,
                "linear",
                self.config,
                self.inputs,
                self.expected,
                restore=True,
            )
        restored = pd.read_csv(dest / "predictions.csv")
        wanted = original[original.model == "linear"].reset_index(drop=True)
        np.testing.assert_allclose(restored.predicted_price, wanted.predicted_price, rtol=1e-10)
        self.assertEqual(sha256(model_path), digest)
        self.assertFalse((self.source / "predictions.csv").exists())
        self.assertTrue(recovery.verify_checkpoint(dest, self.expected))

    def test_rejects_changed_config_and_shifted_scoring_dates(self):
        changed = copy.deepcopy(self.expected)
        changed["config"]["random_seed"] += 1
        with self.assertRaisesRegex(ValueError, "configuration differs"):
            recovery.verify_source(self.source, changed)
        predictions = pd.read_csv(self.source / "predictions.csv")
        predictions.loc[predictions.model == "linear", "origin_date"] = "2099-01-01"
        predictions.to_csv(self.source / "predictions.csv", index=False)
        with self.assertRaisesRegex(ValueError, "origins differ"):
            recovery.checkpoint(
                self.source,
                self.root / "bad",
                "AAPL",
                1,
                "linear",
                self.config,
                self.inputs,
                self.expected,
            )

    def test_checkpoint_tampering_is_not_silently_reused(self):
        dest = self.root / "checkpoint"
        recovery.checkpoint(
            self.source, dest, "AAPL", 1, "linear", self.config, self.inputs, self.expected
        )
        (dest / "model/model.joblib").write_bytes(b"corrupted")
        with self.assertRaisesRegex(ValueError, "damaged"):
            recovery.verify_checkpoint(dest, self.expected)

    def test_parent_restarts_using_verified_checkpoints(self):
        path = self.root / "config.yaml"
        path.write_text(yaml.safe_dump(recovery.normal_config(self.config)))
        args = argparse.Namespace(
            config=str(path), sources=[str(self.source)], run_name="recovered"
        )
        with (
            patch.object(recovery, "REQUIRED_MODELS", ("linear",)),
            patch.object(recovery, "load_prepared", return_value=self.inputs),
        ):
            recovery.run_recovery(args)
            output = self.root / "reports/prd/recovered"
            manifest = recovery.read_json(output / "manifest.json")
            self.assertEqual(manifest["status"], "completed")
            manifest["status"] = "interrupted"
            recovery.atomic_json(output / "manifest.json", manifest)
            with patch.object(
                recovery, "checkpoint", side_effect=AssertionError("Checkpoint should be reused")
            ):
                recovery.run_recovery(args)
            self.assertEqual(recovery.read_json(output / "manifest.json")["status"], "completed")
            self.assertTrue((output / "selected_models.csv").exists())
            self.assertEqual(len(pd.read_csv(output / "metrics.csv")), 6)

    def test_aggregation_selects_validation_winner_not_test_winner(self):
        output = self.root / "combined"
        output.mkdir()
        recovery.atomic_json(
            output / "_recovery_state.json", {"started_utc": "2026-01-01T00:00:00+00:00"}
        )
        blocks = blocks_for(self.inputs[0], self.inputs[1], "AAPL", 1, self.config)
        checkpoints = []
        for model in ("naive", "linear", "lstm"):
            dest = self.root / ("check-" + model)
            (dest / "model").mkdir(parents=True)
            meta = {"ticker": "AAPL", "horizon": 1, "model": model, "params": {}}
            recovery.atomic_json(dest / "model/metadata.json", meta)
            frames = []
            for split in recovery.SPLITS:
                block = blocks[split]
                if model == "naive":
                    prediction = np.zeros(len(block.y))
                elif (model == "lstm" and split == "validation") or (
                    model == "linear" and split == "test"
                ):
                    prediction = block.y.copy()
                else:
                    prediction = block.y + 0.02
                frames.append(
                    predictions_frame(
                        block, prediction, "AAPL", 1, model, split, 0.1 if split == "test" else None
                    )
                )
            recovery.atomic_csv(pd.concat(frames, ignore_index=True), dest / "predictions.csv")
            recovery.atomic_csv(
                pd.DataFrame(
                    [
                        {
                            "ticker": "AAPL",
                            "horizon": 1,
                            "model": model,
                            "candidate": 0,
                            "fold": 1,
                            "mae": 1,
                        }
                    ]
                ),
                dest / "cv_metrics.csv",
            )
            recovery.atomic_json(
                dest / "origin.json",
                {"source": "synthetic fixture", "ticker": "AAPL", "horizon": 1, "model": model},
            )
            files = {
                p.relative_to(dest).as_posix(): sha256(p) for p in dest.rglob("*") if p.is_file()
            }
            recovery.atomic_json(
                dest / "complete.json", {"identity": self.expected, "files": files}
            )
            checkpoints.append(dest)
        with patch.object(recovery, "REQUIRED_MODELS", ("linear", "lstm")):
            recovery.aggregate(
                checkpoints,
                output,
                self.config,
                self.inputs,
                self.expected,
                recovery.read_json(self.source / "manifest.json"),
            )
        selected = pd.read_csv(output / "selected_models.csv")
        self.assertEqual(selected.selected_model.iloc[0], "lstm")
        metrics = pd.read_csv(output / "metrics.csv")
        test = metrics[metrics.split == "test"].set_index("model")
        self.assertLess(test.loc["linear", "mae"], test.loc["lstm", "mae"])

    def test_failed_worker_preserves_checkpoints_and_next_attempt_resumes(self):
        path = self.root / "config.yaml"
        path.write_text(yaml.safe_dump(recovery.normal_config(self.config)))
        args = argparse.Namespace(
            config=str(path), sources=[str(self.source)], run_name="recovered"
        )
        output = self.root / "reports/prd/recovered"

        def train_child(command, **kwargs):
            name = command[command.index("--child-name") + 1]
            model = command[command.index("--model") + 1]
            run_experiment(
                self.config,
                [model],
                ["AAPL"],
                [1],
                name,
                inputs=self.inputs,
                progress=lambda _: None,
            )
            return subprocess.CompletedProcess(command, 0)

        with (
            patch.object(recovery, "REQUIRED_MODELS", ("linear", "svr")),
            patch.object(recovery, "load_prepared", return_value=self.inputs),
        ):
            with patch.object(
                recovery.subprocess, "run", return_value=subprocess.CompletedProcess([], -9)
            ):
                with self.assertRaisesRegex(RuntimeError, "exit -9"):
                    recovery.run_recovery(args)
            self.assertEqual(recovery.read_json(output / "manifest.json")["status"], "interrupted")
            self.assertEqual(
                len(list((output / "_checkpoints").glob("*/attempt-*/complete.json"))), 2
            )
            with patch.object(recovery.subprocess, "run", side_effect=train_child) as worker:
                recovery.run_recovery(args)
            self.assertEqual(worker.call_count, 1)
            self.assertEqual(recovery.read_json(output / "manifest.json")["status"], "completed")
            self.assertEqual(len(pd.read_csv(output / "metrics.csv")), 9)

    @unittest.skipUnless(
        importlib.util.find_spec("tensorflow"), "TensorFlow not installed in this runtime"
    )
    def test_transformer_saved_model_recovery(self):
        self.config["regression"].update(epochs=1, patience=1, batch_size=32)
        source = run_experiment(
            self.config,
            ["transformer"],
            ["AAPL"],
            [1],
            "deep-source",
            inputs=self.inputs,
            progress=lambda _: None,
        )
        expected = recovery.identity(self.config, {})
        (source / "predictions.csv").rename(source / "original_predictions.csv")
        with patch("src.prd.models_dl.fit_deep", side_effect=AssertionError("Must not refit")):
            recovery.checkpoint(
                source,
                self.root / "deep-check",
                "AAPL",
                1,
                "transformer",
                self.config,
                self.inputs,
                expected,
                restore=True,
            )
        self.assertTrue(recovery.verify_checkpoint(self.root / "deep-check", expected))


if __name__ == "__main__":
    unittest.main()
