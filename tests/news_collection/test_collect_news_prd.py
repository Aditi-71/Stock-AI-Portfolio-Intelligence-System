import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import Mock, patch
from urllib.error import URLError

import pandas as pd

import collect_news_prd as app


class CollectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.audit = self.root / "reports/prd/news_collection"
        self.task = ("MSFT", date(2015, 7, 1), date(2015, 7, 31))
        self.state = {"records": {}, "request_times": []}

    def feed(self, day="20150701", n=1):
        return {
            "items": str(n),
            "feed": [
                {
                    "title": f"Article {i}",
                    "time_published": day + "T120000",
                    "summary": "Example",
                    "url": "https://example.com",
                    "source": "fixture",
                }
                for i in range(n)
            ],
        }

    def run_collect(self, tasks=None, fetch=None, maximum=3, budget=25):
        return app.collect(
            self.root,
            tasks or [self.task],
            self.state,
            self.audit,
            "private-api-key",
            max_requests=maximum,
            daily_budget=budget,
            delay=0,
            fetch=fetch or (lambda *_: self.feed()),
            clock=lambda: 100000.0,
        )

    def test_interval_subtraction_and_quarters(self):
        a, b = date(2015, 1, 1), date(2015, 12, 31)
        covered = [(a, date(2015, 3, 31)), (date(2015, 3, 1), date(2015, 6, 30))]
        self.assertEqual(list(app.subtract(a, b, covered)), [(date(2015, 7, 1), b)])
        q = list(app.quarters(a, b))
        self.assertEqual(len(q), 4)
        self.assertEqual(q[-1][1], b)

    def test_existing_files_preserved_and_plan_alternates_tickers(self):
        folder = self.root / "data/raw/news/MSFT"
        folder.mkdir(parents=True)
        old = folder / "MSFT_20150101_20150331.parquet"
        pd.DataFrame({"title": ["old"]}).to_parquet(old)
        before = old.read_bytes()
        config = {"start_date": "2015-01-01", "regression": {"end_date": "2015-09-30"}}
        tasks, inventory = app.make_plan(self.root, config, ["MSFT", "NVDA"], {}, self.audit)
        self.assertEqual(
            tasks[:2],
            [
                ("MSFT", date(2015, 4, 1), date(2015, 6, 30)),
                ("NVDA", date(2015, 1, 1), date(2015, 3, 31)),
            ],
        )
        self.assertEqual(before, old.read_bytes())
        self.assertEqual(inventory[0]["status"], "legacy_unverified")

    def test_error_stops_without_empty_marker_and_redacts_key(self):
        self.assertEqual(self.run_collect(fetch=lambda *_: {"Note": "quota private-api-key"}), 1)
        self.assertFalse(list(self.root.rglob("*.parquet")))
        saved = (self.audit / "collection_status.json").read_text()
        self.assertNotIn("private-api-key", saved)
        self.assertIn("[REDACTED]", saved)
        self.assertEqual(len(self.state["request_times"]), 1)
        self.assertEqual(
            app.pending(self.task, self.state["records"], self.audit, self.root), [self.task]
        )

    def test_empty_feed_is_not_certified_no_news(self):
        self.assertEqual(self.run_collect(fetch=lambda *_: {"items": "0", "feed": []}), 0)
        record = self.state["records"][app.job_key(*self.task)]
        self.assertEqual(record["status"], "empty_unverified")
        self.assertFalse(record["coverage_verified"])
        self.assertFalse(list(self.root.rglob("*.parquet")))
        self.assertEqual(app.pending(self.task, self.state["records"], self.audit, self.root), [])

    def test_capped_response_splits_resumes_and_checks_files(self):
        self.run_collect(fetch=lambda *_: self.feed(n=1000), maximum=1)
        tasks = app.pending(self.task, self.state["records"], self.audit, self.root)
        self.assertEqual(tasks, app.children(*self.task))
        self.assertFalse(list(self.root.rglob("*.parquet")))
        fetch = Mock(side_effect=lambda _, t: self.feed(t[1].strftime("%Y%m%d")))
        self.assertEqual(self.run_collect(tasks=tasks, fetch=fetch), 0)
        self.assertEqual(fetch.call_count, 2)
        self.assertEqual(app.pending(self.task, self.state["records"], self.audit, self.root), [])
        output = next(self.root.rglob("*.parquet"))
        output.write_bytes(b"corrupt")
        with self.assertRaisesRegex(ValueError, "changed or missing"):
            app.pending(self.task, self.state["records"], self.audit, self.root)

    def test_saved_budget_blocks_new_requests_across_runs(self):
        self.state["request_times"] = [99999.0]
        fetch = Mock()
        self.run_collect(fetch=fetch, maximum=1, budget=1)
        fetch.assert_not_called()
        self.assertIn("budget reached", self.state["last_stop"])

    def test_invalid_feed_dates_and_missing_feed_are_errors(self):
        for data in [{}, {"feed": "bad"}, self.feed("20160101"), {"items": "2", "feed": []}]:
            with self.subTest(data=str(data)[:40]):
                with self.assertRaises((ValueError, TypeError)):
                    app.validate_response(data, self.task)
        good = self.feed()
        good["feed"].append({"title": "Boundary", "time_published": "20150801T000000"})
        good["items"] = "2"
        frame, count = app.validate_response(good, self.task)
        self.assertEqual(count, 2)
        self.assertEqual(len(frame), 1)

    def test_transport_failure_does_not_expose_request_url(self):
        with patch.object(
            app, "urlopen", side_effect=URLError("https://example.com?apikey=private-api-key")
        ):
            with self.assertRaisesRegex(ValueError, "Network/JSON") as caught:
                app.request_news("private-api-key", self.task)
        self.assertNotIn("private-api-key", str(caught.exception))

    def test_single_day_cap_blocks_instead_of_looping(self):
        task = ("MSFT", date(2015, 7, 1), date(2015, 7, 1))
        self.assertEqual(self.run_collect(tasks=[task], fetch=lambda *_: self.feed(n=1000)), 1)
        with self.assertRaisesRegex(ValueError, "one day"):
            app.pending(task, self.state["records"], self.audit, self.root)


if __name__ == "__main__":
    unittest.main()
