import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import runner


class MarketSimRunnerTest(unittest.TestCase):
    def test_run_writes_contract_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            (work / "marketdata").mkdir()
            (work / "job.json").write_text(
                json.dumps(
                    {
                        "version": "marketarena.run.v1",
                        "task": "market_replay",
                        "run_id": "run_1",
                        "agent_id": "agent_1",
                        "data_manifest_key": "marketdata/manifest.json",
                        "universe": ["BTC-USD"],
                        "initial_cash_cents": 10000000,
                        "window": {
                            "start": "2026-05-01T00:00:00Z",
                            "end": "2026-05-01T03:00:00Z",
                        },
                    }
                ),
                encoding="utf-8",
            )
            (work / "marketdata" / "manifest.json").write_text(
                json.dumps(
                    {
                        "events": [
                            {"time": "2026-05-01T00:00:00Z", "symbol": "BTC-USD", "close_cents": 10000},
                            {"time": "2026-05-01T01:00:00Z", "symbol": "BTC-USD", "close_cents": 11000},
                            {"time": "2026-05-01T02:00:00Z", "symbol": "BTC-USD", "close_cents": 12000},
                            {"time": "2026-05-01T03:00:00Z", "symbol": "BTC-USD", "close_cents": 9000},
                        ],
                        "news": [
                            {"time": "2026-05-01T01:00:00Z", "symbol": "BTC-USD", "headline": "ETF inflows rise"}
                        ],
                    }
                ),
                encoding="utf-8",
            )

            runner.run(work)

            output = json.loads((work / "output" / "summary.json").read_text(encoding="utf-8"))
            metrics = json.loads((work / "metrics.json").read_text(encoding="utf-8"))
            receipt = json.loads((work / "receipt.json").read_text(encoding="utf-8"))

            self.assertEqual(metrics["output_name"], "output")
            self.assertEqual(output["run_id"], "run_1")
            self.assertGreater(output["orders_count"], 0)
            self.assertIn("output_hash", receipt)
            self.assertEqual(receipt["metadata"]["score"], output["score"])
            self.assertEqual(receipt["metadata"]["trace_object_key"], "output/agent_trace.jsonl")


if __name__ == "__main__":
    unittest.main()
