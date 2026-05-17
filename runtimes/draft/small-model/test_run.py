import importlib.util
import json
import unittest
from pathlib import Path


def load_runner():
    module_path = Path(__file__).with_name("run.py")
    spec = importlib.util.spec_from_file_location("draft_small_model", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DraftRunnerV8Tests(unittest.TestCase):
    def test_build_draft_packets_are_speculative_safe_and_signed(self):
        runner = load_runner()
        job = {
            "window_id": "win-test",
            "workgraph_id": "wg-test",
            "role_id": "draft-role",
            "node_id": "node-test",
            "parent_prefix_hash": "sha256:prefix",
            "model_hash": "sha256:model",
            "drafter_model_id": "tiny-drafter",
            "horizon": 4,
            "branch_count": 3,
            "prompt": "raw prompt must not appear in packet",
        }

        packets = runner.build_draft_packets(job, [[101, 102, 103], [101, 202], [303]])

        self.assertEqual(len(packets), 3)
        encoded = json.dumps(packets, sort_keys=True)
        self.assertNotIn("raw prompt", encoded)
        for packet in packets:
            self.assertEqual(packet["window_id"], "win-test")
            self.assertEqual(packet["workgraph_id"], "wg-test")
            self.assertEqual(packet["role_id"], "draft-role")
            self.assertEqual(packet["parent_prefix_hash"], "sha256:prefix")
            self.assertEqual(packet["model_hash"], "sha256:model")
            self.assertTrue(packet["candidate_tokens"])
            self.assertTrue(packet["signature"].startswith("sha256:"))
            self.assertNotIn("candidate_text_preview", packet)

    def test_main_writes_draft_packets_receipt_metrics_and_output(self):
        import tempfile

        runner = load_runner()
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp_path = Path(raw_tmp)
            runner.WORK_DIR = tmp_path
            (tmp_path / "job.json").write_text(
                json.dumps(
                    {
                        "window_id": "win-main",
                        "workgraph_id": "wg-main",
                        "role_id": "draft-main",
                        "parent_prefix_hash": "sha256:prefix",
                        "model_hash": "sha256:model",
                        "horizon": 3,
                        "branch_count": 2,
                        "prompt": "hello world",
                    }
                ),
                encoding="utf-8",
            )

            self.assertEqual(runner.main(), 0)

            packets = json.loads((tmp_path / "draft_packets.json").read_text(encoding="utf-8"))
            receipt = json.loads((tmp_path / "receipt.json").read_text(encoding="utf-8"))
            metrics = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
            output = json.loads((tmp_path / "output.json").read_text(encoding="utf-8"))

            self.assertTrue(packets)
            self.assertEqual(len(packets), 2)
            self.assertEqual(receipt["receipt_type"], "ryvion.draft_packet_batch.v1")
            self.assertTrue(receipt["output_hash"].startswith("sha256:"))
            self.assertEqual(metrics["output_name"], "output.json")
            self.assertEqual(output["schema_version"], "ryvion.speculative.draft_small_model.output.v1")


if __name__ == "__main__":
    unittest.main()
