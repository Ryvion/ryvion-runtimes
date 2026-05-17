import importlib.util
import json
import os
import socket
import threading
import unittest
from pathlib import Path


def load_runner():
    module_path = Path(__file__).with_name("run.py")
    spec = importlib.util.spec_from_file_location("sglang_verifier_runner_v8", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rpc(socket_path: Path, method: str, params: dict, request_id: str = "1") -> dict:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.connect(str(socket_path))
        client.sendall(json.dumps({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}).encode("utf-8") + b"\n")
        buf = b""
        while not buf.endswith(b"\n"):
            chunk = client.recv(65536)
            if not chunk:
                break
            buf += chunk
        return json.loads(buf.decode("utf-8"))


class SGLangVerifierRunnerV8Tests(unittest.TestCase):
    def test_model_artifact_policy_requires_local_safetensors_and_rejects_bins(self):
        import tempfile

        runner = load_runner()
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            model_dir = tmp / "model"
            model_dir.mkdir()
            (model_dir / "model.safetensors").write_bytes(b"safe")
            policy = runner.validate_model_artifact_policy(model_dir)
            self.assertEqual(policy["format"], "safetensors")
            self.assertFalse(policy["weight_loader_disable_mmap"])

            bad_dir = tmp / "bad"
            bad_dir.mkdir()
            (bad_dir / "pytorch_model.bin").write_bytes(b"unsafe")
            with self.assertRaises(ValueError):
                runner.validate_model_artifact_policy(bad_dir)

            with self.assertRaises(ValueError):
                runner.validate_model_artifact_policy(Path("Qwen/Qwen3-0.6B"))

    def test_unix_socket_contract_verifies_tree_with_fake_backend_and_writes_probe_summary(self):
        import tempfile

        runner = load_runner()
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            socket_path = tmp / "verifier.sock"
            backend = runner.FakeSGLangBackend({
                (10,): {"top_token_id": 10, "logprob": -0.01, "runner_up_logprob": -2.0},
                (10, 11): {"top_token_id": 11, "logprob": -0.02, "runner_up_logprob": -1.7},
                (10, 11, 99): {"top_token_id": 12, "logprob": -3.0, "runner_up_logprob": -0.05},
            })
            server = runner.SGLangVerifierSessionServer(socket_path=socket_path, work_dir=tmp, backend=backend)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            server.wait_ready(timeout_s=2)

            session = {"session_id": "sess-gpu", "workgraph_id": "wg-v8", "model_hash": "sha256:model", "prefix_tokens": []}
            tree = {
                "tree_cid": "sha256:tree",
                "window_id": "win-v8",
                "branches": [
                    {"branch_id": "br-a", "candidate_tokens": [10, 11, 99]},
                    {"branch_id": "br-b", "candidate_tokens": [10, 13]},
                ],
            }

            self.assertEqual(rpc(socket_path, "start_session", {"session": session})["result"]["status"], "started")
            self.assertEqual(rpc(socket_path, "prefill", {"session_id": "sess-gpu", "prefix_tokens": []})["result"]["status"], "prefilled")
            verified = rpc(socket_path, "verify_tree", {"session": session, "tree": tree})["result"]
            self.assertEqual(verified["accepted_token_receipt"]["status"], "verified_exact_greedy")
            self.assertEqual(verified["accepted_token_receipt"]["accepted_len"], 2)
            self.assertEqual(verified["accepted_token_receipt"]["accepted_token_ids"], [10, 11])
            self.assertIn("logprob_margin_bps", verified["probe_summary"]["feature_scores_bps"])
            self.assertEqual(rpc(socket_path, "commit", {"session_id": "sess-gpu", "accepted_token_ids": [10, 11], "accepted_len": 2})["result"]["status"], "committed")
            self.assertEqual(rpc(socket_path, "rollback", {"session_id": "sess-gpu", "branch_ids": ["br-b"]})["result"]["status"], "rolled_back")
            self.assertEqual(rpc(socket_path, "close_session", {"session_id": "sess-gpu"})["result"]["status"], "closed")

            server.shutdown()
            thread.join(timeout=2)

            receipt = json.loads((tmp / "verifier_session_receipt.json").read_text(encoding="utf-8"))
            probe = json.loads((tmp / "probe_summary.json").read_text(encoding="utf-8"))
            metrics = json.loads((tmp / "metrics.json").read_text(encoding="utf-8"))
            output = json.loads((tmp / "output.json").read_text(encoding="utf-8"))

            self.assertEqual(receipt["receipt_type"], "ryvion.accepted_token.v1")
            self.assertEqual(receipt["decoding_mode"], "greedy_exact")
            self.assertEqual(receipt["accepted_len"], 2)
            self.assertNotIn("raw_prompt", json.dumps(probe))
            self.assertEqual(metrics["output_name"], "output.json")
            self.assertEqual(metrics["engine"], "sglang")
            self.assertEqual(output["schema_version"], "ryvion.sglang_verifier_runner_v8.output.v1")

    def test_abort_after_verify_flushes_partial_evidence_without_completed_receipt(self):
        import tempfile

        runner = load_runner()
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            socket_path = tmp / "verifier.sock"
            backend = runner.FakeSGLangBackend({
                (42,): {"top_token_id": 42, "logprob": -0.01, "runner_up_logprob": -1.2},
            })
            server = runner.SGLangVerifierSessionServer(socket_path=socket_path, work_dir=tmp, backend=backend)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            server.wait_ready(timeout_s=2)

            session = {"session_id": "sess-abort", "workgraph_id": "wg-abort", "model_hash": "sha256:model", "prefix_tokens": []}
            tree = {"tree_cid": "sha256:tree-abort", "branches": [{"branch_id": "br-a", "candidate_tokens": [42]}]}

            self.assertEqual(rpc(socket_path, "start_session", {"session": session})["result"]["status"], "started")
            verified = rpc(socket_path, "verify_tree", {"session": session, "tree": tree})["result"]
            self.assertEqual(verified["accepted_token_receipt"]["accepted_len"], 1)
            self.assertFalse((tmp / "receipt.json").exists())

            aborted = rpc(socket_path, "abort", {"session_id": "sess-abort"})["result"]
            self.assertEqual(aborted["status"], "aborted")
            server.shutdown()
            thread.join(timeout=2)

            partial = json.loads((tmp / "receipt.partial.json").read_text(encoding="utf-8"))
            verifier_partial = json.loads((tmp / "verifier_session_receipt.partial.json").read_text(encoding="utf-8"))
            probe_partial = json.loads((tmp / "probe_summary.partial.json").read_text(encoding="utf-8"))

            self.assertEqual(partial["status"], "aborted")
            self.assertEqual(partial["execution_status"], "aborted")
            self.assertEqual(partial["accepted_value"], 1)
            self.assertTrue(partial["committed_before_abort"] is False)
            self.assertEqual(verifier_partial["status"], "aborted_before_commit")
            self.assertEqual(verifier_partial["accepted_len"], 1)
            self.assertEqual(probe_partial["execution_status"], "aborted")

    def test_extract_logprob_records_parses_sglang_style_meta_info(self):
        runner = load_runner()
        meta = {
            "input_token_logprobs": [(None, 100), (-0.1, 10), (-1.5, 11)],
            "input_top_logprobs": [[], [{"10": -0.1, "12": -2.0}], [{"12": -0.2, "11": -1.5}]],
        }
        records = runner.extract_logprob_records(meta, continuation_offset=1, candidate_tokens=[10, 11])
        self.assertEqual(records[0]["top_token_id"], 10)
        self.assertEqual(records[1]["top_token_id"], 12)
        self.assertLess(records[1]["margin"], 0)


if __name__ == "__main__":
    unittest.main()
