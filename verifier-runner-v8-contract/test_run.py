import importlib.util
import json
import socket
import threading
import unittest
from pathlib import Path


def load_runner():
    module_path = Path(__file__).with_name("run.py")
    spec = importlib.util.spec_from_file_location("verifier_runner_v8_contract", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rpc(socket_path: Path, method: str, params: dict, request_id: str = "1") -> dict:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.connect(str(socket_path))
        payload = {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}
        client.sendall(json.dumps(payload).encode("utf-8") + b"\n")
        buf = b""
        while not buf.endswith(b"\n"):
            buf += client.recv(65536)
        return json.loads(buf.decode("utf-8"))


class VerifierRunnerV8ContractTests(unittest.TestCase):
    def test_unix_socket_contract_accepts_verify_commit_rollback_and_writes_receipts(self):
        import tempfile

        runner = load_runner()
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp_path = Path(raw_tmp)
            socket_path = tmp_path / "verifier.sock"
            server = runner.VerifierSessionServer(socket_path=socket_path, work_dir=tmp_path)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            server.wait_ready(timeout_s=2)

            session = {
                "session_id": "sess-test",
                "workgraph_id": "wg-test",
                "model_hash": "sha256:model",
                "prefix_hash": "sha256:prefix",
            }
            tree = {
                "tree_cid": "sha256:tree",
                "window_id": "win-test",
                "branches": [
                    {"branch_id": "br-a", "candidate_tokens": [10, 11, 12]},
                    {"branch_id": "br-b", "candidate_tokens": [10, 22]},
                ],
            }

            self.assertEqual(rpc(socket_path, "start_session", {"session": session})["result"]["status"], "started")
            self.assertEqual(rpc(socket_path, "prefill", {"session_id": "sess-test", "prefix_hash": "sha256:prefix"})["result"]["status"], "prefilled")
            verified = rpc(socket_path, "verify_tree", {"session": session, "tree": tree})["result"]
            self.assertEqual(verified["accepted_token_receipt"]["accepted_len"], 3)
            self.assertEqual(verified["accepted_token_receipt"]["tree_cid"], "sha256:tree")
            self.assertIn("probe_summary", verified)
            self.assertEqual(rpc(socket_path, "commit", {"session_id": "sess-test", "accepted_len": 3})["result"]["status"], "committed")
            self.assertEqual(rpc(socket_path, "rollback", {"session_id": "sess-test", "branch_ids": ["br-b"]})["result"]["status"], "rolled_back")
            self.assertEqual(rpc(socket_path, "close_session", {"session_id": "sess-test"})["result"]["status"], "closed")

            server.shutdown()
            thread.join(timeout=2)

            receipt = json.loads((tmp_path / "verifier_session_receipt.json").read_text(encoding="utf-8"))
            probe = json.loads((tmp_path / "probe_summary.json").read_text(encoding="utf-8"))
            metrics = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
            output = json.loads((tmp_path / "output.json").read_text(encoding="utf-8"))

            self.assertEqual(receipt["receipt_type"], "ryvion.accepted_token.v1")
            self.assertEqual(receipt["method"], "verify_tree")
            self.assertEqual(receipt["accepted_len"], 3)
            self.assertGreaterEqual(probe["confidence_bps"], 9000)
            self.assertEqual(metrics["output_name"], "output.json")
            self.assertEqual(output["schema_version"], "ryvion.verifier_runner_v8_contract.output.v1")


if __name__ == "__main__":
    unittest.main()
