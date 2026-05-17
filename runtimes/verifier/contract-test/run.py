"""CPU mock for Ryvion verifier runner session contract.

This runner intentionally does not implement GPU verification. It exposes a
newline-delimited JSON-RPC server over a Unix socket so node-agent can exercise
the long-lived session lifecycle before SGLang/vLLM data-plane integration.
"""

from __future__ import annotations

import hashlib
import json
import os
import signal
import socket
import sys
import threading
import time
from pathlib import Path

WORK_DIR = Path(os.environ.get("RYV_WORK_DIR", "/work"))
SOCKET_PATH = Path(os.environ.get("RYV_VERIFIER_SESSION_SOCKET", "/work/verifier_session.sock"))


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_json_atomic(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    tmp.replace(path)


class VerifierSessionServer:
    def __init__(self, socket_path: Path = SOCKET_PATH, work_dir: Path = WORK_DIR):
        self.socket_path = Path(socket_path)
        self.work_dir = Path(work_dir)
        self._listener: socket.socket | None = None
        self._shutdown = threading.Event()
        self._ready = threading.Event()
        self.sessions: dict[str, dict] = {}
        self.last_receipt: dict | None = None
        self.last_probe: dict | None = None

    def wait_ready(self, timeout_s: float = 10.0) -> None:
        if not self._ready.wait(timeout_s):
            raise TimeoutError(f"verifier socket not ready: {self.socket_path}")

    def serve_forever(self) -> None:
        self.work_dir.mkdir(parents=True, exist_ok=True)
        try:
            self.socket_path.unlink()
        except FileNotFoundError:
            pass
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        listener.bind(str(self.socket_path))
        listener.listen(16)
        listener.settimeout(0.2)
        self._listener = listener
        self._ready.set()
        while not self._shutdown.is_set():
            try:
                conn, _ = listener.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            threading.Thread(target=self._serve_conn, args=(conn,), daemon=True).start()
        try:
            listener.close()
        finally:
            try:
                self.socket_path.unlink()
            except FileNotFoundError:
                pass

    def shutdown(self) -> None:
        self._shutdown.set()
        if self._listener is not None:
            try:
                self._listener.close()
            except OSError:
                pass

    def _serve_conn(self, conn: socket.socket) -> None:
        with conn:
            reader = conn.makefile("r", encoding="utf-8")
            writer = conn.makefile("w", encoding="utf-8")
            for line in reader:
                if not line.strip():
                    continue
                try:
                    req = json.loads(line)
                    result = self.handle(req.get("method", ""), req.get("params") or {})
                    response = {"jsonrpc": "2.0", "id": req.get("id"), "result": result}
                except Exception as exc:
                    response = {"jsonrpc": "2.0", "id": None, "error": {"code": "mock_verifier_error", "message": str(exc)}}
                writer.write(json.dumps(response, sort_keys=True, separators=(",", ":")) + "\n")
                writer.flush()

    def handle(self, method: str, params: dict) -> dict:
        method = str(method or "").strip()
        if method == "start_session":
            session = params.get("session") or {}
            session_id = str(session.get("session_id") or params.get("session_id") or "sess-mock")
            self.sessions[session_id] = {"session": session, "kv_epoch": 0, "committed": 0}
            return {"status": "started", "session_id": session_id, "kv_epoch": 0}
        if method == "prefill":
            session_id = str(params.get("session_id") or "sess-mock")
            state = self.sessions.setdefault(session_id, {"session": {}, "kv_epoch": 0, "committed": 0})
            state["prefix_hash"] = str(params.get("prefix_hash") or "")
            return {"status": "prefilled", "session_id": session_id, "kv_epoch": state["kv_epoch"]}
        if method == "verify_tree":
            return self._verify_tree(params)
        if method == "commit":
            session_id = str(params.get("session_id") or "sess-mock")
            state = self.sessions.setdefault(session_id, {"session": {}, "kv_epoch": 0, "committed": 0})
            accepted_len = int(params.get("accepted_len") or 0)
            state["committed"] = int(state.get("committed") or 0) + accepted_len
            state["kv_epoch"] = int(state.get("kv_epoch") or 0) + 1
            return {"status": "committed", "session_id": session_id, "accepted_len": accepted_len, "kv_epoch": state["kv_epoch"]}
        if method == "rollback":
            branch_ids = params.get("branch_ids") or params.get("rollback_branch_ids") or []
            return {"status": "rolled_back", "rollback_branch_ids": branch_ids}
        if method in ("close", "close_session"):
            session_id = str(params.get("session_id") or "sess-mock")
            self._write_final_files()
            threading.Timer(0.05, self.shutdown).start()
            return {"status": "closed", "session_id": session_id}
        if method == "abort":
            self._write_abort_files()
            return {"status": "aborted"}
        raise ValueError(f"unsupported method: {method}")

    def _verify_tree(self, params: dict) -> dict:
        session = params.get("session") or {}
        tree = params.get("tree") or params.get("verifier_job", {}).get("tree") or {}
        session_id = str(session.get("session_id") or params.get("session_id") or "sess-mock")
        branches = tree.get("branches") or []
        first = branches[0] if branches else {}
        first_tokens = first.get("candidate_tokens") or []
        accepted_len = min(len(first_tokens), int(params.get("max_accept_len") or 8))
        rollback_branch_ids = [str(branch.get("branch_id")) for branch in branches[1:] if branch.get("branch_id")]
        tree_cid = str(tree.get("tree_cid") or "sha256:" + sha256_hex(json.dumps(tree, sort_keys=True).encode("utf-8")))
        receipt = {
            "schema_version": "ryvion.accepted_token_receipt.v1",
            "receipt_type": "ryvion.accepted_token.v1",
            "method": "verify_tree",
            "session_id": session_id,
            "workgraph_id": str(session.get("workgraph_id") or tree.get("workgraph_id") or ""),
            "window_id": str(tree.get("window_id") or ""),
            "tree_cid": tree_cid,
            "kv_epoch": int(self.sessions.get(session_id, {}).get("kv_epoch") or 0),
            "accepted_len": accepted_len,
            "accepted_tokens": accepted_len,
            "commit_range": {"start": 0, "end": accepted_len},
            "rollback_branch_ids": rollback_branch_ids,
            "status": "verified_mock",
            "latency_ms": 1,
            "energy_mwh": 0,
            "verifier_signature": "sha256:" + sha256_hex(f"{session_id}|{tree_cid}|{accepted_len}".encode("utf-8")),
        }
        probe = {
            "workgraph_id": receipt["workgraph_id"],
            "role_id": "target_verifier",
            "model_hash": str(session.get("model_hash") or tree.get("target_model_hash") or "sha256:mock"),
            "probe_pack_cid": "sha256:mock-probe-pack",
            "feature_scores_bps": {"mock_confidence": 9300, "hallucination_risk": 400},
            "confidence_bps": 9300,
            "answer_confidence_bps": 9300,
            "risk_flags": [],
            "accepted_tokens": accepted_len,
            "early_exit_recommended": False,
            "verifier_signature": receipt["verifier_signature"],
        }
        self.last_receipt = receipt
        self.last_probe = probe
        self._write_final_files()
        return {"status": "verified", "accepted_token_receipt": receipt, "probe_summary": probe}

    def _write_final_files(self) -> None:
        receipt = self.last_receipt or {
            "schema_version": "ryvion.accepted_token_receipt.v1",
            "receipt_type": "ryvion.accepted_token.v1",
            "method": "verify_tree",
            "status": "no_tree_verified",
            "accepted_len": 0,
        }
        probe = self.last_probe or {
            "probe_pack_cid": "sha256:mock-probe-pack",
            "confidence_bps": 0,
            "risk_flags": ["no_tree_verified"],
            "accepted_tokens": 0,
        }
        output = {
            "schema_version": "ryvion.speculative.verifier_contract_test.output.v1",
            "receipt_type": receipt.get("receipt_type"),
            "accepted_len": receipt.get("accepted_len", 0),
            "tree_cid": receipt.get("tree_cid", ""),
            "status": receipt.get("status", ""),
        }
        output_bytes = json.dumps(output, sort_keys=True, separators=(",", ":")).encode("utf-8")
        output_hash = "sha256:" + sha256_hex(output_bytes)
        write_json_atomic(self.work_dir / "verifier_session_receipt.json", receipt)
        write_json_atomic(self.work_dir / "probe_summary.json", probe)
        write_json_atomic(self.work_dir / "output.json", output)
        write_json_atomic(self.work_dir / "receipt.json", {
            "output_hash": output_hash,
            "receipt_type": "ryvion.verifier_session_contract.v1",
            "status": "completed",
            "accepted_value": int(receipt.get("accepted_len") or 0),
            "committed_before_abort": True,
        })
        write_json_atomic(self.work_dir / "metrics.json", {
            "output_name": "output.json",
            "runner": "ryvion-verifier-contract-test",
            "engine": "cpu_mock",
            "accepted_len": int(receipt.get("accepted_len") or 0),
            "duration_ms": 1,
        })

    def _write_abort_files(self) -> None:
        write_json_atomic(self.work_dir / "receipt.partial.json", {
            "output_hash": "sha256:" + sha256_hex(b"verifier_runner_aborted"),
            "receipt_type": "ryvion.verifier_session_contract.v1",
            "status": "aborted",
            "execution_status": "aborted",
            "billing_status": "not_billable_orphaned_compute",
        })


def load_job() -> dict:
    try:
        return json.loads((WORK_DIR / "job.json").read_text(encoding="utf-8"))
    except Exception:
        return {}


def main() -> int:
    server = VerifierSessionServer(socket_path=SOCKET_PATH, work_dir=WORK_DIR)

    def shutdown(signum, frame):
        server._write_abort_files()
        server.shutdown()
        sys.exit(143)

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    print(json.dumps({"event": "verifier_session_socket_starting", "socket": str(SOCKET_PATH)}))
    server.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
