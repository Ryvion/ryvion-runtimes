"""Ryvion SGLang verifier-runner v8.

This runner is the first GPU data-plane implementation for the Ryvion
VerifierSessionContract. It exposes a newline-delimited JSON-RPC service over a
mounted Unix socket, keeps a logical verifier session alive across waves, and
uses SGLang's prefix cache/RadixAttention path by repeatedly scoring committed
prefix + candidate branches through one long-lived offline Engine.

The bridge intentionally stores only token IDs, logprob margins, receipts, and
probe summaries. It never writes raw prompts, raw text output, activations, or
KV tensors to /work.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
import signal
import socket
import sys
import threading
import time
from pathlib import Path
from typing import Any

WORK_DIR = Path(os.environ.get("RYV_WORK_DIR", "/work"))
SOCKET_PATH = Path(os.environ.get("RYV_VERIFIER_SESSION_SOCKET", "/work/verifier_session.sock"))
DEFAULT_MODELS_DIR = Path(os.environ.get("RYV_MODELS_DIR", "/models"))
DEFAULT_CAS_DIR = Path(os.environ.get("RYV_LOCAL_CAS_DIR", "/cas"))

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    tmp.replace(path)


def int_list(value: Any) -> list[int]:
    if not isinstance(value, list):
        return []
    out: list[int] = []
    for item in value:
        try:
            token = int(item)
        except (TypeError, ValueError):
            continue
        if token < 0:
            continue
        out.append(token)
    return out


def normalize_digest(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    if raw.startswith("sha256:"):
        return raw
    if len(raw) == 64 and all(ch in "0123456789abcdefABCDEF" for ch in raw):
        return "sha256:" + raw.lower()
    return raw


def resolve_model_path(job: dict | None = None, session: dict | None = None) -> Path:
    job = job or {}
    session = session or {}
    candidates = [
        os.environ.get("RYV_MODEL_PATH"),
        session.get("model_path"),
        session.get("artifact_path"),
        job.get("model_path"),
        job.get("artifact_path"),
        job.get("local_model_path"),
    ]
    model_id = str(session.get("model_id") or job.get("model_id") or job.get("model") or "").strip()
    if model_id and "/" not in model_id and "\\" not in model_id and "://" not in model_id:
        candidates.extend([DEFAULT_MODELS_DIR / model_id, DEFAULT_CAS_DIR / model_id])
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate)
        if path.exists():
            return path
    raise ValueError("local safetensors model_path required; remote model identifiers are not allowed")


def validate_model_artifact_policy(model_path: Path) -> dict:
    raw = str(model_path)
    if "://" in raw:
        raise ValueError("remote model URLs are forbidden in offline verifier runner")
    if not model_path.exists():
        raise ValueError(f"model path does not exist: {model_path}")
    safetensors = sorted(model_path.rglob("*.safetensors")) if model_path.is_dir() else ([model_path] if model_path.suffix == ".safetensors" else [])
    if not safetensors:
        raise ValueError("model artifact must include .safetensors weights")
    forbidden = []
    if model_path.is_dir():
        for suffix in (".bin", ".pt", ".pth", ".gguf", ".ckpt"):
            forbidden.extend(model_path.rglob(f"*{suffix}"))
    elif model_path.suffix in (".bin", ".pt", ".pth", ".gguf", ".ckpt"):
        forbidden.append(model_path)
    if forbidden:
        raise ValueError("non-safetensors weight formats are forbidden for sglang-verifier-runner-v8")
    return {
        "format": "safetensors",
        "safetensors_files": len(safetensors),
        "weight_loader_disable_mmap": False,
        "artifact_root": str(model_path),
    }


class FakeSGLangBackend:
    """Deterministic test backend.

    Keys are tuple(prefix_plus_candidate_prefix). Values describe what the
    target verifier would greedily choose at that position.
    """

    def __init__(self, token_rules: dict[tuple[int, ...], dict[str, Any]] | None = None):
        self.token_rules = token_rules or {}
        self.model_hash = "sha256:fake-sglang"
        self.artifact_policy = {"format": "safetensors", "weight_loader_disable_mmap": False}

    def score_branch(self, prefix_tokens: list[int], candidate_tokens: list[int], branch_id: str = "") -> list[dict]:
        records: list[dict] = []
        running = list(prefix_tokens)
        for token in candidate_tokens:
            key = tuple(running + [token])
            rule = self.token_rules.get(key, {"top_token_id": token, "logprob": -0.1, "runner_up_logprob": -1.0})
            top_logprob = float(rule.get("top_logprob", rule.get("logprob", -0.1)))
            candidate_logprob = float(rule.get("logprob", top_logprob if int(rule.get("top_token_id", token)) == token else -2.0))
            runner_up = float(rule.get("runner_up_logprob", top_logprob - 1.0))
            top_token_id = int(rule.get("top_token_id", token))
            if top_token_id == token:
                margin = candidate_logprob - runner_up
            else:
                margin = candidate_logprob - top_logprob
            records.append({
                "token_id": int(token),
                "top_token_id": top_token_id,
                "logprob": candidate_logprob,
                "top_logprob": top_logprob,
                "runner_up_logprob": runner_up,
                "margin": margin,
            })
            running.append(token)
        return records

    def shutdown(self) -> None:
        return None


class SGLangBackend:
    def __init__(self, model_path: Path, job: dict | None = None, session: dict | None = None):
        self.model_path = Path(model_path)
        self.artifact_policy = validate_model_artifact_policy(self.model_path)
        self.model_hash = model_hash(self.model_path)
        try:
            import sglang as sgl  # type: ignore
        except Exception as exc:  # pragma: no cover - depends on GPU image
            raise RuntimeError("sglang package is required for production verifier runner") from exc
        engine_kwargs = {
            "model_path": str(self.model_path),
            "trust_remote_code": env_bool("RYV_TRUST_REMOTE_CODE", False),
            "disable_radix_cache": False,
            "page_size": int(os.environ.get("RYV_SGLANG_PAGE_SIZE", "1")),
            "load_format": "safetensors",
            "weight_loader_disable_mmap": False,
            "log_level": os.environ.get("RYV_SGLANG_LOG_LEVEL", "warning"),
        }
        engine_ctor = sgl.Engine
        accepted_kwargs = filter_engine_kwargs(engine_ctor, engine_kwargs)
        if "weight_loader_disable_mmap" not in accepted_kwargs:
            raise RuntimeError("SGLang Engine does not expose weight_loader_disable_mmap; mmap enforcement unavailable")
        self.engine = engine_ctor(**accepted_kwargs)

    def score_branch(self, prefix_tokens: list[int], candidate_tokens: list[int], branch_id: str = "") -> list[dict]:
        if not candidate_tokens:
            return []
        input_ids = list(prefix_tokens) + list(candidate_tokens)
        request = {
            "input_ids": input_ids,
            "sampling_params": {"temperature": 0.0, "max_new_tokens": int(os.environ.get("RYV_SGLANG_VERIFY_MAX_NEW_TOKENS", "0"))},
            "return_logprob": True,
            "logprob_start_len": max(0, len(prefix_tokens)),
            "top_logprobs_num": 2,
            "token_ids_logprob": list(candidate_tokens),
            "return_text_in_logprobs": False,
        }
        output = self.engine.generate(**filter_generate_kwargs(self.engine.generate, request))
        if isinstance(output, list):
            output = output[0] if output else {}
        meta = output.get("meta_info", output) if isinstance(output, dict) else {}
        records = extract_logprob_records(meta, continuation_offset=len(prefix_tokens), candidate_tokens=candidate_tokens)
        if len(records) < len(candidate_tokens):
            raise RuntimeError("SGLang did not return enough token logprob records for exact greedy verification")
        return records[:len(candidate_tokens)]

    def shutdown(self) -> None:
        shutdown = getattr(self.engine, "shutdown", None)
        if callable(shutdown):
            shutdown()


def env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def filter_engine_kwargs(callable_obj, kwargs: dict) -> dict:
    try:
        sig = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return kwargs
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in sig.parameters.values()):
        return kwargs
    return {key: value for key, value in kwargs.items() if key in sig.parameters}


def filter_generate_kwargs(callable_obj, request: dict) -> dict:
    try:
        sig = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return request
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in sig.parameters.values()):
        return request
    if "prompts" in sig.parameters and "input_ids" not in sig.parameters:
        # Older offline Engine variants accept generate(prompts, sampling_params)
        # only. They cannot do token-exact verification.
        raise RuntimeError("SGLang Engine.generate does not expose input_ids; exact token verification unavailable")
    return {key: value for key, value in request.items() if key in sig.parameters}


def model_hash(model_path: Path) -> str:
    h = hashlib.sha256()
    if model_path.is_file():
        h.update(model_path.name.encode("utf-8"))
        h.update(str(model_path.stat().st_size).encode("utf-8"))
        return "sha256:" + h.hexdigest()
    for path in sorted(model_path.rglob("*.safetensors")):
        rel = path.relative_to(model_path)
        h.update(str(rel).encode("utf-8"))
        h.update(str(path.stat().st_size).encode("utf-8"))
    return "sha256:" + h.hexdigest()


def parse_top_logprobs(raw: Any) -> dict[int, float]:
    out: dict[int, float] = {}
    if isinstance(raw, dict):
        iterable = raw.items()
    elif isinstance(raw, list):
        iterable = []
        for item in raw:
            if isinstance(item, dict):
                iterable = list(iterable) + list(item.items())
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                iterable = list(iterable) + [(item[0], item[1])]
    else:
        iterable = []
    for key, value in iterable:
        token_id = token_id_from_logprob_key(key)
        if token_id is None:
            continue
        try:
            out[token_id] = float(value)
        except (TypeError, ValueError):
            continue
    return out


def token_id_from_logprob_key(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        raw = value.strip()
        if raw.startswith("token_id:"):
            raw = raw.split(":", 1)[1]
        try:
            return int(raw)
        except ValueError:
            return None
    if isinstance(value, (list, tuple)) and value:
        return token_id_from_logprob_key(value[-1])
    return None


def logprob_from_record(raw: Any) -> tuple[float | None, int | None]:
    if isinstance(raw, dict):
        lp = raw.get("logprob", raw.get("log_prob"))
        token = raw.get("token_id", raw.get("id"))
        try:
            return float(lp), token_id_from_logprob_key(token)
        except (TypeError, ValueError):
            return None, token_id_from_logprob_key(token)
    if isinstance(raw, (list, tuple)):
        logprob = None
        token = None
        for item in raw:
            if isinstance(item, (float, int)) and logprob is None and (isinstance(item, float) or item < 0):
                logprob = float(item)
            token_candidate = token_id_from_logprob_key(item)
            if token_candidate is not None:
                token = token_candidate
        return logprob, token
    return None, None


def extract_logprob_records(meta: dict, continuation_offset: int, candidate_tokens: list[int]) -> list[dict]:
    token_logprobs = (
        meta.get("input_token_logprobs")
        or meta.get("token_logprobs")
        or meta.get("prompt_token_logprobs")
        or []
    )
    top_logprobs = (
        meta.get("input_top_logprobs")
        or meta.get("top_logprobs")
        or meta.get("prompt_top_logprobs")
        or []
    )
    records: list[dict] = []
    for i, token in enumerate(candidate_tokens):
        idx = continuation_offset + i
        raw_lp = token_logprobs[idx] if idx < len(token_logprobs) else (token_logprobs[i] if i < len(token_logprobs) else None)
        candidate_logprob, parsed_token = logprob_from_record(raw_lp)
        if parsed_token is not None and parsed_token != token and idx < len(token_logprobs):
            candidate_logprob = None
        raw_top = top_logprobs[idx] if idx < len(top_logprobs) else (top_logprobs[i] if i < len(top_logprobs) else None)
        top = parse_top_logprobs(raw_top)
        if candidate_logprob is None and token in top:
            candidate_logprob = top[token]
        top_token_id = token
        top_logprob = candidate_logprob if candidate_logprob is not None else -math.inf
        runner_up_logprob = -math.inf
        if top:
            ordered = sorted(top.items(), key=lambda item: item[1], reverse=True)
            top_token_id, top_logprob = ordered[0]
            if len(ordered) > 1:
                runner_up_logprob = ordered[1][1]
            if token in top:
                candidate_logprob = top[token]
        if candidate_logprob is None:
            candidate_logprob = -math.inf
        if runner_up_logprob == -math.inf:
            runner_up_logprob = top_logprob - 1.0 if math.isfinite(top_logprob) else candidate_logprob - 1.0
        if int(top_token_id) == int(token):
            margin = candidate_logprob - runner_up_logprob
        else:
            # Rejected candidate tokens are scored against the target model's
            # greedy token. A negative margin is useful probe evidence and
            # prevents rejected tokens from looking as confident as the
            # runner-up candidate in a top-k list.
            margin = candidate_logprob - top_logprob
        records.append({
            "token_id": int(token),
            "top_token_id": int(top_token_id),
            "logprob": float(candidate_logprob),
            "top_logprob": float(top_logprob),
            "runner_up_logprob": float(runner_up_logprob),
            "margin": float(margin),
        })
    return records


def choose_best_branch(prefix_tokens: list[int], branches: list[dict], backend) -> dict:
    best = {
        "branch": {},
        "accepted_len": 0,
        "accepted_token_ids": [],
        "records": [],
        "margin_sum": -math.inf,
    }
    for branch in branches:
        branch_id = str(branch.get("branch_id") or "")
        candidate_tokens = int_list(branch.get("candidate_tokens"))
        records = backend.score_branch(prefix_tokens, candidate_tokens, branch_id=branch_id)
        accepted_len = 0
        margin_sum = 0.0
        for record in records:
            if int(record.get("token_id", -1)) != int(record.get("top_token_id", -2)):
                break
            accepted_len += 1
            margin_sum += float(record.get("margin") or 0.0)
        if accepted_len > best["accepted_len"] or (accepted_len == best["accepted_len"] and margin_sum > best["margin_sum"]):
            best = {
                "branch": branch,
                "accepted_len": accepted_len,
                "accepted_token_ids": candidate_tokens[:accepted_len],
                "records": records,
                "margin_sum": margin_sum,
            }
    return best


class SGLangVerifierSessionServer:
    def __init__(self, socket_path: Path = SOCKET_PATH, work_dir: Path = WORK_DIR, backend=None, job: dict | None = None):
        self.socket_path = Path(socket_path)
        self.work_dir = Path(work_dir)
        self.job = job or load_job(self.work_dir / "job.json")
        self.backend = backend
        self._listener: socket.socket | None = None
        self._shutdown = threading.Event()
        self._ready = threading.Event()
        self.sessions: dict[str, dict] = {}
        self.last_receipt: dict | None = None
        self.last_probe: dict | None = None
        self.started_at = time.time()

    def _backend(self, session: dict | None = None):
        if self.backend is not None:
            return self.backend
        model_path = resolve_model_path(self.job, session)
        self.backend = SGLangBackend(model_path=model_path, job=self.job, session=session)
        return self.backend

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
        if self.backend is not None:
            self.backend.shutdown()

    def _serve_conn(self, conn: socket.socket) -> None:
        with conn:
            reader = conn.makefile("r", encoding="utf-8")
            writer = conn.makefile("w", encoding="utf-8")
            for line in reader:
                if not line.strip():
                    continue
                request_id = None
                try:
                    req = json.loads(line)
                    request_id = req.get("id")
                    result = self.handle(str(req.get("method") or ""), req.get("params") or {})
                    response = {"jsonrpc": "2.0", "id": request_id, "result": result}
                except Exception as exc:
                    response = {"jsonrpc": "2.0", "id": request_id, "error": {"code": "sglang_verifier_error", "message": str(exc)}}
                writer.write(json.dumps(response, sort_keys=True, separators=(",", ":")) + "\n")
                writer.flush()

    def handle(self, method: str, params: dict) -> dict:
        if method == "start_session":
            session = params.get("session") or {}
            session_id = str(session.get("session_id") or params.get("session_id") or "sess-sglang")
            prefix_tokens = int_list(session.get("prefix_tokens") or session.get("prefix_input_ids"))
            self._backend(session)
            self.sessions[session_id] = {"session": session, "prefix_tokens": prefix_tokens, "committed_tokens": [], "kv_epoch": 0}
            return {"status": "started", "session_id": session_id, "kv_epoch": 0, "prefix_cache": "sglang_radix_attention"}
        if method == "prefill":
            session_id = str(params.get("session_id") or "sess-sglang")
            state = self.sessions.setdefault(session_id, {"session": {}, "prefix_tokens": [], "committed_tokens": [], "kv_epoch": 0})
            if "prefix_tokens" in params or "prefix_input_ids" in params:
                state["prefix_tokens"] = int_list(params.get("prefix_tokens") or params.get("prefix_input_ids"))
            state["prefix_hash"] = normalize_digest(params.get("prefix_hash"))
            return {"status": "prefilled", "session_id": session_id, "kv_epoch": state["kv_epoch"], "cache_policy": "prefix_cache_hot"}
        if method == "verify_tree":
            return self._verify_tree(params)
        if method == "commit":
            session_id = str(params.get("session_id") or "sess-sglang")
            state = self.sessions.setdefault(session_id, {"session": {}, "prefix_tokens": [], "committed_tokens": [], "kv_epoch": 0})
            accepted_tokens = int_list(params.get("accepted_token_ids"))
            if not accepted_tokens:
                accepted_len = int(params.get("accepted_len") or 0)
                accepted_tokens = (self.last_receipt or {}).get("accepted_token_ids", [])[:accepted_len]
            state["committed_tokens"] = list(state.get("committed_tokens") or []) + list(accepted_tokens)
            state["kv_epoch"] = int(state.get("kv_epoch") or 0) + 1
            if self.last_receipt is not None:
                self.last_receipt["committed_before_abort"] = True
                self.last_receipt["commit_status"] = "committed"
                self._write_partial_evidence_files("committed_partial")
            return {"status": "committed", "session_id": session_id, "accepted_len": len(accepted_tokens), "kv_epoch": state["kv_epoch"]}
        if method == "rollback":
            return {"status": "rolled_back", "rollback_branch_ids": params.get("branch_ids") or params.get("rollback_branch_ids") or []}
        if method in ("close", "close_session"):
            session_id = str(params.get("session_id") or "sess-sglang")
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
        session_id = str(session.get("session_id") or params.get("session_id") or "sess-sglang")
        state = self.sessions.setdefault(session_id, {"session": session, "prefix_tokens": int_list(session.get("prefix_tokens")), "committed_tokens": [], "kv_epoch": 0})
        branches = tree.get("branches") or []
        if not isinstance(branches, list):
            branches = []
        prefix_tokens = list(state.get("prefix_tokens") or []) + list(state.get("committed_tokens") or [])
        backend = self._backend(state.get("session") or session)
        started = time.time()
        best = choose_best_branch(prefix_tokens, branches, backend)
        latency_ms = int((time.time() - started) * 1000)
        accepted_len = int(best["accepted_len"])
        accepted_token_ids = list(best["accepted_token_ids"])
        branch = best["branch"] or {}
        rollback_branch_ids = [str(item.get("branch_id")) for item in branches if item.get("branch_id") and item.get("branch_id") != branch.get("branch_id")]
        records = best["records"]
        margins = [float(record.get("margin") or 0.0) for record in records[:accepted_len]]
        min_margin = min(margins) if margins else 0.0
        avg_margin = sum(margins) / len(margins) if margins else 0.0
        tree_cid = normalize_digest(tree.get("tree_cid")) or "sha256:" + sha256_hex(json.dumps(tree, sort_keys=True).encode("utf-8"))
        model_hash_value = str((state.get("session") or session).get("model_hash") or getattr(backend, "model_hash", "") or "sha256:unknown")
        signature = "sha256:" + sha256_hex(f"{session_id}|{tree_cid}|{accepted_len}|{accepted_token_ids}".encode("utf-8"))
        receipt = {
            "schema_version": "ryvion.accepted_token_receipt.v1",
            "receipt_type": "ryvion.accepted_token.v1",
            "method": "verify_tree",
            "session_id": session_id,
            "workgraph_id": str((state.get("session") or session).get("workgraph_id") or tree.get("workgraph_id") or ""),
            "window_id": str(tree.get("window_id") or ""),
            "tree_cid": tree_cid,
            "branch_id": str(branch.get("branch_id") or ""),
            "kv_epoch": int(state.get("kv_epoch") or 0),
            "decoding_mode": "greedy_exact",
            "temperature_milli": 0,
            "accepted_len": accepted_len,
            "accepted_tokens": accepted_len,
            "accepted_token_ids": accepted_token_ids,
            "commit_range": {"start": len(prefix_tokens), "end": len(prefix_tokens) + accepted_len},
            "rollback_branch_ids": rollback_branch_ids,
            "target_model_hash": model_hash_value,
            "status": "verified_exact_greedy",
            "latency_ms": latency_ms,
            "energy_mwh": 0,
            "verifier_signature": signature,
        }
        probe = {
            "schema_version": "ryvion.probe_summary.v1",
            "workgraph_id": receipt["workgraph_id"],
            "role_id": "target_verifier",
            "model_hash": model_hash_value,
            "probe_pack_cid": "sha256:sglang-logprob-margin-v0",
            "probe_type": "logprob_margin",
            "feature_scores_bps": {
                "accepted_len": accepted_len,
                "logprob_margin_bps": scale_logprob_margin(avg_margin),
                "min_logprob_margin_bps": scale_logprob_margin(min_margin),
            },
            "branch_count": len(branches),
            "accepted_branch_id": receipt["branch_id"],
            "accepted_tokens": accepted_len,
            "confidence_bps": max(0, min(10000, 5000 + scale_logprob_margin(avg_margin))),
            "answer_confidence_bps": max(0, min(10000, 5000 + scale_logprob_margin(min_margin))),
            "risk_flags": [] if accepted_len > 0 else ["no_accepted_tokens"],
            "early_exit_recommended": False,
            "verifier_signature": signature,
        }
        self.last_receipt = receipt
        self.last_probe = probe
        self._write_partial_evidence_files("verified_uncommitted")
        return {"status": "verified", "accepted_token_receipt": receipt, "probe_summary": probe}

    def _write_partial_evidence_files(self, status: str) -> None:
        if self.last_receipt is not None:
            receipt = dict(self.last_receipt)
            receipt["status"] = status
            write_json_atomic(self.work_dir / "verifier_session_receipt.partial.json", receipt)
        if self.last_probe is not None:
            probe = dict(self.last_probe)
            probe["status"] = status
            write_json_atomic(self.work_dir / "probe_summary.partial.json", probe)

    def _write_final_files(self) -> None:
        receipt = self.last_receipt or {
            "schema_version": "ryvion.accepted_token_receipt.v1",
            "receipt_type": "ryvion.accepted_token.v1",
            "method": "verify_tree",
            "status": "no_tree_verified",
            "accepted_len": 0,
            "accepted_token_ids": [],
            "decoding_mode": "greedy_exact",
        }
        probe = self.last_probe or {
            "schema_version": "ryvion.probe_summary.v1",
            "probe_pack_cid": "sha256:sglang-logprob-margin-v0",
            "confidence_bps": 0,
            "risk_flags": ["no_tree_verified"],
            "accepted_tokens": 0,
        }
        output = {
            "schema_version": "ryvion.sglang_verifier_runner_v8.output.v1",
            "receipt_type": receipt.get("receipt_type"),
            "accepted_len": receipt.get("accepted_len", 0),
            "tree_cid": receipt.get("tree_cid", ""),
            "decoding_mode": receipt.get("decoding_mode", "greedy_exact"),
            "status": receipt.get("status", ""),
        }
        output_bytes = json.dumps(output, sort_keys=True, separators=(",", ":")).encode("utf-8")
        output_hash = "sha256:" + sha256_hex(output_bytes)
        write_json_atomic(self.work_dir / "verifier_session_receipt.json", receipt)
        write_json_atomic(self.work_dir / "probe_summary.json", probe)
        write_json_atomic(self.work_dir / "output.json", output)
        write_json_atomic(self.work_dir / "receipt.json", {
            "output_hash": output_hash,
            "receipt_type": "ryvion.sglang_verifier_session.v1",
            "status": "completed",
            "accepted_value": int(receipt.get("accepted_len") or 0),
            "committed_before_abort": True,
            "engine": "sglang",
        })
        duration_ms = int((time.time() - self.started_at) * 1000)
        write_json_atomic(self.work_dir / "metrics.json", {
            "output_name": "output.json",
            "runner": "sglang-verifier-runner-v8",
            "engine": "sglang",
            "accepted_len": int(receipt.get("accepted_len") or 0),
            "duration_ms": duration_ms,
            "prefix_cache": "sglang_radix_attention",
            "weight_loader_disable_mmap": False,
        })

    def _write_abort_files(self) -> None:
        accepted_value = int((self.last_receipt or {}).get("accepted_len") or 0)
        committed_before_abort = bool((self.last_receipt or {}).get("committed_before_abort"))
        billing_status = "partial_committed_before_abort" if committed_before_abort and accepted_value > 0 else "not_billable_orphaned_compute"
        write_json_atomic(self.work_dir / "receipt.partial.json", {
            "output_hash": "sha256:" + sha256_hex(b"sglang_verifier_runner_aborted"),
            "receipt_type": "ryvion.sglang_verifier_session.v1",
            "status": "aborted",
            "execution_status": "aborted",
            "billing_status": billing_status,
            "accepted_value": accepted_value,
            "committed_before_abort": committed_before_abort,
        })
        verifier_partial = dict(self.last_receipt or {
            "schema_version": "ryvion.accepted_token_receipt.v1",
            "receipt_type": "ryvion.accepted_token.v1",
            "method": "verify_tree",
            "accepted_len": 0,
        })
        verifier_partial["status"] = "aborted_after_commit" if committed_before_abort else "aborted_before_commit"
        verifier_partial["execution_status"] = "aborted"
        verifier_partial["billing_status"] = billing_status
        verifier_partial["committed_before_abort"] = committed_before_abort
        write_json_atomic(self.work_dir / "verifier_session_receipt.partial.json", verifier_partial)
        probe_partial = dict(self.last_probe or {
            "schema_version": "ryvion.probe_summary.v1",
            "probe_pack_cid": "sha256:sglang-logprob-margin-v0",
            "confidence_bps": 0,
            "risk_flags": ["aborted_before_probe"],
        })
        probe_partial["status"] = "aborted"
        probe_partial["execution_status"] = "aborted"
        probe_partial["billing_status"] = billing_status
        write_json_atomic(self.work_dir / "probe_summary.partial.json", probe_partial)
        write_json_atomic(self.work_dir / "metrics.partial.json", {
            "output_name": "output.json",
            "runner": "sglang-verifier-runner-v8",
            "engine": "sglang",
            "status": "aborted",
            "accepted_len": accepted_value,
            "committed_before_abort": committed_before_abort,
        })


def scale_logprob_margin(value: float) -> int:
    if not math.isfinite(value):
        return 0
    return int(max(0, min(10000, round(value * 1000))))


def load_job(path: Path = WORK_DIR / "job.json") -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def build_backend(job: dict):
    if env_bool("RYV_ALLOW_TEST_FAKE_BACKEND", False):
        return FakeSGLangBackend()
    model_path = resolve_model_path(job, job.get("session") if isinstance(job.get("session"), dict) else None)
    return SGLangBackend(model_path=model_path, job=job, session=job.get("session") if isinstance(job.get("session"), dict) else None)


def main() -> int:
    job = load_job()
    backend = None
    if env_bool("RYV_EAGER_LOAD_SGLANG", True):
        backend = build_backend(job)
    server = SGLangVerifierSessionServer(socket_path=SOCKET_PATH, work_dir=WORK_DIR, backend=backend, job=job)

    def shutdown(_signum, _frame):
        server._write_abort_files()
        server.shutdown()
        sys.exit(143)

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    print(json.dumps({"event": "sglang_verifier_session_socket_starting", "socket": str(SOCKET_PATH), "network": "offline"}))
    server.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
