"""Ryvion speculative draft runtime.

Reads /work/job.json and emits privacy-safe DraftPacket payloads for the
hub-relayed speculative proposal window. The runner is stateless: it does
not own KV cache and never writes raw prompt/output text into packet payloads.
"""

from __future__ import annotations

import hashlib
import json
import os
import signal
import sys
import time
from pathlib import Path
from typing import Iterable

WORK_DIR = Path(os.environ.get("RYV_WORK_DIR", "/work"))
MODEL_DIR = Path(os.environ.get("RYV_MODEL_DIR", "/models"))
DEFAULT_MODEL_FILE = os.environ.get("RYV_MODEL_FILE", "").strip()
DEFAULT_DRAFTER_ID = os.environ.get("RYV_DRAFTER_MODEL_ID", "ryvion-draft-small-model")
DEFAULT_MODEL_HASH = os.environ.get("RYV_MODEL_HASH", "sha256:local-draft-model")
MAX_CONTEXT = int(os.environ.get("RYV_CTX_SIZE", "2048"))
N_THREADS = int(os.environ.get("RYV_THREADS", "4"))
GPU_LAYERS = int(os.environ.get("RYV_GPU_LAYERS", "0"))

_shutdown = False


def _handle_sigterm(signum, frame):
    global _shutdown
    _shutdown = True
    write_json_atomic(WORK_DIR / "receipt.partial.json", {
        "output_hash": "sha256:" + sha256_hex(b"draft_runner_aborted"),
        "receipt_type": "ryvion.draft_packet_batch.v1",
        "status": "aborted",
        "execution_status": "aborted",
        "billing_status": "not_billable_orphaned_compute",
    })
    print(json.dumps({"event": "sigterm_received"}), file=sys.stderr)
    sys.exit(143)


signal.signal(signal.SIGTERM, _handle_sigterm)
signal.signal(signal.SIGINT, _handle_sigterm)


def load_job(path: Path | None = None) -> dict:
    path = path or WORK_DIR / "job.json"
    with path.open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    return loaded if isinstance(loaded, dict) else {}


def find_model(job: dict) -> Path | None:
    requested = str(job.get("model_file") or DEFAULT_MODEL_FILE or "").strip()
    candidates: list[Path] = []
    if requested:
        candidates.append(MODEL_DIR / requested)
        candidates.append(WORK_DIR / requested)
    candidates.append(WORK_DIR / "model.bin")
    try:
        candidates.extend(sorted(MODEL_DIR.glob("*.gguf")))
    except OSError:
        pass
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def generate_candidate_branches(job: dict) -> list[list[int]]:
    branch_count = clamp_int(job.get("branch_count") or job.get("num_branches") or 4, 1, 32)
    horizon = clamp_int(job.get("horizon") or job.get("max_tokens") or 16, 1, 256)
    model_path = find_model(job)
    if model_path is not None:
        branches = llama_cpp_branches(model_path, job, branch_count, horizon)
        if branches:
            return branches
    return deterministic_branches(job, branch_count, horizon)


def llama_cpp_branches(model_path: Path, job: dict, branch_count: int, horizon: int) -> list[list[int]]:
    try:
        from llama_cpp import Llama  # type: ignore
    except Exception as exc:
        print(json.dumps({"event": "llama_cpp_unavailable", "error": str(exc)}), file=sys.stderr)
        return []
    prompt = prompt_for_generation(job)
    if not prompt:
        prompt = "Ryvion draft prefix"
    try:
        llm = Llama(
            model_path=str(model_path),
            n_ctx=max(256, int(job.get("ctx_size") or MAX_CONTEXT)),
            n_threads=max(1, int(job.get("threads") or N_THREADS)),
            n_gpu_layers=int(job.get("gpu_layers") or GPU_LAYERS),
            verbose=False,
        )
        tokenized = llm.tokenize(prompt.encode("utf-8"), add_bos=False)
        branches: list[list[int]] = []
        for branch_index in range(branch_count):
            temperature = 0.0 if branch_index == 0 else min(1.5, 0.35 + branch_index * 0.15)
            result = llm(prompt, max_tokens=horizon, temperature=temperature, top_p=0.95)
            text = str(result["choices"][0].get("text") or "")
            tokens = llm.tokenize(text.encode("utf-8"), add_bos=False)
            if not tokens:
                tokens = tokenized[-horizon:]
            branches.append([int(token) for token in tokens[:horizon]])
        return branches
    except Exception as exc:
        print(json.dumps({"event": "llama_cpp_generation_failed", "error": str(exc)}), file=sys.stderr)
        return []


def deterministic_branches(job: dict, branch_count: int, horizon: int) -> list[list[int]]:
    seed = "|".join([
        str(job.get("window_id") or ""),
        str(job.get("workgraph_id") or ""),
        str(job.get("role_id") or ""),
        str(job.get("parent_prefix_hash") or ""),
        prompt_for_generation(job),
    ])
    branches: list[list[int]] = []
    for branch_index in range(branch_count):
        branch: list[int] = []
        cursor = sha256_hex(f"{seed}|branch:{branch_index}".encode("utf-8"))
        while len(branch) < horizon:
            cursor = sha256_hex(cursor.encode("utf-8"))
            for offset in range(0, len(cursor), 4):
                if len(branch) >= horizon:
                    break
                token = int(cursor[offset:offset + 4], 16) % 32000
                branch.append(token)
        branches.append(branch)
    return branches


def prompt_for_generation(job: dict) -> str:
    prompt = str(job.get("prefix") or job.get("prompt") or "").strip()
    if prompt:
        return prompt
    messages = job.get("messages")
    if isinstance(messages, list):
        safe_parts = []
        for item in messages:
            if isinstance(item, dict):
                safe_parts.append(str(item.get("content") or ""))
        return "\n".join(safe_parts).strip()
    return ""


def build_draft_packets(job: dict, branches: Iterable[Iterable[int]]) -> list[dict]:
    packets: list[dict] = []
    window_id = required_string(job, "window_id")
    parent_prefix_hash = required_string(job, "parent_prefix_hash", "prefix_hash")
    role_id = str(job.get("role_id") or "draft_worker").strip()
    workgraph_id = str(job.get("workgraph_id") or "").strip()
    node_id = str(job.get("node_id") or os.environ.get("RYV_NODE_ID") or "").strip()
    model_hash = str(job.get("model_hash") or DEFAULT_MODEL_HASH).strip()
    drafter_model_id = str(job.get("drafter_model_id") or DEFAULT_DRAFTER_ID).strip()
    deadline_ms = int(job.get("deadline_ms") or 0)
    energy_mwh = int(job.get("energy_mwh") or job.get("energy_estimate_mwh") or 0)
    horizon = int(job.get("horizon") or job.get("max_tokens") or 0)
    for index, raw_tokens in enumerate(branches):
        tokens = [int(token) for token in raw_tokens if int(token) >= 0]
        if not tokens:
            continue
        if horizon > 0:
            tokens = tokens[:horizon]
        confidence_bps = max(1000, min(9800, int(job.get("confidence_bps") or (9200 - index * 350))))
        packet = {
            "packet_id": "pkt_" + sha256_hex(f"{window_id}|{role_id}|{index}|{tokens}".encode("utf-8"))[:24],
            "window_id": window_id,
            "workgraph_id": workgraph_id,
            "role_id": role_id,
            "node_id": node_id,
            "parent_prefix_hash": parent_prefix_hash,
            "candidate_tokens": tokens,
            "model_hash": model_hash,
            "drafter_model_id": drafter_model_id,
            "horizon": len(tokens),
            "confidence_bps": confidence_bps,
            "energy_mwh": energy_mwh,
            "deadline_ms": deadline_ms,
            "submitted_at": utc_now(),
        }
        packet["signature"] = draft_packet_signature(packet)
        packets.append(packet)
    return packets


def required_string(job: dict, *keys: str) -> str:
    for key in keys:
        value = str(job.get(key) or "").strip()
        if value:
            return value
    raise ValueError(f"{'/'.join(keys)} required")


def draft_packet_signature(packet: dict) -> str:
    signable = {
        "window_id": packet.get("window_id"),
        "workgraph_id": packet.get("workgraph_id"),
        "role_id": packet.get("role_id"),
        "parent_prefix_hash": packet.get("parent_prefix_hash"),
        "candidate_tokens": packet.get("candidate_tokens"),
        "model_hash": packet.get("model_hash"),
        "confidence_bps": packet.get("confidence_bps"),
    }
    return "sha256:" + sha256_hex(json.dumps(signable, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def write_outputs(job: dict, packets: list[dict], started: float) -> None:
    output = {
        "schema_version": "ryvion.draft_runner_v8.output.v1",
        "window_id": str(job.get("window_id") or ""),
        "workgraph_id": str(job.get("workgraph_id") or ""),
        "packet_count": len(packets),
        "packet_hashes": ["sha256:" + sha256_hex(json.dumps(packet, sort_keys=True).encode("utf-8")) for packet in packets],
    }
    output_bytes = json.dumps(output, sort_keys=True, separators=(",", ":")).encode("utf-8")
    output_hash = "sha256:" + sha256_hex(output_bytes)
    write_json_atomic(WORK_DIR / "draft_packets.json", packets)
    write_json_atomic(WORK_DIR / "output.json", output)
    write_json_atomic(WORK_DIR / "receipt.json", {
        "output_hash": output_hash,
        "receipt_type": "ryvion.draft_packet_batch.v1",
        "status": "completed",
        "packet_count": len(packets),
        "window_id": str(job.get("window_id") or ""),
        "workgraph_id": str(job.get("workgraph_id") or ""),
    })
    write_json_atomic(WORK_DIR / "metrics.json", {
        "output_name": "output.json",
        "runner": "ryvion-draft-small-model",
        "engine": "llama_cpp_or_deterministic_fallback",
        "duration_ms": int((time.time() - started) * 1000),
        "packet_count": len(packets),
        "output_bytes": len(output_bytes),
    })


def write_json_atomic(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    tmp.replace(path)


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def clamp_int(value, lo: int, hi: int) -> int:
    try:
        parsed = int(value)
    except Exception:
        parsed = lo
    return max(lo, min(hi, parsed))


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def main() -> int:
    started = time.time()
    try:
        job = load_job()
        branches = generate_candidate_branches(job)
        packets = build_draft_packets(job, branches)
        write_outputs(job, packets, started)
        print(json.dumps({"ok": True, "packet_count": len(packets), "output_name": "output.json"}))
        return 0
    except Exception as exc:
        output_hash = "sha256:" + sha256_hex(str(exc).encode("utf-8"))
        write_json_atomic(WORK_DIR / "receipt.json", {
            "output_hash": output_hash,
            "receipt_type": "ryvion.draft_packet_batch.v1",
            "status": "failed",
            "error_code": "draft_runner_failed",
        })
        write_json_atomic(WORK_DIR / "metrics.json", {
            "output_name": "output.json",
            "runner": "ryvion-draft-small-model",
            "duration_ms": int((time.time() - started) * 1000),
            "error_code": "draft_runner_failed",
        })
        print(json.dumps({"ok": False, "error": str(exc)}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
