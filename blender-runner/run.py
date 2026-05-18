#!/usr/bin/env python3
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

WORK_DIR = Path("/work").resolve()
JOB_PATH = WORK_DIR / "job.json"
RECEIPT_PATH = WORK_DIR / "receipt.json"
METRICS_PATH = WORK_DIR / "metrics.json"

OUTPUT_FORMATS = {
    "PNG": ".png",
    "JPEG": ".jpg",
    "OPEN_EXR": ".exr",
    "TIFF": ".tif",
}

ENGINES = {
    "CYCLES",
    "BLENDER_EEVEE",
    "BLENDER_EEVEE_NEXT",
    "BLENDER_WORKBENCH",
}


def main() -> int:
    started = time.time()
    try:
        job = read_job()
        scene_path = safe_work_path(job.get("scene_path") or "/work/scene.blend", must_exist=True)
        output_dir = safe_work_path(job.get("output_dir") or "/work/output", must_exist=False)
        output_dir.mkdir(parents=True, exist_ok=True)

        output_prefix = safe_name(job.get("output_prefix") or "frame")
        output_format = normalize_output_format(job.get("output_format") or job.get("image_format") or "PNG")
        output_pattern = str(output_dir / f"{output_prefix}_####")

        command = build_blender_command(job, scene_path, output_pattern, output_format)
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        duration_ms = int((time.time() - started) * 1000)
        if result.returncode != 0:
            raise RunnerError(
                f"blender exited with {result.returncode}",
                metadata={
                    "exit_code": result.returncode,
                    "stderr_tail": tail(result.stderr),
                    "stdout_tail": tail(result.stdout),
                },
            )

        outputs = collect_outputs(output_dir, OUTPUT_FORMATS[output_format])
        if not outputs:
            raise RunnerError("blender completed without producing output files")

        output_hash = hash_outputs(output_dir, outputs)
        receipt = {
            "ok": True,
            "status": "completed",
            "engine": "blender",
            "output_hash": output_hash,
            "output_name": output_dir.name,
            "outputs": [str(path.relative_to(WORK_DIR)) for path in outputs],
            "metadata": {
                "duration_ms": duration_ms,
                "frames_rendered": len(outputs),
                "scene_name": scene_path.name,
                "output_format": output_format,
            },
        }
        metrics = {
            "engine": "blender",
            "duration_ms": duration_ms,
            "frames_rendered": len(outputs),
            "output_hash": output_hash,
            "output_name": output_dir.name,
            "output_bytes": sum(path.stat().st_size for path in outputs),
            "output_format": output_format,
        }
        write_json(RECEIPT_PATH, receipt)
        write_json(METRICS_PATH, metrics)
        print(json.dumps({"ok": True, "output_hash": output_hash}))
        return 0
    except Exception as exc:
        duration_ms = int((time.time() - started) * 1000)
        metadata = getattr(exc, "metadata", {})
        write_json(
            RECEIPT_PATH,
            {
                "ok": False,
                "status": "failed",
                "engine": "blender",
                "error": str(exc),
                "metadata": metadata,
            },
        )
        write_json(
            METRICS_PATH,
            {
                "engine": "blender",
                "duration_ms": duration_ms,
                "output_name": "",
                "error": str(exc),
            },
        )
        print(json.dumps({"ok": False, "error": str(exc)}), file=sys.stderr)
        return 1


def read_job() -> dict:
    with JOB_PATH.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise RunnerError("job.json must contain an object")
    return payload


def safe_work_path(value: object, *, must_exist: bool) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise RunnerError("path must be a non-empty string")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = WORK_DIR / path
    resolved = path.resolve()
    if resolved != WORK_DIR and WORK_DIR not in resolved.parents:
        raise RunnerError(f"path escapes /work: {value}")
    if must_exist and not resolved.exists():
        raise RunnerError(f"missing required path: {resolved}")
    return resolved


def safe_name(value: object) -> str:
    if not isinstance(value, str):
        raise RunnerError("output_prefix must be a string")
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value.strip())
    cleaned = cleaned.strip("._")
    if not cleaned:
        raise RunnerError("output_prefix is empty after sanitization")
    return cleaned[:96]


def normalize_output_format(value: object) -> str:
    fmt = str(value).strip().upper()
    if fmt not in OUTPUT_FORMATS:
        raise RunnerError(f"unsupported output_format: {value}")
    return fmt


def build_blender_command(job: dict, scene_path: Path, output_pattern: str, output_format: str) -> list[str]:
    command = [
        "blender",
        "-b",
        str(scene_path),
        "-o",
        output_pattern,
        "-F",
        output_format,
        "-x",
        "1",
    ]
    expr = blender_python_expr(job)
    if expr:
        command.extend(["--python-expr", expr])

    frame = job.get("frame")
    start = job.get("frame_start")
    end = job.get("frame_end")
    if frame is not None:
        command.extend(["-f", str(as_int(frame, "frame", minimum=0))])
    else:
        start_frame = as_int(start if start is not None else 1, "frame_start", minimum=0)
        end_frame = as_int(end if end is not None else start_frame, "frame_end", minimum=start_frame)
        command.extend(["-s", str(start_frame), "-e", str(end_frame), "-a"])
    return command


def blender_python_expr(job: dict) -> str:
    lines = ["import bpy", "scene = bpy.context.scene"]
    engine = job.get("engine")
    if engine:
        engine_name = str(engine).strip().upper()
        if engine_name not in ENGINES:
            raise RunnerError(f"unsupported engine: {engine}")
        lines.append(f"scene.render.engine = {engine_name!r}")
    if job.get("resolution_x") is not None:
        lines.append(f"scene.render.resolution_x = {as_int(job['resolution_x'], 'resolution_x', minimum=1)}")
    if job.get("resolution_y") is not None:
        lines.append(f"scene.render.resolution_y = {as_int(job['resolution_y'], 'resolution_y', minimum=1)}")
    if job.get("samples") is not None:
        samples = as_int(job["samples"], "samples", minimum=1)
        lines.append("hasattr(scene, 'cycles') and setattr(scene.cycles, 'samples', %d)" % samples)
    return "; ".join(lines) if len(lines) > 2 else ""


def as_int(value: object, field: str, *, minimum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        raise RunnerError(f"{field} must be an integer")
    if parsed < minimum:
        raise RunnerError(f"{field} must be >= {minimum}")
    return parsed


def collect_outputs(output_dir: Path, extension: str) -> list[Path]:
    return sorted(path for path in output_dir.iterdir() if path.is_file() and path.suffix.lower() == extension)


def hash_outputs(output_dir: Path, outputs: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in outputs:
        relative = str(path.relative_to(output_dir)).encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def tail(value: str, limit: int = 4000) -> str:
    value = value or ""
    return value[-limit:]


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, path)


class RunnerError(Exception):
    def __init__(self, message: str, *, metadata: dict | None = None) -> None:
        super().__init__(message)
        self.metadata = metadata or {}


if __name__ == "__main__":
    raise SystemExit(main())
