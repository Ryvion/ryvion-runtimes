#!/usr/bin/env python3
"""build_bundle.py — produce a Ryvion EM native bundle (em.bundle.v1).

Builds the on-disk layout documented in ../NATIVE_BUNDLE.md that the node-agent
downloads/verifies/auto-updates and invokes directly (no Docker):

  <out>/<engine>-<version>/
  ├── manifest.json          (em.bundle.v1, Ed25519-signed)
  ├── runner/                (this runner's .py files, frozen)
  │   ├── run.py budget.py geometry.py results.py
  │   ├── engine.py engine_<engine>.py engine_analytic.py <engine>_io.py
  │   └── templates/...
  ├── python/                (portable embedded CPython, per OS x arch)
  ├── engine/                (native FDTD engine + GPU runtime libs)
  └── bundle.sha256          (hash of every staged file)

IMPORTANT — this MUST be RUN ON A TARGET OS + GPU to produce a *real*, runnable
bundle: only there are the right portable CPython, gprMax/openEMS wheels, and
CUDA/ROCm runtime libraries available. On a dev box without those, run with
`--skip-engine --skip-python` to assemble + sign the RUNNER-ONLY skeleton and a
valid manifest (useful for CI/manifest tests); the produced bundle will degrade
to the analytic solver at run time because no native engine is staged. That is
the same fallback discipline the engine modules use.

Signing: pass --signing-key <ed25519 seed file> to sign the manifest. Without a
key the manifest is written with signature "ed25519:UNSIGNED" and the script
exits non-zero unless --allow-unsigned is given (so CI never ships unsigned).

Usage:
  python tools/build_bundle.py --engine gprmax --engine-version 3.1.7+cuda12 \
      --os linux --arch x86_64 --gpu cuda --gpu rocm --out ./dist \
      --engine-root /opt/gprmax-bundle --python-root /opt/python-portable \
      --signing-key ~/.ryvion/em_bundle_ed25519.key

  # dev / CI (runner-only skeleton, manifest still valid):
  python tools/build_bundle.py --engine gprmax --engine-version 3.1.7+dev \
      --os linux --arch x86_64 --gpu none --out ./dist \
      --skip-engine --skip-python --allow-unsigned
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from typing import Dict, List, Optional

BUNDLE_SCHEMA = "em.bundle.v1"

# Runner files frozen into the bundle's runner/ dir. Mirrors the OCI image's
# py_compile set so a bundle runs identically to the container.
RUNNER_FILES = [
    "run.py", "budget.py", "geometry.py", "results.py",
    "engine.py", "engine_analytic.py",
    "engine_gprmax.py", "engine_openems.py", "engine_meep.py",
    "gprmax_io.py", "openems_io.py", "meep_io.py",
    "schema.json",
]
RUNNER_PKG_DIRS = ["templates"]

VALID_ENGINES = ("gprmax", "openems", "meep")
VALID_GPU = ("cuda", "rocm", "metal", "none")


def runner_src_dir() -> str:
    """Directory holding the runner sources (parent of this tools/ dir)."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def stage_runner(dest_runner: str) -> None:
    src = runner_src_dir()
    os.makedirs(dest_runner, exist_ok=True)
    for name in RUNNER_FILES:
        s = os.path.join(src, name)
        if not os.path.exists(s):
            raise FileNotFoundError(f"runner file missing: {s}")
        shutil.copy2(s, os.path.join(dest_runner, name))
    for pkg in RUNNER_PKG_DIRS:
        s = os.path.join(src, pkg)
        d = os.path.join(dest_runner, pkg)
        if os.path.isdir(d):
            shutil.rmtree(d)
        shutil.copytree(s, d, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))


def stage_tree(src_root: str, dest: str, label: str) -> None:
    if not src_root:
        raise ValueError(f"--{label}-root is required unless --skip-{label} is set")
    if not os.path.isdir(src_root):
        raise FileNotFoundError(f"{label} root not found: {src_root}")
    if os.path.isdir(dest):
        shutil.rmtree(dest)
    shutil.copytree(src_root, dest, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def walk_files(root: str) -> List[str]:
    out: List[str] = []
    for dirpath, _dirs, files in os.walk(root):
        for fn in sorted(files):
            out.append(os.path.join(dirpath, fn))
    return sorted(out)


def write_bundle_sha256(bundle_dir: str) -> Dict[str, str]:
    """Hash every staged file (except bundle.sha256/manifest.json themselves),
    write bundle.sha256, and return {relpath: sha256} for the manifest digest.
    """
    digests: Dict[str, str] = {}
    excluded = {"bundle.sha256", "manifest.json"}
    for path in walk_files(bundle_dir):
        rel = os.path.relpath(path, bundle_dir)
        if rel in excluded:
            continue
        digests[rel] = sha256_file(path)
    lines = [f"{digests[rel]}  {rel}" for rel in sorted(digests)]
    with open(os.path.join(bundle_dir, "bundle.sha256"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + ("\n" if lines else ""))
    return digests


def aggregate_digest(digests: Dict[str, str]) -> str:
    """A single sha256 over the sorted (path, hash) pairs — the manifest sha256."""
    h = hashlib.sha256()
    for rel in sorted(digests):
        h.update(rel.encode("utf-8"))
        h.update(b"\x00")
        h.update(digests[rel].encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def total_size(bundle_dir: str) -> int:
    return sum(os.path.getsize(p) for p in walk_files(bundle_dir))


def entrypoint_for(os_name: str) -> List[str]:
    if os_name == "windows":
        return ["python/python.exe", "runner/run.py"]
    return ["python/bin/python3", "runner/run.py"]


def sign_manifest(manifest_bytes: bytes, key_path: Optional[str]) -> str:
    """Ed25519-sign the canonical manifest bytes. Returns 'ed25519:<hex>'.

    The key file holds a 32-byte raw seed (hex or binary). When NO key is given,
    returns 'ed25519:UNSIGNED'. When a key IS given but no Ed25519 backend can
    sign (or the seed is invalid), RAISES — it never silently emits an UNSIGNED
    manifest the operator believes is signed.
    """
    if not key_path:
        return "ed25519:UNSIGNED"
    seed = _load_seed(key_path)
    errors: List[str] = []
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (  # type: ignore
            Ed25519PrivateKey,
        )

        sk = Ed25519PrivateKey.from_private_bytes(seed)
        return "ed25519:" + sk.sign(manifest_bytes).hex()
    except Exception as exc:  # noqa: BLE001
        errors.append(f"cryptography: {exc}")
    try:
        import nacl.signing  # type: ignore

        sk = nacl.signing.SigningKey(seed)
        return "ed25519:" + sk.sign(manifest_bytes).signature.hex()
    except Exception as exc:  # noqa: BLE001
        errors.append(f"pynacl: {exc}")
    raise RuntimeError(
        "signing key provided but could not sign (install `cryptography` or "
        "`pynacl`, and check the seed): " + "; ".join(errors)
    )


def _load_seed(key_path: str) -> bytes:
    with open(key_path, "rb") as fh:
        raw = fh.read().strip()
    # accept hex (64 chars) or raw 32 bytes.
    if len(raw) == 64:
        try:
            return bytes.fromhex(raw.decode("ascii"))
        except Exception:
            pass
    if len(raw) >= 32:
        return raw[:32]
    raise ValueError("signing key must be a 32-byte ed25519 seed (raw or hex)")


def bundled_template_names() -> List[str]:
    """Template names frozen into the bundle, sourced from the runner registry.

    Imported from the staged runner so the manifest always matches the actual
    templates shipped (no hand-maintained list to drift).
    """
    src = runner_src_dir()
    if src not in sys.path:
        sys.path.insert(0, src)
    from templates import registry  # type: ignore

    return registry.names()


def build_manifest(args, digests: Dict[str, str], size_bytes: int) -> Dict[str, object]:
    return {
        "schema": BUNDLE_SCHEMA,
        "engine": args.engine,
        "engine_version": args.engine_version,
        "os": args.os,
        "arch": args.arch,
        "gpu": list(args.gpu),
        "entrypoint": entrypoint_for(args.os),
        "templates": bundled_template_names(),
        "runner_only": bool(args.skip_engine),
        "sha256": aggregate_digest(digests),
        "size_bytes": size_bytes,
        "signature": "ed25519:PENDING",  # replaced after canonical serialisation.
    }


def canonical_manifest_bytes(manifest: Dict[str, object]) -> bytes:
    """Bytes signed/verified: manifest minus the signature field, canonical JSON."""
    unsigned = {k: v for k, v in manifest.items() if k != "signature"}
    return json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _genkey(argv: List[str]) -> int:
    """Generate an Ed25519 EM-bundle signing keypair.

    Writes the PRIVATE seed (hex) to a 0600 file (store it as a secret, never
    commit), and PRINTS the PUBLIC key — pin it as emBundlePinnedSigningKeyB64 in
    ryvion-node (or set RYV_EM_BUNDLE_PUBKEY).
    """
    import base64

    ap = argparse.ArgumentParser(description="Generate an Ed25519 EM-bundle signing keypair.")
    ap.add_argument("--genkey", action="store_true")
    ap.add_argument("--key-out", default="em_bundle_ed25519.key",
                    help="path to write the PRIVATE seed (hex); default ./em_bundle_ed25519.key")
    args = ap.parse_args(argv)

    seed = pub = None
    try:
        from cryptography.hazmat.primitives import serialization  # type: ignore
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey  # type: ignore

        sk = Ed25519PrivateKey.generate()
        seed = sk.private_bytes(serialization.Encoding.Raw, serialization.PrivateFormat.Raw,
                                serialization.NoEncryption())
        pub = sk.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    except Exception:
        try:
            import nacl.signing  # type: ignore

            sk = nacl.signing.SigningKey.generate()
            seed = bytes(sk)
            pub = bytes(sk.verify_key)
        except Exception as exc:  # noqa: BLE001
            sys.stderr.write(f"ERROR: need `cryptography` or `pynacl` to generate a key: {exc}\n")
            return 2

    fd = os.open(args.key_out, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as fh:
        fh.write(seed.hex() + "\n")

    print("Ed25519 EM-bundle signing keypair generated.")
    print(f"  PRIVATE seed -> {args.key_out}  (chmod 600 — store as a SECRET, never commit)")
    print(f"  PUBLIC (base64, pin as emBundlePinnedSigningKeyB64): {base64.standard_b64encode(pub).decode()}")
    print(f"  PUBLIC (hex, or set RYV_EM_BUNDLE_PUBKEY):           {pub.hex()}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:]) if argv is None else list(argv)
    if "--genkey" in argv:
        return _genkey(argv)
    ap = argparse.ArgumentParser(description="Build a Ryvion EM native bundle (em.bundle.v1).")
    ap.add_argument("--engine", required=True, choices=VALID_ENGINES)
    ap.add_argument("--engine-version", required=True)
    ap.add_argument("--os", required=True, choices=("linux", "windows", "darwin"))
    ap.add_argument("--arch", required=True)
    ap.add_argument("--gpu", action="append", default=[], choices=VALID_GPU,
                    help="repeatable; e.g. --gpu cuda --gpu rocm, or --gpu none")
    ap.add_argument("--out", required=True, help="output dir; bundle written under <out>/<engine>-<version>/")
    ap.add_argument("--engine-root", default="", help="dir to copy into engine/ (native FDTD + GPU libs)")
    ap.add_argument("--python-root", default="", help="dir to copy into python/ (portable CPython)")
    ap.add_argument("--skip-engine", action="store_true", help="omit engine/ (runner-only skeleton)")
    ap.add_argument("--skip-python", action="store_true", help="omit python/ (use node's python)")
    ap.add_argument("--signing-key", default="", help="ed25519 seed file (hex or raw)")
    ap.add_argument("--allow-unsigned", action="store_true", help="permit an UNSIGNED manifest")
    args = ap.parse_args(argv)

    if not args.gpu:
        args.gpu = ["none"]

    bundle_dir = os.path.join(args.out, f"{args.engine}-{args.engine_version}")
    if os.path.isdir(bundle_dir):
        shutil.rmtree(bundle_dir)
    os.makedirs(bundle_dir, exist_ok=True)

    # 1. runner sources (always).
    stage_runner(os.path.join(bundle_dir, "runner"))

    # 2. portable python (target-only).
    if not args.skip_python:
        stage_tree(args.python_root, os.path.join(bundle_dir, "python"), "python")
    else:
        sys.stderr.write("WARN: --skip-python -> bundle relies on node's python.\n")

    # 3. native engine + GPU runtime (target-only).
    if not args.skip_engine:
        stage_tree(args.engine_root, os.path.join(bundle_dir, "engine"), "engine")
    else:
        sys.stderr.write("WARN: --skip-engine -> bundle has NO native engine; "
                         "it will degrade to the analytic solver at run time.\n")

    # 4. hash + manifest.
    digests = write_bundle_sha256(bundle_dir)
    size_bytes = total_size(bundle_dir)
    manifest = build_manifest(args, digests, size_bytes)

    try:
        sig = sign_manifest(canonical_manifest_bytes(manifest), args.signing_key or None)
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(f"ERROR: manifest signing failed: {exc}\n")
        return 2
    manifest["signature"] = sig

    with open(os.path.join(bundle_dir, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
        fh.write("\n")

    print(f"built bundle: {bundle_dir}")
    print(f"  engine={args.engine} version={args.engine_version} os={args.os} "
          f"arch={args.arch} gpu={','.join(args.gpu)}")
    print(f"  files={len(digests)} size_bytes={size_bytes} sha256={manifest['sha256'][:16]}…")
    print(f"  signature={sig.split(':')[0]}:{sig.split(':')[1][:16]}…")

    if sig.endswith("UNSIGNED") and not args.allow_unsigned:
        sys.stderr.write("ERROR: manifest is UNSIGNED and --allow-unsigned not set. "
                         "Provide --signing-key (install `cryptography` or `pynacl`).\n")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
