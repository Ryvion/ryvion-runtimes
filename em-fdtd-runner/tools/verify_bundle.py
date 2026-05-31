#!/usr/bin/env python3
"""verify_bundle.py — verify a Ryvion EM native bundle (em.bundle.v1).

The node-agent runs the equivalent of this before invoking a bundle: it
re-hashes every staged file against bundle.sha256, recomputes the aggregate
digest against manifest.sha256, and (when a public key is given) verifies the
Ed25519 signature over the canonical manifest bytes. Mirrors build_bundle.py so
the two stay in lockstep; also used by the unit tests.

Usage:
  python tools/verify_bundle.py <bundle_dir> [--public-key <ed25519 pubkey file>]

Exit 0 on success; non-zero with a diagnostic on any mismatch. An UNSIGNED
manifest is reported but not fatal unless --require-signature is given.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

# Reuse the exact hashing/canonicalisation from the builder so verify == build.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_bundle import (  # type: ignore  # noqa: E402
    aggregate_digest,
    canonical_manifest_bytes,
    sha256_file,
    walk_files,
)


def parse_bundle_sha256(bundle_dir: str) -> Dict[str, str]:
    path = os.path.join(bundle_dir, "bundle.sha256")
    out: Dict[str, str] = {}
    if not os.path.exists(path):
        return out
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            digest, rel = line.split("  ", 1)
            out[rel] = digest
    return out


def recompute_digests(bundle_dir: str) -> Dict[str, str]:
    digests: Dict[str, str] = {}
    excluded = {"bundle.sha256", "manifest.json"}
    for path in walk_files(bundle_dir):
        rel = os.path.relpath(path, bundle_dir)
        if rel in excluded:
            continue
        digests[rel] = sha256_file(path)
    return digests


def verify_signature(manifest: Dict[str, object], pubkey_path: Optional[str]) -> Tuple[bool, str]:
    sig = str(manifest.get("signature", ""))
    if sig.endswith("UNSIGNED") or sig.endswith("PENDING"):
        return (False, "manifest is unsigned")
    if not pubkey_path:
        return (True, "signature present (not checked: no public key given)")
    if not sig.startswith("ed25519:"):
        return (False, f"unknown signature scheme: {sig.split(':', 1)[0]}")
    sig_bytes = bytes.fromhex(sig.split(":", 1)[1])
    pub = _load_pubkey(pubkey_path)
    msg = canonical_manifest_bytes(manifest)
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (  # type: ignore
            Ed25519PublicKey,
        )

        Ed25519PublicKey.from_public_bytes(pub).verify(sig_bytes, msg)
        return (True, "ed25519 signature OK")
    except Exception:
        pass
    try:
        import nacl.signing  # type: ignore

        nacl.signing.VerifyKey(pub).verify(msg, sig_bytes)
        return (True, "ed25519 signature OK")
    except Exception as exc:  # noqa: BLE001
        return (False, f"signature verification failed: {exc}")


def _load_pubkey(path: str) -> bytes:
    with open(path, "rb") as fh:
        raw = fh.read().strip()
    if len(raw) == 64:
        try:
            return bytes.fromhex(raw.decode("ascii"))
        except Exception:
            pass
    return raw[:32]


def verify_bundle(bundle_dir: str, pubkey_path: Optional[str] = None,
                  require_signature: bool = False) -> List[str]:
    """Return a list of problem strings (empty == OK)."""
    problems: List[str] = []
    manifest_path = os.path.join(bundle_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        return [f"no manifest.json in {bundle_dir}"]
    with open(manifest_path, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)

    if manifest.get("schema") != "em.bundle.v1":
        problems.append(f"unexpected schema {manifest.get('schema')!r}")

    recomputed = recompute_digests(bundle_dir)
    listed = parse_bundle_sha256(bundle_dir)

    for rel, digest in recomputed.items():
        if rel not in listed:
            problems.append(f"file not in bundle.sha256: {rel}")
        elif listed[rel] != digest:
            problems.append(f"hash mismatch: {rel}")
    for rel in listed:
        if rel not in recomputed:
            problems.append(f"listed file missing on disk: {rel}")

    agg = aggregate_digest(recomputed)
    if agg != manifest.get("sha256"):
        problems.append("manifest.sha256 does not match recomputed aggregate digest")

    ok_sig, sig_msg = verify_signature(manifest, pubkey_path)
    if not ok_sig and require_signature:
        problems.append(f"signature: {sig_msg}")

    return problems


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Verify a Ryvion EM native bundle.")
    ap.add_argument("bundle_dir")
    ap.add_argument("--public-key", default="")
    ap.add_argument("--require-signature", action="store_true")
    args = ap.parse_args(argv)

    problems = verify_bundle(args.bundle_dir, args.public_key or None, args.require_signature)
    if problems:
        sys.stderr.write("BUNDLE INVALID:\n" + "\n".join(f"  - {p}" for p in problems) + "\n")
        return 1
    print(f"bundle OK: {args.bundle_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
