#!/usr/bin/env python3
r"""sign_descriptor.py — sign the EM bundle DESCRIPTOR that the node verifies.

This produces the node-manifest signature that travels in the hub's EM job spec
as runtime.signature. It is DISTINCT from the bundle-internal manifest signed by
build_bundle.py. The node (ryvion-node internal/runner/em_bundle.go,
emManifestSigningBytes) verifies an Ed25519 signature over a portable,
newline-joined message that binds the bundle IDENTITY:

    ryvion-em-bundle-v1\n<engine>\n<engine_version>\n<bundle_url>\n<bundle_sha256>\n<entrypoint>

os/arch are intentionally excluded so one signature is portable across nodes;
bundle_sha256 still pins the exact artifact. This MUST stay byte-for-byte in sync
with emManifestSigningBytes in ryvion-node.

Usage:
  python tools/sign_descriptor.py --signing-key em_bundle_ed25519.key \
      --engine gprmax --engine-version 3.1.7+cuda12 \
      --bundle-url https://huggingface.co/.../gprmax-3.1.7.tar.gz \
      --bundle-sha256 <hex> --entrypoint runner/run.py

Prints the hex signature. Put it (and the same bundle_url/bundle_sha256/entrypoint)
into the hub's EM job spec runtime section. Run with --selftest to check the
sign+verify round-trip and the exact canonical bytes.
"""
from __future__ import annotations

import argparse
import sys

DOMAIN = "ryvion-em-bundle-v1"


def signing_message(engine: str, engine_version: str, bundle_url: str,
                    bundle_sha256: str, entrypoint: str) -> bytes:
    return "\n".join([DOMAIN, engine, engine_version, bundle_url,
                      bundle_sha256, entrypoint]).encode("utf-8")


def _load_seed(path: str) -> bytes:
    with open(path, "rb") as fh:
        raw = fh.read().strip()
    if len(raw) == 64:
        try:
            return bytes.fromhex(raw.decode("ascii"))
        except Exception:
            pass
    if len(raw) >= 32:
        return raw[:32]
    raise ValueError("signing key must be a 32-byte ed25519 seed (raw or hex)")


def sign(seed: bytes, msg: bytes) -> bytes:
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (  # type: ignore
            Ed25519PrivateKey,
        )

        return Ed25519PrivateKey.from_private_bytes(seed).sign(msg)
    except Exception:
        pass
    import nacl.signing  # type: ignore

    return nacl.signing.SigningKey(seed).sign(msg).signature


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Sign an EM bundle descriptor for the node to verify.")
    ap.add_argument("--signing-key", default="")
    ap.add_argument("--engine", default="")
    ap.add_argument("--engine-version", default="")
    ap.add_argument("--bundle-url", default="")
    ap.add_argument("--bundle-sha256", default="")
    ap.add_argument("--entrypoint", default="")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)

    if args.selftest:
        return _selftest()

    if not args.signing_key:
        sys.stderr.write("ERROR: --signing-key is required\n")
        return 2
    msg = signing_message(args.engine, args.engine_version, args.bundle_url,
                          args.bundle_sha256, args.entrypoint)
    print(sign(_load_seed(args.signing_key), msg).hex())
    return 0


def _selftest() -> int:
    try:
        from cryptography.hazmat.primitives import serialization  # type: ignore
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (  # type: ignore
            Ed25519PrivateKey,
            Ed25519PublicKey,
        )
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(f"selftest needs `cryptography`: {exc}\n")
        return 2

    # Canonical bytes must equal what ryvion-node emManifestSigningBytes builds.
    msg = signing_message("gprmax", "1.0", "https://h/b.tgz", "deadbeef", "run.py")
    expected = b"ryvion-em-bundle-v1\ngprmax\n1.0\nhttps://h/b.tgz\ndeadbeef\nrun.py"
    assert msg == expected, f"canonical bytes drift: {msg!r}"

    sk = Ed25519PrivateKey.generate()
    seed = sk.private_bytes(serialization.Encoding.Raw, serialization.PrivateFormat.Raw,
                            serialization.NoEncryption())
    pub = sk.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    sig = sign(seed, msg)
    Ed25519PublicKey.from_public_bytes(pub).verify(sig, msg)  # raises on mismatch
    print("selftest OK: canonical bytes match and sign+verify round-trips")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
