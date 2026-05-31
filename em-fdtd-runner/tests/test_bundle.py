"""test_bundle.py — unit tests for the native-bundle build + verify tooling.

Exercises tools/build_bundle.py and tools/verify_bundle.py end to end in
RUNNER-ONLY mode (--skip-engine --skip-python), which is the only mode that
works without a target OS+GPU. Asserts the em.bundle.v1 layout, manifest fields,
bundle.sha256 coverage, the aggregate-digest match, and that tampering is
detected. A REAL bundle (with engine/ + python/ + signature) MUST be produced on
a target machine; that path is // TODO(em-verify) here. Run:

    python -m unittest tests.test_bundle -v
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_RUNNER = os.path.dirname(_HERE)
_TOOLS = os.path.join(_RUNNER, "tools")
for p in (_RUNNER, _TOOLS):
    if p not in sys.path:
        sys.path.insert(0, p)

import build_bundle  # noqa: E402
import verify_bundle  # noqa: E402
from templates import registry  # noqa: E402


class TestBundleRoundTrip(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = self.tmp.name
        rc = build_bundle.main([
            "--engine", "gprmax",
            "--engine-version", "3.1.7+test",
            "--os", "linux", "--arch", "x86_64",
            "--gpu", "cuda", "--gpu", "rocm",
            "--out", self.out,
            "--skip-engine", "--skip-python", "--allow-unsigned",
        ])
        self.assertEqual(rc, 0)
        self.bundle = os.path.join(self.out, "gprmax-3.1.7+test")

    def tearDown(self):
        self.tmp.cleanup()

    def test_layout(self):
        self.assertTrue(os.path.isdir(os.path.join(self.bundle, "runner")))
        self.assertTrue(os.path.isfile(os.path.join(self.bundle, "runner", "run.py")))
        self.assertTrue(os.path.isdir(os.path.join(self.bundle, "runner", "templates")))
        self.assertTrue(os.path.isfile(os.path.join(self.bundle, "manifest.json")))
        self.assertTrue(os.path.isfile(os.path.join(self.bundle, "bundle.sha256")))

    def test_runner_files_frozen(self):
        runner = os.path.join(self.bundle, "runner")
        for name in ("engine.py", "engine_gprmax.py", "engine_analytic.py",
                     "gprmax_io.py", "results.py", "budget.py", "geometry.py",
                     "schema.json"):
            self.assertTrue(os.path.isfile(os.path.join(runner, name)),
                            f"missing frozen runner file {name}")

    def test_manifest_fields(self):
        with open(os.path.join(self.bundle, "manifest.json")) as fh:
            m = json.load(fh)
        self.assertEqual(m["schema"], "em.bundle.v1")
        self.assertEqual(m["engine"], "gprmax")
        self.assertEqual(m["engine_version"], "3.1.7+test")
        self.assertEqual(m["os"], "linux")
        self.assertEqual(m["arch"], "x86_64")
        self.assertEqual(sorted(m["gpu"]), ["cuda", "rocm"])
        self.assertEqual(m["entrypoint"], ["python/bin/python3", "runner/run.py"])
        self.assertEqual(sorted(m["templates"]), sorted(registry.names()))
        self.assertTrue(m["runner_only"])
        self.assertEqual(len(m["sha256"]), 64)
        self.assertGreater(m["size_bytes"], 0)
        self.assertTrue(m["signature"].endswith("UNSIGNED"))

    def test_entrypoint_windows(self):
        tmp2 = tempfile.TemporaryDirectory()
        try:
            build_bundle.main([
                "--engine", "openems", "--engine-version", "0.0.36+test",
                "--os", "windows", "--arch", "x86_64", "--gpu", "none",
                "--out", tmp2.name, "--skip-engine", "--skip-python",
                "--allow-unsigned",
            ])
            with open(os.path.join(tmp2.name, "openems-0.0.36+test", "manifest.json")) as fh:
                m = json.load(fh)
            self.assertEqual(m["entrypoint"], ["python/python.exe", "runner/run.py"])
        finally:
            tmp2.cleanup()

    def test_verify_passes(self):
        problems = verify_bundle.verify_bundle(self.bundle)
        self.assertEqual(problems, [], f"unexpected problems: {problems}")

    def test_verify_detects_tampering(self):
        # mutate a frozen runner file -> hash mismatch must be caught.
        target = os.path.join(self.bundle, "runner", "run.py")
        with open(target, "a", encoding="utf-8") as fh:
            fh.write("\n# tampered\n")
        problems = verify_bundle.verify_bundle(self.bundle)
        self.assertTrue(any("hash mismatch" in p for p in problems),
                        f"tamper not detected: {problems}")

    def test_verify_detects_added_file(self):
        with open(os.path.join(self.bundle, "runner", "evil.py"), "w") as fh:
            fh.write("print('x')\n")
        problems = verify_bundle.verify_bundle(self.bundle)
        self.assertTrue(any("not in bundle.sha256" in p for p in problems),
                        f"added file not detected: {problems}")

    def test_aggregate_digest_matches_manifest(self):
        recomputed = verify_bundle.recompute_digests(self.bundle)
        agg = build_bundle.aggregate_digest(recomputed)
        with open(os.path.join(self.bundle, "manifest.json")) as fh:
            m = json.load(fh)
        self.assertEqual(agg, m["sha256"])

    def test_unsigned_without_allow_flag_fails(self):
        tmp2 = tempfile.TemporaryDirectory()
        try:
            rc = build_bundle.main([
                "--engine", "gprmax", "--engine-version", "x",
                "--os", "linux", "--arch", "x86_64", "--gpu", "none",
                "--out", tmp2.name, "--skip-engine", "--skip-python",
            ])
            self.assertEqual(rc, 2)  # refuses to ship UNSIGNED without override.
        finally:
            tmp2.cleanup()


class TestSignedBundle(unittest.TestCase):
    """Sign + verify with a real ed25519 key when cryptography/pynacl is present."""

    def _have_ed25519(self):
        try:
            from cryptography.hazmat.primitives.asymmetric.ed25519 import (  # noqa: F401
                Ed25519PrivateKey,
            )
            return True
        except Exception:
            try:
                import nacl.signing  # noqa: F401
                return True
            except Exception:
                return False

    def test_sign_and_verify(self):
        if not self._have_ed25519():
            self.skipTest("no ed25519 backend (cryptography/pynacl) installed")
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        from cryptography.hazmat.primitives import serialization

        tmp = tempfile.TemporaryDirectory()
        try:
            sk = Ed25519PrivateKey.generate()
            seed = sk.private_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PrivateFormat.Raw,
                encryption_algorithm=serialization.NoEncryption(),
            )
            pub = sk.public_key().public_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PublicFormat.Raw,
            )
            key_path = os.path.join(tmp.name, "k.key")
            pub_path = os.path.join(tmp.name, "k.pub")
            with open(key_path, "wb") as fh:
                fh.write(seed.hex().encode())
            with open(pub_path, "wb") as fh:
                fh.write(pub.hex().encode())

            rc = build_bundle.main([
                "--engine", "gprmax", "--engine-version", "signed",
                "--os", "linux", "--arch", "x86_64", "--gpu", "cuda",
                "--out", tmp.name, "--skip-engine", "--skip-python",
                "--signing-key", key_path,
            ])
            self.assertEqual(rc, 0)
            bundle = os.path.join(tmp.name, "gprmax-signed")
            problems = verify_bundle.verify_bundle(bundle, pub_path, require_signature=True)
            self.assertEqual(problems, [], f"signed verify failed: {problems}")
        finally:
            tmp.cleanup()


if __name__ == "__main__":
    unittest.main()
