# Ryvion runtimes

This repository owns the workload-specific container images and runtime adapters that `ryvion-hub` can assign to `ryvion-node`.

Current scope:

- `embed-runner`
- `agent-runner`
- `image-gen-runner`
- `llm-runner`
- `transcode-runner`
- `vllm-runner`
- `runtimes/draft/small-model`
- `runtimes/verifier/contract-test`
- `runtimes/verifier/sglang`
- `whisper-runner`
- `spatial-stage-runner` (published as `spatial-recon-runner`, `pointcloud-align-runner`, `mesh-optimize-runner`, `scene-render-runner`)

Runner ownership rule:

- Runtime images live in this repository.
- `ryvion-node` is responsible for runtime execution, not for owning runner image definitions.
- `vllm-runner` remains the batched large-model container family used by `ryvion-hub` for the GPU-heavy OpenAI-compatible model tags.
- `runtimes/draft/small-model` is the stateless speculative draft generator. It emits `/work/draft_packets.json` with privacy-safe DraftPacket payloads and uses llama.cpp when a local GGUF is available, otherwise a deterministic fallback for CI and contract tests.
- `runtimes/verifier/contract-test` is the CPU contract-test adapter for the long-lived verifier session contract. It listens on `/work/verifier_session.sock` and supports `start_session`, `prefill`, `verify_tree`, `commit`, `rollback`, `abort`, and `close_session`. It is not a production GPU verifier.
- `runtimes/verifier/sglang` is the GPU target-verifier bridge. It runs offline, listens on `/work/verifier_session.sock`, keeps one SGLang engine hot across verification waves, requires local `.safetensors` artifacts, and writes `verifier_session_receipt.json` plus `probe_summary.json` with logprob-margin evidence.
- `agent-runner` is the default persistent-agent starter image used by `ryvion-hub` when a buyer deploys an agent without providing a custom image.

Runner contract:

1. `ryvion-hub` writes a structured `job.json` into `/work/job.json`.
2. `ryvion-node` mounts `/work` and runs the assigned container image.
3. The runner must:
   - read `/work/job.json`
   - fetch any `payload_url` it needs
   - write its artifact to `/work/output`
   - write `/work/receipt.json`
   - optionally write `/work/metrics.json`

Spatial pipeline stages now perform real geometry work instead of placeholder manifests:

- `spatial-recon-runner`: OpenCV two-view sparse reconstruction from uploaded image sets
- `pointcloud-align-runner`: coarse ICP-style point-cloud alignment
- `mesh-optimize-runner`: mesh generation/cleanup via convex hull + trimesh processing
- `scene-render-runner`: headless preview rendering to PNG for review workflows

Compatibility:

- The CI workflow publishes the exact image names `ryvion-hub` routes today:
	- `ghcr.io/ryvion/embed-runner:0.1.0`
	- `ghcr.io/ryvion/agent-runner:0.1.0`
	- `ghcr.io/ryvion/image-gen-runner:0.1.0`
	- `ghcr.io/ryvion/llm-runner:0.1.0`
	- `ghcr.io/ryvion/transcode-runner:0.1.0`
	- `ghcr.io/ryvion/vllm-runner:{latest,deepseek-r1-671b,deepseek-v3-671b,llama-3_3-70b,qwen-2_5-72b,mistral-large-2}`
	- `ghcr.io/ryvion/ryvion-draft-small-model:0.1.0` (`draft-runner-v8` compatibility alias)
	- `ghcr.io/ryvion/ryvion-verifier-contract-test:0.1.0` (`verifier-runner-v8-contract` compatibility alias)
	- `ghcr.io/ryvion/ryvion-verifier-sglang:0.1.0` (`sglang-verifier-runner-v8` compatibility alias)
	- `ghcr.io/ryvion/whisper-runner:0.1.0`
	- `ghcr.io/ryvion/spatial-recon-runner:0.1.0`
	- `ghcr.io/ryvion/pointcloud-align-runner:0.1.0`
	- `ghcr.io/ryvion/mesh-optimize-runner:0.1.0`
  - `ghcr.io/ryvion/scene-render-runner:0.1.0`
- Each image also receives `latest` and commit SHA tags for operator testing.

Path compatibility:

- Published image names now use mechanism-oriented names first: `ryvion-draft-small-model`, `ryvion-verifier-contract-test`, and `ryvion-verifier-sglang`.
- Legacy image names remain published as compatibility aliases while hub/node assignments migrate.
- Source paths moved under `runtimes/` so the code reads by mechanism instead of by product codename.
- Foresight Mesh remains the product/architecture name; the runtime code is organized as speculative inference.
