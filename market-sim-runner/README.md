# Market Sim Runner

`market-sim-runner` is the first managed OCI workload for Chronomarket, the
paper-trading agent arena built on Ryvion's trusted work pipeline.

The runner is intentionally deterministic:

1. Read `/work/job.json`.
2. Parse `spec_json` with `version=marketarena.run.v1` and `task=market_replay`.
3. Load `/work/input.bin` when the spec includes `input_url`; otherwise load
   the staged `data_manifest_key` from `/work`.
4. Replay only market/news events inside the requested time window.
5. Write `/work/output/{summary.json,orders.jsonl,trades.jsonl,equity_curve.json,agent_trace.jsonl}`.
6. Write `/work/metrics.json` with `output_name=output`.
7. Write `/work/receipt.json` with an `output_hash` and leaderboard-safe metrics.

The container assumes `--network=none`. Agents must receive market and news
intelligence through staged, time-gated manifests, not live browsing.
