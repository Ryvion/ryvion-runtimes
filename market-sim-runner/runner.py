#!/usr/bin/env python3
import hashlib
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class Event:
    time: datetime
    symbol: str
    close_cents: int


def run(work: Path) -> None:
    job = _read_json(work / "job.json")
    spec = _decode_spec(job)
    if spec.get("version") != "marketarena.run.v1" or spec.get("task") != "market_replay":
        raise ValueError("unsupported market replay spec")

    manifest_path = work / "input.bin" if spec.get("input_url") else work / str(spec["data_manifest_key"])
    manifest = _read_json(manifest_path)
    events = _events(manifest.get("events", []), spec)
    if len(events) < 2:
        raise ValueError("market replay requires at least two price events")

    output = work / "output"
    output.mkdir(parents=True, exist_ok=True)

    result = _simulate(spec, events, manifest.get("news", []), output)
    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")

    digest = _hash_directory(output)
    (work / "metrics.json").write_text(
        json.dumps({"output_name": "output", **_receipt_metrics(result)}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (work / "receipt.json").write_text(
        json.dumps({"output_hash": digest, "metadata": _receipt_metadata(result)}, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _decode_spec(job: dict) -> dict:
    if job.get("version") == "marketarena.run.v1":
        return job
    if isinstance(job.get("spec"), dict):
        return job["spec"]
    raw = job.get("spec_json")
    if isinstance(raw, str):
        return json.loads(raw)
    raise ValueError("job.json must include spec_json")


def _events(raw_events: list[dict], spec: dict) -> list[Event]:
    start = _parse_time(spec["window"]["start"])
    end = _parse_time(spec["window"]["end"])
    universe = {str(symbol).upper() for symbol in spec.get("universe", [])}
    events: list[Event] = []
    for item in raw_events:
        t = _parse_time(str(item["time"]))
        symbol = str(item["symbol"]).upper()
        if t < start or t > end:
            continue
        if universe and symbol not in universe:
            continue
        events.append(Event(t, symbol, int(item["close_cents"])))
    events.sort(key=lambda e: (e.time, e.symbol))
    return events


def _simulate(spec: dict, events: list[Event], news: list[dict], output: Path) -> dict:
    cash = int(spec.get("initial_cash_cents") or 100_000_00)
    initial_cash = cash
    position = 0.0
    avg_entry = 0
    last_price = events[0].close_cents
    peak = initial_cash
    max_drawdown_bps = 0
    orders = []
    trades = []
    equity_curve = []
    trace_lines = []
    news_index = _news_by_time(news)

    for i, event in enumerate(events):
        equity = int(cash + position * event.close_cents)
        peak = max(peak, equity)
        if peak > 0:
            drawdown = int(((peak - equity) * 10000) / peak)
            max_drawdown_bps = max(max_drawdown_bps, drawdown)
        equity_curve.append({"time": _format_time(event.time), "equity_cents": equity, "symbol": event.symbol})

        visible_news = news_index.get(_format_time(event.time), [])
        decision = "hold"
        quantity = 0.0
        if i > 0 and event.close_cents > last_price and position == 0:
            invest = int(cash * 0.20)
            quantity = invest / event.close_cents
            if quantity > 0:
                cash -= invest
                position += quantity
                avg_entry = event.close_cents
                decision = "buy"
        elif position > 0 and (event.close_cents < last_price or event.close_cents < avg_entry * 0.92):
            proceeds = int(position * event.close_cents)
            quantity = position
            cash += proceeds
            position = 0.0
            decision = "sell"

        if decision != "hold":
            order = {
                "time": _format_time(event.time),
                "symbol": event.symbol,
                "side": decision,
                "quantity": round(quantity, 8),
                "price_cents": event.close_cents,
                "fill_model": "close_price_baseline_v1",
            }
            orders.append(order)
            trades.append(order)
        trace_lines.append(
            {
                "time": _format_time(event.time),
                "symbol": event.symbol,
                "visible_news_count": len(visible_news),
                "decision": decision,
                "price_cents": event.close_cents,
                "future_events_visible": False,
            }
        )
        last_price = event.close_cents

    final_equity = int(cash + position * events[-1].close_cents)
    return_bps = int(((final_equity - initial_cash) * 10000) / initial_cash)
    turnover_bps = int((sum(int(t["quantity"] * t["price_cents"]) for t in trades) * 10000) / initial_cash)
    sharpe_x1000 = _simple_sharpe(equity_curve)
    score = return_bps - max_drawdown_bps + int(sharpe_x1000 / 10) - int(turnover_bps / 50)

    _write_jsonl(output / "orders.jsonl", orders)
    _write_jsonl(output / "trades.jsonl", trades)
    _write_json(output / "equity_curve.json", equity_curve)
    _write_jsonl(output / "agent_trace.jsonl", trace_lines)

    return {
        "run_id": spec.get("run_id", ""),
        "agent_id": spec.get("agent_id", ""),
        "arena_id": spec.get("arena_id", ""),
        "final_equity_cents": final_equity,
        "return_bps": return_bps,
        "max_drawdown_bps": max_drawdown_bps,
        "sharpe_x1000": sharpe_x1000,
        "turnover_bps": turnover_bps,
        "score": score,
        "orders_count": len(orders),
        "trades_count": len(trades),
    }


def _simple_sharpe(equity_curve: list[dict]) -> int:
    returns = []
    for prev, cur in zip(equity_curve, equity_curve[1:]):
        if prev["equity_cents"] > 0:
            returns.append((cur["equity_cents"] - prev["equity_cents"]) / prev["equity_cents"])
    if not returns:
        return 0
    mean = sum(returns) / len(returns)
    variance = sum((r - mean) ** 2 for r in returns) / len(returns)
    if variance == 0:
        return int(mean * 1000)
    return int((mean / (variance ** 0.5)) * 1000)


def _news_by_time(news: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for item in news:
        key = _format_time(_parse_time(str(item["time"])))
        out.setdefault(key, []).append(item)
    return out


def _receipt_metadata(result: dict) -> dict:
    return {
        **_receipt_metrics(result),
        "object_key": "output",
        "artifact_object_key": "output/summary.json",
        "trace_object_key": "output/agent_trace.jsonl",
    }


def _receipt_metrics(result: dict) -> dict:
    return {
        "final_equity_cents": result["final_equity_cents"],
        "return_bps": result["return_bps"],
        "max_drawdown_bps": result["max_drawdown_bps"],
        "sharpe_x1000": result["sharpe_x1000"],
        "turnover_bps": result["turnover_bps"],
        "score": result["score"],
        "orders_count": result["orders_count"],
        "trades_count": result["trades_count"],
    }


def _hash_directory(path: Path) -> str:
    digest = hashlib.sha256()
    for file in sorted(p for p in path.rglob("*") if p.is_file()):
        digest.update(str(file.relative_to(path)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(file.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")


def _parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def _format_time(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    run(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/work"))
