"""J22 endpoints offline: strategy builder on the model and on a synthetic stored chain, option
rule backtests with DSR/PBO."""

import json
from typing import Any

from fastapi.testclient import TestClient


def _run(
    client: TestClient, items: list[tuple[str, dict[str, Any]]], symbol: str
) -> dict[str, Any]:
    r = client.post(
        "/v1/analyses:run",
        json={
            "subject": {"type": "asset", "id": symbol},
            "context": {"range": "MAX"},
            "analyses": [{"id": i, "params": p} for i, p in items],
        },
    )
    out = {}
    for line in r.text.splitlines():
        x = json.loads(line)
        out[(x.get("result") or x.get("error"))["analysis_id"]] = x
    return out


def test_strategy_builder_and_backtests(client: TestClient) -> None:
    client.post("/v1/ingest/FAKEOS")
    model = _run(client, [("TR-030", {"preset": "iron_condor"})], "FAKEOS")["TR-030"]
    assert "result" in model, model
    p = model["result"]["payload"]
    assert p["source"] == "BSM model"
    assert p["max_loss"] < 0 < p["max_profit"]
    assert len(p["breakevens"]) == 2
    assert len(p["curve"]["spot"]) == len(p["curve"]["expiry"])

    client.post("/v1/options/FAKEOS/ingest")
    items = [
        ("TR-030", {"preset": "covered_call", "pricing": "natural"}),
        ("TR-031", {"rule": "short_strangle_16d", "days": 45, "managed": True, "years": 3}),
    ]
    out = _run(client, items, "FAKEOS")
    chain = out["TR-030"]
    assert "result" in chain, chain
    legs = chain["result"]["payload"]["legs"]
    assert chain["result"]["payload"]["source"] == "stored chain"
    assert legs[1]["expiry"]
    assert legs[1]["premium"] == legs[1]["bid"]  # short at the bid
    typed = _run(client, [("TR-030", {"legs": "+1 C @30; -1 P"})], "FAKEOS")["TR-030"]
    assert len(typed["result"]["payload"]["legs"]) == 2
    bad = _run(client, [("TR-030", {"legs": "buy everything"})], "FAKEOS")["TR-030"]
    assert "Cannot read leg" in bad["error"]["detail"]

    bt = out["TR-031"]
    assert "result" in bt, bt
    q = bt["result"]["payload"]
    assert q["trials"] == 12
    assert 0 <= q["pbo"] <= 1
    assert q["managed_exits"] > 0
    assert len(q["dates"]) == len(q["lines"]["strategy"])
