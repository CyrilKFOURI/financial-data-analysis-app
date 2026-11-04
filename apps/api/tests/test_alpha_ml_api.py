"""J21 endpoints offline: labels & weights, ML pipeline, forecasting baselines, alpha lab."""

import json
from typing import Any

from fastapi.testclient import TestClient


def _run(
    client: TestClient, kind: str, sid: str, items: list[tuple[str, dict[str, Any]]]
) -> dict[str, Any]:
    r = client.post(
        "/v1/analyses:run",
        json={
            "subject": {"type": kind, "id": sid},
            "context": {"range": "MAX"},
            "analyses": [{"id": i, "params": p} for i, p in items],
        },
    )
    out = {}
    for line in r.text.splitlines():
        x = json.loads(line)
        out[(x.get("result") or x.get("error"))["analysis_id"]] = x
    return out


def test_alpha_ml(client: TestClient) -> None:
    client.post("/v1/ingest/FAKEML")
    out = _run(
        client,
        "asset",
        "FAKEML",
        [
            ("TR-021", {"cusum": 1.5}),
            ("TR-022", {"cusum": 1.5}),
            ("TR-023", {"origins": 30, "horizon": 5}),
        ],
    )
    for aid in ("TR-021", "TR-022", "TR-023"):
        assert "result" in out[aid], (aid, out[aid])
    labels = out["TR-021"]["result"]["payload"]
    assert 0 < labels["mean_uniqueness"] <= 1
    ml = out["TR-022"]["result"]["payload"]
    assert "scikit-learn" in ml["models_pending"]
    assert {r["feature"] for r in ml["mda"]} == {
        "ret_5",
        "ret_20",
        "ret_60",
        "vol_20",
        "rsi_14",
        "dist_200",
    }
    fc = out["TR-023"]["result"]["payload"]
    assert [r["model"] for r in fc["rows"]] == [
        "model",
        "naive",
        "seasonal_naive",
        "mean",
    ]  # baselines always shown
    symbols = [f"FAKEA{i:02d}" for i in range(12)]
    uid = client.post(
        "/v1/universes", json={"name": "Alpha", "symbols": "\n".join(symbols)}
    ).json()["id"]
    client.post(f"/v1/universes/{uid}/ingest")
    alpha = _run(client, "universe", str(uid), [("TR-020", {"factor": "momentum_12_1"})])["TR-020"]
    assert "result" in alpha, alpha
    assert len(alpha["result"]["payload"]["quantiles"]) == 5
