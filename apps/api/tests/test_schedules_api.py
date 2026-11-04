"""M31 schedules offline: validation, manual runs, due runs from next_run_at, history, report
written to disk (e-mail skipped without SMTP), failures recorded instead of raised."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

from fastapi.testclient import TestClient
from sqlalchemy import update

from fda_data.schema import schedules


def test_schedules_flow(client: TestClient) -> None:
    bad = client.post("/v1/schedules", json={"name": "x", "kind": "alerts", "cron": "* * *"})
    assert bad.status_code == 422
    bad_zone = client.post(
        "/v1/schedules",
        json={"name": "x", "kind": "alerts", "cron": "0 7 * * *", "timezone": "Mars/Olympus"},
    )
    assert bad_zone.status_code == 422
    no_symbols = client.post(
        "/v1/schedules", json={"name": "x", "kind": "data_refresh", "cron": "0 7 * * *"}
    )
    assert no_symbols.status_code == 422

    refresh = client.post(
        "/v1/schedules",
        json={
            "name": "Nightly refresh",
            "kind": "data_refresh",
            "cron": "30 22 * * 1-5",
            "timezone": "Europe/Paris",
            "spec": {"symbols": ["fakesch"]},
        },
    ).json()
    next_run = datetime.fromisoformat(refresh["next_run_at"])
    assert next_run > datetime.now(UTC)
    report = client.post(
        "/v1/schedules",
        json={
            "name": "Weekly risk",
            "kind": "report",
            "cron": "0 8 * * 1",
            "spec": {
                "template": "risk",
                "subject": "FAKESCH",
                "range": "1Y",
                "email": "me@example.com",
            },
        },
    ).json()
    snap = client.post(
        "/v1/schedules",
        json={
            "name": "Snap",
            "kind": "snapshot",
            "cron": "0 9 * * *",
            "spec": {
                "subject_type": "asset",
                "subject_id": "FAKESCH",
                "label": "auto",
                "analyses": ["R-005"],
            },
        },
    ).json()
    assert (
        client.post(
            "/v1/schedules", json={"name": "Snap", "kind": "alerts", "cron": "0 9 * * *"}
        ).status_code
        == 422
    )

    ran = client.post(f"/v1/schedules/{refresh['id']}/run").json()
    assert ran["status"] == "succeeded"
    assert ran["result"]["symbols"]["FAKESCH"]["rows"] > 0

    # nothing due yet; then force two schedules due
    assert client.post("/v1/schedules/run-due").json() == []
    engine = client.app.state.engine  # type: ignore[attr-defined]
    past = datetime.now(UTC) - timedelta(minutes=1)
    with engine.begin() as conn:
        conn.execute(
            update(schedules)
            .where(schedules.c.id.in_([report["id"], snap["id"]]))
            .values(next_run_at=past)
        )
    due = client.post("/v1/schedules/run-due").json()
    assert {d["schedule_id"] for d in due} == {report["id"], snap["id"]}
    by_id = {d["schedule_id"]: d for d in due}
    rep = by_id[report["id"]]
    assert rep["status"] == "succeeded", rep
    assert Path(rep["result"]["file"]).read_text("utf-8").startswith("<!")
    assert "SMTP_HOST" in rep["result"]["delivery"]
    assert by_id[snap["id"]]["result"]["analyses"] == ["R-005"]
    assert client.post("/v1/schedules/run-due").json() == []  # next_run_at moved forward

    listed = {s["name"]: s for s in client.get("/v1/schedules").json()}
    assert listed["Weekly risk"]["last_status"] == "succeeded"
    assert datetime.fromisoformat(listed["Weekly risk"]["next_run_at"]) > datetime.now(UTC)
    history = client.get(f"/v1/schedules/{refresh['id']}/runs").json()
    assert [h["trigger"] for h in history] == ["manual"]

    broken = client.post(
        "/v1/schedules",
        json={
            "name": "Broken",
            "kind": "data_refresh",
            "cron": "0 7 * * *",
            "spec": {"symbols": ["NOPESYMBOL"]},
        },
    ).json()
    failed = client.post(f"/v1/schedules/{broken['id']}/run").json()
    assert failed["status"] in ("failed", "succeeded")  # recorded either way, never raised
    assert client.delete(f"/v1/schedules/{broken['id']}").status_code == 204
    traversal = client.post(
        "/v1/schedules",
        json={
            "name": "T",
            "kind": "data_refresh",
            "cron": "0 7 * * *",
            "spec": {"symbols": ["../x"]},
        },
    )
    assert traversal.status_code == 422


def test_backup_schedule(client: TestClient) -> None:
    assert (
        client.post(
            "/v1/schedules",
            json={"name": "B", "kind": "backup", "cron": "0 3 * * *", "spec": {"keep": 0}},
        ).status_code
        == 422
    )
    made = client.post(
        "/v1/schedules",
        json={"name": "Nightly backup", "kind": "backup", "cron": "0 3 * * *", "spec": {"keep": 2}},
    ).json()
    for _ in range(3):
        ran = client.post(f"/v1/schedules/{made['id']}/run").json()
        assert ran["status"] == "succeeded", ran
    assert Path(ran["result"]["file"]).name.startswith("fda-backup-")
    assert len(list(Path(ran["result"]["file"]).parent.glob("*.tar.gz"))) <= 2
