"""M31 §2 multi-user acceptance: sign-in, roles, and isolation through every endpoint and tool.

Alice creates one item of every user-owned kind, each named with a secret marker. Bob, another
analyst, then calls every route of the API (from the OpenAPI document) with Alice's ids. No
response may succeed on her items or contain the marker, and her items must be intact
afterwards.
"""

import json
import os
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, func, select, text
from sqlalchemy.engine import make_url

from fda_api import auth
from fda_api.main import app
from fda_api.security import CLIENT_HEADER
from fda_api.tools import all_tools
from fda_data.schema import metadata, portfolios, schedules

from .test_invest_api import ANSWERS

PG = os.getenv("FDA_TEST_PG_URL")
SCRATCH = "fda_multi_user_test"
ROOT = Path(__file__).resolve().parents[3]
SECRET = "ALICE-SECRET"
PASSWORD = "correct horse battery"
STRATEGY = {"symbol": "FAKE", "template": "ema_cross", "grid": {"fast": [5], "slow": [50]}}
HYPOTHESIS = "Trend persistence: medium-term momentum in broad equity indices (under-reaction)."


@pytest.fixture(params=["sqlite", "postgres"])
def accounts(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Iterator[TestClient]:
    """The same suite on SQLite (statement layer) and, when FDA_TEST_PG_URL is set, on a
    migrated Postgres database where the RLS policies also apply."""
    if request.param == "postgres":
        if not PG:
            pytest.skip("set FDA_TEST_PG_URL to run the multi-user suite on Postgres")
        database_url = _scratch_postgres(monkeypatch)
    else:
        database_url = f"sqlite:///{(tmp_path / 'api.db').as_posix()}"
        setup = create_engine(database_url)
        metadata.create_all(setup)
        setup.dispose()
    monkeypatch.setenv("REDIS_URL", "")
    monkeypatch.setenv("FDA_PROVIDERS", "fake")
    monkeypatch.setenv("DATABASE_URL", database_url)
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("FDA_AUTH_MODE", "accounts")
    monkeypatch.setattr(auth, "SCRYPT", auth.SCRYPT | {"n": 2**10})  # fast hashes in tests
    monkeypatch.setattr(auth, "throttle", auth.Throttle())
    with TestClient(app, headers={CLIENT_HEADER: "test"}) as client:
        engine = client.app.state.engine  # type: ignore[attr-defined]
        for email, role in (
            ("admin@x.org", "admin"),
            ("alice@x.org", "analyst"),
            ("bob@x.org", "analyst"),
            ("vera@x.org", "viewer"),
        ):
            body = auth.UserIn(email=email, display_name=email, role=role, password=PASSWORD)
            auth.create_user(engine, body)
        yield client


def _scratch_postgres(monkeypatch: pytest.MonkeyPatch) -> str:
    server = make_url(PG or "")
    admin = create_engine(server, isolation_level="AUTOCOMMIT")
    with admin.connect() as conn:
        conn.execute(text(f"DROP DATABASE IF EXISTS {SCRATCH} WITH (FORCE)"))
        conn.execute(text(f"CREATE DATABASE {SCRATCH}"))
    admin.dispose()
    url = server.set(database=SCRATCH).render_as_string(hide_password=False)
    monkeypatch.setenv("DATABASE_URL", url)
    config = Config(str(ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(ROOT / "infra" / "migrations"))
    command.upgrade(config, "head")
    return url


def login(client: TestClient, email: str) -> None:
    client.cookies.clear()
    response = client.post("/v1/auth/login", json={"email": email, "password": PASSWORD})
    assert response.status_code == 200, response.text


def _lines(response: Any) -> list[dict[str, Any]]:
    return [json.loads(line) for line in response.text.splitlines() if line.strip()]


def _alice_items(client: TestClient) -> dict[str, int]:
    login(client, "alice@x.org")
    assert client.post("/v1/ingest/FAKE").status_code == 202
    ids: dict[str, int] = {}

    def made(response: Any) -> int:
        assert response.status_code in (200, 201), response.text
        return int(response.json()["id"])

    ids["pid"] = made(
        client.post(
            "/v1/portfolios",
            json={
                "name": f"{SECRET}-P",
                "mode": "weights",
                "holdings": [{"symbol": "FAKE", "weight": 1}],
            },
        )
    )
    ids["cid"] = made(
        client.post(
            "/v1/comparisons",
            json={"name": f"{SECRET}-C", "members": f"FAKE, portfolio:{ids['pid']}"},
        )
    )
    rule = {
        "name": f"{SECRET}-A",
        "subject_type": "portfolio",
        "subject_id": str(ids["pid"]),
        "analysis_id": "T-007",
        "path": "value",
        "op": ">",
        "threshold": -1e9,
    }
    ids["rid"] = made(client.post("/v1/alerts", json=rule))
    assert client.post("/v1/alerts/evaluate").status_code == 200  # creates a notification
    made_snapshot = client.post(
        "/v1/snapshots",
        json={
            "subject_type": "portfolio",
            "subject_id": str(ids["pid"]),
            "label": f"{SECRET}-S",
            "analyses": ["T-007"],
        },
    )
    assert made_snapshot.status_code in (200, 201), made_snapshot.text
    ids["uid"] = made(
        client.post("/v1/universes", json={"name": f"{SECRET}-U", "symbols": "FAKE;Co;Tech"})
    )
    ids["sid"] = made(
        client.post(
            "/v1/screens",
            json={"name": f"{SECRET}-SC", "universe_id": ids["uid"], "filters": {"all": []}},
        )
    )
    ids["profile_id"] = made(
        client.post(
            "/v1/invest/profiles",
            json={
                "name": f"{SECRET}-IP",
                "answers": ANSWERS,
            },
        )
    )
    ids["strategy_id"] = made(
        client.post(
            "/v1/strategies",
            json={"name": f"{SECRET}-ST", "hypothesis": HYPOTHESIS, "spec": STRATEGY},
        )
    )
    ids["account_id"] = made(
        client.post("/v1/paper", json={"name": f"{SECRET}-PA", "starting_cash": 1000})
    )
    order = client.post(
        f"/v1/paper/{ids['account_id']}/orders",
        json={"symbol": "FAKE", "side": "buy", "quantity": 1, "note": SECRET},
    )
    assert order.status_code in (200, 201), order.text
    ids["order_id"] = int(order.json().get("id") or order.json()["order"]["id"])
    ids["schedule_id"] = made(
        client.post(
            "/v1/schedules", json={"name": f"{SECRET}-SCH", "kind": "alerts", "cron": "0 7 * * *"}
        )
    )
    return ids


def _fill(path: str, ids: dict[str, int]) -> str | None:
    values = ids | {
        "symbol": "FAKE",
        "code": "DGS10",
        "tx_id": 1,
        "action_id": ids.get("action_id", 1),
    }
    names = re.findall(r"{(\w+)", path)
    if any(n not in values for n in names):
        return None
    return re.sub(r"{(\w+)}", lambda m: str(values[m.group(1)]), path)


def test_isolation_through_every_endpoint(accounts: TestClient) -> None:
    client = accounts
    ids = _alice_items(client)
    with client.app.state.engine.connect() as conn:  # type: ignore[attr-defined]
        before = conn.execute(select(func.count()).select_from(portfolios)).scalar_one()

    login(client, "bob@x.org")
    openapi = client.get("/openapi.json").json()
    owned_params = {
        "pid",
        "cid",
        "rid",
        "uid",
        "sid",
        "profile_id",
        "strategy_id",
        "account_id",
        "order_id",
        "schedule_id",
        "item_id",
    }
    checked = 0
    for path, operations in openapi["paths"].items():
        if path.startswith(("/v1/users", "/v1/auth", "/v1/sharing")) or path == "/mcp":
            continue
        filled = _fill(path, ids)
        if filled is None:
            continue
        touches_alice = bool(set(re.findall(r"{(\w+)", path)) & owned_params)
        for method in operations:
            if method not in ("get", "post", "put", "patch", "delete"):
                continue
            if not touches_alice and method != "get":
                continue  # Bob's own writes are not the point here
            response = client.request(method.upper(), filled, json={})
            checked += 1
            assert SECRET not in response.text, (method, filled, response.text[:300])
            if touches_alice and method != "delete":  # deletes are idempotent (204, no-op)
                empty = response.status_code == 200 and response.text in ("[]", "{}")
                assert response.status_code >= 400 or empty, (method, filled, response.text[:200])
    assert checked > 60

    # analyses, reports and tools on Alice's subjects
    for subject in (
        ("portfolio", ids["pid"]),
        ("comparison", ids["cid"]),
        ("strategy", ids["strategy_id"]),
        ("profile", ids["profile_id"]),
        ("paper", ids["account_id"]),
        ("universe", ids["uid"]),
    ):
        r = client.post(
            "/v1/analyses:run",
            json={
                "subject": {"type": subject[0], "id": str(subject[1])},
                "analyses": [{"id": "R-005"}],
            },
        )
        assert all("error" in line for line in _lines(r)), r.text[:300]
        assert SECRET not in r.text
    report = client.get("/v1/reports/risk", params={"subject": f"portfolio:{ids['pid']}"})
    assert SECRET not in report.text
    listed = client.get(
        "/v1/snapshots", params={"subject_type": "portfolio", "subject_id": str(ids["pid"])}
    )
    assert SECRET not in listed.text
    portfolio_tool = next(t.name for t in all_tools() if t.analysis_id == "R-005")
    called = client.post(
        f"/v1/tools/{portfolio_tool}:call",
        json={"arguments": {"subject_type": "portfolio", "subject_id": str(ids["pid"])}},
    )
    assert called.json()["is_error"] is True, called.text[:300]
    assert SECRET not in called.text

    # Alice's items are untouched
    login(client, "alice@x.org")
    with client.app.state.engine.connect() as conn:  # type: ignore[attr-defined]
        assert conn.execute(select(func.count()).select_from(portfolios)).scalar_one() == before
    assert client.get(f"/v1/portfolios/{ids['pid']}").status_code == 200
    own = client.post(
        f"/v1/tools/{portfolio_tool}:call",
        json={"arguments": {"subject_type": "portfolio", "subject_id": str(ids["pid"])}},
    )
    assert own.json()["is_error"] is False, own.text[:300]  # the same call works for its owner
    assert client.get(f"/v1/paper/{ids['account_id']}").json()["orders"]
    assert any(SECRET in n["message"] for n in client.get("/v1/notifications").json())
    for listing, suffix in (
        ("/v1/comparisons", "C"),
        ("/v1/alerts", "A"),
        ("/v1/universes", "U"),
        ("/v1/invest/profiles", "IP"),
        ("/v1/strategies", "ST"),
        ("/v1/paper", "PA"),
        ("/v1/schedules", "SCH"),
    ):
        assert f"{SECRET}-{suffix}" in client.get(listing).text, listing


def test_sharing_roles_and_sessions(accounts: TestClient) -> None:
    client = accounts
    client.cookies.clear()
    assert client.get("/v1/portfolios").status_code == 401
    assert client.get("/v1/health").status_code == 200
    assert client.get("/v1/auth/me").json() == {"mode": "accounts", "user": None}

    ids = _alice_items(client)
    shared = client.put(f"/v1/sharing/portfolios/{ids['pid']}", json={"shared": True})
    assert shared.status_code == 200, shared.text

    login(client, "bob@x.org")  # shared: readable, analysable, not writable
    assert f"{SECRET}-P" in client.get("/v1/portfolios").text
    run = client.post(
        "/v1/analyses:run",
        json={
            "subject": {"type": "portfolio", "id": str(ids["pid"])},
            "analyses": [{"id": "R-005"}],
        },
    )
    assert "result" in _lines(run)[0], run.text[:300]
    client.delete(f"/v1/portfolios/{ids['pid']}")  # a no-op on someone else's row
    assert f"{SECRET}-P" in client.get("/v1/portfolios").text
    assert (
        client.put(f"/v1/sharing/portfolios/{ids['pid']}", json={"shared": False}).status_code
        == 404
    )
    assert client.get("/v1/users").status_code == 403

    login(client, "vera@x.org")  # viewer: reads shared items, cannot write anything
    assert f"{SECRET}-P" in client.get("/v1/portfolios").text
    assert f"{SECRET}-ST" not in client.get("/v1/strategies").text
    assert client.post("/v1/portfolios", json={"name": "V", "mode": "weights"}).status_code == 403
    assert client.post("/v1/schedules/run-due").status_code == 403
    for tool, arguments in (
        (
            "place_paper_order",
            {"account_id": ids["account_id"], "symbol": "FAKE", "side": "buy", "quantity": 1},
        ),
        ("request_ingestion", {"symbol": "FAKE"}),
    ):
        refused = client.post(f"/v1/tools/{tool}:call", json={"arguments": arguments}).json()
        assert refused["is_error"] is True, refused
        assert "viewers" in refused["content"]["error"]

    login(client, "admin@x.org")
    users = {u["email"]: u for u in client.get("/v1/users").json()}
    assert users["admin@x.org"]["id"] == 1  # the first admin adopted the local user
    assert "password_hash" not in users["admin@x.org"]
    last_admin = client.patch("/v1/users/1", json={"role": "analyst"})
    assert last_admin.status_code == 422
    assert (
        client.post(
            "/v1/users",
            json={
                "email": "short@x.org",
                "display_name": "S",
                "role": "viewer",
                "password": "short",
            },
        ).status_code
        == 422
    )
    bob = users["bob@x.org"]["id"]
    assert client.patch(f"/v1/users/{bob}", json={"is_active": False}).json()["is_active"] is False
    client.cookies.clear()
    assert (
        client.post("/v1/auth/login", json={"email": "bob@x.org", "password": PASSWORD}).status_code
        == 401
    )

    # password change signs out other sessions; wrong passwords are throttled
    login(client, "alice@x.org")
    other = TestClient(app, headers={CLIENT_HEADER: "test"})
    other.post("/v1/auth/login", json={"email": "alice@x.org", "password": PASSWORD})
    assert other.get("/v1/portfolios").status_code == 200
    changed = client.post(
        "/v1/auth/password", json={"current": PASSWORD, "new": "another long passphrase"}
    )
    assert changed.status_code == 204
    assert other.get("/v1/portfolios").status_code == 401
    assert client.get("/v1/portfolios").status_code == 200
    client.cookies.clear()
    for _ in range(5):
        client.post("/v1/auth/login", json={"email": "vera@x.org", "password": "wrong password!"})
    assert (
        client.post(
            "/v1/auth/login", json={"email": "vera@x.org", "password": PASSWORD}
        ).status_code
        == 429
    )


def test_schedules_run_as_their_owner(accounts: TestClient) -> None:
    client = accounts
    ids = _alice_items(client)
    login(client, "bob@x.org")
    assert (
        client.post(
            "/v1/schedules", json={"name": "Bob backup", "kind": "backup", "cron": "0 3 * * *"}
        ).status_code
        == 403
    )
    bob_rule = {
        "name": "bob rule",
        "subject_type": "asset",
        "subject_id": "FAKE",
        "analysis_id": "T-007",
        "path": "value",
        "op": ">",
        "threshold": -1e9,
    }
    assert client.post("/v1/alerts", json=bob_rule).status_code == 201
    engine = client.app.state.engine  # type: ignore[attr-defined]
    from fda_api.schedules import run_schedule

    # a backup schedule stops working once its owner is no longer an admin
    login(client, "admin@x.org")
    second = client.post(
        "/v1/users",
        json={"email": "ops@x.org", "display_name": "Ops", "role": "admin", "password": PASSWORD},
    ).json()
    login(client, "ops@x.org")
    backup_id = client.post(
        "/v1/schedules", json={"name": "Ops backup", "kind": "backup", "cron": "0 3 * * *"}
    ).json()["id"]
    login(client, "admin@x.org")
    client.patch(f"/v1/users/{second['id']}", json={"role": "analyst"})
    demoted = run_schedule(client.app.state, backup_id, "cron")  # type: ignore[attr-defined]
    assert demoted["status"] == "failed"
    assert "admins only" in demoted["result"]["error"]

    result = run_schedule(client.app.state, ids["schedule_id"], "manual")  # type: ignore[attr-defined]
    assert result["status"] == "succeeded", result
    assert result["result"]["evaluated"] == 1  # Alice's rule only, not Bob's
    with engine.connect() as conn:
        owner = conn.execute(
            select(schedules.c.owner_id).where(schedules.c.id == ids["schedule_id"])
        ).scalar()
    assert owner is not None


def test_agent_runs_belong_to_their_user(
    accounts: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fda_api import agent

    from .test_agent import FakeProvider

    client = accounts
    monkeypatch.setattr(agent, "get_provider", lambda name=None: FakeProvider())
    login(client, "alice@x.org")
    client.post("/v1/ingest/FAKE")
    asked = client.post("/v1/agent/ask", json={"question": f"{SECRET} volatility?"})
    run_id = json.loads(asked.text.splitlines()[-1])["run_id"]
    mine = client.get("/v1/agent/runs").json()
    assert [r["id"] for r in mine] == [run_id]
    assert mine[0]["owner_id"] is not None
    login(client, "bob@x.org")
    others = client.get("/v1/agent/runs")
    assert others.json() == []
    assert SECRET not in others.text
    login(client, "vera@x.org")  # viewers do not spend LLM credits
    refused = json.loads(client.post("/v1/agent/ask", json={"question": "hi"}).text.splitlines()[0])
    assert refused["type"] == "error"
    assert "Viewers" in refused["detail"]
