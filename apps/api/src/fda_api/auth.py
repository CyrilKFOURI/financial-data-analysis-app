"""M31 §2 authentication, sessions and roles.

`FDA_AUTH_MODE`:
- `single` (default): no sign-in; every request acts as the built-in local admin (user 1), as
  before multi-user.
- `accounts`: e-mail + password sign-in. Passwords are hashed with scrypt (standard library,
  RFC 7914 parameters N=2^15, r=8, p=1). A random session token goes in an HttpOnly, SameSite=Lax
  cookie, and only its SHA-256 is stored. Sessions slide over 14 days. Failed sign-ins are
  throttled per e-mail and per client address.

Roles: `viewer` reads (shared items and reference data; analyses run), `analyst` owns
portfolios, strategies, schedules…, `admin` also manages users and system rows. The middleware
sets the request's actor (`fda_data.actor`), which scopes every query to the user's rows.
"""

import base64
import hashlib
import hmac
import os
import secrets
import time
from collections import defaultdict, deque
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy import Engine, delete, func, insert, select, text, update

from fda_data.actor import LOCAL_USER_ID, Actor, acting_as
from fda_data.schema import sessions, users

router = APIRouter(prefix="/v1", tags=["auth"])

COOKIE = "fda_session"
SESSION_TTL = timedelta(days=14)
REFRESH_AFTER = timedelta(hours=12)
SCRYPT = {"n": 2**15, "r": 8, "p": 1, "maxmem": 64 * 1024 * 1024, "dklen": 32}
MIN_PASSWORD = 12
PUBLIC_PATHS = frozenset({"/v1/health", "/v1/auth/login", "/v1/auth/me", "/v1/auth/logout"})
MCP_PATH = "/mcp"
# POST endpoints that only read: viewers may call them, and shared rows stay visible in them
READ_POSTS = ("/v1/analyses:run", "/v1/tools/", "/v1/workflows/", "/v1/agent/")
VIEWER_WRITES = frozenset({"/v1/auth/password"})
Role = Literal["viewer", "analyst", "admin"]
EMAIL = r"^[^@\s]+@[^@\s]+\.[^@\s]+$"


def mode() -> str:
    return "accounts" if os.getenv("FDA_AUTH_MODE", "single") == "accounts" else "single"


def ensure_local_user(engine: Engine) -> None:
    """User 1 owns single-user data; databases built without migrations (tests) get it here."""
    with engine.begin() as conn:
        if conn.execute(select(users.c.id).limit(1)).first() is not None:
            return
        conn.execute(
            insert(users).values(
                id=LOCAL_USER_ID, email="local", display_name="Local user", role="admin"
            )
        )
        if conn.dialect.name == "postgresql":
            conn.execute(text("SELECT setval(pg_get_serial_sequence('users', 'id'), 1)"))


# ── passwords ────────────────────────────────────────────────────────────────


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, **SCRYPT)  # type: ignore[arg-type]
    encoded = [base64.b64encode(v).decode() for v in (salt, digest)]
    return "$".join(["scrypt", str(SCRYPT["n"]), str(SCRYPT["r"]), str(SCRYPT["p"]), *encoded])


def verify_password(password: str, stored: str | None) -> bool:
    if not stored:
        hashlib.scrypt(b"timing", salt=b"0" * 16, **SCRYPT)  # type: ignore[arg-type]
        return False
    try:
        scheme, n, r, p, salt, digest = stored.split("$")
        if scheme != "scrypt":
            return False
        actual = hashlib.scrypt(
            password.encode(),
            salt=base64.b64decode(salt),
            n=int(n),
            r=int(r),
            p=int(p),
            maxmem=SCRYPT["maxmem"],
            dklen=len(base64.b64decode(digest)),
        )
    except ValueError:
        return False
    return hmac.compare_digest(actual, base64.b64decode(digest))


def check_password_rules(password: str) -> None:
    if len(password) < MIN_PASSWORD:
        raise HTTPException(422, f"Passwords need at least {MIN_PASSWORD} characters")


# ── throttling ───────────────────────────────────────────────────────────────


class Throttle:
    """At most `limit` failures per key in `window` seconds (in-process, per API process)."""

    def __init__(self, limit: int = 5, window: float = 900) -> None:
        self.limit, self.window = limit, window
        self.failures: dict[str, deque[float]] = defaultdict(deque)

    def blocked(self, *keys: str) -> bool:
        now = time.monotonic()
        for key in keys:
            queue = self.failures[key]
            while queue and now - queue[0] > self.window:
                queue.popleft()
            if len(queue) >= self.limit:
                return True
        return False

    def fail(self, *keys: str) -> None:
        for key in keys:
            self.failures[key].append(time.monotonic())

    def clear(self, *keys: str) -> None:
        for key in keys:
            self.failures.pop(key, None)


throttle = Throttle()


# ── sessions ─────────────────────────────────────────────────────────────────


def _token_id(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def _aware(t: datetime) -> datetime:
    return t if t.tzinfo else t.replace(tzinfo=UTC)


def open_session(engine: Engine, user_id: int, user_agent: str | None) -> str:
    token = secrets.token_urlsafe(32)
    now = datetime.now(UTC)
    with engine.begin() as conn:
        conn.execute(
            insert(sessions).values(
                id=_token_id(token),
                user_id=user_id,
                created_at=now,
                expires_at=now + SESSION_TTL,
                last_seen_at=now,
                user_agent=(user_agent or "")[:200],
            )
        )
        conn.execute(update(users).where(users.c.id == user_id).values(last_login_at=now))
    return token


def session_user(engine: Engine, token: str) -> dict[str, Any] | None:
    sid = _token_id(token)
    query = (
        select(users, sessions.c.expires_at, sessions.c.last_seen_at)
        .join_from(sessions, users, sessions.c.user_id == users.c.id)
        .where(sessions.c.id == sid)
    )
    with engine.connect() as conn:
        row = conn.execute(query).mappings().one_or_none()
    if row is None or not row["is_active"]:
        return None
    now = datetime.now(UTC)
    if _aware(row["expires_at"]) <= now:
        with engine.begin() as conn:
            conn.execute(delete(sessions).where(sessions.c.id == sid))
        return None
    if now - _aware(row["last_seen_at"]) > REFRESH_AFTER:
        with engine.begin() as conn:
            conn.execute(
                update(sessions)
                .where(sessions.c.id == sid)
                .values(last_seen_at=now, expires_at=now + SESSION_TTL)
            )
    return dict(row)


def _secure(request: Request) -> bool:
    return request.url.scheme == "https" or os.getenv("FDA_COOKIE_SECURE") == "1"


def _set_cookie(response: Response, request: Request, token: str) -> None:
    response.set_cookie(
        COOKIE,
        token,
        max_age=int(SESSION_TTL.total_seconds()),
        httponly=True,
        samesite="lax",
        secure=_secure(request),
        path="/",
    )


# ── middleware ───────────────────────────────────────────────────────────────


def _problem(status: int, title: str, detail: str) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        media_type="application/problem+json",
        content={
            "type": f"https://fda.local/problems/{title.lower().replace(' ', '-')}",
            "title": title,
            "status": status,
            "detail": detail,
        },
    )


def _access(request: Request) -> Literal["read", "write"]:
    if request.method in ("GET", "HEAD", "OPTIONS"):
        return "read"
    return "read" if request.url.path.startswith(READ_POSTS) else "write"


def _mcp_user(engine: Engine) -> dict[str, Any] | None:
    """MCP clients act as FDA_MCP_USER (an e-mail), else as the local / first admin user."""
    email = os.getenv("FDA_MCP_USER", "").strip().lower()
    query = select(users).where(users.c.is_active.is_(True))
    query = (
        query.where(users.c.email == email)
        if email
        else query.where(users.c.role == "admin").order_by(users.c.id)
    )
    with engine.connect() as conn:
        row = conn.execute(query.limit(1)).mappings().first()
    return dict(row) if row else None


async def authenticate(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    path = request.url.path
    if not path.startswith(("/v1/", MCP_PATH)):
        return await call_next(request)
    engine: Engine = request.app.state.engine
    access = _access(request)
    if mode() == "single":
        user: dict[str, Any] | None = {"id": LOCAL_USER_ID, "role": "admin"}
    elif path == MCP_PATH:
        user = _mcp_user(engine)  # the MCP endpoint itself checks its bearer token
        access = "read"  # agent writes are proposals approved in the web app
        if user is None:
            return _problem(503, "No MCP user", "Set FDA_MCP_USER to an active user's e-mail.")
    else:
        token = request.cookies.get(COOKIE)
        user = session_user(engine, token) if token else None
        if user is None and path not in PUBLIC_PATHS:
            return _problem(401, "Sign in required", "Sign in to use the workbench.")
    if user is not None:
        role: Role = user["role"]
        if (
            role == "viewer"
            and access == "write"
            and path not in PUBLIC_PATHS
            and path not in VIEWER_WRITES
        ):
            return _problem(403, "Read only", "Viewers cannot change data.")
        if path.startswith("/v1/users") and role != "admin":
            return _problem(403, "Admins only", "Only admins manage users.")
        request.state.user = user
        with acting_as(Actor(int(user["id"]), role, access)):
            return await call_next(request)
    request.state.user = None
    return await call_next(request)


# ── routes ───────────────────────────────────────────────────────────────────


class LoginIn(BaseModel):
    email: str = Field(pattern=EMAIL, max_length=254)
    password: str = Field(min_length=1, max_length=256)


class Me(BaseModel):
    mode: Literal["single", "accounts"]
    user: dict[str, Any] | None


def _public(row: dict[str, Any]) -> dict[str, Any]:
    keep = ("id", "email", "display_name", "role", "is_active", "last_login_at", "created_at")
    return {k: row.get(k) for k in keep if k in row}


@router.get("/auth/me")
def me(request: Request) -> Me:
    user = getattr(request.state, "user", None)
    if mode() == "single":
        with request.app.state.engine.connect() as conn:
            row = conn.execute(select(users).where(users.c.id == LOCAL_USER_ID)).mappings().first()
        return Me(mode="single", user=_public(dict(row)) if row else None)
    return Me(mode="accounts", user=_public(user) if user else None)


@router.post("/auth/login")
def login(request: Request, body: LoginIn, response: Response) -> Me:
    if mode() != "accounts":
        raise HTTPException(409, "Sign-in is off (FDA_AUTH_MODE=single)")
    email = body.email.lower()
    client = request.client.host if request.client else "?"
    keys = (f"email:{email}", f"ip:{client}")
    if throttle.blocked(*keys):
        raise HTTPException(429, "Too many failed sign-ins; try again in 15 minutes")
    engine: Engine = request.app.state.engine
    with engine.connect() as conn:
        row = conn.execute(select(users).where(users.c.email == email)).mappings().first()
    ok = verify_password(body.password, row["password_hash"] if row else None)
    if not ok or row is None or not row["is_active"]:
        throttle.fail(*keys)
        raise HTTPException(401, "Wrong e-mail or password")
    throttle.clear(*keys)
    token = open_session(engine, int(row["id"]), request.headers.get("user-agent"))
    _set_cookie(response, request, token)
    return Me(mode="accounts", user=_public(dict(row)))


@router.post("/auth/logout", status_code=204)
def logout(request: Request, response: Response) -> None:
    token = request.cookies.get(COOKIE)
    if token:
        with request.app.state.engine.begin() as conn:
            conn.execute(delete(sessions).where(sessions.c.id == _token_id(token)))
    response.delete_cookie(COOKIE, path="/")


class PasswordIn(BaseModel):
    current: str = Field(max_length=256)
    new: str = Field(max_length=256)


@router.post("/auth/password", status_code=204)
def change_password(request: Request, body: PasswordIn) -> None:
    user = getattr(request.state, "user", None)
    if mode() != "accounts" or user is None:
        raise HTTPException(409, "Passwords exist in accounts mode only")
    if not verify_password(body.current, user.get("password_hash")):
        raise HTTPException(401, "Current password is wrong")
    check_password_rules(body.new)
    token = request.cookies.get(COOKIE, "")
    with request.app.state.engine.begin() as conn:
        conn.execute(
            update(users)
            .where(users.c.id == user["id"])
            .values(password_hash=hash_password(body.new), updated_at=datetime.now(UTC))
        )
        # other devices are signed out; this one keeps its session
        conn.execute(
            delete(sessions).where(
                sessions.c.user_id == user["id"], sessions.c.id != _token_id(token)
            )
        )


# ── users (admin) ────────────────────────────────────────────────────────────


class UserIn(BaseModel):
    email: str = Field(pattern=EMAIL, max_length=254)
    display_name: str = Field(min_length=1, max_length=80)
    role: Role = "analyst"
    password: str = Field(max_length=256)


class UserPatch(BaseModel):
    display_name: str | None = Field(None, min_length=1, max_length=80)
    role: Role | None = None
    is_active: bool | None = None
    password: str | None = Field(None, max_length=256)


def create_user(engine: Engine, body: UserIn) -> dict[str, Any]:
    check_password_rules(body.password)
    email = body.email.lower()
    with engine.begin() as conn:
        if conn.execute(select(users.c.id).where(users.c.email == email)).first():
            raise HTTPException(422, f"A user with e-mail {email} exists")
        local = conn.execute(select(users).where(users.c.id == LOCAL_USER_ID)).mappings().first()
        values = {
            "email": email,
            "display_name": body.display_name,
            "role": body.role,
            "password_hash": hash_password(body.password),
            "updated_at": datetime.now(UTC),
        }
        if body.role == "admin" and local is not None and local["password_hash"] is None:
            # the first admin adopts the local user and everything it owns
            conn.execute(update(users).where(users.c.id == LOCAL_USER_ID).values(**values))
            uid = LOCAL_USER_ID
        else:
            uid = int(conn.execute(insert(users).values(**values)).inserted_primary_key[0])  # type: ignore[index]
        row = conn.execute(select(users).where(users.c.id == uid)).mappings().one()
    return _public(dict(row))


def _active_admins(conn: Any) -> int:
    query = select(func.count()).where(
        users.c.role == "admin", users.c.is_active.is_(True), users.c.password_hash.is_not(None)
    )
    return int(conn.execute(query).scalar_one())


@router.get("/users")
def list_users(request: Request) -> list[dict[str, Any]]:
    with request.app.state.engine.connect() as conn:
        rows = conn.execute(select(users).order_by(users.c.id)).mappings().all()
    return [_public(dict(r)) for r in rows]


@router.post("/users", status_code=201)
def post_user(request: Request, body: UserIn) -> dict[str, Any]:
    return create_user(request.app.state.engine, body)


@router.patch("/users/{user_id}")
def patch_user(request: Request, user_id: int, body: UserPatch) -> dict[str, Any]:
    values: dict[str, Any] = body.model_dump(exclude_none=True, exclude={"password"})
    if body.password is not None:
        check_password_rules(body.password)
        values["password_hash"] = hash_password(body.password)
    engine: Engine = request.app.state.engine
    with engine.begin() as conn:
        row = conn.execute(select(users).where(users.c.id == user_id)).mappings().first()
        if row is None:
            raise HTTPException(404, f"Unknown user {user_id}")
        demoted = row["role"] == "admin" and (
            values.get("role", "admin") != "admin" or values.get("is_active") is False
        )
        if demoted and _active_admins(conn) <= 1:
            raise HTTPException(422, "The last admin cannot be demoted or deactivated")
        values["updated_at"] = datetime.now(UTC)
        conn.execute(update(users).where(users.c.id == user_id).values(**values))
        if values.get("is_active") is False or "password_hash" in values:
            conn.execute(delete(sessions).where(sessions.c.user_id == user_id))
        updated = conn.execute(select(users).where(users.c.id == user_id)).mappings().one()
    return _public(dict(updated))


# ── sharing ──────────────────────────────────────────────────────────────────


class ShareIn(BaseModel):
    shared: bool


def sharing_router() -> APIRouter:
    from fda_data import scope

    share = APIRouter(prefix="/v1/sharing", tags=["auth"])
    tables = {t.name: t for t in scope.OWNED if t.name != "agent_actions"}

    @share.put("/{kind}/{item_id}")
    def set_shared(request: Request, kind: str, item_id: int, body: ShareIn) -> dict[str, Any]:
        """Share one of your items read-only with every signed-in user (or stop sharing)."""
        table = tables.get(kind)
        if table is None:
            raise HTTPException(404, f"Unknown kind {kind}; one of {', '.join(sorted(tables))}")
        user = getattr(request.state, "user", None) or {}
        with request.app.state.engine.begin() as conn:
            done = conn.execute(
                update(table)
                .where(table.c.id == item_id, table.c.owner_id == user.get("id"))
                .values(shared=body.shared)
            ).rowcount
        if not done:
            raise HTTPException(404, f"Unknown {kind} {item_id}")
        return {"kind": kind, "id": item_id, "shared": body.shared}

    return share
