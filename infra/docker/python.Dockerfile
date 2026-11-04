# syntax=docker/dockerfile:1
# One image for the API, the arq worker, migrations and backups (M31 §4).
# Build from the repository root: docker build -f infra/docker/python.Dockerfile .
FROM ghcr.io/astral-sh/uv:0.12.15 AS uv

FROM python:3.12-slim-bookworm

# pg_dump / pg_restore must match the server major version (Postgres 16), hence the PGDG repo.
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl gnupg \
 && install -d /usr/share/postgresql-common/pgdg \
 && curl -fsSL -o /usr/share/postgresql-common/pgdg/apt.postgresql.org.asc \
      https://www.postgresql.org/media/keys/ACCC4CF8.asc \
 && echo "deb [signed-by=/usr/share/postgresql-common/pgdg/apt.postgresql.org.asc] https://apt.postgresql.org/pub/repos/apt bookworm-pgdg main" \
      > /etc/apt/sources.list.d/pgdg.list \
 && apt-get update \
 && apt-get install -y --no-install-recommends postgresql-client-16 \
 && apt-get purge -y gnupg && apt-get autoremove -y \
 && rm -rf /var/lib/apt/lists/*

COPY --from=uv /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    DATA_DIR=/data

WORKDIR /app
# Third-party dependencies first: this layer is reused until uv.lock changes.
COPY pyproject.toml uv.lock .python-version ./
COPY packages/fda_core/pyproject.toml packages/fda_core/
COPY packages/fda_data/pyproject.toml packages/fda_data/
COPY packages/fda_contracts/pyproject.toml packages/fda_contracts/
COPY apps/api/pyproject.toml apps/api/
COPY apps/worker/pyproject.toml apps/worker/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-workspace

COPY packages packages
COPY apps/api apps/api
COPY apps/worker apps/worker
COPY alembic.ini ./
COPY infra/migrations infra/migrations
RUN --mount=type=cache,target=/root/.cache/uv \
    UV_COMPILE_BYTECODE=0 uv sync --frozen --no-dev --no-editable \
 && useradd --system --uid 10001 --home-dir /app fda \
 && mkdir -p /data && chown fda /data

USER fda
VOLUME ["/data"]
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/v1/health', timeout=4)" || exit 1
CMD ["uvicorn", "fda_api.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips", "*"]
