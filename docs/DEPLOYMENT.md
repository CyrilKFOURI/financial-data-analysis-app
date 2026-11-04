# Deployment

Three ways to run the workbench, all from one `compose.yaml` (M31 §4, TRD §13, feature 273). Secrets and settings live in `.env` (copy `.env.example`); never commit it.

| Profile | Command | What runs |
|---|---|---|
| Native dev | `docker compose up -d`, then `uv run fastapi dev …` and `corepack pnpm --dir apps/web dev` | Postgres (127.0.0.1:5433) and Redis only |
| Laptop / home server | `docker compose --profile app up -d --build` | + migrations, API, worker and web proxy (Caddy) |
| + GPU worker | add `--profile gpu` | a second worker with an NVIDIA GPU reserved |
| Cloud | `docker compose -f compose.yaml -f compose.cloud.yaml --profile app up -d` | API replicas, workers and one scheduler, against managed Postgres and Redis |

## Laptop
```bash
docker compose --profile app up -d --build
```
Open http://127.0.0.1:8080:
- The proxy serves the web app and forwards `/v1` and `/mcp` to the API.
- The API is not published on its own.
- Migrations run first, in the one-shot `migrate` service.
- The in-process scheduler is on (`FDA_SCHEDULER=1`), so schedules run on time.

Data lives in Docker volumes:
- `postgres_data`;
- `fda_data` (lake, reports, backups);
- `redis_data`;
- `caddy_data` (certificates).

## Home server
In `.env`:
```dotenv
POSTGRES_PASSWORD=<long random string, letters and digits only>
FDA_BIND=0.0.0.0              # listen on the LAN, not just the server itself
FDA_HTTP_PORT=80
FDA_HTTPS_PORT=443
FDA_SITE=fda.example.org      # a DNS name pointing at the server → automatic HTTPS; ":80" for plain HTTP on the LAN
FDA_PUBLIC_URL=https://fda.example.org
FDA_AUTH_MODE=accounts        # e-mail + password sign-in, every user sees only their items
FDA_COOKIE_SECURE=1           # behind HTTPS
```
**User accounts.** Create the first admin before switching `FDA_AUTH_MODE` to `accounts`:
```bash
docker compose exec api python -m fda_api.users create --email you@example.org --name "You" --role admin
```
- The first admin adopts the built-in local user, so everything created in single-user mode stays theirs.
- Add other users on Settings · Users & Roles. Roles are viewer, analyst and admin.
- Every query is restricted to the user's own rows by the API and, on Postgres, by row-level security as well.

**Optional extra lock on the proxy.** `FDA_AUTH=basic` adds a shared password in front of everything:
```dotenv
FDA_AUTH=basic
FDA_BASIC_AUTH_USER=fda
FDA_BASIC_AUTH_HASH=<bcrypt hash>
```
**Password hash.** Generate it with:
```bash
docker run --rm caddy:2.10-alpine caddy hash-password --plaintext "choose-a-password"
```
Inside `.env`, write each `$` of the hash as `$$`, because Compose interpolates `$`.

**Exposure.** In single mode with no proxy password, anyone who can reach the port can use the app: use `FDA_AUTH_MODE=accounts`, or keep the server on a trusted LAN or behind a VPN (WireGuard, Tailscale). `/mcp` stays protected by its own bearer token (`FDA_MCP_TOKEN`) and is off while the token is empty.

**Changing the database password later.** Postgres keeps the password it was first created with. Change it with `ALTER USER` inside the database, then update `.env`.

## Backups and the restore drill
**What an archive holds.** `fda_data.backup` writes `fda-backup-<UTC stamp>.tar.gz`, containing:
- `db.dump`: `pg_dump -Fc`, taken from the same snapshot as the row counts;
- the `lake/` and `reports/` trees;
- `manifest.json`: alembic revision, rows per table, SHA-256 of every file.

**Nightly backup.** Create a schedule on the Data · Schedules page, kind *backup*, e.g. cron `0 3 * * *`, keep 14. Archives land in the `fda_data` volume under `/data/backups`.

**Commands.** Run them in the API container:
```bash
docker compose exec api python -m fda_data.backup create --keep 14
docker compose exec api python -m fda_data.backup drill          # newest archive
```
The drill does the following:
- restores into a scratch database `fda_restore_drill` and a temporary lake;
- checks the alembic revision, every table's row count and every file checksum;
- reads one price file back;
- drops the scratch database.

The exit code is non-zero on any mismatch. Run it after each upgrade and at least once a month.

**Off-machine copy.** Copy `/data/backups` to another disk or machine, e.g. a nightly `rsync` from the host path given by:
```bash
docker volume inspect fda_fda_data
```

**Full restore after a loss.**
1. Start `postgres` only, on an empty volume.
2. Create an empty database.
3. Restore into it and into an empty data directory:
   ```bash
   docker compose run --rm migrate python -m fda_data.backup restore /data/backups/<archive> \
     --database-url "$DATABASE_URL" --data-dir /data
   ```
4. Start the profile.

A restore refuses a non-empty database or data directory: it never overwrites live data.

## Cloud
**Required variables.** Set them in the environment, or inject them from a secrets manager:
- `DATABASE_URL`: managed Postgres 16, `postgresql+psycopg://…`;
- `REDIS_URL`;
- `FDA_PUBLIC_URL`;
- `FDA_LAKE_HOST_PATH`: a shared filesystem mounted on every host (NFS, EFS, Azure Files).

**Topology.**
- `FDA_API_REPLICAS` (default 2) API processes sit behind the proxy, which can be replaced by a cloud load balancer.
- `FDA_WORKER_REPLICAS` (default 2) workers.
- A single `scheduler` process runs the schedules, so replicas never run one twice.

**Database password.** URL-encode any special characters in the password inside `DATABASE_URL`.

**Not available yet.** A lake directly on S3/MinIO needs `s3fs`, which needs approval as a new dependency first.

## Upgrades
```bash
git pull
docker compose --profile app up -d --build   # migrate runs before the API starts
docker compose exec api python -m fda_data.backup drill
```
Take a backup before upgrading:
```bash
docker compose exec api python -m fda_data.backup create
```
