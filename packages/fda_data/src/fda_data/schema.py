"""Reference & system tables (BACKEND_SCHEMA.md §3, §5, §7) — SQLAlchemy Core, Alembic-managed."""

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    Column,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    Numeric,
    String,
    Table,
    Text,
    UniqueConstraint,
    func,
)

from fda_data.actor import current_owner

NAMING = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}

metadata = MetaData(naming_convention=NAMING)


def _timestamps() -> list[Column[object]]:
    return [
        Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
        Column(
            "updated_at",
            DateTime(timezone=True),
            nullable=False,
            server_default=func.now(),
            onupdate=func.now(),
        ),
    ]


exchanges = Table(
    "exchanges",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("mic", String(10), nullable=False, unique=True),
    Column("name", Text, nullable=False),
    Column("country", String(2)),
    Column("timezone", Text, nullable=False),
    Column("calendar_code", Text),
)

instruments = Table(
    "instruments",
    metadata,
    Column(
        "id", BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True
    ),
    Column("symbol", String(32), nullable=False),
    Column("name", Text, nullable=False),
    Column("asset_class", String(16), nullable=False),
    Column("asset_subclass", String(32)),
    Column("leverage_factor", Numeric(6, 2)),
    Column("currency", String(3), nullable=False),
    Column("exchange_id", ForeignKey("exchanges.id")),
    Column("country", String(2)),
    Column("sector", Text),
    Column("industry", Text),
    Column("status", String(16), nullable=False, server_default="active"),
    Column("timezone", Text),
    Column("first_trade_date", Date),
    Column("last_trade_date", Date),
    Column("classification_source", String(16)),
    Column("classification_reason", Text),
    Column("classification_confidence", String(8)),
    Column("provider_symbols", JSON, nullable=False, server_default="{}"),
    # Raw provider extras (quote type, fund category, crypto supply…): inputs to type detection.
    Column("profile", JSON, nullable=False, server_default="{}"),
    *_timestamps(),
    UniqueConstraint("symbol", name="uq_instruments_symbol"),
)

instrument_identifiers = Table(
    "instrument_identifiers",
    metadata,
    Column(
        "id", BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True
    ),
    Column("instrument_id", ForeignKey("instruments.id", ondelete="CASCADE"), nullable=False),
    Column("scheme", String(16), nullable=False),
    Column("value", String(64), nullable=False),
    Column("valid_from", Date),
    Column("valid_to", Date),
    UniqueConstraint("scheme", "value", "valid_from", name="uq_identifiers_scheme_value_from"),
)

instrument_overrides = Table(
    "instrument_overrides",
    metadata,
    Column(
        "id", BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True
    ),
    Column("instrument_id", ForeignKey("instruments.id", ondelete="CASCADE"), nullable=False),
    Column("field", String(32), nullable=False),
    Column("value", Text, nullable=False),
    Column("reason", Text),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    UniqueConstraint("instrument_id", "field", name="uq_instrument_overrides_instrument_field"),
)

series_catalog = Table(
    "series_catalog",
    metadata,
    Column(
        "id", BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True
    ),
    Column("dataset", String(32), nullable=False),
    Column("key", Text, nullable=False),
    Column("instrument_id", ForeignKey("instruments.id", ondelete="CASCADE")),
    Column("first_date", Date),
    Column("last_date", Date),
    Column("rows", Integer, nullable=False, server_default="0"),
    Column("source", Text),
    Column("last_ingested_at", DateTime(timezone=True)),
    UniqueConstraint("dataset", "key", name="uq_series_catalog_dataset_key"),
)

ingestion_jobs = Table(
    "ingestion_jobs",
    metadata,
    Column(
        "id", BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True
    ),
    Column("kind", String(32), nullable=False),
    Column("target", JSON, nullable=False),
    Column("status", String(16), nullable=False, server_default="queued"),
    Column("progress", Numeric(5, 4), nullable=False, server_default="0"),
    Column("message", Text),
    Column("provider", Text),
    Column("rows_written", Integer, nullable=False, server_default="0"),
    Column("qc_summary", JSON),
    Column("error", Text),
    Column("attempts", Integer, nullable=False, server_default="0"),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    Column("started_at", DateTime(timezone=True)),
    Column("finished_at", DateTime(timezone=True)),
)

# Point-in-time fundamentals (BACKEND_SCHEMA §4.1). The canonical dictionary lives in
# fda_data/mapping/line_items.yaml (versioned with the code), hence `line_item` is its code.
fundamental_facts = Table(
    "fundamental_facts",
    metadata,
    Column(
        "id", BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True
    ),
    Column("instrument_id", ForeignKey("instruments.id", ondelete="CASCADE"), nullable=False),
    Column("line_item", String(48), nullable=False),
    Column("period_type", String(3), nullable=False),
    Column("fiscal_year", Integer, nullable=False),
    Column("fiscal_quarter", Integer),
    Column("period_start", Date),
    Column("period_end", Date, nullable=False),
    Column("value", Float, nullable=False),
    Column("unit", String(16), nullable=False),
    Column("filed_at", Date, nullable=False),
    Column("source", String(16), nullable=False),
    Column("source_ref", Text),
    Column("form", String(16)),
    Column("is_restated", Boolean, nullable=False, server_default="0"),
    Column("estimated_pit", Boolean, nullable=False, server_default="0"),
    Column("ingested_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    UniqueConstraint(
        "instrument_id",
        "line_item",
        "period_type",
        "period_end",
        "source",
        "filed_at",
        name="uq_fundamental_facts_key",
    ),
    Index("ix_fundamental_facts_lookup", "instrument_id", "line_item", "period_type", "period_end"),
)


# Peer groups (BACKEND_SCHEMA §3.6, DECISIONS D-PE-1/2): auto proposal, editable, frozen once
# edited ("manual") until reset.
peer_groups = Table(
    "peer_groups",
    metadata,
    Column(
        "id", BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True
    ),
    Column("instrument_id", ForeignKey("instruments.id", ondelete="CASCADE"), nullable=False),
    Column("mode", String(8), nullable=False, server_default="auto"),
    Column("algorithm_version", String(16), nullable=False),
    Column("weights", JSON, nullable=False),
    Column("details", JSON, nullable=False, server_default="{}"),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    Column("frozen_at", DateTime(timezone=True)),
    UniqueConstraint("instrument_id", name="uq_peer_groups_instrument_id"),
)

peer_group_members = Table(
    "peer_group_members",
    metadata,
    Column(
        "id", BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True
    ),
    Column("peer_group_id", ForeignKey("peer_groups.id", ondelete="CASCADE"), nullable=False),
    Column("symbol", String(32), nullable=False),
    Column("name", Text),
    Column("score", Float),
    Column("breakdown", JSON, nullable=False, server_default="{}"),
    Column("metrics", JSON, nullable=False, server_default="{}"),
    Column("pinned", Boolean, nullable=False, server_default="0"),
    Column("source", String(8), nullable=False, server_default="auto"),
    UniqueConstraint("peer_group_id", "symbol", name="uq_peer_group_members_group_symbol"),
)

# Latest third-party snapshot per instrument and kind (earnings dates, estimates, analyst
# views, holders, insiders, short interest): data shown with its source and date.
provider_snapshots = Table(
    "provider_snapshots",
    metadata,
    Column(
        "id", BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True
    ),
    Column("instrument_id", ForeignKey("instruments.id", ondelete="CASCADE"), nullable=False),
    Column("kind", String(32), nullable=False),
    Column("source", String(16), nullable=False),
    Column("as_of", DateTime(timezone=True), nullable=False, server_default=func.now()),
    Column("payload", JSON, nullable=False),
    UniqueConstraint("instrument_id", "kind", name="uq_provider_snapshots_instrument_kind"),
)


# Portfolios (BACKEND_SCHEMA §6.2). Single local user (D-PL-1): no user_id yet. Holdings are
# referenced by symbol (instruments are created on ingestion), like peer group members.
def _pk() -> Column[int]:
    return Column(
        "id", BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True
    )


AMOUNT = Numeric(28, 10, asdecimal=False)

# Users & sessions (M31 §2). Row 1 is the built-in single-user owner ("local"); in accounts mode
# the first admin adopts it, so data created before multi-user stays with that admin.
users = Table(
    "users",
    metadata,
    _pk(),
    Column("email", Text, nullable=False),
    Column("display_name", Text, nullable=False),
    Column("role", String(16), nullable=False),  # viewer | analyst | admin
    Column("password_hash", Text),  # scrypt; NULL = cannot sign in (the local user)
    Column("is_active", Boolean, nullable=False, server_default="1"),
    Column("last_login_at", DateTime(timezone=True)),
    *_timestamps(),
    UniqueConstraint("email", name="uq_users_email"),
)

# Only the SHA-256 of a session token is stored: a leaked table cannot be replayed as cookies.
sessions = Table(
    "sessions",
    metadata,
    Column("id", String(64), primary_key=True),
    Column("user_id", ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    Column("expires_at", DateTime(timezone=True), nullable=False),
    Column("last_seen_at", DateTime(timezone=True), nullable=False),
    Column("user_agent", Text),
    Index("ix_sessions_user", "user_id"),
)


def _ownership() -> list[Column[object]]:
    """Owner (NULL = system/reference row) and read-only sharing with every signed-in user."""
    return [
        Column(
            "owner_id",
            ForeignKey("users.id", ondelete="CASCADE"),
            default=current_owner,
            index=True,
        ),
        Column("shared", Boolean, nullable=False, server_default="0", default=False),
    ]


portfolios = Table(
    "portfolios",
    metadata,
    _pk(),
    *_ownership(),
    Column("name", Text, nullable=False),
    Column("mode", String(12), nullable=False),  # weights | transactions
    Column("base_currency", String(3), nullable=False, server_default="USD"),
    Column("benchmark", String(40)),  # instrument symbol or "portfolio:<id>"
    Column("is_benchmark", Boolean, nullable=False, server_default="0"),
    Column("lot_method", String(8), nullable=False, server_default="fifo"),
    Column("current_version_no", Integer, nullable=False, server_default="1"),
    Column("archived_at", DateTime(timezone=True)),
    *_timestamps(),
    UniqueConstraint("owner_id", "name", name="uq_portfolios_owner_id"),
)

portfolio_versions = Table(
    "portfolio_versions",
    metadata,
    _pk(),
    Column("portfolio_id", ForeignKey("portfolios.id", ondelete="CASCADE"), nullable=False),
    Column("version_no", Integer, nullable=False),
    Column("note", Text),
    Column("rebalance", JSON, nullable=False, server_default="{}"),
    Column("cost_bps", Float, nullable=False, server_default="0"),
    Column("start_date", Date),
    Column("initial_value", Float, nullable=False, server_default="100000"),
    Column("definition_hash", String(64), nullable=False),
    Column("is_draft", Boolean, nullable=False, server_default="0"),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    UniqueConstraint("portfolio_id", "version_no", name="uq_portfolio_versions_portfolio_no"),
)

portfolio_targets = Table(
    "portfolio_targets",
    metadata,
    _pk(),
    Column("version_id", ForeignKey("portfolio_versions.id", ondelete="CASCADE"), nullable=False),
    Column("symbol", String(32), nullable=False),
    Column("weight", Float, nullable=False),
    Column("is_cash", Boolean, nullable=False, server_default="0"),
    Column("locked", Boolean, nullable=False, server_default="0"),
    Column("note", Text),
)

import_batches = Table(
    "import_batches",
    metadata,
    _pk(),
    Column("portfolio_id", ForeignKey("portfolios.id", ondelete="CASCADE"), nullable=False),
    Column("source", String(16), nullable=False, server_default="csv"),
    Column("file_name", Text),
    Column("idempotency_key", String(64), nullable=False),
    Column("row_count", Integer, nullable=False, server_default="0"),
    Column("status", String(16), nullable=False),
    Column("errors", JSON, nullable=False, server_default="[]"),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    UniqueConstraint("idempotency_key", name="uq_import_batches_idempotency_key"),
)

# Append-only ledger: corrections are a void + a new row. Cash-only types carry the cash
# amount in `quantity` (price 1).
transactions = Table(
    "transactions",
    metadata,
    _pk(),
    Column("portfolio_id", ForeignKey("portfolios.id", ondelete="CASCADE"), nullable=False),
    Column("trade_date", Date, nullable=False),
    Column("settle_date", Date),
    Column("type", String(16), nullable=False),
    Column("symbol", String(32)),
    Column("quantity", AMOUNT, nullable=False),
    Column("price", AMOUNT, nullable=False, server_default="1"),
    Column("currency", String(3), nullable=False),
    Column("fees", AMOUNT, nullable=False, server_default="0"),
    Column("fx_rate_to_base", AMOUNT),
    Column("account", Text),
    Column("external_id", String(64)),
    Column("import_batch_id", ForeignKey("import_batches.id", ondelete="SET NULL")),
    Column("voided_at", DateTime(timezone=True)),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    UniqueConstraint("portfolio_id", "external_id", name="uq_transactions_portfolio_external"),
    Index("ix_transactions_portfolio_date", "portfolio_id", "trade_date"),
)


# Comparisons (BACKEND_SCHEMA §6.3): members as an ordered JSON list of subject refs
# ({"type": "asset"|"portfolio", "id": ...}) instead of a member table.
comparisons = Table(
    "comparisons",
    metadata,
    _pk(),
    *_ownership(),
    Column("name", Text, nullable=False),
    Column("note", Text),
    Column("members", JSON, nullable=False),
    Column("context", JSON, nullable=False, server_default="{}"),
    *_timestamps(),
    UniqueConstraint("owner_id", "name", name="uq_comparisons_owner_id"),
)

# Analysis alerts (CP-005): a value path in an analysis payload compared with a threshold.
alert_rules = Table(
    "alert_rules",
    metadata,
    _pk(),
    *_ownership(),
    Column("name", Text, nullable=False),
    Column("subject_type", String(16), nullable=False),
    Column("subject_id", String(64), nullable=False),
    Column("analysis_id", String(16), nullable=False),
    Column("params", JSON, nullable=False, server_default="{}"),
    Column("context", JSON, nullable=False, server_default="{}"),
    Column("path", String(200), nullable=False),
    Column("op", String(4), nullable=False),  # > >= < <=
    Column("threshold", Float, nullable=False),
    Column("webhook_url", Text),  # optional outgoing JSON (TR-012); never places orders
    Column("active", Boolean, nullable=False, server_default="1"),
    Column("last_value", Float),
    Column("last_state", Boolean),
    Column("last_evaluated_at", DateTime(timezone=True)),
    *_timestamps(),
)

notifications = Table(
    "notifications",
    metadata,
    _pk(),
    Column("rule_id", ForeignKey("alert_rules.id", ondelete="CASCADE")),
    Column("message", Text, nullable=False),
    Column("value", Float),
    Column("link", Text),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    Column("read_at", DateTime(timezone=True)),
)

# Snapshots (CP-006): selected analysis outputs of a subject frozen at a date.
snapshots = Table(
    "snapshots",
    metadata,
    _pk(),
    *_ownership(),
    Column("subject_type", String(16), nullable=False),
    Column("subject_id", String(64), nullable=False),
    Column("label", Text, nullable=False),
    Column("metrics", JSON, nullable=False),
    Column("versions", JSON, nullable=False, server_default="{}"),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    Index("ix_snapshots_subject", "subject_type", "subject_id"),
)

# Universes (SC-001): constituent lists with membership dates (today's list only from free
# sources: membership is survivorship-aware going forward, not backwards).
universes = Table(
    "universes",
    metadata,
    _pk(),
    *_ownership(),
    Column("name", Text, nullable=False),
    Column("kind", String(16), nullable=False),  # index | crypto | custom | portfolio
    Column("source", String(32)),  # sp500, nasdaq100, dow30, cac40, crypto50, portfolio:<id>
    Column("refreshed_at", DateTime(timezone=True)),
    *_timestamps(),
    UniqueConstraint("owner_id", "name", name="uq_universes_owner_id"),
)

universe_members = Table(
    "universe_members",
    metadata,
    _pk(),
    Column("universe_id", ForeignKey("universes.id", ondelete="CASCADE"), nullable=False),
    Column("symbol", String(32), nullable=False),
    Column("name", Text),
    Column("sector", Text),
    Column("valid_from", Date, nullable=False),
    Column("valid_to", Date),
    Index("ix_universe_members_universe", "universe_id"),
)

# Saved screens (SC-007): filters re-run later; the last result gives entrants/leavers.
saved_screens = Table(
    "saved_screens",
    metadata,
    _pk(),
    *_ownership(),
    Column("name", Text, nullable=False),
    Column("universe_id", ForeignKey("universes.id", ondelete="CASCADE"), nullable=False),
    Column("filters", JSON, nullable=False),
    Column("columns", JSON, nullable=False, server_default="[]"),
    Column("last_run_at", DateTime(timezone=True)),
    Column("last_result", JSON),
    *_timestamps(),
)

# Investor profiles (IN-001): questionnaire answers; every save is a new row (history kept).
investor_profiles = Table(
    "investor_profiles",
    metadata,
    _pk(),
    *_ownership(),
    Column("name", Text, nullable=False),
    Column("answers", JSON, nullable=False),
    Column("scoring_version", String(16), nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
)

# Strategies (M25): a versionless research object — hypothesis note + spec (symbol, template,
# parameters, grid, costs, protocol). Trials are append-only (the DSR/PBO trial counter cannot
# be reset silently).
strategies = Table(
    "strategies",
    metadata,
    _pk(),
    *_ownership(),
    Column("name", Text, nullable=False),
    Column("hypothesis", Text, nullable=False),
    Column("spec", JSON, nullable=False),
    *_timestamps(),
    UniqueConstraint("owner_id", "name", name="uq_strategies_owner_id"),
)

strategy_trials = Table(
    "strategy_trials",
    metadata,
    _pk(),
    Column("strategy_id", ForeignKey("strategies.id", ondelete="CASCADE"), nullable=False),
    Column("config_hash", String(64), nullable=False),
    Column("configurations", Integer, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    UniqueConstraint("strategy_id", "config_hash", name="uq_strategy_trials_config"),
)

# Paper trading (M29, TR-040): accounts and orders only; fills, positions and statements are
# replayed deterministically from the stored bars (so the P&L reconciles to the fills).
paper_accounts = Table(
    "paper_accounts",
    metadata,
    _pk(),
    *_ownership(),
    Column("name", Text, nullable=False),
    Column("currency", String(3), nullable=False),
    Column("starting_cash", AMOUNT, nullable=False),
    Column("costs", JSON, nullable=False),
    *_timestamps(),
    UniqueConstraint("owner_id", "name", name="uq_paper_accounts_owner_id"),
)

paper_orders = Table(
    "paper_orders",
    metadata,
    _pk(),
    Column("account_id", ForeignKey("paper_accounts.id", ondelete="CASCADE"), nullable=False),
    Column("symbol", Text, nullable=False),
    Column("side", String(4), nullable=False),
    Column("quantity", AMOUNT, nullable=False),
    Column("order_type", String(16), nullable=False),
    Column("limit_price", AMOUNT),
    Column("stop_price", AMOUNT),
    Column("trail_pct", AMOUNT),
    Column("tif", String(3), nullable=False),
    Column("decision_date", Date, nullable=False),
    Column("decision_price", AMOUNT, nullable=False),
    Column("cancelled_on", Date),
    Column("source", Text, nullable=False),
    Column("note", Text),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
)

# Agent write actions (M30 §2.2, AI-001/002): every write tool call becomes a pending action that
# only the web app can approve; the row is the audit log (proposed → approved/rejected → result).
agent_actions = Table(
    "agent_actions",
    metadata,
    _pk(),
    *_ownership(),
    Column("tool", Text, nullable=False),
    Column("arguments", JSON, nullable=False),
    Column("summary", Text, nullable=False),
    Column("status", String(16), nullable=False),
    Column("origin", String(16), nullable=False),
    Column("result", JSON),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    Column("decided_at", DateTime(timezone=True)),
)

# Schedules (M31 §3): cron-like recurring jobs run by the API process; every run is recorded.
schedules = Table(
    "schedules",
    metadata,
    _pk(),
    *_ownership(),
    Column("name", Text, nullable=False),
    Column("kind", String(16), nullable=False),
    Column("cron", String(64), nullable=False),
    Column("timezone", String(64), nullable=False),
    Column("spec", JSON, nullable=False),
    Column("enabled", Boolean, nullable=False),
    Column("next_run_at", DateTime(timezone=True)),
    Column("last_run_at", DateTime(timezone=True)),
    Column("last_status", String(16)),
    *_timestamps(),
    UniqueConstraint("owner_id", "name", name="uq_schedules_owner_id"),
)

schedule_runs = Table(
    "schedule_runs",
    metadata,
    _pk(),
    Column("schedule_id", ForeignKey("schedules.id", ondelete="CASCADE"), nullable=False),
    Column("started_at", DateTime(timezone=True), nullable=False),
    Column("finished_at", DateTime(timezone=True)),
    Column("status", String(16), nullable=False),
    Column("trigger", String(16), nullable=False),
    Column("result", JSON),
)

# Agent runs (M30 §3.5, AI-007): question, tool path, answer, citation check and token usage.
agent_runs = Table(
    "agent_runs",
    metadata,
    _pk(),
    *_ownership(),
    Column("provider", String(32), nullable=False),
    Column("model", Text, nullable=False),
    Column("question", Text, nullable=False),
    Column("answer", Text, nullable=False),
    Column("steps", JSON, nullable=False),
    Column("usage", JSON, nullable=False),
    Column("verification", JSON, nullable=False),
    Column("duration_ms", Integer, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
)
