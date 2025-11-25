-- Enable extensions
CREATE EXTENSION IF NOT EXISTS timescaledb;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Raw trade data (hypertable for time-series performance)
CREATE TABLE IF NOT EXISTS trades_raw (
    time            TIMESTAMPTZ NOT NULL,
    symbol          TEXT NOT NULL,
    price           DOUBLE PRECISION NOT NULL,
    quantity        DOUBLE PRECISION NOT NULL,
    is_buyer_maker  BOOLEAN NOT NULL,
    trade_id        BIGINT NOT NULL
);
SELECT create_hypertable('trades_raw', 'time', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_trades_symbol_time ON trades_raw (symbol, time DESC);

-- Aggregated OHLCV candles
CREATE TABLE IF NOT EXISTS candles (
    time        TIMESTAMPTZ NOT NULL,
    symbol      TEXT NOT NULL,
    timeframe   TEXT NOT NULL,
    open        DOUBLE PRECISION,
    high        DOUBLE PRECISION,
    low         DOUBLE PRECISION,
    close       DOUBLE PRECISION,
    volume      DOUBLE PRECISION,
    delta       DOUBLE PRECISION,      -- Net buy/sell delta
    trade_count INTEGER,
    PRIMARY KEY (time, symbol, timeframe)
);
SELECT create_hypertable('candles', 'time', if_not_exists => TRUE);

-- Volume Profile snapshots
CREATE TABLE IF NOT EXISTS volume_profile (
    time        TIMESTAMPTZ NOT NULL,
    symbol      TEXT NOT NULL,
    lookback    TEXT NOT NULL,         -- 'session', 'daily', 'weekly'
    poc         DOUBLE PRECISION,
    vah         DOUBLE PRECISION,
    val         DOUBLE PRECISION,
    profile     JSONB                  -- Full price-level distribution
);
SELECT create_hypertable('volume_profile', 'time', if_not_exists => TRUE);

-- Order flow signals (audit trail)
CREATE TABLE IF NOT EXISTS signals (
    signal_id           UUID DEFAULT uuid_generate_v4() NOT NULL,
    time                TIMESTAMPTZ NOT NULL,
    symbol              TEXT NOT NULL,
    strategy_id         TEXT NOT NULL,
    direction           TEXT,
    grade               TEXT,
    entry_price         DOUBLE PRECISION,
    stop_loss           DOUBLE PRECISION,
    take_profit         DOUBLE PRECISION,
    confluence_score    INTEGER,
    rationale           TEXT,
    metadata            JSONB,
    PRIMARY KEY (time, signal_id)
);
SELECT create_hypertable('signals', 'time', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_signals_uuid ON signals (signal_id);

-- Executed trades (regular table, not hypertable - needs FK support)
CREATE TABLE IF NOT EXISTS executions (
    id              SERIAL PRIMARY KEY,
    signal_id       UUID NOT NULL,        -- Correlates to signals.signal_id
    time_entry      TIMESTAMPTZ NOT NULL,
    time_exit       TIMESTAMPTZ,
    symbol          TEXT NOT NULL,
    side            TEXT NOT NULL,
    entry_price     DOUBLE PRECISION,
    exit_price      DOUBLE PRECISION,
    quantity        DOUBLE PRECISION,
    pnl_realized    DOUBLE PRECISION,
    pnl_percent     DOUBLE PRECISION,
    fees            DOUBLE PRECISION,
    slippage        DOUBLE PRECISION,
    exit_reason     TEXT,              -- 'TP', 'SL', 'MANUAL', 'INVALIDATION'
    metadata        JSONB
);
CREATE INDEX IF NOT EXISTS idx_executions_signal ON executions (signal_id);
CREATE INDEX IF NOT EXISTS idx_executions_time ON executions (time_entry DESC);

-- Session state snapshots
CREATE TABLE IF NOT EXISTS session_state (
    time            TIMESTAMPTZ NOT NULL,
    session_id      TEXT NOT NULL,
    realized_pnl    DOUBLE PRECISION,
    unrealized_pnl  DOUBLE PRECISION,
    win_count       INTEGER,
    loss_count      INTEGER,
    consecutive_losses INTEGER,
    risk_multiplier DOUBLE PRECISION,  -- House money factor
    is_active       BOOLEAN
);
SELECT create_hypertable('session_state', 'time', if_not_exists => TRUE);

-- LLM analysis journal
CREATE TABLE IF NOT EXISTS trade_journal (
    id              SERIAL PRIMARY KEY,
    execution_id    INTEGER REFERENCES executions(id),
    time            TIMESTAMPTZ NOT NULL,
    analysis_type   TEXT,              -- 'post_trade', 'pre_market', 'session_review'
    prompt          TEXT,
    response        TEXT,
    key_insights    JSONB
);
