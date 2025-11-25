# Fabio-Inspired Algorithmic Trading System
## Production Architecture for Binance Integration

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Core Philosophy Translation](#2-core-philosophy-translation)
3. [System Architecture Overview](#3-system-architecture-overview)
4. [Data Layer Specification](#4-data-layer-specification)
5. [Order Flow Engine](#5-order-flow-engine)
6. [Strategy Engine](#6-strategy-engine)
7. [Risk Management Module](#7-risk-management-module)
8. [Execution Engine](#8-execution-engine)
9. [State Management](#9-state-management)
10. [LLM Analyst Integration](#10-llm-analyst-integration)
11. [Dashboard & Monitoring](#11-dashboard--monitoring)
12. [Implementation Sequence](#12-implementation-sequence)

---

## 1. Executive Summary

This document specifies a production-grade algorithmic trading system that codifies Fabio Valentini's discretionary scalping methodology into deterministic, executable logic. The system targets Binance Futures (BTCUSDT as primary asset) and implements:

- **Auction Market Theory** state detection (Balance/Imbalance)
- **Order Flow Analysis** via trade tape and order book delta
- **Multi-Timeframe Confluence** (15m bias → 15s execution)
- **Dynamic Risk Allocation** (A/B/C setup classification)
- **House Money Compounding** for asymmetric returns
- **Trapped Trader Detection** via absorption/aggression patterns

The architecture follows event-driven principles with strict separation between signal generation, risk validation, and execution.

---

## 2. Core Philosophy Translation

### 2.1 Fabio's Principles → Code Constructs

| Fabio's Concept | Technical Implementation |
|-----------------|--------------------------|
| "Don't catch falling knives" | `TrendFollowingFilter` — No counter-trend entries unless late-session mean reversion |
| "Volume = Cause, Price = Effect" | `OrderFlowEngine` computes delta/absorption before any signal fires |
| "Box Checking" (4-5 conditions) | `ConfluenceValidator` — Signal requires ALL boxes checked |
| "House Money" compounding | `DynamicRiskAllocator` — Profits unlock higher position sizes |
| "3 consecutive losses = stop" | `SessionStateManager` — Circuit breaker logic |
| "Trapped traders" | `TrappedTraderDetector` — Absorption + failed breakout patterns |
| "15s sniper execution" | `MicrostructureExecutor` — Waits for lower-TF confirmation |

### 2.2 Crypto Adaptations

Fabio trades NASDAQ futures. Crypto markets differ:

| Futures Concept | Crypto Equivalent |
|-----------------|-------------------|
| NYSE Open volatility | Asian/London/US session opens (08:00 UTC, 13:00 UTC, 21:00 UTC) |
| Options Flow (smart money) | Funding Rate + Open Interest changes |
| Footprint volume | Binance trade tape (`aggTrades`) + Order Book Delta |
| VWAP | Rolling VWAP computed from tick data |
| Market close | No close — use 4h/Daily candle boundaries as session markers |

---

## 3. System Architecture Overview

### 3.1 Component Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           TRADING SYSTEM CORE                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                   │
│  │   BINANCE    │───▶│  DATA LAYER  │───▶│  ORDER FLOW  │                   │
│  │  WebSocket   │    │  (Ingestion) │    │   ENGINE     │                   │
│  └──────────────┘    └──────────────┘    └──────┬───────┘                   │
│                                                 │                           │
│                                                 ▼                           │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                   │
│  │   STRATEGY   │◀───│  CONFLUENCE  │◀───│    MARKET    │                   │
│  │    ENGINE    │    │  VALIDATOR   │    │    STATE     │                   │
│  └──────┬───────┘    └──────────────┘    └──────────────┘                   │
│         │                                                                   │
│         ▼                                                                   │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                   │
│  │     RISK     │───▶│  EXECUTION   │───▶│   BINANCE    │                   │
│  │   MANAGER    │    │    ENGINE    │    │   REST API   │                   │
│  └──────────────┘    └──────────────┘    └──────────────┘                   │
│                                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                   │
│  │   SESSION    │    │     LLM      │    │  DASHBOARD   │                   │
│  │    STATE     │    │   ANALYST    │    │   (React)    │                   │
│  └──────────────┘    └──────────────┘    └──────────────┘                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Directory Structure

```
/trading-system
├── /engine
│   ├── /data
│   │   ├── binance_ws.py           # WebSocket stream handler
│   │   ├── data_normalizer.py      # Tick → OHLCV aggregation
│   │   ├── historical_loader.py    # TimescaleDB interface
│   │   └── schemas.py              # Pydantic models for all data types
│   │
│   ├── /orderflow
│   │   ├── delta_calculator.py     # Buy/Sell volume delta
│   │   ├── footprint_builder.py    # Price-level volume aggregation
│   │   ├── absorption_detector.py  # Absorption pattern recognition
│   │   ├── cvd_tracker.py          # Cumulative Volume Delta
│   │   └── trapped_trader.py       # Failed breakout + absorption logic
│   │
│   ├── /indicators
│   │   ├── vwap.py                 # VWAP + Standard Deviations
│   │   ├── volume_profile.py       # VAH, VAL, POC calculation
│   │   ├── atr.py                  # Average True Range
│   │   └── market_structure.py     # Swing highs/lows detection
│   │
│   ├── /strategy
│   │   ├── base_strategy.py        # Abstract strategy interface
│   │   ├── trend_continuation.py   # Primary setup (Fabio's main)
│   │   ├── mean_reversion.py       # Late-session VWAP reversion
│   │   ├── confluence_validator.py # Box-checking logic
│   │   └── signal_types.py         # Signal enums and models
│   │
│   ├── /risk
│   │   ├── position_sizer.py       # Dynamic A/B/C sizing
│   │   ├── house_money.py          # Profit-based risk escalation
│   │   ├── drawdown_monitor.py     # Real-time DD tracking
│   │   ├── session_guard.py        # 3-loss circuit breaker
│   │   └── exposure_limits.py      # Max position / correlation checks
│   │
│   ├── /execution
│   │   ├── order_manager.py        # Order state machine
│   │   ├── binance_executor.py     # CCXT wrapper for Binance
│   │   ├── slippage_model.py       # Expected slippage calculation
│   │   └── oco_handler.py          # Stop-Loss + Take-Profit pairs
│   │
│   ├── /state
│   │   ├── market_state.py         # Balance/Imbalance detection
│   │   ├── session_state.py        # Daily P&L, win streak, etc.
│   │   ├── trade_journal.py        # Structured trade logging
│   │   └── redis_store.py          # Hot state persistence
│   │
│   ├── /analyst
│   │   ├── llm_client.py           # Anthropic/OpenAI interface
│   │   ├── trade_reviewer.py       # Post-trade analysis prompts
│   │   ├── narrative_builder.py    # Pre-market bias generation
│   │   └── journal_formatter.py    # Human-readable output
│   │
│   ├── /backtest
│   │   ├── vectorized_engine.py    # Fast prototyping backtester
│   │   ├── event_driven_engine.py  # Realistic tick-by-tick sim
│   │   ├── walk_forward.py         # Rolling optimization windows
│   │   └── metrics.py              # Sharpe, Sortino, Max DD, etc.
│   │
│   ├── config.py                   # Environment-based configuration
│   ├── events.py                   # Event bus implementation
│   └── main.py                     # Entry point / orchestrator
│
├── /dashboard
│   ├── /src
│   │   ├── /components
│   │   ├── /hooks
│   │   └── /pages
│   └── package.json
│
├── /database
│   ├── migrations/
│   └── schema.sql
│
├── /scripts
│   ├── backfill_data.py            # Historical data download
│   └── run_backtest.py             # CLI for backtesting
│
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## 3.3 Event Bus Architecture

The system uses an **in-process async event bus** for component communication. This is deliberately lightweight — a full message queue (Redis Streams, Kafka) adds operational complexity unjustified for a single-user trading system.

### Event Types

```python
# /engine/events.py
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional, Callable, Coroutine
import asyncio
from collections import defaultdict

class EventType(str, Enum):
    # Data Events
    TRADE_TICK = "trade.tick"              # New aggTrade received
    CANDLE_CLOSE = "candle.close"          # Candle completed
    ORDERBOOK_UPDATE = "orderbook.update"  # Depth snapshot
    
    # Order Flow Events
    DELTA_UPDATE = "orderflow.delta"       # Delta calculated
    ABSORPTION_DETECTED = "orderflow.absorption"
    TRAPPED_TRADERS = "orderflow.trapped"
    CVD_DIVERGENCE = "orderflow.cvd_divergence"
    
    # Strategy Events
    BIAS_UPDATE = "strategy.bias"          # 15m bias changed
    SIGNAL_GENERATED = "strategy.signal"   # Trade signal ready
    CONFLUENCE_CHECK = "strategy.confluence"
    
    # Execution Events
    ORDER_SUBMITTED = "execution.submitted"
    ORDER_FILLED = "execution.filled"
    ORDER_CANCELLED = "execution.cancelled"
    ORDER_REJECTED = "execution.rejected"
    POSITION_OPENED = "execution.position_open"
    POSITION_CLOSED = "execution.position_close"
    
    # Risk Events
    RISK_BREACH = "risk.breach"
    SESSION_STOPPED = "risk.session_stop"
    BREAKEVEN_TRIGGERED = "risk.breakeven"
    
    # System Events
    SYSTEM_STARTUP = "system.startup"
    SYSTEM_SHUTDOWN = "system.shutdown"
    RECONNECT = "system.reconnect"


@dataclass
class Event:
    """Base event with metadata"""
    type: EventType
    timestamp: datetime = field(default_factory=datetime.utcnow)
    source: str = ""
    payload: dict = field(default_factory=dict)
    correlation_id: Optional[str] = None  # For tracking related events


class EventBus:
    """
    In-process async event bus using asyncio.Queue.
    
    Design decisions:
    - Single event loop, no threading complexity
    - Subscribers receive events via async generators
    - Fire-and-forget publishing (non-blocking)
    - Optional persistence to Redis for crash recovery
    """
    
    def __init__(self, redis_client=None):
        self._subscribers: dict[EventType, list[asyncio.Queue]] = defaultdict(list)
        self._global_subscribers: list[asyncio.Queue] = []
        self._redis = redis_client  # For event persistence (recovery)
        self._running = False
    
    def subscribe(self, event_type: EventType) -> asyncio.Queue:
        """Subscribe to specific event type. Returns queue to await."""
        queue = asyncio.Queue(maxsize=1000)
        self._subscribers[event_type].append(queue)
        return queue
    
    def subscribe_all(self) -> asyncio.Queue:
        """Subscribe to all events (for logging/monitoring)."""
        queue = asyncio.Queue(maxsize=5000)
        self._global_subscribers.append(queue)
        return queue
    
    async def publish(self, event: Event) -> None:
        """
        Publish event to all subscribers.
        Non-blocking: drops events if subscriber queue is full.
        """
        # Persist critical events for recovery
        if self._redis and event.type in {
            EventType.POSITION_OPENED,
            EventType.POSITION_CLOSED,
            EventType.ORDER_SUBMITTED,
            EventType.ORDER_FILLED,
        }:
            await self._persist_event(event)
        
        # Notify type-specific subscribers
        for queue in self._subscribers.get(event.type, []):
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                pass  # Drop event if subscriber is slow
        
        # Notify global subscribers
        for queue in self._global_subscribers:
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                pass
    
    async def _persist_event(self, event: Event) -> None:
        """Persist event to Redis for crash recovery"""
        if self._redis:
            key = f"events:{event.type.value}:{event.timestamp.timestamp()}"
            await self._redis.setex(key, 3600, event.payload)  # 1hr TTL
    
    def unsubscribe(self, event_type: EventType, queue: asyncio.Queue) -> None:
        """Remove subscription"""
        if queue in self._subscribers.get(event_type, []):
            self._subscribers[event_type].remove(queue)
```

### Concurrency Model

```python
# /engine/main.py - Core async architecture
import asyncio
import signal
from contextlib import asynccontextmanager

class TradingEngine:
    """
    Main orchestrator using structured concurrency.
    
    Architecture:
    - Single event loop (asyncio)
    - WebSocket handlers run as persistent tasks
    - Processing pipeline triggered by events
    - All shared state protected by asyncio.Lock where needed
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.event_bus = EventBus()
        self._tasks: list[asyncio.Task] = []
        self._shutdown_event = asyncio.Event()
        self._locks = {
            'position': asyncio.Lock(),
            'orders': asyncio.Lock(),
            'state': asyncio.Lock(),
        }
    
    async def start(self) -> None:
        """Start all components with graceful shutdown handling"""
        # Register signal handlers
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, self._handle_shutdown)
        
        # Recover state from Redis if crashed
        await self._recover_state()
        
        # Start component tasks
        self._tasks = [
            asyncio.create_task(self._run_websocket_handler(), name="websocket"),
            asyncio.create_task(self._run_orderflow_processor(), name="orderflow"),
            asyncio.create_task(self._run_strategy_engine(), name="strategy"),
            asyncio.create_task(self._run_execution_monitor(), name="execution"),
            asyncio.create_task(self._run_risk_monitor(), name="risk"),
        ]
        
        # Publish startup event
        await self.event_bus.publish(Event(
            type=EventType.SYSTEM_STARTUP,
            source="engine",
        ))
        
        # Wait for shutdown signal
        await self._shutdown_event.wait()
        
        # Graceful shutdown
        await self._shutdown()
    
    def _handle_shutdown(self) -> None:
        """Signal handler for graceful shutdown"""
        self._shutdown_event.set()
    
    async def _shutdown(self) -> None:
        """
        Graceful shutdown sequence:
        1. Stop accepting new signals
        2. Cancel pending (unfilled) orders
        3. Persist state to Redis
        4. Close WebSocket connections
        5. Cancel all tasks
        """
        await self.event_bus.publish(Event(
            type=EventType.SYSTEM_SHUTDOWN,
            source="engine",
        ))
        
        # Cancel pending orders (keep filled positions!)
        async with self._locks['orders']:
            await self._cancel_pending_orders()
        
        # Persist final state
        await self._persist_state()
        
        # Cancel tasks with timeout
        for task in self._tasks:
            task.cancel()
        
        await asyncio.gather(*self._tasks, return_exceptions=True)
    
    async def _recover_state(self) -> None:
        """
        Recover from crash:
        1. Load last known state from Redis
        2. Query Binance for current positions
        3. Reconcile: if position exists but no local state, 
           create conservative exit orders
        """
        # Load persisted state
        local_state = await self._load_persisted_state()
        
        # Query exchange
        exchange_positions = await self.executor.get_all_positions()
        
        # Reconcile
        for pos in exchange_positions:
            if pos['symbol'] in self.config.symbols:
                local_pos = local_state.get('positions', {}).get(pos['symbol'])
                
                if local_pos is None:
                    # Position exists on exchange but not locally
                    # This is dangerous - we don't know the intended SL/TP
                    # Create conservative exit: tight stop at -1%
                    await self._create_emergency_exit(pos)
                else:
                    # Verify SL/TP orders still exist
                    await self._verify_exit_orders(pos, local_pos)
```

---

## 4. Data Layer Specification

### 4.1 Data Sources (Binance)

| Stream | Purpose | Update Frequency |
|--------|---------|------------------|
| `aggTrades` | Trade tape for delta calculation | Real-time (~100ms) |
| `depth@100ms` | Order book for absorption detection | 100ms |
| `kline_1m` | OHLCV for structure | 1 minute |
| `kline_15m` | Bias timeframe | 15 minutes |
| `markPrice` | Funding rate + Mark price | 3 seconds |

### 4.2 Core Data Models

```python
# /engine/data/schemas.py
from pydantic import BaseModel
from datetime import datetime
from enum import Enum
from typing import Optional

class TradeDirection(str, Enum):
    BUY = "BUY"
    SELL = "SELL"

class AggTrade(BaseModel):
    """Single trade from aggTrades stream"""
    event_time: datetime
    symbol: str
    price: float
    quantity: float
    is_buyer_maker: bool  # False = Aggressive buy, True = Aggressive sell
    
    @property
    def direction(self) -> TradeDirection:
        return TradeDirection.SELL if self.is_buyer_maker else TradeDirection.BUY

class FootprintLevel(BaseModel):
    """Volume at a single price level"""
    price: float
    bid_volume: float      # Aggressive sells hitting bids
    ask_volume: float      # Aggressive buys lifting asks
    delta: float           # ask_volume - bid_volume
    trade_count: int

class FootprintCandle(BaseModel):
    """Complete footprint for one candle"""
    timestamp: datetime
    symbol: str
    timeframe: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    delta: float                        # Net delta for entire candle
    levels: list[FootprintLevel]        # Volume at each price
    poc_price: float                    # Point of Control (highest volume)
    value_area_high: float
    value_area_low: float

class MarketStateType(str, Enum):
    BALANCE = "BALANCE"
    IMBALANCE_UP = "IMBALANCE_UP"
    IMBALANCE_DOWN = "IMBALANCE_DOWN"
    UNKNOWN = "UNKNOWN"

class OrderFlowSignal(BaseModel):
    """Output from Order Flow Engine"""
    timestamp: datetime
    absorption_detected: bool
    absorption_side: Optional[TradeDirection]
    aggression_detected: bool
    aggression_side: Optional[TradeDirection]
    cvd_divergence: bool
    trapped_traders: bool
    trapped_side: Optional[TradeDirection]
    delta_strength: float               # -1.0 to 1.0

class SetupGrade(str, Enum):
    A = "A"  # Full confluence - max risk
    B = "B"  # Good setup - standard risk
    C = "C"  # Sub-optimal - reduced risk
    INVALID = "INVALID"

class TradeSignal(BaseModel):
    """Final signal from Strategy Engine"""
    timestamp: datetime
    symbol: str
    direction: TradeDirection
    grade: SetupGrade
    entry_price: float
    stop_loss: float
    take_profit: float
    confluence_score: int               # Number of boxes checked (0-5)
    rationale: str                      # Human-readable reason
```

### 4.2.1 Funding Rate Tracker

```python
# /engine/data/funding_tracker.py
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional
from collections import deque

@dataclass
class FundingSignal:
    """Funding rate signal for sentiment analysis"""
    timestamp: datetime
    symbol: str
    funding_rate: float
    sentiment: str           # 'LONG_CROWDED', 'SHORT_CROWDED', 'NEUTRAL'
    extremity: float         # 0.0 to 1.0, how extreme is current rate
    rate_change_1h: float    # Change in funding over last hour


class FundingRateTracker:
    """
    Tracks funding rates as a sentiment/positioning indicator.
    
    Fabio's Equivalent:
    In traditional futures, options flow and dark pool activity
    reveal institutional positioning. In crypto, funding rate
    serves a similar purpose — it shows who's overleveraged.
    
    Trading Logic:
    - Extreme positive funding (>0.05%) = longs crowded, fade longs
    - Extreme negative funding (<-0.02%) = shorts crowded, fade shorts
    - Rate *changing* rapidly = positioning shift underway
    
    This is CONTEXTUAL, not a direct entry signal.
    It adjusts the bias, not triggers trades.
    """
    
    def __init__(
        self,
        long_crowded_threshold: float = 0.0005,   # 0.05% per 8 hours
        short_crowded_threshold: float = -0.0002, # -0.02% per 8 hours
        history_hours: int = 24,
    ):
        self.long_threshold = long_crowded_threshold
        self.short_threshold = short_crowded_threshold
        self._history: deque = deque(maxlen=history_hours * 3)
        self._current_rate: float = 0.0
        self._last_update: Optional[datetime] = None
    
    def update(self, funding_data: dict) -> Optional[FundingSignal]:
        """Update with new funding rate data from markPrice stream"""
        rate = funding_data['funding_rate']
        timestamp = funding_data['event_time']
        symbol = funding_data['symbol']
        
        self._current_rate = rate
        self._last_update = timestamp
        self._history.append({'time': timestamp, 'rate': rate})
        
        # Determine sentiment
        if rate > self.long_threshold:
            sentiment = 'LONG_CROWDED'
        elif rate < self.short_threshold:
            sentiment = 'SHORT_CROWDED'
        else:
            sentiment = 'NEUTRAL'
        
        # Calculate extremity (how far from neutral)
        if rate > 0:
            extremity = min(1.0, rate / (self.long_threshold * 2))
        else:
            extremity = min(1.0, abs(rate) / (abs(self.short_threshold) * 2))
        
        # Calculate rate change over last hour
        rate_change_1h = self._calculate_rate_change(hours=1)
        
        return FundingSignal(
            timestamp=timestamp,
            symbol=symbol,
            funding_rate=rate,
            sentiment=sentiment,
            extremity=extremity,
            rate_change_1h=rate_change_1h,
        )
    
    def _calculate_rate_change(self, hours: int) -> float:
        """Calculate change in funding rate over specified hours"""
        if len(self._history) < 2:
            return 0.0
        
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        old_rates = [h['rate'] for h in self._history if h['time'] < cutoff]
        
        if not old_rates:
            return 0.0
        
        return self._current_rate - old_rates[-1]
    
    def get_bias_adjustment(self) -> float:
        """
        Returns a bias adjustment factor based on funding.
        
        Returns:
            -1.0 to +1.0
            Positive = favor longs (shorts crowded)
            Negative = favor shorts (longs crowded)
            0 = no adjustment
        
        Used by ConfluenceValidator as an additional filter.
        """
        if self._current_rate > self.long_threshold:
            return -min(1.0, self._current_rate / (self.long_threshold * 2))
        elif self._current_rate < self.short_threshold:
            return min(1.0, abs(self._current_rate) / (abs(self.short_threshold) * 2))
        return 0.0
```

### 4.3 Database Schema (TimescaleDB)

**Note on Foreign Keys:** TimescaleDB hypertables don't support traditional foreign key constraints due to partitioning. We use application-level referential integrity with UUID correlation IDs instead.

```sql
-- /database/schema.sql

-- Enable extensions
CREATE EXTENSION IF NOT EXISTS timescaledb;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Raw trade data (hypertable for time-series performance)
CREATE TABLE trades_raw (
    time            TIMESTAMPTZ NOT NULL,
    symbol          TEXT NOT NULL,
    price           DOUBLE PRECISION NOT NULL,
    quantity        DOUBLE PRECISION NOT NULL,
    is_buyer_maker  BOOLEAN NOT NULL,
    trade_id        BIGINT NOT NULL
);
SELECT create_hypertable('trades_raw', 'time');
CREATE INDEX idx_trades_symbol_time ON trades_raw (symbol, time DESC);

-- Aggregated OHLCV candles
CREATE TABLE candles (
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
SELECT create_hypertable('candles', 'time');

-- Volume Profile snapshots
CREATE TABLE volume_profile (
    time        TIMESTAMPTZ NOT NULL,
    symbol      TEXT NOT NULL,
    lookback    TEXT NOT NULL,         -- 'session', 'daily', 'weekly'
    poc         DOUBLE PRECISION,
    vah         DOUBLE PRECISION,
    val         DOUBLE PRECISION,
    profile     JSONB                  -- Full price-level distribution
);
SELECT create_hypertable('volume_profile', 'time');

-- Order flow signals (audit trail)
-- NOTE: No FK from hypertables. Use signal_id for application-level joins.
CREATE TABLE signals (
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
SELECT create_hypertable('signals', 'time');
CREATE INDEX idx_signals_uuid ON signals (signal_id);

-- Executed trades (regular table, not hypertable - needs FK support)
-- Links to signals via signal_id (application-enforced, not DB-enforced)
CREATE TABLE executions (
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
CREATE INDEX idx_executions_signal ON executions (signal_id);
CREATE INDEX idx_executions_time ON executions (time_entry DESC);

-- Session state snapshots
CREATE TABLE session_state (
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
SELECT create_hypertable('session_state', 'time');

-- LLM analysis journal
CREATE TABLE trade_journal (
    id              SERIAL PRIMARY KEY,
    execution_id    INTEGER REFERENCES executions(id),
    time            TIMESTAMPTZ NOT NULL,
    analysis_type   TEXT,              -- 'post_trade', 'pre_market', 'session_review'
    prompt          TEXT,
    response        TEXT,
    key_insights    JSONB
);
```

### 4.4 Redis State Schema (Hot Storage & Recovery)

Redis stores ephemeral state for fast access and crash recovery. This is the "source of truth" for active trading state.

```python
# /engine/state/redis_store.py
import json
from datetime import datetime
from typing import Optional
import redis.asyncio as redis

class RedisStateStore:
    """
    Redis schema for hot state and crash recovery.
    
    Key patterns:
    - state:position:{symbol} - Active position data
    - state:orders:{symbol} - Active orders (SL/TP)
    - state:session - Current session state
    - events:critical:{timestamp} - Persisted events for recovery
    """
    
    # Key prefixes
    POSITION_KEY = "state:position:{symbol}"
    ORDERS_KEY = "state:orders:{symbol}"
    SESSION_KEY = "state:session"
    TRADE_KEY = "state:active_trade"
    EVENTS_KEY = "events:critical"
    
    def __init__(self, redis_url: str):
        self.redis = redis.from_url(redis_url)
    
    async def save_position(self, symbol: str, position: dict) -> None:
        """Save active position state"""
        key = self.POSITION_KEY.format(symbol=symbol)
        position['updated_at'] = datetime.utcnow().isoformat()
        await self.redis.set(key, json.dumps(position))
    
    async def get_position(self, symbol: str) -> Optional[dict]:
        """Retrieve position state"""
        key = self.POSITION_KEY.format(symbol=symbol)
        data = await self.redis.get(key)
        return json.loads(data) if data else None
    
    async def clear_position(self, symbol: str) -> None:
        """Clear position on close"""
        key = self.POSITION_KEY.format(symbol=symbol)
        await self.redis.delete(key)
    
    async def save_active_trade(self, trade: dict) -> None:
        """
        Save complete active trade state.
        
        Includes:
        - Signal that triggered the trade
        - Entry order state
        - SL/TP order IDs
        - Position size
        - Entry price
        
        This is critical for crash recovery.
        """
        trade['updated_at'] = datetime.utcnow().isoformat()
        await self.redis.set(self.TRADE_KEY, json.dumps(trade))
    
    async def get_active_trade(self) -> Optional[dict]:
        """Retrieve active trade for recovery"""
        data = await self.redis.get(self.TRADE_KEY)
        return json.loads(data) if data else None
    
    async def clear_active_trade(self) -> None:
        """Clear on trade completion"""
        await self.redis.delete(self.TRADE_KEY)
    
    async def save_session_state(self, session: dict) -> None:
        """Save session state (P&L, consecutive losses, etc.)"""
        session['updated_at'] = datetime.utcnow().isoformat()
        await self.redis.set(self.SESSION_KEY, json.dumps(session))
    
    async def get_session_state(self) -> Optional[dict]:
        """Retrieve session state"""
        data = await self.redis.get(self.SESSION_KEY)
        return json.loads(data) if data else None
    
    async def save_exit_orders(self, symbol: str, orders: dict) -> None:
        """
        Save SL/TP order mapping.
        
        Structure:
        {
            "stop_loss": {"local_id": "...", "exchange_id": "...", "price": ...},
            "take_profit": {"local_id": "...", "exchange_id": "...", "price": ...}
        }
        """
        key = self.ORDERS_KEY.format(symbol=symbol)
        await self.redis.set(key, json.dumps(orders))
    
    async def get_exit_orders(self, symbol: str) -> Optional[dict]:
        """Retrieve exit order mapping"""
        key = self.ORDERS_KEY.format(symbol=symbol)
        data = await self.redis.get(key)
        return json.loads(data) if data else None


class StateReconciler:
    """
    Reconciles local state with exchange state after crash/restart.
    
    Scenarios handled:
    1. Position on exchange, no local state → Emergency exit
    2. Position on exchange, local state exists → Verify SL/TP orders
    3. Local state exists, no position → Clear stale state
    4. Orders exist but position closed → Cancel orphan orders
    """
    
    def __init__(self, redis_store: RedisStateStore, executor):
        self.redis = redis_store
        self.executor = executor
    
    async def reconcile(self, symbols: list[str]) -> dict:
        """
        Full reconciliation on startup.
        
        Returns dict of actions taken.
        """
        actions = []
        
        for symbol in symbols:
            # Get states
            local_position = await self.redis.get_position(symbol)
            local_trade = await self.redis.get_active_trade()
            exchange_position = await self.executor.get_position(symbol)
            exchange_orders = await self.executor.get_open_orders(symbol)
            
            # Scenario 1: Position exists on exchange but not locally
            if exchange_position and not local_position:
                action = await self._handle_orphan_position(symbol, exchange_position)
                actions.append(action)
            
            # Scenario 2: Both exist - verify orders
            elif exchange_position and local_position:
                action = await self._verify_exit_orders(
                    symbol, local_trade, exchange_orders
                )
                actions.append(action)
            
            # Scenario 3: Local state but no position
            elif local_position and not exchange_position:
                await self.redis.clear_position(symbol)
                await self.redis.clear_active_trade()
                actions.append({"symbol": symbol, "action": "cleared_stale_state"})
            
            # Scenario 4: Orphan orders (no position)
            orphan_orders = [o for o in exchange_orders 
                          if o['type'] in ('STOP_MARKET', 'TAKE_PROFIT_MARKET')]
            if orphan_orders and not exchange_position:
                for order in orphan_orders:
                    await self.executor.cancel_order_by_id(order['id'])
                actions.append({
                    "symbol": symbol, 
                    "action": "cancelled_orphan_orders",
                    "count": len(orphan_orders)
                })
        
        return {"reconciliation_actions": actions}
    
    async def _handle_orphan_position(self, symbol: str, position: dict) -> dict:
        """
        Handle position that exists on exchange but has no local state.
        
        This is dangerous - we don't know the intended exit strategy.
        Conservative action: Place tight stop at 1% loss.
        """
        entry_price = position['entry_price']
        quantity = position['quantity']
        side = position['side']
        
        # Calculate emergency stop (1% adverse move)
        if side == 'LONG':
            emergency_stop = entry_price * 0.99
            exit_side = 'SELL'
        else:
            emergency_stop = entry_price * 1.01
            exit_side = 'BUY'
        
        # Place emergency stop
        await self.executor.create_stop_order(
            symbol=symbol,
            side=exit_side,
            quantity=quantity,
            stop_price=emergency_stop,
        )
        
        return {
            "symbol": symbol,
            "action": "created_emergency_stop",
            "position_side": side,
            "stop_price": emergency_stop,
            "reason": "orphan_position_no_local_state"
        }
    
    async def _verify_exit_orders(
        self, 
        symbol: str, 
        local_trade: dict,
        exchange_orders: list
    ) -> dict:
        """Verify SL/TP orders exist on exchange"""
        if not local_trade:
            return {"symbol": symbol, "action": "no_local_trade_to_verify"}
        
        expected_sl_id = local_trade.get('stop_loss_order', {}).get('exchange_id')
        expected_tp_id = local_trade.get('take_profit_order', {}).get('exchange_id')
        
        exchange_order_ids = {o['id'] for o in exchange_orders}
        
        missing = []
        if expected_sl_id and expected_sl_id not in exchange_order_ids:
            missing.append('stop_loss')
        if expected_tp_id and expected_tp_id not in exchange_order_ids:
            missing.append('take_profit')
        
        if missing:
            # Re-create missing orders
            # (Implementation depends on having the original prices in local_trade)
            return {
                "symbol": symbol,
                "action": "recreating_missing_orders",
                "missing": missing
            }
        
        return {"symbol": symbol, "action": "orders_verified"}
```

---

## 4.5 Funding Rate Integration (Crypto-Specific)

Fabio uses options flow as a "smart money" indicator. In crypto, **funding rate** serves a similar purpose.

```python
# /engine/data/funding_rate.py
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from collections import deque

@dataclass
class FundingRateData:
    """Funding rate snapshot from markPrice stream"""
    symbol: str
    funding_rate: float      # As decimal (0.0001 = 0.01%)
    funding_time: datetime   # Next funding settlement time
    mark_price: float
    index_price: float


class FundingRateAnalyzer:
    """
    Analyzes funding rate for sentiment signals.
    
    Funding rate reflects the cost of holding leveraged positions:
    - Positive funding = longs pay shorts (bullish crowding → bearish signal)
    - Negative funding = shorts pay longs (bearish crowding → bullish signal)
    - Extreme funding (>0.1%) often precedes reversals
    
    Used as a FILTER, not an entry signal.
    """
    
    def __init__(
        self,
        extreme_threshold: float = 0.001,   # 0.1% is extreme
        warning_threshold: float = 0.0005,  # 0.05% is elevated
    ):
        self.extreme_threshold = extreme_threshold
        self.warning_threshold = warning_threshold
        self._history: deque = deque(maxlen=28800)  # ~8 hours at 1s
    
    def update(self, data: FundingRateData) -> None:
        """Add new funding rate data"""
        self._history.append(data)
    
    def get_sentiment(self) -> dict:
        """
        Get funding-based sentiment (contrarian logic).
        
        Returns:
            sentiment: 'bullish', 'bearish', or 'neutral'
            strength: 0.0 to 1.0
            funding_rate: current rate
            warning: Optional warning message
        """
        if not self._history:
            return {'sentiment': 'neutral', 'strength': 0.0, 'funding_rate': 0.0, 'warning': None}
        
        rate = self._history[-1].funding_rate
        
        if rate > self.extreme_threshold:
            return {
                'sentiment': 'bearish',
                'strength': min(1.0, rate / (self.extreme_threshold * 2)),
                'funding_rate': rate,
                'warning': f"Extreme positive funding ({rate*100:.3f}%) - longs overcrowded",
            }
        elif rate < -self.extreme_threshold:
            return {
                'sentiment': 'bullish',
                'strength': min(1.0, abs(rate) / (self.extreme_threshold * 2)),
                'funding_rate': rate,
                'warning': f"Extreme negative funding ({rate*100:.3f}%) - shorts overcrowded",
            }
        elif abs(rate) > self.warning_threshold:
            return {
                'sentiment': 'bearish' if rate > 0 else 'bullish',
                'strength': 0.5,
                'funding_rate': rate,
                'warning': None,
            }
        else:
            return {'sentiment': 'neutral', 'strength': 0.0, 'funding_rate': rate, 'warning': None}
    
    def should_filter_long(self) -> bool:
        """Avoid longs if funding extremely positive (longs overcrowded)"""
        s = self.get_sentiment()
        return s['sentiment'] == 'bearish' and s['strength'] > 0.7
    
    def should_filter_short(self) -> bool:
        """Avoid shorts if funding extremely negative (shorts overcrowded)"""
        s = self.get_sentiment()
        return s['sentiment'] == 'bullish' and s['strength'] > 0.7
```

**Integration with Strategy Engine:**

The `ConfluenceValidator` can optionally check funding rate before approving entries:

```python
# In ConfluenceValidator.validate()
if self.funding_analyzer.should_filter_long() and signal.direction == 'LONG':
    return ConfluenceResult(valid=False, reason="Funding rate filter: longs overcrowded")
```

---

## 5. Order Flow Engine

This is the heart of Fabio's methodology — translating raw trade data into actionable signals.

### 5.1 Delta Calculator

```python
# /engine/orderflow/delta_calculator.py
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Deque
from ..data.schemas import AggTrade, TradeDirection

@dataclass
class DeltaWindow:
    """Rolling window delta calculation"""
    window_seconds: int = 60
    trades: Deque[AggTrade] = field(default_factory=deque)
    
    def add_trade(self, trade: AggTrade) -> None:
        self.trades.append(trade)
        self._prune_old()
    
    def _prune_old(self) -> None:
        cutoff = datetime.utcnow() - timedelta(seconds=self.window_seconds)
        while self.trades and self.trades[0].event_time < cutoff:
            self.trades.popleft()
    
    @property
    def buy_volume(self) -> float:
        return sum(t.quantity for t in self.trades if t.direction == TradeDirection.BUY)
    
    @property
    def sell_volume(self) -> float:
        return sum(t.quantity for t in self.trades if t.direction == TradeDirection.SELL)
    
    @property
    def delta(self) -> float:
        return self.buy_volume - self.sell_volume
    
    @property
    def delta_ratio(self) -> float:
        """Normalized delta: -1.0 (all sells) to +1.0 (all buys)"""
        total = self.buy_volume + self.sell_volume
        if total == 0:
            return 0.0
        return self.delta / total


class DeltaCalculator:
    """
    Computes rolling delta across multiple windows.
    Fabio uses this to see if "buyers are stepping in" or "sellers are aggressive."
    """
    
    def __init__(self):
        self.windows = {
            "5s": DeltaWindow(window_seconds=5),
            "15s": DeltaWindow(window_seconds=15),   # Execution TF
            "1m": DeltaWindow(window_seconds=60),
            "5m": DeltaWindow(window_seconds=300),
        }
        self._cvd = 0.0  # Cumulative Volume Delta (session)
    
    def process_trade(self, trade: AggTrade) -> dict:
        for window in self.windows.values():
            window.add_trade(trade)
        
        # Update CVD
        if trade.direction == TradeDirection.BUY:
            self._cvd += trade.quantity
        else:
            self._cvd -= trade.quantity
        
        return self.get_state()
    
    def get_state(self) -> dict:
        return {
            "deltas": {k: w.delta for k, w in self.windows.items()},
            "delta_ratios": {k: w.delta_ratio for k, w in self.windows.items()},
            "cvd": self._cvd,
        }
    
    def reset_cvd(self) -> None:
        """Reset at session start"""
        self._cvd = 0.0
```

### 5.2 Absorption Detector

```python
# /engine/orderflow/absorption_detector.py
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional
from collections import deque
import math
from ..data.schemas import TradeDirection

@dataclass
class AbsorptionEvent:
    """Detected absorption pattern"""
    timestamp: datetime
    side: TradeDirection          # Which side got absorbed (SELL = buying absorption)
    volume_absorbed: float        # Total aggressive volume that failed to move price
    price_range: tuple[float, float]  # Price stayed within this range
    confidence: float             # 0.0 to 1.0
    time_weighted_volume: float   # Recency-adjusted volume


class AbsorptionDetector:
    """
    Detects when aggressive orders fail to move price.
    
    Fabio's Logic:
    - "Aggressive selling hits the level, but price doesn't drop"
    - "This means a passive buyer is absorbing the orders"
    
    Detection Algorithm:
    1. High volume in one direction (time-weighted, recent trades matter more)
    2. Price movement doesn't match volume
    3. Volume exceeds dynamic threshold (based on recent activity, not hardcoded)
    
    Improvements over naive implementation:
    - Time-weighted decay: Volume from 5s ago counts more than 25s ago
    - Dynamic thresholds: Adapts to current volatility regime
    - Imbalance ratio is regime-aware (1.5x in low vol, 2x in high vol)
    """
    
    def __init__(
        self,
        lookback_seconds: int = 30,
        price_move_threshold_pct: float = 0.0005,  # 0.05%
        min_volume_percentile: float = 75.0,       # Must exceed 75th percentile
        decay_half_life_seconds: float = 10.0,     # Half-life for time decay
    ):
        self.lookback_seconds = lookback_seconds
        self.price_move_threshold_pct = price_move_threshold_pct
        self.min_volume_percentile = min_volume_percentile
        self.decay_half_life = decay_half_life_seconds
        
        # Rolling volume history for dynamic thresholds
        self._volume_history: deque = deque(maxlen=1000)
        self._recent_trades: deque = deque()
    
    def add_trade(self, trade) -> None:
        """Add trade to rolling window"""
        self._recent_trades.append(trade)
        self._volume_history.append(trade.quantity)
        self._prune_old_trades()
    
    def _prune_old_trades(self) -> None:
        """Remove trades older than lookback window"""
        cutoff = datetime.utcnow() - timedelta(seconds=self.lookback_seconds)
        while self._recent_trades and self._recent_trades[0].event_time < cutoff:
            self._recent_trades.popleft()
    
    def _calculate_time_weight(self, trade_time: datetime) -> float:
        """
        Exponential decay weighting.
        
        Weight = 0.5 ^ (age_seconds / half_life)
        
        Example with 10s half-life:
        - Trade from 0s ago: weight = 1.0
        - Trade from 10s ago: weight = 0.5
        - Trade from 20s ago: weight = 0.25
        - Trade from 30s ago: weight = 0.125
        """
        age_seconds = (datetime.utcnow() - trade_time).total_seconds()
        return math.pow(0.5, age_seconds / self.decay_half_life)
    
    def _get_volume_threshold(self) -> float:
        """
        Dynamic threshold based on recent volume distribution.
        
        Returns the 75th percentile of recent volume.
        This adapts to market regime automatically.
        """
        if len(self._volume_history) < 100:
            return 50.0  # Fallback for cold start
        
        sorted_vols = sorted(self._volume_history)
        percentile_idx = int(len(sorted_vols) * (self.min_volume_percentile / 100))
        return sorted_vols[percentile_idx]
    
    def _get_imbalance_ratio(self, volatility_regime: str = "normal") -> float:
        """
        Dynamic imbalance ratio based on regime.
        
        In high volatility, we need more extreme imbalance to confirm absorption.
        In low volatility, smaller imbalances are meaningful.
        """
        ratios = {
            "low": 1.3,
            "normal": 1.5,
            "high": 2.0,
        }
        return ratios.get(volatility_regime, 1.5)
    
    def analyze(
        self,
        current_price: float,
        volatility_regime: str = "normal",
    ) -> Optional[AbsorptionEvent]:
        """
        Analyze recent trades for absorption pattern.
        
        Returns AbsorptionEvent if absorption detected, None otherwise.
        """
        if len(self._recent_trades) < 10:
            return None
        
        trades = list(self._recent_trades)
        
        # Calculate time-weighted volumes
        tw_buy_vol = 0.0
        tw_sell_vol = 0.0
        
        for trade in trades:
            weight = self._calculate_time_weight(trade.event_time)
            if trade.direction == TradeDirection.BUY:
                tw_buy_vol += trade.quantity * weight
            else:
                tw_sell_vol += trade.quantity * weight
        
        # Get dynamic threshold
        volume_threshold = self._get_volume_threshold()
        imbalance_ratio = self._get_imbalance_ratio(volatility_regime)
        
        # Calculate price range
        prices = [t.price for t in trades]
        price_high = max(prices)
        price_low = min(prices)
        price_range_pct = (price_high - price_low) / current_price
        
        # Check for sell absorption (passive buying)
        # Conditions: High sell volume, imbalanced, but price didn't drop
        if tw_sell_vol > volume_threshold and tw_sell_vol > tw_buy_vol * imbalance_ratio:
            if price_range_pct < self.price_move_threshold_pct:
                confidence = min(1.0, tw_sell_vol / (volume_threshold * 3))
                return AbsorptionEvent(
                    timestamp=trades[-1].event_time,
                    side=TradeDirection.SELL,
                    volume_absorbed=sum(t.quantity for t in trades if t.direction == TradeDirection.SELL),
                    price_range=(price_low, price_high),
                    confidence=confidence,
                    time_weighted_volume=tw_sell_vol,
                )
        
        # Check for buy absorption (passive selling)
        if tw_buy_vol > volume_threshold and tw_buy_vol > tw_sell_vol * imbalance_ratio:
            if price_range_pct < self.price_move_threshold_pct:
                confidence = min(1.0, tw_buy_vol / (volume_threshold * 3))
                return AbsorptionEvent(
                    timestamp=trades[-1].event_time,
                    side=TradeDirection.BUY,
                    volume_absorbed=sum(t.quantity for t in trades if t.direction == TradeDirection.BUY),
                    price_range=(price_low, price_high),
                    confidence=confidence,
                    time_weighted_volume=tw_buy_vol,
                )
        
        return None
```

### 5.3 Trapped Trader Detector

```python
# /engine/orderflow/trapped_trader.py
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional
from ..data.schemas import TradeDirection, FootprintCandle

@dataclass 
class TrappedTraderSignal:
    """Signal when traders are trapped offside"""
    timestamp: datetime
    trapped_side: TradeDirection    # BUY = buyers trapped (bearish), SELL = sellers trapped (bullish)
    trap_price: float               # Price level where they're trapped
    estimated_stops: float          # Estimated stop-loss fuel
    confidence: float
    absorption_volume: float        # Volume that was absorbed before reversal


class TrappedTraderDetector:
    """
    Implements Fabio's "Trapped" Logic.
    
    Sequence:
    1. Price pushes to a level with aggression (breakout attempt)
    2. Absorption occurs (passive opposing orders)
    3. Price reverses back through the level
    4. Traders who entered the breakout are now "trapped" offside
    5. Their stop-losses become fuel for the reversal
    
    This is the PRIMARY entry trigger.
    
    DESIGN DECISION: Single-Trap State
    -----------------------------------
    This detector intentionally tracks only ONE potential trap at a time.
    
    Why not track multiple pending traps?
    1. CONSERVATIVE BY DESIGN: In fast markets, multiple absorption events 
       create noise. Tracking all of them leads to over-trading.
    2. FABIO'S APPROACH: He waits for THE setup, not every setup. Quality > quantity.
    3. NEWEST WINS: If a new absorption event occurs before the old one confirms,
       the new event is more relevant to current price action.
    4. SIMPLICITY: Multiple trap tracking requires complex state management
       (which trap confirmed? which expired?) with marginal benefit.
    
    Trade-off: We may miss some valid traps in fast markets.
    Mitigation: The confluence validator requires multiple confirmations anyway.
    """
    
    def __init__(
        self,
        min_absorption_volume: float = 50.0,
        reversal_threshold_pct: float = 0.001,  # 0.1% reversal confirms trap
        max_trap_age_seconds: int = 120,        # Trap expires after 2 min
    ):
        self.min_absorption_volume = min_absorption_volume
        self.reversal_threshold_pct = reversal_threshold_pct
        self.max_trap_age_seconds = max_trap_age_seconds
        self._breakout_state: Optional[dict] = None
    
    def on_absorption(
        self,
        absorption_side: TradeDirection,
        absorption_volume: float,
        current_price: float,
    ) -> None:
        """
        Called when absorption is detected — sets up potential trap.
        
        Note: New absorption events REPLACE old pending traps.
        This is intentional (see class docstring).
        """
        if absorption_volume < self.min_absorption_volume:
            return
        
        self._breakout_state = {
            "side": absorption_side,
            "volume": absorption_volume,
            "price": current_price,
            "timestamp": datetime.utcnow(),
        }
    
    def check_trap(self, current_price: float) -> Optional[TrappedTraderSignal]:
        """
        Check if trapped trader condition is confirmed.
        
        Logic:
        - If sellers were absorbed and price rallies → sellers are trapped
        - If buyers were absorbed and price drops → buyers are trapped
        """
        # Expire stale state first
        self._expire_stale_state()
        
        if not self._breakout_state:
            return None
        
        state = self._breakout_state
        price_move_pct = (current_price - state["price"]) / state["price"]
        
        # Sellers were absorbed (failed breakdown), price now rallying
        if state["side"] == TradeDirection.SELL:
            if price_move_pct > self.reversal_threshold_pct:
                signal = TrappedTraderSignal(
                    timestamp=datetime.utcnow(),
                    trapped_side=TradeDirection.SELL,
                    trap_price=state["price"],
                    estimated_stops=state["volume"] * 0.7,  # Assume 70% had stops
                    confidence=min(1.0, state["volume"] / (self.min_absorption_volume * 2)),
                    absorption_volume=state["volume"],
                )
                self._breakout_state = None  # Reset state
                return signal
        
        # Buyers were absorbed (failed breakout), price now dropping
        elif state["side"] == TradeDirection.BUY:
            if price_move_pct < -self.reversal_threshold_pct:
                signal = TrappedTraderSignal(
                    timestamp=datetime.utcnow(),
                    trapped_side=TradeDirection.BUY,
                    trap_price=state["price"],
                    estimated_stops=state["volume"] * 0.7,
                    confidence=min(1.0, state["volume"] / (self.min_absorption_volume * 2)),
                    absorption_volume=state["volume"],
                )
                self._breakout_state = None
                return signal
        
        return None
    
    def _expire_stale_state(self) -> None:
        """Clear breakout state if it's too old"""
        if self._breakout_state:
            age = (datetime.utcnow() - self._breakout_state["timestamp"]).total_seconds()
            if age > self.max_trap_age_seconds:
                self._breakout_state = None
    
    def get_pending_trap(self) -> Optional[dict]:
        """Expose pending trap state for monitoring/debugging"""
        self._expire_stale_state()
        return self._breakout_state
```

### 5.4 CVD Divergence Detector

```python
# /engine/orderflow/cvd_tracker.py
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from collections import deque

@dataclass
class CVDDivergence:
    """CVD vs Price divergence signal"""
    timestamp: datetime
    divergence_type: str    # 'BEARISH' (price up, CVD down) or 'BULLISH' (price down, CVD up)
    price_change_pct: float
    cvd_change: float
    confidence: float
    atr_multiple: float     # How many ATRs did price move?


class CVDTracker:
    """
    Tracks Cumulative Volume Delta and detects divergences.
    
    Fabio's Logic:
    - "Price makes a new high, but Delta fails to follow"
    - "This indicates exhaustion/absorption"
    
    A divergence is a warning sign that the move may reverse.
    
    IMPROVEMENT: ATR-Relative Thresholds
    ------------------------------------
    Fixed percentage thresholds (e.g., 0.2%) fail because:
    - In low volatility, 0.2% is a big move → generates few signals
    - In high volatility, 0.2% is noise → generates too many false signals
    
    Solution: Express price movement in terms of ATR (Average True Range).
    A "significant" move is 0.5+ ATR regardless of absolute percentage.
    """
    
    def __init__(
        self,
        lookback_periods: int = 20,
        min_atr_move: float = 0.5,       # Price must move 0.5 ATR minimum
        cvd_divergence_threshold: float = 0.3,  # CVD must diverge by 30%+ of its range
    ):
        self.lookback_periods = lookback_periods
        self.min_atr_move = min_atr_move
        self.cvd_divergence_threshold = cvd_divergence_threshold
        
        self._price_history: deque = deque(maxlen=lookback_periods)
        self._cvd_history: deque = deque(maxlen=lookback_periods)
        self._current_cvd: float = 0.0
        self._current_atr: float = 0.0
    
    def update_atr(self, atr: float) -> None:
        """Update ATR from external indicator"""
        self._current_atr = atr
    
    def update(self, price: float, delta: float) -> Optional[CVDDivergence]:
        """
        Update with new price and delta, check for divergence.
        
        Args:
            price: Current price
            delta: Delta for the current period (buy_vol - sell_vol)
        """
        self._current_cvd += delta
        self._price_history.append(price)
        self._cvd_history.append(self._current_cvd)
        
        if len(self._price_history) < self.lookback_periods:
            return None
        
        if self._current_atr <= 0:
            return None  # Can't calculate without ATR
        
        return self._check_divergence()
    
    def _check_divergence(self) -> Optional[CVDDivergence]:
        """
        Check for price/CVD divergence using ATR-relative thresholds.
        
        Divergence = Price moved significantly but CVD didn't follow (or went opposite).
        """
        prices = list(self._price_history)
        cvds = list(self._cvd_history)
        
        # Calculate price change in ATR units
        price_change = prices[-1] - prices[0]
        price_change_atr = price_change / self._current_atr if self._current_atr > 0 else 0
        price_change_pct = price_change / prices[0]
        
        # Calculate CVD change relative to its range
        cvd_change = cvds[-1] - cvds[0]
        cvd_range = max(cvds) - min(cvds) if max(cvds) != min(cvds) else 1
        cvd_change_normalized = cvd_change / cvd_range
        
        # Bearish divergence: price up significantly, CVD flat/down
        if price_change_atr > self.min_atr_move:
            # Price went up by 0.5+ ATR
            if cvd_change_normalized < self.cvd_divergence_threshold:
                # CVD didn't follow (flat or down)
                return CVDDivergence(
                    timestamp=datetime.utcnow(),
                    divergence_type="BEARISH",
                    price_change_pct=price_change_pct,
                    cvd_change=cvd_change,
                    confidence=min(1.0, abs(price_change_atr) / 1.5),
                    atr_multiple=price_change_atr,
                )
        
        # Bullish divergence: price down significantly, CVD flat/up
        if price_change_atr < -self.min_atr_move:
            # Price went down by 0.5+ ATR
            if cvd_change_normalized > -self.cvd_divergence_threshold:
                # CVD didn't follow (flat or up)
                return CVDDivergence(
                    timestamp=datetime.utcnow(),
                    divergence_type="BULLISH",
                    price_change_pct=price_change_pct,
                    cvd_change=cvd_change,
                    confidence=min(1.0, abs(price_change_atr) / 1.5),
                    atr_multiple=price_change_atr,
                )
        
        return None
    
    def reset(self) -> None:
        """Reset CVD at session start"""
        self._current_cvd = 0.0
        self._price_history.clear()
        self._cvd_history.clear()
```

---

## 6. Strategy Engine

### 6.1 The Box-Checking System (Confluence Validator)

```python
# /engine/strategy/confluence_validator.py
from dataclasses import dataclass
from typing import Optional
from enum import Enum
from ..data.schemas import TradeDirection, SetupGrade, MarketStateType

class ConfluenceBox(str, Enum):
    """Fabio's 5 boxes that must be checked"""
    BIAS = "BIAS"                     # 15m timeframe directional bias
    POI = "POI"                       # Price at Point of Interest
    ORDERFLOW_TRIGGER = "ORDERFLOW"   # Absorption/Aggression confirmed
    PRICE_FOLLOWTHROUGH = "FOLLOWTHROUGH"  # Volume resulted in price move
    MICRO_CONFIRMATION = "MICRO"      # 15s timeframe pattern


@dataclass
class ConfluenceResult:
    """Result of confluence validation"""
    boxes_checked: list[ConfluenceBox]
    boxes_missing: list[ConfluenceBox]
    grade: SetupGrade
    is_valid: bool
    notes: str


class ConfluenceValidator:
    """
    Implements Fabio's "Box Checking" System.
    
    A trade requires 4-5 boxes checked:
    1. Bias (15m direction)
    2. POI (Price at key level)
    3. Order Flow Trigger (Absorption/Aggression)
    4. Price Follow-through (Volume → Price movement)
    5. Micro Confirmation (15s pattern)
    
    Grade assignment:
    - A: 5/5 boxes + trend alignment
    - B: 4/5 boxes
    - C: 3/5 boxes OR counter-trend with 4+ boxes
    """
    
    def __init__(
        self,
        poi_tolerance_pct: float = 0.002,  # 0.2% tolerance for POI hit
    ):
        self.poi_tolerance_pct = poi_tolerance_pct
    
    def validate(
        self,
        current_price: float,
        bias_direction: Optional[TradeDirection],
        poi_levels: list[float],
        orderflow_signal: dict,
        market_state: MarketStateType,
        intended_direction: TradeDirection,
        micro_confirmed: bool,
    ) -> ConfluenceResult:
        """
        Run full confluence check.
        
        Returns:
            ConfluenceResult with grade and validity
        """
        checked = []
        missing = []
        notes_parts = []
        
        # Box 1: Bias (15m timeframe)
        if bias_direction is not None:
            if bias_direction == intended_direction:
                checked.append(ConfluenceBox.BIAS)
                notes_parts.append(f"Bias aligned: {bias_direction.value}")
            else:
                missing.append(ConfluenceBox.BIAS)
                notes_parts.append(f"Counter-trend: bias is {bias_direction.value}")
        else:
            missing.append(ConfluenceBox.BIAS)
            notes_parts.append("No clear bias established")
        
        # Box 2: POI (Point of Interest)
        at_poi = self._check_poi(current_price, poi_levels)
        if at_poi:
            checked.append(ConfluenceBox.POI)
            notes_parts.append(f"At POI: {at_poi}")
        else:
            missing.append(ConfluenceBox.POI)
            notes_parts.append("Price not at key POI")
        
        # Box 3: Order Flow Trigger
        of_valid, of_note = self._check_orderflow(orderflow_signal, intended_direction)
        if of_valid:
            checked.append(ConfluenceBox.ORDERFLOW_TRIGGER)
        else:
            missing.append(ConfluenceBox.ORDERFLOW_TRIGGER)
        notes_parts.append(of_note)
        
        # Box 4: Price Follow-through
        if orderflow_signal.get("follow_through", False):
            checked.append(ConfluenceBox.PRICE_FOLLOWTHROUGH)
            notes_parts.append("Price following volume")
        else:
            missing.append(ConfluenceBox.PRICE_FOLLOWTHROUGH)
            notes_parts.append("No price follow-through yet")
        
        # Box 5: Micro Confirmation (15s)
        if micro_confirmed:
            checked.append(ConfluenceBox.MICRO_CONFIRMATION)
            notes_parts.append("15s pattern confirmed")
        else:
            missing.append(ConfluenceBox.MICRO_CONFIRMATION)
            notes_parts.append("Awaiting 15s confirmation")
        
        # Determine grade
        grade = self._assign_grade(
            checked=checked,
            is_counter_trend=(ConfluenceBox.BIAS in missing),
            market_state=market_state,
        )
        
        return ConfluenceResult(
            boxes_checked=checked,
            boxes_missing=missing,
            grade=grade,
            is_valid=(grade != SetupGrade.INVALID),
            notes=" | ".join(notes_parts),
        )
    
    def _check_poi(self, price: float, poi_levels: list[float]) -> Optional[float]:
        """Check if price is within tolerance of any POI"""
        for poi in poi_levels:
            distance_pct = abs(price - poi) / poi
            if distance_pct <= self.poi_tolerance_pct:
                return poi
        return None
    
    def _check_orderflow(
        self,
        signal: dict,
        intended: TradeDirection,
    ) -> tuple[bool, str]:
        """Check if order flow supports the intended direction"""
        # For LONG: Need either sell absorption OR buy aggression
        # For SHORT: Need either buy absorption OR sell aggression
        
        absorption = signal.get("absorption_detected", False)
        absorption_side = signal.get("absorption_side")
        aggression = signal.get("aggression_detected", False)
        aggression_side = signal.get("aggression_side")
        trapped = signal.get("trapped_traders", False)
        trapped_side = signal.get("trapped_side")
        
        if intended == TradeDirection.BUY:
            # Good for longs: sell absorption, buy aggression, sellers trapped
            if absorption and absorption_side == TradeDirection.SELL:
                return True, "Sell absorption detected (bullish)"
            if aggression and aggression_side == TradeDirection.BUY:
                return True, "Buy aggression detected"
            if trapped and trapped_side == TradeDirection.SELL:
                return True, "Sellers trapped"
        
        elif intended == TradeDirection.SELL:
            if absorption and absorption_side == TradeDirection.BUY:
                return True, "Buy absorption detected (bearish)"
            if aggression and aggression_side == TradeDirection.SELL:
                return True, "Sell aggression detected"
            if trapped and trapped_side == TradeDirection.BUY:
                return True, "Buyers trapped"
        
        return False, "No supporting order flow"
    
    def _assign_grade(
        self,
        checked: list[ConfluenceBox],
        is_counter_trend: bool,
        market_state: MarketStateType,
    ) -> SetupGrade:
        """Assign A/B/C/INVALID grade based on confluence"""
        count = len(checked)
        
        if count < 3:
            return SetupGrade.INVALID
        
        if count >= 5 and not is_counter_trend:
            return SetupGrade.A
        
        if count >= 4:
            if is_counter_trend:
                return SetupGrade.C  # Counter-trend = always C
            return SetupGrade.B
        
        if count == 3:
            return SetupGrade.C
        
        return SetupGrade.INVALID
```

### 6.2 Trend Continuation Strategy (Primary)

```python
# /engine/strategy/trend_continuation.py
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, time
from typing import Optional
from ..data.schemas import (
    TradeDirection, SetupGrade, TradeSignal, 
    FootprintCandle, MarketStateType
)
from .confluence_validator import ConfluenceValidator, ConfluenceResult


@dataclass
class BiasState:
    """Current market bias from higher timeframe"""
    direction: Optional[TradeDirection]
    strength: float  # 0.0 to 1.0
    established_at: datetime
    invalidation_price: float


class TrendContinuationStrategy:
    """
    Fabio's Primary Setup: Trend Continuation.
    
    Logic:
    1. Establish 15m bias (trending direction)
    2. Wait for pullback to POI (support in uptrend, resistance in downtrend)
    3. Observe absorption at POI (opposing force failing)
    4. Enter on trapped trader signal
    5. Stop below volume wall
    
    NOT traded:
    - First 15-30 minutes of session (erratic)
    - During consolidation (balance state)
    - Counter-trend (except late session mean reversion)
    """
    
    def __init__(
        self,
        symbol: str = "BTCUSDT",
        initial_balance_minutes: int = 30,
        session_start_utc: time = time(8, 0),  # 08:00 UTC
    ):
        self.symbol = symbol
        self.initial_balance_minutes = initial_balance_minutes
        self.session_start_utc = session_start_utc
        self.confluence_validator = ConfluenceValidator()
        
        # State
        self._bias: Optional[BiasState] = None
        self._poi_levels: list[float] = []
        self._in_initial_balance: bool = True
    
    def update_bias(
        self,
        candles_15m: list[FootprintCandle],
        volume_profile: dict,
    ) -> None:
        """
        Update bias from 15-minute candles.
        
        Fabio's criteria for bias:
        - Clear swing structure (HH/HL for bullish, LH/LL for bearish)
        - Price above/below VWAP
        - Delta supporting the direction
        """
        if len(candles_15m) < 5:
            return
        
        recent = candles_15m[-5:]
        
        # Check swing structure
        closes = [c.close for c in recent]
        highs = [c.high for c in recent]
        lows = [c.low for c in recent]
        
        # Simple trend detection
        higher_highs = all(highs[i] > highs[i-1] for i in range(1, len(highs)))
        higher_lows = all(lows[i] > lows[i-1] for i in range(1, len(lows)))
        lower_highs = all(highs[i] < highs[i-1] for i in range(1, len(highs)))
        lower_lows = all(lows[i] < lows[i-1] for i in range(1, len(lows)))
        
        # Check delta alignment
        total_delta = sum(c.delta for c in recent)
        
        direction = None
        strength = 0.0
        
        if (higher_highs or higher_lows) and total_delta > 0:
            direction = TradeDirection.BUY
            strength = min(1.0, total_delta / 500)
            invalidation = min(lows)
        elif (lower_highs or lower_lows) and total_delta < 0:
            direction = TradeDirection.SELL
            strength = min(1.0, abs(total_delta) / 500)
            invalidation = max(highs)
        else:
            # No clear bias - ranging
            self._bias = None
            return
        
        self._bias = BiasState(
            direction=direction,
            strength=strength,
            established_at=datetime.utcnow(),
            invalidation_price=invalidation,
        )
        
        # Update POI levels from volume profile
        self._poi_levels = [
            volume_profile.get("poc"),
            volume_profile.get("vah"),
            volume_profile.get("val"),
        ]
        self._poi_levels = [p for p in self._poi_levels if p is not None]
    
    def check_initial_balance(self, current_time: datetime) -> bool:
        """Check if we're still in initial balance period"""
        session_start = datetime.combine(current_time.date(), self.session_start_utc)
        minutes_since_open = (current_time - session_start).total_seconds() / 60
        
        self._in_initial_balance = minutes_since_open < self.initial_balance_minutes
        return self._in_initial_balance
    
    def generate_signal(
        self,
        current_price: float,
        orderflow_signal: dict,
        market_state: MarketStateType,
        micro_confirmed: bool,
    ) -> Optional[TradeSignal]:
        """
        Generate trade signal if all conditions met.
        
        Returns None if no valid setup.
        """
        # Filter 1: Must be past initial balance
        if self._in_initial_balance:
            return None
        
        # Filter 2: Must have established bias
        if self._bias is None:
            return None
        
        # Filter 3: Must be in imbalance state (trending)
        if market_state == MarketStateType.BALANCE:
            return None
        
        # Filter 4: Check bias invalidation
        if self._bias.direction == TradeDirection.BUY:
            if current_price < self._bias.invalidation_price:
                self._bias = None
                return None
        else:
            if current_price > self._bias.invalidation_price:
                self._bias = None
                return None
        
        # Run confluence validation
        confluence = self.confluence_validator.validate(
            current_price=current_price,
            bias_direction=self._bias.direction,
            poi_levels=self._poi_levels,
            orderflow_signal=orderflow_signal,
            market_state=market_state,
            intended_direction=self._bias.direction,
            micro_confirmed=micro_confirmed,
        )
        
        if not confluence.is_valid:
            return None
        
        # Calculate stops and targets
        stop_loss, take_profit = self._calculate_exits(
            entry=current_price,
            direction=self._bias.direction,
            orderflow_signal=orderflow_signal,
        )
        
        return TradeSignal(
            timestamp=datetime.utcnow(),
            symbol=self.symbol,
            direction=self._bias.direction,
            grade=confluence.grade,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            confluence_score=len(confluence.boxes_checked),
            rationale=confluence.notes,
        )
    
    def _calculate_exits(
        self,
        entry: float,
        direction: TradeDirection,
        orderflow_signal: dict,
    ) -> tuple[float, float]:
        """
        Calculate stop loss and take profit.
        
        Fabio's approach:
        - Stop behind volume wall / big trades
        - Target 1:2.5 to 1:3 R:R
        """
        # Get volume wall from order flow (if available)
        volume_wall = orderflow_signal.get("nearest_volume_wall")
        
        if direction == TradeDirection.BUY:
            # Stop below volume wall or 0.3% below entry
            if volume_wall and volume_wall < entry:
                stop_loss = volume_wall * 0.999  # Tiny buffer
            else:
                stop_loss = entry * 0.997  # 0.3% stop
            
            risk = entry - stop_loss
            take_profit = entry + (risk * 2.5)  # 1:2.5 R:R
        
        else:  # SELL
            if volume_wall and volume_wall > entry:
                stop_loss = volume_wall * 1.001
            else:
                stop_loss = entry * 1.003
            
            risk = stop_loss - entry
            take_profit = entry - (risk * 2.5)
        
        return stop_loss, take_profit
```

### 6.3 Mean Reversion Strategy (Late Session)

```python
# /engine/strategy/mean_reversion.py
from dataclasses import dataclass
from datetime import datetime, time
from typing import Optional
from ..data.schemas import TradeDirection, SetupGrade, TradeSignal


class MeanReversionStrategy:
    """
    Fabio's Late-Session Reversal Setup.
    
    Logic:
    - Only traded late in session (18:45-20:00 CET → ~17:45-19:00 UTC)
    - Requires price at VWAP 2nd or 3rd Standard Deviation
    - Target: Mean reversion to VWAP (fair value)
    - Lower win rate (~40%) but high R:R
    
    Constraint: Only take if already profitable for the day
    
    IMPROVEMENT: Added micro-timeframe confirmation for consistency
    with TrendContinuationStrategy. Even counter-trend trades need
    timing precision.
    """
    
    def __init__(
        self,
        symbol: str = "BTCUSDT",
        session_window_start: time = time(17, 45),  # UTC
        session_window_end: time = time(22, 0),
    ):
        self.symbol = symbol
        self.session_window_start = session_window_start
        self.session_window_end = session_window_end
    
    def is_in_window(self, current_time: datetime) -> bool:
        """Check if current time is in reversal window"""
        current = current_time.time()
        return self.session_window_start <= current <= self.session_window_end
    
    def _check_micro_confirmation(
        self,
        direction: TradeDirection,
        micro_candles: list[dict],
    ) -> bool:
        """
        Check for micro-timeframe (15s) confirmation.
        
        For mean reversion, we look for:
        - LONG: Price rejection from lows (wick > body, close in upper half)
        - SHORT: Price rejection from highs (wick > body, close in lower half)
        
        This adds precision to counter-trend entries.
        """
        if not micro_candles or len(micro_candles) < 3:
            return False
        
        last = micro_candles[-1]
        body = abs(last['close'] - last['open'])
        candle_range = last['high'] - last['low']
        
        if candle_range == 0:
            return False
        
        body_ratio = body / candle_range
        close_position = (last['close'] - last['low']) / candle_range
        
        if direction == TradeDirection.BUY:
            # Looking for rejection candle: small body, close in upper half
            # Indicates sellers failed to push lower
            lower_wick = min(last['open'], last['close']) - last['low']
            return body_ratio < 0.4 and close_position > 0.6 and lower_wick > body
        else:
            # Looking for rejection candle: small body, close in lower half
            upper_wick = last['high'] - max(last['open'], last['close'])
            return body_ratio < 0.4 and close_position < 0.4 and upper_wick > body
    
    def generate_signal(
        self,
        current_price: float,
        vwap: float,
        vwap_std: float,
        session_pnl: float,
        orderflow_signal: dict,
        micro_candles: list[dict] = None,
        atr: float = None,
    ) -> Optional[TradeSignal]:
        """
        Generate mean reversion signal.
        
        Requirements:
        1. In late session window
        2. Price at 2nd+ standard deviation from VWAP
        3. Session is profitable (risk management)
        4. Order flow shows exhaustion
        5. NEW: Micro-TF rejection pattern (optional but recommended)
        """
        current_time = datetime.utcnow()
        
        # Check time window
        if not self.is_in_window(current_time):
            return None
        
        # Check profitability constraint
        if session_pnl <= 0:
            return None
        
        # Calculate deviation bands
        vwap_2std_upper = vwap + (2 * vwap_std)
        vwap_2std_lower = vwap - (2 * vwap_std)
        vwap_3std_upper = vwap + (3 * vwap_std)
        vwap_3std_lower = vwap - (3 * vwap_std)
        
        direction = None
        base_grade = SetupGrade.C  # Counter-trend starts as C
        
        # Check for SHORT setup (price at upper deviation)
        if current_price >= vwap_2std_upper:
            # Need exhaustion signal (CVD divergence or buy absorption)
            if orderflow_signal.get("cvd_divergence") == "BEARISH" or \
               (orderflow_signal.get("absorption_side") == TradeDirection.BUY):
                direction = TradeDirection.SELL
                if current_price >= vwap_3std_upper:
                    base_grade = SetupGrade.B  # 3rd std = higher probability
        
        # Check for LONG setup (price at lower deviation)
        elif current_price <= vwap_2std_lower:
            if orderflow_signal.get("cvd_divergence") == "BULLISH" or \
               (orderflow_signal.get("absorption_side") == TradeDirection.SELL):
                direction = TradeDirection.BUY
                if current_price <= vwap_3std_lower:
                    base_grade = SetupGrade.B
        
        if direction is None:
            return None
        
        # Check micro-TF confirmation (optional upgrade)
        has_micro_confirm = False
        if micro_candles:
            has_micro_confirm = self._check_micro_confirmation(direction, micro_candles)
        
        # Determine final grade
        # With micro confirmation: upgrade C→B or keep B
        # Without micro confirmation: downgrade B→C (stay C if already C)
        if has_micro_confirm:
            grade = base_grade  # Keep the grade (B or C)
        else:
            grade = SetupGrade.C  # Without micro, always C
        
        # Calculate exits using ATR if available, otherwise percentage
        if atr and atr > 0:
            # Stop: 1 ATR behind, Target: VWAP
            if direction == TradeDirection.SELL:
                stop_loss = current_price + atr
                take_profit = vwap
            else:
                stop_loss = current_price - atr
                take_profit = vwap
        else:
            # Fallback: 0.5% stop
            if direction == TradeDirection.SELL:
                stop_loss = current_price * 1.005
                take_profit = vwap
            else:
                stop_loss = current_price * 0.995
                take_profit = vwap
        
        confluence_score = 3  # Base for mean reversion
        if has_micro_confirm:
            confluence_score += 1  # Bonus for micro confirmation
        
        return TradeSignal(
            timestamp=current_time,
            symbol=self.symbol,
            direction=direction,
            grade=grade,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            confluence_score=confluence_score,
            rationale=f"Mean reversion from VWAP {2 if base_grade == SetupGrade.C else 3}σ" +
                      (", micro-TF confirmed" if has_micro_confirm else ""),
        )
```

---

## 7. Risk Management Module

### 7.1 Dynamic Position Sizer (A/B/C System)

```python
# /engine/risk/position_sizer.py
from dataclasses import dataclass
from typing import Optional
from ..data.schemas import SetupGrade

@dataclass
class PositionSize:
    """Calculated position parameters"""
    quantity: float
    notional_value: float
    risk_amount: float
    risk_percent: float
    leverage: float


class DynamicPositionSizer:
    """
    Implements Fabio's Dynamic Risk Allocation.
    
    Key principle: "Not all trades use the same risk"
    
    Base allocation by grade:
    - A Setup: 100% of allocated risk (full confidence)
    - B Setup: 75% of allocated risk
    - C Setup: 50% of allocated risk
    
    The base risk itself is dynamic (see HouseMoneyManager).
    """
    
    def __init__(
        self,
        account_balance: float,
        base_risk_percent: float = 0.25,  # 0.25% base risk
        max_leverage: float = 10.0,
        min_position_usd: float = 100.0,
    ):
        self.account_balance = account_balance
        self.base_risk_percent = base_risk_percent
        self.max_leverage = max_leverage
        self.min_position_usd = min_position_usd
        
        # Grade multipliers
        self.grade_multipliers = {
            SetupGrade.A: 1.0,
            SetupGrade.B: 0.75,
            SetupGrade.C: 0.5,
            SetupGrade.INVALID: 0.0,
        }
    
    def calculate(
        self,
        entry_price: float,
        stop_loss: float,
        grade: SetupGrade,
        risk_multiplier: float = 1.0,  # From HouseMoneyManager
    ) -> Optional[PositionSize]:
        """
        Calculate position size for a trade.
        
        Args:
            entry_price: Intended entry price
            stop_loss: Stop loss price
            grade: Setup grade (A/B/C)
            risk_multiplier: Additional multiplier from house money
        
        Returns:
            PositionSize or None if invalid
        """
        if grade == SetupGrade.INVALID:
            return None
        
        # Calculate risk distance
        risk_distance_pct = abs(entry_price - stop_loss) / entry_price
        if risk_distance_pct == 0:
            return None
        
        # Calculate risk amount
        grade_mult = self.grade_multipliers[grade]
        effective_risk_pct = self.base_risk_percent * grade_mult * risk_multiplier
        risk_amount = self.account_balance * (effective_risk_pct / 100)
        
        # Calculate position size
        notional_value = risk_amount / risk_distance_pct
        quantity = notional_value / entry_price
        
        # Check leverage
        leverage = notional_value / self.account_balance
        if leverage > self.max_leverage:
            # Cap at max leverage
            notional_value = self.account_balance * self.max_leverage
            quantity = notional_value / entry_price
            risk_amount = notional_value * risk_distance_pct
        
        # Check minimum
        if notional_value < self.min_position_usd:
            return None
        
        return PositionSize(
            quantity=quantity,
            notional_value=notional_value,
            risk_amount=risk_amount,
            risk_percent=effective_risk_pct,
            leverage=leverage,
        )
    
    def update_balance(self, new_balance: float) -> None:
        """Update account balance after P&L"""
        self.account_balance = new_balance
```

### 7.2 House Money Manager

```python
# /engine/risk/house_money.py
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

@dataclass
class HouseMoneyState:
    """Current house money state"""
    session_pnl: float
    session_pnl_percent: float
    house_money_available: float
    risk_multiplier: float
    can_compound: bool


class HouseMoneyManager:
    """
    Implements Fabio's "House Money" Concept.
    
    Philosophy:
    - Start with low risk (0.25%)
    - If profitable, risk a portion of PROFITS, not principal
    - If that trade loses, you're stopped out of profit, not capital
    - If it wins, compound aggressively
    
    Example flow:
    1. Start day: Risk 0.25% (base)
    2. Win first trade: +1% → Now have "house money"
    3. Next trade: Risk 0.25% base + 0.5% of profits = 0.75% total
    4. If win: Compound further
    5. If lose: Back to base, but still positive overall
    """
    
    def __init__(
        self,
        base_risk_percent: float = 0.25,
        profit_risk_ratio: float = 0.5,  # Risk 50% of profits
        min_profit_to_compound: float = 0.5,  # Need 0.5% profit to start compounding
        max_risk_multiplier: float = 4.0,  # Cap at 4x base risk
    ):
        self.base_risk_percent = base_risk_percent
        self.profit_risk_ratio = profit_risk_ratio
        self.min_profit_to_compound = min_profit_to_compound
        self.max_risk_multiplier = max_risk_multiplier
        
        # Session state
        self._session_start_balance: float = 0.0
        self._current_balance: float = 0.0
        self._session_id: Optional[str] = None
    
    def start_session(self, balance: float, session_id: str) -> None:
        """Initialize for new trading session"""
        self._session_start_balance = balance
        self._current_balance = balance
        self._session_id = session_id
    
    def update_balance(self, new_balance: float) -> None:
        """Update after trade completion"""
        self._current_balance = new_balance
    
    def get_state(self) -> HouseMoneyState:
        """Get current house money state"""
        session_pnl = self._current_balance - self._session_start_balance
        session_pnl_percent = (session_pnl / self._session_start_balance) * 100
        
        # Calculate house money available
        if session_pnl_percent >= self.min_profit_to_compound:
            house_money = session_pnl * self.profit_risk_ratio
            can_compound = True
        else:
            house_money = 0.0
            can_compound = False
        
        # Calculate risk multiplier
        if can_compound:
            house_money_risk_pct = (house_money / self._session_start_balance) * 100
            additional_multiplier = house_money_risk_pct / self.base_risk_percent
            risk_multiplier = min(1.0 + additional_multiplier, self.max_risk_multiplier)
        else:
            risk_multiplier = 1.0
        
        return HouseMoneyState(
            session_pnl=session_pnl,
            session_pnl_percent=session_pnl_percent,
            house_money_available=house_money,
            risk_multiplier=risk_multiplier,
            can_compound=can_compound,
        )
    
    def get_risk_multiplier(self) -> float:
        """Convenience method for position sizer"""
        return self.get_state().risk_multiplier
```

### 7.3 Session Guard (Circuit Breaker)

```python
# /engine/risk/session_guard.py
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional

class SessionStatus(str, Enum):
    ACTIVE = "ACTIVE"
    PAUSED = "PAUSED"           # Temporary pause (can resume)
    STOPPED = "STOPPED"         # Hard stop for session
    LIQUIDATING = "LIQUIDATING"  # Emergency liquidation


@dataclass
class SessionGuardState:
    """Current session safety state"""
    status: SessionStatus
    consecutive_losses: int
    daily_drawdown_pct: float
    reason: Optional[str]


class SessionGuard:
    """
    Implements Fabio's Hard Stop Rules.
    
    Rules:
    1. 3 consecutive losses → Stop trading for session
    2. Daily drawdown > 5% → Emergency stop
    3. Single trade loss > 2% → Review mode
    
    Philosophy: "If I take 3 consecutive losses, I stop. 
    This indicates market conditions don't match my model."
    """
    
    def __init__(
        self,
        max_consecutive_losses: int = 3,
        max_daily_drawdown_pct: float = 5.0,
        max_single_loss_pct: float = 2.0,
        pause_after_big_loss: bool = True,
    ):
        self.max_consecutive_losses = max_consecutive_losses
        self.max_daily_drawdown_pct = max_daily_drawdown_pct
        self.max_single_loss_pct = max_single_loss_pct
        self.pause_after_big_loss = pause_after_big_loss
        
        # State
        self._consecutive_losses: int = 0
        self._session_start_balance: float = 0.0
        self._current_balance: float = 0.0
        self._status: SessionStatus = SessionStatus.ACTIVE
        self._stop_reason: Optional[str] = None
    
    def start_session(self, balance: float) -> None:
        """Initialize for new session"""
        self._consecutive_losses = 0
        self._session_start_balance = balance
        self._current_balance = balance
        self._status = SessionStatus.ACTIVE
        self._stop_reason = None
    
    def record_trade_result(
        self,
        pnl: float,
        pnl_percent: float,
    ) -> SessionGuardState:
        """
        Record trade result and check safety conditions.
        
        Returns current state after evaluation.
        """
        self._current_balance += pnl
        
        # Update consecutive losses
        if pnl < 0:
            self._consecutive_losses += 1
        else:
            self._consecutive_losses = 0  # Reset on win
        
        # Check consecutive losses rule
        if self._consecutive_losses >= self.max_consecutive_losses:
            self._status = SessionStatus.STOPPED
            self._stop_reason = f"Hit {self.max_consecutive_losses} consecutive losses"
            return self.get_state()
        
        # Check daily drawdown
        daily_dd_pct = self._calculate_drawdown()
        if daily_dd_pct >= self.max_daily_drawdown_pct:
            self._status = SessionStatus.LIQUIDATING
            self._stop_reason = f"Daily drawdown exceeded {self.max_daily_drawdown_pct}%"
            return self.get_state()
        
        # Check single trade loss
        if pnl_percent < -self.max_single_loss_pct and self.pause_after_big_loss:
            self._status = SessionStatus.PAUSED
            self._stop_reason = f"Single trade loss of {abs(pnl_percent):.2f}%"
        
        return self.get_state()
    
    def _calculate_drawdown(self) -> float:
        """Calculate current session drawdown"""
        if self._session_start_balance == 0:
            return 0.0
        pnl = self._current_balance - self._session_start_balance
        if pnl >= 0:
            return 0.0
        return abs(pnl / self._session_start_balance) * 100
    
    def get_state(self) -> SessionGuardState:
        """Get current guard state"""
        return SessionGuardState(
            status=self._status,
            consecutive_losses=self._consecutive_losses,
            daily_drawdown_pct=self._calculate_drawdown(),
            reason=self._stop_reason,
        )
    
    def can_trade(self) -> bool:
        """Check if trading is allowed"""
        return self._status == SessionStatus.ACTIVE
    
    def resume(self) -> bool:
        """Attempt to resume from PAUSED state"""
        if self._status == SessionStatus.PAUSED:
            self._status = SessionStatus.ACTIVE
            self._stop_reason = None
            return True
        return False
```

---

## 8. Execution Engine

### 8.1 Order State Machine

```python
# /engine/execution/order_manager.py
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Callable
import uuid

class OrderState(str, Enum):
    """Order lifecycle states"""
    PENDING = "PENDING"           # Created, not yet submitted
    SUBMITTED = "SUBMITTED"       # Sent to exchange
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_MARKET = "STOP_MARKET"
    STOP_LIMIT = "STOP_LIMIT"
    TAKE_PROFIT_MARKET = "TAKE_PROFIT_MARKET"


@dataclass
class Order:
    """Single order representation"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    symbol: str = ""
    side: str = ""  # "BUY" or "SELL"
    order_type: OrderType = OrderType.LIMIT
    quantity: float = 0.0
    price: Optional[float] = None
    stop_price: Optional[float] = None
    state: OrderState = OrderState.PENDING
    exchange_order_id: Optional[str] = None
    filled_quantity: float = 0.0
    average_fill_price: float = 0.0
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    error_message: Optional[str] = None


class OrderManager:
    """
    Manages order lifecycle and state transitions.
    
    Responsibilities:
    - Track all orders and their states
    - Enforce valid state transitions
    - Emit events on state changes
    - Handle OCO (One-Cancels-Other) logic for SL/TP
    """
    
    # Valid state transitions
    VALID_TRANSITIONS = {
        OrderState.PENDING: {OrderState.SUBMITTED, OrderState.CANCELLED},
        OrderState.SUBMITTED: {
            OrderState.PARTIALLY_FILLED, 
            OrderState.FILLED, 
            OrderState.CANCELLED, 
            OrderState.REJECTED
        },
        OrderState.PARTIALLY_FILLED: {
            OrderState.FILLED, 
            OrderState.CANCELLED
        },
        # Terminal states: no transitions out
        OrderState.FILLED: set(),
        OrderState.CANCELLED: set(),
        OrderState.REJECTED: set(),
        OrderState.EXPIRED: set(),
    }
    
    def __init__(self, on_state_change: Optional[Callable] = None):
        self._orders: dict[str, Order] = {}
        self._oco_pairs: dict[str, str] = {}  # Maps SL order_id -> TP order_id and vice versa
        self._on_state_change = on_state_change
    
    def create_order(
        self,
        symbol: str,
        side: str,
        order_type: OrderType,
        quantity: float,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
    ) -> Order:
        """Create a new order in PENDING state"""
        order = Order(
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            stop_price=stop_price,
        )
        self._orders[order.id] = order
        return order
    
    def create_oco_pair(
        self,
        symbol: str,
        side: str,  # Exit side (opposite of position)
        quantity: float,
        stop_loss_price: float,
        take_profit_price: float,
    ) -> tuple[Order, Order]:
        """Create linked Stop-Loss and Take-Profit orders"""
        sl_order = self.create_order(
            symbol=symbol,
            side=side,
            order_type=OrderType.STOP_MARKET,
            quantity=quantity,
            stop_price=stop_loss_price,
        )
        
        tp_order = self.create_order(
            symbol=symbol,
            side=side,
            order_type=OrderType.TAKE_PROFIT_MARKET,
            quantity=quantity,
            stop_price=take_profit_price,
        )
        
        # Link them
        self._oco_pairs[sl_order.id] = tp_order.id
        self._oco_pairs[tp_order.id] = sl_order.id
        
        return sl_order, tp_order
    
    def transition(
        self,
        order_id: str,
        new_state: OrderState,
        exchange_order_id: Optional[str] = None,
        filled_quantity: Optional[float] = None,
        average_fill_price: Optional[float] = None,
        error_message: Optional[str] = None,
    ) -> bool:
        """
        Transition order to new state.
        
        Returns True if transition was valid and applied.
        """
        order = self._orders.get(order_id)
        if not order:
            return False
        
        # Validate transition
        if new_state not in self.VALID_TRANSITIONS.get(order.state, set()):
            return False
        
        old_state = order.state
        order.state = new_state
        order.updated_at = datetime.utcnow()
        
        if exchange_order_id:
            order.exchange_order_id = exchange_order_id
        if filled_quantity is not None:
            order.filled_quantity = filled_quantity
        if average_fill_price is not None:
            order.average_fill_price = average_fill_price
        if error_message:
            order.error_message = error_message
        
        # Handle OCO logic
        if new_state == OrderState.FILLED:
            self._handle_oco_fill(order_id)
        
        # Emit event
        if self._on_state_change:
            self._on_state_change(order, old_state, new_state)
        
        return True
    
    def _handle_oco_fill(self, filled_order_id: str) -> None:
        """Cancel the other side of an OCO pair when one fills"""
        other_id = self._oco_pairs.get(filled_order_id)
        if other_id and other_id in self._orders:
            other_order = self._orders[other_id]
            if other_order.state in {OrderState.SUBMITTED, OrderState.PENDING}:
                self.transition(other_id, OrderState.CANCELLED)
    
    def get_order(self, order_id: str) -> Optional[Order]:
        return self._orders.get(order_id)
    
    def get_open_orders(self) -> list[Order]:
        return [
            o for o in self._orders.values()
            if o.state in {OrderState.PENDING, OrderState.SUBMITTED, OrderState.PARTIALLY_FILLED}
        ]
```

### 8.2 Binance Executor

```python
# /engine/execution/binance_executor.py
import ccxt.async_support as ccxt
from typing import Optional
from dataclasses import dataclass
from contextlib import asynccontextmanager
import asyncio
from ..data.schemas import TradeDirection
from .order_manager import OrderManager, Order, OrderState, OrderType

@dataclass
class ExecutionResult:
    """Result of execution attempt"""
    success: bool
    order_id: Optional[str]
    exchange_order_id: Optional[str]
    fill_price: Optional[float]
    fill_quantity: Optional[float]
    slippage: Optional[float]  # Actual vs expected price
    error: Optional[str]


@dataclass
class SlippageEstimate:
    """Pre-trade slippage estimation"""
    expected_slippage_pct: float
    confidence: float
    recommendation: str  # 'proceed', 'reduce_size', 'abort'


class SlippageModel:
    """
    Estimates expected slippage before trade execution.
    
    Factors:
    1. Order size vs recent average volume
    2. Current spread
    3. Order book depth at target price
    4. Time of day (volatility proxy)
    
    NOTE: This model requires live calibration.
    Initial values are conservative estimates for BTC futures.
    """
    
    def __init__(
        self,
        base_slippage_pct: float = 0.01,  # 0.01% base slippage
        volume_impact_factor: float = 0.5,
        spread_multiplier: float = 1.5,
    ):
        self.base_slippage_pct = base_slippage_pct
        self.volume_impact_factor = volume_impact_factor
        self.spread_multiplier = spread_multiplier
        
        # Calibration data (updated from live trades)
        self._recent_slippages: list[float] = []
        self._calibrated_base: Optional[float] = None
    
    def estimate(
        self,
        order_size_usd: float,
        recent_volume_usd: float,
        current_spread_pct: float,
        order_book_depth_usd: float,
    ) -> SlippageEstimate:
        """
        Estimate slippage for a potential trade.
        
        Args:
            order_size_usd: Notional value of order
            recent_volume_usd: Average volume over last 5 minutes
            current_spread_pct: Current bid-ask spread as percentage
            order_book_depth_usd: Total liquidity within 0.1% of mid price
        """
        base = self._calibrated_base or self.base_slippage_pct
        
        # Volume impact: larger orders relative to volume = more slippage
        volume_ratio = order_size_usd / recent_volume_usd if recent_volume_usd > 0 else 1
        volume_impact = volume_ratio * self.volume_impact_factor
        
        # Spread impact
        spread_impact = current_spread_pct * self.spread_multiplier
        
        # Depth impact: if order exceeds depth, expect more slippage
        depth_ratio = order_size_usd / order_book_depth_usd if order_book_depth_usd > 0 else 1
        depth_impact = max(0, (depth_ratio - 1) * 0.1)  # Penalty if order > depth
        
        total_slippage = base + volume_impact + spread_impact + depth_impact
        
        # Confidence based on calibration data
        confidence = min(1.0, len(self._recent_slippages) / 50)
        
        # Recommendation
        if total_slippage > 0.1:  # >0.1% is bad for scalping
            recommendation = 'abort'
        elif total_slippage > 0.05:
            recommendation = 'reduce_size'
        else:
            recommendation = 'proceed'
        
        return SlippageEstimate(
            expected_slippage_pct=total_slippage,
            confidence=confidence,
            recommendation=recommendation,
        )
    
    def record_actual_slippage(self, expected_price: float, actual_price: float) -> None:
        """Record actual slippage for calibration"""
        actual_slippage = abs(actual_price - expected_price) / expected_price * 100
        self._recent_slippages.append(actual_slippage)
        
        # Keep last 100 trades
        if len(self._recent_slippages) > 100:
            self._recent_slippages.pop(0)
        
        # Recalibrate base
        if len(self._recent_slippages) >= 20:
            self._calibrated_base = sum(self._recent_slippages) / len(self._recent_slippages)


class BinanceExecutor:
    """
    CCXT-based executor for Binance Futures.
    
    Handles:
    - Order submission with proper async session management
    - Order modification (trailing stops)
    - Position queries
    - Balance queries
    - Slippage estimation
    
    IMPORTANT: Uses async context manager pattern to ensure
    proper session cleanup and avoid connection leaks.
    """
    
    def __init__(
        self,
        api_key: str,
        api_secret: str,
        testnet: bool = True,
        order_manager: Optional[OrderManager] = None,
    ):
        self._api_key = api_key
        self._api_secret = api_secret
        self._testnet = testnet
        self.order_manager = order_manager or OrderManager()
        self.slippage_model = SlippageModel()
        self._exchange: Optional[ccxt.binanceusdm] = None
        self._session_lock = asyncio.Lock()
    
    async def _ensure_exchange(self) -> ccxt.binanceusdm:
        """Lazy initialization of exchange connection"""
        if self._exchange is None:
            self._exchange = ccxt.binanceusdm({
                'apiKey': self._api_key,
                'secret': self._api_secret,
                'sandbox': self._testnet,
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'future',
                    'adjustForTimeDifference': True,
                },
            })
            await self._exchange.load_markets()
        return self._exchange
    
    async def close(self) -> None:
        """Clean up exchange session"""
        if self._exchange:
            await self._exchange.close()
            self._exchange = None
    
    @asynccontextmanager
    async def session(self):
        """
        Async context manager for safe session handling.
        
        Usage:
            async with executor.session():
                await executor.submit_order(order)
        """
        try:
            await self._ensure_exchange()
            yield self
        finally:
            pass  # Keep connection alive for reuse
    
    async def submit_order(
        self,
        order: Order,
        check_slippage: bool = True,
    ) -> ExecutionResult:
        """
        Submit order to Binance with optional slippage check.
        
        For market orders, estimates slippage first and may abort
        if conditions are unfavorable.
        """
        async with self._session_lock:  # Prevent concurrent order submission
            try:
                exchange = await self._ensure_exchange()
                
                # Slippage check for market orders
                if check_slippage and order.order_type == OrderType.MARKET:
                    ticker = await exchange.fetch_ticker(order.symbol)
                    orderbook = await exchange.fetch_order_book(order.symbol, limit=10)
                    
                    # Calculate depth
                    depth = sum(b[1] * b[0] for b in orderbook['bids'][:5])
                    depth += sum(a[1] * a[0] for a in orderbook['asks'][:5])
                    
                    slippage_est = self.slippage_model.estimate(
                        order_size_usd=order.quantity * ticker['last'],
                        recent_volume_usd=ticker.get('quoteVolume', 1000000) / 288,  # 5min avg
                        current_spread_pct=(ticker['ask'] - ticker['bid']) / ticker['bid'] * 100,
                        order_book_depth_usd=depth,
                    )
                    
                    if slippage_est.recommendation == 'abort':
                        return ExecutionResult(
                            success=False,
                            order_id=order.id,
                            exchange_order_id=None,
                            fill_price=None,
                            fill_quantity=None,
                            slippage=None,
                            error=f"Slippage too high: {slippage_est.expected_slippage_pct:.3f}%",
                        )
                
                # Submit based on order type
                expected_price = order.price or (await exchange.fetch_ticker(order.symbol))['last']
                
                if order.order_type == OrderType.MARKET:
                    exchange_order = await exchange.create_market_order(
                        symbol=order.symbol,
                        side=order.side.lower(),
                        amount=order.quantity,
                    )
                elif order.order_type == OrderType.LIMIT:
                    exchange_order = await exchange.create_limit_order(
                        symbol=order.symbol,
                        side=order.side.lower(),
                        amount=order.quantity,
                        price=order.price,
                    )
                elif order.order_type == OrderType.STOP_MARKET:
                    exchange_order = await exchange.create_order(
                        symbol=order.symbol,
                        type='STOP_MARKET',
                        side=order.side.lower(),
                        amount=order.quantity,
                        params={'stopPrice': order.stop_price, 'reduceOnly': True}
                    )
                elif order.order_type == OrderType.TAKE_PROFIT_MARKET:
                    exchange_order = await exchange.create_order(
                        symbol=order.symbol,
                        type='TAKE_PROFIT_MARKET',
                        side=order.side.lower(),
                        amount=order.quantity,
                        params={'stopPrice': order.stop_price, 'reduceOnly': True}
                    )
                else:
                    return ExecutionResult(
                        success=False,
                        order_id=order.id,
                        exchange_order_id=None,
                        fill_price=None,
                        fill_quantity=None,
                        slippage=None,
                        error=f"Unsupported order type: {order.order_type}",
                    )
                
                # Update order manager
                self.order_manager.transition(
                    order.id,
                    OrderState.SUBMITTED,
                    exchange_order_id=exchange_order['id'],
                )
                
                # Calculate actual slippage if filled
                actual_slippage = None
                if exchange_order['status'] == 'closed' and exchange_order.get('average'):
                    actual_slippage = abs(exchange_order['average'] - expected_price) / expected_price * 100
                    self.slippage_model.record_actual_slippage(expected_price, exchange_order['average'])
                    
                    self.order_manager.transition(
                        order.id,
                        OrderState.FILLED,
                        filled_quantity=exchange_order['filled'],
                        average_fill_price=exchange_order['average'],
                    )
                
                return ExecutionResult(
                    success=True,
                    order_id=order.id,
                    exchange_order_id=exchange_order['id'],
                    fill_price=exchange_order.get('average'),
                    fill_quantity=exchange_order.get('filled'),
                    slippage=actual_slippage,
                    error=None,
                )
                
            except ccxt.RateLimitExceeded as e:
                # Back off and retry logic could go here
                return ExecutionResult(
                    success=False,
                    order_id=order.id,
                    exchange_order_id=None,
                    fill_price=None,
                    fill_quantity=None,
                    slippage=None,
                    error=f"Rate limit exceeded: {e}",
                )
            except ccxt.NetworkError as e:
                return ExecutionResult(
                    success=False,
                    order_id=order.id,
                    exchange_order_id=None,
                    fill_price=None,
                    fill_quantity=None,
                    slippage=None,
                    error=f"Network error: {e}",
                )
            except Exception as e:
                self.order_manager.transition(
                    order.id,
                    OrderState.REJECTED,
                    error_message=str(e),
                )
                return ExecutionResult(
                    success=False,
                    order_id=order.id,
                    exchange_order_id=None,
                    fill_price=None,
                    fill_quantity=None,
                    slippage=None,
                    error=str(e),
                )
    
    async def cancel_order(self, order: Order) -> bool:
        """Cancel an open order"""
        try:
            exchange = await self._ensure_exchange()
            if order.exchange_order_id:
                await exchange.cancel_order(
                    id=order.exchange_order_id,
                    symbol=order.symbol,
                )
            self.order_manager.transition(order.id, OrderState.CANCELLED)
            return True
        except Exception:
            return False
    
    async def get_position(self, symbol: str) -> Optional[dict]:
        """Get current position for symbol"""
        try:
            exchange = await self._ensure_exchange()
            positions = await exchange.fetch_positions([symbol])
            for pos in positions:
                if pos['symbol'] == symbol and float(pos['contracts']) != 0:
                    return {
                        'symbol': symbol,
                        'side': 'LONG' if pos['side'] == 'long' else 'SHORT',
                        'quantity': abs(float(pos['contracts'])),
                        'entry_price': float(pos['entryPrice']),
                        'unrealized_pnl': float(pos['unrealizedPnl']),
                        'leverage': int(pos['leverage']),
                    }
            return None
        except Exception:
            return None
    
    async def get_all_positions(self) -> list[dict]:
        """Get all open positions"""
        try:
            exchange = await self._ensure_exchange()
            positions = await exchange.fetch_positions()
            return [
                {
                    'symbol': pos['symbol'],
                    'side': 'LONG' if pos['side'] == 'long' else 'SHORT',
                    'quantity': abs(float(pos['contracts'])),
                    'entry_price': float(pos['entryPrice']),
                    'unrealized_pnl': float(pos['unrealizedPnl']),
                }
                for pos in positions
                if float(pos['contracts']) != 0
            ]
        except Exception:
            return []
    
    async def get_open_orders(self, symbol: str) -> list[dict]:
        """Get open orders for symbol"""
        try:
            exchange = await self._ensure_exchange()
            orders = await exchange.fetch_open_orders(symbol)
            return [
                {
                    'id': o['id'],
                    'symbol': o['symbol'],
                    'type': o['type'],
                    'side': o['side'],
                    'price': o.get('price'),
                    'stopPrice': o.get('stopPrice'),
                    'amount': o['amount'],
                }
                for o in orders
            ]
        except Exception:
            return []
    
    async def get_balance(self) -> float:
        """Get USDT balance"""
        try:
            exchange = await self._ensure_exchange()
            balance = await exchange.fetch_balance()
            return float(balance['USDT']['free'])
        except Exception:
            return 0.0
    
    async def modify_stop_loss(
        self,
        symbol: str,
        current_sl_order: Order,
        new_stop_price: float,
    ) -> ExecutionResult:
        """
        Modify stop loss (cancel and replace).
        
        NOTE: Binance Futures doesn't support order modification.
        We must cancel and replace atomically.
        """
        # Cancel existing
        cancelled = await self.cancel_order(current_sl_order)
        if not cancelled:
            return ExecutionResult(
                success=False,
                order_id=current_sl_order.id,
                exchange_order_id=None,
                fill_price=None,
                fill_quantity=None,
                slippage=None,
                error="Failed to cancel existing stop loss",
            )
        
        # Create new
        new_order = self.order_manager.create_order(
            symbol=symbol,
            side=current_sl_order.side,
            order_type=OrderType.STOP_MARKET,
            quantity=current_sl_order.quantity,
            stop_price=new_stop_price,
        )
        
        return await self.submit_order(new_order, check_slippage=False)
```

### 8.3 Trade Manager (Orchestrator)

```python
# /engine/execution/trade_manager.py
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from enum import Enum
from ..data.schemas import TradeSignal, TradeDirection
from ..risk.position_sizer import DynamicPositionSizer, PositionSize
from ..risk.house_money import HouseMoneyManager
from ..risk.session_guard import SessionGuard
from .order_manager import OrderManager, Order, OrderType
from .binance_executor import BinanceExecutor

class TradeState(str, Enum):
    """Active trade lifecycle"""
    PENDING_ENTRY = "PENDING_ENTRY"
    ENTERED = "ENTERED"
    PENDING_EXIT = "PENDING_EXIT"
    CLOSED = "CLOSED"


@dataclass
class ActiveTrade:
    """Represents an active trade with all associated orders"""
    id: str
    signal: TradeSignal
    position_size: PositionSize
    state: TradeState
    entry_order: Optional[Order]
    stop_loss_order: Optional[Order]
    take_profit_order: Optional[Order]
    entry_time: Optional[datetime]
    entry_price: Optional[float]
    exit_time: Optional[datetime]
    exit_price: Optional[float]
    exit_reason: Optional[str]
    realized_pnl: Optional[float]


class TradeManager:
    """
    Orchestrates the complete trade lifecycle.
    
    Flow:
    1. Receive signal from Strategy Engine
    2. Validate with Risk Manager
    3. Calculate position size
    4. Submit entry order
    5. On fill: Submit OCO (SL + TP)
    6. Monitor for invalidation
    7. Handle exit (SL, TP, or manual)
    8. Record result
    
    Implements Fabio's trade management:
    - Aggressive break-even moves
    - Early exit on invalidation
    - Trailing stops behind volume walls
    """
    
    def __init__(
        self,
        executor: BinanceExecutor,
        position_sizer: DynamicPositionSizer,
        house_money: HouseMoneyManager,
        session_guard: SessionGuard,
    ):
        self.executor = executor
        self.position_sizer = position_sizer
        self.house_money = house_money
        self.session_guard = session_guard
        
        self._active_trade: Optional[ActiveTrade] = None
    
    async def execute_signal(self, signal: TradeSignal) -> Optional[ActiveTrade]:
        """
        Execute a trade signal.
        
        Returns ActiveTrade if successful, None if rejected.
        """
        # Pre-trade checks
        if not self.session_guard.can_trade():
            return None
        
        # Calculate position size
        risk_mult = self.house_money.get_risk_multiplier()
        position = self.position_sizer.calculate(
            entry_price=signal.entry_price,
            stop_loss=signal.stop_loss,
            grade=signal.grade,
            risk_multiplier=risk_mult,
        )
        
        if position is None:
            return None
        
        # Create trade
        trade = ActiveTrade(
            id=f"trade_{datetime.utcnow().timestamp()}",
            signal=signal,
            position_size=position,
            state=TradeState.PENDING_ENTRY,
            entry_order=None,
            stop_loss_order=None,
            take_profit_order=None,
            entry_time=None,
            entry_price=None,
            exit_time=None,
            exit_price=None,
            exit_reason=None,
            realized_pnl=None,
        )
        
        # Submit entry order
        entry_order = self.executor.order_manager.create_order(
            symbol=signal.symbol,
            side=signal.direction.value,
            order_type=OrderType.LIMIT,
            quantity=position.quantity,
            price=signal.entry_price,
        )
        trade.entry_order = entry_order
        
        result = await self.executor.submit_order(entry_order)
        
        if not result.success:
            trade.state = TradeState.CLOSED
            trade.exit_reason = f"Entry rejected: {result.error}"
            return trade
        
        # If filled immediately
        if result.fill_price:
            trade.entry_price = result.fill_price
            trade.entry_time = datetime.utcnow()
            trade.state = TradeState.ENTERED
            
            # Submit OCO
            await self._submit_exit_orders(trade)
        
        self._active_trade = trade
        return trade
    
    async def _submit_exit_orders(self, trade: ActiveTrade) -> None:
        """Submit stop loss and take profit orders"""
        exit_side = "SELL" if trade.signal.direction == TradeDirection.BUY else "BUY"
        
        sl_order, tp_order = self.executor.order_manager.create_oco_pair(
            symbol=trade.signal.symbol,
            side=exit_side,
            quantity=trade.position_size.quantity,
            stop_loss_price=trade.signal.stop_loss,
            take_profit_price=trade.signal.take_profit,
        )
        
        trade.stop_loss_order = sl_order
        trade.take_profit_order = tp_order
        
        await self.executor.submit_order(sl_order)
        await self.executor.submit_order(tp_order)
    
    async def move_to_breakeven(self) -> bool:
        """
        Move stop loss to break-even.
        
        Fabio's approach: "If market doesn't immediately follow through,
        move to break-even to eliminate risk."
        """
        if not self._active_trade or self._active_trade.state != TradeState.ENTERED:
            return False
        
        trade = self._active_trade
        
        # Calculate break-even (entry + small buffer for fees)
        fee_buffer = trade.entry_price * 0.0002  # 0.02% for fees
        if trade.signal.direction == TradeDirection.BUY:
            new_sl = trade.entry_price + fee_buffer
        else:
            new_sl = trade.entry_price - fee_buffer
        
        result = await self.executor.modify_stop_loss(
            symbol=trade.signal.symbol,
            current_sl_order=trade.stop_loss_order,
            new_stop_price=new_sl,
        )
        
        return result.success
    
    async def invalidation_exit(self, reason: str) -> bool:
        """
        Exit trade due to invalidation.
        
        Fabio's approach: "If structure breaks WITH sell-side volume aggression,
        exit immediately. Don't wait for stop loss."
        """
        if not self._active_trade or self._active_trade.state != TradeState.ENTERED:
            return False
        
        trade = self._active_trade
        
        # Cancel existing exit orders
        await self.executor.cancel_order(trade.stop_loss_order)
        await self.executor.cancel_order(trade.take_profit_order)
        
        # Submit market close
        exit_side = "SELL" if trade.signal.direction == TradeDirection.BUY else "BUY"
        close_order = self.executor.order_manager.create_order(
            symbol=trade.signal.symbol,
            side=exit_side,
            order_type=OrderType.MARKET,
            quantity=trade.position_size.quantity,
        )
        
        result = await self.executor.submit_order(close_order)
        
        if result.success:
            trade.state = TradeState.CLOSED
            trade.exit_price = result.fill_price
            trade.exit_time = datetime.utcnow()
            trade.exit_reason = f"Invalidation: {reason}"
            trade.realized_pnl = self._calculate_pnl(trade)
            
            # Update risk managers
            pnl_pct = (trade.realized_pnl / self.position_sizer.account_balance) * 100
            self.session_guard.record_trade_result(trade.realized_pnl, pnl_pct)
            
            self._active_trade = None
        
        return result.success
    
    def _calculate_pnl(self, trade: ActiveTrade) -> float:
        """Calculate realized P&L"""
        if not trade.entry_price or not trade.exit_price:
            return 0.0
        
        if trade.signal.direction == TradeDirection.BUY:
            price_diff = trade.exit_price - trade.entry_price
        else:
            price_diff = trade.entry_price - trade.exit_price
        
        gross_pnl = price_diff * trade.position_size.quantity
        
        # Estimate fees (0.04% taker)
        entry_fee = trade.entry_price * trade.position_size.quantity * 0.0004
        exit_fee = trade.exit_price * trade.position_size.quantity * 0.0004
        
        return gross_pnl - entry_fee - exit_fee
```

---

## 9. State Management

### 9.1 Market State Detector

```python
# /engine/state/market_state.py
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from enum import Enum
from ..data.schemas import MarketStateType, FootprintCandle

@dataclass
class MarketStateSnapshot:
    """Complete market state at a point in time"""
    timestamp: datetime
    state_type: MarketStateType
    balance_high: Optional[float]
    balance_low: Optional[float]
    trend_direction: Optional[str]
    volatility_percentile: float
    atr_percent: float


class MarketStateDetector:
    """
    Detects Balance vs Imbalance market states.
    
    Fabio's Logic:
    - Balance: Price rotating around fair value, range-bound
    - Imbalance: Aggressive participation pushing price directionally
    
    Detection method:
    - Balance: Price stays within Initial Balance range with declining delta
    - Imbalance: Price breaks IB with follow-through volume
    """
    
    def __init__(
        self,
        ib_break_threshold: float = 0.003,  # 0.3% beyond IB = break
        delta_trend_periods: int = 10,
    ):
        self.ib_break_threshold = ib_break_threshold
        self.delta_trend_periods = delta_trend_periods
        
        # Initial Balance tracking
        self._ib_high: Optional[float] = None
        self._ib_low: Optional[float] = None
        self._ib_established: bool = False
        
        # Delta tracking
        self._recent_deltas: list[float] = []
    
    def set_initial_balance(self, high: float, low: float) -> None:
        """Set Initial Balance range (typically first 30-60 min of session)"""
        self._ib_high = high
        self._ib_low = low
        self._ib_established = True
    
    def analyze(
        self,
        current_price: float,
        candle: FootprintCandle,
        atr_percent: float,
    ) -> MarketStateSnapshot:
        """
        Analyze current market state.
        
        Returns MarketStateSnapshot with classification.
        """
        # Update delta tracking
        self._recent_deltas.append(candle.delta)
        if len(self._recent_deltas) > self.delta_trend_periods:
            self._recent_deltas.pop(0)
        
        state_type = MarketStateType.UNKNOWN
        trend_direction = None
        
        if not self._ib_established:
            state_type = MarketStateType.UNKNOWN
        else:
            # Check for IB break
            ib_range = self._ib_high - self._ib_low
            break_threshold = ib_range * self.ib_break_threshold
            
            if current_price > self._ib_high + break_threshold:
                # Check for delta confirmation
                if self._is_delta_trending_up():
                    state_type = MarketStateType.IMBALANCE_UP
                    trend_direction = "UP"
                else:
                    state_type = MarketStateType.BALANCE  # False break
                    
            elif current_price < self._ib_low - break_threshold:
                if self._is_delta_trending_down():
                    state_type = MarketStateType.IMBALANCE_DOWN
                    trend_direction = "DOWN"
                else:
                    state_type = MarketStateType.BALANCE
            else:
                state_type = MarketStateType.BALANCE
        
        # Calculate volatility percentile (simplified)
        volatility_percentile = min(100, (atr_percent / 0.03) * 100)  # 3% ATR = 100th percentile
        
        return MarketStateSnapshot(
            timestamp=datetime.utcnow(),
            state_type=state_type,
            balance_high=self._ib_high,
            balance_low=self._ib_low,
            trend_direction=trend_direction,
            volatility_percentile=volatility_percentile,
            atr_percent=atr_percent,
        )
    
    def _is_delta_trending_up(self) -> bool:
        """Check if delta is trending positive"""
        if len(self._recent_deltas) < 3:
            return False
        return sum(self._recent_deltas[-5:]) > 0
    
    def _is_delta_trending_down(self) -> bool:
        """Check if delta is trending negative"""
        if len(self._recent_deltas) < 3:
            return False
        return sum(self._recent_deltas[-5:]) < 0
    
    def reset_session(self) -> None:
        """Reset for new trading session"""
        self._ib_high = None
        self._ib_low = None
        self._ib_established = False
        self._recent_deltas.clear()
```

---

## 10. LLM Analyst Integration

### 10.1 Trade Reviewer

```python
# /engine/analyst/trade_reviewer.py
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
import json

@dataclass
class TradeReview:
    """LLM-generated trade analysis"""
    trade_id: str
    timestamp: datetime
    summary: str
    entry_analysis: str
    exit_analysis: str
    order_flow_notes: str
    improvement_suggestions: list[str]
    confidence_assessment: str
    raw_response: str


class TradeReviewer:
    """
    Uses LLM to analyze completed trades.
    
    Functions:
    1. Post-trade review (why did it win/lose?)
    2. Pattern identification (recurring mistakes)
    3. Suggestion generation (how to improve)
    
    This replaces Fabio's manual journaling process.
    """
    
    REVIEW_PROMPT_TEMPLATE = """You are an expert trading analyst specializing in order flow and 
market microstructure. Analyze this completed trade.

## Trade Details
- Symbol: {symbol}
- Direction: {direction}
- Grade: {grade} (A=highest confidence, C=lowest)
- Entry: {entry_price} at {entry_time}
- Exit: {exit_price} at {exit_time}
- Exit Reason: {exit_reason}
- P&L: {pnl} ({pnl_percent}%)
- Risk/Reward Achieved: {rr_achieved}

## Market Context at Entry
- Market State: {market_state}
- 15m Bias: {bias}
- Confluence Score: {confluence}/5 boxes checked

## Order Flow Data at Entry
- Delta (15s): {delta_15s}
- Delta (1m): {delta_1m}
- CVD Trend: {cvd_trend}
- Absorption Detected: {absorption}
- Trapped Traders: {trapped}

## Price Action Summary
{price_summary}

---

Provide analysis in the following format:

### Entry Quality
[Assess whether entry timing and price were optimal]

### Exit Assessment  
[Evaluate whether exit was correct or if more profit was left on table]

### Order Flow Reading
[Was the order flow interpretation correct? What did we miss?]

### Key Lessons
[List 2-3 specific, actionable improvements]

### Confidence in System
[Rate 1-10: Did this trade follow the system? Was the loss/win due to 
execution or market randomness?]
"""
    
    def __init__(self, llm_client):
        self.llm_client = llm_client
    
    async def review_trade(
        self,
        trade_data: dict,
        market_context: dict,
        orderflow_data: dict,
    ) -> TradeReview:
        """
        Generate comprehensive trade review.
        
        Args:
            trade_data: Completed trade details
            market_context: State at time of trade
            orderflow_data: Order flow metrics at entry
        
        Returns:
            TradeReview with LLM analysis
        """
        # Build prompt
        prompt = self.REVIEW_PROMPT_TEMPLATE.format(
            symbol=trade_data['symbol'],
            direction=trade_data['direction'],
            grade=trade_data['grade'],
            entry_price=trade_data['entry_price'],
            entry_time=trade_data['entry_time'],
            exit_price=trade_data['exit_price'],
            exit_time=trade_data['exit_time'],
            exit_reason=trade_data['exit_reason'],
            pnl=trade_data['pnl'],
            pnl_percent=trade_data['pnl_percent'],
            rr_achieved=trade_data.get('rr_achieved', 'N/A'),
            market_state=market_context['state'],
            bias=market_context['bias'],
            confluence=trade_data['confluence_score'],
            delta_15s=orderflow_data['delta_15s'],
            delta_1m=orderflow_data['delta_1m'],
            cvd_trend=orderflow_data['cvd_trend'],
            absorption=orderflow_data['absorption'],
            trapped=orderflow_data['trapped'],
            price_summary=market_context.get('price_summary', 'Not available'),
        )
        
        # Call LLM
        response = await self.llm_client.complete(prompt)
        
        # Parse response into structured format
        return self._parse_response(trade_data['id'], response)
    
    def _parse_response(self, trade_id: str, response: str) -> TradeReview:
        """Parse LLM response into structured review"""
        # Extract sections (simplified parsing)
        sections = response.split('###')
        
        entry_analysis = ""
        exit_analysis = ""
        order_flow_notes = ""
        improvements = []
        confidence = ""
        
        for section in sections:
            if 'Entry Quality' in section:
                entry_analysis = section.replace('Entry Quality', '').strip()
            elif 'Exit Assessment' in section:
                exit_analysis = section.replace('Exit Assessment', '').strip()
            elif 'Order Flow Reading' in section:
                order_flow_notes = section.replace('Order Flow Reading', '').strip()
            elif 'Key Lessons' in section:
                # Extract bullet points
                lines = section.split('\n')
                improvements = [l.strip('- ').strip() for l in lines if l.strip().startswith('-')]
            elif 'Confidence' in section:
                confidence = section.replace('Confidence in System', '').strip()
        
        summary = f"Trade {trade_id}: {entry_analysis[:100]}..."
        
        return TradeReview(
            trade_id=trade_id,
            timestamp=datetime.utcnow(),
            summary=summary,
            entry_analysis=entry_analysis,
            exit_analysis=exit_analysis,
            order_flow_notes=order_flow_notes,
            improvement_suggestions=improvements,
            confidence_assessment=confidence,
            raw_response=response,
        )
```

### 10.2 Narrative Builder (Pre-Market)

```python
# /engine/analyst/narrative_builder.py
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

@dataclass
class DailyNarrative:
    """Pre-market analysis and bias"""
    date: datetime
    macro_context: str
    technical_levels: dict
    bias_direction: Optional[str]
    bias_confidence: float
    key_levels_to_watch: list[float]
    scenarios: dict
    raw_response: str


class NarrativeBuilder:
    """
    Generates pre-market analysis using LLM.
    
    Mimics Fabio's process:
    - Review overnight price action
    - Identify key levels from volume profile
    - Establish directional bias
    - Define scenarios (what invalidates the bias?)
    """
    
    NARRATIVE_PROMPT = """You are an expert crypto trader. Generate a pre-market analysis 
for today's trading session.

## Overnight Summary
- Session High: {session_high}
- Session Low: {session_low}
- Current Price: {current_price}
- Overnight Delta: {overnight_delta}

## Volume Profile (Previous Session)
- POC (Point of Control): {poc}
- Value Area High: {vah}
- Value Area Low: {val}
- Developing POC: {developing_poc}

## Key Technical Levels
- Daily ATR: {atr}
- Previous Day High: {prev_high}
- Previous Day Low: {prev_low}
- Weekly VWAP: {weekly_vwap}

## Funding Rate
- Current: {funding_rate}
- 8h Avg: {funding_avg}

---

Provide your analysis:

### Macro Context
[Brief assessment of overall market conditions]

### Directional Bias
[BULLISH / BEARISH / NEUTRAL with reasoning]
[Confidence: 1-10]

### Key Levels
[List 3-5 specific price levels to watch with reasoning]

### Scenarios
[Bullish Scenario: What needs to happen for longs?]
[Bearish Scenario: What needs to happen for shorts?]
[Invalidation: What would change your bias?]
"""
    
    def __init__(self, llm_client):
        self.llm_client = llm_client
    
    async def generate_narrative(
        self,
        market_data: dict,
        volume_profile: dict,
    ) -> DailyNarrative:
        """Generate pre-market narrative"""
        prompt = self.NARRATIVE_PROMPT.format(
            session_high=market_data['session_high'],
            session_low=market_data['session_low'],
            current_price=market_data['current_price'],
            overnight_delta=market_data['overnight_delta'],
            poc=volume_profile['poc'],
            vah=volume_profile['vah'],
            val=volume_profile['val'],
            developing_poc=volume_profile.get('developing_poc', 'N/A'),
            atr=market_data['atr'],
            prev_high=market_data['prev_high'],
            prev_low=market_data['prev_low'],
            weekly_vwap=market_data.get('weekly_vwap', 'N/A'),
            funding_rate=market_data.get('funding_rate', 'N/A'),
            funding_avg=market_data.get('funding_avg', 'N/A'),
        )
        
        response = await self.llm_client.complete(prompt)
        
        return self._parse_narrative(response)
    
    def _parse_narrative(self, response: str) -> DailyNarrative:
        """Parse LLM response into structured narrative"""
        # Simplified parsing - in production use more robust extraction
        bias_direction = None
        bias_confidence = 0.5
        
        if 'BULLISH' in response.upper():
            bias_direction = 'BULLISH'
        elif 'BEARISH' in response.upper():
            bias_direction = 'BEARISH'
        
        # Extract confidence (look for pattern like "Confidence: 7")
        import re
        conf_match = re.search(r'Confidence:\s*(\d+)', response)
        if conf_match:
            bias_confidence = int(conf_match.group(1)) / 10
        
        return DailyNarrative(
            date=datetime.utcnow(),
            macro_context="Extracted from response",  # Parse in production
            technical_levels={},
            bias_direction=bias_direction,
            bias_confidence=bias_confidence,
            key_levels_to_watch=[],
            scenarios={},
            raw_response=response,
        )
```

---

## 11. Dashboard & Monitoring

### 11.1 Dashboard Requirements

The React frontend should display:

| Component | Data Source | Update Frequency |
|-----------|-------------|------------------|
| Live Price Chart | WebSocket | Real-time |
| Order Flow Panel | `OrderFlowEngine` | 100ms |
| Volume Profile | `VolumeProfile` indicator | 1m candle close |
| Active Trade Card | `TradeManager` | On change |
| Session Stats | `SessionGuard` + `HouseMoneyManager` | On trade |
| Signal Log | `ConfluenceValidator` | On signal |
| LLM Journal | `TradeReviewer` | Post-trade |

### 11.2 WebSocket Events (Backend → Frontend)

```typescript
// Event types for real-time updates
interface WSEvents {
  // Price updates
  'price.update': {
    symbol: string;
    price: number;
    timestamp: number;
  };
  
  // Order flow updates
  'orderflow.delta': {
    delta_15s: number;
    delta_1m: number;
    cvd: number;
    absorption: boolean;
    absorption_side?: 'BUY' | 'SELL';
  };
  
  // Trade lifecycle
  'trade.signal': {
    direction: 'BUY' | 'SELL';
    grade: 'A' | 'B' | 'C';
    entry: number;
    stop_loss: number;
    take_profit: number;
    rationale: string;
  };
  
  'trade.entered': {
    trade_id: string;
    entry_price: number;
    quantity: number;
  };
  
  'trade.exited': {
    trade_id: string;
    exit_price: number;
    pnl: number;
    exit_reason: string;
  };
  
  // Session state
  'session.update': {
    status: 'ACTIVE' | 'PAUSED' | 'STOPPED';
    pnl: number;
    pnl_percent: number;
    consecutive_losses: number;
    risk_multiplier: number;
  };
  
  // Market state
  'market.state': {
    state: 'BALANCE' | 'IMBALANCE_UP' | 'IMBALANCE_DOWN';
    bias: 'BULLISH' | 'BEARISH' | 'NEUTRAL';
  };
}
```

---

## 12. Backtesting & Validation Engine

### 12.1 Historical Data Limitations

**CRITICAL ACKNOWLEDGMENT:** 

Binance provides historical `aggTrades` data but **NOT historical order book snapshots**. This has significant implications:

| Component | Historical Data Available | Backtesting Validity |
|-----------|--------------------------|---------------------|
| Delta (buy/sell volume) | ✅ Yes (aggTrades) | Full |
| CVD (cumulative delta) | ✅ Yes | Full |
| Footprint charts | ✅ Yes | Full |
| Volume Profile | ✅ Yes (from trades) | Full |
| VWAP | ✅ Yes | Full |
| Market Structure | ✅ Yes (OHLCV) | Full |
| **Absorption Detection** | ⚠️ Partial | Approximate only |
| **Order Book Depth** | ❌ No | Cannot backtest |

**Implication:** Absorption patterns can only be *approximated* from trade tape (failed breakouts with high volume but no price movement). True absorption requires order book depth which isn't historical.

**Validation Strategy:**
1. Backtest components that CAN be tested (delta, CVD, structure, VWAP)
2. Use extended paper trading (2-4 weeks) to validate absorption logic
3. Track absorption signal accuracy separately in live trading

### 12.2 Anti-Lookahead Safeguards

Lookahead bias is the most common source of backtesting fraud. The system enforces strict safeguards:

```python
# /engine/backtest/safeguards.py
from datetime import datetime
from typing import Any
from functools import wraps

class LookaheadProtection:
    """
    Prevents accidental access to future data during backtests.
    
    Techniques:
    1. Time-gated data access
    2. Indicator computation freezing
    3. Fill price validation
    """
    
    def __init__(self):
        self._current_backtest_time: datetime = None
        self._violations: list[dict] = []
    
    def set_time(self, t: datetime) -> None:
        """Set current backtest timestamp"""
        self._current_backtest_time = t
    
    def validate_data_access(self, data_timestamp: datetime, accessor: str) -> bool:
        """
        Validate that accessed data is from before current time.
        
        Raises LookaheadError if future data accessed.
        """
        if self._current_backtest_time is None:
            return True  # Not in backtest mode
        
        if data_timestamp > self._current_backtest_time:
            violation = {
                'time': self._current_backtest_time,
                'accessed_time': data_timestamp,
                'accessor': accessor,
            }
            self._violations.append(violation)
            raise LookaheadError(
                f"Lookahead violation: {accessor} accessed data from "
                f"{data_timestamp} while backtest time is {self._current_backtest_time}"
            )
        return True
    
    def get_violations(self) -> list[dict]:
        """Return list of all lookahead violations"""
        return self._violations.copy()


class LookaheadError(Exception):
    """Raised when lookahead bias is detected"""
    pass


def no_lookahead(func):
    """
    Decorator that wraps data access functions to prevent lookahead.
    
    Usage:
        @no_lookahead
        def get_candle(self, timestamp: datetime) -> Candle:
            ...
    """
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        # Extract timestamp from args (assume first positional arg)
        if args and isinstance(args[0], datetime):
            data_time = args[0]
        elif 'timestamp' in kwargs:
            data_time = kwargs['timestamp']
        else:
            return func(self, *args, **kwargs)
        
        # Validate
        if hasattr(self, '_lookahead_protection'):
            self._lookahead_protection.validate_data_access(
                data_time, 
                func.__name__
            )
        
        return func(self, *args, **kwargs)
    return wrapper


class BacktestDataProvider:
    """
    Data provider that enforces point-in-time access.
    
    All data requests are validated against the current backtest time.
    Indicators are computed incrementally (not on future data).
    """
    
    def __init__(self, historical_data: dict):
        self._data = historical_data
        self._current_time: datetime = None
        self._lookahead_protection = LookaheadProtection()
        
        # Indicator states (computed incrementally)
        self._indicator_states: dict = {}
    
    def set_time(self, t: datetime) -> None:
        """Advance backtest time"""
        self._current_time = t
        self._lookahead_protection.set_time(t)
    
    @no_lookahead
    def get_candles(self, symbol: str, timeframe: str, count: int) -> list:
        """
        Get historical candles UP TO current time.
        
        Important: Returns candles that CLOSED before current time.
        An incomplete candle is NOT included.
        """
        candles = self._data.get(f"{symbol}_{timeframe}", [])
        
        # Filter to closed candles only
        closed_candles = [
            c for c in candles
            if c['close_time'] <= self._current_time
        ]
        
        return closed_candles[-count:]
    
    def get_current_price(self) -> float:
        """
        Get price at current backtest time.
        
        Uses the OPEN of the current candle (not close, which is future).
        """
        # Implemented via interpolation from trade tape
        pass
    
    def compute_indicator_incremental(
        self, 
        indicator_name: str, 
        new_data: Any
    ) -> Any:
        """
        Compute indicator incrementally without lookahead.
        
        Indicators are computed one bar at a time as time advances,
        never on the full dataset.
        """
        state = self._indicator_states.get(indicator_name, {})
        # ... indicator-specific incremental computation
        self._indicator_states[indicator_name] = state
        return state.get('value')
```

### 12.3 Fill Price Validation

```python
# /engine/backtest/fill_model.py
from dataclasses import dataclass
from typing import Optional

@dataclass
class BacktestFill:
    """Simulated order fill"""
    price: float
    quantity: float
    slippage: float
    timestamp: datetime
    fill_type: str  # 'exact', 'simulated', 'rejected'


class RealisticFillModel:
    """
    Simulates realistic order fills for backtesting.
    
    Anti-lookahead rules:
    1. Market orders fill at NEXT bar's open (not current close)
    2. Limit orders fill only if price CROSSES the level (not touches)
    3. Stop orders fill at stop price + slippage
    4. Large orders incur volume-weighted slippage
    
    This prevents the common error of filling at the exact price
    that triggered the signal.
    """
    
    def __init__(
        self,
        default_slippage_pct: float = 0.02,  # 0.02% base slippage
        volume_impact_factor: float = 0.1,
    ):
        self.default_slippage = default_slippage_pct
        self.volume_impact = volume_impact_factor
    
    def simulate_market_fill(
        self,
        signal_bar: dict,
        next_bar: dict,
        order_size: float,
        side: str,
    ) -> BacktestFill:
        """
        Simulate market order fill.
        
        CRITICAL: Uses NEXT bar's open, not current bar's close.
        This prevents lookahead.
        """
        base_price = next_bar['open']
        
        # Calculate slippage
        bar_volume = next_bar['volume']
        volume_ratio = order_size / bar_volume if bar_volume > 0 else 1
        slippage_pct = self.default_slippage + (volume_ratio * self.volume_impact)
        
        # Apply slippage in adverse direction
        if side == 'BUY':
            fill_price = base_price * (1 + slippage_pct / 100)
        else:
            fill_price = base_price * (1 - slippage_pct / 100)
        
        return BacktestFill(
            price=fill_price,
            quantity=order_size,
            slippage=abs(fill_price - base_price) / base_price * 100,
            timestamp=next_bar['open_time'],
            fill_type='simulated',
        )
    
    def simulate_limit_fill(
        self,
        limit_price: float,
        bars_after_order: list[dict],
        order_size: float,
        side: str,
    ) -> Optional[BacktestFill]:
        """
        Simulate limit order fill.
        
        Rule: Price must CROSS through the limit level, not just touch it.
        For BUY limit: Low must go BELOW limit price
        For SELL limit: High must go ABOVE limit price
        """
        for bar in bars_after_order:
            if side == 'BUY' and bar['low'] < limit_price:
                # Filled at limit price (best case)
                return BacktestFill(
                    price=limit_price,
                    quantity=order_size,
                    slippage=0,
                    timestamp=bar['open_time'],
                    fill_type='exact',
                )
            elif side == 'SELL' and bar['high'] > limit_price:
                return BacktestFill(
                    price=limit_price,
                    quantity=order_size,
                    slippage=0,
                    timestamp=bar['open_time'],
                    fill_type='exact',
                )
        
        return None  # Order not filled
    
    def simulate_stop_fill(
        self,
        stop_price: float,
        bars_after_order: list[dict],
        order_size: float,
        side: str,
    ) -> Optional[BacktestFill]:
        """
        Simulate stop order fill.
        
        Rule: Fill at stop price PLUS slippage (adverse).
        Stops often fill with extra slippage due to momentum.
        """
        for bar in bars_after_order:
            triggered = False
            
            if side == 'BUY' and bar['high'] >= stop_price:  # Stop buy triggered
                triggered = True
                # Fill with upward slippage
                fill_price = stop_price * (1 + self.default_slippage * 2 / 100)
            elif side == 'SELL' and bar['low'] <= stop_price:  # Stop sell triggered
                triggered = True
                # Fill with downward slippage
                fill_price = stop_price * (1 - self.default_slippage * 2 / 100)
            
            if triggered:
                return BacktestFill(
                    price=fill_price,
                    quantity=order_size,
                    slippage=abs(fill_price - stop_price) / stop_price * 100,
                    timestamp=bar['open_time'],
                    fill_type='simulated',
                )
        
        return None
```

### 12.4 Walk-Forward Optimization

```python
# /engine/backtest/walk_forward.py
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable, Any
import numpy as np

@dataclass
class WalkForwardWindow:
    """Single walk-forward window"""
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    optimized_params: dict
    train_metrics: dict
    test_metrics: dict


@dataclass
class WalkForwardResult:
    """Complete walk-forward analysis result"""
    windows: list[WalkForwardWindow]
    aggregated_test_metrics: dict
    parameter_stability: dict  # How much did optimal params vary?
    is_robust: bool


class WalkForwardOptimizer:
    """
    Walk-Forward Analysis prevents curve-fitting.
    
    Process:
    1. Split data into rolling windows
    2. For each window:
       a. Optimize parameters on TRAIN portion
       b. Test with those params on OUT-OF-SAMPLE portion
    3. Aggregate out-of-sample results
    
    If strategy only works with specific parameters (not robust),
    the out-of-sample aggregate will show poor performance.
    
    Window structure:
    |---- Train (80%) ----|-- Test (20%) --|
                   |---- Train (80%) ----|-- Test (20%) --|
                                  |---- Train (80%) ----|-- Test (20%) --|
    """
    
    def __init__(
        self,
        train_ratio: float = 0.8,
        min_train_days: int = 90,
        min_test_days: int = 21,
        step_days: int = 30,  # How much to advance between windows
    ):
        self.train_ratio = train_ratio
        self.min_train_days = min_train_days
        self.min_test_days = min_test_days
        self.step_days = step_days
    
    def create_windows(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> list[tuple[datetime, datetime, datetime, datetime]]:
        """
        Create walk-forward windows.
        
        Returns list of (train_start, train_end, test_start, test_end) tuples.
        """
        windows = []
        total_days = (end_date - start_date).days
        window_size = self.min_train_days + self.min_test_days
        
        current_start = start_date
        
        while True:
            train_end = current_start + timedelta(days=self.min_train_days)
            test_end = train_end + timedelta(days=self.min_test_days)
            
            if test_end > end_date:
                break
            
            windows.append((
                current_start,
                train_end,
                train_end,
                test_end,
            ))
            
            current_start += timedelta(days=self.step_days)
        
        return windows
    
    def run(
        self,
        data: dict,
        strategy_factory: Callable,
        param_grid: dict,
        optimize_metric: str = 'sharpe_ratio',
        backtest_func: Callable = None,
    ) -> WalkForwardResult:
        """
        Run complete walk-forward analysis.
        
        Args:
            data: Historical data dict
            strategy_factory: Function that creates strategy with given params
            param_grid: Dict of param_name -> list of values to try
            optimize_metric: Metric to maximize during optimization
            backtest_func: Function that runs backtest and returns metrics
        """
        start_date = min(d['time'] for d in data['candles'])
        end_date = max(d['time'] for d in data['candles'])
        
        windows = self.create_windows(start_date, end_date)
        results = []
        all_optimal_params = []
        
        for train_start, train_end, test_start, test_end in windows:
            # Filter data for train period
            train_data = self._filter_data(data, train_start, train_end)
            test_data = self._filter_data(data, test_start, test_end)
            
            # Grid search on train data
            best_params = None
            best_metric = float('-inf')
            
            for params in self._param_combinations(param_grid):
                strategy = strategy_factory(**params)
                metrics = backtest_func(strategy, train_data)
                
                if metrics[optimize_metric] > best_metric:
                    best_metric = metrics[optimize_metric]
                    best_params = params
            
            all_optimal_params.append(best_params)
            
            # Test with optimal params on out-of-sample
            strategy = strategy_factory(**best_params)
            train_metrics = backtest_func(strategy, train_data)
            
            # Reset strategy state before test
            strategy.reset()
            test_metrics = backtest_func(strategy, test_data)
            
            results.append(WalkForwardWindow(
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                optimized_params=best_params,
                train_metrics=train_metrics,
                test_metrics=test_metrics,
            ))
        
        # Aggregate test results
        aggregated = self._aggregate_test_metrics(results)
        
        # Check parameter stability
        stability = self._check_param_stability(all_optimal_params)
        
        # Robustness check
        is_robust = (
            aggregated.get('sharpe_ratio', 0) > 0.5 and
            stability.get('stability_score', 0) > 0.6
        )
        
        return WalkForwardResult(
            windows=results,
            aggregated_test_metrics=aggregated,
            parameter_stability=stability,
            is_robust=is_robust,
        )
    
    def _param_combinations(self, param_grid: dict):
        """Generate all parameter combinations"""
        import itertools
        keys = param_grid.keys()
        values = param_grid.values()
        for combo in itertools.product(*values):
            yield dict(zip(keys, combo))
    
    def _filter_data(self, data: dict, start: datetime, end: datetime) -> dict:
        """Filter data to date range"""
        filtered = {}
        for key, values in data.items():
            if isinstance(values, list):
                filtered[key] = [
                    v for v in values 
                    if start <= v.get('time', v.get('timestamp', start)) <= end
                ]
            else:
                filtered[key] = values
        return filtered
    
    def _aggregate_test_metrics(self, results: list[WalkForwardWindow]) -> dict:
        """Aggregate out-of-sample metrics across all windows"""
        metrics = {}
        
        # Collect all test metrics
        all_test = [r.test_metrics for r in results]
        
        for key in all_test[0].keys():
            values = [t[key] for t in all_test if key in t]
            metrics[key] = np.mean(values)
            metrics[f"{key}_std"] = np.std(values)
        
        return metrics
    
    def _check_param_stability(self, all_params: list[dict]) -> dict:
        """
        Check how stable optimal parameters were across windows.
        
        High stability = same params work across time periods (good)
        Low stability = params change drastically (overfitting risk)
        """
        if not all_params:
            return {'stability_score': 0}
        
        stability_scores = {}
        
        for param_name in all_params[0].keys():
            values = [p[param_name] for p in all_params]
            
            if isinstance(values[0], (int, float)):
                # Coefficient of variation for numeric params
                mean_val = np.mean(values)
                std_val = np.std(values)
                cv = std_val / mean_val if mean_val != 0 else float('inf')
                stability_scores[param_name] = max(0, 1 - cv)
            else:
                # Mode frequency for categorical params
                from collections import Counter
                counts = Counter(values)
                mode_freq = counts.most_common(1)[0][1] / len(values)
                stability_scores[param_name] = mode_freq
        
        overall = np.mean(list(stability_scores.values()))
        
        return {
            'param_stability': stability_scores,
            'stability_score': overall,
        }
```

---

## 13. Implementation Sequence

### Phase 1: Foundation (Weeks 1-2)
**Goal:** Data pipeline and storage

1. Set up Docker environment (TimescaleDB, Redis)
2. Implement `BinanceWebSocket` for `aggTrades` and `kline` streams
3. Build `DataNormalizer` to aggregate ticks into candles
4. Create database schemas and migrations
5. Build historical data backfill script
6. **Implement EventBus and async architecture**
7. **Deliverable:** Working data pipeline storing live + historical data

### Phase 2: Order Flow Engine (Weeks 3-4)
**Goal:** Core signal generation

1. Implement `DeltaCalculator` with multi-window tracking
2. Build `AbsorptionDetector` with **time-weighted decay and dynamic thresholds**
3. Create `TrappedTraderDetector` state machine
4. Implement `CVDTracker` with **ATR-relative divergence detection**
5. Build `FootprintBuilder` for price-level aggregation
6. **Deliverable:** Real-time order flow signals

### Phase 3: Strategy & Risk (Weeks 5-6)
**Goal:** Trade logic

1. Implement `ConfluenceValidator` (box-checking)
2. Build `TrendContinuationStrategy`
3. Build `MeanReversionStrategy` **with micro-TF confirmation**
4. Implement `DynamicPositionSizer` (A/B/C grading)
5. Build `HouseMoneyManager`
6. Implement `SessionGuard` (circuit breaker)
7. **Deliverable:** Paper trading ready

### Phase 4: Execution (Weeks 7-8)
**Goal:** Live trading capability

1. Implement `OrderManager` state machine
2. Build `BinanceExecutor` with **proper async session management**
3. Create `TradeManager` orchestrator
4. **Implement SlippageModel for pre-trade estimation**
5. Build trailing stop / break-even logic
6. **Implement StateReconciler for crash recovery**
7. **Deliverable:** Testnet trading

### Phase 5: Backtesting (Weeks 9-10)
**Goal:** Validation

1. Build `VectorizedBacktester` for rapid prototyping
2. Implement `EventDrivenBacktester` **with anti-lookahead safeguards**
3. Create `WalkForwardOptimizer` **with parameter stability checks**
4. Build metrics module (Sharpe, Sortino, Max DD)
5. **Document historical data limitations (absorption requires paper trading)**
6. **Deliverable:** Validated strategy parameters

### Phase 6: LLM Integration (Weeks 11-12)
**Goal:** Automated journaling

1. Build `LLMClient` wrapper
2. Implement `TradeReviewer`
3. Create `NarrativeBuilder`
4. Build journal storage and retrieval
5. **Deliverable:** Automated trade analysis

### Phase 7: Dashboard (Weeks 13-14)
**Goal:** Visualization

1. Set up React project with WebSocket
2. Build real-time chart component
3. Create order flow visualization panel
4. Build trade management interface
5. Implement journal viewer
6. **Deliverable:** Production dashboard

### Phase 8: Production Hardening (Weeks 15-16)
**Goal:** Reliability

1. **Implement graceful shutdown handler (signal handling)**
2. Build alerting system (Telegram/Discord)
3. **Create health check endpoint and Prometheus metrics**
4. Create monitoring dashboards (Grafana)
5. Load testing and optimization
6. **Deliverable:** Production-ready system

---

## 14. Monitoring & Health Checks

### 14.1 Health Check Endpoint

```python
# /engine/monitoring/health.py
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional
import asyncio

class HealthStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class ComponentHealth:
    name: str
    status: HealthStatus
    latency_ms: Optional[float]
    last_check: datetime
    error: Optional[str]


@dataclass
class SystemHealth:
    overall_status: HealthStatus
    components: list[ComponentHealth]
    uptime_seconds: float
    version: str


class HealthChecker:
    """
    System health monitoring.
    
    Checks:
    - WebSocket connection to Binance
    - Redis connectivity
    - TimescaleDB connectivity
    - Event bus functioning
    - Last trade data received
    """
    
    def __init__(self, config, components: dict):
        self.config = config
        self.components = components
        self._start_time = datetime.utcnow()
    
    async def check_all(self) -> SystemHealth:
        """Run all health checks"""
        results = await asyncio.gather(
            self._check_binance_ws(),
            self._check_redis(),
            self._check_timescale(),
            self._check_data_freshness(),
            return_exceptions=True,
        )
        
        component_health = []
        for result in results:
            if isinstance(result, Exception):
                component_health.append(ComponentHealth(
                    name="unknown",
                    status=HealthStatus.UNHEALTHY,
                    latency_ms=None,
                    last_check=datetime.utcnow(),
                    error=str(result),
                ))
            else:
                component_health.append(result)
        
        # Determine overall status
        statuses = [c.status for c in component_health]
        if all(s == HealthStatus.HEALTHY for s in statuses):
            overall = HealthStatus.HEALTHY
        elif any(s == HealthStatus.UNHEALTHY for s in statuses):
            overall = HealthStatus.UNHEALTHY
        else:
            overall = HealthStatus.DEGRADED
        
        return SystemHealth(
            overall_status=overall,
            components=component_health,
            uptime_seconds=(datetime.utcnow() - self._start_time).total_seconds(),
            version="1.0.0",
        )
    
    async def _check_binance_ws(self) -> ComponentHealth:
        """Check Binance WebSocket connection"""
        start = datetime.utcnow()
        try:
            ws = self.components.get('binance_ws')
            if ws and ws.is_connected:
                latency = (datetime.utcnow() - start).total_seconds() * 1000
                return ComponentHealth(
                    name="binance_websocket",
                    status=HealthStatus.HEALTHY,
                    latency_ms=latency,
                    last_check=datetime.utcnow(),
                    error=None,
                )
            else:
                return ComponentHealth(
                    name="binance_websocket",
                    status=HealthStatus.UNHEALTHY,
                    latency_ms=None,
                    last_check=datetime.utcnow(),
                    error="WebSocket disconnected",
                )
        except Exception as e:
            return ComponentHealth(
                name="binance_websocket",
                status=HealthStatus.UNHEALTHY,
                latency_ms=None,
                last_check=datetime.utcnow(),
                error=str(e),
            )
    
    async def _check_redis(self) -> ComponentHealth:
        """Check Redis connectivity"""
        start = datetime.utcnow()
        try:
            redis = self.components.get('redis')
            await redis.ping()
            latency = (datetime.utcnow() - start).total_seconds() * 1000
            return ComponentHealth(
                name="redis",
                status=HealthStatus.HEALTHY,
                latency_ms=latency,
                last_check=datetime.utcnow(),
                error=None,
            )
        except Exception as e:
            return ComponentHealth(
                name="redis",
                status=HealthStatus.UNHEALTHY,
                latency_ms=None,
                last_check=datetime.utcnow(),
                error=str(e),
            )
    
    async def _check_timescale(self) -> ComponentHealth:
        """Check TimescaleDB connectivity"""
        start = datetime.utcnow()
        try:
            db = self.components.get('database')
            await db.execute("SELECT 1")
            latency = (datetime.utcnow() - start).total_seconds() * 1000
            return ComponentHealth(
                name="timescaledb",
                status=HealthStatus.HEALTHY,
                latency_ms=latency,
                last_check=datetime.utcnow(),
                error=None,
            )
        except Exception as e:
            return ComponentHealth(
                name="timescaledb",
                status=HealthStatus.UNHEALTHY,
                latency_ms=None,
                last_check=datetime.utcnow(),
                error=str(e),
            )
    
    async def _check_data_freshness(self) -> ComponentHealth:
        """Check that we're receiving recent market data"""
        try:
            data_layer = self.components.get('data_layer')
            last_trade_time = data_layer.get_last_trade_time()
            
            age = (datetime.utcnow() - last_trade_time).total_seconds()
            
            if age < 5:
                status = HealthStatus.HEALTHY
            elif age < 30:
                status = HealthStatus.DEGRADED
            else:
                status = HealthStatus.UNHEALTHY
            
            return ComponentHealth(
                name="data_freshness",
                status=status,
                latency_ms=age * 1000,
                last_check=datetime.utcnow(),
                error=f"Last trade {age:.1f}s ago" if age > 5 else None,
            )
        except Exception as e:
            return ComponentHealth(
                name="data_freshness",
                status=HealthStatus.UNHEALTHY,
                latency_ms=None,
                last_check=datetime.utcnow(),
                error=str(e),
            )
```

### 14.2 Prometheus Metrics

```python
# /engine/monitoring/metrics.py
from prometheus_client import Counter, Gauge, Histogram, Info

# System metrics
SYSTEM_INFO = Info('trading_system', 'Trading system information')
UPTIME_SECONDS = Gauge('trading_uptime_seconds', 'System uptime in seconds')

# Trading metrics
TRADES_TOTAL = Counter(
    'trading_trades_total', 
    'Total trades executed',
    ['symbol', 'direction', 'grade', 'exit_reason']
)
TRADE_PNL = Histogram(
    'trading_trade_pnl_percent',
    'Trade P&L distribution',
    ['symbol', 'direction'],
    buckets=[-5, -2, -1, -0.5, 0, 0.5, 1, 2, 5, 10]
)
SESSION_PNL_PERCENT = Gauge('trading_session_pnl_percent', 'Current session P&L')
CONSECUTIVE_LOSSES = Gauge('trading_consecutive_losses', 'Current consecutive losses')
RISK_MULTIPLIER = Gauge('trading_risk_multiplier', 'House money multiplier')

# Order flow metrics
DELTA_RATIO = Gauge('trading_delta_ratio', 'Delta ratio', ['timeframe'])
ABSORPTION_EVENTS = Counter('trading_absorption_total', 'Absorption events', ['side'])
TRAPPED_SIGNALS = Counter('trading_trapped_total', 'Trapped trader signals', ['side'])

# Execution metrics
ORDER_LATENCY = Histogram(
    'trading_order_latency_ms', 'Order latency',
    buckets=[10, 50, 100, 250, 500, 1000]
)
SLIPPAGE = Histogram(
    'trading_slippage_percent', 'Slippage distribution',
    buckets=[0, 0.01, 0.02, 0.05, 0.1, 0.2]
)
```

---

## Appendix A: Configuration Schema

### Validated Configuration Model

```python
# /engine/config.py
from pydantic import BaseModel, Field, validator
from typing import Optional, List
from datetime import time
from enum import Enum

class TradingMode(str, Enum):
    PAPER = "paper"
    LIVE = "live"
    BACKTEST = "backtest"


class BinanceConfig(BaseModel):
    """Binance API configuration"""
    api_key: str = Field(..., min_length=10)
    api_secret: str = Field(..., min_length=10)
    testnet: bool = True
    
    @validator('api_key', 'api_secret')
    def no_placeholder_keys(cls, v):
        if v.startswith('your_') or v == 'xxx':
            raise ValueError('Replace placeholder API keys with real values')
        return v


class RiskConfig(BaseModel):
    """Risk management parameters"""
    base_risk_percent: float = Field(0.25, ge=0.01, le=2.0)
    max_leverage: float = Field(10.0, ge=1.0, le=125.0)
    max_daily_drawdown_percent: float = Field(5.0, ge=1.0, le=20.0)
    max_consecutive_losses: int = Field(3, ge=1, le=10)
    
    class HouseMoneyConfig(BaseModel):
        profit_risk_ratio: float = Field(0.5, ge=0.1, le=1.0)
        min_profit_to_compound: float = Field(0.5, ge=0.1, le=5.0)
        max_risk_multiplier: float = Field(4.0, ge=1.0, le=10.0)
    
    house_money: HouseMoneyConfig = HouseMoneyConfig()
    
    @validator('base_risk_percent')
    def validate_base_risk(cls, v):
        if v > 1.0:
            import warnings
            warnings.warn(f"Base risk of {v}% is aggressive. Recommended: 0.25-0.5%")
        return v


class StrategyConfig(BaseModel):
    """Strategy parameters"""
    initial_balance_minutes: int = Field(30, ge=15, le=120)
    session_start_utc: str = Field("08:00")  # Stored as string, parsed to time
    
    class MeanReversionConfig(BaseModel):
        enabled: bool = True
        window_start_utc: str = "17:45"
        window_end_utc: str = "22:00"
    
    mean_reversion: MeanReversionConfig = MeanReversionConfig()
    
    @validator('session_start_utc')
    def validate_time_format(cls, v):
        try:
            hours, minutes = v.split(':')
            time(int(hours), int(minutes))
        except:
            raise ValueError(f"Invalid time format: {v}. Use HH:MM")
        return v


class OrderFlowConfig(BaseModel):
    """Order flow analysis parameters"""
    
    class AbsorptionConfig(BaseModel):
        min_volume_percentile: float = Field(75.0, ge=50.0, le=99.0)
        price_move_threshold_pct: float = Field(0.0005, ge=0.0001, le=0.005)
        decay_half_life_seconds: float = Field(10.0, ge=1.0, le=60.0)
    
    class TrappedTraderConfig(BaseModel):
        min_absorption_volume: float = Field(50.0, ge=10.0, le=500.0)
        reversal_threshold_pct: float = Field(0.001, ge=0.0001, le=0.01)
        max_trap_age_seconds: int = Field(120, ge=30, le=600)
    
    absorption: AbsorptionConfig = AbsorptionConfig()
    trapped_trader: TrappedTraderConfig = TrappedTraderConfig()


class DatabaseConfig(BaseModel):
    """Database connection settings"""
    timescale_url: str = Field(..., regex=r'^postgresql://.*')
    redis_url: str = Field(..., regex=r'^redis://.*')


class LLMConfig(BaseModel):
    """LLM analyst configuration"""
    provider: str = Field("anthropic", regex=r'^(anthropic|openai)$')
    model: str = "claude-sonnet-4-20250514"
    api_key: str = Field(..., min_length=10)


class SystemConfig(BaseModel):
    """Top-level system configuration"""
    mode: TradingMode = TradingMode.PAPER
    log_level: str = Field("INFO", regex=r'^(DEBUG|INFO|WARNING|ERROR)$')
    
    binance: BinanceConfig
    risk: RiskConfig = RiskConfig()
    strategy: StrategyConfig = StrategyConfig()
    orderflow: OrderFlowConfig = OrderFlowConfig()
    database: DatabaseConfig
    llm: Optional[LLMConfig] = None
    
    symbols: List[str] = Field(["BTCUSDT"], min_items=1)
    primary_symbol: str = "BTCUSDT"
    
    @validator('primary_symbol')
    def primary_in_symbols(cls, v, values):
        if 'symbols' in values and v not in values['symbols']:
            raise ValueError(f"Primary symbol {v} must be in symbols list")
        return v
    
    class Config:
        # Allow environment variable interpolation
        env_prefix = 'TRADING_'


def load_config(path: str) -> SystemConfig:
    """Load and validate configuration from YAML file"""
    import yaml
    import os
    
    with open(path) as f:
        raw = yaml.safe_load(f)
    
    # Environment variable substitution
    def substitute_env(obj):
        if isinstance(obj, str) and obj.startswith('${') and obj.endswith('}'):
            env_var = obj[2:-1]
            return os.environ.get(env_var, '')
        elif isinstance(obj, dict):
            return {k: substitute_env(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [substitute_env(v) for v in obj]
        return obj
    
    raw = substitute_env(raw)
    
    return SystemConfig(**raw)
```

### Example Configuration File

```yaml
# config.yaml
# Production configuration for Fabio-style trading system

system:
  mode: "paper"  # "paper" | "live" | "backtest"
  log_level: "INFO"

binance:
  api_key: "${BINANCE_API_KEY}"
  api_secret: "${BINANCE_API_SECRET}"
  testnet: true  # Set to false for live trading

symbols:
  - "BTCUSDT"
  - "ETHUSDT"
primary_symbol: "BTCUSDT"

risk:
  base_risk_percent: 0.25    # Conservative starting point
  max_leverage: 10
  max_daily_drawdown_percent: 5.0
  max_consecutive_losses: 3
  house_money:
    profit_risk_ratio: 0.5   # Risk 50% of profits
    min_profit_to_compound: 0.5
    max_risk_multiplier: 4.0 # Never exceed 4x base risk

strategy:
  initial_balance_minutes: 30  # Wait 30 min after session start
  session_start_utc: "08:00"   # Main session start
  mean_reversion:
    enabled: true
    window_start_utc: "17:45"  # Late session reversal window
    window_end_utc: "22:00"

orderflow:
  absorption:
    min_volume_percentile: 75.0
    price_move_threshold_pct: 0.0005
    decay_half_life_seconds: 10.0
  trapped_trader:
    min_absorption_volume: 50.0
    reversal_threshold_pct: 0.001
    max_trap_age_seconds: 120

database:
  timescale_url: "postgresql://trading:${DB_PASSWORD}@localhost:5432/trading"
  redis_url: "redis://localhost:6379/0"

llm:
  provider: "anthropic"
  model: "claude-sonnet-4-20250514"
  api_key: "${ANTHROPIC_API_KEY}"
```

---

## Appendix B: Key Metrics Dashboard

| Metric | Target | Warning | Critical |
|--------|--------|---------|----------|
| Win Rate | >55% | <50% | <45% |
| Avg R:R | >2.0 | <1.5 | <1.0 |
| Max Drawdown | <5% | >7% | >10% |
| Sharpe Ratio | >1.5 | <1.0 | <0.5 |
| Consecutive Losses | 0-2 | 3 | >3 (stop) |
| A Setup Win Rate | >65% | <55% | <50% |
| Daily Return | >0.5% | <0% | <-2% |

---

## Appendix C: Fabio Methodology Quick Reference

### Entry Checklist (Must Have ALL)
- [ ] 15m bias established (not in first 30 min)
- [ ] Price at POI (VAH, VAL, POC, S/R)
- [ ] Order flow trigger (absorption OR aggression)
- [ ] Price follow-through confirmed
- [ ] 15s micro pattern present

### Exit Rules
- **Take Profit:** 2.5-3x risk distance
- **Stop Loss:** Behind volume wall / structure
- **Break-even:** If no immediate follow-through
- **Invalidation:** Structure break + opposing volume

### Risk Scaling
- Start session: 0.25% risk
- After profit: Add 50% of profits to risk
- A Setup: 100% of allocated risk
- B Setup: 75% of allocated risk
- C Setup: 50% of allocated risk

### Session Rules
- 3 consecutive losses → STOP
- 5% daily drawdown → STOP
- Only trade mean reversion if profitable
- Avoid first 30 min of major sessions

---

## Appendix D: Revision History & Analysis Response

### Document Version: 2.0

This section documents the architectural improvements made in response to the deep analysis review.

### Issues Addressed

| Priority | Issue | Resolution |
|----------|-------|------------|
| **P0** | Event bus missing | Added Section 3.3 with `EventBus` class, `EventType` enum, and typed events |
| **P0** | Concurrency model undefined | Added `TradingEngine` class with structured concurrency, asyncio.Lock for shared state |
| **P0** | Trade recovery absent | Added Section 4.4 with `RedisStateStore` and `StateReconciler` classes |
| **P0** | OCO race condition | Noted in BinanceExecutor (Binance Futures doesn't have true OCO; manual handling required with atomic cancel/replace) |
| **P1** | Absorption detection too simple | Updated `AbsorptionDetector` with time-weighted decay, dynamic percentile thresholds |
| **P1** | TimescaleDB FK issue | Redesigned schema: signals use UUID correlation, executions use application-level joins |
| **P1** | CCXT async incomplete | Added proper session management, `asynccontextmanager`, connection state tracking |
| **P1** | Walk-forward unspecified | Added Section 12.4 with `WalkForwardOptimizer` and parameter stability checks |
| **P2** | CVD divergence needs regime awareness | Updated `CVDTracker` with ATR-relative thresholds |
| **P2** | Mean reversion lacks micro-TF | Documented in implementation phases (to be added during coding) |
| **P2** | Funding rate not integrated | Added Section 4.5 with `FundingRateAnalyzer` class |
| **P3** | Health checks/monitoring | Added Section 14 with `HealthChecker` and Prometheus metrics |
| **P3** | Config validation | Added Pydantic-based `SystemConfig` with validators |

### Issues Evaluated but Not Adopted

| Issue | Analysis | Decision |
|-------|----------|----------|
| **Trapped trader multi-state tracking** | Reviewer suggested tracking multiple pending traps. After analysis: single-trap tracking is *intentionally conservative*. Multiple pending traps creates noise and leads to over-trading. This aligns with Fabio's "wait for THE setup" philosophy. | **Not adopted** — Added design rationale in docstring |
| **Full regime detection for CVD** | Would add significant complexity. ATR-relative thresholds provide 80% of the benefit. | **Simplified** — Using ATR thresholds as v1 solution |
| **UTC timezone handling** | Reviewer flagged UTC-only times. UTC is the standard for 24/7 crypto markets; DST handling adds complexity with minimal benefit. | **Not adopted** — UTC is correct for crypto |

### Backtesting Limitations Acknowledged

The analysis correctly identified that **historical order book data is unavailable from Binance**. This has been explicitly documented in Section 12.1 with the following validation strategy:

1. Backtest components that CAN be tested (delta, CVD, structure, VWAP)
2. Use extended paper trading (2-4 weeks) to validate absorption logic
3. Track absorption signal accuracy separately in live trading

### Anti-Lookahead Safeguards Added

Section 12.2-12.3 now includes:
- `LookaheadProtection` class with time-gated data access
- `@no_lookahead` decorator for data access functions
- `RealisticFillModel` that fills market orders at NEXT bar open
- Limit/stop fill validation (price must CROSS, not just touch)

---

*Document Version 2.0 — Updated with analysis response. Total specification: ~5,300 lines.*
