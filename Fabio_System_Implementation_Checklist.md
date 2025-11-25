# Fabio Trading System - Implementation Checklist

This document serves as the master checklist for implementing the Fabio Trading System. It is derived directly from the `Fabio Trading System Architecture.md` plan. All future AI agents and developers must follow this sequence to ensure architectural integrity.

---

## Phase 1: Foundation (Data & Infrastructure)
**Goal:** Establish a robust data pipeline and storage layer.

- [ ] **Environment Setup**
    - [ ] Verify Docker environment (TimescaleDB, Redis).
    - [ ] Install dependencies from `requirements.txt`.
    - [ ] Configure `config.py` with environment variables (API keys, DB URLs).

- [ ] **Database Schema**
    - [ ] Execute `database/schema.sql` to create TimescaleDB hypertables (`trades_raw`, `candles`, `volume_profile`, `signals`, `session_state`) and standard tables (`executions`, `trade_journal`).
    - [ ] Verify Redis connection and persistence configuration.

- [ ] **Data Ingestion (`engine/data`)**
    - [ ] Implement `BinanceWebSocket` (`binance_ws.py`) to handle `aggTrades`, `depth@100ms`, `kline_1m`, `kline_15m`, and `markPrice` streams.
    - [ ] Implement `DataNormalizer` (`data_normalizer.py`) to aggregate ticks into OHLCV candles if needed, or process raw streams.
    - [ ] Implement `HistoricalLoader` (`historical_loader.py`) to fetch and store historical data for backtesting.
    - [ ] Implement `FundingRateTracker` (`funding_rate.py` - *Note: Create this file if missing*) to track funding rates and sentiment.

- [ ] **Event Architecture**
    - [ ] Implement `EventBus` in `engine/events.py` (Asyncio Queue based, support for multiple subscribers).
    - [ ] Define all `EventType` enums (Data, OrderFlow, Strategy, Execution, Risk, System).

---

## Phase 2: Order Flow Engine
**Goal:** Translate raw data into actionable order flow signals.

- [ ] **Delta Calculation (`engine/orderflow`)**
    - [ ] Implement `DeltaCalculator` (`delta_calculator.py`) to compute rolling delta for 5s, 15s, 1m, 5m windows.
    - [ ] Implement CVD (Cumulative Volume Delta) tracking.

- [ ] **Absorption Detection**
    - [ ] Implement `AbsorptionDetector` (`absorption_detector.py`).
    - [ ] Logic: Detect high volume (time-weighted) with minimal price movement.
    - [ ] Implement dynamic thresholds based on recent volume percentiles.

- [ ] **Trapped Trader Detection**
    - [ ] Implement `TrappedTraderDetector` (`trapped_trader.py`).
    - [ ] Logic: Breakout attempt + Absorption + Reversal = Trapped Traders.
    - [ ] Ensure only ONE pending trap state is tracked at a time (conservative approach).

- [ ] **CVD Divergence**
    - [ ] Implement `CVDTracker` (`cvd_tracker.py`).
    - [ ] Logic: Detect Price vs CVD divergence using ATR-relative thresholds.

- [ ] **Footprint Construction**
    - [ ] Implement `FootprintBuilder` (`footprint_builder.py`) to aggregate volume at price levels for each candle.

---

## Phase 3: Strategy & Risk
**Goal:** Implement the core trading logic and safety mechanisms.

- [ ] **Confluence Validator (`engine/strategy`)**
    - [ ] Implement `ConfluenceValidator` (`confluence_validator.py`).
    - [ ] Logic: Check 5 boxes (Bias, POI, OrderFlow, Follow-through, Micro-confirmation).
    - [ ] Implement Grading System (A/B/C/Invalid).

- [ ] **Trend Continuation Strategy**
    - [ ] Implement `TrendContinuationStrategy` (`trend_continuation.py`).
    - [ ] Logic: 15m Bias + Pullback to POI + Absorption/Trap Signal.
    - [ ] Implement Initial Balance filter (no trading first 30 mins).

- [ ] **Mean Reversion Strategy**
    - [ ] Implement `MeanReversionStrategy` (`mean_reversion.py`).
    - [ ] Logic: Late session only, Price at VWAP 2nd/3rd SD, Micro-TF confirmation.

- [ ] **Risk Management (`engine/risk`)**
    - [ ] Implement `DynamicPositionSizer` (`position_sizer.py`) based on Setup Grade (A=100%, B=75%, C=50%).
    - [ ] Implement `HouseMoneyManager` (`house_money.py`) to compound profits (risk portion of session P&L).
    - [ ] Implement `SessionGuard` (`session_guard.py`) for circuit breakers (3 consecutive losses, max daily drawdown).

---

## Phase 4: Execution Engine
**Goal:** robust, safe, and reliable trade execution.

- [ ] **Order Management (`engine/execution`)**
    - [ ] Implement `OrderManager` (`order_manager.py`) state machine (Pending -> Submitted -> Filled/Cancelled).
    - [ ] Implement OCO (One-Cancels-Other) logic for SL/TP pairs.

- [ ] **Binance Executor**
    - [ ] Implement `BinanceExecutor` (`binance_executor.py`) using `ccxt`.
    - [ ] Implement async session management.
    - [ ] Implement `SlippageModel` (`slippage_model.py`) for pre-trade estimation.

- [ ] **Trade Orchestration**
    - [ ] Implement `TradeManager` (`trade_manager.py` - *Note: Create if missing*).
    - [ ] Logic: Coordinate Signal -> Risk Check -> Entry -> OCO Exit.
    - [ ] Implement Break-even move logic.
    - [ ] Implement Invalidation exit logic (immediate close on structure break).

- [ ] **State Reconciliation**
    - [ ] Implement `StateReconciler` (`redis_store.py` or separate) to sync local state with Binance after restart.

---

## Phase 5: Backtesting & Validation
**Goal:** Validate strategy logic (with known data limitations).

- [ ] **Backtest Engine (`engine/backtest`)**
    - [ ] Implement `VectorizedBacktester` (`vectorized_engine.py`) for fast initial checks.
    - [ ] Implement `EventDrivenBacktester` (`event_driven_engine.py`) for realistic simulation.
    - [ ] **CRITICAL:** Implement `LookaheadProtection` (`safeguards.py`) to prevent future data access.
    - [ ] Implement `RealisticFillModel` (`fill_model.py`) to simulate slippage and fill probability.

- [ ] **Optimization**
    - [ ] Implement `WalkForwardOptimizer` (`walk_forward.py`) to test parameter stability over time.

---

## Phase 6: LLM Integration
**Goal:** Automated analysis and journaling.

- [ ] **LLM Client (`engine/analyst`)**
    - [ ] Implement `LLMClient` (`llm_client.py`) to interface with OpenAI/Anthropic.

- [ ] **Analysis Modules**
    - [ ] Implement `TradeReviewer` (`trade_reviewer.py`) for post-trade analysis.
    - [ ] Implement `NarrativeBuilder` (`narrative_builder.py`) for pre-market bias generation.

---

## Phase 7: Dashboard
**Goal:** Real-time visualization and monitoring.

- [ ] **Frontend (`dashboard/`)**
    - [ ] Initialize React app.
    - [ ] Implement WebSocket client to receive `price.update`, `orderflow.delta`, `trade.signal`.
    - [ ] Build components: Live Chart, Order Flow Panel, Active Trade Card, Session Stats.

---

## Phase 8: Production Hardening
**Goal:** Reliability and safety.

- [ ] **System Health**
    - [ ] Implement `HealthChecker` (`monitoring/health.py` - *Note: Create directory/file*).
    - [ ] Checks: WS connection, Redis, DB, Event Bus.

- [ ] **Deployment**
    - [ ] Finalize `docker-compose.yml`.
    - [ ] Implement graceful shutdown handling in `main.py`.
    - [ ] Set up logging and alerting.

---

## Usage Guide for Agents
1.  **Check this file first** to see the current progress.
2.  **Mark items as completed** `[x]` only when the code is implemented and verified.
3.  **Do not skip steps** - dependencies matter (e.g., Data Layer must exist before Order Flow).
4.  **Reference the Architecture Document** for specific logic implementation details (formulas, thresholds, etc.).
