# Fabio Trading System

## Overview
The **Fabio Trading System** is a high-frequency, event-driven algorithmic trading platform designed for Binance Futures. It leverages advanced order flow analysis, automated strategy execution, and real-time risk management to identify and capitalize on market inefficiencies.

The system is built with a modular architecture, separating data ingestion, order flow calculation, strategy logic, and execution into distinct, loosely coupled components communicating via an asynchronous event bus.

## Key Features

### 1. Order Flow Engine
- **Real-time Delta & CVD**: Calculates buying and selling pressure tick-by-tick.
- **Footprint Charts**: Aggregates volume-at-price to visualize market microstructure.
- **Absorption & Trap Detection**: Identifies algorithmic setups like absorption at key levels and trapped traders.

### 2. Strategy Engine
- **Confluence Validator**: Implements a rigorous "box-checking" logic to grade setups (A/B/C) based on multiple factors (Bias, POI, Order Flow, etc.).
- **Trend Continuation**: Identifies pullbacks to key levels aligned with the higher timeframe trend.
- **Mean Reversion**: Fades extreme deviations from VWAP during late sessions.

### 3. Risk Management
- **Dynamic Position Sizing**: Adjusts trade size based on setup grade and account balance.
- **"House Money" Logic**: Aggressively compounds profits while protecting initial capital.
- **Circuit Breakers**: Automatically halts trading upon reaching max daily drawdown or consecutive loss limits.

### 4. Execution & State
- **Low-Latency Execution**: Direct integration with Binance Futures via `ccxt`.
- **State Persistence**: Redis-backed state management for crash recovery and fault tolerance.
- **OCO Management**: Automated Stop Loss and Take Profit handling.

### 5. AI & Analytics
- **LLM Integration**: Uses OpenAI/Anthropic to generate pre-market narratives and post-trade reviews.
- **Dashboard**: React-based real-time dashboard for monitoring order flow and active trades.

## Architecture

The system is composed of the following modules:

- **`engine/`**: Core trading logic.
  - `data/`: Data ingestion (WebSocket) and normalization.
  - `orderflow/`: Delta, CVD, and Footprint calculations.
  - `strategy/`: Signal generation and validation.
  - `risk/`: Position sizing and risk checks.
  - `execution/`: Order management and exchange interface.
  - `state/`: Redis persistence.
  - `analyst/`: LLM-based analysis.
- **`dashboard/`**: React frontend.
- **`monitoring/`**: Health checks and system status.

## Tech Stack

- **Language**: Python 3.9+
- **Database**: TimescaleDB (PostgreSQL)
- **Cache**: Redis
- **Exchange**: Binance Futures
- **Frontend**: React.js
- **Containerization**: Docker & Docker Compose

## Getting Started

### Prerequisites
- Docker & Docker Compose
- Python 3.9+
- Node.js (for Dashboard)

### Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/fabio-trading-system.git
    cd fabio-trading-system
    ```

2.  **Environment Setup**:
    Create a `.env` file in `trading-system/` with your credentials:
    ```env
    BINANCE_API_KEY=your_key
    BINANCE_SECRET_KEY=your_secret
    OPENAI_API_KEY=your_openai_key
    REDIS_HOST=localhost
    DB_HOST=localhost
    ```

3.  **Run with Docker**:
    ```bash
    cd trading-system
    docker-compose up --build
    ```

4.  **Run Dashboard**:
    ```bash
    cd trading-system/dashboard
    npm install
    npm start
    ```

## Development

- **Run Backtests**:
  ```bash
  python -m scripts.run_backtest --strategy TrendContinuation --symbol BTCUSDT
  ```

- **Check System Health**:
  ```bash
  python -m monitoring.health
  ```

## License
MIT
