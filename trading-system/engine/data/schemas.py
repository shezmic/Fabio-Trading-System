from pydantic import BaseModel
from datetime import datetime
from enum import Enum
from typing import Optional, List

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
    levels: List[FootprintLevel]        # Volume at each price
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
