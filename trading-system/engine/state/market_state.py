from enum import Enum
from dataclasses import dataclass
from typing import Optional

class MarketRegime(str, Enum):
    BALANCE = "BALANCE"
    IMBALANCE_UP = "IMBALANCE_UP"
    IMBALANCE_DOWN = "IMBALANCE_DOWN"
    UNKNOWN = "UNKNOWN"

@dataclass
class MarketStateSnapshot:
    regime: MarketRegime
    initial_balance_high: float
    initial_balance_low: float
    current_price: float

class MarketStateDetector:
    """
    Detects the current market state (Balance vs Imbalance).
    Uses Initial Balance (first 30-60 mins) and VWAP slope.
    """
    
    def __init__(self):
        self.ib_high = float('-inf')
        self.ib_low = float('inf')
        self.ib_established = False
        self.regime = MarketRegime.UNKNOWN
        
    def update(self, price: float, time_in_session_minutes: int):
        # 1. Establish Initial Balance (IB) in first 60 mins
        if time_in_session_minutes <= 60:
            self.ib_high = max(self.ib_high, price)
            self.ib_low = min(self.ib_low, price)
            self.regime = MarketRegime.BALANCE
            if time_in_session_minutes == 60:
                self.ib_established = True
                
        # 2. Detect Breakout (Imbalance) after IB established
        elif self.ib_established:
            if price > self.ib_high:
                self.regime = MarketRegime.IMBALANCE_UP
            elif price < self.ib_low:
                self.regime = MarketRegime.IMBALANCE_DOWN
            else:
                self.regime = MarketRegime.BALANCE
                
    def get_state(self) -> MarketStateSnapshot:
        return MarketStateSnapshot(
            regime=self.regime,
            initial_balance_high=self.ib_high,
            initial_balance_low=self.ib_low,
            current_price=0.0 # Should be passed or stored
        )
