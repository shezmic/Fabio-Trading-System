from typing import List, Optional, Tuple
from enum import Enum

class StructureType(str, Enum):
    SWING_HIGH = "SWING_HIGH"
    SWING_LOW = "SWING_LOW"
    MSS_BULLISH = "MSS_BULLISH" # Market Structure Shift
    MSS_BEARISH = "MSS_BEARISH"

class MarketStructureDetector:
    """
    Detects Swing Highs, Swing Lows, and Market Structure Shifts.
    Uses a fractal approach (e.g., High surrounded by 2 lower highs).
    """
    
    def __init__(self, fractal_period: int = 2):
        self.fractal_period = fractal_period # candles on each side
        self.candles = [] # List of (high, low) tuples or objects
        self.last_swing_high = None # price
        self.last_swing_low = None # price
        
    def update(self, high: float, low: float) -> List[Tuple[StructureType, float]]:
        """
        Update with new candle. Returns list of detected events.
        """
        self.candles.append((high, low))
        if len(self.candles) < (self.fractal_period * 2 + 1):
            return []
            
        # We analyze the candle at index -(fractal_period + 1)
        # i.e., the middle of the window
        
        events = []
        center_idx = len(self.candles) - 1 - self.fractal_period
        center_high, center_low = self.candles[center_idx]
        
        # Check Swing High
        is_swing_high = True
        for i in range(1, self.fractal_period + 1):
            # Check left neighbors
            if self.candles[center_idx - i][0] >= center_high:
                is_swing_high = False
                break
            # Check right neighbors
            if self.candles[center_idx + i][0] >= center_high:
                is_swing_high = False
                break
                
        if is_swing_high:
            self.last_swing_high = center_high
            events.append((StructureType.SWING_HIGH, center_high))
            
        # Check Swing Low
        is_swing_low = True
        for i in range(1, self.fractal_period + 1):
            # Check left neighbors
            if self.candles[center_idx - i][1] <= center_low:
                is_swing_low = False
                break
            # Check right neighbors
            if self.candles[center_idx + i][1] <= center_low:
                is_swing_low = False
                break
                
        if is_swing_low:
            self.last_swing_low = center_low
            events.append((StructureType.SWING_LOW, center_low))
            
        # Check MSS (Break of Structure)
        # Usually we check if CURRENT price closes above last swing high
        # But here we are analyzing historical swing points.
        # Real-time MSS check needs current close vs last confirmed swing.
        
        return events
        
    def check_mss(self, current_close: float) -> Optional[StructureType]:
        """
        Check if current price breaks structure.
        """
        if self.last_swing_high and current_close > self.last_swing_high:
            return StructureType.MSS_BULLISH
            
        if self.last_swing_low and current_close < self.last_swing_low:
            return StructureType.MSS_BEARISH
            
        return None
