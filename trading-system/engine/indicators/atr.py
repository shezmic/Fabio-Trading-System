from collections import deque
import numpy as np

class ATRCalculator:
    """
    Calculates Average True Range (ATR).
    """
    
    def __init__(self, period: int = 14):
        self.period = period
        self.tr_history = deque(maxlen=period * 2) # Keep enough for smoothing
        self.prev_close = None
        self.current_atr = 0.0
        
    def update(self, high: float, low: float, close: float) -> float:
        """
        Update with new candle. Returns current ATR.
        """
        if self.prev_close is None:
            tr = high - low
        else:
            tr = max(
                high - low,
                abs(high - self.prev_close),
                abs(low - self.prev_close)
            )
            
        self.tr_history.append(tr)
        self.prev_close = close
        
        if len(self.tr_history) < self.period:
            # Simple Average for initial
            self.current_atr = sum(self.tr_history) / len(self.tr_history)
        else:
            # RMA (Wilder's Smoothing)
            # ATR = ((Prior ATR * (n-1)) + Current TR) / n
            if self.current_atr == 0:
                 self.current_atr = sum(list(self.tr_history)[:self.period]) / self.period
            else:
                self.current_atr = ((self.current_atr * (self.period - 1)) + tr) / self.period
                
        return self.current_atr
