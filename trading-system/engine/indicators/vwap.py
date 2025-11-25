import math
from dataclasses import dataclass
from typing import Optional
from engine.data.schemas import AggTrade

@dataclass
class VWAPResult:
    vwap: float
    upper_band_1: float
    lower_band_1: float
    upper_band_2: float
    lower_band_2: float
    upper_band_3: float
    lower_band_3: float
    std_dev: float

class VWAPCalculator:
    """
    Calculates rolling VWAP and Standard Deviation Bands.
    Formula:
    VWAP = Sum(Price * Volume) / Sum(Volume)
    Variance = (Sum(Price^2 * Volume) / Sum(Volume)) - VWAP^2
    StdDev = Sqrt(Variance)
    """
    
    def __init__(self):
        self.cum_volume = 0.0
        self.cum_pv = 0.0      # Cumulative (Price * Volume)
        self.cum_p2v = 0.0     # Cumulative (Price^2 * Volume)
        
    def reset(self):
        """Reset calculation (e.g., at session start)"""
        self.cum_volume = 0.0
        self.cum_pv = 0.0
        self.cum_p2v = 0.0
        
    def update(self, trade: AggTrade) -> Optional[VWAPResult]:
        price = trade.price
        qty = trade.quantity
        
        self.cum_volume += qty
        self.cum_pv += (price * qty)
        self.cum_p2v += (price * price * qty)
        
        if self.cum_volume == 0:
            return None
            
        vwap = self.cum_pv / self.cum_volume
        
        # Calculate Variance
        # Var = E[X^2] - (E[X])^2
        mean_x2 = self.cum_p2v / self.cum_volume
        variance = mean_x2 - (vwap * vwap)
        
        # Handle floating point errors (variance can be slightly negative close to 0)
        variance = max(0.0, variance)
        std_dev = math.sqrt(variance)
        
        return VWAPResult(
            vwap=vwap,
            upper_band_1=vwap + std_dev,
            lower_band_1=vwap - std_dev,
            upper_band_2=vwap + (2 * std_dev),
            lower_band_2=vwap - (2 * std_dev),
            upper_band_3=vwap + (3 * std_dev),
            lower_band_3=vwap - (3 * std_dev),
            std_dev=std_dev
        )
