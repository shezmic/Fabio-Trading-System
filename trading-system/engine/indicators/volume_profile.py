from dataclasses import dataclass
from typing import Dict, List, Optional
from collections import defaultdict
from engine.data.schemas import AggTrade

@dataclass
class VolumeProfileResult:
    poc: float
    vah: float
    val: float
    total_volume: float
    levels: Dict[float, float] # Price -> Volume

class VolumeProfileCalculator:
    """
    Calculates Session Volume Profile.
    Identifies Point of Control (POC) and Value Area (VAH/VAL).
    """
    
    def __init__(self, tick_size: float = 0.1, value_area_pct: float = 0.70):
        self.tick_size = tick_size
        self.value_area_pct = value_area_pct
        self.levels: Dict[float, float] = defaultdict(float)
        self.total_volume = 0.0
        
    def reset(self):
        self.levels.clear()
        self.total_volume = 0.0
        
    def _round_to_tick(self, price: float) -> float:
        return round(price / self.tick_size) * self.tick_size
        
    def update(self, trade: AggTrade) -> Optional[VolumeProfileResult]:
        price_level = self._round_to_tick(trade.price)
        self.levels[price_level] += trade.quantity
        self.total_volume += trade.quantity
        
        # Optimization: Don't recalculate full stats on every tick if not needed.
        # But for now, we provide a method to get stats on demand.
        return None 
        
    def get_profile(self) -> Optional[VolumeProfileResult]:
        if not self.levels:
            return None
            
        # 1. Find POC (Level with max volume)
        poc_price = max(self.levels, key=self.levels.get)
        max_vol = self.levels[poc_price]
        
        # 2. Calculate Value Area
        target_volume = self.total_volume * self.value_area_pct
        
        # Sort levels by price
        sorted_prices = sorted(self.levels.keys())
        poc_idx = sorted_prices.index(poc_price)
        
        current_volume = max_vol
        upper_idx = poc_idx
        lower_idx = poc_idx
        
        # Expand from POC
        while current_volume < target_volume:
            # Try to expand up
            next_upper_vol = 0
            if upper_idx < len(sorted_prices) - 1:
                next_upper_vol = self.levels[sorted_prices[upper_idx + 1]]
                
            # Try to expand down
            next_lower_vol = 0
            if lower_idx > 0:
                next_lower_vol = self.levels[sorted_prices[lower_idx - 1]]
                
            # Expand in direction of higher volume (or both if needed, but usually one step at a time)
            # Standard algo: compare two immediate neighbors
            
            if (upper_idx < len(sorted_prices) - 1) and (lower_idx > 0):
                if next_upper_vol >= next_lower_vol:
                    current_volume += next_upper_vol
                    upper_idx += 1
                else:
                    current_volume += next_lower_vol
                    lower_idx -= 1
            elif upper_idx < len(sorted_prices) - 1:
                current_volume += next_upper_vol
                upper_idx += 1
            elif lower_idx > 0:
                current_volume += next_lower_vol
                lower_idx -= 1
            else:
                break
                
        vah = sorted_prices[upper_idx]
        val = sorted_prices[lower_idx]
        
        return VolumeProfileResult(
            poc=poc_price,
            vah=vah,
            val=val,
            total_volume=self.total_volume,
            levels=dict(self.levels)
        )
