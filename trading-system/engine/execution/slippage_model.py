from dataclasses import dataclass
from typing import Optional

@dataclass
class SlippageEstimate:
    estimated_slippage_pct: float
    estimated_cost: float
    recommendation: str # 'PROCEED', 'REDUCE_SIZE', 'ABORT'

class SlippageModel:
    """
    Estimates slippage based on trade size, volatility, and liquidity.
    """
    
    def __init__(self, spread_multiplier: float = 1.5):
        self.spread_multiplier = spread_multiplier
        
    def estimate(self, 
                 symbol: str, 
                 quantity: float, 
                 current_price: float, 
                 avg_spread: float, 
                 avg_volume_1m: float) -> SlippageEstimate:
        
        # Basic linear model:
        # Slippage = Spread + (Impact_Coefficient * (Order_Size / Volume))
        
        # 1. Spread Cost
        spread_cost_pct = (avg_spread / current_price) if current_price > 0 else 0
        
        # 2. Market Impact
        # If order size > 1% of 1m volume, impact increases quadratically
        volume_ratio = quantity / avg_volume_1m if avg_volume_1m > 0 else 1.0
        impact_coeff = 0.1 # Tunable parameter
        
        impact_cost_pct = impact_coeff * (volume_ratio ** 2)
        
        total_slippage_pct = (spread_cost_pct * self.spread_multiplier) + impact_cost_pct
        estimated_cost = total_slippage_pct * quantity * current_price
        
        # Recommendation
        if total_slippage_pct > 0.005: # > 0.5% slippage
            recommendation = 'ABORT'
        elif total_slippage_pct > 0.001: # > 0.1% slippage
            recommendation = 'REDUCE_SIZE'
        else:
            recommendation = 'PROCEED'
            
        return SlippageEstimate(
            estimated_slippage_pct=total_slippage_pct,
            estimated_cost=estimated_cost,
            recommendation=recommendation
        )
