from engine.data.schemas import SetupGrade, OrderFlowSignal
from engine.state.market_state import MarketStateSnapshot, MarketRegime

class ConfluenceValidator:
    """
    Validates trade setups by checking for confluence across multiple factors.
    Grades setups as A, B, or C.
    """
    
    def __init__(self):
        self.factors = {
            "bias": False,
            "poi": False,
            "orderflow": False,
            "market_state": False,
            "micro_structure": False
        }
        
    def validate(self, 
                 bias_aligned: bool, 
                 near_poi: bool, 
                 order_flow: OrderFlowSignal, 
                 market_state: MarketStateSnapshot,
                 micro_confirmed: bool) -> SetupGrade:
        
        # 1. Bias Alignment (Must be aligned with higher timeframe)
        self.factors["bias"] = bias_aligned
        
        # 2. POI (Price must be near key level)
        self.factors["poi"] = near_poi
        
        # 3. Order Flow (Absorption, Trapped Traders, or CVD Divergence)
        # Strong signal if any of these are present in the right direction
        of_confluence = (
            order_flow.absorption_detected or 
            order_flow.trapped_traders or 
            order_flow.cvd_divergence
        )
        self.factors["orderflow"] = of_confluence
        
        # 4. Market State (Are we in a favorable regime?)
        # e.g., Trend Continuation requires Imbalance or Breakout from Balance
        state_confluence = False
        if market_state.regime != MarketRegime.UNKNOWN:
            state_confluence = True # Simplified for now
        self.factors["market_state"] = state_confluence
        
        # 5. Micro Structure (15s confirmation)
        self.factors["micro_structure"] = micro_confirmed
        
        # Scoring
        score = sum(self.factors.values())
        
        if score == 5:
            return SetupGrade.A
        elif score == 4:
            return SetupGrade.B
        elif score == 3:
            return SetupGrade.C
        else:
            return SetupGrade.INVALID
            
    def get_missing_factors(self) -> list[str]:
        return [k for k, v in self.factors.items() if not v]
