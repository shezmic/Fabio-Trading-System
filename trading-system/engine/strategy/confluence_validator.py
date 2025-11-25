from engine.data.schemas import SetupGrade, TradeDirection
from engine.events import EventBus

class ConfluenceValidator:
    """
    Validates trade setups by checking for confluence across multiple factors.
    Implements Fabio's "Box Checking" methodology.
    """
    
    def __init__(self):
        # Confluence factors
        self.factors = {
            'bias': False,          # 15m/1h Trend Alignment
            'poi': False,           # Price at Key Level (VWAP, VAL/VAH)
            'orderflow': False,     # Absorption/Trapped Traders
            'follow_through': False, # Aggressive move in direction
            'micro_structure': False # 15s/1m Structure Break
        }
    
    def reset(self):
        """Reset factors for a new potential setup"""
        for k in self.factors:
            self.factors[k] = False
            
    def update_factor(self, factor: str, value: bool):
        if factor in self.factors:
            self.factors[factor] = value
            
    def validate(self) -> SetupGrade:
        """
        Grade the setup based on checked boxes.
        """
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
