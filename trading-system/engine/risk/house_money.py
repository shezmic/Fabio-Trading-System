class HouseMoneyManager:
    """
    Manages 'House Money' logic:
    - Track session P&L.
    - If P&L > 0, allow risking a portion of profits (e.g., 50%) on top of base risk.
    - If P&L < 0, revert to base risk or reduce risk.
    """
    
    def __init__(self):
        self.session_pnl = 0.0
        self.risk_multiplier = 1.0
        
    def update_pnl(self, pnl: float):
        self.session_pnl += pnl
        self._update_multiplier()
        
    def _update_multiplier(self):
        if self.session_pnl > 0:
            # Example: Risk 1.5x if profitable
            self.risk_multiplier = 1.5 
        else:
            self.risk_multiplier = 1.0
            
    def get_risk_multiplier(self) -> float:
        return self.risk_multiplier
