class HouseMoneyManager:
    """
    Manages 'House Money' logic:
    - Track session P&L.
    - If P&L > 0, allow risking a portion of profits (e.g., 50%) on top of base risk.
    - If P&L < 0, revert to base risk or reduce risk.
    """
    
    def __init__(self, initial_balance: float, base_risk_pct: float = 0.01):
        self.initial_balance = initial_balance
        self.base_risk_pct = base_risk_pct
        self.session_pnl = 0.0
        self.max_risk_multiplier = 4.0
        self.compounding_rate = 0.5 # Risk 50% of profits
        
    def update_pnl(self, pnl: float):
        self.session_pnl += pnl
        
    def get_risk_multiplier(self) -> float:
        """
        Calculate risk multiplier based on House Money.
        Multiplier = 1.0 + (Risk_from_Profit / Base_Risk_Amount)
        """
        if self.session_pnl <= 0:
            return 1.0
            
        # How much of the profit are we willing to risk?
        risk_from_profit = self.session_pnl * self.compounding_rate
        
        # Base risk amount
        base_risk_amount = self.initial_balance * self.base_risk_pct
        
        # Additional multiplier
        additional_multiplier = risk_from_profit / base_risk_amount
        
        # Total multiplier (capped)
        return min(1.0 + additional_multiplier, self.max_risk_multiplier)
