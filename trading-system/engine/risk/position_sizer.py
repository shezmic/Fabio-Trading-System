from engine.data.schemas import SetupGrade

class PositionSizer:
    """
    Calculates position size based on:
    1. Account Balance
    2. Risk per Trade (e.g., 1%)
    3. Setup Grade (A=100%, B=75%, C=50% of risk)
    4. Stop Loss distance
    """
    
    def __init__(self, base_risk_pct: float = 0.01):
        self.base_risk_pct = base_risk_pct
        
    def calculate_size(self, balance: float, entry_price: float, stop_loss: float, grade: SetupGrade) -> float:
        """
        Returns quantity to trade.
        """
        risk_amount = balance * self.base_risk_pct
        
        # Adjust risk based on grade
        if grade == SetupGrade.A:
            risk_amount *= 1.0
        elif grade == SetupGrade.B:
            risk_amount *= 0.75
        elif grade == SetupGrade.C:
            risk_amount *= 0.50
        else:
            return 0.0
            
        # Calculate size based on SL distance
        # Risk = Size * |Entry - SL|
        # Size = Risk / |Entry - SL|
        
        sl_distance = abs(entry_price - stop_loss)
        if sl_distance == 0:
            return 0.0
            
        size = risk_amount / sl_distance
        return size
