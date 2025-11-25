import pandas as pd
from engine.backtest.metrics import PerformanceMetrics

class VectorizedBacktester:
    """
    Fast backtester using pandas for initial strategy validation.
    """
    
    def __init__(self, data: pd.DataFrame):
        self.data = data
        
    def run(self, strategy_logic):
        """
        Run strategy logic on dataframe.
        strategy_logic: function that takes df and returns signals series
        """
        df = self.data.copy()
        
        # Apply strategy
        df['signal'] = strategy_logic(df)
        
        # Calculate returns
        df['returns'] = df['close'].pct_change()
        df['strategy_returns'] = df['signal'].shift(1) * df['returns']
        
        # Equity curve
        df['equity'] = (1 + df['strategy_returns']).cumprod()
        
        # Extract trades (simplified)
        trades = df[df['signal'] != 0].copy()
        trades['pnl'] = trades['strategy_returns'] # Approximation
        
        metrics = PerformanceMetrics.calculate(trades, df['equity'])
        return metrics, df
