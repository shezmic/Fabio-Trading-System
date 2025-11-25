import pandas as pd
import numpy as np

class PerformanceMetrics:
    """
    Calculates trading performance metrics.
    """
    
    @staticmethod
    def calculate(trades: pd.DataFrame, equity_curve: pd.Series) -> dict:
        """
        Calculate metrics from trade list and equity curve.
        """
        if trades.empty:
            return {}
            
        total_trades = len(trades)
        wins = trades[trades['pnl'] > 0]
        losses = trades[trades['pnl'] <= 0]
        
        win_rate = len(wins) / total_trades if total_trades > 0 else 0
        avg_win = wins['pnl'].mean() if not wins.empty else 0
        avg_loss = losses['pnl'].mean() if not losses.empty else 0
        profit_factor = abs(wins['pnl'].sum() / losses['pnl'].sum()) if not losses.empty and losses['pnl'].sum() != 0 else float('inf')
        
        # Drawdown
        running_max = equity_curve.cummax()
        drawdown = (equity_curve - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Sharpe (assuming daily returns for simplicity, can adjust for intraday)
        returns = equity_curve.pct_change().dropna()
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if not returns.empty and returns.std() != 0 else 0
        
        return {
            "total_trades": total_trades,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": sharpe,
            "total_pnl": trades['pnl'].sum()
        }
