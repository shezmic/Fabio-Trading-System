class WalkForwardOptimizer:
    """
    Performs walk-forward optimization.
    Train on window N, Test on window N+1.
    """
    
    def __init__(self, backtester):
        self.backtester = backtester
        
    def optimize(self, data, param_grid):
        """
        Run optimization loop.
        """
        pass
