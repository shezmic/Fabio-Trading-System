from collections import deque
from datetime import datetime, timedelta
from engine.events import EventBus, Event, EventType
from engine.data.schemas import AggTrade, TradeDirection

class DeltaCalculator:
    """
    Calculates rolling delta (Buy Volume - Sell Volume) over various time windows.
    Also tracks Cumulative Volume Delta (CVD) for the session.
    """
    
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.cvd = 0.0
        self.trades = deque() # Store recent trades for rolling window calculation
        
        # Rolling windows in seconds
        self.windows = {
            '5s': 5,
            '15s': 15,
            '1m': 60,
            '5m': 300
        }
        
    async def process_trade(self, trade: AggTrade):
        """
        Process a new trade and update delta metrics.
        """
        # Update CVD
        delta = trade.quantity if trade.direction == TradeDirection.BUY else -trade.quantity
        self.cvd += delta
        
        # Add to rolling window
        self.trades.append(trade)
        self._prune_trades()
        
        # Calculate rolling deltas
        deltas = self._calculate_rolling_deltas()
        
        # Publish update
        await self.event_bus.publish(Event(
            type=EventType.DELTA_UPDATE,
            source="delta_calculator",
            payload={
                "symbol": trade.symbol,
                "timestamp": trade.event_time.isoformat(),
                "cvd": self.cvd,
                "rolling_deltas": deltas
            }
        ))
        
    def _prune_trades(self):
        """Remove trades older than the largest window"""
        max_window = max(self.windows.values())
        cutoff = datetime.utcnow() - timedelta(seconds=max_window)
        # Note: In a real high-freq system, we'd use a more efficient structure than deque + linear scan/pop
        # But for Python prototyping this is fine.
        # Assuming trades are monotonic in time.
        
        while self.trades and self.trades[0].event_time < cutoff:
            self.trades.popleft()
            
    def _calculate_rolling_deltas(self):
        """Calculate delta for each defined window"""
        now = datetime.utcnow()
        results = {}
        
        # Optimization: We could maintain running sums for each window
        # instead of re-iterating. For now, simple iteration.
        for name, seconds in self.windows.items():
            cutoff = now - timedelta(seconds=seconds)
            window_delta = 0.0
            
            # Iterate backwards for efficiency? Or just filter.
            # Since deque is sorted by time, we can iterate.
            for trade in self.trades:
                if trade.event_time >= cutoff:
                    d = trade.quantity if trade.direction == TradeDirection.BUY else -trade.quantity
                    window_delta += d
            
            results[name] = window_delta
            
        return results
