import asyncio
from typing import List
from datetime import datetime
from engine.events import EventBus, Event, EventType
from engine.data.schemas import AggTrade, TradeDirection
from engine.backtest.metrics import PerformanceMetrics
import pandas as pd

class EventDrivenBacktester:
    """
    Realistic backtester that processes events tick-by-tick.
    """
    
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.trades = []
        self.equity_curve = []
        self.current_balance = 10000.0
        
    async def run(self, data_feed):
        """
        Run simulation over data feed.
        """
        # 1. Replay data events
        for tick in data_feed:
            await self.event_bus.publish(Event(
                type=EventType.TRADE_TICK,
                source="backtest",
                payload=tick
            ))
            # Wait for processing? In async, we might need to ensure order
            
        # 2. Collect results
        return self._calculate_metrics()
        
    def _calculate_metrics(self):
        # Convert trades list to DF
        # Calculate metrics
        pass
