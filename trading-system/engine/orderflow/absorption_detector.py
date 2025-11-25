from engine.events import EventBus, Event, EventType
from engine.data.schemas import AggTrade, TradeDirection

class AbsorptionDetector:
    """
    Detects absorption: High volume traded with minimal price movement.
    Limit orders absorbing aggressive market orders.
    """
    
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.volume_history = [] # List of recent trade volumes for percentile calc
        self.min_price = float('inf')
        self.max_price = float('-inf')
        self.current_window_vol = 0.0
        self.last_update_time = None
        
    def _get_volume_threshold(self) -> float:
        """
        Dynamic threshold: 75th percentile of recent significant volumes.
        Fallback to 10.0 BTC if insufficient history.
        """
        if len(self.volume_history) < 10:
            return 10.0
        
        sorted_vols = sorted(self.volume_history)
        idx = int(len(sorted_vols) * 0.75)
        return sorted_vols[idx]

    async def process_trade(self, trade: AggTrade):
        """
        Check for absorption patterns.
        """
        # Time decay logic (reset window if too much time passed)
        if self.last_update_time and (trade.event_time - self.last_update_time).total_seconds() > 5:
             self.current_window_vol = 0
             self.min_price = float('inf')
             self.max_price = float('-inf')
             
        self.last_update_time = trade.event_time
        self.current_window_vol += trade.quantity
        self.min_price = min(self.min_price, trade.price)
        self.max_price = max(self.max_price, trade.price)
        
        price_range = self.max_price - self.min_price
        threshold = self._get_volume_threshold()
        
        # Logic: If volume > threshold AND price range is tight (absorption)
        # Price range threshold should also be dynamic (e.g., relative to ATR), but hardcoded for now as per spec P1
        if self.current_window_vol > threshold and price_range < 5.0:
            # Record this volume for future threshold adaptation
            self.volume_history.append(self.current_window_vol)
            if len(self.volume_history) > 100:
                self.volume_history.pop(0)

            await self.event_bus.publish(Event(
                type=EventType.ABSORPTION_DETECTED,
                source="absorption_detector",
                payload={
                    "symbol": trade.symbol,
                    "volume": self.current_window_vol,
                    "price_range": price_range,
                    "threshold": threshold,
                    "timestamp": trade.event_time.isoformat()
                }
            ))
            
            # Reset window
            self.current_window_vol = 0
            self.min_price = float('inf')
            self.max_price = float('-inf')
