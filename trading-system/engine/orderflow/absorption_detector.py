from engine.events import EventBus, Event, EventType
from engine.data.schemas import AggTrade, TradeDirection

class AbsorptionDetector:
    """
    Detects absorption: High volume traded with minimal price movement.
    Limit orders absorbing aggressive market orders.
    """
    
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.volume_threshold = 10.0 # BTC, dynamic later
        self.price_range_threshold = 5.0 # USDT
        
        self.current_window_vol = 0.0
        self.min_price = float('inf')
        self.max_price = float('-inf')
        
    async def process_trade(self, trade: AggTrade):
        """
        Check for absorption patterns.
        """
        self.current_window_vol += trade.quantity
        self.min_price = min(self.min_price, trade.price)
        self.max_price = max(self.max_price, trade.price)
        
        price_range = self.max_price - self.min_price
        
        # Simple logic: If volume > threshold AND price range < threshold
        if self.current_window_vol > self.volume_threshold and price_range < self.price_range_threshold:
            # Determine side: if price is at low, buying absorption?
            # If price is at high, selling absorption?
            
            # This is a simplified check. Real absorption needs order book depth context too.
            
            await self.event_bus.publish(Event(
                type=EventType.ABSORPTION_DETECTED,
                source="absorption_detector",
                payload={
                    "symbol": trade.symbol,
                    "volume": self.current_window_vol,
                    "price_range": price_range,
                    "timestamp": trade.event_time.isoformat()
                }
            ))
            
            # Reset window (or use rolling window)
            self.current_window_vol = 0
            self.min_price = float('inf')
            self.max_price = float('-inf')
