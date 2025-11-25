from engine.events import EventBus, Event, EventType

class CVDTracker:
    """
    Tracks Cumulative Volume Delta (CVD) and detects divergences with price.
    """
    
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.prev_cvd = 0.0
        self.prev_price = 0.0
        
    async def update(self, cvd: float, price: float, symbol: str):
        """
        Check for divergence:
        - Price makes Higher High, CVD makes Lower High (Bearish Divergence)
        - Price makes Lower Low, CVD makes Higher Low (Bullish Divergence)
        """
        # This requires tracking swing highs/lows, which is complex.
        # Simplified: Price up, CVD down?
        
        price_change = price - self.prev_price
        cvd_change = cvd - self.prev_cvd
        
        if price_change > 0 and cvd_change < 0:
            # Price up, aggressive selling (absorption?)
            await self.event_bus.publish(Event(
                type=EventType.CVD_DIVERGENCE,
                source="cvd_tracker",
                payload={
                    "symbol": symbol,
                    "type": "BEARISH_DIVERGENCE", # Price up, Net Selling
                    "price": price,
                    "cvd": cvd
                }
            ))
            
        elif price_change < 0 and cvd_change > 0:
            # Price down, aggressive buying (absorption?)
            await self.event_bus.publish(Event(
                type=EventType.CVD_DIVERGENCE,
                source="cvd_tracker",
                payload={
                    "symbol": symbol,
                    "type": "BULLISH_DIVERGENCE", # Price down, Net Buying
                    "price": price,
                    "cvd": cvd
                }
            ))
            
        self.prev_cvd = cvd
        self.prev_price = price
