from engine.events import EventBus, Event, EventType

class CVDTracker:
    """
    Tracks Cumulative Volume Delta (CVD) and detects divergences with price.
    """
    
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.prev_cvd = 0.0
        self.prev_price = 0.0
        self.atr = 10.0 # Default, should be updated from ATR indicator
        
    def update_atr(self, atr: float):
        self.atr = atr
        
    async def update(self, cvd: float, price: float, symbol: str):
        """
        Check for divergence:
        - Price makes Higher High, CVD makes Lower High (Bearish Divergence)
        - Price makes Lower Low, CVD makes Higher Low (Bullish Divergence)
        """
        price_change = price - self.prev_price
        cvd_change = cvd - self.prev_cvd
        
        # Threshold: Price move must be significant relative to ATR (e.g., 0.5 * ATR)
        if abs(price_change) < (0.5 * self.atr):
            self.prev_cvd = cvd
            self.prev_price = price
            return

        if price_change > 0 and cvd_change < 0:
            # Price up significantly, but CVD down (Bearish Divergence / Absorption)
            await self.event_bus.publish(Event(
                type=EventType.CVD_DIVERGENCE,
                source="cvd_tracker",
                payload={
                    "symbol": symbol,
                    "type": "BEARISH_DIVERGENCE", 
                    "price": price,
                    "cvd": cvd,
                    "atr": self.atr
                }
            ))
            
        elif price_change < 0 and cvd_change > 0:
            # Price down significantly, but CVD up (Bullish Divergence / Absorption)
            await self.event_bus.publish(Event(
                type=EventType.CVD_DIVERGENCE,
                source="cvd_tracker",
                payload={
                    "symbol": symbol,
                    "type": "BULLISH_DIVERGENCE", 
                    "price": price,
                    "cvd": cvd,
                    "atr": self.atr
                }
            ))
            
        self.prev_cvd = cvd
        self.prev_price = price
