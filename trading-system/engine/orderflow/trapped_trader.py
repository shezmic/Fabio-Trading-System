from engine.events import EventBus, Event, EventType

class TrappedTraderDetector:
    """
    Detects Trapped Traders:
    1. Breakout attempt (High volume at key level)
    2. Absorption (Price stalls)
    3. Reversal (Price moves back into range)
    """
    
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.potential_trap = None # State: {'price': float, 'side': 'BUY'/'SELL', 'time': datetime, 'volume': float}
        
    async def on_absorption(self, absorption_event: dict):
        """
        Called when absorption is detected.
        Sets a potential trap state.
        """
        # If absorption detected, it COULD be a trap if price reverses.
        # We store the state and wait for price update.
        self.potential_trap = {
            'price': absorption_event['price_range'], # Using range/midpoint would be better
            'time': datetime.utcnow(),
            'volume': absorption_event['volume']
            # Side needs to be inferred from absorption logic (Buying vs Selling absorption)
        }
        
    async def on_price_update(self, price: float):
        """
        Check if price reverses from the trap level.
        """
        if self.potential_trap:
            # Check if price moved away significantly (e.g. > 0.2%)
            # If so, confirm trap and publish event.
            # Then clear state.
            pass
            
    def _expire_stale_state(self):
        """Clear potential trap if too much time passed without reversal"""
        if self.potential_trap:
            if (datetime.utcnow() - self.potential_trap['time']).total_seconds() > 60:
                self.potential_trap = None
