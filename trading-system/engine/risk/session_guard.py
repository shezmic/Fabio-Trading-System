from engine.events import EventBus, Event, EventType

class SessionGuard:
    """
    Circuit Breakers:
    - Max Daily Drawdown
    - Max Consecutive Losses
    """
    
    def __init__(self, event_bus: EventBus, max_daily_loss: float, max_consecutive_losses: int = 3):
        self.event_bus = event_bus
        self.max_daily_loss = max_daily_loss
        self.max_consecutive_losses = max_consecutive_losses
        
        self.current_loss = 0.0
        self.consecutive_losses = 0
        self.is_active = True
        
    async def on_trade_close(self, pnl: float):
        if not self.is_active:
            return

        if pnl < 0:
            self.current_loss += abs(pnl)
            self.consecutive_losses += 1
        else:
            # Reset consecutive losses on win? Or keep counting if net negative?
            # Fabio: "3 consecutive losses = stop"
            self.consecutive_losses = 0
            
        # Check triggers
        if self.consecutive_losses >= self.max_consecutive_losses:
            await self._trigger_stop("Max Consecutive Losses Reached")
            
        if self.current_loss >= self.max_daily_loss:
            await self._trigger_stop("Max Daily Loss Reached")
            
    async def _trigger_stop(self, reason: str):
        self.is_active = False
        await self.event_bus.publish(Event(
            type=EventType.SESSION_STOPPED,
            source="session_guard",
            payload={"reason": reason}
        ))
