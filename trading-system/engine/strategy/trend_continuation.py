from engine.events import EventBus, Event, EventType
from engine.strategy.confluence_validator import ConfluenceValidator
from engine.data.schemas import TradeSignal, TradeDirection, SetupGrade
from datetime import datetime

class TrendContinuationStrategy:
    """
    Fabio's primary strategy:
    1. Identify Trend (15m/1h)
    2. Wait for Pullback to POI (VWAP, EMA, etc.)
    3. Look for Order Flow confirmation (Absorption/Trap)
    4. Enter on Micro-structure break
    """
    
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.validator = ConfluenceValidator()
        self.bias = None # 'LONG' or 'SHORT'
        
    async def on_bias_update(self, bias: str):
        self.bias = bias
        self.validator.update_factor('bias', True) # Simplified
        
    async def on_orderflow_signal(self, signal: dict):
        """
        Handle absorption or trap signals.
        """
        # Logic to check if signal aligns with bias
        # If so, update validator
        pass
        
    async def check_signal(self, symbol: str, price: float):
        """
        Called on price updates or candle closes to check for entry.
        """
        grade = self.validator.validate()
        
        if grade != SetupGrade.INVALID:
            # Generate Signal
            signal = TradeSignal(
                timestamp=datetime.utcnow(),
                symbol=symbol,
                direction=TradeDirection.BUY if self.bias == 'LONG' else TradeDirection.SELL,
                grade=grade,
                entry_price=price,
                stop_loss=price * 0.99, # Placeholder
                take_profit=price * 1.02, # Placeholder
                confluence_score=sum(self.validator.factors.values()),
                rationale=f"Trend Continuation {grade.value} Setup"
            )
            
            await self.event_bus.publish(Event(
                type=EventType.SIGNAL_GENERATED,
                source="trend_continuation",
                payload=signal.dict()
            ))
            
            self.validator.reset()
