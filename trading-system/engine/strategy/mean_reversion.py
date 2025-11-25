from engine.events import EventBus, Event, EventType
from engine.strategy.confluence_validator import ConfluenceValidator
from engine.data.schemas import TradeSignal, TradeDirection, SetupGrade
from datetime import datetime

class MeanReversionStrategy:
    """
    Late-session strategy:
    Fade moves at extreme deviations (2nd/3rd SD of VWAP)
    """
    
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.validator = ConfluenceValidator()
        
    async def check_signal(self, symbol: str, price: float, vwap_data: dict):
        # Check if price is at 2SD/3SD
        # Check for Reversal Order Flow
        pass
