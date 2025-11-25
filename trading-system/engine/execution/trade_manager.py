from engine.events import EventBus, Event, EventType
from engine.execution.binance_executor import BinanceExecutor
from engine.execution.order_manager import OrderManager
from engine.risk.position_sizer import PositionSizer
from engine.risk.session_guard import SessionGuard
from engine.state.redis_store import RedisStateStore

class TradeManager:
    """
    Orchestrates the trade lifecycle:
    Signal -> Risk Check -> Execution -> Management -> Exit
    """
    
    def __init__(self, event_bus: EventBus, executor: BinanceExecutor, state_store: RedisStateStore):
        self.event_bus = event_bus
        self.executor = executor
        self.state_store = state_store
        self.order_manager = OrderManager(event_bus)
        self.position_sizer = PositionSizer()
        # self.session_guard = SessionGuard(...) 
        
    async def on_signal(self, signal: dict):
        """
        Handle incoming trade signal.
        """
        # 1. Check Session Guard (is trading allowed?)
        
        # 2. Calculate Position Size
        # size = self.position_sizer.calculate_size(...)
        
        # 3. Execute Entry
        # await self.executor.create_order(...)
        
        # 4. Place SL/TP (OCO)
        
        pass
