from engine.events import EventBus, Event, EventType
from engine.execution.binance_executor import BinanceExecutor
from engine.execution.order_manager import OrderManager
from engine.risk.position_sizer import PositionSizer
from engine.risk.session_guard import SessionGuard
from engine.state.redis_store import RedisStateStore
from engine.data.schemas import TradeSignal

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
        
    async def on_signal(self, signal: TradeSignal):
        """
        Handle incoming trade signal.
        """
        await self.execute_signal(signal)

    async def execute_signal(self, signal: TradeSignal):
        """
        Orchestrate trade execution:
        1. Check Risk (Session Guard, Exposure)
        2. Calculate Position Size
        3. Submit Entry Order
        4. Submit OCO Exits (SL/TP)
        """
        # 1. Risk Check (Placeholder)
        # if not self.session_guard.can_trade(): return
        
        # 2. Position Size - Now using PositionSizer
        account_balance = 10000.0  # TODO: Fetch from exchange
        quantity = self.position_sizer.calculate_size(
            balance=account_balance,
            entry_price=signal.entry_price,
            stop_loss=signal.stop_loss,
            grade=signal.grade
        )
        
        # 3. Submit Entry
        order = await self.executor.create_order(
            symbol=signal.symbol,
            side=signal.direction,
            type="MARKET",
            quantity=quantity
        )
        
        if order and order.get('status') == 'FILLED':
            # 4. Submit Exits
            await self._submit_exit_orders(signal, quantity, float(order['avgPrice']))
            
    async def _submit_exit_orders(self, signal: TradeSignal, quantity: float, entry_price: float):
        """
        Submit Stop Loss and Take Profit orders.
        """
        # Determine SL/TP prices if not in signal
        sl_price = signal.stop_loss
        tp_price = signal.take_profit
        
        side = "SELL" if signal.direction == "BUY" else "BUY"
        
        # Submit Stop Loss
        await self.executor.create_order(
            symbol=signal.symbol,
            side=side,
            type="STOP_MARKET",
            quantity=quantity,
            price=sl_price, # Stop Price
            params={'stopPrice': sl_price}
        )
        
        # Submit Take Profit
        await self.executor.create_order(
            symbol=signal.symbol,
            side=side,
            type="TAKE_PROFIT_MARKET",
            quantity=quantity,
            price=tp_price, # Stop Price for TP
            params={'stopPrice': tp_price}
        )
        
    async def move_to_breakeven(self, symbol: str, entry_price: float):
        """
        Move Stop Loss to Entry Price.
        """
        # Logic: Cancel existing SL, submit new SL at entry_price
        pass
