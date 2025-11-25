from enum import Enum
from typing import Dict, Optional
from datetime import datetime
from engine.events import EventBus, Event, EventType

class OrderStatus(str, Enum):
    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"

class OrderManager:
    """
    Manages the lifecycle of individual orders.
    """
    
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.orders: Dict[str, Dict] = {} # internal id -> order data
        
    def create_order(self, symbol: str, side: str, quantity: float, price: Optional[float] = None, order_type: str = "MARKET") -> str:
        """
        Register a new order intention. Returns internal order ID.
        """
        import uuid
        order_id = str(uuid.uuid4())
        
        self.orders[order_id] = {
            "id": order_id,
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "price": price,
            "type": order_type,
            "status": OrderStatus.PENDING,
            "created_at": datetime.utcnow(),
            "exchange_id": None
        }
        
        return order_id
        
    async def update_status(self, order_id: str, status: OrderStatus, exchange_id: Optional[str] = None):
        if order_id in self.orders:
            self.orders[order_id]["status"] = status
            if exchange_id:
                self.orders[order_id]["exchange_id"] = exchange_id
                
            # Publish event
            event_type = {
                OrderStatus.SUBMITTED: EventType.ORDER_SUBMITTED,
                OrderStatus.FILLED: EventType.ORDER_FILLED,
                OrderStatus.CANCELLED: EventType.ORDER_CANCELLED,
                OrderStatus.REJECTED: EventType.ORDER_REJECTED
            }.get(status)
            
            if event_type:
                await self.event_bus.publish(Event(
                    type=event_type,
                    source="order_manager",
                    payload=self.orders[order_id]
                ))
