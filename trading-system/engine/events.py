from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional, Callable, Coroutine, List, Dict
import asyncio
from collections import defaultdict
import json

class EventType(str, Enum):
    # Data Events
    TRADE_TICK = "trade.tick"              # New aggTrade received
    CANDLE_CLOSE = "candle.close"          # Candle completed
    ORDERBOOK_UPDATE = "orderbook.update"  # Depth snapshot
    
    # Order Flow Events
    DELTA_UPDATE = "orderflow.delta"       # Delta calculated
    ABSORPTION_DETECTED = "orderflow.absorption"
    TRAPPED_TRADERS = "orderflow.trapped"
    CVD_DIVERGENCE = "orderflow.cvd_divergence"
    
    # Strategy Events
    BIAS_UPDATE = "strategy.bias"          # 15m bias changed
    SIGNAL_GENERATED = "strategy.signal"   # Trade signal ready
    CONFLUENCE_CHECK = "strategy.confluence"
    
    # Execution Events
    ORDER_SUBMITTED = "execution.submitted"
    ORDER_FILLED = "execution.filled"
    ORDER_CANCELLED = "execution.cancelled"
    ORDER_REJECTED = "execution.rejected"
    POSITION_OPENED = "execution.position_open"
    POSITION_CLOSED = "execution.position_close"
    
    # Risk Events
    RISK_BREACH = "risk.breach"
    SESSION_STOPPED = "risk.session_stop"
    BREAKEVEN_TRIGGERED = "risk.breakeven"
    
    # System Events
    SYSTEM_STARTUP = "system.startup"
    SYSTEM_SHUTDOWN = "system.shutdown"
    RECONNECT = "system.reconnect"


@dataclass
class Event:
    """Base event with metadata"""
    type: EventType
    timestamp: datetime = field(default_factory=datetime.utcnow)
    source: str = ""
    payload: dict = field(default_factory=dict)
    correlation_id: Optional[str] = None  # For tracking related events


class EventBus:
    """
    In-process async event bus using asyncio.Queue.
    """
    
    def __init__(self, redis_client=None):
        self._subscribers: Dict[EventType, List[asyncio.Queue]] = defaultdict(list)
        self._global_subscribers: List[asyncio.Queue] = []
        self._redis = redis_client  # For event persistence (recovery)
    
    def subscribe(self, event_type: EventType) -> asyncio.Queue:
        """Subscribe to specific event type. Returns queue to await."""
        queue = asyncio.Queue(maxsize=1000)
        self._subscribers[event_type].append(queue)
        return queue
    
    def subscribe_all(self) -> asyncio.Queue:
        """Subscribe to all events (for logging/monitoring)."""
        queue = asyncio.Queue(maxsize=5000)
        self._global_subscribers.append(queue)
        return queue
    
    async def publish(self, event: Event) -> None:
        """
        Publish event to all subscribers.
        Non-blocking: drops events if subscriber queue is full.
        """
        # Persist critical events for recovery
        if self._redis and event.type in {
            EventType.POSITION_OPENED,
            EventType.POSITION_CLOSED,
            EventType.ORDER_SUBMITTED,
            EventType.ORDER_FILLED,
        }:
            await self._persist_event(event)
        
        # Notify type-specific subscribers
        for queue in self._subscribers.get(event.type, []):
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                pass  # Drop event if subscriber is slow
        
        # Notify global subscribers
        for queue in self._global_subscribers:
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                pass
    
    async def _persist_event(self, event: Event) -> None:
        """Persist event to Redis for crash recovery"""
        if self._redis:
            key = f"events:{event.type.value}:{event.timestamp.timestamp()}"
            # Convert datetime to string for JSON serialization if needed, 
            # but payload should be dict.
            # Here we assume payload is json serializable.
            await self._redis.setex(key, 3600, json.dumps(event.payload))  # 1hr TTL
    
    def unsubscribe(self, event_type: EventType, queue: asyncio.Queue) -> None:
        """Remove subscription"""
        if queue in self._subscribers.get(event_type, []):
            self._subscribers[event_type].remove(queue)
