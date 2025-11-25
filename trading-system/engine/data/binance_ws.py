import asyncio
import json
import logging
import websockets
from engine.config import config
from engine.events import EventBus, Event, EventType

logger = logging.getLogger("BinanceWS")

class BinanceWebSocket:
    """
    Handles connection to Binance Futures WebSocket streams.
    """
    
    BASE_URL = "wss://fstream.binance.com/ws"
    
    def __init__(self, event_bus: EventBus, symbols: list[str]):
        self.event_bus = event_bus
        self.symbols = [s.lower() for s in symbols]
        self._running = False
        self._ws = None
    
    async def run(self):
        """Main loop to maintain connection"""
        self._running = True
        streams = []
        for s in self.symbols:
            streams.extend([
                f"{s}@aggTrade",
                f"{s}@depth@100ms",
                f"{s}@kline_1m",
                f"{s}@markPrice"
            ])
        
        stream_url = f"{self.BASE_URL}/{'/'.join(streams)}"
        
        while self._running:
            try:
                logger.info(f"Connecting to Binance WS: {stream_url}")
                async with websockets.connect(stream_url) as ws:
                    self._ws = ws
                    logger.info("Connected to Binance WS")
                    
                    while self._running:
                        msg = await ws.recv()
                        await self._handle_message(json.loads(msg))
                        
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                await asyncio.sleep(5)  # Reconnect delay
    
    async def _handle_message(self, msg: dict):
        """Route message to appropriate event"""
        event_type = msg.get('e')
        
        if event_type == 'aggTrade':
            await self.event_bus.publish(Event(
                type=EventType.TRADE_TICK,
                source="binance",
                payload=msg
            ))
        elif event_type == 'depthUpdate':
            await self.event_bus.publish(Event(
                type=EventType.ORDERBOOK_UPDATE,
                source="binance",
                payload=msg
            ))
        elif event_type == 'kline':
            kline = msg.get('k')
            if kline and kline.get('x'): # Candle closed
                await self.event_bus.publish(Event(
                    type=EventType.CANDLE_CLOSE,
                    source="binance",
                    payload=msg
                ))
        # Add other handlers as needed
