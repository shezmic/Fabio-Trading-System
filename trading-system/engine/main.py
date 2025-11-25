import asyncio
import signal
import logging
from contextlib import asynccontextmanager
from engine.config import config
from engine.events import EventBus, Event, EventType

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("TradingEngine")

class TradingEngine:
    """
    Main orchestrator using structured concurrency.
    """
    
    def __init__(self):
        self.config = config
        self.event_bus = EventBus()
        self._tasks: list[asyncio.Task] = []
        self._shutdown_event = asyncio.Event()
        self._locks = {
            'position': asyncio.Lock(),
            'orders': asyncio.Lock(),
            'state': asyncio.Lock(),
        }
    
    async def start(self) -> None:
        """Start all components with graceful shutdown handling"""
        logger.info("Starting Trading Engine...")
        
        # Register signal handlers
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, self._handle_shutdown)
        
        # Start component tasks
        self._tasks = [
            # asyncio.create_task(self._run_websocket_handler(), name="websocket"),
            # asyncio.create_task(self._run_orderflow_processor(), name="orderflow"),
            # asyncio.create_task(self._run_strategy_engine(), name="strategy"),
            # asyncio.create_task(self._run_execution_monitor(), name="execution"),
            # asyncio.create_task(self._run_risk_monitor(), name="risk"),
        ]
        
        # Publish startup event
        await self.event_bus.publish(Event(
            type=EventType.SYSTEM_STARTUP,
            source="engine",
        ))
        
        logger.info("Trading Engine Running. Press Ctrl+C to stop.")
        
        # Wait for shutdown signal
        await self._shutdown_event.wait()
        
        # Graceful shutdown
        await self._shutdown()
    
    def _handle_shutdown(self) -> None:
        """Signal handler for graceful shutdown"""
        logger.info("Shutdown signal received.")
        self._shutdown_event.set()
    
    async def _shutdown(self) -> None:
        """
        Graceful shutdown sequence
        """
        logger.info("Shutting down...")
        
        await self.event_bus.publish(Event(
            type=EventType.SYSTEM_SHUTDOWN,
            source="engine",
        ))
        
        # Cancel tasks with timeout
        for task in self._tasks:
            task.cancel()
        
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
            
        logger.info("Shutdown complete.")

if __name__ == "__main__":
    engine = TradingEngine()
    try:
        asyncio.run(engine.start())
    except KeyboardInterrupt:
        pass # Handled by signal handler usually, but just in case
