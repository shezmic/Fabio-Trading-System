import ccxt.async_support as ccxt
import asyncio
from typing import Optional, Dict, Any
from engine.config import config
import logging

logger = logging.getLogger("BinanceExecutor")

class BinanceExecutor:
    """
    Handles order execution on Binance Futures via CCXT.
    """
    
    def __init__(self):
        self.exchange = ccxt.binance({
            'apiKey': config.BINANCE_API_KEY,
            'secret': config.BINANCE_SECRET_KEY,
            'options': {
                'defaultType': 'future',
            },
            'enableRateLimit': True
        })
        # Sandbox mode if needed
        if config.ENV == "development":
            self.exchange.set_sandbox_mode(True) # Warning: Check if Binance Futures Sandbox is supported/wanted
            
    async def initialize(self):
        await self.exchange.load_markets()
        
    async def close(self):
        await self.exchange.close()
        
    async def create_order(self, symbol: str, side: str, type: str, quantity: float, price: Optional[float] = None, params: Dict = {}) -> Dict:
        """
        Create an order.
        side: 'buy' or 'sell'
        type: 'limit', 'market', 'stop_market', etc.
        """
        try:
            return await self.exchange.create_order(symbol, type, side, quantity, price, params)
        except Exception as e:
            logger.error(f"Order creation failed: {e}")
            raise e
            
    async def cancel_order(self, order_id: str, symbol: str):
        try:
            return await self.exchange.cancel_order(order_id, symbol)
        except Exception as e:
            logger.error(f"Order cancellation failed: {e}")
            raise e

    async def get_position(self, symbol: str) -> Optional[Dict]:
        """Get current position for symbol"""
        try:
            positions = await self.exchange.fetch_positions([symbol])
            for p in positions:
                if p['symbol'] == symbol:
                    return p
            return None
        except Exception as e:
            logger.error(f"Fetch position failed: {e}")
            return None
