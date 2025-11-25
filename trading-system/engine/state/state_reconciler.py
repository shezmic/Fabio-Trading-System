import logging
from typing import Dict, List
from engine.execution.binance_executor import BinanceExecutor
from engine.state.redis_store import RedisStateStore

logger = logging.getLogger("StateReconciler")

class StateReconciler:
    """
    Reconciles local state (Redis) with exchange state (Binance) on startup.
    Handles crash recovery scenarios.
    """
    
    def __init__(self, executor: BinanceExecutor, redis_store: RedisStateStore):
        self.executor = executor
        self.redis = redis_store
        
    async def reconcile(self):
        """
        Main reconciliation loop.
        1. Get all open positions from Exchange.
        2. Get all open positions from Redis.
        3. Compare and fix discrepancies.
        """
        logger.info("Starting State Reconciliation...")
        
        exchange_positions = await self.executor.get_all_positions()
        # Filter for active positions (size != 0)
        active_exchange_positions = [p for p in exchange_positions if float(p['positionAmt']) != 0]
        
        # TODO: Load local positions from Redis (need to implement get_all_positions in RedisStore)
        # local_positions = await self.redis.get_all_positions() 
        local_positions = {} # Placeholder
        
        # Check for Orphans (Position on Exchange, but not in Redis)
        for pos in active_exchange_positions:
            symbol = pos['symbol']
            if symbol not in local_positions:
                logger.warning(f"Orphan position detected for {symbol}: {pos}")
                await self._handle_orphan_position(pos)
                
        # Check for Ghosts (Position in Redis, but not on Exchange)
        for symbol, pos in local_positions.items():
            # Check if it exists in active_exchange_positions
            found = False
            for ex_pos in active_exchange_positions:
                if ex_pos['symbol'] == symbol:
                    found = True
                    break
            
            if not found:
                logger.warning(f"Ghost position detected for {symbol}. Clearing local state.")
                await self.redis.clear_position(symbol)
                
        logger.info("State Reconciliation Complete.")
        
    async def _handle_orphan_position(self, position: dict):
        """
        Handle a position found on exchange but missing locally.
        Action: Create emergency exit orders (Stop Loss) if none exist.
        """
        symbol = position['symbol']
        amt = float(position['positionAmt'])
        entry_price = float(position['entryPrice'])
        
        # Check if there are open orders for this symbol
        open_orders = await self.executor.get_open_orders(symbol)
        
        if not open_orders:
            logger.warning(f"Orphan position {symbol} has NO open orders. Placing emergency Stop Loss.")
            # Place SL at 1% from entry
            side = "SELL" if amt > 0 else "BUY"
            sl_price = entry_price * 0.99 if amt > 0 else entry_price * 1.01
            
            # await self.executor.create_order(...) # TODO: Implement emergency order placement
        else:
            logger.info(f"Orphan position {symbol} has {len(open_orders)} open orders. Assuming safe.")
            
        # Save to Redis to sync state
        # await self.redis.save_position(...)
