import json
from datetime import datetime
from typing import Optional, Dict, Any
import redis.asyncio as redis
from engine.config import config

class RedisStateStore:
    """
    Redis schema for hot state and crash recovery.
    """
    
    def __init__(self):
        self.redis = redis.Redis(
            host=config.REDIS_HOST,
            port=config.REDIS_PORT,
            db=config.REDIS_DB,
            decode_responses=True
        )
        
    async def save_position(self, symbol: str, position_data: Dict[str, Any]):
        """Save active position state"""
        key = f"state:position:{symbol}"
        await self.redis.set(key, json.dumps(position_data))
        
    async def load_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Load active position state"""
        key = f"state:position:{symbol}"
        data = await self.redis.get(key)
        return json.loads(data) if data else None
        
    async def clear_position(self, symbol: str):
        """Clear position state (after close)"""
        key = f"state:position:{symbol}"
        await self.redis.delete(key)
        
    async def save_order(self, order_id: str, order_data: Dict[str, Any]):
        """Save active order"""
        key = f"state:order:{order_id}"
        await self.redis.set(key, json.dumps(order_data))
        
    async def load_order(self, order_id: str) -> Optional[Dict[str, Any]]:
        """Load active order"""
        key = f"state:order:{order_id}"
        data = await self.redis.get(key)
        return json.loads(data) if data else None

    async def save_session_state(self, state: Dict[str, Any]):
        """Save session metrics (PnL, wins/losses)"""
        key = "state:session"
        await self.redis.set(key, json.dumps(state))
        
    async def load_session_state(self) -> Optional[Dict[str, Any]]:
        """Load session metrics"""
        key = "state:session"
        data = await self.redis.get(key)
        return json.loads(data) if data else None
