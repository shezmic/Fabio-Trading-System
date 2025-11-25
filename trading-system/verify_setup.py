import asyncio
import asyncpg
import redis.asyncio as redis
from engine.config import config
from engine.events import EventBus, Event, EventType

async def verify_db():
    print("Verifying Database Connection...")
    try:
        conn = await asyncpg.connect(
            user=config.DB_USER,
            password=config.DB_PASSWORD,
            database=config.DB_NAME,
            host=config.DB_HOST,
            port=config.DB_PORT
        )
        print("✅ Database Connected")
        
        # Check if tables exist
        tables = await conn.fetch("SELECT tablename FROM pg_tables WHERE schemaname = 'public'")
        table_names = [t['tablename'] for t in tables]
        required_tables = ['trades_raw', 'candles', 'volume_profile', 'signals', 'executions', 'session_state', 'trade_journal']
        
        missing = [t for t in required_tables if t not in table_names]
        if missing:
            print(f"❌ Missing tables: {missing}")
        else:
            print("✅ All tables present")
            
        await conn.close()
    except Exception as e:
        print(f"❌ Database Error: {e}")

async def verify_redis():
    print("\nVerifying Redis Connection...")
    try:
        r = redis.Redis(
            host=config.REDIS_HOST,
            port=config.REDIS_PORT,
            db=config.REDIS_DB
        )
        await r.ping()
        print("✅ Redis Connected")
        await r.close()
    except Exception as e:
        print(f"❌ Redis Error: {e}")

async def verify_event_bus():
    print("\nVerifying Event Bus...")
    bus = EventBus()
    queue = bus.subscribe(EventType.SYSTEM_STARTUP)
    
    await bus.publish(Event(type=EventType.SYSTEM_STARTUP, source="test"))
    
    try:
        event = await asyncio.wait_for(queue.get(), timeout=1.0)
        if event.type == EventType.SYSTEM_STARTUP:
            print("✅ Event Bus Working")
        else:
            print("❌ Event Bus received wrong event")
    except asyncio.TimeoutError:
        print("❌ Event Bus Timeout")

async def main():
    await verify_db()
    await verify_redis()
    await verify_event_bus()

if __name__ == "__main__":
    asyncio.run(main())
