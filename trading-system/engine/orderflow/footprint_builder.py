from collections import defaultdict
from datetime import datetime
from engine.data.schemas import AggTrade, FootprintCandle, FootprintLevel, TradeDirection
from engine.events import EventBus, Event, EventType

class FootprintBuilder:
    """
    Aggregates trades into Footprint candles (Volume at Price).
    """
    
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.current_candle = None
        self.levels = defaultdict(lambda: {'bid': 0.0, 'ask': 0.0, 'count': 0})
        self.candle_open_time = None
        
    async def process_trade(self, trade: AggTrade):
        """
        Accumulate volume at price level.
        """
        # In a real system, we need to know when a candle closes.
        # This class should listen to CANDLE_CLOSE events or manage its own timer.
        # For now, let's assume we receive external triggers or just accumulate.
        
        price = trade.price
        qty = trade.quantity
        
        if trade.direction == TradeDirection.BUY:
            self.levels[price]['ask'] += qty
        else:
            self.levels[price]['bid'] += qty
            
        self.levels[price]['count'] += 1
        
    async def on_candle_close(self, candle_data: dict):
        """
        Finalize the footprint when a candle closes.
        """
        # Construct FootprintCandle object
        levels_list = []
        total_volume = 0.0
        total_delta = 0.0
        max_vol = 0.0
        poc_price = 0.0
        
        for price, data in self.levels.items():
            bid_vol = data['bid']
            ask_vol = data['ask']
            vol = bid_vol + ask_vol
            delta = ask_vol - bid_vol
            
            total_volume += vol
            total_delta += delta
            
            if vol > max_vol:
                max_vol = vol
                poc_price = price
            
            levels_list.append(FootprintLevel(
                price=price,
                bid_volume=bid_vol,
                ask_volume=ask_vol,
                delta=delta,
                trade_count=data['count']
            ))
            
        # Calculate Value Area (70% of volume)
        # Sort levels by volume descending to find VA? 
        # Standard VA is usually around POC.
        # Simplified VA calculation for now.
        
        fp = FootprintCandle(
            timestamp=datetime.utcnow(), # Should use candle time
            symbol=candle_data.get('s', 'UNKNOWN'),
            timeframe="1m", # dynamic?
            open=float(candle_data.get('o', 0)),
            high=float(candle_data.get('h', 0)),
            low=float(candle_data.get('l', 0)),
            close=float(candle_data.get('c', 0)),
            volume=total_volume,
            delta=total_delta,
            levels=levels_list,
            poc_price=poc_price,
            value_area_high=0.0, # TODO: Implement VA logic
            value_area_low=0.0
        )
        
        # Reset for next candle
        self.levels.clear()
        
        # Publish
        # await self.event_bus.publish(...)
        return fp
