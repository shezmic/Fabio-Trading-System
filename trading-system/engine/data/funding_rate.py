from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, Deque
from collections import deque
from engine.events import EventBus, Event, EventType

@dataclass
class FundingSignal:
    """Funding rate signal for sentiment analysis"""
    timestamp: datetime
    symbol: str
    funding_rate: float
    sentiment: str           # 'LONG_CROWDED', 'SHORT_CROWDED', 'NEUTRAL'
    extremity: float         # 0.0 to 1.0, how extreme is current rate
    rate_change_1h: float    # Change in funding over last hour


class FundingRateTracker:
    """
    Tracks funding rates as a sentiment/positioning indicator.
    """
    
    def __init__(
        self,
        event_bus: EventBus,
        long_crowded_threshold: float = 0.0005,   # 0.05% per 8 hours
        short_crowded_threshold: float = -0.0002, # -0.02% per 8 hours
        history_hours: int = 24,
    ):
        self.event_bus = event_bus
        self.long_threshold = long_crowded_threshold
        self.short_threshold = short_crowded_threshold
        self._history: Deque = deque(maxlen=history_hours * 3)
        self._current_rate: float = 0.0
        self._last_update: Optional[datetime] = None
    
    async def update(self, funding_data: dict) -> Optional[FundingSignal]:
        """Update with new funding rate data from markPrice stream"""
        # Parse binance payload
        # "r": "0.00010000" // Funding rate
        # "T": 167...       // Next funding time
        # "E": 167...       // Event time
        
        try:
            rate = float(funding_data.get('r', 0.0))
            timestamp = datetime.fromtimestamp(funding_data.get('E', 0) / 1000.0)
            symbol = funding_data.get('s')
            
            self._current_rate = rate
            self._last_update = timestamp
            self._history.append({'time': timestamp, 'rate': rate})
            
            # Determine sentiment
            if rate > self.long_threshold:
                sentiment = 'LONG_CROWDED'
            elif rate < self.short_threshold:
                sentiment = 'SHORT_CROWDED'
            else:
                sentiment = 'NEUTRAL'
            
            # Calculate extremity (how far from neutral)
            if rate > 0:
                extremity = min(1.0, rate / (self.long_threshold * 2)) if self.long_threshold else 0
            else:
                extremity = min(1.0, abs(rate) / (abs(self.short_threshold) * 2)) if self.short_threshold else 0
            
            # Calculate rate change over last hour
            rate_change_1h = self._calculate_rate_change(hours=1)
            
            signal = FundingSignal(
                timestamp=timestamp,
                symbol=symbol,
                funding_rate=rate,
                sentiment=sentiment,
                extremity=extremity,
                rate_change_1h=rate_change_1h,
            )
            
            # Publish event if significant change or extreme
            if sentiment != 'NEUTRAL':
                 await self.event_bus.publish(Event(
                    type=EventType.BIAS_UPDATE, # Or a specific FUNDING_UPDATE event
                    source="funding_tracker",
                    payload={"sentiment": sentiment, "rate": rate}
                ))
            
            return signal
            
        except Exception as e:
            print(f"Error updating funding rate: {e}")
            return None
    
    def _calculate_rate_change(self, hours: int) -> float:
        """Calculate change in funding rate over specified hours"""
        if len(self._history) < 2:
            return 0.0
        
        cutoff = datetime.now() - timedelta(hours=hours) # Use now() as approx
        if self._last_update:
            cutoff = self._last_update - timedelta(hours=hours)
            
        old_rates = [h['rate'] for h in self._history if h['time'] < cutoff]
        
        if not old_rates:
            return 0.0
        
        return self._current_rate - old_rates[-1]
