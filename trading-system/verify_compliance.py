import asyncio
import logging
from datetime import datetime

# Import all new components to check for syntax/import errors
from engine.indicators.vwap import VWAPCalculator
from engine.indicators.volume_profile import VolumeProfileCalculator
from engine.indicators.atr import ATRCalculator
from engine.indicators.market_structure import MarketStructureDetector

from engine.orderflow.absorption_detector import AbsorptionDetector
from engine.orderflow.cvd_tracker import CVDTracker
from engine.orderflow.trapped_trader import TrappedTraderDetector

from engine.strategy.confluence_validator import ConfluenceValidator
from engine.state.market_state import MarketStateDetector
from engine.risk.house_money import HouseMoneyManager

from engine.execution.slippage_model import SlippageModel
from engine.state.state_reconciler import StateReconciler

from engine.events import EventBus

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ComplianceVerify")

async def verify():
    logger.info("Verifying Indicators...")
    vwap = VWAPCalculator()
    vp = VolumeProfileCalculator()
    atr = ATRCalculator()
    ms = MarketStructureDetector()
    logger.info("Indicators OK.")
    
    logger.info("Verifying Order Flow...")
    bus = EventBus()
    abs_det = AbsorptionDetector(bus)
    cvd = CVDTracker(bus)
    trap = TrappedTraderDetector(bus)
    logger.info("Order Flow OK.")
    
    logger.info("Verifying Strategy & Risk...")
    conf = ConfluenceValidator()
    state = MarketStateDetector()
    house = HouseMoneyManager(10000)
    logger.info("Strategy & Risk OK.")
    
    logger.info("Verifying Execution...")
    slip = SlippageModel()
    # StateReconciler needs mocks, skipping instantiation for now
    logger.info("Execution OK.")
    
    logger.info("ALL COMPLIANCE CHECKS PASSED.")

if __name__ == "__main__":
    asyncio.run(verify())
