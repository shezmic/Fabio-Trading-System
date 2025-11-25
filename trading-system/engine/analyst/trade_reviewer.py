from engine.analyst.llm_client import LLMClient
from engine.data.schemas import TradeSignal

class TradeReviewer:
    """
    Reviews executed trades against the trading plan.
    """
    
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client
        
    async def review_trade(self, trade_data: dict, signal: TradeSignal) -> str:
        """
        Generate a post-trade review.
        """
        prompt = f"""
        Review this trade:
        Symbol: {trade_data.get('symbol')}
        Entry: {trade_data.get('entry_price')}
        Exit: {trade_data.get('exit_price')}
        PnL: {trade_data.get('pnl')}
        
        Original Signal Rationale: {signal.rationale}
        Confluence Score: {signal.confluence_score}
        
        Did the trade follow the plan? What could be improved?
        """
        
        return await self.llm.generate_completion(prompt)
