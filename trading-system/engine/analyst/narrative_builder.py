from engine.analyst.llm_client import LLMClient

class NarrativeBuilder:
    """
    Analyzes market context to build a pre-session narrative/bias.
    """
    
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client
        
    async def generate_narrative(self, market_data: dict) -> str:
        """
        Generate daily bias/narrative.
        """
        prompt = f"""
        Analyze the following market data for {market_data.get('symbol')}:
        Price: {market_data.get('price')}
        24h Change: {market_data.get('change_24h')}%
        Funding Rate: {market_data.get('funding_rate')}
        
        Determine the daily bias (Bullish/Bearish/Neutral) and key levels to watch.
        """
        
        return await self.llm.generate_completion(prompt)
