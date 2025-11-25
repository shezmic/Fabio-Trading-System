import asyncio
import argparse
from engine.backtest.vectorized_engine import VectorizedBacktester
# from engine.data.historical_loader import HistoricalLoader

async def main():
    parser = argparse.ArgumentParser(description="Run Backtest")
    parser.add_argument("--symbol", type=str, default="BTCUSDT")
    parser.add_argument("--strategy", type=str, default="TrendContinuation")
    args = parser.parse_args()
    
    print(f"Running backtest for {args.strategy} on {args.symbol}...")
    
    # Load data
    # loader = HistoricalLoader()
    # data = await loader.load_candles(...)
    
    # Run backtest
    # bt = VectorizedBacktester(data)
    # results = bt.run(...)
    
    print("Backtest complete.")

if __name__ == "__main__":
    asyncio.run(main())
