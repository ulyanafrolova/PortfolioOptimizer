import os
import asyncio
from dotenv import load_dotenv
from openai import AsyncOpenAI

from config import CONFIG
from data_provider import fetch_historical_prices, fetch_news_yfinance
from portfolio_math import calculate_annual_metrics, get_portfolio_performance, optimize_portfolio
from sentiment_analyzer import get_llm_sentiment, run_sentiment_analysis

async def main():
    print("--- Starting Portfolio Pipeline ---")
    
    # 1. Initialize
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is missing in environment variables.")
    openai_client = AsyncOpenAI(api_key=api_key)
    
    # 2. Get data
    prices = fetch_historical_prices(CONFIG.tickers, CONFIG.history_period)    
    valid_tickers = prices.columns.tolist()
    print(f"\nActive tickers for analysis ({len(valid_tickers)}): {valid_tickers}")
    
    # 3. Calculations    
    mean_returns, cov_matrix = calculate_annual_metrics(prices, CONFIG.trading_days_per_year)
    adjusted_returns = mean_returns.copy()
    
    # 4. Asynchronious AI Sentiment Overlay
    print("\nRunning asynchronous LLM analysis...")
    sentiment_scores = await run_sentiment_analysis(openai_client, valid_tickers)
    
    for ticker, score in sentiment_scores.items():
        adjustment = CONFIG.alpha * score
        adjusted_returns[ticker] += adjustment
        print(f"[{ticker}] AI Score: {score:>5.2f} | Adj: {adjustment:+.4f}")

    # 5. Optimization
    print("\nRunning SLSQP optimization...")
    optimal_weights = optimize_portfolio(adjusted_returns, cov_matrix, CONFIG.risk_free_rate)
    
    # 6. Result
    opt_return, opt_volatility = get_portfolio_performance(optimal_weights, adjusted_returns, cov_matrix)
    
    print("\n=== FINAL OPTIMAL PORTFOLIO ===")
    for i, ticker in enumerate(valid_tickers):
        if optimal_weights[i] > 0.01: # Показываем только веса больше 1%
            print(f"{ticker:<5}: {optimal_weights[i]*100:>5.2f}%")
            
    print("-" * 20)
    print(f"Expected Return: {opt_return * 100:.2f}%")
    print(f"Volatility (Risk): {opt_volatility * 100:.2f}%")
    print(f"Sharpe Ratio: {(opt_return - CONFIG.risk_free_rate) / opt_volatility:.2f}")
    
if __name__ == "__main__":
    asyncio.run(main())
    