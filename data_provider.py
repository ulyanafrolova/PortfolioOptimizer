import yfinance as yf
import pandas as pd
import requests

def fetch_historical_prices(tickers: tuple, period: str) -> pd.DataFrame:
    """
    Downloads and cleans historical adjusted close prices
    """
    print("Downloading historical data...")
    data = yf.download(list(tickers), period=period)['Close'] 
    print("\nMissing data per ticker before cleaning:")
    print(data.isna().sum())
    max_missing = len(data) * 0.05
    data = data.loc[:, data.isna().sum() <= max_missing]
    data = data.ffill()
    data = data.dropna(axis=0)
    print(f"Data shape after cleaning: {data.shape}")
    return data

def fetch_news_yfinance(ticker:str, limit:int=5) -> list:
    """
    Pulls the latest news headlines for a ticker using Yahoo Finance.
    Handles API structure changes robustly using .get()
    """
    try:
        stock = yf.Ticker(ticker)
        news = stock.news 
        headlines = []
        for item in news[:limit]:
            title = item.get('title') 
            if not title and 'content' in item:
                title = item['content'].get('title')
            if title:
                headlines.append(title)
                
        return headlines
    
    except requests.exceptions.RequestException as e:
        print(f"Network error fetching news for {ticker}: {e}")
        return []
    except (KeyError, TypeError) as e:
        print(f"Data structure error for {ticker}: {e}")
        return []