from dataclasses import dataclass

@dataclass(frozen=True)
class PortfolioConfig:
    tickers: tuple = (
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'V',
        'JPM', 'JNJ', 'XOM', 'PG', 'SPY',
        'GLD', 'TLT', 'EFA', 'VWO', 'VNQ'
    )
    history_period: str = "5y"
    trading_days_per_year:int = 252
    alpha: float = 0.02
    risk_free_rate: float = 0.045

CONFIG = PortfolioConfig()