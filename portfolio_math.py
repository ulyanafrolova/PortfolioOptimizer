import numpy as np
import pandas as pd
from scipy.optimize import minimize

def calculate_annual_metrics(prices: pd.DataFrame, trading_days: int):
    """
    Calculates annual_mean_returns and annual_cov_matrix.
    """
    log_returns = np.log(prices / prices.shift(1))
    log_returns = log_returns.dropna()
    
    annual_mean_returns = log_returns.mean() * trading_days
    annual_cov_matrix = log_returns.cov() * trading_days

    return annual_mean_returns, annual_cov_matrix 
    
def get_portfolio_performance(weights: np.ndarray, mean_returns: pd.Series, cov_matrix: pd.DataFrame) -> tuple[float, float]:
    """
    Calculates the expected return and volatility (risk) of a portfolio.
    """
    assert len(weights) == cov_matrix.shape[0], "Weights array length must match covariance matrix dimensions."    
    port_return = np.sum(weights * mean_returns)
    port_volatility = np.sqrt(weights @ cov_matrix @ weights)   
     
    return port_return, port_volatility

def min_func_sharpe(weights, mean_returns, cov_matrix, risk_free_rate=0.045):
    """
    Calculates the negative Sharpe ratio
    """
    p_return, p_volatility = get_portfolio_performance(weights, mean_returns, cov_matrix)
    sharpe_ratio = (p_return - risk_free_rate) / p_volatility
    return -sharpe_ratio

def optimize_portfolio(mean_returns: pd.Series, cov_matrix: pd.DataFrame, risk_free_rate: float) -> np.ndarray:
    """
    Finds the optimal portfolio asset weights that maximize the Sharpe ratio 
    using Sequential Least Squares Programming (SLSQP).
    """
    num_assets = len(mean_returns)
    initial_guess = np.repeat(1 / num_assets, num_assets)
    adjusted_mean_returns = mean_returns.copy()
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))
    
    optimized_results = minimize(
        min_func_sharpe, 
        initial_guess, 
        args=(adjusted_mean_returns, cov_matrix, risk_free_rate),
        method='SLSQP', 
        bounds=bounds, 
        constraints=constraints
    )
    
    if not optimized_results.success:
        print(f"Warning: Optimizer failed to converge. Reason: {optimized_results.message}")
        
    return optimized_results.x