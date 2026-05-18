from pypfopt import EfficientFrontier, objective_functions
from config import settings

def optimize_portfolio(expected_returns, cov_matrix, method='max_sharpe', risk_aversion=None):
    """Optimize portfolio weights based on expected returns and covariance."""
    # Ensure correct index and columns, add weight bounds (Max 10% per stock)
    bounds = (0.0, settings.MAX_STOCK_WEIGHT)
    
    # Check if number of assets allows for MAX_STOCK_WEIGHT (e.g. need at least 10 assets if max is 10%)
    if len(expected_returns) * settings.MAX_STOCK_WEIGHT < 1.0:
        bounds = (0.0, 1.0) # relaxing if constraints are impossible
        
    ef = EfficientFrontier(expected_returns, cov_matrix, weight_bounds=bounds)
    
    # Optionally penalize excessive turnover or extreme weights
    ef.add_objective(objective_functions.L2_reg, gamma=0.1) # Small L2 regularization for stability
    
    if method == 'max_sharpe':
        ef.max_sharpe()
    elif method == 'min_vol':
        ef.min_volatility()
    elif method == 'max_quadratic_utility':
        ra = risk_aversion if risk_aversion else settings.RISK_AVERSION
        ef.max_quadratic_utility(risk_aversion=ra)
    else:
        raise ValueError(f"Unknown optimization method: {method}")
        
    weights = ef.clean_weights()
    return weights