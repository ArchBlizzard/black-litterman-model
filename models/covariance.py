from pypfopt import risk_models

def get_covariance_matrix(prices, method='ledoit_wolf'):
    """Calculate covariance matrix."""
    if method == 'sample':
        return risk_models.sample_cov(prices)
    elif method == 'exponential':
        return risk_models.exp_cov(prices)
    elif method == 'ledoit_wolf':
        return risk_models.CovarianceShrinkage(prices).ledoit_wolf()
    else:
        raise ValueError(f"Unknown covariance method: {method}")