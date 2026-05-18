import pandas as pd

def generate_signals(prices, current_date):
    """Generate dynamic views based on a multi-factor alpha model."""
    historical_prices = prices.loc[:current_date]
    if len(historical_prices) < 252 + 1:
        return {}
        
    # Calculate momentums
    p_t = historical_prices.iloc[-1]
    mom_3m = (p_t / historical_prices.iloc[-63]) - 1
    mom_6m = (p_t / historical_prices.iloc[-126]) - 1
    mom_12m = (p_t / historical_prices.iloc[-252]) - 1
    
    # Calculate rolling Sharpe (6-month)
    returns = historical_prices.iloc[-126:].pct_change().dropna()
    vols = returns.std() * (252 ** 0.5)
    ann_rets = (1 + returns.mean()) ** 252 - 1
    rolling_sharpe = ann_rets / (vols + 1e-6)
    
    low_volatility = -vols # Negative because lower vol is better

    # Z-score normalize factors cross-sectionally
    def z_score(series):
        return (series - series.mean()) / (series.std() + 1e-6)
        
    z_3m = z_score(mom_3m)
    z_6m = z_score(mom_6m)
    z_12m = z_score(mom_12m)
    z_sharpe = z_score(rolling_sharpe)
    z_low_vol = z_score(low_volatility)
    
    # Composite Score (Multi-Factor Alpha)
    score = 0.35 * z_3m + 0.35 * z_6m + 0.15 * z_low_vol + 0.15 * z_sharpe
    
    ranks = score.rank(pct=True)
    
    views = {}
    for asset, rank in ranks.items():
        if rank > 0.7:
            views[asset] = 0.05  # Bullish view: 5% expected return
        elif rank < 0.3:
            views[asset] = -0.05 # Bearish view: -5% expected return
            
    # Save the absolute score mapping for confidence modeling
    global _LAST_SCORES 
    _LAST_SCORES = score.to_dict()
            
    return views

def generate_confidence(prices, current_date, views):
    """Generate confidence based on signal strength & inverse volatility."""
    lookback = 126
    historical_prices = prices.loc[:current_date]
    if len(historical_prices) < lookback:
        return {k: 0.5 for k in views.keys()}
        
    returns = historical_prices.iloc[-lookback:].pct_change().dropna()
    vols = returns.std() * (252 ** 0.5)
    
    scores = globals().get('_LAST_SCORES', {})
    
    confidences = {}
    for asset in views.keys():
        vol = vols.get(asset, 0.2)
        score_mag = abs(scores.get(asset, 0.0))
        
        if vol > 0:
            # Confidence proportional to signal strength over volatility
            conf = score_mag / vol
            confidences[asset] = conf
        else:
            confidences[asset] = 0.5
            
    # Normalize between 0.1 and 0.95
    if not confidences:
        return {}
        
    c_vals = list(confidences.values())
    c_min, c_max = min(c_vals), max(c_vals)
    
    if c_max == c_min:
        return {k: 0.5 for k in confidences.keys()}
        
    for asset in confidences:
        norm = (confidences[asset] - c_min) / (c_max - c_min)
        confidences[asset] = 0.10 + norm * 0.85
        
    return confidences