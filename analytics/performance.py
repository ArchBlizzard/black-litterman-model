import numpy as np
import pandas as pd

def compute_metrics(returns, risk_free_rate=0.0):
    """Compute performance metrics."""
    # Ensure returns have no exact zeros or nans before calc
    returns = returns[returns != 0.0].dropna()
    if len(returns) == 0:
        return {'CAGR': 0, 'Ann Volatility': 0, 'Sharpe Ratio': 0, 'Sortino Ratio': 0, 'Max Drawdown': 0, 'Calmar Ratio': 0}
        
    # Annualized return
    cum_ret = (1 + returns).prod()
    years = len(returns) / 252
    cagr = cum_ret ** (1 / years) - 1 if years > 0 and cum_ret > 0 else 0
    
    # Volatility
    ann_vol = returns.std() * np.sqrt(252)
    
    # Sharpe
    sharpe = (cagr - risk_free_rate) / ann_vol if ann_vol > 0 else 0
    
    # Maximum Drawdown
    cum_returns = (1 + returns).cumprod()
    rolling_max = cum_returns.cummax()
    drawdowns = (cum_returns - rolling_max) / rolling_max
    max_dd = drawdowns.min()
    
    # Calmar Ratio
    calmar = cagr / abs(max_dd) if abs(max_dd) > 0 else 0
    
    # Sortino
    downside_returns = returns[returns < 0]
    downside_vol = downside_returns.std() * np.sqrt(252)
    sortino = (cagr - risk_free_rate) / downside_vol if downside_vol > 0 else 0
    
    # Information Ratio (Proxy using simple standard deviation of returns if benchmark not given here, 
    # but actual IR should be relative. For simplicity, reporting proxy absolute IR as annualized ret/vol without risk free)
    information_ratio = cagr / ann_vol if ann_vol > 0 else 0
    
    return {
        'CAGR': cagr,
        'Ann Volatility': ann_vol,
        'Sharpe Ratio': sharpe,
        'Sortino Ratio': sortino,
        'Max Drawdown': max_dd,
        'Calmar Ratio': calmar,
        'Information Ratio': information_ratio
    }

def compute_regime_metrics(returns, regime_history, risk_free_rate=0.0):
    """Compute metrics broken down by market regime."""
    results = []
    # Align indices
    aligned = pd.concat([returns, regime_history], axis=1).dropna()
    aligned.columns = ['Returns', 'Regime']
    
    for regime_val, regime_name in [(1.0, 'Bull'), (-1.0, 'Bear')]:
        mask = aligned['Regime'] == regime_val
        if not mask.any():
            continue
        regime_rets = aligned.loc[mask, 'Returns']
        
        # We manually annualize considering the subset of days
        metrics = compute_metrics(regime_rets, risk_free_rate)
        
        # Override CAGR/Vol for accurate partial-year representation
        cum_ret = (1 + regime_rets).prod() - 1
        metrics['Period Return'] = cum_ret
        metrics['Regime'] = regime_name
        
        results.append(metrics)
        
    if results:
        df = pd.DataFrame(results).set_index('Regime')
        return df
    return pd.DataFrame()