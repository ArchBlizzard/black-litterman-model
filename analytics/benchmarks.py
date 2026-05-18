import pandas as pd
import numpy as np

def construct_equal_weight_benchmark(prices):
    """Construct an equal weight portfolio benchmark."""
    returns = prices.pct_change().dropna()
    weights = np.ones(prices.shape[1]) / prices.shape[1]
    ew_returns = returns.dot(weights)
    return ew_returns

def construct_mcap_weight_benchmark(prices, mcaps):
    """Construct a market-cap weighted benchmark."""
    returns = prices.pct_change().dropna()
    # Normalize mcaps to weights
    total_mcap = sum(mcaps.values())
    weights = np.array([mcaps.get(col, 0)/total_mcap for col in prices.columns])
    cw_returns = returns.dot(weights)
    return cw_returns

def generate_benchmark_comparison(strategy_returns, prices, mcaps=None, risk_free_rate=0.0):
    """Generate returns comparison against basic benchmarks."""
    from analytics.performance import compute_metrics
    import yfinance as yf
    
    benchmarks = {
        'Strategy': strategy_returns
    }
    
    # Optional NIFTY Proxy or Exact NIFTY if found
    if 'NIFTY' in prices.columns:
        benchmarks['NIFTY 50'] = prices['NIFTY'].pct_change().dropna()
    elif '^NSEI' in prices.columns:
        benchmarks['NIFTY 50'] = prices['^NSEI'].pct_change().dropna()
    else:
        # Download NIFTY 50 strictly aligned to the prices dates
        try:
            nifty = yf.download('^NSEI', start=prices.index[0], end=prices.index[-1], progress=False)['Close']
            if isinstance(nifty, pd.DataFrame):
                nifty = nifty.squeeze()
            nifty_rets = nifty.pct_change().dropna()
            
            # Align timezone to match the input returns
            if rets.index.tz is not None:
                if nifty_rets.index.tz is None:
                    nifty_rets.index = nifty_rets.index.tz_localize(rets.index.tz)
                else:
                    nifty_rets.index = nifty_rets.index.tz_convert(rets.index.tz)
            else:
                if nifty_rets.index.tz is not None:
                    nifty_rets.index = nifty_rets.index.tz_localize(None)

            # Reindex to ensure strict alignment
            nifty_rets = nifty_rets.reindex(rets.index).fillna(0.0)
            strat_idx = strategy_returns.index.tz_localize(None) if strategy_returns.index.tz is not None else strategy_returns.index
            nifty_rets.index = nifty_rets.index.normalize()
            strat_idx = strat_idx.normalize()
            
            # Map index
            benchmarks['NIFTY 50'] = nifty_rets.reindex(strat_idx).fillna(0)
        except Exception:
            pass
        
    benchmarks['Equal Weight'] = construct_equal_weight_benchmark(prices)
    if mcaps:
        benchmarks['Market Cap Weight'] = construct_mcap_weight_benchmark(prices, mcaps)
    
    # Align dates
    df = pd.DataFrame(benchmarks).dropna()
    
    metrics_list = []
    for name, r in df.items():
        met = compute_metrics(r, risk_free_rate)
        met['Portfolio'] = name
        metrics_list.append(met)
        
    result_df = pd.DataFrame(metrics_list).set_index('Portfolio')
    return result_df, df