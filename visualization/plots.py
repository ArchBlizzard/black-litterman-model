import os
import matplotlib.pyplot as plt
from config import settings

def plot_benchmark_comparison(returns_df, regime_history=None, plot_dir=settings.PLOT_DIR):
    """Plot cum returns of strategy vs benchmarks with regime overlays."""
    cum_rets = (1 + returns_df).cumprod()
    
    plt.figure(figsize=(12, 6))
    for col in cum_rets.columns:
        if col == 'Strategy':
            plt.plot(cum_rets.index, cum_rets[col], label=col, linewidth=2, color='tab:blue')
        else:
            plt.plot(cum_rets.index, cum_rets[col], label=col, alpha=0.7, linestyle='--')
            
    if regime_history is not None:
        # Overlay market regimes
        bear_mask = regime_history == -1
        # Fill regions where bear market is True
        import matplotlib.transforms as mtransforms
        plt.fill_between(regime_history.index, 0, 1, where=bear_mask, 
                         color='red', alpha=0.15, transform=plt.gca().get_xaxis_transform(),
                         label='Bear Regime (-1)')
                         
    plt.title('Strategy vs Benchmarks Cumulative Return')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'benchmark_comparison.png'))
    plt.close()

def plot_underwater_drawdown(strategy_returns, plot_dir=settings.PLOT_DIR):
    """Plot the underwater drawdown curve."""
    cum_returns = (1 + strategy_returns).cumprod()
    rolling_max = cum_returns.cummax()
    drawdowns = (cum_returns - rolling_max) / rolling_max

    plt.figure(figsize=(12, 5))
    plt.fill_between(drawdowns.index, drawdowns, 0, color='tab:red', alpha=0.3)
    plt.title('Strategy Underwater Drawdown')
    plt.ylabel('Drawdown (%)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'underwater_drawdown.png'))
    plt.close()

def plot_rolling_volatility(strategy_returns, window=126, plot_dir=settings.PLOT_DIR):
    """Plot rolling volatility."""
    roll_vol = strategy_returns.rolling(window).std() * (252 ** 0.5)
    plt.figure(figsize=(12, 5))
    plt.plot(roll_vol.index, roll_vol, color='tab:orange')
    plt.axhline(settings.VOLATILITY_TARGET, color='tab:grey', linestyle='--', label='Target Volatility')
    plt.title('Rolling 6-Month Volatility')
    plt.ylabel('Annualized Volatility')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'rolling_volatility.png'))
    plt.close()

def plot_rolling_sharpe(strategy_returns, window=126, regime_history=None, plot_dir=settings.PLOT_DIR):
    """Plot rolling Sharpe ratio."""
    roll_ret = strategy_returns.rolling(window).mean() * 252
    roll_vol = strategy_returns.rolling(window).std() * (252 ** 0.5)
    roll_sharpe = roll_ret / (roll_vol + 1e-6)
    
    plt.figure(figsize=(12, 5))
    plt.plot(roll_sharpe.index, roll_sharpe, color='tab:green', linewidth=1.5)
    plt.axhline(0, color='black', linestyle='--', alpha=0.5)
    
    if regime_history is not None:
        bear_mask = regime_history == -1
        plt.fill_between(regime_history.index, plt.gca().get_ylim()[0], plt.gca().get_ylim()[1], 
                         where=bear_mask, color='red', alpha=0.1, label='Bear Regime')
                         
    plt.title('Rolling 6-Month Sharpe Ratio')
    plt.ylabel('Annualized Sharpe Ratio')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'rolling_sharpe.png'))
    plt.close()