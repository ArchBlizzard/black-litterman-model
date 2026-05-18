import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from backtest.engine import WalkForwardBacktest
from analytics.performance import compute_metrics
from config import settings

def run_sensitivity_analysis(prices, mcaps, plot_dir=settings.PLOT_DIR):
    """
    Run stress testing and sensitivity analysis on parameters.
    """
    print("Running Sensitivity Analysis (this may take a minute)...")
    
    # 1. Tau vs Risk Aversion Sensitivity
    taus = [0.01, 0.05, 0.10, 0.15]
    risk_aversions = [1.0, 3.0, 5.0, 7.0]
    
    sharpe_matrix = pd.DataFrame(index=taus, columns=risk_aversions)
    drawdown_matrix = pd.DataFrame(index=taus, columns=risk_aversions)
    
    original_tau = settings.TAU
    original_ra = settings.RISK_AVERSION
    
    for t in taus:
        for ra in risk_aversions:
            settings.TAU = t
            settings.RISK_AVERSION = ra
            
            bt = WalkForwardBacktest(prices, mcaps)
            rets, _, _ = bt.run(optimization_method='max_sharpe')
            
            metrics = compute_metrics(rets)
            sharpe_matrix.loc[t, ra] = metrics['Sharpe Ratio']
            drawdown_matrix.loc[t, ra] = metrics['Max Drawdown']

    # Restore settings
    settings.TAU = original_tau
    settings.RISK_AVERSION = original_ra
    
    sharpe_matrix = sharpe_matrix.astype(float)
    drawdown_matrix = drawdown_matrix.astype(float)
    
    # Plot heatmaps
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    sns.heatmap(sharpe_matrix, annot=True, cmap='RdYlGn', fmt=".2f", ax=axes[0])
    axes[0].set_title("Sharpe Ratio Sensitivity")
    axes[0].set_xlabel("Risk Aversion")
    axes[0].set_ylabel("Tau")
    
    sns.heatmap(drawdown_matrix, annot=True, cmap='RdYlGn', fmt=".2%", ax=axes[1])
    axes[1].set_title("Max Drawdown Sensitivity")
    axes[1].set_xlabel("Risk Aversion")
    axes[1].set_ylabel("Tau")
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'sensitivity_heatmaps.png'))
    plt.close()
    
    return sharpe_matrix, drawdown_matrix