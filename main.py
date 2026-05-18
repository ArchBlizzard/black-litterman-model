import pandas as pd
import os
import matplotlib.pyplot as plt
import warnings

# Suppress all warnings for clean console output
warnings.filterwarnings("ignore")

from config import settings
from backtest.engine import WalkForwardBacktest
from analytics.performance import compute_metrics, compute_regime_metrics
from analytics.stress_testing import run_sensitivity_analysis
from analytics.benchmarks import generate_benchmark_comparison
from visualization.plots import plot_benchmark_comparison, plot_underwater_drawdown, plot_rolling_volatility, plot_rolling_sharpe

def main():
    print("Loading data...")
    prices_path = os.path.join(settings.DATA_DIR, 'prices.csv')
    mcap_path = os.path.join(settings.DATA_DIR, 'mcap.csv')
    
    # Load prices
    prices = pd.read_csv(prices_path, index_col=0, parse_dates=True)
    
    # Load mcaps
    mcaps_df = pd.read_csv(mcap_path, index_col=0)
    # Convert mcaps to a dictionary mapping ticker to market cap
    mcaps = mcaps_df.iloc[:, 0].to_dict()
    
    # Ensure mcaps only includes tickers that are in prices
    mcaps = {k: v for k, v in mcaps.items() if k in prices.columns}
    
    print("Running Walk-Forward Backtest...")
    bt = WalkForwardBacktest(prices, mcaps)
    rets, weights, turnover = bt.run(optimization_method='max_sharpe')
    
    # Analyze
    print("Computing metrics...")
    metrics = compute_metrics(rets)
    
    print("\n--- Strategy Performance Metrics ---")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
        
    print("\nGenerating Benchmark Comparisons...")
    benchmark_metrics, returns_df = generate_benchmark_comparison(rets, prices, mcaps)
    print("\n--- Benchmark Table ---")
    print(benchmark_metrics)
    
    print("\nGenerating Regime-Specific Analytics...")
    regime_metrics = compute_regime_metrics(rets, bt.regime_history)
    print("\n--- Regime Analytics ---")
    print(regime_metrics)
    
    print("\nGenerating Plots...")
    # Plot benchmark returns
    plot_benchmark_comparison(returns_df, regime_history=bt.regime_history)
    plot_underwater_drawdown(rets)
    plot_rolling_volatility(rets)
    plot_rolling_sharpe(rets, regime_history=bt.regime_history)
    
    # Run sensitivity analysis
    sharpe_matrix, drawdowns_matrix = run_sensitivity_analysis(prices, mcaps)
    
    # Summary Table Output
    with open(os.path.join(settings.REPORT_DIR, 'metrics.txt'), 'w') as f:
        f.write("--- Strategy Performance Metrics ---\n")
        pd.Series(metrics).to_csv(f, sep='\t')
        f.write("\n--- Benchmark Comparison ---\n")
        benchmark_metrics.to_csv(f, sep='\t')
        f.write("\n--- Regime Analytics ---\n")
        regime_metrics.to_csv(f, sep='\t')
        
    # Generate Research Report
    generate_research_report(metrics, benchmark_metrics, regime_metrics, sharpe_matrix)
            
    print("Done! Check outputs/ directory for results.")

def generate_research_report(metrics, benchmark_metrics, regime_metrics, sharpe_matrix):
    """Generate a clean markdown research report."""
    report_path = os.path.join(settings.REPORT_DIR, 'Research_Report.md')
    with open(report_path, 'w') as f:
        f.write("# Quantitative Portfolio Research Report\n\n")
        f.write("## 1. Strategy Performance Summary\n")
        for k, v in metrics.items():
            f.write(f"- **{k}**: {v:.4f}\n")
            
        f.write("\n## 2. Benchmark Comparison\n")
        f.write(benchmark_metrics.to_markdown())
        
        f.write("\n\n## 3. Regime Analytics\n")
        f.write("Performance breakdown during Bull (MA50 > MA200) vs Bear (MA50 < MA200) market regimes:\n\n")
        f.write(regime_metrics[['CAGR', 'Ann Volatility', 'Sharpe Ratio', 'Max Drawdown', 'Period Return']].to_markdown())
        
        f.write("\n\n## 4. Parameter Sensitivity (Sharpe Ratio)\n")
        f.write("Analysis of sensitivity to Tau (rows) and Risk Aversion (columns):\n\n")
        f.write(sharpe_matrix.to_markdown())
        
        f.write("\n\n## 5. Visualizations\n")
        f.write("Please check the `outputs/plots/` directory for:\n")
        f.write("1. `benchmark_comparison.png` - Cumulative returns with bear market regime overlays.\n")
        f.write("2. `underwater_drawdown.png` - Institutional peak-to-trough drawdown curves.\n")
        f.write("3. `rolling_volatility.png` - Realized volatility vs configured target volatility.\n")
        f.write("4. `rolling_sharpe.png` - Out-of-sample rolling Sharpe ratio stability.\n")

if __name__ == "__main__":
    main()