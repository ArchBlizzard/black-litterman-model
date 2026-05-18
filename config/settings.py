import os

# Project paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
PLOT_DIR = os.path.join(OUTPUT_DIR, 'plots')
REPORT_DIR = os.path.join(OUTPUT_DIR, 'reports')

# Backtest Settings
LOOKBACK_WINDOW = 252  # 1 year of trading days
REBALANCE_FREQ = 'ME'  # Monthly (Month End, strictly compliant with Pandas 2.2+)
TRANSACTION_COST = 0.001 # 10 bps
VOLATILITY_TARGET = 0.12 # 12% annualized target volatility

# Black-Litterman Settings
TAU = 0.05
RISK_AVERSION = 3.0

# Portfolio Optimization Constraints
MAX_STOCK_WEIGHT = 0.10 # 10% maximum individual weight

# Regime Settings
BULL_MA_FAST = 50
BULL_MA_SLOW = 200
BEAR_RISK_AVERSION_MULTIPLIER = 1.5   # Increase risk aversion in bear market
BEAR_CONFIDENCE_MULTIPLIER = 0.5      # Downscale confidence in bear market