# Quantitative Black-Litterman Framework

This project implements an institutional-grade Black-Litterman portfolio optimization system in Python.

## Core Features
- **Rolling Walk-Forward Backtesting**: Monthly rebalancing using a 252-day lookback window.
- **Dynamic Views**: Cross-sectional momentum-based signals generate dynamic bullish/bearish views.
- **Dynamic Confidence Model**: Signal confidence is inversely proportional to historical volatility.
- **Transaction Costs**: Realistic 10bps transaction cost models included in net return calculation.
- **Risk Metrics**: Sharpe, Sortino, Max Drawdown calculation and more.
- **Modular Architecture**: Professional structure strictly separating data, signals, models, and backtesting.

## Installation

```bash
pip install -r requirements.txt
```

## Running the Backtest

```bash
python main.py
```

## Structure
- `config/`: Global project configurations.
- `models/`: Covariance, Regime, and Black-Litterman implementations.
- `signals/`: Dynamic Alpha and Confidence signals.
- `backtest/`: Walk-forward backtesting execution engine.
- `analytics/`: Risk and return computations.
- `visualization/`: Dashboard generation and plotting.
- `outputs/`: Automatically populated with metrics and plots post-run.
