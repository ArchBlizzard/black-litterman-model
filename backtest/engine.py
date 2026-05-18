import pandas as pd
import numpy as np
from config import settings
from models.covariance import get_covariance_matrix
from models.black_litterman import BlackLittermanModel
from models.optimizer import optimize_portfolio
from models.regime import detect_regime
from signals.views import generate_signals, generate_confidence
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WalkForwardBacktest:
    def __init__(self, prices, mcaps):
        self.prices = prices
        self.mcaps = mcaps
        self.returns = prices.pct_change().dropna()
        self.weights_history = pd.DataFrame(index=self.returns.index, columns=prices.columns).fillna(0.0)
        self.portfolio_returns = pd.Series(index=self.returns.index, dtype=float).fillna(0.0)
        self.turnover = pd.Series(index=self.returns.index, dtype=float).fillna(0.0)
        self.regime_history = pd.Series(index=self.returns.index, dtype=float).fillna(1.0)
        self.exposure_history = pd.Series(index=self.returns.index, dtype=float).fillna(1.0)
        
        # Determine Proxy for NIFTY if not available directly -> use equal weight for regime detection if needed
        # Or better, if NIFTY is a column, use it. Else use mean price proxy.
        if 'NIFTY' in prices.columns:
            self.market_proxy = prices['NIFTY']
        elif '^NSEI' in prices.columns:
            self.market_proxy = prices['^NSEI']
        else:
            self.market_proxy = prices.mean(axis=1)
            
    def run(self, optimization_method='max_quadratic_utility'):
        # Generate monthly rebalance dates
        monthly_dates = self.prices.resample(settings.REBALANCE_FREQ).last().index
        
        current_weights = pd.Series(0.0, index=self.prices.columns)
        
        for i in range(len(monthly_dates) - 1):
            rebalance_date = monthly_dates[i]
            next_rebalance_date = monthly_dates[i+1]
            
            # Check for sufficient history
            history = self.prices.loc[:rebalance_date]
            if len(history) < settings.LOOKBACK_WINDOW:
                continue
                
            # Regime Detection
            regime = detect_regime(self.market_proxy, rebalance_date)
            self.regime_history.loc[rebalance_date] = regime
            
            # Adjust parameters based on regime
            current_risk_aversion = settings.RISK_AVERSION
            if regime == -1:
                current_risk_aversion *= settings.BEAR_RISK_AVERSION_MULTIPLIER
                
            logger.info(f"Rebalancing on {rebalance_date.date()} | Regime: {regime} | Risk Av: {current_risk_aversion}")
            
            # 1. Get training data
            train_prices = history.iloc[-settings.LOOKBACK_WINDOW:]
            
            # 2. Get covariance
            cov = get_covariance_matrix(train_prices)
            
            # 3. Generate views & confidences
            views_dict = generate_signals(self.prices, rebalance_date)
            
            if hasattr(self.mcaps, 'loc'):
                # Assuming single row or series
                if isinstance(self.mcaps, pd.DataFrame):
                     latest_mcaps = self.mcaps.loc[:rebalance_date].iloc[-1].to_dict()
                else:
                     latest_mcaps = self.mcaps.to_dict()
            else:
                latest_mcaps = self.mcaps
                
            try:
                if not views_dict:
                    new_weights = pd.Series(1.0/len(self.prices.columns), index=self.prices.columns)
                else:
                    confidences_dict = generate_confidence(self.prices, rebalance_date, views_dict)
                    
                    if regime == -1:
                        # Downscale confidences in bear market
                        confidences_dict = {k: v * settings.BEAR_CONFIDENCE_MULTIPLIER for k, v in confidences_dict.items()}
                    
                    # 4. Black Litterman
                    bl = BlackLittermanModel(cov, latest_mcaps, views_dict, confidences_dict)
                    bl_rets, bl_cov = bl.compute_posterior()
                    
                    # 5. Optimize
                    new_weights_dict = optimize_portfolio(
                        bl_rets, 
                        bl_cov, 
                        method=optimization_method, 
                        risk_aversion=current_risk_aversion
                    )
                    new_weights = pd.Series(new_weights_dict)
            except Exception as e:
                logger.warning(f"Optimization failed on {rebalance_date}: {e}. Retaining previous weights.")
                new_weights = current_weights
                
            # Compute turnover
            turnover_val = np.sum(np.abs(new_weights - current_weights))
            self.turnover.loc[rebalance_date] = turnover_val
            
            # Volatility Targeting (Exposure Scaling)
            # COV is already annualized by PyPortfolioOpt by default, so do NOT multiply by 252 again.
            port_var = np.dot(new_weights.T, np.dot(cov, new_weights))
            port_vol = np.sqrt(port_var) if port_var > 0 else 0
            
            target_exposure = 1.0
            if port_vol > 0:
                target_exposure = settings.VOLATILITY_TARGET / port_vol
            
            # Max leverage constraint (e.g. 1.0)
            target_exposure = min(target_exposure, 1.0)
            if regime == -1:
                target_exposure = min(target_exposure, 0.8) # max 80% exposure in bear markets
                
            self.exposure_history.loc[rebalance_date] = target_exposure
            
            # Apply weights to the next period
            period_mask = (self.returns.index > rebalance_date) & (self.returns.index <= next_rebalance_date)
            period_returns = self.returns.loc[period_mask]
            
            for dt in period_returns.index:
                # Deduct transaction costs on rebalance date
                cost = 0
                if dt == period_returns.index[0]:
                    cost = turnover_val * settings.TRANSACTION_COST
                
                # Apply exposure scaling
                ret = (np.dot(period_returns.loc[dt], new_weights) * target_exposure) - cost
                self.portfolio_returns.loc[dt] = ret
                self.weights_history.loc[dt] = new_weights * target_exposure
                
            current_weights = new_weights

        # Forward fill the regime and exposure histories for plotting
        self.regime_history = self.regime_history.ffill()
        self.exposure_history = self.exposure_history.ffill()
        
        return self.portfolio_returns, self.weights_history, self.turnover