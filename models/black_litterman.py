import numpy as np
import pandas as pd
from pypfopt import black_litterman, risk_models

class BlackLittermanModel:
    def __init__(self, cov_matrix, mcaps, views_dict, confidences_dict, tau=0.05, risk_aversion=3.0):
        self.cov_matrix = cov_matrix
        self.mcaps = mcaps
        self.views_dict = views_dict
        self.confidences_dict = confidences_dict
        self.tau = tau
        self.risk_aversion = risk_aversion
        
    def get_market_prior(self):
        """Calculate market-implied prior returns."""
        market_prior = black_litterman.market_implied_prior_returns(
            self.mcaps, self.risk_aversion, self.cov_matrix
        )
        return market_prior
        
    def compute_posterior(self):
        """Compute BL posterior returns and covariance."""
        prior = self.get_market_prior()
        
        # PyPortfolioOpt expects a list of confidences in the same order as absolute_views
        # self.views_dict is {ticker: view}
        # self.confidences_dict is {ticker: confidence}
        confidences_list = [self.confidences_dict[ticker] for ticker in self.views_dict.keys()]
        
        bl = black_litterman.BlackLittermanModel(
            self.cov_matrix, 
            pi=prior, 
            absolute_views=self.views_dict, 
            omega="idzorek", 
            view_confidences=confidences_list,
            tau=self.tau
        )
        ret = bl.bl_returns()
        cov = bl.bl_cov()
        return ret, cov