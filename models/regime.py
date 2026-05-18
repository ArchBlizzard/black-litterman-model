import pandas as pd
from config import settings

def detect_regime(nifty_prices, current_date):
    """
    Detect market regime (bull or bear) using NIFTY moving averages.
    Returns +1 for bull, -1 for bear.
    """
    history = nifty_prices.loc[:current_date]
    if len(history) < settings.BULL_MA_SLOW:
        return 1 # Default to bull if not enough history
        
    ma_fast = history.iloc[-settings.BULL_MA_FAST:].mean()
    ma_slow = history.iloc[-settings.BULL_MA_SLOW:].mean()
    
    if ma_fast > ma_slow:
        return 1
    else:
        return -1