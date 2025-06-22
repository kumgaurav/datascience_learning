"""
Tabs Module

Contains all tab components for the stock detail page.
"""

from .rsi_tab import RSITab
from .rsi_analysis_tab import RSIAnalysisTab
from .trading_signals_tab import TradingSignalsTab
from .price_ma_bb_tab import PriceMABBTab
from .rsi_stoch_tab import RSIStochTab
from .macd_tab import MACDTab
from .volume_obv_tab import VolumeOBVTab
from .atr_tab import ATRTab
from .best_performers_tab import BestPerformersTab
from .consistent_performers_tab import ConsistentPerformersTab
from .volatile_stocks_tab import VolatileStocksTab
from .earnings_stocks_tab import EarningsStocksTab

__all__ = [
    'RSITab',
    'RSIAnalysisTab', 
    'TradingSignalsTab',
    'PriceMABBTab',
    'RSIStochTab',
    'MACDTab',
    'VolumeOBVTab',
    'ATRTab',
    'BestPerformersTab',
    'ConsistentPerformersTab',
    'VolatileStocksTab',
    'EarningsStocksTab'
] 