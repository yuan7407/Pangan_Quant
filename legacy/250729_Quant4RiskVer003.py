"""
ETF强化型量化交易策略 Ver 0.03
250729_Quant4RiskVer003.py
基于 市场趋势判断，结合均线、MACD、RSI和成交量分析的高风险高回报量化交易策略，通过server酱推送通知到绑定微信上。
作者：Michael Yuan
"""

import datetime
import numpy as np
import pandas as pd
import backtrader as bt
import yfinance as yf
import random
import time
import plotly.graph_objects as go
import math
import requests
import json
import threading
import queue
import traceback
from plotly.subplots import make_subplots
from collections import deque
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

@dataclass
class TradingParameters:
    """
交易策略参数配置
"""
    # 基础参数
    symbol: str
    initial_cash: float
    start_date: str = None
    end_date: str = None
    commission: float = 0.001
    slippage: float = 0.002

    # 技术指标参数
    fast_ma: int = 12           # 从5改为12
    slow_ma: int = 24          # 从10改为24
    mid_ma: int = 48           # 从20改为48
    long_ma: int = 96          # 从50改为96
    macd_fast: int = 8         # 保持8
    macd_slow: int = 17        # 保持17
    macd_signal: int = 6       # 保持6
    atr_period: int = 14       # 从7改为14
    rsi_period: int = 14       # 从9改为14
    volatility_window: int = 20 # 从10改为20

    momentum_period: int = 10    # 从5改为10
