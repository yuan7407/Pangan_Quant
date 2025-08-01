"""
ETF强化型量化交易策略 Ver 0.19
250706_Quant4ETFVer019.py
基于 TD序列 和 市场趋势判断，结合均线、MACD、RSI和成交量分析的综合性 ETF量化交易策略，通过server酱推送通知到绑定微信上。
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
    """交易策略参数配置"""
    # 基础参数
    symbol: str
    initial_cash: float
    start_date: str = None
    end_date: str = None
    commission: float = 0.0001
    slippage: float = 0.001

    # TD序列参数
    td_count: int = 5
    td_check_days: int = 1

    # 技术指标参数
    fast_ma: int = 5
    slow_ma: int = 20
    mid_ma: int = 50
    long_ma: int = 100
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    atr_period: int = 14
    rsi_period: int = 14
    volatility_window: int = 20

    # 技术指标额外参数
    atr_ma_period: int = 20
    momentum_period: int = 10
    volume_base_period: int = 15
    volume_adjust_multiplier: int = 5

    # 成交量参数
    base_volume_mult: float = 0.8
    volume_adjust: float = 0.02

    # 风险管理参数
    risk_pct_per_trade: float = 0.05
    max_daily_loss: float = 0.15
    max_drawdown: float = 0.30
    trade_loss_pct: float = 0.15
    equity_risk: float = 0.3
    min_bars_between_trades: int = 10

    # 仓位参数
    base_position_pct: float = 0.6
    min_position_size: int = 10
    max_position_pct: float = 0.95

    # 获利了结参数
    profit_trigger_1: float = 0.40
    profit_trigger_2: float = 0.80
    position_scaler: float = 1.2
    td_require_regime: bool = True
    scale_cash_mult: float = 1.05

    # 止损参数
    stop_atr: float = 4.0
    trail_stop_mult: float = 3.0
    profit_lock_pct: float = 0.5
    force_min_stop_pct: float = 0.10
    exit_confirm_threshold: int = 1

    # 时间周期减仓参数
    time_based_profit_min_days: int = 180
    time_based_profit_min_pct: float = 0.10
    time_based_profit_size_pct: float = 0.2

    # 功能标志
    enable_trailing_stop: bool = True
    enable_basic_stop: bool = True
    enable_trend_reversal_stop: bool = True
    final_close: bool = True
    max_single_loss: float = 0.02
    position_scaling_mode: str = "kelly"  # 仓位计算模式：fixed/kelly/risk_parity

    # 市场状态参数
    strong_uptrend_threshold: float = 8
    uptrend_threshold: float = 4
    sideways_threshold: float = 0.5
    downtrend_threshold: float = -9

    max_consecutive_losses: int = 2
    loss_cooldown_days: int = 10  # 亏损后冷静期10天

    # 如果需要确认机制，添加新的有效参数
    regime_change_confirmation_days: int = 1

    # ETF特殊处理参数
    strong_uptrend_mult: float = 1.5
    uptrend_mult: float = 1.2
    sideways_mult: float = 1.0
    downtrend_mult: float = 0.7

    # 重新定义追踪止损参数的语义：直接设定合理的激活阈值
    trailing_activation_base: float = 0.03
    trailing_strong_uptrend_mult: float = 2.5
    trailing_uptrend_mult: float = 2.0
    trailing_sideways_mult: float = 2.5
    trailing_weak_trend_mult: float = 3.0
    trailing_downtrend_mult: float = 4.0

    new_position_protection_days: int = 3

    # 回调买入参数
    dip_base_pct: float = 0.02
    dip_per_batch: float = 0.005

    # 额外参数
    sell_cooldown: int = 2
    batches_allowed: int = 6

    lookback_days_for_dip: int = 20  # 回调检查的回望天数
    min_bars_after_add: int = 3  # 加仓后最小间隔天数
    min_meaningful_shares: int = 50  # 最小有意义的股数，避免过小交易

    td_sequence_weight: float = 1.0
    trend_weight: float = 1.0
    momentum_weight: float = 1.5
    pattern_weight: float = 1.0
    signal_threshold: float = 1.5

    # 趋势得分参数
    etf_trend_bias: float = 5.0
    trend_price_up_score: float = 3.0
    trend_price_accel_score: float = 3.0
    trend_price_decel_penalty: float = 2.0
    trend_long_up_score: float = 2.0
    trend_long_down_penalty: float = 3.0

    # 价格均线关系参数
    trend_price_above_fast_score: float = 3.0
    trend_price_below_fast_penalty: float = 2.0
    trend_price_above_slow_score: float = 2.0
    trend_price_below_slow_penalty: float = 3.0

    # 均线交叉参数
    trend_ma_bull_score: float = 3.0
    trend_ma_bear_penalty: float = 4.0
    trend_ma_golden_cross_score: float = 4.0
    trend_ma_death_cross_penalty: float = 5.0

    # 止损参数
    stop_long_days: int = 30
    stop_long_days_mult: float = 0.8
    stop_min_level_mult: float = 0.97

    # 追踪止损参数
    trailing_stop_min_profit: float = 0.25
    trailing_stop_min_drop: float = 0.15
    trailing_stop_long_holding: int = 120
    trailing_stop_rapid_drop: float = 0.08
    trailing_stop_trail_factor_high: float = 0.20
    trailing_stop_trail_factor_medium: float = 0.18
    trailing_stop_trail_factor_low: float = 0.15
    trailing_stop_trail_factor_min: float = 0.13
    trailing_stop_mult: float = 1.2
    trailing_stop_long_holding_factor: float = 0.85
    trailing_stop_medium_holding_factor: float = 0.92

    dynamic_stop_enabled: bool = True  # 启用动态止损
    stop_profit_threshold: float = 0.05  # 盈利5%后放宽止损
    stop_profit_relaxation: float = 1.5  # 止损放宽倍数

    # 仓位管理参数
    position_portfolio_factor_base: float = 80000.0
    position_min_size_factor: float = 1.5
    position_cash_reserve: float = 0.05
    position_portfolio_high: float = 0.9
    position_portfolio_medium: float = 0.7
    position_batch_decay: float = 0.02
    position_min_add_pct: float = 0.1

    serverchan_sendkey: str = "SCT119824TW3DsGlBjkhAV9lJWIOLohv1P"  # Server酱的SendKey
    enable_trade_notification: bool = True  # 是否启用交易通知
    enable_risk_notification: bool = True   # 是否启用风险提醒
    notification_min_pnl: float = 100.0     # 最小通知盈亏金额
    live_mode: bool = False  # 是否为实时交易模式
    data_check_interval: int = 60  # 数据检查间隔（秒）
    max_data_delay: int = 300  # 最大数据延迟（秒）
    generate_live_report: bool = True  # 实时运行结束后是否生成历史回测报告
    history_days: int = 3650

    # 调试设置
    debug_mode: bool = True

    def __post_init__(self):
        """初始化后处理，设置默认日期"""
        if self.start_date is None or self.end_date is None:
            default_start, default_end = self._get_default_dates()
            if self.start_date is None:
                self.start_date = default_start
            if self.end_date is None:
                self.end_date = default_end

        # 添加极端值警告
        if self.trailing_activation_base <= 0 or self.trailing_activation_base >= 1:
            print(f"[WARNING] trailing_activation_base={self.trailing_activation_base} 是极端值!")

    def _get_default_dates(self) -> Tuple[str, str]:
        """获取默认的回测日期范围"""
        end_date = datetime.datetime.now()
        start_date = end_date - datetime.timedelta(days=365 * 1)
        return start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")

    def validate(self) -> None:
        """验证参数合法性"""
        if self.initial_cash <= 0:
            raise ValueError("初始资金必须大于0")
        if self.commission < 0 or self.commission > 0.1:
            raise ValueError("手续费率必须在0-10%之间")
        if self.slippage < 0 or self.slippage > 0.1:
            raise ValueError("滑点率必须在0-10%之间")
         # 增加日期逻辑验证
        if self.start_date and self.end_date:
            start_dt = datetime.datetime.strptime(self.start_date, "%Y-%m-%d")
            end_dt = datetime.datetime.strptime(self.end_date, "%Y-%m-%d")
            if start_dt >= end_dt:
                raise ValueError("开始日期必须早于结束日期")

        try:
            datetime.datetime.strptime(self.start_date, "%Y-%m-%d")
            datetime.datetime.strptime(self.end_date, "%Y-%m-%d")
        except ValueError:
            raise ValueError("日期格式错误，应为YYYY-MM-DD")

class ParamAccessor:

    def get_param(obj, param_name, default_value=None):
        """简化版参数获取方法"""
        if param_name is None:
            raise ValueError("参数名不能为None！")

        # 优先从trading_params获取
        trading_params = None
        if hasattr(obj, 'trading_params') and obj.trading_params is not None:
            trading_params = obj.trading_params
        elif hasattr(obj, 'strategy') and hasattr(obj.strategy, 'trading_params'):
            trading_params = obj.strategy.trading_params

        if trading_params and hasattr(trading_params, param_name):
            return getattr(trading_params, param_name)

        # 如果找不到，抛出异常而不是返回默认值
        raise ValueError(f"参数'{param_name}'未找到！")

class TradeManager:
    """交易管理器：处理交易记录、统计和日志"""

    def __init__(self):
        """初始化"""
        # 交易记录
        #self.trading_params = trading_params  # 添加参数传递支持
        self.executed_trades = []
        self.trade_count = 0
        self.holding_days = 0

        # 日志设置
        self.debug_enabled = True

        # 初始化统计
        self._init_statistics()

    def set_strategy(self, strategy):
        """设置策略引用并获取参数"""
        self.strategy = strategy

    def _init_statistics(self):
        """初始化统计变量"""
        # 基础统计
        self.total_pnl = 0.0
        self.total_commission = 0.0
        self.winning_trades = 0
        self.losing_trades = 0

        # 收益统计
        self.profits_list = []  # 添加 profits_list 的初始化
        self.max_profit_trade = 0.0
        self.max_loss_trade = 0.0
        self.avg_profit = 0.0
        self.avg_loss = 0.0

        # 连续交易
        self.max_winning_streak = 0
        self.max_losing_streak = 0
        self.current_streak = 0

        # 持仓统计
        self.avg_holding_period = 0
        self.max_holding_period = 0
        self.min_holding_period = float('inf')

        # 日内统计
        self.daily_pnl = {}
        self.best_day = 0
        self.worst_day = 0

        # 投资组合
        self.portfolio_values = []
        self.peak_value = 0

        # 统计缓存
        self._stats_cache = {}
        self._last_update = 0

    def _update_statistics(self, trade: Dict) -> None:
        """更新交易统计

        参数:
            trade: 交易记录字典
        """
        try:
            # 确保日期字段是datetime类型
            entry_date = pd.to_datetime(trade.get('entry_date'))
            exit_date = pd.to_datetime(trade.get('exit_date'))

            if isinstance(entry_date, float):
                entry_date = datetime.datetime.fromordinal(int(entry_date)).date()
            if isinstance(exit_date, float):
                exit_date = datetime.datetime.fromordinal(int(exit_date)).date()

            # 1. 计算持仓天数
            holding_days = (exit_date - entry_date).days if entry_date and exit_date else 0
            self.holding_days += holding_days

            # 1. 更新基础统计
            if trade.get('pnl'):
                self.total_pnl += trade['pnl']
                self.profits_list.append(trade['pnl'])

                # 更新胜负次数
                if trade['pnl'] > 0:
                    self.winning_trades += 1
                    self.max_profit_trade = max(self.max_profit_trade, trade['pnl'])
                    # 更新连胜记录
                    self.current_streak = max(1, self.current_streak + 1)
                    self.max_winning_streak = max(self.max_winning_streak, self.current_streak)
                else:
                    self.losing_trades += 1
                    self.max_loss_trade = min(self.max_loss_trade, trade['pnl'])
                    # 更新连亏记录
                    self.current_streak = min(-1, self.current_streak - 1)
                    self.max_losing_streak = min(self.max_losing_streak, self.current_streak)

            # 2. 更新持仓周期统计
            if trade.get('entry_date') and trade.get('exit_date'):
                holding_days = (trade['exit_date'] - trade['entry_date']).days
                self.holding_days += holding_days

                # 更新平均持仓时间
                if self.trade_count > 0:
                    self.avg_holding_period = self.holding_days / self.trade_count

                # 更新最长最短持仓
                self.max_holding_period = max(self.max_holding_period, holding_days)
                self.min_holding_period = min(self.min_holding_period, holding_days)

            # 3. 更新日内统计
            trade_date = exit_date if exit_date else entry_date
            if trade_date:
                self.daily_pnl[trade_date] = self.daily_pnl.get(trade_date, 0) + trade.get('pnl', 0)


            # 更新日内最佳最差
            self.best_day = max(self.daily_pnl.values(), default=0)
            self.worst_day = min(self.daily_pnl.values(), default=0)

            # 4. 更新手续费统计
            if trade.get('commission'):
                self.total_commission += trade['commission']

            # 5. 清除缓存
            self._stats_cache.clear()
            self._last_update = len(self.executed_trades)

        except Exception as e:
            self.log(f"更新统计失败: {str(e)}")
            raise  # 让错误暴露出来，而不是静默返回0

    def add_trade(self, trade: Dict) -> None:
        try:
            # 严格验证关键字段
            entry_price = trade.get('entry_price', 0)
            size = trade.get('size', 0)

            # 数据有效性检查
            if trade.get('type') == 'entry':
                if trade.get('size', 0) <= 0 and trade.get('orig_size', 0) <= 0:
                    return
                if trade.get('entry_price', 0) <= 1.0:
                    return

            # 创建深拷贝避免引用问题
            import copy
            trade_copy = copy.deepcopy(trade)

            # 确保size为正数
            trade_copy['size'] = abs(size)
            if 'orig_size' not in trade_copy:
                trade_copy['orig_size'] = abs(size)

            # 验证必需字段
            required_fields = ['entry_date', 'entry_price', 'size', 'commission']
            if trade_copy.get('status') == 'closed':
                required_fields.extend(['exit_date', 'exit_price', 'pnl'])

            missing_fields = [field for field in required_fields if field not in trade_copy]
            if missing_fields:
                if self.strategy:
                    self.strategy.log(f"交易字段不完整 (缺少: {', '.join(missing_fields)})", level="WARNING")
                return

            # 确保日期字段正确转换
            try:
                if trade_copy.get('entry_date') is not None:
                    trade_copy['entry_date'] = pd.to_datetime(trade_copy['entry_date']).date()
                if 'exit_date' in trade_copy and trade_copy['exit_date'] is not None:
                    trade_copy['exit_date'] = pd.to_datetime(trade_copy['exit_date']).date()
            except Exception as date_err:
                # 尝试修复日期问题
                if isinstance(trade_copy.get('entry_date'), float):
                    try:
                        trade_copy['entry_date'] = datetime.datetime.fromordinal(int(trade_copy['entry_date'])).date()
                    except:
                        return
                if isinstance(trade_copy.get('exit_date'), float):
                    try:
                        trade_copy['exit_date'] = datetime.datetime.fromordinal(int(trade_copy['exit_date'])).date()
                    except:
                        pass

            # 确保数字字段为数字类型
            numeric_fields = ['entry_price', 'size', 'commission']
            if trade_copy.get('status') == 'closed':
                numeric_fields.extend(['exit_price', 'pnl'])

            for field in numeric_fields:
                if field in trade_copy:
                    if trade_copy[field] is None:
                        trade_copy[field] = 0.0
                    elif not isinstance(trade_copy[field], (int, float)):
                        try:
                            trade_copy[field] = float(trade_copy[field])
                        except (ValueError, TypeError):
                            trade_copy[field] = 0.0

            # 添加缺少的字段
            if 'type' not in trade_copy:
                trade_copy['type'] = 'entry'

            # === 优化：删除详细的交易记录日志，避免与notify_order重复 ===
            # 不再输出详细的交易记录，因为notify_order已经处理了

            # 添加到交易列表并更新计数
            self.executed_trades.append(trade_copy)
            self.trade_count += 1

            # 更新统计数据
            self._update_statistics(trade_copy)

        except Exception as e:
           if self.strategy:
               self.strategy.log(f"添加交易记录时出现错误: {str(e)}", level='ERROR')
           else:
               print(f"添加交易记录时出现错误: {str(e)}")
           raise

    def update_portfolio_value(self, value: float) -> None:
        """更新投资组合价值"""
        self.portfolio_values.append(value)
        self.peak_value = max(self.peak_value, value)

    def _calculate_max_drawdown(self) -> float:
        """计算最大回撤"""
        if not self.portfolio_values:
            return 0.0

        values = np.array(self.portfolio_values)
        peak = np.maximum.accumulate(values)
        drawdown = (values - peak) / peak

        return abs(drawdown.min()) if len(drawdown) > 0 else 0

class MarketState:
    """市场状态管理类"""

    def __init__(self, params):
        """初始化市场状态管理器"""
        self.trading_params = params
        self.strategy = None  # 将在之后设置
        self.indicators = None  # 将在之后设置

        # TD序列变量
        self.td_up = 0
        self.td_down = 0
        self.td_high = 0
        self.td_low = float('inf')

        # 缓存
        self._market_regime = None
        self._trend_strength = None
        self._last_update = 0
        self._last_trend_update = 0  # 添加此行跟踪趋势强度更新
        self._prev_regime = "sideways"  # 默认值
        self._regime_change_counter = 0
        self.is_crash_mode = False  # 新增标志位
        # 市场状态变量
        self._initialize_market_vars()

    def _initialize_market_vars(self):
        """初始化市场状态变量"""
        self.market_score = 0
        self.trend_score = 0
        self.vol_score = 0
        self.mom_score = 0
        self.last_market_regime = None
        self.regime_duration = 0
        # 添加额外属性确保安全
        self.buy_signal = False

    def set_strategy(self, strategy):
        """设置策略引用"""
        self.strategy = strategy

        # 必须！从策略中获取trading_params
        if hasattr(strategy, 'trading_params') and strategy.trading_params is not None:
            self.trading_params = strategy.trading_params
            print(f"{self.__class__.__name__} 成功获取trading_params引用")
        else:
            # 记录警告但不报错，让问题在后续使用时暴露
            print(f"警告: {self.__class__.__name__} 无法获取trading_params")
        # 确保初始化后立即从策略中获取参数，而不是使用硬编码值
        # 动态计算最小交易规模
        portfolio_value = self.strategy.broker.getvalue()
        self.max_position_value = portfolio_value * 0.95

        # 获取最小持仓规模参数（无默认值）
        min_position_size = ParamAccessor.get_param(self, 'min_position_size')  # 修改: 移除默认25
        self.min_position_size = max(15, int(min_position_size))
        # 获取批次限制参数（无默认值）
        self.batches_allowed = ParamAccessor.get_param(self, 'batches_allowed')  # 修改: 移除默认6/4
        # 卖出冷却期参数（无默认值）
        self._sell_cooldown = ParamAccessor.get_param(self, 'sell_cooldown')     # 修改: 移除默认3

        self.strategy.log(f"初始化仓位管理 - 投资组合价值: {portfolio_value:.2f}, 最小交易规模: {self.min_position_size}", level="INFO")

    def set_indicators(self, indicators):
        """设置技术指标引用"""
        self.indicators = indicators

    def get_trend_strength(self, force_recalculate=False):
        # 添加参数传递调试
        if hasattr(self, 'trading_params'):
            self.strategy.log(f"Trend strength - 直接访问td_count: {getattr(self.trading_params, 'td_count', 'not found')}")
        elif hasattr(self.strategy, 'trading_params'):
            self.strategy.log(f"Trend strength - 通过strategy访问td_count: {getattr(self.strategy.trading_params, 'td_count', 'not found')}")
        else:
            self.strategy.log("Trend strength - 无法访问trading_params")


        old_trend_strength = getattr(self, '_trend_strength', None)
        current_bar = len(self.strategy.data)

        # 在以下情况下重新计算趋势强度:
        # 1. 强制重新计算
        # 2. 趋势强度尚未计算
        # 3. 当前K线与上次计算时的K线不同
        if force_recalculate or self._trend_strength is None or getattr(self, '_last_trend_update', 0) != current_bar:
            self._trend_strength = self._calculate_trend_strength()
            self._last_trend_update = current_bar

        # 输出趋势强度变化日志，帮助追踪参数影响
        if old_trend_strength != self._trend_strength and old_trend_strength is not None:
            if abs(self._trend_strength - old_trend_strength) > 8.0:  # 只显示大幅变化
                self.strategy.log(f"趋势强度变化: {old_trend_strength:.2f} → {self._trend_strength:.2f}", level="INFO")

        return self._trend_strength

    # 位置：在 MarketState 类中，替换 _calculate_trend_strength 方法
    # 上下文：位于 get_trend_strength 之后
    def _calculate_trend_strength(self):
        """计算趋势强度 - 模块化版本"""
        try:
            if self.strategy is None or len(self.strategy.data.close) < 3:
                return 0.0

            close_data = self.strategy.data.close.get(size=10)
            etf_bias = ParamAccessor.get_param(self, 'etf_trend_bias')
            price_score = etf_bias
            momentum_score = 1.0 + (etf_bias * 0.5)

            # 子函数：收益率分析
            def calculate_returns_score():
                score = 0.0
                returns = {}
                for period in [1, 3, 5, 10, 20]:
                    if period < len(close_data):
                        returns[period] = close_data[-1] / close_data[-period] - 1

                trend_price_up_score = ParamAccessor.get_param(self, 'trend_price_up_score')
                if returns.get(1, 0) > -0.01:
                    score += trend_price_up_score * 0.8
                elif returns.get(3, 0) > 0:
                    score += trend_price_up_score * 0.6

                if len(returns) >= 3:
                    trend_price_accel_score = ParamAccessor.get_param(self, 'trend_price_accel_score')
                    trend_price_decel_penalty = ParamAccessor.get_param(self, 'trend_price_decel_penalty')
                    if returns.get(1, 0) > returns.get(3, 0):
                        score += trend_price_accel_score
                    elif returns.get(1, 0) < returns.get(3, 0) - 0.02:
                        score -= trend_price_decel_penalty

                trend_long_up_score = ParamAccessor.get_param(self, 'trend_long_up_score')
                trend_long_down_penalty = ParamAccessor.get_param(self, 'trend_long_down_penalty')
                if returns.get(10, 0) > 0.05:
                    score += trend_long_up_score
                elif returns.get(20, 0) < -0.05:
                    score -= trend_long_down_penalty
                return score

            # 子函数：均线关系
            def calculate_ma_score():
                score = 0.0
                current_price = close_data[-1]
                trend_price_above_fast_score = ParamAccessor.get_param(self, 'trend_price_above_fast_score')
                trend_price_below_fast_penalty = ParamAccessor.get_param(self, 'trend_price_below_fast_penalty')
                trend_price_above_slow_score = ParamAccessor.get_param(self, 'trend_price_above_slow_score')
                trend_price_below_slow_penalty = ParamAccessor.get_param(self, 'trend_price_below_slow_penalty')

                if current_price > self.strategy.ma_fast[0]:
                    score += trend_price_above_fast_score
                elif current_price < self.strategy.ma_fast[0] * 0.95:
                    score -= trend_price_below_fast_penalty

                if current_price > self.strategy.ma_slow[0]:
                    score += trend_price_above_slow_score
                elif current_price < self.strategy.ma_slow[0] * 0.95:
                    score -= trend_price_below_slow_penalty
                return score

            # 子函数：均线交叉
            def calculate_ma_cross_score():
                score = 0.0
                if len(self.strategy.ma_fast) > 2 and len(self.strategy.ma_slow) > 2:
                    trend_ma_bull_score = ParamAccessor.get_param(self, 'trend_ma_bull_score')
                    trend_ma_bear_penalty = ParamAccessor.get_param(self, 'trend_ma_bear_penalty')
                    trend_ma_golden_cross_score = ParamAccessor.get_param(self, 'trend_ma_golden_cross_score')
                    trend_ma_death_cross_penalty = ParamAccessor.get_param(self, 'trend_ma_death_cross_penalty')

                    if (self.strategy.ma_fast[0] > self.strategy.ma_mid[0] and
                        self.strategy.ma_mid[0] > self.strategy.ma_slow[0]):
                        score += trend_ma_bull_score
                    elif (self.strategy.ma_fast[0] < self.strategy.ma_mid[0] and
                          self.strategy.ma_mid[0] < self.strategy.ma_slow[0]):
                        score -= trend_ma_bear_penalty

                    if (self.strategy.ma_fast[-2] <= self.strategy.ma_slow[-2] and
                        self.strategy.ma_fast[-1] > self.strategy.ma_slow[-1]):
                        score += trend_ma_golden_cross_score
                    elif (self.strategy.ma_fast[-2] >= self.strategy.ma_slow[-2] and
                          self.strategy.ma_fast[-1] < self.strategy.ma_slow[-1]):
                        score -= trend_ma_death_cross_penalty
                return score

            # 长期趋势
            long_term_trend_up = self.strategy.ma_long[0] > self.strategy.ma_long[-1]
            if long_term_trend_up:
                price_score += ParamAccessor.get_param(self, 'trend_long_up_score')
            else:
                price_score -= ParamAccessor.get_param(self, 'trend_long_down_penalty')
                price_score = min(price_score, ParamAccessor.get_param(self, 'sideways_threshold'))

            # 整合得分
            price_score += calculate_returns_score() + calculate_ma_score() + calculate_ma_cross_score()
            normalized_score = max(0, min(20, price_score + momentum_score))

            if ParamAccessor.get_param(self, 'debug_mode'):
                self.strategy.log(
                    f"趋势计算 - 价格评分:{price_score:.1f}, 动量评分:{momentum_score:.1f}, 最终:{normalized_score:.1f}",
                    level="INFO"
                )
            return normalized_score

        except Exception as e:
            self.strategy.log(f"趋势强度计算错误: {str(e)}")
            return 1.0

    def update_td_sequence(self):
        """改进TD序列更新，确保参数直接影响行为"""
        try:
            td_count = ParamAccessor.get_param(self, 'td_count')
            if not self.strategy.position and td_count > 0:
                # 空仓时，将td_count要求减半（但至少为2）
                effective_td_count = max(2, td_count // 2)
            else:
                effective_td_count = td_count

            current_price = self.strategy.data.close[0]
            prev_price = self.strategy.data.close[-4]

            # 更新TD计数
            if current_price > prev_price:
                self.td_up += 1
                self.td_down = 0
                self.td_high = max(self.td_high, current_price)

                # 获取参数 - 使用ParamAccessor确保正确获取
                td_count = ParamAccessor.get_param(self, 'td_count')
                td_check_days = ParamAccessor.get_param(self, 'td_check_days')

                # 检查均线条件 - 直接使用td_check_days参数
                ma_check_list = []
                for i in range(td_check_days):  # 这里直接使用td_check_days
                    if i < len(self.strategy.data.close) and i < len(self.strategy.ma_slow):
                        check = self.strategy.data.close[-i] > self.strategy.ma_slow[-i]
                        ma_check_list.append(check)

                ma_check = all(ma_check_list) if ma_check_list else False

                # === 核心修改：直接设置买入信号，确保td_count直接影响买入决策 ===
                td_require_regime = ParamAccessor.get_param(self, 'td_require_regime')
                market_regime = self.get_market_regime()
                trend_strength = self.get_trend_strength()

                # 计算回调幅度
                dip_base_pct = ParamAccessor.get_param(self, 'dip_base_pct')
                current_price = self.strategy.data.close[0]
                dip_from_high = (self.td_high - current_price) / self.td_high if self.td_high > 0 else 0

                # 只有当td_up大于等于td_count时才考虑生成买入信号
                if self.td_up >= td_count and dip_from_high >= dip_base_pct:
                    # 检查市场状态约束
                    valid_regimes = ["uptrend", "strong_uptrend", "sideways"] if td_require_regime else ["uptrend", "strong_uptrend", "sideways", "downtrend"]
                    regime_ok = market_regime in valid_regimes

                    # 获取市场状态阈值参数
                    strong_uptrend_threshold = ParamAccessor.get_param(self, 'strong_uptrend_threshold')
                    uptrend_threshold = ParamAccessor.get_param(self, 'uptrend_threshold')
                    sideways_threshold = ParamAccessor.get_param(self, 'sideways_threshold')
                    downtrend_threshold = ParamAccessor.get_param(self, 'downtrend_threshold')

                    # 根据市场状态动态调整趋势要求
                    if market_regime == "strong_uptrend":
                        trend_required = strong_uptrend_threshold * 0.5
                    elif market_regime == "uptrend":
                        trend_required = uptrend_threshold * 0.8
                    elif market_regime == "sideways":
                        trend_required = sideways_threshold
                    else:  # downtrend
                        trend_required = abs(downtrend_threshold) * 0.8

                    trend_ok = trend_strength > trend_required

                    # 设置买入信号强度（供get_buy_signal使用）
                    if (not td_require_regime or regime_ok) and trend_ok:
                        # TD信号强度根据超过td_count的程度来设置
                        if self.td_up >= td_count * 2:
                            self.td_signal_strength = 2.0  # 强信号
                        elif self.td_up >= td_count * 1.5:
                            self.td_signal_strength = 1.5  # 中等信号
                        else:
                            self.td_signal_strength = 1.0  # 基础信号

                        self.buy_signal = True
                        self.strategy.log(f"TD序列买入信号生成 - 计数:{self.td_up}/{td_count}, 强度:{self.td_signal_strength}, 市场:{market_regime}", level="INFO")
                    else:
                        self.td_signal_strength = 0.5  # 弱信号
                        self.buy_signal = False
                else:
                    self.td_signal_strength = 0.0
                    self.buy_signal = False

            elif current_price < prev_price:
                self.td_down += 1
                self.td_up = 0
                self.td_low = min(self.td_low, current_price)
                self.td_signal_strength = 0.0
                self.buy_signal = False

        except Exception as e:
            self.strategy.log(f"TD序列更新错误: {str(e)}")
            self.buy_signal = False
            self.td_signal_strength = 0.0
            raise

    def reset_td_sequence(self):
        """重置TD序列状态"""
        self.td_up = 0
        self.td_down = 0
        self.td_high = 0
        self.td_low = float('inf')
        self.buy_signal = False
        self.td_signal_strength = 0.0
        self.strategy.log("TD序列状态已重置", level="DEBUG")

    def get_buy_signal(self) -> bool:
        """统一的买入信号检测 - 简化版本，复用趋势计算"""
        try:
            # 基础检查
            if self.strategy is None:
                return False

            # 检查是否已在当前bar更新过TD序列
            current_bar = len(self.strategy)
            if not hasattr(self, '_last_td_update_bar') or self._last_td_update_bar != current_bar:
                self.update_td_sequence()
                self._last_td_update_bar = current_bar

            # 获取关键参数
            td_count = ParamAccessor.get_param(self, 'td_count')
            signal_threshold = ParamAccessor.get_param(self, 'signal_threshold')
            td_sequence_weight = ParamAccessor.get_param(self, 'td_sequence_weight')
            trend_weight = ParamAccessor.get_param(self, 'trend_weight')
            momentum_weight = ParamAccessor.get_param(self, 'momentum_weight')
            pattern_weight = ParamAccessor.get_param(self, 'pattern_weight')

            # 获取市场状态和趋势强度
            market_regime = self.get_market_regime()
            trend_strength = self.get_trend_strength()

            # 核心检查：TD序列
            if td_count > 0 and self.strategy.position:
                if not getattr(self, 'buy_signal', False):
                    return False

            # 早期退出：极端下跌市场
            if market_regime == "downtrend" and trend_strength < -5:
                self.strategy.log(
                    f"强下跌趋势拒绝买入 - 市场:{market_regime}, 趋势:{trend_strength:.1f}",
                    level="INFO"
                )
                return False

            # 近期大幅下跌检查
            if len(self.strategy.data.close) >= 10:
                drop_5d = (self.strategy.data.close[0] / self.strategy.data.close[-5] - 1)
                drop_10d = (self.strategy.data.close[0] / self.strategy.data.close[-10] - 1)
                if drop_5d < -0.08 or drop_10d < -0.12:
                    volume_confirm = False
                    if hasattr(self.strategy, 'volume_ma') and self.strategy.volume_ma[0] > 0:
                        volume_ratio = self.strategy.data.volume[0] / self.strategy.volume_ma[0]
                        volume_confirm = 1.2 < volume_ratio < 2.5
                    if not volume_confirm:
                        self.strategy.log("大跌但成交量不足，暂不买入", level="INFO")
                        return False

            # 信号计算
            td_signal = getattr(self, 'td_signal_strength', 0.0) * td_sequence_weight
            trend_signal = min(trend_strength / 10, 1.0) * trend_weight  # 归一化趋势信号
            momentum_signal = 0.0
            pattern_signal = 0.0
            volume_signal = 0.0

            # 动量信号
            try:
                if hasattr(self.strategy, 'macd') and hasattr(self.strategy, 'rsi'):
                    macd_bullish = self.strategy.macd.macd[0] > self.strategy.macd.signal[0]
                    rsi_bullish = 30 < self.strategy.rsi[0] < 70
                    momentum_signal = (1.0 if macd_bullish and rsi_bullish else
                                     0.7 if macd_bullish else
                                     0.5 if rsi_bullish else 0.0) * momentum_weight
            except:
                pass

            # 形态信号
            try:
                bullish_candle = self.strategy.data.close[0] > self.strategy.data.open[0]
                price_above_prev = self.strategy.data.close[0] > self.strategy.data.close[-1]
                pattern_signal = (1.0 if bullish_candle and price_above_prev else
                                 0.5 if bullish_candle or price_above_prev else 0.0) * pattern_weight
            except:
                pass

            # 成交量信号
            try:
                base_volume_mult = ParamAccessor.get_param(self, 'base_volume_mult')
                if self.strategy.volume_ma[0] > 0:
                    volume_ratio = self.strategy.data.volume[0] / self.strategy.volume_ma[0]
                    volume_signal = (1.0 if volume_ratio > (1.0 + base_volume_mult) else
                                    0.5 if volume_ratio > 1.0 else 0.0)
            except:
                pass

            # 市场状态加成
            market_bonus = {
                "strong_uptrend": 0.6 if trend_strength > 8 else 0.0,
                "uptrend": 0.5 if trend_strength > 4 else 0.0,
                "sideways": 0.2 if trend_strength > 0 else 0.0,
                "downtrend": -0.25
            }.get(market_regime, 0.0)

            # 综合信号
            total_signal = td_signal + trend_signal + momentum_signal + pattern_signal + volume_signal + market_bonus

            # 动态阈值
            threshold_mult = {"strong_uptrend": 0.5, "uptrend": 0.5, "sideways": 0.7}.get(market_regime, 1.0)
            effective_threshold = signal_threshold * threshold_mult

            result = total_signal > effective_threshold

            # 趋势追踪和空仓处理（保留原逻辑）
            if not result and market_regime in ["strong_uptrend", "uptrend"] and trend_strength > 10.0:
                if hasattr(self.strategy, 'volume_ma') and self.strategy.volume_ma[0] > 0:
                    volume_ratio = self.strategy.data.volume[0] / self.strategy.volume_ma[0]
                    if volume_ratio > 0.8:
                        result = True
                        self.strategy.log(f"趋势追踪买入 - 市场:{market_regime}, 趋势强度:{trend_strength:.1f}", level="INFO")

            if td_count > 0 and td_signal > 0.8 and trend_signal > 0.3:
                result = True
                self.strategy.log(f"TD信号兜底买入 - TD:{td_signal:.2f}, 趋势:{trend_signal:.2f}", level="INFO")

            if not self.strategy.position:
                bars_since_last_trade = len(self.strategy) - getattr(self.strategy, 'last_trade_bar', 0)
                if bars_since_last_trade > 30 and total_signal > effective_threshold * 0.5 and trend_strength > 0:  # 从60天改为30天，阈值从0.7改为0.5
                    result = True
                    self.strategy.log(f"空仓{bars_since_last_trade}天，降低买入要求", level="INFO")

            if result:
                self.strategy.log(
                    f"买入信号 - 总分:{total_signal:.2f} > 阈值{effective_threshold:.2f} "
                    f"(TD:{td_signal:.1f} 趋势:{trend_signal:.1f} 动量:{momentum_signal:.1f})",
                    level="INFO"
                )

            return result

        except Exception as e:
            self.strategy.log(f"买入信号生成错误: {str(e)}")
            return False

    def get_market_regime(self, force_update=False):
        """简化但保留核心功能的市场状态检测，包括崩盘检测"""
        if not self.strategy:
            self._market_regime = "sideways"
            self.is_crash_mode = False
            return "sideways"

        current_bar = len(self.strategy.data)
        if current_bar < 20:
            self._market_regime = "sideways"
            self.is_crash_mode = False
            self._last_update = current_bar
            return self._market_regime

        if not force_update and self._last_update == current_bar:
            return self._market_regime

        try:
            regime = "sideways"
            self.is_crash_mode = False
            close = self.strategy.data.close[0]
            ma_fast = self.strategy.ma_fast[0]
            ma_mid = self.strategy.ma_mid[0]
            ma_slow = self.strategy.ma_slow[0]

            if ma_fast == 0:
                self._market_regime = regime
                self.is_crash_mode = False
                self._last_update = current_bar
                return regime

            trend_strength = self.get_trend_strength(force_recalculate=True)
            strong_uptrend_threshold = ParamAccessor.get_param(self, 'strong_uptrend_threshold')
            uptrend_threshold = ParamAccessor.get_param(self, 'uptrend_threshold')
            sideways_threshold = ParamAccessor.get_param(self, 'sideways_threshold')
            downtrend_threshold = ParamAccessor.get_param(self, 'downtrend_threshold')

            # 崩盘检测
            if len(self.strategy.data.close) >= 20:
                price_5d = (close / self.strategy.data.close[-5] - 1)
                price_10d = (close / self.strategy.data.close[-10] - 1)
                volume_spike = False
                if hasattr(self.strategy, 'volume_ma') and self.strategy.volume_ma[0] > 0:
                    volume_ratio = self.strategy.data.volume[0] / self.strategy.volume_ma[0]
                    volume_spike = volume_ratio > 2.0
                if price_5d < -0.10 and price_10d < -0.15 and volume_spike:
                    regime = "crash"
                    self.is_crash_mode = True
                    self.strategy.log(
                        f"市场崩盘检测 - 5天跌{price_5d*100:.1f}%, 10天跌{price_10d*100:.1f}%, "
                        f"成交量放大{volume_ratio:.1f}倍",
                        level="CRITICAL"
                    )

            # 正常市场状态判断
            if regime != "crash":
                if trend_strength > strong_uptrend_threshold:
                    regime = "strong_uptrend"
                elif trend_strength > uptrend_threshold:
                    regime = "uptrend"
                elif trend_strength > sideways_threshold:
                    regime = "sideways"
                elif trend_strength <= downtrend_threshold:
                    regime = "downtrend"

            # 均线验证
            price_above_fast = close > ma_fast
            price_above_slow = close > ma_slow
            ma_fast_up = len(self.strategy.ma_fast) > 5 and self.strategy.ma_fast[0] > self.strategy.ma_fast[-5]
            if regime == "strong_uptrend" and (not price_above_fast or not ma_fast_up):
                regime = "uptrend"
            elif regime == "downtrend" and price_above_fast and price_above_slow and ma_fast_up:
                regime = "sideways"

            # ETF 特殊规则
            symbol = ParamAccessor.get_param(self, 'symbol')
            if symbol.upper() in ['QQQ', 'SPY', 'IWM', 'DIA'] and regime == "sideways" and trend_strength > 0 and price_above_slow:
                regime = "uptrend"

            # 状态切换确认
            regime_change_confirmation_days = ParamAccessor.get_param(self, 'regime_change_confirmation_days')
            extreme_change = (
                (self._prev_regime in ["strong_uptrend", "uptrend"] and regime in ["downtrend", "crash"]) or
                (self._prev_regime in ["downtrend", "crash"] and regime in ["strong_uptrend", "uptrend"])
            )
            if extreme_change:
                if self._prev_regime != regime:
                    self._regime_change_counter = 1
                    regime = self._prev_regime
                else:
                    self._regime_change_counter += 1
                    if self._regime_change_counter >= regime_change_confirmation_days:
                        self._prev_regime = regime
                        self._regime_change_counter = 0
                    else:
                        regime = self._prev_regime
            else:
                self._prev_regime = regime
                self._regime_change_counter = 0

            self._market_regime = regime
            self._last_update = current_bar
            return regime

        except Exception as e:
            self.strategy.log(f"市场状态错误: {str(e)}")
            self.is_crash_mode = False
            return "sideways"


    def is_market_recovering(self) -> bool:
        """判断市场是否正在从崩盘中恢复"""
        try:
            if len(self.strategy.data.close) < 20:
                return False

            # 检查是否有过崩盘
            if not hasattr(self.strategy, '_last_crash_bar'):
                return True  # 没有崩盘记录，认为正常

            bars_since_crash = len(self.strategy) - self.strategy._last_crash_bar
            if bars_since_crash < 5:
                return False  # 崩盘后5天内不认为恢复

            # 检查恢复迹象
            # 1. 最近5天的表现
            recent_5d_return = (self.strategy.data.close[0] / self.strategy.data.close[-5] - 1)

            # 2. 成交量恢复正常
            volume_ratio = self.strategy.data.volume[0] / self.strategy.volume_ma[0]
            volume_normal = 0.7 < volume_ratio < 1.5

            # 3. 价格稳定性
            recent_volatility = np.std([self.strategy.data.close[-i] for i in range(5)]) / np.mean([self.strategy.data.close[-i] for i in range(5)])

            # 4. 技术指标改善
            macd_improving = self.strategy.macd.macd[0] > self.strategy.macd.macd[-3]

            # 综合判断
            recovery_signals = [
                recent_5d_return > -0.02,  # 5天跌幅小于2%
                volume_normal,  # 成交量正常
                recent_volatility < 0.02,  # 波动率降低
                macd_improving,  # MACD改善
                bars_since_crash > 10  # 已经过了10天
            ]

            recovery_count = sum(recovery_signals)

            if recovery_count >= 3:
                self.strategy.log(
                    f"市场恢复迹象 - 满足{recovery_count}/5个条件，"
                    f"距离崩盘已{bars_since_crash}天",
                    level="INFO"
                )
                # 清除崩盘标记
                delattr(self.strategy, '_last_crash_bar')
                return True

            elif bars_since_crash > 15 and recovery_count >= 2:
                self.strategy.log(
                    f"市场部分恢复 - 满足{recovery_count}/5个条件，"
                    f"距离崩盘已{bars_since_crash}天，允许恢复交易",
                    level="INFO"
                )
                delattr(self.strategy, '_last_crash_bar')
                return True

            return False

        except Exception as e:
            self.strategy.log(f"市场恢复检测错误: {str(e)}")
            return False

    def is_long_term_sideways(self, window: int = 60, range_threshold: float = 0.05) -> bool:
        """
        判断近期是否处于长期横盘状态

        参数：
            window: 用于判断的交易日数（默认60日）
            range_threshold: 价格波动相对幅度阈值，若最大与最小价格的差相对于最小价格低于该值，则认为横盘（默认5%）

        返回：
            True 表示横盘，False 表示非横盘
        """
        try:
            # 从数据中获取最近 window 个收盘价
            close_data = self.strategy.data.close.get(size=window)
            if len(close_data) < window // 2:  # 至少要有一半的数据
                # 数据不足时不认为是横盘状态
                return False

            max_price = max(close_data)
            min_price = min(close_data)

            # 计算相对价格范围
            rel_range = (max_price - min_price) / min_price

            self.strategy.log(
                f"长期横盘检测 - 最近{len(close_data)}日价格区间: {rel_range*100:.2f}%",
                level="INFO"
            )

            return rel_range < range_threshold

        except Exception as e:
            self.strategy.log(f"长期横盘检测错误: {str(e)}", level="INFO")
            raise

class RiskManager:
    """风险管理类"""

    def __init__(self, params):
        self.trading_params = params
        self.strategy = None  # 将在之后设置

        # 初始化基本变量，但推迟依赖于策略的初始化
        self.peak_value = 0  # 将在设置策略时更新
        self.daily_value = 0
        self.last_date = None
        self.current_drawdown = 0
        self.worst_drawdown = 0

        # 设置日内损失阈值（取负值）
        self.daily_loss_threshold = -abs(ParamAccessor.get_param(self, 'max_daily_loss'))

        # 用于防止同一天多次触发
        self.last_loss_day = None
        self.last_daily_loss = 0
        self.risk_score = 0
        self.consecutive_losses = 0
        self.last_trade_pnl = 0

        # 波动率跟踪
        self.vol_ma20 = 0
        self.vol_high = 0

        # 缓存
        self._cache = {}
        self._last_update = 0

        # 预初始化风险参数（不依赖 strategy）
        # 使用ParamAccessor确保使用设置的值
        self.max_drawdown = -abs(ParamAccessor.get_param(self, 'max_drawdown'))
        self.max_daily_loss = -abs(ParamAccessor.get_param(self, 'max_daily_loss'))
        self.max_position = ParamAccessor.get_param(self, 'max_position_pct')
        self._last_cleanup_bar = 0

    def set_indicators(self, indicators):
        """设置技术指标引用"""
        self.indicators = indicators

    def set_strategy(self, strategy):
        """设置策略引用"""
        self.strategy = strategy

        # 确保从策略获取trading_params
        # 必须！从策略中获取trading_params
        if hasattr(strategy, 'trading_params') and strategy.trading_params is not None:
            self.trading_params = strategy.trading_params
            print(f"{self.__class__.__name__} 成功获取trading_params引用")
            # 调试输出: 确认 profit_trigger_1 参数是否传入
            print(f"RiskManager: profit_trigger_1 = {getattr(self.trading_params, 'profit_trigger_1', None)}")  # 调试打印
        else:
            print(f"警告: {self.__class__.__name__} 无法获取trading_params")

        # 现在有了策略，更新初始值
        self.peak_value = self.strategy.broker.getvalue()
        self.daily_value = self.peak_value
        # 重新初始化依赖于策略的参数
        self._init_risk_params()

    # 修复后代码
    def _init_risk_params(self):
        """初始化风险参数"""

        # 使用原始参数值，不做乘法修改 - 使用ParamAccessor
        self.max_drawdown = -abs(ParamAccessor.get_param(self, 'max_drawdown'))
        self.max_daily_loss = -abs(ParamAccessor.get_param(self, 'max_daily_loss'))
        self.max_position = ParamAccessor.get_param(self, 'max_position_pct')

        # 设置不同市场状态下的交易风险参数
        trade_loss_pct = ParamAccessor.get_param(self, 'trade_loss_pct')
        self.max_trade_loss = {
            'strong_uptrend': -abs(trade_loss_pct) * 2.0,
            'uptrend': -abs(trade_loss_pct) * 1.5,
            'sideways': -abs(trade_loss_pct) * 1.0,
            'downtrend': -abs(trade_loss_pct) * 0.5
        }

        # 设置不同市场状态下的盈利目标参数
        profit_trigger_1 = ParamAccessor.get_param(self, 'profit_trigger_1')
        self.profit_target = {
            'strong_uptrend': profit_trigger_1 * 1.8,
            'uptrend': profit_trigger_1 * 1.5,
            'sideways': profit_trigger_1 * 1.2,
            'downtrend': profit_trigger_1 * 1.0
        }

        # 设置波动率参数
        volatility_window = ParamAccessor.get_param(self, 'volatility_window')
        self.vol_threshold = {
            'strong_uptrend': volatility_window / 10.0 * 1.5,
            'uptrend': volatility_window / 10.0 * 1.2,
            'sideways': volatility_window / 10.0 * 1.0 ,
            'downtrend': volatility_window / 10.0 * 0.8
        }

        # 设置风险评分阈值
        equity_risk = ParamAccessor.get_param(self, 'equity_risk')
        self.risk_thresholds = {
            'low': equity_risk * 100,  # 转换为0-100的范围
            'medium': equity_risk * 150,
            'high': equity_risk * 200
        }
        if hasattr(self.strategy, 'broker'):
            self.daily_value = self.strategy.broker.getvalue()

    def check_risk_limits(self, current_price):
        """检查是否达到风险限制，增强版 - 修复持续限制问题"""
        try:
            current_bar = len(self.strategy)
            if current_bar - self._last_cleanup_bar > 365:
                self._last_cleanup_bar = current_bar

                # 清理超过1年的风险限制
                if hasattr(self, '_drawdown_triggered') and self._drawdown_triggered:
                    days_in_limit = current_bar - getattr(self, '_drawdown_trigger_day', 0)
                    if days_in_limit > 365:
                        self._drawdown_triggered = False
                        self._drawdown_trigger_day = 0
                        self.strategy.log(f"自动清理超期风险限制 - 已限制{days_in_limit}天", level="CRITICAL")

                # 清理过期的亏损记录
                if hasattr(self.strategy, 'last_loss_day') and self.strategy.last_loss_day > 0:
                    days_since_loss = current_bar - self.strategy.last_loss_day
                    if days_since_loss > 365:
                        self.strategy.last_trade_pnl = 0
                        self.strategy.last_loss_day = 0
                        self.strategy.consecutive_losses = 0
                        self.strategy.log(f"自动清理超期亏损记录 - 已{days_since_loss}天", level="CRITICAL")

            if hasattr(self, '_last_risk_check_bar') and self._last_risk_check_bar == current_bar:
                return getattr(self, '_last_risk_check_result', False)
            self._last_risk_check_bar = current_bar
            # 获取当前账户价值和峰值
            portfolio_value = self.strategy.broker.getvalue()

            # 计算当前回撤
            current_drawdown = (portfolio_value / self.peak_value - 1)

            # 使用equity_risk参数动态调整风险容忍度
            equity_risk = ParamAccessor.get_param(self, 'equity_risk')
            dynamic_max_drawdown = self.max_drawdown * (1 + equity_risk)

            # ========== 1. 整体回撤检查 ==========
            if current_drawdown < dynamic_max_drawdown:
                # 检查是否有显著恢复
                if not hasattr(self, '_drawdown_triggered'):
                    self._drawdown_triggered = True
                    self._drawdown_trigger_day = len(self.strategy)
                    self.strategy.log(f"触发最大回撤保护: {current_drawdown:.2%}", level="INFO")
                    # 添加风险通知
                    if self.strategy.notifier and self.strategy.trading_params.enable_risk_notification:
                        self.strategy.notifier.send_message(
                            "风险警报 - 最大回撤",
                            f"当前回撤: {current_drawdown:.2%}\n触发保护机制"
                        )
                    return True
                else:
                    # 改进的恢复条件
                    recovery_threshold = dynamic_max_drawdown + 0.05  # 恢复到限制线上5%
                    days_since_trigger = len(self.strategy) - getattr(self, '_drawdown_trigger_day', len(self.strategy))

                    # 多重恢复条件：任一满足即可解除限制
                    recovery_conditions = [
                        current_drawdown > recovery_threshold,  # 回撤恢复到阈值以上
                        days_since_trigger > 30,  # 超过30天自动解除
                        portfolio_value > self.peak_value * 0.95,  # 恢复到峰值的95%
                        (current_drawdown > dynamic_max_drawdown * 0.8 and days_since_trigger > 15),  # 部分恢复+时间
                        # 新增：如果回撤已经大幅改善且超过60天，强制解除
                        (days_since_trigger > 60 and current_drawdown > dynamic_max_drawdown * 0.5)
                    ]

                    if any(recovery_conditions):
                        self._drawdown_triggered = False
                        self._drawdown_trigger_day = 0
                        self.strategy.log(
                            f"风险限制解除 - 当前回撤: {current_drawdown:.2%}, "
                            f"距离触发: {days_since_trigger}天, "
                            f"账户价值: ${portfolio_value:.2f}",
                            level="CRITICAL"
                        )
                        return False
                    return True
            else:
                # 回撤在限制之内，重置状态
                if hasattr(self, '_drawdown_triggered'):
                    self._drawdown_triggered = False
                    self._drawdown_trigger_day = 0

            # ========== 2. 日内损失检查 ==========
            if self.daily_value > 0:
                daily_loss = (portfolio_value / self.daily_value - 1)
                if daily_loss < self.max_daily_loss:
                    self.strategy.log(f"触发日内损失保护: {daily_loss:.2%}", level="CRITICAL")
                    return True

            # ========== 3. 单笔交易风险检查（改进版）==========
            current_day = len(self.strategy.data)

            # 确保属性存在
            if not hasattr(self.strategy, 'last_loss_day'):
                self.strategy.last_loss_day = 0
            if not hasattr(self.strategy, 'last_trade_pnl'):
                self.strategy.last_trade_pnl = 0

            # 只在有亏损记录时检查
            if self.strategy.last_trade_pnl < 0:
                days_since_loss = current_day - self.strategy.last_loss_day
                risk_pct_per_trade = ParamAccessor.get_param(self, 'risk_pct_per_trade')
                trade_risk_limit = -abs(risk_pct_per_trade) * portfolio_value

                # 极端亏损阈值（3倍正常风险）
                extreme_loss_threshold = trade_risk_limit * 3

                # 只在极端亏损且时间较短时限制
                if self.strategy.last_trade_pnl < extreme_loss_threshold and days_since_loss <= 5:
                    # 检查市场是否已经企稳
                    if hasattr(self.strategy, 'market_state'):
                        current_regime = self.strategy.market_state.get_market_regime()
                        current_trend = self.strategy.market_state.get_trend_strength()

                        # 如果市场明显转好，可以提前解除限制
                        if current_regime in ["uptrend", "strong_uptrend"] and current_trend > 5:
                            self.strategy.log(
                                f"市场转好（{current_regime}，趋势{current_trend:.1f}），"
                                f"解除亏损限制（亏损{self.strategy.last_trade_pnl:.2f}）",
                                level="INFO"
                            )
                            # 重置亏损记录
                            self.strategy.last_trade_pnl = 0
                            self.strategy.last_loss_day = 0
                            return False
                elif days_since_loss > 30:  # 超过30天强制解除
                    self.strategy.log(f"亏损限制超期，强制解除", level="CRITICAL")
                    self.strategy.last_trade_pnl = 0
                    self.strategy.last_loss_day = 0
                    return False

                    # 仍需限制
                    self.strategy.log(
                        f"极端亏损限制交易: {self.strategy.last_trade_pnl:.2f} < {extreme_loss_threshold:.2f} "
                        f"(剩余限制天数: {5 - days_since_loss})",
                        level="CRITICAL"
                    )
                    return True

            # 获取参数
            max_consecutive_losses = ParamAccessor.get_param(self, 'max_consecutive_losses')
            loss_cooldown_days = ParamAccessor.get_param(self, 'loss_cooldown_days')

            # 确保连续亏损机制正常工作
            if hasattr(self.strategy, 'consecutive_losses') and self.strategy.consecutive_losses >= max_consecutive_losses:
                if hasattr(self.strategy, 'last_loss_day') and self.strategy.last_loss_day > 0:
                    days_since_last_loss = len(self.strategy) - self.strategy.last_loss_day

                    if days_since_last_loss < loss_cooldown_days:
                        self.strategy.log(
                            f"连续亏损保护 - 已连续亏损{self.strategy.consecutive_losses}次，"
                            f"冷静期还剩{loss_cooldown_days - days_since_last_loss}天",
                            level="CRITICAL"
                        )
                        return True
                    else:
                        # 冷静期结束，重置连续亏损计数
                        self.strategy.consecutive_losses = 0
                        self.strategy.log("连续亏损冷静期结束，恢复交易", level="INFO")

            # ========== 4. 额外的安全检查 ==========
            # 检查账户是否接近清零
            if portfolio_value < self.strategy.broker.startingcash * 0.2:
                self.strategy.log(
                    f"账户价值过低保护: ${portfolio_value:.2f} < 初始资金的20%",
                    level="CRITICAL"
                )
                return True

            return False
        except Exception as e:
            self.strategy.log(f"风险检查错误: {str(e)}")
            # 出错时默认不限制交易，避免因错误导致无法交易
            return False

    def update_tracking(self, current_price, current_value):
        """统一更新所有追踪数据 - 简化版"""
        try:
            current_date = self.strategy.data.datetime.date(0)

            # 1. 更新日内价格追踪
            if self.last_date != current_date:
            # 新的交易日
                self.daily_value = current_value
                self.last_date = current_date
            else:
                # 更新日内高低点
                self.daily_high = max(getattr(self, 'daily_high', current_price), current_price)
                self.daily_low = min(getattr(self, 'daily_low', current_price), current_price)

            # 2. 更新基础风险追踪（保留有用部分）
            self.peak_value = max(self.peak_value, current_value)
            drawdown = (current_value / self.peak_value - 1)
            self.worst_drawdown = min(self.worst_drawdown, drawdown)

            # 3. 更新波动率追踪（保留，因为可能在其他地方使用）
            if hasattr(self, 'indicators') and self.indicators:
                try:
                    current_vol = self.indicators.atr[0]
                    self.vol_ma20 = (getattr(self, 'vol_ma20', current_vol) * 19 + current_vol) / 20
                    self.vol_high = max(getattr(self, 'vol_high', current_vol), current_vol)
                except:
                    self.strategy.log(f"ATR获取失败，跳过波动率更新", level="WARNING")

        except Exception as e:
            self.strategy.log(f"Tracking update error: {str(e)}")
            raise

    def check_stops(self):
        """统一的止损检查"""
        return self.stop_manager.check_all_stops()

class StopManager:
    """停损管理器"""

    def __init__(self, params):
        self.trading_params = params
        self.strategy = None
        # 移除此处的ParamAccessor.get_param调用
        self.trailing_activated = False
        self.highest_profit = 0
        self.profit_target_hits = 0
        self.exit_signals = 0
        self.last_check_price = 0
        self.reversal_confirmed = False
        # 初始化vol_threshold但后续应该动态获取
        self.vol_threshold = {}  # 空字典，后续动态计算

    def set_strategy(self, strategy):
        """设置策略引用并重新初始化参数"""
        self.strategy = strategy
        # 确保从策略获取trading_params
        if hasattr(strategy, 'trading_params') and strategy.trading_params is not None:
            self.trading_params = strategy.trading_params
            print(f"{self.__class__.__name__} 成功获取trading_params引用")
            # 初始化vol_threshold
            self._init_thresholds()  # 新方法，见下
        else:
            print(f"警告: {self.__class__.__name__} 无法获取trading_params")

    def _init_thresholds(self):
        """
        初始化各种阈值设置，从交易参数中动态获取值而不是使用硬编码
        在 set_strategy 后调用，确保所有参数正确初始化
        """
        # 获取市场状态阈值参数
        strong_uptrend_threshold = ParamAccessor.get_param(self, 'strong_uptrend_threshold')
        uptrend_threshold = ParamAccessor.get_param(self, 'uptrend_threshold')
        sideways_threshold = ParamAccessor.get_param(self, 'sideways_threshold')
        downtrend_threshold = ParamAccessor.get_param(self, 'downtrend_threshold')

        # ===== 获取追踪止损相关参数 =====
        trailing_activation_base = ParamAccessor.get_param(self, 'trailing_activation_base')
        self.strategy.log(f"[DEBUG] Got trailing_activation_base = {trailing_activation_base}", level="CRITICAL")
        trailing_strong_uptrend_mult = ParamAccessor.get_param(self, 'trailing_strong_uptrend_mult')
        self.strategy.log(f"[DEBUG] Got trailing_strong_uptrend_mult = {trailing_strong_uptrend_mult}", level="CRITICAL")
        trailing_uptrend_mult = ParamAccessor.get_param(self, 'trailing_uptrend_mult')
        trailing_sideways_mult = ParamAccessor.get_param(self, 'trailing_sideways_mult')
        trailing_weak_trend_mult = ParamAccessor.get_param(self, 'trailing_weak_trend_mult')
        trailing_downtrend_mult = ParamAccessor.get_param(self, 'trailing_downtrend_mult')

        # 获取ETF相关参数
        strong_uptrend_mult = ParamAccessor.get_param(self, 'strong_uptrend_mult')
        uptrend_mult = ParamAccessor.get_param(self, 'uptrend_mult')
        sideways_mult = ParamAccessor.get_param(self, 'sideways_mult')
        downtrend_mult = ParamAccessor.get_param(self, 'downtrend_mult')

        # 设置波动率阈值 - 使用参数值而非硬编码
        self.vol_threshold = {
            'strong_uptrend': max(1.0, strong_uptrend_threshold * 0.1),
            'uptrend': max(0.8, uptrend_threshold * 0.15),
            'sideways': max(0.6, sideways_threshold * 0.2),
            'downtrend': max(0.4, abs(downtrend_threshold) * 0.25)
        }

        # 初始化追踪止损激活阈值
        self.trail_activation_threshold = {
            'strong_uptrend': trailing_activation_base * trailing_strong_uptrend_mult,
            'uptrend': trailing_activation_base * trailing_uptrend_mult,
            'sideways': trailing_activation_base * trailing_sideways_mult,
            'weak_trend': trailing_activation_base * trailing_weak_trend_mult,
            'downtrend': trailing_activation_base * trailing_downtrend_mult
        }

        # 记录初始化的阈值，帮助调试
        self.strategy.log(f"止损阈值初始化 - 波动率阈值: {self.vol_threshold}", level="INFO")
        self.strategy.log(f"追踪止损激活阈值: {self.trail_activation_threshold}", level="INFO")

        # ===== 关键调试输出：显示获取到的参数值 =====
        self.strategy.log(f"追踪止损参数详情 - activation_base: {trailing_activation_base}, "
                         f"strong_mult: {trailing_strong_uptrend_mult}, uptrend_mult: {trailing_uptrend_mult}, "
                         f"sideways_mult: {trailing_sideways_mult}, weak_mult: {trailing_weak_trend_mult}, "
                         f"down_mult: {trailing_downtrend_mult}", level="CRITICAL")

    def _check_basic_stop(self, current_price, trend_strength):
        """修复基础止损逻辑，强制使用参数原始值并启用长期持仓参数"""
        try:
            # 直接从trading_params获取参数，不使用默认值
            if not hasattr(self.strategy, 'trading_params') or self.strategy.trading_params is None:
                raise ValueError("基础止损: trading_params未设置")

            # 直接获取所有止损参数 - 使用ParamAccessor
            stop_atr = ParamAccessor.get_param(self, 'stop_atr')
            force_min_stop_pct = ParamAccessor.get_param(self, 'force_min_stop_pct')
            stop_long_days = ParamAccessor.get_param(self, 'stop_long_days')
            stop_long_days_mult = ParamAccessor.get_param(self, 'stop_long_days_mult')
            stop_min_level_mult = ParamAccessor.get_param(self, 'stop_min_level_mult')

            if not hasattr(self.strategy, 'entry_price'):
                return False

            # 获取市场状态
            days_in_position = self.strategy.get_holding_days()
            market_regime = self.strategy.market_state.get_market_regime()

            # 获取ATR
            atr = self.strategy.atr[0]

            effective_stop_atr = stop_atr
            effective_force_min_stop_pct = force_min_stop_pct

            # ===== 修复：正确使用stop_long_days_mult参数 =====
            if days_in_position >= stop_long_days:
                # 使用stop_long_days_mult参数调整止损
                effective_stop_atr = stop_atr
                effective_force_min_stop_pct = force_min_stop_pct * stop_long_days_mult

                self.strategy.log(
                    f"长期持仓止损调整 - 持仓{days_in_position}天 >= {stop_long_days}天阈值, "
                    f"ATR倍数: {stop_atr} → {effective_stop_atr:.2f}, "
                    f"止损比例: {force_min_stop_pct:.2%} → {effective_force_min_stop_pct:.2%}",
                    level="INFO"
                )

            if hasattr(self.strategy, 'entry_price') and self.strategy.entry_price > 0:
                current_pnl = (current_price / self.strategy.entry_price - 1)

                # 根据盈亏状态动态调整止损
                if current_pnl > 0.20:  # 盈利超过20%
                    # 大幅盈利，使用更紧的止损保护利润
                    effective_force_min_stop_pct = min(effective_force_min_stop_pct, 0.08)
                    effective_stop_atr = min(effective_stop_atr, stop_atr * 0.6)
                    self.strategy.log(
                        f"动态止损[高盈利保护] - 盈利{current_pnl*100:.1f}%, "
                        f"止损收紧至{effective_force_min_stop_pct*100:.1f}%",
                        level="INFO"
                    )
                elif current_pnl > 0.10:  # 盈利10-20%
                    # 中等盈利，适度收紧止损
                    effective_force_min_stop_pct = min(effective_force_min_stop_pct, 0.10)
                    effective_stop_atr = min(effective_stop_atr, stop_atr * 0.8)
                    self.strategy.log(
                        f"动态止损[中盈利保护] - 盈利{current_pnl*100:.1f}%, "
                        f"止损调整至{effective_force_min_stop_pct*100:.1f}%",
                        level="DEBUG"
                    )
                elif current_pnl > 0.05:  # 盈利5-10%
                    # 小幅盈利，保持原止损
                    pass
                elif current_pnl > -0.05:  # 小幅亏损或平手
                    # 给予更多空间等待反弹
                    effective_force_min_stop_pct = max(effective_force_min_stop_pct, 0.15)
                    effective_stop_atr = max(effective_stop_atr, stop_atr * 1.2)
                    self.strategy.log(
                        f"动态止损[震荡容忍] - 当前{current_pnl*100:.1f}%, "
                        f"止损放宽至{effective_force_min_stop_pct*100:.1f}%",
                        level="DEBUG"
                    )
                else:  # 亏损超过5%
                    # 已经亏损较多，避免扩大损失，维持正常止损
                    self.strategy.log(
                        f"动态止损[控制风险] - 亏损{-current_pnl*100:.1f}%, "
                        f"维持止损{effective_force_min_stop_pct*100:.1f}%",
                        level="DEBUG"
                    )

            # ===== 修复：移除硬编码的ETF倍数，使用参数化的方式 =====
            # 获取ETF相关参数
            strong_uptrend_mult = ParamAccessor.get_param(self, 'strong_uptrend_mult')
            uptrend_mult = ParamAccessor.get_param(self, 'uptrend_mult')
            sideways_mult = ParamAccessor.get_param(self, 'sideways_mult')
            downtrend_mult = ParamAccessor.get_param(self, 'downtrend_mult')

            # 根据市场状态选择倍数
            if market_regime == "strong_uptrend" and trend_strength > 8:
                etf_multiplier = strong_uptrend_mult
            elif market_regime == "uptrend":
                etf_multiplier = uptrend_mult
            elif market_regime == "sideways":
                etf_multiplier = sideways_mult
            else:
                etf_multiplier = downtrend_mult

            entry_based_stop = self.strategy.entry_price * (1 - effective_force_min_stop_pct)
            high_since_entry = getattr(self.strategy, 'high_since_entry', self.strategy.entry_price)

            # 计算ATR基础止损
            atr_based_stop = high_since_entry - (atr * effective_stop_atr * etf_multiplier)

            # 计算最终ATR止损（不能低于最低止损水平）
            atr_stop = max(atr_based_stop, min_stop_level)

             # 使用两者中较高的作为止损位（给予更多空间）
            stop_level = max(entry_based_stop, atr_based_stop)

            # 确保止损位不会高于入场价（避免立即止损）
            entry_price_stop_mult = 0.94  # 从0.96改为0.94，给更多空间
            stop_level = min(stop_level, self.strategy.entry_price * entry_price_stop_mult)

            # 确保止损不会太远
            max_stop_price = self.strategy.entry_price * (1 - min(effective_force_min_stop_pct * 1.5, 0.15))

            # 计算最终止损价格
            final_stop = max(atr_stop, stop_level, max_stop_price, 0.01)

            # 只在真正触发止损时输出
            if current_price < stop_level:
                drop_pct = (self.strategy.entry_price - current_price) / self.strategy.entry_price
                self.strategy.log(
                    f"基础止损触发 - 当前价{current_price:.2f} < 止损价{stop_level:.2f}, "
                    f"从入场价跌幅: {drop_pct:.1%}",
                    level="INFO"
                )
                return True

            # 只在真正触发止损时输出
            if current_price < final_stop:
                self.strategy.log(f"基础止损触发 - 当前价{current_price:.2f} < 止损价{final_stop:.2f}", level="INFO")
                return True

            # 返回止损决策
            return current_price < final_stop

        except Exception as e:
            # 记录错误并抛出，不使用默认值
            self.strategy.log(f"基础止损计算错误: {str(e)}")
            raise  # 让错误暴露出来

    def _check_trailing_stop(self, current_price, market_regime):
        """修复的追踪止损检查，只使用TradingParameters中实际存在的参数"""
        try:
            # 1. 首先确保trading_params可用
            if not hasattr(self, 'trading_params') or self.trading_params is None:
                raise ValueError("追踪止损: trading_params未设置")

            # 2. 立即获取所有相关参数（在任何条件检查之前）
            trail_stop_mult = ParamAccessor.get_param(self, 'trail_stop_mult')
            trailing_activation_base = ParamAccessor.get_param(self, 'trailing_activation_base')
            trailing_stop_min_profit = ParamAccessor.get_param(self, 'trailing_stop_min_profit')
            trailing_stop_min_drop = ParamAccessor.get_param(self, 'trailing_stop_min_drop')
            trailing_stop_long_holding = ParamAccessor.get_param(self, 'trailing_stop_long_holding')
            trailing_stop_rapid_drop = ParamAccessor.get_param(self, 'trailing_stop_rapid_drop')
            trailing_stop_mult = ParamAccessor.get_param(self, 'trailing_stop_mult')
            trailing_stop_long_holding_factor = ParamAccessor.get_param(self, 'trailing_stop_long_holding_factor')
            trailing_stop_medium_holding_factor = ParamAccessor.get_param(self, 'trailing_stop_medium_holding_factor')

            # 修复：关键参数获取
            trailing_strong_uptrend_mult = ParamAccessor.get_param(self, 'trailing_strong_uptrend_mult')
            trailing_uptrend_mult = ParamAccessor.get_param(self, 'trailing_uptrend_mult')
            trailing_sideways_mult = ParamAccessor.get_param(self, 'trailing_sideways_mult')
            trailing_weak_trend_mult = ParamAccessor.get_param(self, 'trailing_weak_trend_mult')
            trailing_downtrend_mult = ParamAccessor.get_param(self, 'trailing_downtrend_mult')

            # 3. 基本条件检查（移到参数获取之后）
            if not self.strategy.position or not hasattr(self.strategy, 'entry_price') or self.strategy.entry_price == 0:
                # 即使退出，也要记录参数值以便调试
                self.strategy.log(f"追踪止损跳过 - 但参数已获取: activation_base={trailing_activation_base}, "
                                 f"strong_mult={trailing_strong_uptrend_mult}", level="INFO")
                return False

            # 4. 获取其他配置信息
            trend_strength = self.strategy.market_state.get_trend_strength()
            holding_days = self.strategy.get_holding_days()

             # 5. 计算当前盈利水平
            profit_pct = (current_price / self.strategy.entry_price) - 1

             # ===== 6. 修复激活逻辑，使用动态获取的参数 =====
            # ETF专用激活逻辑
            if not self.trailing_activated:
                # 根据市场状态设置激活阈值
                if market_regime == "strong_uptrend":
                    activation_threshold = trailing_activation_base * trailing_strong_uptrend_mult
                    # 添加详细日志
                    self.strategy.log(
                        f"追踪止损激活检查 - 当前盈利:{profit_pct:.2%}, "
                        f"激活阈值:{activation_threshold:.2%} "
                        f"(base={trailing_activation_base}, mult={trailing_strong_uptrend_mult})",
                        level="DEBUG"
                    )
                elif market_regime == "uptrend":
                    activation_threshold = trailing_activation_base * trailing_uptrend_mult
                elif market_regime == "sideways":
                    activation_threshold = trailing_activation_base * trailing_sideways_mult
                elif market_regime == "downtrend":
                    activation_threshold = trailing_activation_base * trailing_downtrend_mult
                else:
                    activation_threshold = trailing_activation_base * trailing_weak_trend_mult

                # 持仓时间影响
                orig_threshold = activation_threshold
                if holding_days > 60:  # 长期持仓，降低门槛
                    activation_threshold *= trailing_stop_long_holding_factor
                elif holding_days > 30:  # 中期持仓
                    activation_threshold *= trailing_stop_medium_holding_factor

                if orig_threshold != activation_threshold:
                    self.strategy.log(f"持仓时间调整激活阈值: {orig_threshold:.4f} -> {activation_threshold:.4f}", level="INFO")

                if profit_pct > activation_threshold and not self.trailing_activated:
                    self.trailing_activated = True
                    self.highest_profit = profit_pct
                    self.strategy.log(f"追踪止损已激活 - 当前盈利: {profit_pct:.2%}", level='INFO')
                    return False

                return False

            # 7. 已激活的追踪止损逻辑（保持原有代码不变）
            trailing_stop_trail_factor_high = ParamAccessor.get_param(self, 'trailing_stop_trail_factor_high')
            trailing_stop_trail_factor_medium = ParamAccessor.get_param(self, 'trailing_stop_trail_factor_medium')
            trailing_stop_trail_factor_low = ParamAccessor.get_param(self, 'trailing_stop_trail_factor_low')
            trailing_stop_trail_factor_min = ParamAccessor.get_param(self, 'trailing_stop_trail_factor_min')

            # 设置基础追踪参数 - 直接使用trail_stop_mult
            base_trail = trail_stop_mult * 0.4  # 确保参数有影响

            if market_regime == "strong_uptrend":
                trail_multiplier = trailing_strong_uptrend_mult
            elif market_regime == "uptrend":
                trail_multiplier = trailing_uptrend_mult
            elif market_regime == "sideways":
                trail_multiplier = trailing_sideways_mult
            elif market_regime == "downtrend":
                trail_multiplier = trailing_downtrend_mult
            else:
                trail_multiplier = trailing_weak_trend_mult

            orig_multiplier = trail_multiplier
            if trend_strength > 15.0:
                trail_multiplier *= 1.2
            elif trend_strength > 10.0:
                trail_multiplier *= 1.1
            elif trend_strength < 0:
                trail_multiplier *= 0.8

            # 追踪止损计算
            atr = self.strategy.atr[0]
            high_since_entry = getattr(self.strategy, 'high_since_entry', self.strategy.data.high[0])
            current_profit = (current_price / self.strategy.entry_price) - 1
            self.highest_profit = max(self.highest_profit, current_profit)

            # ETF专用追踪系数
            if profit_pct > 0.50:  # 从highest_profit改为profit_pct
                trail_stop_mult = trailing_stop_trail_factor_high * 1.2  # 增加20%容忍度
            elif profit_pct > 0.30:
                trail_stop_mult = trailing_stop_trail_factor_medium * 1.2
            elif self.highest_profit > 0.20:
                trail_stop_mult = trailing_stop_trail_factor_low
            else:
                trail_stop_mult = trailing_stop_trail_factor_min

            # 持仓时间调整追踪因子
            orig_factor = trail_stop_mult
            if holding_days > 180:  # 长期持仓
                trail_stop_mult *= trailing_stop_long_holding_factor
            elif holding_days > 90:
                trail_stop_mult *= trailing_stop_medium_holding_factor

            # 计算止损价格
            trail_stop = high_since_entry * (1 - trail_stop_mult)
            drop_from_peak = (high_since_entry - current_price) / high_since_entry
            # 设置触发条件
            min_profit_to_stop = trailing_stop_min_profit
            min_drop_threshold = trailing_stop_min_drop

            # 持仓时间影响
            if holding_days > trailing_stop_long_holding:
                min_profit_to_stop *= trailing_stop_long_holding_factor
                min_drop_threshold *= trailing_stop_long_holding_factor

            # 记录显著回撤
            if drop_from_peak > trailing_stop_rapid_drop * 1.5:
                self.strategy.log(
                    f"ETF追踪止损显著回撤 - 从高点回撤: {drop_from_peak:.2%}, 最高盈利: {self.highest_profit:.2%}, "
                    f"当前盈利: {profit_pct:.2%}, 持有天数: {holding_days}",
                    level='INFO'
                )

            # 止损触发条件判断
            price_below_stop = current_price < trail_stop
            still_profitable = profit_pct > min_profit_to_stop
            significant_drop = drop_from_peak > min_drop_threshold

            # 详细记录各个判断条件
            self.strategy.log(f"ETF追踪止损判断 - 价格低于止损位? {price_below_stop}, "
                             f"仍有盈利({profit_pct:.2%} > {min_profit_to_stop:.2%})? {still_profitable}, "
                             f"显著回撤({drop_from_peak:.2%} > {min_drop_threshold:.2%})? {significant_drop}", level="INFO")

            # 各种情况的止损触发
            if holding_days > trailing_stop_long_holding and price_below_stop and drop_from_peak > min_drop_threshold:
                self.strategy.log(f"ETF追踪止损触发 - 长期持仓且价格低于止损位")
                return True
            elif price_below_stop and still_profitable and significant_drop:
                self.strategy.log(f"ETF追踪止损触发 - 满足所有条件")
                return True
            elif drop_from_peak > min_drop_threshold * 1.67:
                self.strategy.log(f"ETF追踪止损触发 - 极端回撤超过25%")
                return True
            return False

        except Exception as e:
            self.strategy.log(f"追踪止损错误: {str(e)}")
            raise  # 让错误暴露出来

    def check_all_stops(self):
        """简化的止损检查"""
        if not self.strategy.position or not hasattr(self.strategy, 'entry_price'):
            return False

        try:
            current_price = self.strategy.data.close[0]

            # 按优先级检查，一旦触发立即返回
            if self._check_basic_stop(current_price, self.strategy.market_state.get_trend_strength()):
                self.strategy.log("基础止损触发", level="INFO")
                return True

            if self._check_trailing_stop(current_price, self.strategy.market_state.get_market_regime()):
                self.strategy.log("追踪止损触发", level="INFO")
                return True

            if self._check_profit_protection(current_price, self.strategy.market_state.get_market_regime()):
                self.strategy.log("获利保护触发", level="INFO")
                return True

            return False

        except Exception as e:
            self.strategy.log(f"止损检查错误: {str(e)}")
            return False

    def _check_profit_protection(self, current_price, market_regime):
        try:
            if not hasattr(self.strategy, 'entry_price'):
                return False

            profit_pct = (current_price / self.strategy.entry_price) - 1

            # 使用ParamAccessor获取触发参数，不传入默认值
            t1 = ParamAccessor.get_param(self, 'profit_trigger_1')
            t2 = ParamAccessor.get_param(self, 'profit_trigger_2')
            # 获取锁定比例参数
            profit_lock_pct = ParamAccessor.get_param(self, 'profit_lock_pct')

            triggers = [t1, t2]

            # 如果尚未达到下一个触发点，检查当前是否突破
            if self.profit_target_hits < len(triggers) and profit_pct >= triggers[self.profit_target_hits]:
                self.profit_target_hits += 1

            # 如果已至少命中一个目标，检查追踪止盈条件
            if self.profit_target_hits > 0:
                # 当前生效的触发阈值（已命中的最高一级）
                current_trigger = triggers[self.profit_target_hits - 1]

                # 仅当价格仍高于该触发阈值时才应用追踪止盈
                if profit_pct >= current_trigger:
                    # 关键修改: 使用profit_lock_pct参数，而不是硬编码的0.5
                    lock_in = current_trigger * profit_lock_pct
                    protect_stop_price = self.strategy.entry_price * (1.0 + lock_in)
                    # 只在真正触发时输出
                    if current_price < protect_stop_price:
                        self.strategy.log(f"获利保护止损触发 @ {current_price:.2f}", level="INFO")
                    return current_price < protect_stop_price
                else:
                    return False

            return False

        except Exception as e:
            self.strategy.log(f"获利保护检查错误: {str(e)}")
            raise

    #@track_method(category="止损方法")
    def _confirm_exit(self, reason):
        """确认退出信号"""
        try:
            # 记录本次信号确认
            self.exit_signals += 1

            # 使用ParamAccessor获取确认次数阈值
            confirm_threshold = ParamAccessor.get_param(self, 'exit_confirm_threshold')

            # 改为使用参数，默认为1
            if self.exit_signals >= confirm_threshold:
                self.strategy.log(f"确认{reason}信号 - 执行平仓")
                return True

            self.strategy.log(f"检测到{reason}信号 - 等待确认 ({self.exit_signals}/{confirm_threshold})", level='INFO')
            return False

        except Exception as e:
            self.strategy.log(f"Exit confirmation error: {str(e)}")
            raise

class PositionManager:
    """头寸管理器"""

    def __init__(self, params):
        self.trading_params = params
        self.strategy = None  # 将在之后设置
        self.market_state = None  # 将在之后设置
        self.risk_manager = None  # 将在之后设置
        self.last_add_bar = None

        # 头寸管理参数 - 使用成员变量存储，避免后续访问params出错
        # 注意：这些值会在set_strategy时被更新
        self.max_position_value = 0  # 将在设置策略时更新
        self.min_position_size = 25  # 默认值，将在设置策略时更新
        self._max_batches_logged = False
        self._last_sell_day = 0

        self.size_multipliers = {}  # 将在方法中动态获取

        self.did_time_exit = False

        # 跟踪部分加仓状态 - 不再硬编码batches_allowed
        self.added_batches = 0

        self._last_sell_day = 0
        self._sell_cooldown = 3  # 修改：卖出后3天内不再卖出(原为7天)

        # 缓存
        self._position_cache = {}
        self._last_update = 0
        self._last_add_price = 0  # 最后加仓价格
        self.last_time_profit_day = 0  # 时间周期获利
        self._executed_profit_levels = set()  # 确保初始化
        # 新增：追踪最后买入价格
        self._last_buy_price = None  # 添加这行
        self._buy_price_history = []  # 添加这行，记录所有买入价格历史

    def _clean_expired_buy_history(self):
        """清理过期的买入价格历史"""
        if not hasattr(self, '_buy_price_history') or not self._buy_price_history:
            return

        current_bar = len(self.strategy)
        # 保留最近365天内的买入记录
        self._buy_price_history = [
            record for record in self._buy_price_history
            if current_bar - record.get('bar', 0) <= 365
        ]

        # 如果所有记录都过期了，清空列表
        if not self._buy_price_history:
            self.strategy.log("清理所有过期买入记录", level="INFO")

    def set_strategy(self, strategy):
        """设置策略引用"""
        self.strategy = strategy
        # 确保从策略获取trading_params
        # 必须！从策略中获取trading_params
        if hasattr(strategy, 'trading_params') and strategy.trading_params is not None:
            self.trading_params = strategy.trading_params
            print(f"{self.__class__.__name__} 成功获取trading_params引用")
        else:
            # 记录警告但不报错，让问题在后续使用时暴露
            print(f"警告: {self.__class__.__name__} 无法获取trading_params")
        # 现在有了策略，调用初始化方法
        self._init_position_limits()

    def set_market_state(self, market_state):
        """设置市场状态引用"""
        self.market_state = market_state

    def set_risk_manager(self, risk_manager):
        """设置风险管理器引用"""
        self.risk_manager = risk_manager

    def _init_position_limits(self):
        """初始化头寸限制 - 使用参数原始值而不是检查有效性"""
        if self.strategy is None:
            return  # 如果策略未设置，直接返回

        # 基础设置
        self.max_position_value = self.strategy.broker.getvalue() * 0.95

        # 动态计算最小交易规模
        portfolio_value = self.strategy.broker.getvalue()

        # 使用ParamAccessor获取参数 - 不提供默认值
        position_portfolio_factor_base = ParamAccessor.get_param(self, 'position_portfolio_factor_base')
        position_min_size_factor = ParamAccessor.get_param(self, 'position_min_size_factor')

        # 计算投资组合因子
        portfolio_factor = max(position_min_size_factor, portfolio_value / position_portfolio_factor_base)

        # 增加最小持仓规模，并随投资组合增长
        # 使用ParamAccessor获取参数 - 不提供默认值
        min_position_size = ParamAccessor.get_param(self, 'min_position_size')

        # 直接使用参数，不检查是否为0、负数或9999
        self.min_position_size = max(10, int(min_position_size))  # 删除portfolio_factor的影响

        # 获取batches_allowed参数 - 使用ParamAccessor无默认值
        self.batches_allowed = ParamAccessor.get_param(self, 'batches_allowed')

        # 获取sell_cooldown参数 - 使用ParamAccessor无默认值
        self._sell_cooldown = ParamAccessor.get_param(self, 'sell_cooldown')

        # 记录关键参数
        self.strategy.log(
            f"初始化仓位管理 - 投资组合价值: {portfolio_value:.2f}, 最小交易规模: {self.min_position_size}, "
            f"批次限制: {self.batches_allowed}, 卖出冷却期: {self._sell_cooldown}",
            level="INFO"
        )

        # 更积极的规模乘数 - 完全依赖参数原始值
        # 使用ParamAccessor获取参数
        strong_uptrend_mult = ParamAccessor.get_param(self, 'strong_uptrend_mult')
        uptrend_mult = ParamAccessor.get_param(self, 'uptrend_mult')
        sideways_mult = ParamAccessor.get_param(self, 'sideways_mult')
        downtrend_mult = ParamAccessor.get_param(self, 'downtrend_mult')

        # 直接更新乘数字典，不使用默认值或检查有效性
        self.size_multipliers = {
            'strong_uptrend': strong_uptrend_mult,
            'uptrend': uptrend_mult,
            'sideways': sideways_mult,
            'downtrend': downtrend_mult
        }

    def _check_position_reset(self):
        """检查并重置仓位状态"""
        if not self.strategy.position:
            # 重置所有状态
            if self.added_batches > 0:
                self.strategy.log(f"平仓后重置批次计数: {self.added_batches} → 0", level="INFO")
            self.did_time_exit = False
            self.added_batches = 0
            self._last_sell_day = 0
            self._last_add_bar = None
            self._max_batches_logged = False
            self._executed_profit_levels = set()
            self._buy_price_history = []
            self._last_buy_price = None
            return True
        return False

    def _check_sell_cooldown(self, current_day, market_regime, trend_strength):
        """检查卖出冷却期
        返回: (in_cooldown, passive_mode)
        """
        in_sell_cooldown = False
        passive_mode = False

        sell_cooldown = ParamAccessor.get_param(self, 'sell_cooldown')

        if current_day - self._last_sell_day < sell_cooldown:
            cooldown_remaining = sell_cooldown - (current_day - self._last_sell_day)

            # 在强势上涨中缩短冷却期
            if market_regime == 'strong_uptrend' and trend_strength > 8 and cooldown_remaining <= 2:
                self.strategy.log(f"强势上涨 - 提前结束卖出冷却期", level="INFO")
                # 这里应该返回 False, False
                return False, False
            # 在下跌趋势中延长冷却期
            elif market_regime == 'downtrend' and trend_strength < 0:
                self.strategy.log(
                    f"卖出冷却期 ({current_day - self._last_sell_day}/{sell_cooldown} 天) - 下跌趋势中暂停减仓",
                    level="INFO"
                )
                in_sell_cooldown = True
                passive_mode = True
            else:
                self.strategy.log(
                    f"卖出冷却期 ({current_day - self._last_sell_day}/{sell_cooldown} 天)",
                    level="INFO"
                )
                in_sell_cooldown = True
                passive_mode = True

            return in_sell_cooldown, passive_mode

        # 如果不在冷却期，应该返回 False, False
        return False, False

    def _handle_small_position(self, current_position_size, min_meaningful_trade,
                              holding_days, trend_strength, profit_pct, current_day):
        abs_position_size = abs(current_position_size)
        """处理小仓位整合"""
        # 计算仓位占总资产比例
        portfolio_value = self.strategy.broker.getvalue()
        position_value = abs_position_size * self.strategy.data.close[0]
        position_ratio = position_value / portfolio_value

        # 修改：如果仓位占比小于10%，视为小仓位
        if position_ratio < 0.10 and abs_position_size > 0:
            # 修改1：大幅降低清理门槛
            if holding_days > 90:  # 从180天改为90天
                if profit_pct > 0.05:  # 从20%降低到5%
                    self.strategy.log(
                        f"清理小仓位 - 持仓({abs_position_size}股，占比{position_ratio:.1%})，"
                        f"持有{holding_days}天，盈利{profit_pct*100:.1f}%",
                        level="INFO"
                    )
                    return {
                        'action': 'sell',
                        'size': abs_position_size,
                        'reason': 'Small position cleanup'
                    }
                # 新增：持有超过180天，即使小幅亏损也清理
                elif holding_days > 180 and profit_pct > -0.05:  # 亏损不超过5%
                    self.strategy.log(
                        f"强制清理长期小仓位 - 持仓({abs_position_size}股)，"
                        f"持有{holding_days}天",
                        level="INFO"
                    )
                    return {
                        'action': 'sell',
                        'size': abs_position_size,
                        'reason': 'Force cleanup small position'
                    }
                # 新增：如果仓位极小（<5%），60天后就清理
                elif position_ratio < 0.05 and holding_days > 60:
                    self.strategy.log(
                        f"清理极小仓位 - 仅占{position_ratio:.1%}",
                        level="INFO"
                    )
                    return {
                        'action': 'sell',
                        'size': abs_position_size,
                        'reason': 'Micro position cleanup'
                    }
        return None

    def _check_position_concentration(self, position_ratio, portfolio_value,
                                     current_position_size, current_price,
                                     min_meaningful_trade, profit_pct, current_day):
        """检查仓位集中度风险"""
        if position_ratio > 0.7 and profit_pct > 0.05:
            self.strategy.log(f"风险控制 - 仓位过度集中({position_ratio:.1%})，执行减仓", level="INFO")

            # 减仓至最多60%
            target_ratio = 0.6
            target_value = portfolio_value * target_ratio
            target_size = int(target_value / current_price)
            reduce_size = current_position_size - target_size

            # 确保减仓量有意义
            reduce_size = max(reduce_size, min_meaningful_trade)
            reduce_size = min(reduce_size, current_position_size)

            if reduce_size >= min_meaningful_trade:
                self._last_sell_day = current_day
                return {
                    'action': 'sell',
                    'size': reduce_size,
                    'reason': 'Position concentration risk'
                }
        return None

    def _check_add_position(self, current_price, market_regime):
        """检查是否应该加仓"""
        # TD序列信号检查
        td_buy_signal = False
        if hasattr(self.market_state, 'buy_signal') and self.market_state.buy_signal:
            td_buy_signal = True
            self.strategy.log(f"TD序列触发的加仓信号检测", level="INFO")

        # 回调买入检查
        dip_signal = self._should_buy_the_dip(current_price)

        if td_buy_signal and not dip_signal:
            self.strategy.log(f"TD信号触发但回调条件不满足，暂不加仓", level="INFO")
            return None

        if dip_signal:
            if self.added_batches > 0:
                # 使用买入价格历史记录中的最后一个
                if hasattr(self, '_buy_price_history') and self._buy_price_history:
                    last_buy_info = self._buy_price_history[-1]
                    last_buy_price = last_buy_info['price']
                    price_change = (current_price - last_buy_price) / last_buy_price
                    min_price_drop = ParamAccessor.get_param(self, 'dip_per_batch')

                    # 修正：当前价格应该比上次买入价格低才能加仓
                    if current_price >= last_buy_price:
                        return False

                    # 检查跌幅是否足够
                    actual_drop = -price_change  # 转为正数表示跌幅
                    if actual_drop < min_price_drop:
                        return False
                    else:
                        self.strategy.log(
                            f"加仓价格满足 - 比上次低{actual_drop*100:.2f}%",
                            level="CRITICAL"
                        )

            if self.strategy.position and self.strategy.get_holding_days() > 90:
                current_pnl = (current_price / self.strategy.entry_price - 1)
                if current_pnl > 0.15 and market_regime in ["uptrend", "strong_uptrend"]:
                    # 检查是否有短期回调（不要求低于买入价）
                    short_term_high = max(self.strategy.data.high.get(size=10))
                    short_dip = (short_term_high - current_price) / short_term_high

                    if short_dip >= 0.02:  # 从10天高点回调2%即可
                        self.strategy.log(
                            f"ETF趋势加仓机会 - 持仓{self.strategy.get_holding_days()}天，"
                            f"盈利{current_pnl*100:.1f}%，短期回调{short_dip*100:.1f}%",
                            level="INFO"
                        )
                        # 允许加仓，绕过严格的价格检查
                        dip_signal = True  # 强制设置回调信号

            # 批次限制检查
            batches_allowed = ParamAccessor.get_param(self, 'batches_allowed')

            # 结合当前仓位比例进一步限制批次
            position_ratio = self._position_cache.get('position_pct', 0)
            if position_ratio > 0.7:  # 从0.5改为0.7
                batches_max = batches_allowed - 2  # 动态调整而不是固定值
            elif position_ratio > 0.5:  # 从0.3改为0.5
                batches_max = batches_allowed - 1
            else:
                batches_max = batches_allowed

            # 如果持仓时间很长且有较大回调，允许额外加仓
            if self.strategy.position and self.strategy.get_holding_days() > 60:
                recent_high = max(self.strategy.data.high.get(size=20))
                current_dip = (recent_high - current_price) / recent_high
                if current_dip > 0.10:  # 从高点回调超过10%
                    batches_max += 2  # 允许额外2次加仓
                    self.strategy.log(f"大幅回调，临时增加批次限制至{batches_max}", level="INFO")

            if self.added_batches >= batches_max:
                self.strategy.log(f"达到最大加仓次数: {self.added_batches}/{batches_max}", level="INFO")
                return None

            # 计算加仓规模
            add_size = self._calculate_position_size()

            if add_size == 0:
                self.strategy.log("跳过加仓 - 计算的加仓规模无效", level="INFO")
                return None

            # 确保加仓规模有意义
            portfolio_value = self.strategy.broker.getvalue()
            portfolio_factor = max(1.0, portfolio_value / 100000)

            min_meaningful_trade = max(15, int(15 * portfolio_factor))
            add_size = max(add_size, min_meaningful_trade)

            # 根据已加仓次数动态调整加仓规模
            position_batch_decay = ParamAccessor.get_param(self, 'position_batch_decay')
            batch_factor = max(0.5, 1.0 - (self.added_batches * position_batch_decay))
            add_size = int(add_size * batch_factor)

            if add_size >= min_meaningful_trade:
                return {
                    'action': 'buy',
                    'size': add_size,
                    'reason': f'{market_regime} meaningful dip buying'
                }

        return None

    def _check_profit_taking(self, profit_pct, passive_mode, min_meaningful_trade,
                            current_position_size, current_day):
        """检查获利了结条件"""
        # 即使在被动模式下，如果盈利很高，也允许止盈
        if not passive_mode or profit_pct > 0.15:
            take_profit_size = self._check_enhanced_profit_levels(profit_pct)

            self.strategy.log(f"获利检查结果: {take_profit_size}, 盈利率: {profit_pct:.2%}", level="INFO")

            if take_profit_size > 0:
                # 确保卖出规模有意义
                if take_profit_size < min_meaningful_trade:
                    if current_position_size <= min_meaningful_trade * 2:
                        take_profit_size = current_position_size
                    else:
                        take_profit_size = min_meaningful_trade

                # 确保卖出比例有意义 - 至少15%的持仓
                min_percent_size = int(current_position_size * 0.15)
                old_size = take_profit_size
                take_profit_size = max(take_profit_size, min_percent_size)

                # 确保不超过持仓上限
                old_size = take_profit_size
                take_profit_size = min(take_profit_size, current_position_size)

                if take_profit_size >= min_meaningful_trade:
                    self._last_sell_day = current_day
                    self._executed_profit_action = {
                        'action': 'sell',
                        'size': take_profit_size,
                        'reason': f'Take profit at {profit_pct:.1%}'
                    }
                    return self._executed_profit_action

        return None

    def _check_stop_loss(self, holding_days, trend_strength, market_regime,
                        profit_pct, current_position_size, min_meaningful_trade, current_day):
        """检查止损条件"""
        # 使用参数而不是硬编码，并提前触发
        trade_loss_pct = ParamAccessor.get_param(self, 'trade_loss_pct')  # 0.15
        if (trend_strength < -2 and market_regime in ["downtrend"]) or profit_pct < -trade_loss_pct * 0.8:
            # 根据持有天数调整止损阈值
            if holding_days > 90:
                loss_threshold = -0.20
            elif holding_days > 60:
                loss_threshold = -0.18
            elif holding_days > 30:
                loss_threshold = -0.15
            else:
                loss_threshold = -0.12

            if profit_pct < loss_threshold:
                size_to_sell = max(int(current_position_size * 0.40), min_meaningful_trade)
                size_to_sell = min(size_to_sell, current_position_size)

                if size_to_sell >= min_meaningful_trade:
                    self._last_sell_day = current_day
                    return {
                        'action': 'sell',
                        'size': size_to_sell,
                        'reason': f"{market_regime} intelligent stop loss"
                    }
        return None

    def _check_sideways_reduction(self, market_regime, holding_days, profit_pct,
                                 current_position_size, min_meaningful_trade,
                                 current_day, passive_mode):
        """检查横盘市场减仓"""
        if not passive_mode:
            if market_regime == "sideways" and holding_days > 90:
                # 使用 is_long_term_sideways 方法
                if self.strategy.market_state.is_long_term_sideways(window=30, range_threshold=0.04):
                    if profit_pct > 0.03:
                        sell_size = max(int(current_position_size * 0.5), min_meaningful_trade)
                        sell_size = min(sell_size, current_position_size)

                        if sell_size >= min_meaningful_trade:
                            self._last_sell_day = current_day
                            return {
                                'action': 'sell',
                                'size': sell_size,
                                'reason': f"Sideways market optimization"
                            }
        return None

    def _check_long_term_review(self, holding_days, profit_pct, current_position_size,
                               min_meaningful_trade, current_day):
        """超长期仓位回顾"""
        if holding_days % 90 == 0 and holding_days >= 180:
            if profit_pct < 0.10:
                sell_size = max(int(current_position_size * 0.33), min_meaningful_trade)
                sell_size = min(sell_size, current_position_size)

                if sell_size >= min_meaningful_trade:
                    self._last_sell_day = current_day
                    return {
                        'action': 'sell',
                        'size': sell_size,
                        'reason': f"Long-term position review"
                    }
        return None

    def manage_position(self, current_price, market_regime, trend_strength):
        """全面优化的仓位管理，专注于有意义的交易，避免琐碎交易"""
        try:
            self._clean_expired_buy_history()

            # 初始化返回结果
            result = {
                'action': None,
                'size': 0,
                'reason': ''
            }

            # 初始化所有可能使用的变量，避免UnboundLocalError
            in_sell_cooldown = False
            passive_mode = False
            stop_loss_result = None
            profit_taking_result = None
            add_position_result = None
            concentration_result = None
            small_position_result = None
            sideways_result = None
            long_term_result = None
            holding_days = 0
            profit_pct = 0
            current_position_size = 0
            min_meaningful_trade = 15

            # 更新缓存信息
            self._update_position_cache(current_price, market_regime, trend_strength)

            # 获取基础信息
            current_day = len(self.strategy)

            # 1. 检查并重置仓位状态
            if self._check_position_reset():
                return result
            # 新增：针对极小仓位的特殊处理
            portfolio_value = self.strategy.broker.getvalue()
            if current_position_size > 0 and current_position_size < 30:  # 小于30股
                position_value = current_position_size * current_price
                position_ratio = position_value / portfolio_value

                # 如果仓位占比小于2%，考虑清理
                if position_ratio < 0.02:
                    holding_days = self.strategy.get_holding_days()
                    profit_pct = (current_price / self.strategy.entry_price - 1) if self.strategy.entry_price > 0 else 0

                    # 持有超过6个月且有任何盈利就清理
                    if holding_days > 180 and profit_pct > 0:
                        self.strategy.log(
                            f"清理微小仓位 - {current_position_size}股(占比{position_ratio:.1%})，"
                            f"持有{holding_days}天，盈利{profit_pct:.1%}",
                            level="INFO"
                        )
                        return {
                            'action': 'sell',
                            'size': current_position_size,
                            'reason': 'Micro position cleanup'
            }

            # 获取当前持仓信息
            current_position_size = self.strategy.position.size
            portfolio_value = self.strategy.broker.getvalue()
            position_value = current_position_size * current_price
            position_ratio = position_value / portfolio_value
            holding_days = self.strategy.get_holding_days()

            # 获取缓存的盈利信息
            cache = self._position_cache
            profit_pct = cache.get('profit_pct', 0)

            # 计算最小交易规模
            portfolio_factor = max(1.0, portfolio_value / 100000)
            min_meaningful_trade = max(15, int(15 * portfolio_factor))

            # 3. 处理小仓位
            small_position_result = self._handle_small_position(
                current_position_size, min_meaningful_trade, holding_days,
                trend_strength, profit_pct, current_day
            )
            if small_position_result:
                return small_position_result

            # 4. 检查仓位集中度
            concentration_result = self._check_position_concentration(
                position_ratio, portfolio_value, current_position_size,
                current_price, min_meaningful_trade, profit_pct, current_day
            )
            if concentration_result:
                return concentration_result

            # 5. 检查加仓条件
            add_position_result = self._check_add_position(current_price, market_regime)
            if add_position_result:
                # 更新批次计数
                self.added_batches += 1
                self._last_add_bar = len(self.strategy)
                self.strategy.log(f"有意义加仓 - 规模: {add_position_result['size']}股, 原因: {add_position_result['reason']}", level="INFO")
                return add_position_result

            # 6. 检查获利了结
            profit_taking_result = self._check_profit_taking(
                profit_pct, passive_mode, min_meaningful_trade,
                current_position_size, current_day
            )
            if profit_taking_result:
                self.strategy.log(f"获利止盈 - 规模: {profit_taking_result['size']}股, 利润: {profit_pct:.2%}", level="INFO")
                return profit_taking_result

            # 7. 新仓位保护期（改进版）
            new_position_protection_days = ParamAccessor.get_param(self, 'new_position_protection_days')

            # 根据市场状态动态调整保护期
            if market_regime == "downtrend":
                # 下跌市场延长保护期
                effective_protection_days = int(new_position_protection_days * 1.5)
            elif market_regime == "strong_uptrend":
                # 强势上涨缩短保护期
                effective_protection_days = max(2, int(new_position_protection_days * 0.6))
            else:
                effective_protection_days = new_position_protection_days

            if holding_days < effective_protection_days:
                self.strategy.log(
                    f"新仓位保护期 ({holding_days}/{effective_protection_days} 天) - "
                    f"市场{market_regime} - 暂不减仓",
                    level="DEBUG"
                )
                return result

            # 8. 止损检查
            stop_loss_result = self._check_stop_loss(
                holding_days, trend_strength, market_regime, profit_pct,
                current_position_size, min_meaningful_trade, current_day
            )
            if stop_loss_result:
                self.strategy.log(f"智能止损 - 规模: {stop_loss_result['size']}股, 亏损: {profit_pct:.2%}", level="INFO")
                return stop_loss_result

            # 2. 检查卖出冷却期
            in_sell_cooldown, passive_mode = self._check_sell_cooldown(current_day, market_regime, trend_strength)

            # 9. 横盘市场减仓
            sideways_result = self._check_sideways_reduction(
                market_regime, holding_days, profit_pct, current_position_size,
                min_meaningful_trade, current_day, passive_mode
            )
            if sideways_result:
                self.strategy.log(
                    f"横盘市场优化 - 规模: {sideways_result['size']}股, 持有{holding_days}天, 利润: {profit_pct:.2%}",
                    level="INFO"
                )
                return sideways_result

            # 10. 超长期仓位回顾
            long_term_result = self._check_long_term_review(
                holding_days, profit_pct, current_position_size,
                min_meaningful_trade, current_day
            )
            if long_term_result:
                self.strategy.log(
                    f"长期仓位评估 - 规模: {long_term_result['size']}股, 持有{holding_days}天, 利润不足: {profit_pct:.2%}",
                    level="INFO"
                )
                return long_term_result

            return result


        except Exception as e:
            self.strategy.log(f"Position management error: {str(e)}", level='INFO')
            raise

    def _check_enhanced_profit_levels(self, profit_pct):
        """更平衡的获利了结策略，修复参数访问方式"""
        # 添加最小获利保护
        if profit_pct < 0.05:  # 利润低于5%不触发获利了结
            return 0

        # 对于长期持仓，提高获利门槛
        holding_days = self.strategy.get_holding_days()
        if holding_days < 30 and profit_pct < 0.10:  # 持仓不足30天且利润低于10%
            return 0

        # 动态确定最小有意义的交易量
        portfolio_value = self.strategy.broker.getvalue()
        portfolio_factor = max(1.0, portfolio_value / 100000)

        min_meaningful_trade = max(20, int(20 * portfolio_factor))

        # 获取市场状态和趋势信息
        market_regime = self.strategy.market_state.get_market_regime()
        trend_strength = self.strategy.market_state.get_trend_strength()
        holding_days = self.strategy.get_holding_days()

        # 修改：直接使用ParamAccessor获取参数无默认值
        profit_trigger_1 = ParamAccessor.get_param(self, 'profit_trigger_1')
        profit_trigger_2 = ParamAccessor.get_param(self, 'profit_trigger_2')

        if market_regime == "strong_uptrend":
            take_profit_levels = [
                {'threshold': profit_trigger_1 * 2.5, 'reduce_pct': 0.30},  # 37.5%时减仓30%
                {'threshold': profit_trigger_1 * 1.8, 'reduce_pct': 0.20},  # 27%时减仓20%
                {'threshold': profit_trigger_1 * 1.2, 'reduce_pct': 0.15}   # 18%时减仓15%
            ]
        elif market_regime == "uptrend":
            take_profit_levels = [
                {'threshold': profit_trigger_1 * 1.8, 'reduce_pct': 0.18},  # 27%
                {'threshold': profit_trigger_1 * 1.4, 'reduce_pct': 0.15},  # 21%
                {'threshold': profit_trigger_1 * 1.0, 'reduce_pct': 0.12},  # 15%
                {'threshold': profit_trigger_1 * 0.6, 'reduce_pct': 0.10}   # 9%
            ]
        else:
            take_profit_levels = [
                {'threshold': profit_trigger_1 * 1.4, 'reduce_pct': 0.25},  # 21%
                {'threshold': profit_trigger_1 * 1.0, 'reduce_pct': 0.20},  # 15%
                {'threshold': profit_trigger_1 * 0.7, 'reduce_pct': 0.15},  # 10.5%
                {'threshold': profit_trigger_1 * 0.4, 'reduce_pct': 0.10}   # 6%
            ]

        # 以下代码中的逻辑保持不变，只是使用了上面定义的take_profit_levels
        # 检查超买状态
        overbought = False
        try:
            # 更合理的超买判断
            if hasattr(self.strategy, 'rsi') and self.strategy.rsi[0] > 75:
                overbought = True

            # 价格偏离均线判断
            if (hasattr(self.strategy, 'ma_fast') and
                self.strategy.data.close[0] > self.strategy.ma_fast[0] * 1.15):
                overbought = True
        except:
            pass

        # 获取已执行获利级别
        executed_levels = getattr(self, '_executed_profit_levels', set())

        # 获取当前持仓数量
        current_position_size = self.strategy.position.size
        min_percent_size = max(int(current_position_size * 0.08), min_meaningful_trade)

        # 初始化结果
        result = 0

        # 跳过获利了结的条件更为合理
        if trend_strength > 10.0 and market_regime in ["strong_uptrend", "uptrend"]:
            # 只有极强趋势(16分以上)才考虑跳过获利了结
            if profit_pct < 0.35:  # 从50%降到35%
                self.strategy.log(f"趋势良好，暂不获利了结 - 趋势强度: {trend_strength:.1f}", level="INFO")
                return 0

        # 检查各个获利级别
        for level in take_profit_levels:
            adjusted_threshold = level['threshold']
            adjusted_reduce_pct = level['reduce_pct']

            # 超买状态调整
            if overbought:
                adjusted_threshold *= 0.85  # 超买状态降低获利门槛
                adjusted_reduce_pct *= 1.20  # 增加卖出比例

            # 持仓时间调整
            if holding_days > 180:  # 持有超过6个月
                adjusted_threshold *= 0.80  # 降低20%获利门槛
            elif holding_days > 90:  # 持有超过3个月
                adjusted_threshold *= 0.90  # 降低10%获利门槛

            # 检查是否达到获利门槛
            if profit_pct >= adjusted_threshold and adjusted_threshold not in executed_levels:
                reduce_pct = adjusted_reduce_pct

                # 根据持仓时间调整卖出比例
                if holding_days > 120:
                    reduce_pct = max(reduce_pct, 0.15)  # 长期持仓至少减15%

                # 趋势强度调整
                if trend_strength > 15.0:
                    reduce_pct *= 0.85  # 减少15%的减仓比例

                # 计算卖出数量
                sell_size = int(current_position_size * reduce_pct)
                sell_size = max(sell_size, min_percent_size)
                sell_size = min(sell_size, current_position_size)

                # 确保仍有足够的最小持仓
                remaining = current_position_size - sell_size
                if remaining < min_meaningful_trade and remaining > 0:
                    if remaining < min_meaningful_trade * 0.5:
                        sell_size = current_position_size
                    else:
                        sell_size = current_position_size - min_meaningful_trade

                # 极端情况：在强ETF趋势中小额获利了结
                if market_regime in ["strong_uptrend"] and trend_strength > 14.0:
                    if sell_size < current_position_size * 0.08:  # 卖出不到8%持仓
                        if profit_pct < 0.40:  # 利润低于40%时完全跳过
                            self.strategy.log(
                                f"强ETF趋势小额获利跳过 - 利润: {profit_pct:.2%}, 趋势强度: {trend_strength:.1f}",
                                level='INFO'
                            )
                            sell_size = 0
                        else:
                            sell_size = int(sell_size * 0.75)  # 减少25%卖出量

                if sell_size > 0 and sell_size >= min_meaningful_trade:
                    self.strategy.log(
                        f"获利信号 - 利润: {profit_pct:.2%}, 目标: {adjusted_threshold:.2%}, 卖出: {sell_size}股 ({reduce_pct:.0%})",
                        level='INFO'
                    )
                    executed_levels.add(adjusted_threshold)
                    setattr(self, '_executed_profit_levels', executed_levels)
                    result = sell_size
                    break  # 找到合适的级别后退出

        # 如果没有触发上面的获利条件，检查长期持仓获利
        if result == 0:
            # 长期持仓时间获利逻辑 - 使用实际参数
            time_based_profit_min_days = ParamAccessor.get_param(self, 'time_based_profit_min_days')
            time_based_profit_min_pct = ParamAccessor.get_param(self, 'time_based_profit_min_pct')

            # 根据实际参数计算时间获利门槛
            if holding_days > 270:  # 9个月以上
                time_profit_threshold = time_based_profit_min_pct * 0.8  # 降低要求
            elif holding_days > 180:  # 6个月以上
                time_profit_threshold = time_based_profit_min_pct * 1.0
            elif holding_days > 120:  # 4个月以上
                time_profit_threshold = time_based_profit_min_pct * 1.2  # 增加要求
            else:
                time_profit_threshold = float('inf')  # 不触发

            if holding_days > time_based_profit_min_days:
                self.strategy.log(
                    f"时间获利检查 - 持有{holding_days}天 > {time_based_profit_min_days}天, "
                    f"利润{profit_pct:.2%} vs 要求{time_profit_threshold:.2%}",
                    level="INFO"
                )

            # 长期持仓时间获利触发
            if holding_days > time_based_profit_min_days and profit_pct > time_profit_threshold:
                # 趋势检查
                if trend_strength > 16.0 and market_regime in ["strong_uptrend"]:
                    time_reduce_pct *= 0.5  # 强趋势下减半卖出比例，而不是完全跳过
                    self.strategy.log(
                        f"强趋势ETF跳过时间获利 - 持有{holding_days}天, 趋势强度: {trend_strength:.1f}",
                        level='INFO'
                    )

                # 获取时间周期减仓比例参数
                time_based_profit_size_pct = ParamAccessor.get_param(self, 'time_based_profit_size_pct')

                # 计算更合理的卖出比例
                if holding_days > 270:  # 9个月以上
                    time_reduce_pct = time_based_profit_size_pct * 1.0
                elif holding_days > 180:  # 6个月以上
                    time_reduce_pct = time_based_profit_size_pct * 0.8
                elif holding_days > 120:  # 4个月以上
                    time_reduce_pct = time_based_profit_size_pct * 0.6
                else:
                    time_reduce_pct = time_based_profit_size_pct * 0.5

                # ETF趋势调整
                if market_regime in ["strong_uptrend", "uptrend"] and trend_strength > 12.0:
                    time_reduce_pct *= 0.8  # 在强趋势中减少20%卖出量

                # 计算卖出规模
                sell_size = max(int(current_position_size * time_reduce_pct), min_meaningful_trade)
                sell_size = min(sell_size, current_position_size)

                # 确保最小持仓规模
                remaining = current_position_size - sell_size
                if remaining < min_meaningful_trade and remaining > 0:
                    if remaining < min_meaningful_trade * 0.5:
                        sell_size = current_position_size
                    else:
                        sell_size = current_position_size - min_meaningful_trade

                if sell_size > 0 and sell_size >= min_meaningful_trade:
                    self.strategy.log(
                        f"长期持仓获利 - 持有{holding_days}天, 利润{profit_pct:.2%}, 减仓{sell_size}股({time_reduce_pct:.0%})",
                        level='INFO'
                    )
                    self._last_profit_check_result = sell_size
                    return sell_size

        return result

    # 在PositionManager类中修改_update_position_cache函数
    #@track_method(category="仓位管理方法")
    def _update_position_cache(self, current_price, market_regime, trend_strength):
        """Update internal cache for position info."""
        current_bar = len(self.strategy.data)

        # Avoid re-calculating if this bar hasn't advanced
        if self._last_update == current_bar:
            return

        try:
            portfolio_value = self.strategy.broker.getvalue()
            if portfolio_value <= 1e-9:
                portfolio_value = 1e-9

            # 明确标记缓存已更新 - 新增
            cache_changed = False

            if self.strategy.position:
                # 1) 计算当前PnL%
                if not hasattr(self.strategy, 'entry_price') or \
                   self.strategy.entry_price is None or \
                   self.strategy.entry_price == 0:
                    profit_pct = 0.0
                else:
                    profit_pct = (current_price / self.strategy.entry_price) - 1

                # 2) 计算持仓比例
                position_value = self.strategy.position.size * current_price
                position_pct = position_value / portfolio_value

                # 检查缓存是否有实质性变化 - 新增
                old_cache = self._position_cache.copy() if hasattr(self, '_position_cache') else {}

                self._position_cache.update({
                    'profit_pct': profit_pct,
                    'position_value': position_value,
                    'position_pct': position_pct,
                    'market_regime': market_regime,
                    'trend_strength': trend_strength
                })

                # 检查是否有重要值变化 - 新增
                if (not old_cache or
                    abs(old_cache.get('profit_pct', 0) - profit_pct) > 0.01 or
                    old_cache.get('market_regime') != market_regime or
                    abs(old_cache.get('trend_strength', 0) - trend_strength) > 0.5):
                    cache_changed = True

            self._last_update = current_bar

            # 明确标记缓存变化 - 新增
            self._cache_modified = cache_changed

        except Exception as e:
            self.strategy.log(f"Cache update error: {str(e)}", level='INFO')
            raise  # 让错误暴露出来，而不是静默返回0

    def get_dynamic_cash_utilization(self):
        """根据市场状态和账户状况动态调整现金利用率"""
        # 基础利用率
        base_utilization = 0.7  # 保守起见，最多使用70%资金

        # 获取当前账户盈亏状态
        current_value = self.strategy.broker.getvalue()
        initial_value = self.strategy.broker.startingcash
        account_pnl = (current_value / initial_value - 1)

        # 根据账户盈亏调整
        if account_pnl > 0.20:  # 盈利20%以上
            utilization_mult = 1.1
        elif account_pnl > 0.10:  # 盈利10-20%
            utilization_mult = 1.0
        elif account_pnl > 0:  # 小幅盈利
            utilization_mult = 0.9
        else:  # 亏损状态
            utilization_mult = 0.7  # 大幅降低仓位

        # 根据市场状态调整
        market_regime = self.market_state.get_market_regime()
        if market_regime == "strong_uptrend":
            regime_mult = 1.1
        elif market_regime == "uptrend":
            regime_mult = 1.0
        elif market_regime == "sideways":
            regime_mult = 0.8
        else:  # downtrend
            regime_mult = 0.6

        # 最终利用率
        final_utilization = base_utilization * utilization_mult * regime_mult

        # 限制在合理范围内
        return min(0.85, max(0.3, final_utilization))

    def calculate_kelly_position(self):
        """使用简化凯利公式计算最优仓位"""
        # 从历史交易计算胜率和盈亏比
        closed_trades = [t for t in self.trade_manager.executed_trades if t['status'] == 'closed']

        if len(closed_trades) < 10:  # 交易样本不足，使用保守仓位
            return 0.25

        # 计算胜率
        winning_trades = [t for t in closed_trades if t['pnl'] > 0]
        win_rate = len(winning_trades) / len(closed_trades)

        # 计算平均盈亏比
        avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
        losing_trades = [t for t in closed_trades if t['pnl'] < 0]
        avg_loss = abs(np.mean([t['pnl'] for t in losing_trades])) if losing_trades else 1

        win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 1

        # 凯利公式：f = (p * b - q) / b
        # p = 胜率, q = 败率, b = 盈亏比
        kelly_fraction = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio

        # 使用1/4凯利值（更保守）
        conservative_kelly = kelly_fraction * 0.25

        # 限制在合理范围
        return min(0.4, max(0.05, conservative_kelly))

    def _calculate_position_size(self):
        """
        修复的仓位大小计算逻辑 - 保留所有重要功能
        """
        try:
            cash_available = self.strategy.broker.get_cash()
            current_price = self.strategy.data.close[0]
            portfolio_value = self.strategy.broker.getvalue()

            # 缓存市场状态，避免重复调用
            market_regime = self.market_state.get_market_regime()
            trend_strength = self.market_state.get_trend_strength()

            cash_reserve_pct = ParamAccessor.get_param(self, 'position_cash_reserve')
            base_position_pct = ParamAccessor.get_param(self, 'base_position_pct')

            if not self.strategy.position:
                if market_regime in ["uptrend", "strong_uptrend"]:
                    base_position_pct = base_position_pct * 1.5
                    self.strategy.log(f"空仓+上涨趋势，提高初始仓位至{base_position_pct:.1%}", level="INFO")

            # 获取仓位参数
            position_portfolio_factor_base = ParamAccessor.get_param(self, 'position_portfolio_factor_base')
            position_min_size_factor = ParamAccessor.get_param(self, 'position_min_size_factor')
            position_portfolio_high = ParamAccessor.get_param(self, 'position_portfolio_high')
            position_portfolio_medium = ParamAccessor.get_param(self, 'position_portfolio_medium')
            position_batch_decay = ParamAccessor.get_param(self, 'position_batch_decay')
            position_min_add_pct = ParamAccessor.get_param(self, 'position_min_add_pct')

            # 获取市场数据
            market_regime = self.market_state.get_market_regime()
            trend_strength = self.market_state.get_trend_strength()

            # 1. 最小仓位大小
            min_position_size = ParamAccessor.get_param(self, 'min_position_size')
            portfolio_factor = max(position_min_size_factor, portfolio_value / position_portfolio_factor_base)

            # 增加最小持仓规模，并随投资组合增长
            self.min_position_size = max(15, int(min_position_size * portfolio_factor))

            # ETF固定最小仓位
            min_meaningful_trade = max(15, int(15 * max(1.0, portfolio_value / 100000)))

            # 资金保护：确保始终保留一定现金
            current_portfolio_value = self.strategy.broker.getvalue()  # 包含持仓的总价值
            min_cash_reserve = current_portfolio_value * cash_reserve_pct  # 基于当前总资产
            # 根据市场状态动态调整现金储备
            if market_regime in ["strong_uptrend", "uptrend"]:
                actual_reserve_needed = min_cash_reserve * 0.5  # 上涨趋势减少现金储备
            elif self.strategy.position and self.strategy.position.size > 0:
                # 已有仓位时进一步减少储备要求
                actual_reserve_needed = min_cash_reserve * 0.3
            else:
                actual_reserve_needed = min_cash_reserve

            if cash_available < actual_reserve_needed:
                if cash_available < actual_reserve_needed * 0.5:
                    self.strategy.log(
                        f"现金不足 - 可用${cash_available:.2f} < 需要${actual_reserve_needed:.2f}",
                        level="INFO"
                    )
                    return 0

            if cash_available < min_cash_reserve:
                # 不要完全阻止交易，而是减少交易规模
                if cash_available > min_cash_reserve * 0.5:  # 至少有一半储备金
                    # 按比例减少交易规模
                    reduction_factor = cash_available / min_cash_reserve
                    self.strategy.log(
                        f"资金紧张，按比例{reduction_factor:.2f}减少交易规模",
                        level="INFO"
                    )
                    # 继续执行，但后续会减少position_size
                else:
                    self.strategy.log(
                        f"资金严重不足 - 可用${cash_available:.2f} < 最低储备50%",
                        level="CRITICAL"
                    )
                    return 0

            # 2. 基础仓位
            base_position_pct = ParamAccessor.get_param(self, 'base_position_pct')

            # 3. 根据市场状态调整的仓位大小
            size_multipliers = {
                'strong_uptrend': ParamAccessor.get_param(self, 'strong_uptrend_mult'),
                'uptrend': ParamAccessor.get_param(self, 'uptrend_mult'),
                'sideways': ParamAccessor.get_param(self, 'sideways_mult'),
                'downtrend': ParamAccessor.get_param(self, 'downtrend_mult')
            }

            # 4. 应用市场状态乘数
            trend_multiplier = size_multipliers.get(market_regime, 1.0)

            # 5. 应用趋势强度调整
            trend_factor = 1.0
            if trend_strength > 0:
                if trend_strength > 10.0:
                    trend_factor = 2.0
                elif trend_strength > 8.0:
                    trend_factor = 1.8
                elif trend_strength > 6.0:
                    trend_factor = 1.6
                elif trend_strength > 4.0:
                    trend_factor = 1.4

            trend_multiplier *= trend_factor

            # 【保留原功能】强势趋势下的资金利用优化
            if market_regime in ["strong_uptrend", "uptrend"] and trend_strength > 8.0:
                cash_utilization = self.get_dynamic_cash_utilization()
                cash_available = cash_available * cash_utilization  # 直接修改缓存值
                self.strategy.log(f"动态资金利用率: {cash_utilization:.1%} - 可用资金: {cash_available:.2f}", level="INFO")

            # 【保留原功能】根据市场状态和风险情况动态调整仓位上限
            max_position_pct = ParamAccessor.get_param(self, 'max_position_pct')
            if market_regime == "downtrend" or trend_strength < -5:
                # 下跌趋势最多使用30%资金
                max_position_pct = min(max_position_pct, 0.3)
                self.strategy.log(f"下跌趋势限制仓位至30%", level="CRITICAL")
            elif market_regime == "sideways" and trend_strength < 2:
                # 震荡市场最多使用50%资金
                max_position_pct = min(max_position_pct, 0.5)
                self.strategy.log(f"震荡市场限制仓位至50%", level="INFO")

            # 基于最近交易亏损的风险控制
            if hasattr(self.strategy, 'last_trade_pnl') and self.strategy.last_trade_pnl < 0:
                days_since_loss = len(self.strategy) - self.strategy.last_loss_day
                if days_since_loss < 10:  # 10天内有亏损
                    # 温和的仓位调整，不是指数衰减
                    loss_impact = min(abs(self.strategy.last_trade_pnl) / portfolio_value, 0.2)  # 最多影响20%
                    max_position_pct *= (1 - loss_impact * 0.5)  # 最多减少10%仓位
                    self.strategy.log(
                        f"近期亏损调整仓位 - 亏损{self.strategy.last_trade_pnl:.2f}, "
                        f"距今{days_since_loss}天, 仓位调整至{max_position_pct:.1%}",
                        level="INFO"
                    )

            # 6. 计算仓位百分比 - 但不低于最小值
            raw_position_pct = base_position_pct * trend_multiplier
            position_pct = max(raw_position_pct, 0.1)  # 确保最小仓位百分比

            # 7. 最大仓位
            position_pct = min(position_pct, max_position_pct)

            # 8. 现金储备（已在前面处理）
            max_affordable = int(cash_available / current_price)

            # 9. 基于投资组合价值计算目标
            target_value = portfolio_value * position_pct
            target_shares = int(target_value / current_price)

            # 【保留原功能】确保最小有意义仓位大小
            dynamic_min_size = max(30, int(portfolio_value / current_price * 0.05))  # 不超过投资组合的5%
            min_meaningful_trade = min(self.min_position_size, dynamic_min_size)

            # 10. 现有仓位调整
            if self.strategy.position and self.strategy.position.size > 0:
                current_position_size = self.strategy.position.size
                portfolio_portion = current_position_size * current_price / portfolio_value

                # 根据现有仓位调整加仓目标
                if portfolio_portion > 0.95:  # 只有超过95%才限制
                    target_shares = int(target_shares * 0.9)  # 减少限制幅度
                    self.strategy.log(f"调整加仓规模 - 当前持仓比例: {portfolio_portion:.2%}", level="INFO")
                elif portfolio_portion > position_portfolio_medium:
                    target_shares = int(target_shares * 0.9)
                    self.strategy.log(f"适度调整加仓规模 - 当前持仓比例: {portfolio_portion:.2%}", level="INFO")

                # 根据已添加批次进一步调整
                batch_factor = max(0.8, 1.0 - (self.added_batches * position_batch_decay))
                target_shares = int(target_shares * batch_factor)

                # 基于现有仓位的最小加仓规模
                min_based_on_position = max(
                    min_meaningful_trade,
                    int(current_position_size * position_min_add_pct)
                )

                # 使用计算目标和基于仓位最小加仓中的较大值
                target_shares = max(target_shares, min_based_on_position)

            # 11. 受限于可用资金
            position_size = min(target_shares, max_affordable)

            # 12. 强制最小仓位大小 - 但不强制极小交易
            if 0 < position_size < min_meaningful_trade:
                if cash_available >= current_price * 5:  # 至少能买5股
                    position_size = max(5, int(cash_available / current_price * 0.9))  # 使用90%的资金
                    self.strategy.log(f"资金有限，调整为小额交易: {position_size}股", level="INFO")
                else:
                    return 0

            # 13. 如果是ETF且资金充足，增加规模
            position_scaler = ParamAccessor.get_param(self, 'position_scaler')
            scale_cash_mult = ParamAccessor.get_param(self, 'scale_cash_mult')

            target_size = int(position_size * (1 + position_scaler / 10.0))
            affordable = min(target_size, max_affordable)
            if affordable > position_size and cash_available > affordable * current_price * scale_cash_mult:
                self.strategy.log(f"ETF 放大仓位 {position_size}->{affordable} 股 "
                                  f"(scaler={position_scaler})", level='INFO')
                position_size = affordable

            # 根据账户盈亏状态调整仓位
            current_value = self.strategy.broker.getvalue()
            initial_value = self.strategy.broker.startingcash
            account_pnl = (current_value / initial_value - 1)

            if account_pnl < -0.05:  # 账户亏损超过5%
                # 减小仓位，控制风险
                position_pct *= 0.7
                self.strategy.log(
                    f"账户亏损{account_pnl:.1%}，减小仓位至{position_pct:.1%}",
                    level="INFO"
                )
            elif account_pnl > 0.10:  # 账户盈利超过10%
                # 可以适度增加仓位
                position_pct *= 1.2
                position_pct = min(position_pct, 0.8)  # 但不超过80%

            max_trade_value = portfolio_value * 0.95  # 允许使用95%资金
            max_affordable_shares = int(max_trade_value / current_price)

            # 绝对数量上限（提高上限）
            absolute_max_shares = int(portfolio_value / current_price * 0.95)  # 去掉1000的下限

            # 应用保护限制
            original_position_size = position_size
            position_size = min(position_size, max_affordable_shares, absolute_max_shares)

            # 【保留原功能】详细的调试信息
            if position_size != original_position_size:
                reduction_pct = (1 - position_size/original_position_size) * 100
                self.strategy.log(
                    f"仓位保护 - 原计算:{original_position_size}股 → 限制后:{position_size}股 (削减{reduction_pct:.1f}%) "
                    f"(最大交易价值:{max_trade_value:.0f}, 绝对上限:{absolute_max_shares}股, 现金:{cash_available:.0f})",
                    level="CRITICAL"
                )

            # 最终验证：确保返回值合理
            if position_size <= 0 or position_size > absolute_max_shares:
                self.strategy.log(f"仓位计算异常，返回0: {position_size}", level="CRITICAL")
                return 0

            # 记录详细决策
            self.strategy.log(
                f"头寸规模计算 - 目标占比: {position_pct:.2%}, 目标股数: {target_shares}, 实际股数: {position_size}, 市场状态: {market_regime}, 趋势强度: {trend_strength:.1f}",
                level='INFO'
            )
            return position_size

        except Exception as e:
            self.strategy.log(f"持仓计算错误: {str(e)}", level='CRITICAL')
            return 0

    def _should_buy_the_dip(self, current_price):
        """统一的回调买入检测逻辑"""
        try:
            # 立即获取所有需要的参数
            base_dip_pct = ParamAccessor.get_param(self, 'dip_base_pct')
            additional_pct_per_batch = ParamAccessor.get_param(self, 'dip_per_batch')
            batches_allowed = ParamAccessor.get_param(self, 'batches_allowed')
            min_bars_between_trades = ParamAccessor.get_param(self, 'min_bars_between_trades')
            min_position_size = ParamAccessor.get_param(self, 'min_position_size')

            # 计算本次所需的回调幅度
            required_dip_pct = base_dip_pct + (self.added_batches * additional_pct_per_batch)

            # 1. 时间控制检查
            if not self._check_time_interval(min_bars_between_trades):
                return False

            # 2. 批次控制检查
            if not self._check_batch_limit(batches_allowed):
                return False

            # 3. 资金检查
            if not self._check_available_funds(current_price, min_position_size):
                return False

            # 4. 根据是否有仓位分别处理
            if self.strategy.position:
                return self._check_position_dip(current_price, required_dip_pct)
            else:
                return self._check_new_position_dip(current_price, base_dip_pct)

        except Exception as e:
            self.strategy.log(f"回调买入逻辑错误: {str(e)}")
            return False

    def _check_time_interval(self, min_bars_between_trades):
        """检查加仓时间间隔"""
        if self.last_add_bar is not None:
            days_since_last_add = len(self.strategy) - self.last_add_bar
            self.strategy.log(f"回调检查 - 距上次加仓: {days_since_last_add}天, 最小间隔: {min_bars_between_trades}天", level="INFO")

            if days_since_last_add < min_bars_between_trades:
                self.strategy.log(f"[参数影响] 时间间隔不足，跳过回调买入", level="DEBUG")
                return False
        return True

    def _check_batch_limit(self, batches_allowed):
        """检查批次限制"""
        self.strategy.log(f"批次控制 - 已添加批次: {self.added_batches}, 最大允许: {batches_allowed}", level="INFO")

        if self.added_batches >= batches_allowed:
            if not hasattr(self, '_max_batches_logged') or not self._max_batches_logged:
                self.strategy.log(f"[参数影响] 达到最大加仓次数: {self.added_batches}/{batches_allowed}，后续将跳过回调买入", level="DEBUG")
                self._max_batches_logged = True
            return False
        return True

    def _check_available_funds(self, current_price, min_position_size):
        """检查可用资金 - 简化版"""
        cash_available = self.strategy.broker.get_cash()

        # 只要能买得起最小交易单位（如10股）就允许交易
        min_shares = min_position_size  # 使用传入的参数，而不是硬编码
        min_required_cash = current_price * min_shares * 1.01  # 留1%缓冲

        if cash_available < min_required_cash:
            # 只在真正资金不足时输出一次
            if not hasattr(self, '_last_fund_warning_bar') or len(self.strategy) - self._last_fund_warning_bar > 50:
                self.strategy.log(
                    f"资金不足 - 可用{cash_available:.2f} < 最低要求{min_required_cash:.2f}(买{min_shares}股)",
                    level="INFO"
                )
                self._last_fund_warning_bar = len(self.strategy)
            return False
        return True

    def _check_position_dip(self, current_price, required_dip_pct):
        """检查持仓时的加仓回调条件 - 完全重写，修复逻辑错误"""

        # 获取市场状态
        market_regime = self.market_state.get_market_regime()
        trend_strength = self.market_state.get_trend_strength()
        holding_days = self.strategy.get_holding_days()

        # === 核心修改：使用近期高点而不是历史买入价格 ===

        lookback_days_param = ParamAccessor.get_param(self, 'lookback_days_for_dip')
        lookback_days = min(lookback_days_param, len(self.strategy.data.close)-1)

        # 确保有足够的数据
        if lookback_days > len(self.strategy.data.close)-1:
            lookback_days = len(self.strategy.data.close)-1

        # 获取近期高点
        try:
            recent_high = max(self.strategy.data.high.get(size=lookback_days))
        except:
            recent_high = self.strategy.data.high[0]

        # 计算从高点的回调幅度
        dip_from_high = (recent_high - current_price) / recent_high if recent_high > 0 else 0

        # === 动态调整回调要求 ===
        adjusted_dip_pct = required_dip_pct

        # 1. 强势上涨趋势：大幅降低回调要求
        if market_regime == "strong_uptrend" and trend_strength > 12.0:
            adjusted_dip_pct *= 0.3  # 只需要0.6%的回调
            self.strategy.log(f"强势上涨大幅降低回调要求: {required_dip_pct:.3f} → {adjusted_dip_pct:.3f}", level="INFO")
        elif market_regime == "strong_uptrend":
            adjusted_dip_pct *= 0.5
        elif market_regime == "uptrend":
            adjusted_dip_pct *= 0.7
        elif market_regime == "sideways":
            adjusted_dip_pct *= 1.0  # 保持不变
        else:  # downtrend
            adjusted_dip_pct *= 1.5

        # 2. 长期持仓进一步放宽
        if holding_days > 180:
            adjusted_dip_pct *= 0.5
            self.strategy.log(f"长期持仓放宽回调要求: {adjusted_dip_pct*2:.3f} → {adjusted_dip_pct:.3f}", level="INFO")
        elif holding_days > 90:
            adjusted_dip_pct *= 0.7

        # 3. 批次调整 - 后续批次要求更高的回调
        if self.added_batches > 0:
            batch_multiplier = 1.0 + (self.added_batches * 0.3)  # 每批次增加30%要求
            adjusted_dip_pct *= batch_multiplier
            self.strategy.log(f"第{self.added_batches+1}批次调整: 回调要求×{batch_multiplier:.1f}", level="DEBUG")

        # === 判断是否满足回调条件 ===
        meets_dip = dip_from_high >= adjusted_dip_pct

        if not meets_dip:
            if ParamAccessor.get_param(self, 'debug_mode'):
                self.strategy.log(
                    f"回调条件未满足: 需要从高点{recent_high:.2f}回调{adjusted_dip_pct*100:.1f}%, "
                    f"当前只回调了{dip_from_high*100:.1f}%",
                    level="DEBUG"
                )
            return False

        # === 止跌确认（保留原有逻辑但简化）===
        # 只在下跌趋势中要求止跌信号
        if market_regime == "downtrend":
            if not self._check_stop_falling_signal():
                return False

        # === 成交量确认（简化）===
        volume_ok = True
        if hasattr(self.strategy, 'volume_ma') and self.strategy.volume_ma[0] > 0:
            volume_ratio = self.strategy.data.volume[0] / self.strategy.volume_ma[0]
            volume_ok = volume_ratio > 0.5  # 成交量不能太低

        if not volume_ok:
            self.strategy.log("成交量过低，暂缓加仓", level="DEBUG")
            return False

        # === 成功触发 ===
        self.strategy.log(
            f"回调买入触发 - 从{lookback_days}天高点{recent_high:.2f}回调{dip_from_high*100:.1f}%至{current_price:.2f}, "
            f"满足调整后要求{adjusted_dip_pct*100:.1f}%",
            level="INFO"
        )

        # 记录本次加仓价格
        self._last_add_price = current_price

        return True

    def _check_stop_falling_signal(self):
        """检查止跌信号"""
        if len(self.strategy.data.close) < 3:
            return True

        current_price = self.strategy.data.close[0]

        # 检查是否有止跌迹象：今天收盘价高于昨天
        if current_price <= self.strategy.data.close[-1]:
            self.strategy.log(f"等待止跌信号 - 今收{current_price:.2f} <= 昨收{self.strategy.data.close[-1]:.2f}", level="INFO")
            return False

        # 确认：最近3天至少有2天收阳
        recent_up_days = sum(1 for i in range(3) if self.strategy.data.close[-i] > self.strategy.data.open[-i])
        if recent_up_days < 2:
            self.strategy.log(f"止跌信号不足 - 近3天仅{recent_up_days}天收阳", level="INFO")
            return False

        return True

    def _check_new_position_dip(self, current_price, base_dip_pct):
        """检查新建仓位的回调条件"""
        market_regime = self.market_state.get_market_regime()
        trend_strength = self.market_state.get_trend_strength()
        recent_high = max(self.strategy.data.high.get(size=20))
        current_dip_pct = (recent_high - current_price) / recent_high

        # === 优化：移除或改为DEBUG级别 ===
        if ParamAccessor.get_param(self, 'debug_mode'):
            self.strategy.log(
                f"[参数影响] 新建仓位回调检查: 从高点{recent_high:.2f}回调{current_dip_pct:.4f}, "
                f"要求回调{base_dip_pct:.4f}",
                level="DEBUG"
            )

        if current_dip_pct < base_dip_pct * 0.8:  # 给予20%的宽容度
            # 如果市场状态良好，进一步放宽
            if market_regime in ["uptrend", "strong_uptrend", "sideways"] and trend_strength > 0:
                self.strategy.log(
                    f"市场状态良好({market_regime})，放宽新建仓要求",
                    level="INFO"
                )
                return True  # 直接允许建仓
            return False

        self.strategy.log(f"[参数影响] 新建仓位满足回调条件 {current_dip_pct:.4f} >= {base_dip_pct:.4f}", level="CRITICAL")

        # 新建仓需要额外确认
        if not self._check_new_position_confirmation(market_regime, trend_strength):
            return False

        # 如果回调幅度很大，进一步放宽要求
        if current_dip_pct > base_dip_pct * 2.0:
            if trend_strength > 0 and market_regime != "downtrend":
                self.strategy.log(
                    f"[参数影响] 大幅回调触发宽松买入条件 - 回调{current_dip_pct:.2%} > {base_dip_pct*2:.2%}",
                    level="CRITICAL"
                )
                return True

        return True

    def _check_new_position_confirmation(self, market_regime, trend_strength):
        """新建仓位的额外确认"""
        # 确保不是下跌趋势
        ma_fast_down = self.strategy.ma_fast[0] < self.strategy.ma_fast[-5] if len(self.strategy.ma_fast) > 5 else False
        ma_slow_down = self.strategy.ma_slow[0] < self.strategy.ma_slow[-10] if len(self.strategy.ma_slow) > 10 else False

        if ma_fast_down and ma_slow_down:
            self.strategy.log("新建仓位谨慎 - 快慢均线都在下降，等待企稳", level="CRITICAL")
            return False

        # 获取阈值参数
        uptrend_threshold = ParamAccessor.get_param(self, 'uptrend_threshold')
        sideways_threshold = ParamAccessor.get_param(self, 'sideways_threshold')

        # 检查市场状态和趋势强度
        if market_regime in ["strong_uptrend", "uptrend", "sideways"] and trend_strength > max(2.0, sideways_threshold * 0.5):
            try:
                # 技术条件检查
                current_price = self.strategy.data.close[0]
                rising_ma = self.strategy.ma_fast[0] > self.strategy.ma_fast[-5]
                price_above_ma = current_price > self.strategy.ma_slow[0]
                macd_signal = self.strategy.macd.macd[0] > self.strategy.macd.signal[0]
                price_up = current_price > self.strategy.data.close[-1]

                conditions_met = sum([rising_ma, price_above_ma, macd_signal, price_up])
                required_conditions = 2

                if conditions_met >= required_conditions:
                    self.strategy.log(
                        f"[参数影响] 非ETF新建仓信号 - {market_regime}中趋势强度 {trend_strength:.1f}, "
                        f"满足{conditions_met}/{len([rising_ma, price_above_ma, macd_signal, price_up])}个条件, "
                        f"参数base_dip_pct={ParamAccessor.get_param(self, 'dip_base_pct')}直接影响",
                        level="CRITICAL"
                    )
                    return True

            except Exception as e:
                self.strategy.log(f"非ETF新建仓位检查错误: {str(e)}", level="INFO")

        return False

class EnhancedStrategy(bt.Strategy):
    def __init__(self, trading_params=None):
        """策略初始化"""
        # 先调用父类的初始化
        super().__init__()

        # 存储TradingParameters实例
        self.trading_params = trading_params
        self.last_trade_pnl = 0  # 添加这行
        self.last_loss_day = 0   # 添加这行
        self.last_trade_bar = 0  # 确保这行存在
        self.consecutive_losses = 0
        self.last_trade_was_loss = False

        # 将trading_params中的参数直接转换为Backtrader的params
        if trading_params is not None:
            param_count = 0
            key_params = {}
            for param_name in dir(trading_params):
                if not param_name.startswith('_') and not callable(getattr(trading_params, param_name)):
                    try:
                        value = getattr(trading_params, param_name)
                        setattr(self.params, param_name, value)
                        param_count += 1

                        # 只记录核心参数
                        if param_name in ['symbol', 'initial_cash', 'start_date', 'end_date',
                                         'td_count', 'stop_atr', 'profit_trigger_1']:
                            key_params[param_name] = value
                    except Exception as e:
                        print(f"设置参数 {param_name} 失败: {str(e)}")

            print(f"\n策略初始化 - {trading_params.symbol} | 资金: ${trading_params.initial_cash:,.0f}")

        self.symbol = self.trading_params.symbol if self.trading_params else ""
        self._initialize_indicators()

        # 初始化管理器
        self.trade_manager = TradeManager()
        self.market_state = MarketState(self.trading_params if self.trading_params else self.params)
        self.risk_manager = RiskManager(self.trading_params if self.trading_params else self.params)
        self.stop_manager = StopManager(self.trading_params if self.trading_params else self.params)
        self.position_manager = PositionManager(self.trading_params if self.trading_params else self.params)

        self.trade_manager.params = self.trading_params if self.trading_params else self.params

        # 交易状态 - 先初始化基本属性
        self.order = None
        self.entry_price = 0
        self.stop_level = None
        self.buy_signal = False
        self.high_since_entry = 0
        self._executed_trades = []
        self.last_time_profit_day = 0

        # 投资组合跟踪初始化
        self.portfolio_values = []
        self.daily_values = []
        self.daily_returns = []
        self.peak_value = self.broker.getvalue()
        self.daily_value = self.peak_value
        self.last_date = None

        # 设置各管理器的strategy引用
        self.trade_manager.set_strategy(self)
        self.market_state.set_strategy(self)
        self.risk_manager.set_strategy(self)
        self.stop_manager.set_strategy(self)
        self.position_manager.set_strategy(self)

        # 设置其他必要的引用
        self.market_state.set_indicators(self)
        self.position_manager.set_market_state(self.market_state)
        self.position_manager.set_risk_manager(self.risk_manager)

        # 记录重要参数
        params_to_log = ['symbol', 'initial_cash', 'start_date', 'end_date',
                          'atr_period', 'rsi_period', 'fast_ma', 'slow_ma']
        param_str = ", ".join([f"{p}:{ParamAccessor.get_param(self, p)}" for p in params_to_log])
        self.log(f"策略初始化 - {param_str}", level="INFO")

        self._param_cache = {}
        if self.trading_params:
            for attr_name in dir(self.trading_params):
                if not attr_name.startswith('_') and not callable(getattr(self.trading_params, attr_name)):
                    self._param_cache[attr_name] = getattr(self.trading_params, attr_name)

        # 初始化通知器
        self.notifier = None
        if self.trading_params and self.trading_params.serverchan_sendkey:
            self.notifier = ServerChanNotifier(self.trading_params.serverchan_sendkey)
            self.log("Server酱通知器已初始化", level="INFO")


    @property
    def trades(self):
        """返回交易管理器"""
        return self.trade_manager

    def log(self, txt, dt=None, level='INFO'):
        """优化的日志方法，重点显示交易执行和关键决策信息"""
        # 决定是否显示此日志
        should_print = False

        # === 最高优先级：交易执行和重要决策 ===
        if level == 'CRITICAL':
            should_print = True
        elif level == 'INFO':
            # 只显示真正重要的交易信息
            important_keywords = [
                # 实际交易执行（只保留一套）
                '买入执行', '卖出执行',
                # 交易决策
                '准备买入', '准备卖出',
                # 重要的止损和获利
                '止损触发', '获利止盈', '时间周期减仓',
                # 重要的市场状态变化
                '市场状态变化',
                # 风险控制
                '风险限制触发', '仓位保护',
                # 重要的加减仓决策
                '加仓决策', '减仓决策'
            ]
            if any(keyword in txt for keyword in important_keywords):
                should_print = True

            # 过滤掉噪音信息 - 在原有基础上增加几个高频重复项
            noise_keywords = [
                'ENTRY', 'EXIT',  # 删除重复的交易记录
                'TD序列买入信号', 'TD序列触发的加仓信号检测',  # 删除TD噪音
                '[关键修复]', '[参数影响]', '[DEBUG]',  # 删除调试信息
                '从trading_params获取参数', '参数验证',  # 删除参数获取日志
                '回调检查', '回调条件判断', '追踪止损参数详情',  # 删除详细检查日志
                '趋势强度变化',  # 只保留大幅变化
                # 新增：过滤高频重复的回调检查信息
                '[关键修复] 加仓回调检查: 持仓期最高价',  # 过滤每日重复的回调检查
                '新仓位保护期','新建仓位满足回调条件',  # 新增：过滤重复的回调检查
                '卖出冷却期',  # 过滤冷却期的重复提示
                # === 新增：过滤更多重复信息 ===
                '加仓价格满足',  # 过滤频繁的价格检查
                '[参数影响] 新建仓位回调检查',  # 过滤参数影响的详细检查
                '资金严重不足'  # 只保留"资金不足"的简单提示
            ]
            if any(noise in txt for noise in noise_keywords):
                should_print = False

            if "资金不足" in txt or "现金储备" in txt or "仓位保护" in txt:
                should_print = True  # 提升为显示级别

            # 特殊处理：趋势强度只显示大幅变化
            if '趋势强度变化' in txt and '→' in txt:
                try:
                    parts = txt.split('→')
                    if len(parts) == 2:
                        old_val = float(parts[0].split(':')[-1].strip())
                        new_val = float(parts[1].strip())
                        if abs(new_val - old_val) > 8.0:  # 只显示变化超过8的
                            should_print = True
                except:
                    pass

        elif level == 'WARNING':
            # 显示重要警告，但过滤掉重复警告
            if '风险状态' in txt or '极端' in txt or '异常' in txt:
                should_print = True
        elif level == 'ERROR':
            should_print = True

        # DEBUG级别完全不显示
        if level == 'DEBUG':
            should_print = False

        # 如果debug_mode为False，忽略所有DEBUG级别日志
        if not ParamAccessor.get_param(self, 'debug_mode') and level == 'DEBUG':
            return

        if should_print:
            if dt is None:
                try:
                    dt = self.data.datetime.date(0)
                except:
                    dt = datetime.datetime.now()

            print(f'{dt.strftime("%Y-%m-%d")} {txt}')

    #@track_param(['fast_ma', 'slow_ma', 'macd_fast', 'macd_slow', 'macd_signal', 'atr_period', 'rsi_period', 'volatility_window', 'volume_adjust'], "技术指标计算")
    def _initialize_indicators(self):
        """使用优化参数初始化技术指标"""
        try:
            if not hasattr(self, 'trading_params') or self.trading_params is None:
                raise ValueError("无法访问trading_params，请确保已正确设置")
            # 获取基本参数值 - 直接从trading_params读取，不使用默认值
            fast_ma = self.trading_params.fast_ma
            slow_ma = self.trading_params.slow_ma
            mid_ma = self.trading_params.mid_ma
            long_ma = self.trading_params.long_ma
            macd_fast = self.trading_params.macd_fast
            macd_slow = self.trading_params.macd_slow
            macd_signal = self.trading_params.macd_signal
            atr_period = self.trading_params.atr_period
            rsi_period = self.trading_params.rsi_period
            volatility_window = self.trading_params.volatility_window
            volume_adjust = self.trading_params.volume_adjust

            # 同样直接读取额外参数
            atr_ma_period = self.trading_params.atr_ma_period
            momentum_period = self.trading_params.momentum_period
            volume_base_period = self.trading_params.volume_base_period
            volume_adjust_multiplier = self.trading_params.volume_adjust_multiplier

            # 记录使用的参数值，方便调试
            self.log(f"技术指标初始化 - ATR周期:{atr_period}, RSI周期:{rsi_period}, "
                    f"波动率窗口:{volatility_window}, 成交量调整:{volume_adjust}", level="INFO")

            # 存储原始参数值，以便在信号生成时归因
            self._indicator_params = {
                'atr_period': atr_period,
                'rsi_period': rsi_period,
                'volatility_window': volatility_window,
                'volume_adjust': volume_adjust,
                'fast_ma': fast_ma,
                'slow_ma': slow_ma,
                'macd_fast': macd_fast,
                'macd_slow': macd_slow,
                'macd_signal': macd_signal,
                'atr_ma_period': atr_ma_period,
                'volume_base_period': volume_base_period,
                'volume_adjust_multiplier': volume_adjust_multiplier
            }

            # 移动平均线 - 使用获取的参数值
            self.ma_fast = bt.indicators.SMA(
                self.data.close,
                period=min(fast_ma, len(self.data.close)-1)
            )
            self.ma_slow = bt.indicators.SMA(
                self.data.close,
                period=slow_ma
            )
            self.ma_mid = bt.indicators.SMA(
                self.data.close,
                period=mid_ma
            )
            self.ma_long = bt.indicators.SMA(
                self.data.close,
                period=long_ma
            )

            # MACD指标 - 使用参数值
            self.macd = bt.indicators.MACD(
                self.data.close,
                period_me1=macd_fast,
                period_me2=macd_slow,
                period_signal=macd_signal
            )

            # ATR波动率指标
            self.atr = bt.indicators.ATR(
                self.data,
                period=atr_period
            )
            self.atr_ma = bt.indicators.SMA(
                self.atr,
                period=atr_ma_period  # 使用参数代替硬编码值
            )

            # RSI
            self.rsi = bt.indicators.RSI(
                self.data.close,
                period=rsi_period
            )

            # 成交量均线
            volume_period = volume_base_period
            self.volume_ma = bt.indicators.SMA(
                self.data.volume,
                period=volume_period
            )

            # 波动率
            self.volatility = bt.indicators.StdDev(
                self.data.close,
                period=volatility_window
            )

            # 记录指标初始化完成
            self.log(f"技术指标初始化完成", level="INFO")

        except Exception as e:
            self.log(f"指标初始化错误: {str(e)}")
            raise  # 让错误暴露出来，而不是静默返回0

    def get_holding_days(self):
        """获取当前持仓天数的统一方法"""
        if not self.position or not hasattr(self, 'entry_time'):
            return 0
        return len(self) - self.entry_time

    # 在 EnhancedStrategy 类中添加
    def get_min_meaningful_trade(self):
        """获取最小有意义的交易规模"""
        portfolio_value = self.broker.getvalue()
        portfolio_factor = max(1.0, portfolio_value / 100000)

        return max(15, int(15 * portfolio_factor))

    def check_stops(self):
        """代理到StopManager"""
        return self.stop_manager.check_all_stops()

    def stop(self):
        """
        在回测结束前最后一刻调用。
        如果开启了 'final_close' 参数，便强制平仓，以便统计完整的最终盈亏。
        """
        # 使用ParamAccessor访问final_close参数
        final_close = ParamAccessor.get_param(self, 'final_close', True)

        if final_close and self.position:
            self.log("End of backtest - closing final position to realize P&L", level='INFO')
            self.close()  # 强制平仓，确保 TradeManager 记录到已平仓交易

    def notify_data(self, data, status):
        """监控数据状态变化"""
        status_name = data._getstatusname(status)

        # 只在实时模式下记录和通知
        if self.trading_params.live_mode:
            self.log(f'数据状态: {status_name}', level="INFO")

            if status == data.LIVE:
                self.log("策略已切换至实时数据模式", level="INFO")
                if self.notifier:
                    # 检查是否真的是实时数据
                    current_time = self.data.datetime.datetime(0)
                    now = datetime.datetime.now()
                    if (now - current_time).total_seconds() < 86400:  # 24小时内
                        self.notifier.send_message("实时交易启动", "策略开始接收实时数据")
            elif status == data.DISCONNECTED:
                self.log("数据连接断开", level="CRITICAL")
                if self.notifier:
                    self.notifier.send_message("数据连接断开", "请检查数据源连接")

    def next(self):
        """主策略逻辑 - 优化版本，解决小仓位陷阱问题"""
        # ========== 1. 基础检查 ==========
        if len(self.data) < 5:
            return

        # ========== 2. 实时模式数据检查 ==========
        if self.trading_params.live_mode:
            current_time = self.data.datetime.datetime(0)
            now = datetime.datetime.now()
            time_to_now = (now - current_time).total_seconds()

            # 只有当数据时间接近当前时间时才进行延迟检查
            if time_to_now < 86400:  # 24小时内的数据才检查延迟
                if hasattr(self, '_last_data_time'):
                    time_diff = (current_time - self._last_data_time).total_seconds()
                    if time_diff > self.trading_params.max_data_delay:
                        self.log(f"数据延迟过大: {time_diff}秒", level="WARNING")
                        if self.notifier:
                            self.notifier.send_message(
                                "数据延迟警告",
                                f"数据延迟 {time_diff} 秒，超过阈值"
                            )
                self._last_data_time = current_time

        # ========== 3. 亏损记录管理 ==========
        if hasattr(self, 'last_loss_day') and self.last_loss_day > 0:
            days_since_loss = len(self) - self.last_loss_day
            if days_since_loss > 10:  # 10天后重置亏损记录
                old_pnl = self.last_trade_pnl
                self.log(
                    f"重置过期亏损记录 - 已{days_since_loss}天，清除历史亏损${old_pnl:.2f}",
                    level="CRITICAL"
                )
                self.last_trade_pnl = 0
                self.last_loss_day = 0
                self.consecutive_losses = 0

        # ========== 4. 空仓状态重置 ==========
        if not self.position:
            if hasattr(self, 'position_manager'):
                self.position_manager._check_position_reset()

        # ========== 5. 定期账户状态输出（每100个bar）==========
        if len(self.data) % 100 == 0:
            total_value = self.broker.getvalue()
            cash = self.broker.get_cash()
            position_value = self.position.size * self.data.close[0] if self.position else 0

            self.log(f"账户状态 - 总价值:${total_value:.2f}, 现金:${cash:.2f}, "
                    f"持仓价值:${position_value:.2f}, 持仓数量:{self.position.size if self.position else 0}",
                    level="CRITICAL")

        # ========== 6. 定期风险状态重置（每365个bar）==========
        if len(self.data) % 365 == 0:
            if hasattr(self.risk_manager, '_drawdown_triggered') and self.risk_manager._drawdown_triggered:
                days_in_limit = len(self.data) - getattr(self.risk_manager, '_drawdown_trigger_day', 0)
                if days_in_limit > 180:
                    self.risk_manager._drawdown_triggered = False
                    self.risk_manager._drawdown_trigger_day = 0
                    self.log(f"强制解除长期风险限制 - 已限制{days_in_limit}天", level="CRITICAL")

        # ========== 7. 定期小仓位检查（每90天）- 新增 ==========
        if len(self.data) % 90 == 0:
            if self.position and self.position.size > 0:
                portfolio_value = self.broker.getvalue()
                position_value = self.position.size * self.data.close[0]
                position_ratio = position_value / portfolio_value

                if position_ratio < 0.10:  # 仓位占比小于10%
                    holding_days = self.get_holding_days()
                    profit_pct = (self.data.close[0] / self.entry_price - 1) if self.entry_price > 0 else 0

                    self.log(f"定期小仓位检查 - {self.position.size}股(占比{position_ratio:.1%})，"
                            f"持有{holding_days}天，盈亏{profit_pct:.1%}", level="CRITICAL")

                    # 如果持有超过180天的小仓位，考虑清理
                    if holding_days > 180 and position_ratio < 0.05:
                        self.log("触发长期小仓位清理", level="CRITICAL")
                        self.order = self.sell(size=self.position.size)
                        return

        # ========== 8. 异常状态检测 ==========
        if len(self.data) > 5:
            portfolio_value = self.broker.getvalue()
            if portfolio_value < 0:
                self.log(f"检测到资金为负: ${portfolio_value:.2f}，停止交易", level='CRITICAL')
                if self.position:
                    self.order = self.close()
                return

        try:
            # ========== 9. 获取当前价格和更新追踪 ==========
            current_price = self.data.close[0]
            self.risk_manager.update_tracking(current_price, self.broker.getvalue())
            self.portfolio_values.append(self.broker.getvalue())

            # ========== 10. 风险限制检查 ==========
            risk_limited = self.risk_manager.check_risk_limits(current_price)
            if risk_limited:
                if self.position:
                    self.log(f"风险限制触发 - 执行平仓", level='INFO')
                    self.order = self.close()
                return

            # ========== 11. 市场崩盘检查 ==========
            if self.market_state.get_market_regime() == "crash":
                self.log("市场崩盘状态 - 暂停所有买入", level="CRITICAL")
                if not hasattr(self, '_last_crash_bar'):
                    self._last_crash_bar = len(self)
                if self.position:
                    self.log("市场崩盘 - 考虑减仓保护", level="CRITICAL")
                return

            # ========== 12. 更新持仓最高价 ==========
            if self.position:
                if not hasattr(self, 'high_since_entry') or self.high_since_entry == 0:
                    self.high_since_entry = self.data.high[0]
                else:
                    self.high_since_entry = max(self.high_since_entry, self.data.high[0])

            # ========== 13. 日内信息追踪 ==========
            current_date = self.data.datetime.date(0)
            if current_date != getattr(self, '_last_date', None):
                self.daily_open = current_price
                self.daily_value = self.broker.getvalue()
                self._last_date = current_date
                if len(self.portfolio_values) > 1:
                    daily_return = (self.portfolio_values[-1] / self.portfolio_values[-2] - 1)
                    self.daily_returns.append(daily_return)

            # ========== 14. 更新市场状态 ==========
            self.market_state.update_td_sequence()
            market_regime = self.market_state.get_market_regime()
            trend_strength = self.market_state.get_trend_strength()

            # ========== 15. 判断是否为小仓位 - 新增 ==========
            is_small_position = False
            if self.position and self.position.size > 0:
                portfolio_value = self.broker.getvalue()
                position_value = self.position.size * current_price
                position_ratio = position_value / portfolio_value

                # 如果仓位占比小于5%，视为小仓位，可以继续寻找新的买入机会
                if position_ratio < 0.05:
                    is_small_position = True
                    self.log(f"检测到小仓位：{self.position.size}股(占比{position_ratio:.1%})，"
                            f"继续寻找买入机会", level="INFO")

            # ========== 16. 主要交易逻辑 ==========
            if self.position and not is_small_position:
                # === 正常持仓管理逻辑 ===
                self._was_in_position = True

                # 16.1 检查止损
                if self.stop_manager.check_all_stops():
                    self.log(f"止损触发 - 价格: {current_price:.2f}", level='INFO')
                    self.order = self.close()
                    return

                # 16.2 获取仓位管理建议
                position_action = self.position_manager.manage_position(
                    current_price,
                    market_regime,
                    trend_strength
                )

                # 16.3 执行建议的操作
                if position_action['action'] == 'buy':
                    # 加仓
                    self.log(f"加仓决策 - 规模: {position_action['size']}股, 原因: {position_action['reason']}",
                             level="INFO")
                    self.order = self.buy(size=position_action['size'])
                    # 记录买入价格
                    self.position_manager._last_buy_price = current_price
                    self.position_manager._buy_price_history.append({
                        'price': current_price,
                        'size': position_action['size'],
                        'bar': len(self)
                    })
                    self.position_manager.added_batches += 1
                    self.position_manager.last_add_bar = len(self)
                    return

                elif position_action['action'] == 'sell':
                    # 减仓或平仓
                    self.log(f"【准备卖出】 - 原因: {position_action['reason']}, 数量: {position_action['size']}股",
                            level="INFO")

                    # 检查是否为小额获利了结
                    if 'Take profit' in position_action['reason']:
                        profit_pct = (current_price / self.entry_price - 1) if self.entry_price > 0 else 0
                        if profit_pct < 0.10 and position_action['size'] < self.position.size * 0.2:
                            self.log(f"跳过小额获利了结 - 利润{profit_pct:.1%}，卖出比例过小", level="INFO")
                            return

                    self.order = self.sell(size=position_action['size'])
                    return

            else:
                # === 空仓或小仓位时的买入逻辑 ===

                # 16.4 重置TD序列（如果刚平仓）
                if hasattr(self, '_was_in_position') and self._was_in_position:
                    self.market_state.reset_td_sequence()
                    self._was_in_position = False

                # 16.5 获取TD参数
                td_count = ParamAccessor.get_param(self, 'td_count')
                signal_threshold = ParamAccessor.get_param(self, 'signal_threshold')
                self.log(f"买入信号参数检查 - td_count={td_count}, 信号阈值={signal_threshold}", level="INFO")

                # 16.6 更新TD序列
                self.market_state.update_td_sequence()

                # 16.7 检查买入信号
                base_buy = self.market_state.get_buy_signal()

                # 16.8 检查市场恢复（如果之前有崩盘）
                if hasattr(self, '_last_crash_bar'):
                    bars_since_crash = len(self) - self._last_crash_bar
                    # 这里可以添加市场恢复的判断逻辑

                # 16.9 检查回调条件
                dip_signal = self.position_manager._should_buy_the_dip(current_price)

                # 16.10 TD序列必要性检查
                if td_count > 0:
                    td_satisfied = hasattr(self.market_state, 'td_up') and self.market_state.td_up >= td_count
                    if not td_satisfied:
                        self.log(
                            f"TD序列未满足({getattr(self.market_state, 'td_up', 0)} < {td_count})，跳过买入",
                            level="INFO"
                        )
                        return

                # 16.11 构建信号来源
                signal_source = []
                if base_buy:
                    signal_source.append("买入信号")
                if dip_signal:
                    signal_source.append("回调信号")

                # 16.12 确定是否买入
                should_buy = False
                if td_count > 0:
                    should_buy = base_buy
                    if should_buy and dip_signal:
                        signal_source.append("(回调加强)")
                else:
                    should_buy = base_buy or dip_signal

                # 16.13 执行买入
                if should_buy:
                    # 检查最小交易间隔
                    min_bars = ParamAccessor.get_param(self, 'min_bars_between_trades')
                    if hasattr(self, 'last_trade_bar'):
                        bars_since_last = len(self) - self.last_trade_bar
                        if bars_since_last < min_bars:
                            return

                    # 计算仓位大小
                    size = self.position_manager._calculate_position_size()

                    # 如果是小仓位情况，调整买入数量
                    if is_small_position:
                        # 减去现有小仓位，避免过度杠杆
                        current_position_value = self.position.size * current_price
                        available_for_new = self.broker.getvalue() * 0.9 - current_position_value
                        max_new_shares = int(available_for_new / current_price)
                        size = min(size, max_new_shares)
                        self.log(f"小仓位调整：新买入{size}股（已有{self.position.size}股）", level="INFO")

                    if size > 0:
                        self.log(f"准备买入 - 信号来源: {'+'.join(signal_source)}, 数量: {size}股", level='INFO')
                        self.order = self.buy(size=size)
                        if self.order:
                            self.entry_price = current_price
                            self.high_since_entry = current_price
                            self.entry_time = len(self)
                            self.last_trade_bar = len(self)
                            self.last_time_profit_day = 0
                            self._executed_profit_levels = set()
                            self.position_manager._last_buy_price = current_price
                            self.position_manager._buy_price_history = [{
                                'price': current_price,
                                'size': size,
                                'bar': len(self)
                            }]

        except Exception as e:
            self.log(f"策略执行错误: {str(e)}", level='ERROR')
            self.log(traceback.format_exc(), level='ERROR')
            raise

    def notify_order(self, order):
        if order.status == order.Completed:
            try:
                # 确保 trade_manager 存在
                if not hasattr(self, 'trade_manager'):
                    self.trade_manager = TradeManager()
                    self.trade_manager.set_strategy(self)

                # 获取当前交易日期
                current_date = self.data.datetime.date(0)
                if isinstance(current_date, (int, float)):
                    current_date = pd.Timestamp(datetime.datetime.fromordinal(int(current_date)))
                elif isinstance(current_date, datetime.date):
                    current_date = pd.Timestamp(current_date)

                # ============ BUY BRANCH ============
                if order.isbuy():
                    executed_size = order.executed.size
                    executed_price = order.executed.price
                    executed_comm = order.executed.comm

                    # 记录买入交易
                    trade_dict = {
                        'entry_date': current_date,
                        'entry_price': executed_price,
                        'exit_date': None,
                        'exit_price': None,
                        'size': executed_size,
                        'orig_size': executed_size,
                        'commission': executed_comm,
                        'pnl': -executed_comm,
                        'type': 'entry',
                        'status': 'open'
                    }

                    # 添加到 trade_manager
                    self.trade_manager.add_trade(trade_dict)

                    # 追踪记录
                    if not hasattr(self, '_executed_trades'):
                        self._executed_trades = []
                    self._executed_trades.append(trade_dict.copy())

                    # 继续更新策略级别信息
                    self.entry_price = order.executed.price
                    self.high_since_entry = max(self.high_since_entry, self.entry_price) if hasattr(self, 'high_since_entry') else self.entry_price
                    if not hasattr(self, 'entry_time'):
                        self.entry_time = len(self)

                    # 只保留一个简洁的日志
                    self.log(f'【 买入执行 】 - 价格: {executed_price:.2f}, '
                            f'数量: {executed_size}, 手续费: {executed_comm:.2f}', level='INFO')

                    # 发送买入通知
                    if self.notifier and self.trading_params.enable_trade_notification:
                        # 检查是否是实时交易（数据时间接近当前时间）
                        current_data_time = self.data.datetime.datetime(0)
                        now = datetime.datetime.now()
                        if (now - current_data_time).total_seconds() < 86400:  # 24小时内
                            self.notifier.send_trade_alert(
                                action="买入",
                                symbol=self.symbol,
                                price=executed_price,
                                size=executed_size
                            )

                    if hasattr(self, 'stop_manager'):
                        self.stop_manager.trailing_activated = False
                        self.stop_manager.highest_profit = 0

                # ============ SELL BRANCH ============
                else:
                    # 卖单执行
                    executed_size = abs(order.executed.size)
                    executed_price = order.executed.price
                    executed_comm = order.executed.comm

                    # 获取入场价格
                    entry_price = self.entry_price if hasattr(self, 'entry_price') and self.entry_price > 0 else executed_price

                    # 判断是否完全平仓
                    is_closing_position = (abs(self.position.size) < 1e-9)

                    # 查找对应的开仓交易
                    open_trades = [t for t in self.trade_manager.executed_trades if t.get('status') == 'open' and t.get('size', 0) > 0]

                    if open_trades:
                        # 按时间顺序处理（FIFO）
                        remaining_to_close = executed_size
                        total_pnl = 0
                        weighted_entry_price = 0
                        total_closed = 0

                        for open_trade in sorted(open_trades, key=lambda x: x['entry_date']):
                            if remaining_to_close <= 0:
                                break

                            # 计算本次平仓数量
                            close_size = min(open_trade['size'], remaining_to_close)

                            # 计算盈亏
                            trade_pnl = (executed_price - open_trade['entry_price']) * close_size
                            total_pnl += trade_pnl

                            # 更新加权平均入场价
                            weighted_entry_price += open_trade['entry_price'] * close_size
                            total_closed += close_size

                            # 更新开仓记录
                            open_trade['size'] -= close_size
                            if open_trade['size'] <= 0:
                                open_trade['status'] = 'closed'
                                open_trade['exit_date'] = current_date
                                open_trade['exit_price'] = executed_price

                            remaining_to_close -= close_size

                        # 计算平均入场价
                        if total_closed > 0:
                            avg_entry_price = weighted_entry_price / total_closed
                        else:
                            avg_entry_price = entry_price

                        # 记录平仓交易
                        closed_trade = {
                            'entry_date': getattr(self, 'entry_time', current_date),
                            'entry_price': avg_entry_price,
                            'exit_date': current_date,
                            'exit_price': executed_price,
                            'size': executed_size,
                            'commission': executed_comm,
                            'pnl': total_pnl - executed_comm,
                            'type': 'exit',
                            'status': 'closed'
                        }

                        self.trade_manager.add_trade(closed_trade)

                        # 简洁的日志输出
                        self.log(f'【 卖出执行 】 - 价格: {executed_price:.2f}, '
                                f'数量: {executed_size}, 盈亏: {closed_trade["pnl"]:+.2f}',
                                level='INFO')

                        # 在盈亏计算后添加通知
                        if self.notifier and self.trading_params.enable_trade_notification:
                            # 只有盈亏超过阈值才通知
                            if abs(closed_trade['pnl']) >= self.trading_params.notification_min_pnl:
                                # 检查是否是实时交易
                                current_data_time = self.data.datetime.datetime(0)
                                now = datetime.datetime.now()
                                if (now - current_data_time).total_seconds() < 86400:  # 24小时内
                                    self.notifier.send_trade_alert(
                                        action="卖出",
                                        symbol=self.symbol,
                                        price=executed_price,
                                        size=executed_size,
                                        pnl=closed_trade['pnl']
                                    )

                    else:
                        # 没有找到开仓记录的处理
                        simple_pnl = (executed_price - entry_price) * executed_size - executed_comm

                        closed_trade = {
                            'entry_date': getattr(self, 'entry_time', current_date),
                            'entry_price': entry_price,
                            'exit_date': current_date,
                            'exit_price': executed_price,
                            'size': executed_size,
                            'commission': executed_comm,
                            'pnl': simple_pnl,
                            'type': 'exit',
                            'status': 'closed'
                        }

                        self.trade_manager.add_trade(closed_trade)

                        self.log(f'【 卖出执行 】 - 价格: {executed_price:.2f}, '
                                f'数量: {executed_size}, 盈亏: {simple_pnl:+.2f}',
                                level='INFO')

                    # 完全平仓时重置状态
                    if is_closing_position:
                        self.log(f'完全平仓', level='INFO')
                        self.entry_price = 0
                        self.high_since_entry = 0
                        if hasattr(self, 'entry_time'):
                            delattr(self, 'entry_time')
                        if hasattr(self, 'stop_manager'):
                            self.stop_manager.trailing_activated = False
                            self.stop_manager.highest_profit = 0
                            self.stop_manager.profit_target_hits = 0

            except Exception as e:
                self.log(f"订单处理错误: {str(e)}")
                self.log(traceback.format_exc(), level='INFO')

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(f'订单失败 - 状态: {order.status}', level='INFO')

        # 最后，清除当前订单
        self.order = None

    def notify_trade(self, trade):
        """处理交易平仓通知 - 只更新统计，不重复记录"""
        if trade.isclosed:
            try:
                # 更新策略级别的统计信息
                self.total_trade_pnl = getattr(self, 'total_trade_pnl', 0) + trade.pnl
                self.last_trade_pnl = trade.pnl
                self.last_trade_commission = trade.commission

                # 记录亏损交易信息（用于风险管理）
                if trade.pnl < 0:
                    self.last_loss_day = len(self)
                    self.last_trade_pnl = trade.pnl

                    # 更新连续亏损计数
                    if hasattr(self, 'last_trade_was_loss') and self.last_trade_was_loss:
                        self.consecutive_losses = getattr(self, 'consecutive_losses', 0) + 1
                    else:
                        self.consecutive_losses = 1
                    self.last_trade_was_loss = True

                    self.log(
                        f"记录亏损交易 - 连续亏损{self.consecutive_losses}次，"
                        f"bar={self.last_loss_day}, 亏损={trade.pnl:.2f}",
                        level="CRITICAL"
                    )
                else:
                    # 盈利交易，重置连续亏损
                    self.consecutive_losses = 0
                    self.last_trade_was_loss = False

                # 更新交易管理器中的统计
                # 注意：具体的交易记录已经在notify_order中添加了
                # 这里只需要让交易管理器更新平仓交易的最终盈亏
                open_trades = [t for t in self.trade_manager.executed_trades
                              if t['status'] == 'open' and t['entry_date'] is not None]

                if open_trades and trade.pnl != 0:
                    # 找到对应的开仓交易并更新
                    # 这里可以根据时间顺序匹配，或者使用其他逻辑
                    oldest_open = min(open_trades, key=lambda x: x['entry_date'])
                    oldest_open['exit_date'] = self.data.datetime.date(0)
                    oldest_open['exit_price'] = trade.price
                    oldest_open['pnl'] = trade.pnl
                    oldest_open['status'] = 'closed'

            except Exception as e:
                self.log(f"交易统计更新失败: {str(e)}")

    # 辅助方法用于处理日期
    def _convert_date(self, date_val):
        """安全地将各种日期格式转换为 datetime.date"""
        if date_val is None:
            return None
        try:
            if isinstance(date_val, (float, int)):
                return datetime.datetime.fromordinal(int(date_val)).date()
            elif isinstance(date_val, (datetime.datetime, datetime.date)):
                return date_val.date() if isinstance(date_val, datetime.datetime) else date_val
            elif isinstance(date_val, str):
                return pd.to_datetime(date_val).date()
            elif isinstance(date_val, pd.Timestamp):
                return date_val.date()
            else:
                self.strategy.log(f"未知日期格式: {type(date_val)}", level='WARNING')
                return None
        except Exception as e:
            self.strategy.log(f"日期转换失败: {str(e)}", level='ERROR')
            return None

class ServerChanNotifier:
    """Server酱微信通知器"""

    def __init__(self, sendkey: str):
        """
        初始化通知器
        参数:
            sendkey: Server酱的SendKey
        """
        self.sendkey = sendkey
        self.base_url = "https://sctapi.ftqq.com"
        self.enabled = bool(sendkey)

    def send_message(self, title: str, content: str = "") -> bool:
        """发送消息到微信"""
        if not self.enabled:
            return False

        try:
            url = f"{self.base_url}/{self.sendkey}.send"
            data = {
                "title": title,
                "desp": content
            }

            response = requests.post(url, data=data, timeout=10)
            response.raise_for_status()

            result = response.json()
            if result.get("code") == 0:
                print(f"[通知] 消息发送成功: {title}")
                return True
            else:
                print(f"[通知] 消息发送失败: {result.get('message', 'Unknown error')}")
                return False

        except Exception as e:
            print(f"[通知] 发送消息时出错: {str(e)}")
            return False

    def send_trade_alert(self, action: str, symbol: str, price: float,
                        size: int, pnl: Optional[float] = None) -> bool:
        """发送交易提醒"""
        title = f"{symbol} - {action}"

        content = f"""
                    ### 交易详情
                    - **标的**: {symbol}
                    - **操作**: {action}
                    - **价格**: ${price:.2f}
                    - **数量**: {size}股
                    - **时间**: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                    """

        if pnl is not None:
            content += f"- **盈亏**: ${pnl:+.2f}\n"

        return self.send_message(title, content)

class RealTimeDataFeed(bt.feeds.DataBase):
    """实时数据源 - 支持Yahoo Finance和Alpha Vantage"""

    params = (
        ('symbol', 'QQQ'),
        ('use_alphavantage', True),  # 是否使用Alpha Vantage
        ('api_key', 'ZYF7PH3KNEQX341D'),  # Alpha Vantage API密钥
        ('update_interval', 30),  # 更新间隔（秒）
    )

    def __init__(self):
        super().__init__()
        self.data_queue = queue.Queue(maxsize=1000)
        self.running = False
        self.update_thread = None
        self._last_price = None
        self._last_update = None

    def start(self):
        """启动实时数据获取"""
        self.running = True
        self.update_thread = threading.Thread(target=self._fetch_data_loop)
        self.update_thread.daemon = True
        self.update_thread.start()

    def stop(self):
        """停止数据获取"""
        self.running = False
        if self.update_thread:
            self.update_thread.join()

    def _fetch_data_loop(self):
        """数据获取循环"""
        while self.running:
            try:
                # 使用Yahoo Finance获取实时数据
                if not self.p.use_alphavantage:
                    self._fetch_yahoo_data()
                else:
                    self._fetch_alphavantage_data()

                # 等待下一次更新
                time.sleep(self.p.update_interval)

            except Exception as e:
                print(f"获取数据错误: {str(e)}")
                time.sleep(10)  # 错误后等待10秒重试

    def _fetch_yahoo_data(self):
        """从Yahoo Finance获取数据"""
        try:
            import yfinance as yf
            ticker = yf.Ticker(self.p.symbol)

            # 获取最新价格
            info = ticker.info
            current_price = info.get('regularMarketPrice', 0)

            if current_price > 0:
                # 创建数据点
                data_point = {
                    'datetime': datetime.datetime.now(),
                    'open': current_price,
                    'high': current_price,
                    'low': current_price,
                    'close': current_price,
                    'volume': info.get('regularMarketVolume', 0)
                }

                # 添加到队列
                if not self.data_queue.full():
                    self.data_queue.put(data_point)

        except Exception as e:
            print(f"Yahoo Finance数据获取失败: {str(e)}")

    def _fetch_alphavantage_data(self):
        """从Alpha Vantage获取数据（备用）"""
        if not self.p.api_key:
            return

        try:
            url = f"https://www.alphavantage.co/query"
            params = {
                "function": "GLOBAL_QUOTE",
                "symbol": self.p.symbol,
                "apikey": self.p.api_key
            }

            response = requests.get(url, params=params, timeout=10)
            data = response.json()

            if "Global Quote" in data:
                quote = data["Global Quote"]
                current_price = float(quote.get("05. price", 0))

                if current_price > 0:
                    data_point = {
                        'datetime': datetime.datetime.now(),
                        'open': float(quote.get("02. open", current_price)),
                        'high': float(quote.get("03. high", current_price)),
                        'low': float(quote.get("04. low", current_price)),
                        'close': current_price,
                        'volume': int(quote.get("06. volume", 0))
                    }

                    if not self.data_queue.full():
                        self.data_queue.put(data_point)

        except Exception as e:
            print(f"Alpha Vantage数据获取失败: {str(e)}")

    def _load(self):
        """加载数据到backtrader"""
        try:
            # 尝试从队列获取数据（非阻塞）
            try:
                data = self.data_queue.get_nowait()

                # 设置数据线
                self.lines.datetime[0] = bt.date2num(data['datetime'])
                self.lines.open[0] = data['open']
                self.lines.high[0] = data['high']
                self.lines.low[0] = data['low']
                self.lines.close[0] = data['close']
                self.lines.volume[0] = data['volume']

                return True

            except queue.Empty:
                # 没有新数据时返回 False（不是 None！）
                # 这告诉 backtrader 暂时没有新数据，但数据流没有结束
                return False

        except Exception as e:
            print(f"加载数据错误: {str(e)}")
            return False

class SimpleLiveDataFeed(bt.feeds.DataBase):
    """简化的实时数据源 - 使用定时轮询"""

    lines = ('close', 'volume', 'open', 'high', 'low', 'openinterest',)

    params = (
        ('symbol', 'QQQ'),
        ('interval', 60),
        ('preload_bars', 100),  # 保留但可被 history_days 覆盖
        ('history_days', 3650),  # 新增参数，默认 10 年
    )

    def __init__(self):
        super().__init__()
        self.last_update = None
        self.current_data = None
        self.hist_data = []
        self.hist_index = 0

        # 加载历史数据作为初始数据
        self._load_initial_data()

    def _load_initial_data(self):
        """加载初始历史数据"""
        try:
            end_date = datetime.datetime.now()
            # 使用 params.history_days 动态计算起始日期
            start_date = end_date - datetime.timedelta(days=self.p.history_days)

            ticker = yf.Ticker(self.p.symbol)
            hist_data = ticker.history(
                start=start_date,
                end=end_date,
                interval='1d'
            )

            if not hist_data.empty:
                self.hist_data = []
                for idx, row in hist_data.iterrows():
                    self.hist_data.append({
                        'datetime': idx,
                        'open': row['Open'],
                        'high': row['High'],
                        'low': row['Low'],
                        'close': row['Close'],
                        'volume': row['Volume']
                    })
                print(f"加载了 {len(self.hist_data)} 条历史数据（请求范围：{self.p.history_days} 天）")
                # Initialize hist_index after successful data loading
                self.hist_index = 0
            else:
                self.hist_data = []
                self.hist_index = 0
                print("未获取到历史数据")
        except Exception as e:
            print(f"加载历史数据失败: {str(e)}")
            self.hist_data = []
            self.hist_index = 0

    def _load(self):
        """加载下一个数据点"""
        try:
            # 确保 hist_index 已初始化
            if not hasattr(self, 'hist_index'):
                self.hist_index = 0

            # 如果还有历史数据，先加载历史数据
            if hasattr(self, 'hist_data') and self.hist_data and self.hist_index < len(self.hist_data):
                data = self.hist_data[self.hist_index]
                self.hist_index += 1

                # 每100条数据打印一次进度
                if self.hist_index % 100 == 0:
                    print(f"历史数据加载进度: {self.hist_index}/{len(self.hist_data)}")

                # 设置数据线 - 添加安全检查
                try:
                    self.lines.datetime[0] = bt.date2num(data['datetime'])
                    self.lines.open[0] = data['open']
                    self.lines.high[0] = data['high']
                    self.lines.low[0] = data['low']
                    self.lines.close[0] = data['close']
                    self.lines.volume[0] = data['volume']
                    self.lines.openinterest[0] = 0
                except IndexError as e:
                    print(f"数据线分配错误: {str(e)}")
                    return False

                return True

            # 历史数据加载完毕，切换到实时模式
            if hasattr(self, 'hist_data') and self.hist_index == len(self.hist_data):
                print(f"历史数据加载完成，切换到实时模式...")
                self.hist_index += 1  # 确保只打印一次

            # 检查是否需要更新实时数据
            now = datetime.datetime.now()
            if self.last_update is None or (now - self.last_update).seconds >= self.p.interval:
                # 获取最新数据
                new_data = self._fetch_current_price()
                if new_data and self._is_trading_hours(now):
                    self.last_update = now
                    print(f"获取到实时数据: {self.p.symbol} @ ${new_data['close']:.2f}")

                    # 设置数据线 - 添加安全检查
                    try:
                        self.lines.datetime[0] = bt.date2num(new_data['datetime'])
                        self.lines.open[0] = new_data['open']
                        self.lines.high[0] = new_data['high']
                        self.lines.low[0] = new_data['low']
                        self.lines.close[0] = new_data['close']
                        self.lines.volume[0] = new_data['volume']
                        self.lines.openinterest[0] = 0
                    except IndexError as e:
                        print(f"实时数据线分配错误: {str(e)}")
                        return False

                    return True
                else:
                    # 非交易时间，每30分钟提醒一次
                    if not hasattr(self, '_last_non_trading_msg'):
                        self._last_non_trading_msg = now

                    # 每5分钟提醒一次（而不是只提醒一次）
                    if (now - self._last_non_trading_msg).seconds >= 60:  # 1分钟
                        print(f"[{now.strftime('%H:%M:%S')}] 非交易时间，等待市场开盘...")
                        self._last_non_trading_msg = now

            # 在非交易时间，定期返回一个虚拟数据点以保持系统活跃
            if not self._is_trading_hours(now):
                # 每5分钟返回一个重复的数据点
                if not hasattr(self, '_last_dummy_time'):
                    self._last_dummy_time = now

                if (now - self._last_dummy_time).seconds >= 60:  # 1分钟
                    # 使用最后的有效数据创建虚拟数据点
                    if hasattr(self, '_last_valid_data'):
                        data = self._last_valid_data.copy()
                        data['datetime'] = now

                        try:
                            self.lines.datetime[0] = bt.date2num(data['datetime'])
                            self.lines.open[0] = data['open']
                            self.lines.high[0] = data['high']
                            self.lines.low[0] = data['low']
                            self.lines.close[0] = data['close']
                            self.lines.volume[0] = data['volume']
                            self.lines.openinterest[0] = 0

                            self._last_dummy_time = now
                            return True
                        except:
                            pass

            # 没有新数据，返回False（不是None）
            return False

        except Exception as e:
            print(f"数据加载错误: {str(e)}")
            return False

    def _is_trading_hours(self, dt):
        """检查是否在美股交易时间"""
        # 简化版本：周一到周五，9:30-16:00（需要考虑时区）
        if dt.weekday() >= 5:  # 周末
            return False

        # 这里应该转换为美东时间
        hour = dt.hour
        minute = dt.minute

        # 美股交易时间：9:30 - 16:00
        if hour < 9 or hour >= 16:
            return False
        if hour == 9 and minute < 30:
            return False

        return True

    def _fetch_current_price(self):
        """获取当前价格"""
        try:
            import yfinance as yf

            ticker = yf.Ticker(self.p.symbol)
            info = ticker.info

            current_price = info.get('regularMarketPrice', 0)
            if current_price > 0:
                return {
                    'datetime': datetime.datetime.now(),
                    'open': current_price,
                    'high': current_price,
                    'low': current_price,
                    'close': current_price,
                    'volume': info.get('regularMarketVolume', 0)
                }
                # 保存最后的有效数据
                self._last_valid_data = new_data.copy()
            return None

        except Exception as e:
            # 静默处理错误，避免频繁输出
            return None

class HybridDataFeed:
    """混合数据源 - 结合历史数据和实时数据"""

    @staticmethod
    def create_feed(symbol, start_date, end_date, live_mode=False, api_key="ZYF7PH3KNEQX341D"):
        """创建数据源"""
        if live_mode:
            # 实时模式：先加载历史数据，再切换到实时
            print(f"创建实时数据源: {symbol}")

            # 创建实时数据源
            feed = RealTimeDataFeed(
                symbol=symbol,
                api_key=api_key,
                update_interval=30  # 每分钟更新
            )

            # 先加载最近30天的历史数据作为基础
            hist_start = datetime.datetime.now() - datetime.timedelta(days=30)
            hist_df = yfinance_download(symbol, hist_start.strftime("%Y-%m-%d"),
                                      end_date, interval="1d")

            # 将历史数据添加到feed
            # 这里需要特殊处理，确保历史数据能正确加载

            return feed
        else:
            # 回测模式：使用历史数据
            print(f"创建历史数据源: {symbol}")
            df = yfinance_download(symbol, start_date, end_date, interval="1d")

            # 确保列名正确
            df.columns = [col.lower() for col in df.columns]

            # 创建Pandas数据源
            feed = bt.feeds.PandasData(
                dataname=df,
                datetime='date',
                open='open',
                high='high',
                low='low',
                close='close',
                volume='volume',
                openinterest=-1
            )

            return feed

class TradeAnalyzer:
    """
    交易分析器：负责计算交易统计和生成分析报告

    功能：
    - 计算交易统计
    - 分析盈亏情况
    - 计算买入持有收益
    - 验证交易记录一致性
    """

    @staticmethod
    def calculate_statistics(strategy) -> Dict:
        """Calculate complete trading statistics"""
        # Access trades through trade_manager
        trade_manager = strategy.trade_manager

        # 初始化stats字典
        stats = {}

        # 计算组合值
        portfolio_values = np.array(strategy.portfolio_values)

        # 基础统计
        initial_value = strategy.broker.startingcash
        final_value = strategy.broker.getvalue()
        closed_trades = [t for t in trade_manager.executed_trades if t['status'] == 'closed']

        # 收益统计
        days = len(portfolio_values)
        total_return = (final_value / initial_value - 1)
        annual_return = ((1 + total_return) ** (252/days) - 1) * 100 if days > 0 else 0

        stats['收益统计'] = {
            '初始资金': initial_value,
            '最终资金': final_value,
            '净收益': final_value - initial_value,
            '收益率': total_return * 100,
            '年化收益率': annual_return,
        }

        # 交易统计
        if closed_trades:
            winning_trades = sum(1 for t in closed_trades if t['pnl'] > 0)
            total_pnl = sum(t['pnl'] for t in closed_trades)
            stats['交易统计'] = {
                '总交易次数': len(trade_manager.executed_trades),
                '胜率': (winning_trades / len(closed_trades) * 100),
                '平均每笔收益': total_pnl / len(closed_trades),
                '最大单笔收益': max((t['pnl'] for t in closed_trades), default=0),
                '最大单笔亏损': min((t['pnl'] for t in closed_trades), default=0),
            }
        else:
            stats['交易统计'] = {
                '总交易次数': 0,
                '胜率': 0,
                '平均每笔收益': 0,
                '最大单笔收益': 0,
                '最大单笔亏损': 0,
            }

        # 风险统计
        # 使用portfolio_values计算日收益率
        if len(portfolio_values) > 1:
            daily_returns = np.diff(portfolio_values) / portfolio_values[:-1]
            volatility = np.std(daily_returns) * np.sqrt(252)
            sharpe_ratio = TradeAnalyzer._calculate_sharpe_ratio(daily_returns)
            sortino_ratio = TradeAnalyzer._calculate_sortino_ratio(daily_returns)
        else:
            volatility = 0
            sharpe_ratio = 0
            sortino_ratio = 0

        max_drawdown = TradeAnalyzer._calculate_max_drawdown(strategy)

        stats['风险统计'] = {
            '最大回撤': max_drawdown * 100,
            '收益波动率': volatility * 100,
            '夏普比率': sharpe_ratio,
            'Sortino比率': sortino_ratio,
        }

        # Position information
        current_position = strategy.position.size if strategy.position else 0
        stats['持仓信息'] = {
            '当前持仓': current_position,
            '持仓成本': strategy.position.price if strategy.position else 0,
            '持仓市值': current_position * strategy.data.close[0] if current_position > 0 else 0,
        }

        # Cost statistics
        total_commission = sum(t['commission'] for t in trade_manager.executed_trades)
        stats['成本统计'] = {
            '总交易成本': total_commission,
            '成本占比': (total_commission / initial_value) * 100,
        }

        return stats

    @staticmethod
    def _calculate_max_drawdown(strategy) -> float:
        """计算最大回撤"""
        portfolio_values = np.array(strategy.portfolio_values)
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - peak) / peak
        return abs(drawdown.min()) if len(drawdown) > 0 else 0

    @staticmethod
    def _calculate_sharpe_ratio(returns: np.array, risk_free_rate: float = 0.03) -> float:
        """计算夏普比率"""
        try:
            if len(returns) == 0:
                return 0.0

            # 计算超额收益
            excess_returns = returns - risk_free_rate/252

            # 计算标准差,添加小值避免除0
            std = np.std(excess_returns) + 1e-6

            # 计算夏普比率
            sharpe = np.mean(excess_returns) / std * np.sqrt(252)

            # 限制极端值
            return max(min(sharpe, 100), -100)

        except Exception as e:
            print(f"Sharpe ratio calculation error: {str(e)}")
            raise  # 让错误暴露出来，而不是静默返回0

    @staticmethod
    def _calculate_sortino_ratio(returns: np.array, risk_free_rate: float = 0.03) -> float:
        """计算索提诺比率"""
        try:
            if len(returns) == 0:
                return 0.0

            # 计算超额收益
            excess_returns = returns - risk_free_rate/252

            # 计算下行波动率
            downside_returns = excess_returns[excess_returns < 0]
            if len(downside_returns) == 0:
                return 0.0

            # 计算下行标准差,添加小值避免除0
            downside_std = np.std(downside_returns) + 1e-6

            # 计算索提诺比率
            sortino = np.mean(excess_returns) / downside_std * np.sqrt(252)

            # 限制极端值
            return max(min(sortino, 100), -100)

        except Exception as e:
            print(f"Sortino ratio calculation error: {str(e)}")
            raise  # 让错误暴露出来，而不是静默返回0

    @staticmethod
    def calculate_buy_hold_return(df: pd.DataFrame, initial_cash: float) -> Dict:
        """
        计算买入持有策略的收益表现

        参数:
            df: 包含价格数据的DataFrame
            initial_cash: 初始投资金额

        返回:
            买入持有策略的统计指标
        """
        try:
            def get_col(dataframe, target):
                for col in dataframe.columns:
                    if str(col).lower().startswith(target.lower()):
                        return col
                raise KeyError(f"Column '{target}' not found in DataFrame columns: {dataframe.columns}")

            # 获取收盘价所在列（假设名称以 "close" 开头）
            close_col = get_col(df, 'close')
            first_valid_close = df[close_col].iloc[0]
            last_valid_close = df[close_col].iloc[-1]

            shares = int(initial_cash / first_valid_close)
            final_value = shares * last_valid_close
            total_return = ((final_value / initial_cash) - 1) * 100

            # 计算最大回撤
            cummax = df[close_col].cummax()
            drawdown = (df[close_col] - cummax) / cummax * 100
            max_drawdown = abs(drawdown.min())

            return {
                '初始价格': first_valid_close,
                '最终价格': last_valid_close,
                '持有股数': shares,
                '最终价值': final_value,
                '收益率': total_return,
                '最大回撤': max_drawdown
            }
        except Exception as e:
            print(f"计算买入持有收益时出错: {str(e)}")
            return {
                '初始价格': 0,
                '最终价格': 0,
                '持有股数': 0,
                '最终价值': 0,
                '收益率': 0,
                '最大回撤': 0
            }

class TradeVisualizer:
    """
    交易可视化器：生成交易图表和分析报告

    功能：
    - 生成K线图 (深色背景)
    - 添加交易标记
    - 显示统计信息
    - 创建交易记录表格
    """

    def __init__(
        self,
        df: pd.DataFrame,
        strategy,
        stats: Dict,
        symbol: str,
        initial_cash: float,
        buy_hold_stats: Dict
    ):
        """
        初始化可视化器

        参数：
            df: OHLCV数据
            strategy: 策略实例
            stats: 统计数据
            symbol: 交易品种代码
            initial_cash: 初始资金
        """
        self.df = df
        self.strategy = strategy
        self.stats = stats
        self.symbol = symbol
        self.initial_cash = initial_cash
        self.buy_hold_stats = buy_hold_stats
        self.broker_value = strategy.broker.getvalue()
        self.trade_manager = strategy.trade_manager  # 交易管理器

    def get_col(self, col_name: str) -> str:
        """
        按不区分大小写的方式查找 DataFrame 中对应的列名。
        如果没有精确匹配，则查找以该列名开头的列。
        """
        for col in self.df.columns:
            if str(col).lower() == col_name.lower():
                return str(col)
        for col in self.df.columns:
            if str(col).lower().startswith(col_name.lower()):
                return str(col)
        raise KeyError(f"Column '{col_name}' not found in {list(self.df.columns)}")

    def create_candlestick_chart(self) -> go.Figure:
        """
        创建K线图和交易标记，并使用统一的深色主题/背景。
        """
        import datetime

        # 1) Identify columns
        date_col = self.get_col('date')
        open_col = self.get_col('open')
        high_col = self.get_col('high')
        low_col = self.get_col('low')
        close_col = self.get_col('close')

        # 2) Ensure date is datetime.date
        if not isinstance(self.df[date_col].iloc[0], datetime.date):
            self.df[date_col] = pd.to_datetime(self.df[date_col]).dt.date

        # === Dark theme color definitions ===
        dark_bg_page    = "#1F1F22"   # intended to unify the entire “page” area
        dark_bg_plot    = "#252529"   # behind candlestick & subplots
        grid_color      = "rgba(255,255,255,0.08)"
        text_color      = "#ECECEC"
        accent_up       = "#26A69A"
        accent_down     = "#EF5350"
        table_header_bg = "rgba(255,255,255,0.05)"
        table_cells_bg  = "rgba(255,255,255,0.02)"
        table_header_font = "#FFFFFF"

        # 3) Layout with fully unified background
        layout = go.Layout(
            template=None,
            height=1000,
            # Increase top margin a bit more to ensure second line & new labels
            margin=dict(t=230, r=40, b=50, l=40),
            showlegend=True,
            # Make both "paper" (outer) and "plot" backgrounds dark
            paper_bgcolor=dark_bg_page,
            plot_bgcolor=dark_bg_plot,
            font=dict(family="Inter, Arial, sans-serif", color=text_color),
            title=None,
            annotations=[],
            xaxis=dict(
                title=None,
                showticklabels=True,
                showgrid=True,
                gridcolor=grid_color,
                zeroline=False,
                rangeslider=dict(visible=False),
                domain=[0, 1]
            ),
            yaxis=dict(
                title=None,
                showticklabels=True,
                showgrid=True,
                gridcolor=grid_color,
                zeroline=False,
                tickformat=',.2f',
                domain=[0.46, 1]
            ),
        )

        # 4) Build figure with subplots
        fig = go.Figure(layout=layout)
        fig = make_subplots(
            figure=fig,
            rows=3,
            cols=1,
            row_heights=[0.6, 0.2, 0.3],
            vertical_spacing=0.06,
            specs=[
                [{"secondary_y": False}],
                [{"type": "table"}],
                [{"type": "table"}]
            ]
        )

        # 5) Candlestick trace
        candlestick = go.Candlestick(
            x=self.df[date_col],
            open=self.df[open_col],
            high=self.df[high_col],
            low=self.df[low_col],
            close=self.df[close_col],
            name=self.symbol,
            showlegend=True,
            hoverinfo='x+y',
            increasing=dict(line=dict(color=accent_up), fillcolor=accent_up),
            decreasing=dict(line=dict(color=accent_down), fillcolor=accent_down),
            whiskerwidth=0.8,
            line=dict(width=1),
            hoverlabel=dict(
                bgcolor='#2F2F31',
                font=dict(family='Inter, Arial, sans-serif', color='#ECECEC')
            )
        )
        fig.add_trace(candlestick, row=1, col=1)

        # 6) Generate trade records + add tables
        try:
            trade_records = self._process_trade_records()

            # Add a "Transaction Records" label *above* the trade table
            # Moved up a bit more to leave a bigger gap from the chart
            fig.add_annotation(
                text="<b>交易记录</b>",
                x=0.5,
                y=0.42,  # lowered from 0.46 to 0.435 => more gap from chart
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=16, color="#CCCCCC"),
                xanchor='center',
                yanchor='bottom'
            )

            # Table of trades - with “no border” look
            table = go.Table(
                header=dict(
                    values=['Date', 'Type', 'Size', 'Price', 'Daily P&L', 'Total P&L'],
                    font=dict(size=12, color=table_header_font),
                    fill_color=table_header_bg,
                    align='left',
                    height=30,
                    line_color='rgba(255,255,255,0)'  # remove header border
                ),
                cells=dict(
                    values=[
                        [r.get('Date', 'N/A') for r in trade_records],
                        [r.get('Type', '') for r in trade_records],
                        [r.get('Size', 0) for r in trade_records],
                        ['{:.2f}'.format(float(r.get('Price', 0))) for r in trade_records],
                        ['{:.2f}'.format(float(r.get('Daily P&L', 0))) for r in trade_records],
                        ['{:.2f}'.format(float(r.get('Total P&L', 0))) for r in trade_records]
                    ],
                    align='left',
                    font=dict(size=11, color="#EDEDED"),
                    fill_color=table_cells_bg,
                    height=24,
                    line_color='rgba(255,255,255,0)'  # remove cell borders
                )
            )
            fig.add_trace(table, row=2, col=1)

            # Add buy/sell markers
            self._add_trade_markers(fig, trade_records)

            # Stats table (e.g. backtest results) with no frame
            # but let's also add a "Backtest Results" label above row=3
            fig.add_annotation(
                text="<b>回测结果</b>",
                x=0.5,
                y=0.18,  # above the bottom table
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=16, color="#CCCCCC"),
                xanchor='center',
                yanchor='bottom'
            )

            stats_table = self._create_stats_table(current_price=self.strategy.data.close[0])
            fig.add_trace(stats_table, row=3, col=1)

        except Exception as e:
            print(f"Error in creating chart: {str(e)}")
            fig.add_annotation(
                text=f"Error processing trade data: {str(e)}",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(color=accent_down, size=14)
            )

        # 7) Chart Title: main + second line
        title_main = f"<b>{self.symbol} 量化策略回测分析</b>"
        stats_line = (
            f"初始资金: &#36;{self.initial_cash:,.2f}  |  "
            f"最终资金: &#36;{self.broker_value:,.2f}  |  "
            f"总收益率: {((self.broker_value - self.initial_cash)/self.initial_cash*100):+.2f}%"
        )

        # Main title annotation
        fig.add_annotation(
            text=title_main,
            x=0.5,
            y=1.15,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(family='Inter, Arial, sans-serif', size=24, color=text_color),
            xanchor='center',
            yanchor='bottom'
        )

        # Second line annotation
        fig.add_annotation(
            text=stats_line,
            x=0.5,
            y=1.10,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(family='Inter, Arial, sans-serif', size=14, color="#AAAAAA"),
            xanchor='center',
            yanchor='bottom'
        )

        return fig

    def _create_stats_table(self, current_price: float) -> go.Table:
        """
        创建统计信息表格, 深色风格, 无边框, 轻微玻璃感
        """
        broker_value = self.strategy.broker.getvalue()
        # Partially transparent fill colors to mimic a “frosted glass” style
        row_fill_1 = 'rgba(255,255,255,0.04)'
        row_fill_2 = 'rgba(255,255,255,0.01)'

        return go.Table(
            header=dict(
                values=['指标', '数值'],
                font=dict(size=12, color='#FFFFFF'),
                fill_color='rgba(255,255,255,0.05)',
                align='left',
                height=30,
                line_color='rgba(255,255,255,0)'  # no border
            ),
            cells=dict(
                values=[
                    [
                        '=== 收益统计 ===',
                        '初始资金',
                        '最终资金',
                        '净收益',
                        '总收益率',
                        '年化收益率',
                        '=== 交易统计 ===',
                        '总交易次数',
                        '胜率',
                        '平均每笔收益',
                        '最大单笔收益',
                        '最大单笔亏损',
                        '=== 风险统计 ===',
                        '最大回撤',
                        '收益波动率(年化)',
                        '夏普比率',
                        'Sortino比率',
                        '=== 成本统计 ===',
                        '总交易成本',
                        '成本占比',
                        '=== 基准对比分析 ===',
                        '策略收益率',
                        '买入持有收益率',
                        '超额收益率',
                        '策略最大回撤',
                        '买入持有最大回撤',
                        '=== 当前持仓 ===',
                        '持仓数量',
                        '持仓成本',
                        '持仓市值',
                        '浮动盈亏'
                    ],
                    [
                        '',
                        f'${self.initial_cash:,.2f}',
                        f'${broker_value:,.2f}',
                        f'${(broker_value - self.initial_cash):,.2f}',
                        f'{(broker_value/self.initial_cash - 1)*100:.2f}%',
                        f'{self.stats["收益统计"]["年化收益率"]:.2f}%',
                        '',
                        f'{self.stats["交易统计"]["总交易次数"]}',
                        f'{self.stats["交易统计"]["胜率"]:.2f}%',
                        f'${self.stats["交易统计"]["平均每笔收益"]:,.2f}',
                        f'${self.stats["交易统计"]["最大单笔收益"]:,.2f}',
                        f'${self.stats["交易统计"]["最大单笔亏损"]:,.2f}',
                        '',
                        f'{self.stats["风险统计"]["最大回撤"]:.2f}%',
                        f'{self.stats["风险统计"]["收益波动率"]:.2f}%',
                        f'{self.stats["风险统计"]["夏普比率"]:.2f}',
                        f'{self.stats["风险统计"]["Sortino比率"]:.2f}',
                        '',
                        f'${self.stats["成本统计"]["总交易成本"]:,.2f}',
                        f'{self.stats["成本统计"]["成本占比"]:.2f}%',
                        '',
                        f'{self.stats["收益统计"]["收益率"]:.2f}%',
                        f'{self.buy_hold_stats["收益率"]:.2f}%',
                        f'{self.stats["收益统计"]["收益率"] - self.buy_hold_stats["收益率"]:.2f}%',
                        f'{self.stats["风险统计"]["最大回撤"]:.2f}%',
                        f'{self.buy_hold_stats["最大回撤"]:.2f}%',
                        '',
                        f'{self.stats["持仓信息"]["当前持仓"]:,d}股',
                        f'${self.stats["持仓信息"]["持仓成本"]:.2f}',
                        f'${self.stats["持仓信息"]["持仓市值"]:,.2f}',
                        f'${(current_price - self.stats["持仓信息"]["持仓成本"]) * self.stats["持仓信息"]["当前持仓"]:,.2f}'
                    ]
                ],
                align='left',
                font=dict(size=11, color='#E0E0E0'),
                fill_color=[
                    [row_fill_1 if i % 2 == 0 else row_fill_2 for i in range(31)],
                    [row_fill_1 if i % 2 == 0 else row_fill_2 for i in range(31)]
                ],
                height=25,
                line_color='rgba(255,255,255,0)'  # no cell borders
            )
        )

    def _process_trade_records(self) -> List[Dict]:
        """
        处理执行的交易记录, 并生成 Buy/Sell 日志
        """
        trade_records = []
        running_pnl = 0.0
        executed_trades = self.trade_manager.executed_trades or []

        for trade in executed_trades:
            if trade.get('type') == 'entry':
                b_date = trade.get('entry_date')
                if b_date is None:
                    continue
                buy_price = float(trade.get('entry_price', 0.0))
                # 如果该 entry 记录的 size 被平仓后为 0，则使用原始数量进行显示
                buy_size = trade.get('orig_size', trade.get('size', 0))
                # 新增：跳过无效的entry记录
                if buy_size <= 0 or buy_price <= 1.0:
                    continue
                # 新增：只显示未完全平仓的entry记录，或者使用原始数量
                if trade.get('status') == 'closed' and trade.get('size', 0) <= 0:
                    # 对于已完全平仓的记录，使用原始数量
                    buy_size = trade.get('orig_size', 0)
                    if buy_size <= 0:  # 如果仍然无效，跳过
                        continue

                comm = float(trade.get('commission', 0.0))
                daily_pnl = -comm  # 买入时仅记录手续费
                trade_records.append({
                    'Date': str(b_date),
                    'Type': 'B',
                    'Size': buy_size,
                    'Price': buy_price,
                    'Daily P&L': daily_pnl,
                    'Total P&L': running_pnl
                })
            elif trade.get('type') == 'exit':
                s_date = trade.get('exit_date')
                if s_date is None:
                    continue
                sell_price = float(trade.get('exit_price', 0.0))
                sell_size  = trade.get('size', 0)
                realized_pnl = float(trade.get('pnl', 0.0))
                updated_pnl  = running_pnl + realized_pnl
                trade_records.append({
                    'Date': str(s_date),
                    'Type': 'S',
                    'Size': sell_size,
                    'Price': sell_price,
                    'Daily P&L': realized_pnl,
                    'Total P&L': updated_pnl
                })
                running_pnl = updated_pnl
        # 按交易日期排序，确保买入记录出现在平仓记录之前
        trade_records.sort(key=lambda r: pd.to_datetime(r['Date']))

        return trade_records


    def _add_trade_markers(self, fig: go.Figure, trade_records: List[Dict]) -> None:
        """
        改进交易标记：买入时不显示$0.00盈亏标注；仅在卖出时显示非零P&L。
        """
        for record in trade_records:
            if record['Type'] == 'B':
                self._add_buy_marker(fig, record)
                # skip P&L annotation for buys

            elif record['Type'] == 'S':
                self._add_sell_marker(fig, record)
                if abs(record.get('Daily P&L', 0.0)) > 1e-9:
                    self._add_pnl_annotation(fig, record)

    def _add_pnl_annotation(self, fig: go.Figure, record: Dict) -> None:
        """
        在卖出位置添加盈亏标注
        """
        fig.add_annotation(
            x=record['Date'],
            y=record['Price'],
            text=f"P&L: ${record['Daily P&L']:,.2f}",
            showarrow=False,
            font=dict(size=9, color='#FFFFFF'),
            bgcolor='rgba(0, 0, 0, 0.7)',
            bordercolor='rgba(255,255,255,0.2)',
            borderwidth=1,
            yshift=-25
        )

    def _add_buy_marker(self, fig: go.Figure, record: Dict) -> None:
        fig.add_trace(
            go.Scatter(
                x=[record['Date']],
                y=[record['Price']],
                mode='markers',
                marker=dict(symbol='triangle-up', size=12, color='#0ECB81'),
                name='Buy',
                showlegend=False
            ),
            row=1, col=1
        )
        fig.add_annotation(
            x=record['Date'],
            y=record['Price'],
            xref="x1",
            yref="y1",
            text=f"B<br>Qty: {record['Size']}",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=1.5,
            arrowcolor='#0ECB81',
            font=dict(color='#0ECB81', size=10),
            yshift=-15
        )

    def _add_sell_marker(self, fig: go.Figure, record: Dict) -> None:
        fig.add_trace(
            go.Scatter(
                x=[record['Date']],
                y=[record['Price']],
                mode='markers',
                marker=dict(symbol='triangle-down', size=12, color='#E02F44'),
                name='Sell',
                showlegend=False
            ),
            row=1, col=1
        )
        fig.add_annotation(
            x=record['Date'],
            y=record['Price'],
            xref="x1",
            yref="y1",
            text=f"S<br>Qty: {record['Size']}",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=1.5,
            arrowcolor='#E02F44',
            font=dict(color='#E02F44', size=10),
            yshift=15
        )

def alphavantage_download(symbol: str, start: str, end: str, interval: str = "1d",
                    prepost: bool = True, api_key: str = None) -> pd.DataFrame:
    try:
        if not api_key:
            raise ValueError("需要提供Alpha Vantage API密钥")

        print(f"使用Alpha Vantage获取 {symbol} 数据...")

        # Alpha Vantage API基础URL
        base_url = "https://www.alphavantage.co/query"

        # 根据interval选择合适的API函数
        if interval in ["1h", "1H"]:
            function = "TIME_SERIES_INTRADAY"
            interval_param = "60min"
            time_key = "Time Series (60min)"
        elif interval in ["30m"]:
            function = "TIME_SERIES_INTRADAY"
            interval_param = "30min"
            time_key = "Time Series (30min)"
        elif interval in ["15m"]:
            function = "TIME_SERIES_INTRADAY"
            interval_param = "15min"
            time_key = "Time Series (15min)"
        elif interval in ["5m"]:
            function = "TIME_SERIES_INTRADAY"
            interval_param = "5min"
            time_key = "Time Series (5min)"
        elif interval in ["1m"]:
            function = "TIME_SERIES_INTRADAY"
            interval_param = "1min"
            time_key = "Time Series (1min)"
        else:
            # 默认使用日线数据
            function = "TIME_SERIES_DAILY"
            time_key = "Time Series (Daily)"
            interval_param = None

        # 构建API请求参数
        params = {
            "function": function,
            "symbol": symbol,
            "apikey": api_key,
            "outputsize": "full",  # 获取完整数据
            "datatype": "json"
        }

        # 如果是分钟数据，添加interval参数
        if interval_param:
            params["interval"] = interval_param

        print(f"API请求: function={function}, symbol={symbol}")

        # 发送API请求
        response = requests.get(base_url, params=params, timeout=30)
        response.raise_for_status()

        data = response.json()
        # 在后面添加这几行：
        print(f"API返回的键: {list(data.keys())}")
        if "Information" in data:
            print(f"Information内容: {data['Information']}")

        # 检查错误响应
        if "Error Message" in data:
            raise Exception(f"Alpha Vantage API错误: {data['Error Message']}")

        if "Note" in data:
            print("⚠️  API调用频率限制提醒：Alpha Vantage免费版每分钟5次，每天500次")
            # 等待一段时间后重试
            print("等待15秒后重试...")
            time.sleep(15)
            response = requests.get(base_url, params=params, timeout=30)
            data = response.json()

            if "Note" in data:
                raise Exception(f"API调用频率超限: {data['Note']}")

        if time_key not in data:
            available_keys = list(data.keys())
            raise Exception(f"数据格式错误，找不到键: {time_key}，可用键: {available_keys}")

        # 转换数据格式
        time_series = data[time_key]

        if not time_series:
            raise Exception("返回的时间序列数据为空")

        df_data = []
        for date_str, values in time_series.items():
            try:
                df_data.append({
                    'Date': pd.to_datetime(date_str),
                    'Open': float(values['1. open']),
                    'High': float(values['2. high']),
                    'Low': float(values['3. low']),
                    'Close': float(values['4. close']),
                    'Volume': int(values['5. volume'])
                })
            except (ValueError, KeyError) as e:
                print(f"跳过无效数据行: {date_str}, 错误: {e}")
                continue

        if not df_data:
            raise Exception("没有有效的数据行")

        df = pd.DataFrame(df_data)
        df.set_index('Date', inplace=True)
        df.sort_index(inplace=True)

        # 移除重复的时间索引
        df = df[~df.index.duplicated(keep='first')]

        # 筛选日期范围
        if start:
            start_date = pd.to_datetime(start)
            df = df[df.index >= start_date]

        if end:
            end_date = pd.to_datetime(end)
            df = df[df.index <= end_date]

        if df.empty:
            raise Exception(f"在指定日期范围 {start} 到 {end} 内没有数据")

        # 重置索引以获得Date列（保持与原函数兼容）
        df.reset_index(inplace=True)

        print(f"成功获取 {len(df)} 条 {symbol} 数据记录")
        print(f"日期范围: {df['Date'].min().strftime('%Y-%m-%d')} 到 {df['Date'].max().strftime('%Y-%m-%d')}")

        return df

    except Exception as e:
        print(f"Alpha Vantage数据获取失败: {str(e)}")
        print("请检查：")
        print("1. API密钥是否有效")
        print("2. 股票代码是否正确")
        print("3. 网络连接是否正常")
        print("4. 是否超过API调用限制（免费版：5次/分钟，500次/天）")
        raise e

def yfinance_download(symbol: str, start: str, end: str, interval: str = "1d") -> pd.DataFrame:
    """
    使用yfinance获取股票数据作为备用方案
    """
    try:
        print(f"使用YFinance获取 {symbol} 数据...")

        yf_interval = {
            "1d": "1d",
            "1h": "1h",
            "30m": "30m",
            "15m": "15m",
            "5m": "5m",
            "1m": "1m"
        }.get(interval, "1d")

        ticker = yf.Ticker(symbol)
        df = ticker.history(
            start=start,
            end=end,
            interval=yf_interval,
            auto_adjust=True,
            prepost=True
        )

        if df.empty:
            raise Exception(f"YFinance未返回{symbol}的数据，可能原因：无效代码、日期范围错误或网络问题")

        df.reset_index(inplace=True)
        df.columns = [str(col).title() for col in df.columns]

        required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise Exception(f"YFinance数据缺少必需列: {missing_cols}")

        print(f"YFinance成功获取 {len(df)} 条 {symbol} 数据记录")
        print(f"日期范围: {df['Date'].min().strftime('%Y-%m-%d')} 到 {df['Date'].max().strftime('%Y-%m-%d')}")
        return df

    except Exception as e:
        print(f"YFinance数据获取失败: {str(e)}")
        print("请检查：")
        print("1. 股票代码是否正确")
        print("2. 日期范围是否有效")
        print("3. 网络连接是否正常")
        print("4. YFinance库是否需要更新")
        raise e

def twelvedata_download(symbol: str, start: str, end: str, interval: str = "1d", api_key: str = None) -> pd.DataFrame:
    """
    使用Twelve Data API获取股票数据
    免费tier: 每天800次请求，每分钟8次
    """
    try:
        print(f"使用Twelve Data获取 {symbol} 数据...")

        # 如果没有提供API密钥，使用demo key
        if not api_key or api_key == "4e06770f76fe42b9bc3b6760b14118f6":
            print("⚠️  使用Twelve Data demo密钥，功能有限")
            api_key = "demo"

        # Twelve Data API基础URL
        base_url = "https://api.twelvedata.com/time_series"

        # 映射interval格式
        interval_map = {
            "1d": "1day",
            "1h": "1h",
            "30m": "30min",
            "15m": "15min",
            "5m": "5min",
            "1m": "1min"
        }
        td_interval = interval_map.get(interval, "1day")

        # 构建请求参数
        params = {
            "symbol": symbol,
            "interval": td_interval,
            "start_date": start,
            "end_date": end,
            "format": "JSON",
            "apikey": api_key
        }

        # 发送请求
        response = requests.get(base_url, params=params, timeout=30)
        response.raise_for_status()

        data = response.json()

        # 检查错误
        if "status" in data and data["status"] == "error":
            raise Exception(f"Twelve Data API错误: {data.get('message', 'Unknown error')}")

        if "values" not in data:
            raise Exception("Twelve Data返回数据格式错误")

        # 转换数据
        df_data = []
        for item in data["values"]:
            df_data.append({
                'Date': pd.to_datetime(item['datetime']),
                'Open': float(item['open']),
                'High': float(item['high']),
                'Low': float(item['low']),
                'Close': float(item['close']),
                'Volume': int(item.get('volume', 0))
            })

        if not df_data:
            raise Exception("没有有效的数据行")

        df = pd.DataFrame(df_data)
        df.sort_values('Date', inplace=True)
        df.reset_index(drop=True, inplace=True)

        print(f"Twelve Data成功获取 {len(df)} 条 {symbol} 数据记录")
        print(f"日期范围: {df['Date'].min().strftime('%Y-%m-%d')} 到 {df['Date'].max().strftime('%Y-%m-%d')}")

        return df

    except Exception as e:
        print(f"Twelve Data数据获取失败: {str(e)}")
        raise e


def yahoo_direct_download(symbol: str, start: str, end: str, interval: str = "1d") -> pd.DataFrame:
    """
    直接使用Yahoo Finance HTTP API获取数据（绕过yfinance库）
    不需要API密钥，但需要模拟正常浏览器行为
    """
    try:
        print(f"使用Yahoo Finance直接API获取 {symbol} 数据...")

        # 添加随机延迟，避免被识别为爬虫
        delay = random.uniform(1.0, 3.0)  # 1-3秒随机延迟
        print(f"等待 {delay:.1f} 秒...")
        time.sleep(delay)

        # 转换日期为时间戳
        start_ts = int(pd.Timestamp(start).timestamp())
        end_ts = int(pd.Timestamp(end).timestamp())

        # 构建URL（保持原有代码）
        interval_map = {
            "1d": "1d",
            "1h": "1h",
            "30m": "30m",
            "15m": "15m",
            "5m": "5m",
            "1m": "1m"
        }
        yf_interval = interval_map.get(interval, "1d")

        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        params = {
            'period1': start_ts,
            'period2': end_ts,
            'interval': yf_interval,
            'includePrePost': 'true',
            'events': 'div|split|earn'
        }

        # 使用随机User-Agent，更好地模拟真实浏览器
        user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0'
        ]

        headers = {
            'User-Agent': random.choice(user_agents),
            'Accept': 'application/json',
            'Accept-Language': 'en-US,en;q=0.9',
            'Cache-Control': 'no-cache',
            'Pragma': 'no-cache'
        }

        # 发送请求
        response = requests.get(url, params=params, headers=headers, timeout=30)
        response.raise_for_status()

        data = response.json()

        # 解析数据
        result = data['chart']['result'][0]
        timestamps = result['timestamp']
        quote = result['indicators']['quote'][0]

        # 构建DataFrame
        df = pd.DataFrame({
            'Date': pd.to_datetime(timestamps, unit='s'),
            'Open': quote['open'],
            'High': quote['high'],
            'Low': quote['low'],
            'Close': quote['close'],
            'Volume': quote['volume']
        })

        # 清理空值
        df.dropna(inplace=True)
        df.reset_index(drop=True, inplace=True)

        print(f"Yahoo Direct成功获取 {len(df)} 条 {symbol} 数据记录")
        print(f"日期范围: {df['Date'].min().strftime('%Y-%m-%d')} 到 {df['Date'].max().strftime('%Y-%m-%d')}")

        return df

    except Exception as e:
        print(f"Yahoo Direct数据获取失败: {str(e)}")
        raise e

class SimpleTradingMonitor:
    """简单的交易监控系统"""

    def __init__(self, cerebro, strategy_params, notifier=None):
        self.cerebro = cerebro
        self.strategy_params = strategy_params
        self.strategy = None  # 初始化为Null，后续设置
        self.notifier = notifier
        self.is_running = False
        self.monitor_thread = None

    def start(self):
        """启动监控"""
        self.is_running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()

        if self.notifier:
            self.notifier.send_message(
                "交易系统启动",
                f"开始监控 {self.strategy_params.symbol}"
            )

    def stop(self):
        """停止监控"""
        self.is_running = False
        if self.monitor_thread:
            self.monitor_thread.join()

        if self.notifier:
            self.notifier.send_message(
                "交易系统停止",
                "监控已结束"
            )

    def _monitor_loop(self):
        """监控循环"""
        check_interval = 30
        last_position_check = None
        error_count = 0  # 添加错误计数器
        max_errors = 3   # 最大错误次数

        while self.is_running:
            try:
                now = datetime.datetime.now()

                # 获取策略状态 - 修正：使用 is not None 判断
                if hasattr(self, 'strategy') and self.strategy is not None:
                    # 更安全的属性访问
                    try:
                        # 使用getattr安全获取broker
                        broker = getattr(self.strategy, 'broker', None)
                        if broker is not None:
                            portfolio_value = broker.getvalue()
                            cash = broker.get_cash()
                        else:
                            portfolio_value = 0
                            cash = 0

                        # 安全获取position
                        position = getattr(self.strategy, 'position', None)

                        # 确保data存在并且有数据
                        current_price = 0.0
                        if hasattr(self.strategy, 'data'):
                            data = self.strategy.data
                            if hasattr(data, 'close') and len(data.close) > 0:
                                try:
                                    current_price = data.close[0]
                                except:
                                    pass

                        # 重置错误计数
                        error_count = 0

                        # 检查持仓变化
                        current_position = 0
                        if position is not None:
                            try:
                                current_position = position.size
                            except:
                                current_position = 0

                        if last_position_check != current_position:
                            print(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] 持仓变化: {last_position_check} → {current_position}")
                            if self.notifier:
                                self.notifier.send_message(
                                    "持仓变化",
                                    f"当前持仓: {current_position}股\n账户价值: ${portfolio_value:,.2f}"
                                )
                            last_position_check = current_position

                        # 定期状态报告（每30分钟）
                        if not hasattr(self, '_last_full_report') or (now - self._last_full_report).seconds >= 1800:
                            status_msg = f"""
    [{now.strftime('%Y-%m-%d %H:%M:%S')}] 系统状态报告
    - 账户总值: ${portfolio_value:,.2f}
    - 可用现金: ${cash:,.2f}
    - 当前持仓: {current_position}股
    - 当前价格: ${current_price:.2f}
    - 交易时间: {'是' if self._is_trading_hours(now) else '否'}
    """
                            print(status_msg)
                            self._last_full_report = now

                    except Exception as e:
                        # 内部错误，不发送通知
                        if 'addindicator' not in str(e) and 'nonzero' not in str(e):  # 忽略特定错误
                            print(f"监控数据访问错误: {str(e)}")
                else:
                    print(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] 等待策略初始化...")

                time.sleep(check_interval)

            except Exception as e:
                error_count += 1
                print(f"监控错误 ({error_count}/{max_errors}): {str(e)}")

                # 只有在错误次数达到阈值时才发送通知
                if error_count >= max_errors and self.notifier:
                    self.notifier.send_message("监控异常", f"连续{error_count}次错误，最后错误: {str(e)}")
                    error_count = 0  # 重置计数器，避免重复发送

    def _is_trading_hours(self, dt):
        """检查是否在交易时间"""
        # 简化版本，实际需要考虑美东时间和节假日
        weekday = dt.weekday()
        if weekday >= 5:  # 周末
            return False

        hour = dt.hour
        return 9 <= hour <= 16

def _get_next_market_open(current_time):
    """计算下一个美股开盘时间（简化版）"""
    # 这是简化版本，实际应该考虑时区和节假日
    current_hour = current_time.hour
    current_weekday = current_time.weekday()

    # 如果是周末
    if current_weekday >= 5:  # 周六或周日
        days_to_monday = 7 - current_weekday
        next_open_date = current_time + datetime.timedelta(days=days_to_monday)
        return f"{next_open_date.strftime('%Y-%m-%d')} 09:30 EST"

    # 如果是工作日
    if current_hour < 9 or (current_hour == 9 and current_time.minute < 30):
        # 今天还没开盘
        return f"{current_time.strftime('%Y-%m-%d')} 09:30 EST"
    elif current_hour >= 16:
        # 今天已收盘
        if current_weekday == 4:  # 周五
            # 下周一开盘
            next_open_date = current_time + datetime.timedelta(days=3)
        else:
            # 明天开盘
            next_open_date = current_time + datetime.timedelta(days=1)
        return f"{next_open_date.strftime('%Y-%m-%d')} 09:30 EST"
    else:
        # 正在交易时间
        return "当前正在交易时间"

def main():
    """主执行函数"""
    try:
        # 获取交易参数，支持默认值
        symbol = input("输入股票代码（例如：QQQ） [默认: QQQ]: ").strip().upper() or "QQQ"
        initial_cash_str = input("输入初始资金（例如：100000） [默认: 100000]: ").strip() or "100000"
        initial_cash = float(initial_cash_str)

        # 新增：询问是否为实时模式
        live_mode_input = input("是否启用实时交易模式? y/n (默认n): ").strip().lower()
        live_mode = (live_mode_input == 'y')


        serverchan_key = "SCT119824TW3DsGlBjkhAV9lJWIOLohv1P"

        # 临时直接设置API密钥（替换为您的实际密钥）
        ALPHA_VANTAGE_API_KEY = "ZYF7PH3KNEQX341D"
        TWELVE_DATA_API_KEY = "4e06770f76fe42b9bc3b6760b14118f6"
        api_key = ALPHA_VANTAGE_API_KEY

        if not api_key or api_key == "YOUR_ACTUAL_API_KEY_HERE":
            api_key = input("请输入您的Alpha Vantage API密钥: ").strip()

        # 日期设置
        generate_live_report = False  # 默认值
        if live_mode:
            # 实时模式：允许用户指定历史数据长度
            history_days_input = input("输入历史数据天数（默认3650天）: ").strip() or "3650"
            history_days = int(history_days_input)

            end_date = datetime.datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.datetime.now() - datetime.timedelta(days=history_days)).strftime("%Y-%m-%d")
            print(f"\n实时模式：加载从 {start_date} 到 {end_date} 的历史数据")

            # 询问是否在实时运行后生成回测报告
            generate_report = input("实时运行结束后是否生成历史回测报告? y/n (默认y): ").strip().lower() or "y"
            generate_live_report = (generate_report == 'y')

            interval_input = "1d"  # 实时模式默认使用日线
            include_prepost = True
        else:
            # 可选：指定日期范围
            start_input = input("输入开始日期(YYYY-MM-DD)，或者直接回车使用默认: ").strip()
            end_input = input("输入结束日期(YYYY-MM-DD)，或者直接回车使用默认: ").strip()

            # 可选：数据间隔及是否包含盘前/盘后数据（默认1d和y）
            interval_input = input("输入数据间隔(如'1d','1h','15m')，默认'1d': ").strip() or "1d"
            prepost_input = input("是否包含盘前/盘后交易? y/n (默认y): ").strip().lower() or "y"
            include_prepost = (prepost_input == 'y')

            start_date = start_input if start_input else None
            end_date = end_input if end_input else None

        # 创建参数对象 - 只指定用户输入的基本参数，其他使用类中定义的默认值
        trading_params = TradingParameters(
            symbol=symbol,
            initial_cash=initial_cash,
            start_date=start_date,
            end_date=end_date,
            # 新增参数
            live_mode=live_mode,
            serverchan_sendkey=serverchan_key,
            enable_trade_notification=bool(serverchan_key),
            enable_risk_notification=bool(serverchan_key),
            generate_live_report=generate_live_report
            # 其他参数使用TradingParameters类中定义的默认值
        )
        trading_params.validate()

        # 创建通知器（提前创建以便在数据下载阶段也能使用）
        notifier = None
        if trading_params.serverchan_sendkey:
            notifier = ServerChanNotifier(trading_params.serverchan_sendkey)
            print("Server酱通知已启用")

        # ===== 数据获取部分 =====
        df = None
        df_feed = None

        if live_mode:
            # 实时模式：创建混合数据源
            print(f"\n创建实时数据源 {symbol}...")

            # 先获取历史数据作为基础
            print(f"下载最近30天的历史数据...")

            # 使用现有的多源API机制获取历史数据
            api_sources = [
                ("Alpha Vantage", lambda: alphavantage_download(
                    symbol=trading_params.symbol,
                    start=trading_params.start_date,
                    end=trading_params.end_date,
                    interval=interval_input,
                    prepost=include_prepost,
                    api_key=ALPHA_VANTAGE_API_KEY
                )),
                ("Twelve Data", lambda: twelvedata_download(
                    symbol=trading_params.symbol,
                    start=trading_params.start_date,
                    end=trading_params.end_date,
                    interval=interval_input,
                    api_key=TWELVE_DATA_API_KEY
                )),
                ("Yahoo Direct", lambda: yahoo_direct_download(
                    symbol=trading_params.symbol,
                    start=trading_params.start_date,
                    end=trading_params.end_date,
                    interval=interval_input
                )),
                ("YFinance", lambda: yfinance_download(
                    symbol=trading_params.symbol,
                    start=trading_params.start_date,
                    end=trading_params.end_date,
                    interval=interval_input
                ))
            ]

            # 尝试获取历史数据
            successful_source = None
            for i, (source_name, download_func) in enumerate(api_sources):
                try:
                    print(f"\n尝试使用 {source_name}...")
                    if i > 0:
                        delay = random.uniform(0.5, 1.5)
                        print(f"切换API前等待 {delay:.1f} 秒...")
                        time.sleep(delay)

                    df = download_func()
                    if df is not None and not df.empty:
                        successful_source = source_name
                        print(f"✅ {source_name} 数据获取成功！")
                        break

                except Exception as e:
                    print(f"❌ {source_name} 失败: {str(e)}")
                    continue

            if df is None or df.empty:
                raise ValueError(f"未能获取 {trading_params.symbol} 的历史数据")

            # 创建实时数据源
            realtime_feed = RealTimeDataFeed(
                symbol=trading_params.symbol,
                use_alphavantage=(successful_source == "Alpha Vantage"),
                api_key=ALPHA_VANTAGE_API_KEY if successful_source == "Alpha Vantage" else "",
                update_interval=60  # 每分钟更新
            )

            # 启动实时数据获取
            realtime_feed.start()

            # 数据处理逻辑保持不变
            df.reset_index(inplace=True)

        else:
            # ===== 使用多源API自动切换机制 =====
            print(f"\n下载 {symbol} 的历史数据...")

            # API源列表，按优先级排序
            api_sources = [
                ("Alpha Vantage", lambda: alphavantage_download(
                    symbol=trading_params.symbol,
                    start=trading_params.start_date,
                    end=trading_params.end_date,
                    interval=interval_input,
                    prepost=include_prepost,
                    api_key=ALPHA_VANTAGE_API_KEY
                )),
                ("Twelve Data", lambda: twelvedata_download(
                    symbol=trading_params.symbol,
                    start=trading_params.start_date,
                    end=trading_params.end_date,
                    interval=interval_input,
                    api_key=TWELVE_DATA_API_KEY
                )),
                ("Yahoo Direct", lambda: yahoo_direct_download(
                    symbol=trading_params.symbol,
                    start=trading_params.start_date,
                    end=trading_params.end_date,
                    interval=interval_input
                )),
                ("YFinance", lambda: yfinance_download(
                    symbol=trading_params.symbol,
                    start=trading_params.start_date,
                    end=trading_params.end_date,
                    interval=interval_input
                ))
            ]

            # 尝试每个API源
            df = None
            successful_source = None

            for i, (source_name, download_func) in enumerate(api_sources):
                try:
                    print(f"\n尝试使用 {source_name}...")

                    # 如果不是第一个源，添加延迟
                    if i > 0:
                        delay = random.uniform(0.5, 1.5)
                        print(f"切换API前等待 {delay:.1f} 秒...")
                        time.sleep(delay)

                    df = download_func()
                    if df is not None and not df.empty:
                        successful_source = source_name
                        print(f"✅ {source_name} 数据获取成功！")
                        break

                except Exception as e:
                    print(f"❌ {source_name} 失败: {str(e)}")
                    if "rate limit" in str(e).lower() or "too many requests" in str(e).lower():
                        print("  提示: API额度已用完，尝试下一个数据源...")
                    continue

        if df is None or df.empty:
            raise ValueError(f"未找到股票代码 {trading_params.symbol} 的数据")

        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"数据下载错误: 期望 pandas DataFrame，得到 {type(df)}")

        df.reset_index(inplace=True)

        # 定义 fix_col() 函数：将列名转换为小写字符串；若为元组则用下划线连接非空部分
        def fix_col(col):
            if isinstance(col, tuple):
                if len(set(col)) == 1:
                    return str(col[0]).lower()
                else:
                    col_parts = [str(x) for x in col if x and str(x).strip() != '']
                    return '_'.join(col_parts).lower() if col_parts else str(col).lower()
            else:
                return str(col).lower()

        # 如果没有 'date' 列，则检查 'datetime' 或 'Date' 列，否则用索引创建日期列
        if 'date' not in df.columns:
            if 'datetime' in df.columns:
                df.rename(columns={'datetime': 'date'}, inplace=True)
            elif 'Date' in df.columns:
                df.rename(columns={'Date': 'date'}, inplace=True)
            elif 'Datetime' in df.columns:
                df.rename(columns={'Datetime': 'date'}, inplace=True)
            else:
                df['date'] = df.index

        # 将所有列名转换为小写，并去除重复的列
        df.columns = [fix_col(c) for c in df.columns]
        df = df.loc[:, ~df.columns.duplicated()]

        # 去除列名后缀（例如 TSLA 返回的列可能为 "open_tsla"），以当前 symbol 生成后缀
        ticker_suffix = "_" + trading_params.symbol.lower()
        df.columns = [col[:-len(ticker_suffix)] if col.endswith(ticker_suffix) else col for col in df.columns]

        # 检查必需列是否存在
        required = ['date', 'open', 'high', 'low', 'close', 'volume']
        missing = [col for col in required if col not in df.columns]
        if missing:
            if 'close' in missing and 'adj close' in df.columns:
                df.rename(columns={'adj close': 'close'}, inplace=True)
                missing = [col for col in required if col not in df.columns]
            if missing:
                raise KeyError(f"{missing} not in index. Available columns: {df.columns.tolist()}")

        # 当使用小时/分钟级别数据时的特殊处理
        is_intraday = interval_input.endswith('h') or interval_input.endswith('m')
        if is_intraday:
            print("\n执行小时/分钟级别数据的时间修复...")

            # 将日期列转换为datetime类型
            df['date'] = pd.to_datetime(df['date'])

            # 如果有时区信息，移除它
            if hasattr(df['date'].iloc[0], 'tz') and df['date'].iloc[0].tz is not None:
                print("检测到时区信息，正在移除...")
                df['date'] = df['date'].dt.tz_localize(None)

            # 检查是否为1970-01-01
            if df['date'].iloc[0].year == 1970:
                print("警告: 检测到无效日期 (1970年)，尝试修复...")
                # 假设索引是ordinal日期，尝试转换
                if isinstance(df.index, pd.DatetimeIndex):
                    df['date'] = df.index
                    print("使用DataFrame索引作为日期替代")

            # 再次检查
            if df['date'].iloc[0].year == 1970:
                print("警告: 日期仍然是1970年，这可能会导致回测出现问题")
                print("提示: 尝试使用其他数据源或不同的时间间隔")

            # 输出修复后的日期样本
            print("时间修复后的前5行日期样本:")
            for i in range(min(5, len(df))):
                print(f"  {i+1}. {df['date'].iloc[i]}")
        else:
            # 如果是日级别数据，确保日期列是datetime类型
            df['date'] = pd.to_datetime(df['date'])

        # 只保留必要的列
        df = df[required]

        # 为 Backtrader 生成数据 df_bt
        df_bt = df.copy()
        # 原始数据用于可视化
        df_viz = df.copy()

        qqq_df_bt = None
        qqq_df_viz = None

        # 设置 Cerebro
        cerebro = bt.Cerebro()
        # 传递trading_params给策略
        cerebro.addstrategy(EnhancedStrategy, trading_params=trading_params)

        # 设置正确的时间框架和压缩率
        timeframe = bt.TimeFrame.Days  # 默认为天
        compression = 1  # 默认压缩率

        # 根据时间间隔设置正确的参数
        if interval_input.endswith('m'):
            timeframe = bt.TimeFrame.Minutes
            compression = int(interval_input[:-1])
            print(f"设置时间框架: Minutes, 压缩率: {compression}")
        elif interval_input.endswith('h'):
            # Backtrader使用Minutes表示小时，将小时转换为分钟
            timeframe = bt.TimeFrame.Minutes
            compression = int(interval_input[:-1]) * 60  # 每小时60分钟
            print(f"设置时间框架: Minutes (小时级别), 压缩率: {compression}")
        else:
            # 默认为日级别
            print(f"设置时间框架: Days, 压缩率: {compression}")

        # 创建数据源
        if live_mode:
            df_feed = SimpleLiveDataFeed(
                symbol=trading_params.symbol,
                interval=60,
                history_days=trading_params.history_days  # 使用 trading_params 中的值
            )

            # 注意：SimpleLiveDataFeed会自动加载历史数据
            print("实时数据源已创建")
        else:
            # 回测模式：使用历史数据
            df_feed = bt.feeds.PandasData(
                dataname=df_bt,
                datetime=0,    # 日期列索引
                open=1,
                high=2,
                low=3,
                close=4,
                volume=5,
                openinterest=-1,
                timeframe=timeframe,
                compression=compression
            )

        cerebro.adddata(df_feed)

        cerebro.broker.setcommission(commission=trading_params.commission)
        cerebro.broker.set_slippage_fixed(trading_params.slippage)
        cerebro.broker.setcash(trading_params.initial_cash)

        # 实时模式：创建监控系统
        monitor = None
        if live_mode:
            monitor = SimpleTradingMonitor(cerebro, trading_params, notifier)
            # monitor.start() 将在 cerebro.run() 之后调用
            print("\n实时交易系统已创建，等待启动...")

        strategy = None
        strategies = None  # 新增变量存储策略列表
        try:
            print("\n运行回测..." if not live_mode else "\n运行实时交易...")
            strategies = cerebro.run()  # 先保存到strategies
            if strategies and len(strategies) > 0:
                strategy = strategies[0]

            if live_mode and monitor:
                # 设置策略引用并启动监控
                monitor.strategy = strategy
                # 延迟启动监控，确保策略完全初始化
                def delayed_monitor_start():
                    time.sleep(2)  # 等待2秒
                    # 不能直接用 if monitor.strategy，因为backtrader重载了__bool__
                    if hasattr(monitor, 'strategy') and monitor.strategy is not None:
                        monitor.start()
                        print("\n实时交易监控已启动，按Ctrl+C停止...")
                    else:
                        print("\n警告：策略未正确初始化，监控未启动")

                # 在新线程中延迟启动
                import threading
                start_thread = threading.Thread(target=delayed_monitor_start)
                start_thread.daemon = True
                start_thread.start()

                if notifier:
                    notifier.send_message(
                        "实时交易系统启动",
                        f"开始监控 {symbol}\n初始资金: ${initial_cash:,.2f}"
                    )

                # 实时模式：保持运行
                last_status_time = datetime.datetime.now()
                status_interval = 60
                while True:
                    try:
                        now = datetime.datetime.now()

                        # 定期输出状态信息
                        if (now - last_status_time).seconds >= status_interval:
                            # 检查是否在交易时间
                            is_trading = monitor._is_trading_hours(now) if monitor else False
                            status_msg = f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] 系统运行中 - "
                            if is_trading:
                                status_msg += "交易时间"
                            else:
                                # 计算下一个交易日开盘时间
                                next_open = _get_next_market_open(now)
                                status_msg += f"非交易时间，等待开盘 (预计: {next_open})"

                            print(status_msg)
                            last_status_time = now

                        time.sleep(60)  # 每分钟检查一次

                    except KeyboardInterrupt:
                        raise  # 向上传递中断信号
                    except Exception as e:
                        print(f"主循环错误: {str(e)}")
                        time.sleep(60)

        except KeyboardInterrupt:
            print("\n收到停止信号...")
            # 尝试从cerebro获取策略实例
            if strategy is None and cerebro._brokers:
                try:
                    # 从cerebro的内部结构获取策略
                    for strat in cerebro.strats:
                        if strat and hasattr(strat, '_analyzers'):
                            strategy = strat
                            break
                except:
                    pass
        except Exception as e:
            print(f"\n发生错误: {str(e)}")

        finally:
            # 清理资源
            if live_mode:
                if hasattr(df_feed, 'stop'):
                    df_feed.stop()
                if monitor:
                    monitor.stop()
                if notifier:
                    notifier.send_message(
                        "实时交易系统停止",
                        "系统已安全关闭"
                    )

                # 生成历史回测报告
                if trading_params.generate_live_report:
                        if strategy is not None:
                            print("\n生成历史回测报告...")
                            stats = TradeAnalyzer.calculate_statistics(strategy)
                            buy_hold_stats = TradeAnalyzer.calculate_buy_hold_return(df_viz, initial_cash)

                            print("\n" + "=" * 20 + " 策略回测详细报告 " + "=" * 20)
                            print("\n=== 收益统计 ===")
                            print(f"初始资金: ${stats['收益统计']['初始资金']:,.2f}")
                            print(f"最终资金: ${stats['收益统计']['最终资金']:,.2f}")
                            print(f"净收益: ${stats['收益统计']['净收益']:,.2f}")
                            print(f"总收益率: {stats['收益统计']['收益率']:.2f}%")
                            print(f"年化收益率: {stats['收益统计']['年化收益率']:.2f}%")

                            print("\n=== 交易统计 ===")
                            print(f"总交易次数: {stats['交易统计']['总交易次数']}")
                            print(f"胜率: {stats['交易统计']['胜率']:.2f}%")
                            print(f"平均每笔收益: ${stats['交易统计']['平均每笔收益']:,.2f}")
                            print(f"最大单笔收益: ${stats['交易统计']['最大单笔收益']:,.2f}")
                            print(f"最大单笔亏损: ${stats['交易统计']['最大单笔亏损']:,.2f}")

                            print("\n=== 风险统计 ===")
                            print(f"最大回撤: {stats['风险统计']['最大回撤']:.2f}%")
                            print(f"收益波动率(年化): {stats['风险统计']['收益波动率']:.2f}%")
                            print(f"夏普比率: {stats['风险统计']['夏普比率']:.2f}")
                            print(f"索提诺比率: {stats['风险统计']['Sortino比率']:.2f}")

                            print("\n=== 当前持仓信息 ===")
                            print(f"持仓数量: {stats['持仓信息']['当前持仓']:,d}股")
                            if stats['持仓信息']['当前持仓'] > 0:
                                print(f"持仓成本: ${stats['持仓信息']['持仓成本']:.2f}")
                                print(f"持仓市值: ${stats['持仓信息']['持仓市值']:,.2f}")

                            print("\n=== 成本统计 ===")
                            print(f"总交易成本: ${stats['成本统计']['总交易成本']:,.2f}")
                            print(f"成本占初始资金比例: {stats['成本统计']['成本占比']:.2f}%")

                            print("\n=== 基准对比分析 ===")
                            print(f"策略收益率: {stats['收益统计']['收益率']:.2f}%")
                            print(f"买入持有收益率: {buy_hold_stats['收益率']:.2f}%")
                            excess_return = stats['收益统计']['收益率'] - buy_hold_stats['收益率']
                            print(f"超额收益率: {excess_return:.2f}%")
                            print(f"策略最大回撤: {stats['风险统计']['最大回撤']:.2f}%")
                            print(f"买入持有最大回撤: {buy_hold_stats['最大回撤']:.2f}%")

                            open_trades = [t for t in strategy.trades.executed_trades if t['status'] == 'open']
                            if open_trades:
                                print("\n当前未平仓交易:")
                                for trade in open_trades:
                                    curr_price = strategy.data.close[0]
                                    unrealized_pnl = (curr_price - trade['entry_price']) * trade['size']
                                    print(f"买入日期: {trade['entry_date']}")
                                    print(f"买入价格: ${trade['entry_price']:.2f}")
                                    print(f"当前价格: ${curr_price:.2f}")
                                    print(f"持仓数量: {trade['size']}股")
                                    print(f"浮动盈亏: ${unrealized_pnl:.2f}")

                            print("\n" + "=" * 50)
                            print("\n正在生成图表分析...")
                            visualizer = TradeVisualizer(
                                df=df_viz,
                                strategy=strategy,
                                stats=stats,
                                symbol=symbol,
                                initial_cash=initial_cash,
                                buy_hold_stats=buy_hold_stats
                            )

                            fig = visualizer.create_candlestick_chart()
                            fig.show()
                        else:
                            print("\n无法生成历史回测报告：策略未初始化")

        # 以下是回测结果处理，实时模式不会执行到这里
        if not live_mode:
            stats = TradeAnalyzer.calculate_statistics(strategy)
            buy_hold_stats = TradeAnalyzer.calculate_buy_hold_return(df_viz, initial_cash)

            print("\n" + "=" * 20 + " 策略回测详细报告 " + "=" * 20)
            print("\n=== 收益统计 ===")
            print(f"初始资金: ${stats['收益统计']['初始资金']:,.2f}")
            print(f"最终资金: ${stats['收益统计']['最终资金']:,.2f}")
            print(f"净收益: ${stats['收益统计']['净收益']:,.2f}")
            print(f"总收益率: {stats['收益统计']['收益率']:.2f}%")
            print(f"年化收益率: {stats['收益统计']['年化收益率']:.2f}%")

            print("\n=== 交易统计 ===")
            print(f"总交易次数: {stats['交易统计']['总交易次数']}")
            print(f"胜率: {stats['交易统计']['胜率']:.2f}%")
            print(f"平均每笔收益: ${stats['交易统计']['平均每笔收益']:,.2f}")
            print(f"最大单笔收益: ${stats['交易统计']['最大单笔收益']:,.2f}")
            print(f"最大单笔亏损: ${stats['交易统计']['最大单笔亏损']:,.2f}")

            print("\n=== 风险统计 ===")
            print(f"最大回撤: {stats['风险统计']['最大回撤']:.2f}%")
            print(f"收益波动率(年化): {stats['风险统计']['收益波动率']:.2f}%")
            print(f"夏普比率: {stats['风险统计']['夏普比率']:.2f}")
            print(f"索提诺比率: {stats['风险统计']['Sortino比率']:.2f}")

            print("\n=== 当前持仓信息 ===")
            print(f"持仓数量: {stats['持仓信息']['当前持仓']:,d}股")
            if stats['持仓信息']['当前持仓'] > 0:
                print(f"持仓成本: ${stats['持仓信息']['持仓成本']:.2f}")
                print(f"持仓市值: ${stats['持仓信息']['持仓市值']:,.2f}")

            print("\n=== 成本统计 ===")
            print(f"总交易成本: ${stats['成本统计']['总交易成本']:,.2f}")
            print(f"成本占初始资金比例: {stats['成本统计']['成本占比']:.2f}%")

            print("\n=== 基准对比分析 ===")
            print(f"策略收益率: {stats['收益统计']['收益率']:.2f}%")
            print(f"买入持有收益率: {buy_hold_stats['收益率']:.2f}%")
            excess_return = stats['收益统计']['收益率'] - buy_hold_stats['收益率']
            print(f"超额收益率: {excess_return:.2f}%")
            print(f"策略最大回撤: {stats['风险统计']['最大回撤']:.2f}%")
            print(f"买入持有最大回撤: {buy_hold_stats['最大回撤']:.2f}%")

            open_trades = [t for t in strategy.trades.executed_trades if t['status'] == 'open']
            if open_trades:
                print("\n当前未平仓交易:")
                for trade in open_trades:
                    curr_price = strategy.data.close[0]
                    unrealized_pnl = (curr_price - trade['entry_price']) * trade['size']
                    print(f"买入日期: {trade['entry_date']}")
                    print(f"买入价格: ${trade['entry_price']:.2f}")
                    print(f"当前价格: ${curr_price:.2f}")
                    print(f"持仓数量: {trade['size']}股")
                    print(f"浮动盈亏: ${unrealized_pnl:.2f}")

            print("\n" + "=" * 50)
            print("\n正在生成图表分析...")
            visualizer = TradeVisualizer(
                df=df_viz,
                strategy=strategy,
                stats=stats,
                symbol=symbol,
                initial_cash=initial_cash,
                buy_hold_stats=buy_hold_stats
            )

            fig = visualizer.create_candlestick_chart()
            fig.show()

    except Exception as e:
        print(f"\n发生错误: {str(e)}")
        print(traceback.format_exc())

        # 发送错误通知
        if 'notifier' in locals() and notifier:
            notifier.send_message(
                "系统错误",
                f"错误信息: {str(e)}"
            )


if __name__ == "__main__":
    main()
