"""
高风险高爆发高回报强化型量化交易策略 Ver 0.05
250809_Quant4RiskVer005.py
基于 市场趋势判断，结合均线、MACD、RSI和成交量分析的高风险高回报量化交易策略，通过server酱推送通知到绑定微信上。
作者：Michael Yuan
"""

import datetime
import os
import pandas_market_calendars as mcal
import numpy as np
import pandas as pd
import backtrader as bt
import yfinance as yf
import random
import time
import plotly.graph_objects as go
import matplotlib
matplotlib.use("Agg")  # 确保在无GUI环境下也能生成PNG
import matplotlib.pyplot as plt
import math
import requests
import json
import base64
import threading
import queue
import traceback
import datetime as dt
from plotly.subplots import make_subplots
from collections import deque
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from zoneinfo import ZoneInfo
import warnings

_XNYS_SCHEDULE_CACHE: Dict[str, tuple] = {}

def upload_image_imgbb(file_path: str, api_key: str, endpoint: str = "https://api.imgbb.com/1/upload") -> str:
    """上传图片到 imgbb，返回图片URL；失败返回空串。
    KISS：multipart + key + image(base64)，异常即返回空。
    """
    try:
        with open(file_path, 'rb') as f:
            encoded = base64.b64encode(f.read())
        data = {
            'key': api_key,
            'image': encoded
        }
        r = requests.post(endpoint, data=data, timeout=30)
        r.raise_for_status()
        resp = r.json()
        if resp.get('success') and resp.get('data', {}).get('url'):
            return resp['data']['url']
    except Exception:
        pass
    return ""

def generate_backtest_visual_png(symbol: str,
                                 start_str: str,
                                 end_str: str,
                                 interval: str,
                                 initial_cash: float,
                                 include_prepost: bool = True) -> tuple:
    """运行一次轻量回测，使用TradeVisualizer生成高保真PNG。
    返回 (png_path, stats_dict)。若失败，返回("", None)。
    说明：需要kaleido方可输出Plotly PNG；若缺失会在日志提示。
    """
    try:
        print(f"[每日总结] 回测(可视化) - 下载 {symbol} {interval} 数据: {start_str} ~ {end_str}")
        df = yfinance_download(symbol=symbol, start=start_str, end=end_str, interval=interval, prepost=include_prepost)
        # 兼容列名：统一到小写，日期列命名为 'date'
        rename_map = {c: c.lower() for c in df.columns}
        df = df.rename(columns=rename_map)
        if 'datetime' in df.columns and 'date' not in df.columns:
            df = df.rename(columns={'datetime': 'date'})
        if 'date' not in df.columns:
            raise Exception("历史数据缺少date列")
        df['date'] = pd.to_datetime(df['date'])

        # 构建回测
        cerebro = bt.Cerebro()
        params = TradingParameters(
            symbol=symbol,
            initial_cash=initial_cash,
            start_date=start_str,
            end_date=end_str,
            data_interval=interval,
            live_mode=False,
            prepost=include_prepost,
            exclude_premarket=not include_prepost,
            exclude_afterhours=not include_prepost
        )
        cerebro.addstrategy(EnhancedStrategy, trading_params=params)
        feed = bt.feeds.PandasData(
            dataname=df,
            datetime='date', open='open', high='high', low='low', close='close', volume='volume'
        )
        cerebro.adddata(feed)
        cerebro.broker.setcommission(commission=params.commission)
        cerebro.broker.set_slippage_fixed(params.slippage)
        cerebro.broker.setcash(params.initial_cash)
        strategies = cerebro.run()
        if not strategies:
            raise Exception("回测未得到策略实例")
        strategy = strategies[0]
        stats = TradeAnalyzer.calculate_statistics(strategy)
        buy_hold = TradeAnalyzer.calculate_buy_hold_return(df, initial_cash)

        # 使用已有TradeVisualizer生成图
        # 将回测得到的交易明细注入到strategy.trade_manager，以便可视化器绘制买卖点
        try:
            if hasattr(strategy, 'trades') and hasattr(strategy.trades, 'executed_trades') and strategy.trades.executed_trades:
                pass
            elif hasattr(strategy, 'trade_manager') and hasattr(strategy.trade_manager, 'executed_trades') and strategy.trade_manager.executed_trades:
                # 兼容命名：确保可视化使用的是同一容器
                strategy.trades = strategy.trade_manager
        except Exception:
            pass

        viz = TradeVisualizer(df=df.copy(), strategy=strategy, stats=stats, symbol=symbol, initial_cash=initial_cash, buy_hold_stats=buy_hold)
        fig = viz.create_candlestick_chart()
        # 提高分辨率（导出像素）
        fig.update_layout(width=1920, height=1080)
        # 将图片输出到 logs/daily_summary_screenshots
        out_dir = os.path.join(os.getcwd(), 'logs', 'daily_summary_screenshots')
        try:
            os.makedirs(out_dir, exist_ok=True)
        except Exception:
            pass
        out_png = os.path.join(out_dir, f"summary_{symbol}_{end_str}.png")
        try:
            # 使用scale放大，获得更高像素的导出
            fig.write_image(out_png, format='png', scale=2)
            print("[每日总结] 使用TradeVisualizer+kaleido导出PNG成功")
            return out_png, stats
        except Exception as e:
            print("[每日总结] 需安装kaleido以导出高保真图像：pip install -U kaleido")
            print(f"[每日总结] 导出失败: {str(e)}")
            return "", stats
    except Exception as e:
        print(f"[每日总结] 可视化回测失败: {str(e)}")
        return "", None

def get_market_session(dt_local: datetime.datetime, include_prepost: bool = True) -> tuple:
    """返回当前美股会话类型和是否视为可交易。
    返回 (session, is_trading)。session ∈ {'pre', 'regular', 'after', 'closed'}。
    - 使用 XNYS 交易日历判断是否交易日及收盘时间（含半日市）。
    - prepost=True 时，'pre' 与 'after' 也视为可交易。
    """
    try:
        # 统一到美东时间（若无时区，默认按美东处理；回测数据已按美东会话筛选）
        if dt_local.tzinfo is None:
            dt_local = dt_local.replace(tzinfo=ZoneInfo("America/New_York"))
        dt_et = dt_local.astimezone(ZoneInfo("America/New_York"))

        # 非工作日快速返回
        if dt_et.weekday() >= 5:
            return ("closed", False)

        # 获取当天XNYS日程（缓存 + 抑制DeprecationWarning）
        day_str = dt_et.strftime('%Y-%m-%d')
        if day_str in _XNYS_SCHEDULE_CACHE:
            open_et, close_et = _XNYS_SCHEDULE_CACHE[day_str]
        else:
            cal = mcal.get_calendar('XNYS')
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', DeprecationWarning)
                sched = cal.schedule(start_date=day_str, end_date=day_str)
            if sched.empty:
                return ("closed", False)
            open_et = sched['market_open'].iloc[0].tz_convert('America/New_York')
            close_et = sched['market_close'].iloc[0].tz_convert('America/New_York')
            _XNYS_SCHEDULE_CACHE[day_str] = (open_et, close_et)

        # 定义盘前/盘后窗口（ET）
        pre_start = open_et.replace(hour=4, minute=0, second=0, microsecond=0)
        # 盘前结束为开盘时刻
        pre_end = open_et
        # 盘后开始为收盘时刻（半日市自动提前）
        after_start = close_et
        after_end = close_et.replace(hour=20, minute=0, second=0, microsecond=0)

        if pre_start <= dt_et < pre_end:
            return ("pre", include_prepost)
        if open_et <= dt_et < close_et:
            return ("regular", True)
        if after_start <= dt_et < after_end:
            return ("after", include_prepost)

        return ("closed", False)

    except Exception:
        # 兜底：按时间大致判断（ET）
        dt_et = dt_local if dt_local.tzinfo and str(dt_local.tzinfo) == 'America/New_York' else dt_local.astimezone(ZoneInfo("America/New_York"))
        minutes = dt_et.hour * 60 + dt_et.minute
        if 4 * 60 <= minutes < 9 * 60 + 30:
            return ("pre", include_prepost)
        if 9 * 60 + 30 <= minutes < 16 * 60:
            return ("regular", True)
        if 16 * 60 <= minutes < 20 * 60:
            return ("after", include_prepost)
        return ("closed", False)

@dataclass
class TradingParameters:
    """交易策略参数配置"""
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
    volume_surge_mult: float = 2.0  # 从2.5降到2.0（降低成交量要求）
    breakout_period: int = 24   # 从20改为24

    # 技术指标额外参数
    atr_ma_period: int = 20
    volume_base_period: int = 15
    volume_adjust_multiplier: int = 5

    # 成交量参数
    base_volume_mult: float = 0.8
    volume_adjust: float = 0.02

    # 风险管理参数
    risk_pct_per_trade: float = 0.40  # 从0.30提高到0.40
    max_daily_loss: float = 0.50      # 从0.40提高到0.50
    max_drawdown: float = 0.80        # 从0.70提高到0.80 
    equity_risk: float = 0.3
    min_bars_between_trades: int = 1

    # 仓位参数
    base_position_pct: float = 0.8    # 从0.4提高到0.6
    max_position_pct: float = 0.95    # 从0.8提高到0.95
    min_position_size: int = 100      # 从10提到100

    # 获利了结参数 - 激进化调整
    profit_trigger_1: float = 0.50  # 50%触发首次止盈
    profit_trigger_2: float = 1.00  # 100%触发第二次止盈

    # 止损参数
    stop_atr: float = 3.0
    trade_loss_pct: float = 0.12  # 18%基础止损（高风险但更合理）
    force_min_stop_pct: float = 0.35  # 35%强制止损
    
    # [deprecated] trail_stop_mult 未再使用
    # trail_stop_mult: float = 3.0
    profit_lock_pct: float = 0.5
    exit_confirm_threshold: int = 1
    # 新增：ATR与峰值回撤双阈值止损参数
    atr_drawdown_stop_enabled: bool = True
    atr_multiple_stop: float = 3.0          # ATR倍数阈值
    peak_drawdown_stop_pct: float = 0.15    # 从入场峰值回撤阈值（20%）

    # 功能标志
    final_close: bool = True
    position_scaling_mode: str = "kelly"  # 仓位计算模式：fixed/kelly/risk_parity

    # 市场状态参数
    strong_uptrend_threshold: float = 5
    uptrend_threshold: float = 2
    sideways_threshold: float = 0.5
    downtrend_threshold: float = -9

    max_consecutive_losses: int = 2

    # ETF特殊处理参数
    strong_uptrend_mult: float = 1.5
    uptrend_mult: float = 1.2
    sideways_mult: float = 1.0
    downtrend_mult: float = 0.7

    # 重新定义追踪止损参数的语义：直接设定合理的激活阈值
    trailing_activation_base: float = 0.15  # 15%激活追踪
    trailing_strong_uptrend_mult: float = 2.5
    trailing_uptrend_mult: float = 2.0
    trailing_sideways_mult: float = 2.5
    trailing_weak_trend_mult: float = 3.0
    trailing_downtrend_mult: float = 4.0

    new_position_protection_days: int = 3

    # 回调买入参数（更保守，降低下行环境中不断摊薄的风险）
    dip_base_pct: float = 0.10  # 默认10%
    dip_per_batch: float = 0.04  # 批次间隔4%

    batches_allowed: int = 8

    lookback_days_for_dip: int = 20  # 回调检查的回望天数
    min_bars_after_add: int = 3  # 加仓后最小间隔天数
    min_meaningful_shares: int = 30

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
    # 仓位管理动态参数
    max_position_ratio_uptrend: float = 0.85  # 上涨趋势最大仓位比例
    max_position_ratio_normal: float = 0.70   # 正常市场最大仓位比例
    concentration_profit_threshold: float = 0.15  # 仓位集中度检查的盈利门槛

    dynamic_stop_enabled: bool = True  # 启用动态止损
    stop_profit_threshold: float = 0.05  # 盈利5%后放宽止损
    stop_profit_relaxation: float = 1.5  # 止损放宽倍数

    position_batch_decay: float = 0.10  # 从0.15降到0.10，减缓衰减
    position_min_add_pct: float = 0.10  # 降低加仓门槛
    position_add_base_pct: float = 0.30  # 基础加仓占用可用现金比例（原硬编码0.30）

    serverchan_sendkey: str = "SCT119824TW3DsGlBjkhAV9lJWIOLohv1P"  # Server酱的SendKey
    # 新增：企业微信群机器人 & Server酱渠道
    wecom_webhook_url: str = "https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=dd52bd7c-a8a2-4406-8332-714faaa3185e"
    serverchan_channel: str = "9|1"  # 例如 "9|1" 同时服务号与企业微信群机器人
    enable_trade_notification: bool = True  # 是否启用交易通知
    enable_risk_notification: bool = True   # 是否启用风险提醒
    notification_min_pnl: float = 100.0     # 最小通知盈亏金额
    live_mode: bool = False  # 是否为实时交易模式
    data_check_interval: int = 60  # 数据检查间隔（秒）
    max_data_delay: int = 3600  # 最大数据延迟（秒）
    generate_live_report: bool = True  # 实时运行结束后是否生成历史回测报告
    history_days: int = 3650

    small_position_ratio: float = 0.10  # 小仓位阈值10%
    micro_position_ratio: float = 0.05  # 极小仓位阈值5%
    micro_position_shares: int = 30  # 极小股数阈值

    # 新增：核心仓位与当日止盈限制
    core_position_min_ratio: float = 0.30  # 至少保留30%核心仓位
    max_take_profit_actions_per_day: int = 2  # 单日最多止盈次数
    take_profit_cooldown_bars: int = 4
    allow_profit_add: bool = True           # 默认禁止"盈利加仓"
    # 强势市场的动态限制
    tp_strong_up_max_actions_per_day: int = 1  # 强势上行时，每日最多止盈次数

    # 盈利加仓限制
    profit_add_rsi_cap: float = 70.0        # RSI高于此值不做盈利加仓，避免高位追涨
    profit_add_ma_fast_premium: float = 0.06 # 价格较快线均线溢价超过6%不盈利加仓
    profit_add_only_strong: bool = True     # 仅在 strong_uptrend 执行盈利加仓


    # 凯利公式相关参数（激进化）
    kelly_multiplier: float = 1.5  # 提高到1.5倍
    min_kelly_position: float = 0.60  # 最低60%仓位
    max_kelly_position: float = 0.95  # 最高95%仓位

    # 调试设置
    debug_mode: bool = True

    # 新增日内交易参数
    intraday_mode: bool = False  # 是否为日内模式
    data_interval: str = "1d"     # 数据间隔
    trading_start_time: str = "09:30"  # 交易开始时间
    trading_end_time: str = "16:00"    # 交易结束时间
    exclude_premarket: bool = False      # 排除盘前交易
    exclude_afterhours: bool = False     # 排除盘后交易
    prepost: bool = True                 # 是否视为可交易（盘前/盘后）

    # 日内技术指标周期倍数（用于自动转换）
    intraday_multiplier: int = 26  # 15分钟K线：每天6.5小时 * 4 = 26根
    min_data_points: int = 200
    intraday_min_bars_between_trades: int = 3  # 日内模式最小交易间隔bar数

    # 盈利加仓的额外过滤与冷却
    profit_add_cooldown_bars: int = 6
    max_profit_adds_per_day: int = 1
    profit_add_long_ma_premium_cap: float = 0.25  # 相对长均线溢价上限（25%）

    # 强势/上行市场的动态参数
    dynamic_core_min_ratio_strong: float = 0.50
    dynamic_core_min_ratio_uptrend: float = 0.40
    dynamic_concentration_threshold_strong: float = 0.95
    dynamic_concentration_profit_threshold_strong: float = 0.25

    # 批次衰减（解决“前期一次性用完批次，后期无法加仓”）
    enable_add_batches_decay: bool = True
    add_batches_decay_days: int = 8

    # 分笔止损与时限退出（避免长期浮亏占用仓位导致胜率虚高/后段无交易）
    per_lot_stop_loss_pct: float = 0.18
    max_holding_bars_loss_exit: int = 200
    stagnation_relax_bars: int = 80
    # 单笔卖出规模上限（降低最大单笔亏损、平滑回撤），按当前持仓比例
    max_single_exit_ratio: float = 0.15
    # 当日亏损保护（仅日内模式生效）：当日回撤超过阈值时，触发小比例去风险
    daily_loss_guard_pct: float = 0.08
    daily_loss_guard_sell_ratio: float = 0.20
    # 回调加仓的额外门槛：跌破中/慢均线时需要RSI反弹（<30→>35）才允许加仓
    dip_add_require_above_ma: bool = True
    dip_add_require_rsi_bounce: bool = True
    # 日内节流：卖出后N个bar内禁止加仓；每日最大加仓次数与每日最大老化亏损退出次数
    post_sell_cooldown_bars: int = 5
    max_adds_per_day: int = 2
    max_aging_exits_per_day: int = 1
    aging_exit_min_spacing_bars: int = 6

    # 心跳/推送配置
    heartbeat_interval_seconds: int = 1800  # 系统心跳推送间隔（默认30分钟）
    heartbeat_include_returns: bool = True  # 心跳是否包含价格/当日/总收益率
    # 分渠道心跳配置
    enable_serverchan_hourly_heartbeat: bool = True
    serverchan_heartbeat_seconds: int = 3600  # Server酱每小时
    enable_wecom_daily_heartbeat: bool = True
    wecom_daily_heartbeat_hour_local: int = 10   # Asia/Shanghai 本地小时
    wecom_daily_heartbeat_minute_local: int = 15 # Asia/Shanghai 本地分钟

    # 心跳专用渠道：仅用于限制心跳只发到服务号，避免企业微信收到小时心跳
    # 说明：很多用户在Server酱后台把多个渠道（如服务号/企业微信群机器人）组合在一起，例如 "9|1"。
    # 为避免心跳打扰企业微信，这里增加单独的心跳渠道覆盖。默认仅服务号（常见为'9'）。
    serverchan_heartbeat_channel: str = "9"

    # 是否在开始监控时立即发送一次“每日交易总结”（企业微信）
    send_daily_summary_on_start: bool = True

    # 外部健康检查心跳URL（healthchecks.io等），留空则不启用
    healthcheck_ping_url: str = "https://hc-ping.com/df7b5083-70f5-4cd8-838c-550855ec88d5"

    # 加速时钟（KISS）：>1 表示加速，例如 60 表示把 60 秒压缩为 1 秒；=1 表示真实时间
    # 仅用于控制/监控循环等“等待类”sleep，网络重试与速率限制的sleep不受影响
    time_scale: float = 1.0
    # imgbb图床API Key（用于每日总结图表上传）
    imgbb_api_key: str = "1a7c2fc0bf1a308ce82031f35a72d4b9"
    # 每日总结用的“简易回测”初始资金（避免因实盘初始资金过小而不触发交易）
    summary_backtest_initial_cash: float = 100000.0

    # 实时运行日志节流
    enable_periodic_signal_log: bool = False   # 是否定期打印信号检查（默认关闭）
    enable_periodic_account_log: bool = False  # 是否定期打印账户状态（默认关闭）
    periodic_log_interval_bars: int = 100      # 定期打印的bar间隔

    hot_stocks_2025: List[str] = None  # 将在__post_init__中初始化

    def __post_init__(self):
        """初始化后处理，设置默认日期"""
        # 初始化2025年热门股票列表（可用于激进参数提示，不再限制运行）
        self.hot_stocks_2025 = [
            'NVDA', 'AMD', 'TSLA', 'COIN', 'PLTR',
            'SMCI', 'ARM', 'CRWD', 'SNOW', 'CELH',
            'ABNB', 'NET', 'DDOG', 'MARA', 'RIOT',
            'SOFI', 'TEM', 'GOOGL','META','HOOD'
        ]

        if self.start_date is None or self.end_date is None:
            default_start, default_end = self._get_default_dates()
            if self.start_date is None:
                self.start_date = default_start
            if self.end_date is None:
                self.end_date = default_end

        # 根据数据间隔自动设置日内模式
        if self.data_interval in ["15m", "30m", "5m", "1h"]:
            self.intraday_mode = True
            # 小时级也参与等效日级缩放
            self._save_original_periods()
            self._adjust_intraday_periods()
            try:
                print(f"1小时数据已按等效日级缩放周期")
            except Exception:
                pass

        # 日内模式下，提高最小交易间隔，减少过度交易
        if self.intraday_mode:
            self.min_bars_between_trades = max(self.min_bars_between_trades, self.intraday_min_bars_between_trades)

        # 准确设置每个交易日的bars数，用于年化和波动率聚合
        try:
            di = (self.data_interval or "1d").lower()
            if di in ("1d", "1day"):
                self.intraday_multiplier = 1
            elif di in ("1h", "60m", "1hour"):
                # 美股单日约6.5小时，向上取整
                self.intraday_multiplier = 7
            elif di in ("30m", "30min"):
                self.intraday_multiplier = 13
            elif di in ("15m", "15min"):
                self.intraday_multiplier = 26
            elif di in ("5m", "5min"):
                self.intraday_multiplier = 78
            else:
                # 其他周期：按1天估算
                self.intraday_multiplier = max(1, self.intraday_multiplier)
        except Exception:
            pass

        # 添加极端值警告
        if self.trailing_activation_base <= 0 or self.trailing_activation_base >= 1:
            print(f"[WARNING] trailing_activation_base={self.trailing_activation_base} 是极端值!")

    def _get_default_dates(self) -> Tuple[str, str]:
        """获取默认的回测日期范围"""
        end_date = datetime.datetime.now()
        start_date = end_date - datetime.timedelta(days=365 * 1)
        return start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")

    def _save_original_periods(self):
        """保存原始周期值"""
        self._original_fast_ma = self.fast_ma
        self._original_slow_ma = self.slow_ma
        self._original_mid_ma = self.mid_ma
        self._original_long_ma = self.long_ma
        self._original_atr_period = self.atr_period
        self._original_rsi_period = self.rsi_period
        self._original_volatility_window = self.volatility_window

    def _limit_date_range_for_intraday(self):
        """限制日内数据的日期范围"""
        end_date = datetime.datetime.now()

        # 15分钟数据限制为最近30天
        if self.data_interval == "1h":
            max_days = 90  # 从30天改为90天，获取更多数据
        elif self.data_interval == "15m":
            max_days = 180  # 从7天改为30天
        else:
            max_days = 365  # 从60天改为180天

        start_date = end_date - datetime.timedelta(days=max_days)

        # 更新日期
        if self.end_date is None or pd.to_datetime(self.end_date) > end_date:
            self.end_date = end_date.strftime("%Y-%m-%d")

        if self.start_date is None or pd.to_datetime(self.start_date) < start_date:
            self.start_date = start_date.strftime("%Y-%m-%d")
            print(f"[警告] 15分钟数据限制日期范围为最近{max_days}天")
            print(f"调整日期范围: {self.start_date} 至 {self.end_date}")

    def _adjust_intraday_periods(self):
        """根据日内模式调整技术指标周期"""
        if not self.intraday_mode:
            return

        # 保存原始周期值（已在_save_original_periods中完成）

        # 1小时数据：不提前返回，允许按数据量自适应缩放
        if self.data_interval == "1h":
            pass

        # 15分钟数据的特殊处理
        elif self.data_interval == "15m":
            # 使用更保守的周期设置以避免数据不足
            self.fast_ma = min(self.fast_ma, 20)
            self.slow_ma = min(self.slow_ma, 40)
            self.mid_ma = min(self.mid_ma, 60)
            self.long_ma = min(self.long_ma, 100)

            # 其他指标也做相应限制
            self.atr_period = min(self.atr_period, 30)
            self.rsi_period = min(self.rsi_period, 30)
            self.volatility_window = min(self.volatility_window, 40)
            self.momentum_period = min(self.momentum_period, 20)
            self.breakout_period = min(self.breakout_period, 50)

        print(f"日内模式已启用 - 数据间隔: {self.data_interval}")
        print(f"技术指标周期 - 快线MA: {self.fast_ma}, 慢线MA: {self.slow_ma}, "
              f"中线MA: {self.mid_ma}, 长线MA: {self.long_ma}")

    def adjust_periods_for_data_length(self, data_length: int):
        """根据实际数据长度动态调整技术指标周期"""
        if not self.intraday_mode:
            return

        print(f"数据长度: {data_length}条")

        # 如果数据太少，进一步降低周期要求
        if data_length < 200:
            print(f"数据长度不足200条，使用最小周期设置...")
            self.fast_ma = 10
            self.slow_ma = 20
            self.mid_ma = 30
            self.long_ma = 40

            self.atr_period = 14
            self.rsi_period = 14
            self.volatility_window = 20

            self.macd_fast = 8
            self.macd_slow = 17
            self.macd_signal = 6
        elif data_length < self.min_data_points:
            # 按比例缩小
            scale = data_length / self.min_data_points
            self.fast_ma = max(5, int(self.fast_ma * scale))
            self.slow_ma = max(10, int(self.slow_ma * scale))
            self.mid_ma = max(15, int(self.mid_ma * scale))
            self.long_ma = max(20, int(self.long_ma * scale))

        print(f"最终周期设置 - MA快线: {self.fast_ma}, MA慢线: {self.slow_ma}, MA中线: {self.mid_ma}")

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

        # 找不到则回退默认值（若提供），否则抛错
        if default_value is not None:
            return default_value
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
            import traceback
            self.log(f"更新统计失败: {str(e)}")
            traceback.print_exc()
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

        self.strategy.log(f"初始化仓位管理 - 投资组合价值: {portfolio_value:.2f}, 最小交易规模: {self.min_position_size}", level="INFO")

    def set_indicators(self, indicators):
        """设置技术指标引用"""
        self.indicators = indicators

    def get_trend_strength(self, force_recalculate=False):
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

    def get_buy_signal(self) -> bool:
        """激进买入信号 - 增加触发频率（修正版）"""
        # 统一按照"天数→bars"的方式换算窗口，避免在1小时K上误用日级窗口
        def bars(days: int) -> int:
            intraday_mode = ParamAccessor.get_param(self, 'intraday_mode')
            if intraday_mode:
                mult = max(1, int(ParamAccessor.get_param(self, 'intraday_multiplier') or 26))
                return max(1, int(days * mult))
            return max(1, int(days))

        # 最小数据检查
        if len(self.strategy.data.close) < bars(3):
            return False

        current_price = self.strategy.data.close[0]
        try:
            # 条件1：突破信号（使用参数化窗口 breakout_period，并按bars换算）
            breakout_days = ParamAccessor.get_param(self, 'breakout_period') or 20
            breakout_window = bars(int(breakout_days))
            if len(self.strategy.data.high) >= 2:
                size = min(breakout_window, len(self.strategy.data.high))
                if size >= 2:
                    vals = list(self.strategy.data.high.get(size=size))[:-1]
                    vals = [v for v in vals if pd.notna(v)]
                    if vals:
                        recent_high = max(vals)
                        if recent_high > 0 and current_price > recent_high * 1.01:  # 从1.015改为1.01:
                            rise = max(0.0, (current_price / recent_high - 1) * 100)
                            self.strategy.log(f"突破信号 - 突破{rise:.1f}%", level="CRITICAL")
                            return True
            
            # 条件2：MACD金叉（保持原有）
            if hasattr(self.strategy, 'macd') and len(self.strategy.macd) > 1:
                if (self.strategy.macd.macd[-1] <= self.strategy.macd.signal[-1] and
                    self.strategy.macd.macd[0] > self.strategy.macd.signal[0]):
                    self.strategy.log("MACD金叉", level="CRITICAL")
                    return True
            
            # 条件3：短期动量（参数化 + bars换算）
            momentum_days = ParamAccessor.get_param(self, 'momentum_period') or 10
            momentum_window = bars(int(momentum_days))
            if len(self.strategy.data.close) >= momentum_window:
                mom_ret = (current_price / self.strategy.data.close[-momentum_window] - 1)
                if mom_ret > 0.03:
                    self.strategy.log(f"短期动量信号 - {momentum_days}天涨{mom_ret*100:.1f}%", level="CRITICAL")
                    return True
            
            # 条件4：均线支撑买入（完整的安全检查，空仓/小仓位放宽）
            if hasattr(self.strategy, 'ma_fast') and hasattr(self.strategy, 'ma_slow'):
                try:
                    # 确保指标有足够的数据
                    if (hasattr(self.strategy.ma_fast, '__len__') and 
                        hasattr(self.strategy.ma_slow, '__len__') and
                        len(self.strategy.ma_fast) > 0 and 
                        len(self.strategy.ma_slow) > 0):
                        
                        # 使用更安全的访问方式
                        try:
                            ma_fast_value = float(self.strategy.ma_fast[0])
                            ma_slow_value = float(self.strategy.ma_slow[0])
                            
                            if ma_fast_value > 0 and ma_slow_value > 0:  # 确保均线有效
                                price_distance = abs(current_price - ma_fast_value) / ma_fast_value
                                # 放宽：空仓或小仓位(<5%)时，提高距离阈值到4%
                                position_ratio = 0.0
                                if self.strategy.position:
                                    position_ratio = (self.strategy.position.size * current_price) / max(1e-9, self.strategy.broker.getvalue())
                                near_threshold = 0.04 if position_ratio < 0.05 else 0.02
                                if price_distance < near_threshold and ma_fast_value > ma_slow_value:
                                    self.strategy.log("均线支撑买入", level="CRITICAL")
                                    return True
                        except (TypeError, ValueError) as e:
                            # 数值转换失败，静默跳过
                            pass
                except (IndexError, AttributeError) as e:
                    # 指标还未就绪，静默跳过
                    pass
            
            # 条件5：RSI超卖（保持原有但更宽松）
            if hasattr(self.strategy, 'rsi') and len(self.strategy.rsi) > 1:
                if self.strategy.rsi[0] > 30 and self.strategy.rsi[-1] < 30:
                    self.strategy.log("RSI超卖反弹", level="CRITICAL")
                    return True
            
            # 条件6：强势反弹（按天数→bars）
            if len(self.strategy.data.close) >= bars(5):
                size5 = min(bars(5), len(self.strategy.data.low))
                vals_low = list(self.strategy.data.low.get(size=size5))
                vals_low = [v for v in vals_low if pd.notna(v)]
                if vals_low:
                    recent_low = min(vals_low)
                else:
                    recent_low = 0
                if recent_low > 0:
                    bounce = (current_price / recent_low - 1)
                    if bounce > 0.03 and hasattr(self.strategy, 'rsi') and self.strategy.rsi[0] > 40:
                        if hasattr(self.strategy, 'volume_ma') and self.strategy.data.volume[0] > self.strategy.volume_ma[0] * 1.2:
                            self.strategy.log(f"强势反弹确认 - 反弹{bounce*100:.1f}%+放量", level="CRITICAL")
                            return True
            
            # 条件7：连续上涨信号（按天数→bars，空仓更宽松）
            if len(self.strategy.data.close) >= bars(2) + 1:
                recent_up = (self.strategy.data.close[0] > self.strategy.data.close[-bars(1)] and
                              self.strategy.data.close[-bars(1)] > self.strategy.data.close[-bars(2)])
                if recent_up:
                    two_day_gain = (self.strategy.data.close[0] / self.strategy.data.close[-bars(2)] - 1)
                    # 空仓/小仓位放宽阈值
                    position_ratio = 0.0
                    if self.strategy.position:
                        position_ratio = (self.strategy.position.size * current_price) / max(1e-9, self.strategy.broker.getvalue())
                    gain_threshold = 0.01 if position_ratio < 0.05 else 0.02
                    if two_day_gain > gain_threshold:
                        if hasattr(self.strategy, 'volume_ma'):
                            vol_mult = 0.6 if position_ratio < 0.05 else 0.8
                            if self.strategy.data.volume[0] > self.strategy.volume_ma[0] * vol_mult:
                                self.strategy.log(f"连续上涨信号 - 2天涨{two_day_gain*100:.1f}%+放量", level="CRITICAL")
                                return True

            # 条件8：成交量放大（新增）
            if hasattr(self.strategy, 'volume_ma') and self.strategy.volume_ma[0] > 0:
                vol_ratio = self.strategy.data.volume[0] / self.strategy.volume_ma[0]
                if vol_ratio > 1.5 and current_price > self.strategy.data.close[-1]:
                    self.strategy.log(f"放量上涨信号 - 成交量{vol_ratio:.1f}倍", level="CRITICAL")
                    return True

            # 条件9：微调触发器（提升中后期触发概率）
            try:
                # 当未交易天数较多时，适度放宽触发（避免“前密后疏”）
                bars_since_last = (len(self.strategy) - getattr(self.strategy, 'last_trade_bar', 0)) if hasattr(self.strategy, 'last_trade_bar') else 9999
                if bars_since_last >= 60:  # 两到三个月未交易
                    # 放宽动量与突破阈值
                    if len(self.strategy.data.close) >= momentum_window:
                        mom_ret = (current_price / self.strategy.data.close[-momentum_window] - 1)
                        if mom_ret > 0.02:
                            self.strategy.log("久未交易放宽：动量信号触发", level="INFO")
                            return True
                    if len(self.strategy.data.high) >= 2:
                        vals = list(self.strategy.data.high.get(size=min(breakout_window, len(self.strategy.data.high))))[:-1]
                        vals = [v for v in vals if pd.notna(v)]
                        if vals:
                            recent_high = max(vals)
                            if recent_high > 0 and current_price > recent_high * 1.005:
                                self.strategy.log("久未交易放宽：微突破触发", level="INFO")
                                return True
            except Exception:
                pass
    
        
        except (IndexError, AttributeError) as e:
            # 某个指标访问失败，返回False
            self.strategy.log(f"买入信号检查中断: {str(e)}", level="DEBUG")
            return False
        
        return False

    def get_market_regime(self, force_update=False):
        """增强版市场状态判断 - 适用于高风险高回报股票"""
        if not self.strategy:
            return "sideways"

        try:
            # 获取基础趋势强度
            trend_strength = self.get_trend_strength(force_recalculate=True)
            
            # 获取更多市场指标
            close = self.strategy.data.close[0]
            ma_fast = self.strategy.ma_fast[0] if hasattr(self.strategy, 'ma_fast') else close
            ma_slow = self.strategy.ma_slow[0] if hasattr(self.strategy, 'ma_slow') else close
            volume = self.strategy.data.volume[0]
            volume_ma = self.strategy.volume_ma[0] if hasattr(self.strategy, 'volume_ma') else volume
            
            # 计算价格动量
            price_momentum = 0
            if len(self.strategy.data.close) > 5:
                price_momentum = (close / self.strategy.data.close[-5] - 1) * 100
            
            # 计算成交量异常
            volume_surge = volume / volume_ma if volume_ma > 0 else 1.0
            
            # 动态市场状态判断
            regime = "sideways"  # 默认状态
            
            # 强势上涨：趋势强+量价配合
            if (trend_strength > 12 and price_momentum > 3 and 
                close > ma_fast > ma_slow and volume_surge > 1.2):
                regime = "strong_uptrend"
                
            # 上涨：趋势正向+技术面支撑
            elif (trend_strength > 6 and price_momentum > 1 and 
                  close > ma_fast and volume_surge > 0.8):
                regime = "uptrend"
                
            # 震荡：趋势弱但有支撑
            elif trend_strength > 0 and close > ma_slow * 0.95:
                regime = "sideways"
                
            # 下跌：趋势负+技术面转弱
            else:
                regime = "downtrend"
            
            # 添加急跌保护
            if price_momentum < -5 and volume_surge > 2:
                regime = "downtrend"
                
            # 记录状态
            self._market_regime = regime
            
            # 输出详细日志
            if ParamAccessor.get_param(self, 'debug_mode'):
                self.strategy.log(
                    f"市场状态更新 - {regime} "
                    f"[趋势:{trend_strength:.1f}, 动量:{price_momentum:.1f}%, "
                    f"量比:{volume_surge:.1f}]",
                    level="INFO"
                )
            
            return regime

        except Exception as e:
            self.strategy.log(f"市场状态错误: {str(e)}")
            return "sideways"
class RiskManager:
    """风险管理类 - 合并止损管理功能"""

    def __init__(self, params):
        self.trading_params = params
        self.strategy = None  # 将在之后设置

        # 初始化基本变量
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

        # 波动率/追踪止损状态
        self.vol_ma20 = 0
        self.vol_high = 0
        self.trailing_activated = False
        self.highest_profit = 0
        self.highest_price = None
        self.profit_target_hits = 0
        
        # StopManager相关属性
        self.exit_signals = 0
        self.last_check_price = 0
        self.reversal_confirmed = False
        self.vol_threshold = {}

        # 缓存
        self._cache = {}
        self._last_update = 0

        # 预初始化风险参数（正数语义：0.8 表示允许 80% 回撤/损失）
        self.max_drawdown = abs(ParamAccessor.get_param(self, 'max_drawdown'))
        self.max_daily_loss = abs(ParamAccessor.get_param(self, 'max_daily_loss'))
        self.max_position = ParamAccessor.get_param(self, 'max_position_pct')
        self._last_cleanup_bar = 0

    def set_indicators(self, indicators):
        """设置技术指标引用"""
        self.indicators = indicators

    def set_strategy(self, strategy):
        """设置策略引用"""
        self.strategy = strategy

        # 确保从策略获取trading_params
        if hasattr(strategy, 'trading_params') and strategy.trading_params is not None:
            self.trading_params = strategy.trading_params
            print(f"{self.__class__.__name__} 成功获取trading_params引用")
        else:
            print(f"警告: {self.__class__.__name__} 无法获取trading_params")

        # 现在有了策略，更新初始值
        self.peak_value = self.strategy.broker.getvalue()
        self.daily_value = self.peak_value
        # 重新初始化依赖于策略的参数
        self._init_risk_params()
        self._init_thresholds()

    def _init_risk_params(self):
        """初始化风险参数"""
        # 使用原始参数值（正数语义）
        self.max_drawdown = abs(ParamAccessor.get_param(self, 'max_drawdown'))
        self.max_daily_loss = abs(ParamAccessor.get_param(self, 'max_daily_loss'))
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
            'sideways': volatility_window / 10.0 * 1.0,
            'downtrend': volatility_window / 10.0 * 0.8
        }

        # 追踪止损激活阈值
        trailing_activation_base = ParamAccessor.get_param(self, 'trailing_activation_base')
        self.trail_activation_threshold = {
            'strong_uptrend': trailing_activation_base * ParamAccessor.get_param(self, 'trailing_strong_uptrend_mult'),
            'uptrend': trailing_activation_base * ParamAccessor.get_param(self, 'trailing_uptrend_mult'),
            'sideways': trailing_activation_base * ParamAccessor.get_param(self, 'trailing_sideways_mult'),
            'weak_trend': trailing_activation_base * ParamAccessor.get_param(self, 'trailing_weak_trend_mult'),
            'downtrend': trailing_activation_base * ParamAccessor.get_param(self, 'trailing_downtrend_mult'),
        }

        # 设置风险评分阈值
        equity_risk = ParamAccessor.get_param(self, 'equity_risk')
        self.risk_thresholds = {
            'low': equity_risk * 100,
            'medium': equity_risk * 150,
            'high': equity_risk * 200
        }
        
        if hasattr(self.strategy, 'broker'):
            self.daily_value = self.strategy.broker.getvalue()

    def _init_thresholds(self):
        """初始化各种阈值设置（来自StopManager）"""
        # 获取市场状态阈值参数
        strong_uptrend_threshold = ParamAccessor.get_param(self, 'strong_uptrend_threshold')
        uptrend_threshold = ParamAccessor.get_param(self, 'uptrend_threshold')
        sideways_threshold = ParamAccessor.get_param(self, 'sideways_threshold')
        downtrend_threshold = ParamAccessor.get_param(self, 'downtrend_threshold')

        # 获取追踪止损相关参数
        trailing_activation_base = ParamAccessor.get_param(self, 'trailing_activation_base')
        trailing_strong_uptrend_mult = ParamAccessor.get_param(self, 'trailing_strong_uptrend_mult')
        trailing_uptrend_mult = ParamAccessor.get_param(self, 'trailing_uptrend_mult')
        trailing_sideways_mult = ParamAccessor.get_param(self, 'trailing_sideways_mult')
        trailing_weak_trend_mult = ParamAccessor.get_param(self, 'trailing_weak_trend_mult')
        trailing_downtrend_mult = ParamAccessor.get_param(self, 'trailing_downtrend_mult')

        # 设置波动率阈值
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

    def _get_risk_metrics(self, current_price):
        """统一计算风险指标，避免重复计算"""
        portfolio_value = self.strategy.broker.getvalue()
        current_bar = len(self.strategy)

        # 计算回撤（负数表示回撤）
        current_drawdown = (portfolio_value / self.peak_value - 1) if self.peak_value > 0 else 0

        # 计算日内损失
        daily_loss = (portfolio_value / self.daily_value - 1) if self.daily_value > 0 else 0

        # 获取动态风险容忍度（保持正数语义，不放大到>100%）
        equity_risk = ParamAccessor.get_param(self, 'equity_risk')
        # 可按账户风险偏好略微收紧/放宽（这里维持不变，避免>100%阈值）
        dynamic_max_drawdown = max(0.01, min(0.95, self.max_drawdown))

        return {
            'portfolio_value': portfolio_value,
            'current_bar': current_bar,
            'current_drawdown': current_drawdown,
            'daily_loss': daily_loss,
            'dynamic_max_drawdown': dynamic_max_drawdown,
            'current_price': current_price
        }

    def check_risk_limits(self, current_price):
        """检查是否达到风险限制，增强版 - 修复持续限制问题"""
        try:
            # 获取统一的风险指标
            metrics = self._get_risk_metrics(current_price)
            current_bar = metrics['current_bar']
            portfolio_value = metrics['portfolio_value']
            current_drawdown = metrics['current_drawdown']
            daily_loss = metrics['daily_loss']
            dynamic_max_drawdown = metrics['dynamic_max_drawdown']

            # 定期清理过期限制
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

            # 使用缓存避免重复检查
            if hasattr(self, '_last_risk_check_bar') and self._last_risk_check_bar == current_bar:
                return getattr(self, '_last_risk_check_result', False)
            self._last_risk_check_bar = current_bar

            # 1. 整体回撤检查（current_drawdown 为负数，使用绝对值比较）
            if abs(current_drawdown) > dynamic_max_drawdown:
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
                    # 改进的恢复条件（使用绝对回撤幅度下降作为解除依据）
                    days_since_trigger = len(self.strategy) - getattr(self, '_drawdown_trigger_day', len(self.strategy))
                    recovery_conditions = [
                        abs(current_drawdown) < dynamic_max_drawdown * 0.8,  # 回撤收敛回80%阈值以内
                        days_since_trigger > 30,
                        portfolio_value > self.peak_value * 0.95,
                        (abs(current_drawdown) < dynamic_max_drawdown * 0.9 and days_since_trigger > 15),
                        (days_since_trigger > 60 and abs(current_drawdown) < dynamic_max_drawdown)
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
                if hasattr(self, '_drawdown_triggered'):
                    self._drawdown_triggered = False
                    self._drawdown_trigger_day = 0

            # 2. 日内损失检查（daily_loss 为负数，超出阈值触发）
            if self.daily_value > 0:
                if abs(daily_loss) > self.max_daily_loss:
                    self.strategy.log(f"触发日内损失保护: {daily_loss:.2%}", level="CRITICAL")
                    return True

            # 3. 单笔交易风险检查
            current_day = len(self.strategy.data)

            if not hasattr(self.strategy, 'last_loss_day'):
                self.strategy.last_loss_day = 0
            if not hasattr(self.strategy, 'last_trade_pnl'):
                self.strategy.last_trade_pnl = 0

            if self.strategy.last_trade_pnl < 0:
                days_since_loss = current_day - self.strategy.last_loss_day
                risk_pct_per_trade = ParamAccessor.get_param(self, 'risk_pct_per_trade')
                trade_risk_limit = -abs(risk_pct_per_trade) * portfolio_value

                extreme_loss_threshold = trade_risk_limit * 3

                if self.strategy.last_trade_pnl < extreme_loss_threshold and days_since_loss <= 5:
                    if hasattr(self.strategy, 'market_state'):
                        current_regime = self.strategy.market_state.get_market_regime()
                        current_trend = self.strategy.market_state.get_trend_strength()

                        if current_regime in ["uptrend", "strong_uptrend"] and current_trend > 5:
                            self.strategy.log(
                                f"市场转好（{current_regime}，趋势{current_trend:.1f}），"
                                f"解除亏损限制（亏损{self.strategy.last_trade_pnl:.2f}）",
                                level="INFO"
                            )
                            self.strategy.last_trade_pnl = 0
                            self.strategy.last_loss_day = 0
                            return False

                    self.strategy.log(
                        f"极端亏损限制交易: {self.strategy.last_trade_pnl:.2f} < {extreme_loss_threshold:.2f} "
                        f"(剩余限制天数: {5 - days_since_loss})",
                        level="CRITICAL"
                    )
                    return True
                elif days_since_loss > 30:
                    self.strategy.log(f"亏损限制超期，强制解除", level="CRITICAL")
                    self.strategy.last_trade_pnl = 0
                    self.strategy.last_loss_day = 0
                    return False

            # 4. 账户价值过低保护
            if portfolio_value < self.strategy.broker.startingcash * 0.2:
                self.strategy.log(
                    f"账户价值过低保护: ${portfolio_value:.2f} < 初始资金的20%",
                    level="CRITICAL"
                )
                return True

            self._last_risk_check_result = False
            return False

        except Exception as e:
            import traceback
            self.strategy.log(f"风险检查错误: {str(e)}")
            traceback.print_exc()
            return False

    def update_tracking(self, current_price, current_value):
        """统一更新所有追踪数据"""
        try:
            current_date = self.strategy.data.datetime.date(0)

            # 1. 更新日内价格追踪
            if self.last_date != current_date:
                self.daily_value = current_value
                self.last_date = current_date
            else:
                self.daily_high = max(getattr(self, 'daily_high', current_price), current_price)
                self.daily_low = min(getattr(self, 'daily_low', current_price), current_price)

            # 2. 更新基础风险追踪
            self.peak_value = max(self.peak_value, current_value)
            drawdown = (current_value / self.peak_value - 1)
            self.worst_drawdown = min(self.worst_drawdown, drawdown)

            # 3. 更新波动率追踪
            if hasattr(self, 'indicators') and self.indicators:
                try:
                    current_vol = self.indicators.atr[0]
                    self.vol_ma20 = (getattr(self, 'vol_ma20', current_vol) * 19 + current_vol) / 20
                    self.vol_high = max(getattr(self, 'vol_high', current_vol), current_vol)
                except:
                    self.strategy.log(f"ATR获取失败，跳过波动率更新", level="WARNING")

        except Exception as e:
            import traceback
            self.strategy.log(f"Tracking update error: {str(e)}")
            traceback.print_exc()
            raise

    def check_stops(self):
        """统一的止损检查（主入口）"""
        return self.check_all_stops()

    def _get_stop_info(self):
        """统一获取止损相关信息"""
        if not self.strategy.position or not hasattr(self.strategy, 'entry_price'):
            return None

        current_price = self.strategy.data.close[0]
        # 使用剩余开仓的加权平均成本，而不是最后一次买入价
        try:
            entry_price = self.strategy.get_average_entry_price()
        except Exception:
            entry_price = self.strategy.entry_price

        if entry_price == 0:
            return None

        profit_pct = (current_price / entry_price - 1)
        holding_days = self.strategy.get_holding_days()
        market_regime = self.strategy.market_state.get_market_regime()
        trend_strength = self.strategy.market_state.get_trend_strength()
        atr = self.strategy.atr[0] if hasattr(self.strategy, 'atr') else 0
        high_since_entry = getattr(self.strategy, 'high_since_entry', entry_price)

        return {
            'current_price': current_price,
            'entry_price': entry_price,
            'profit_pct': profit_pct,
            'holding_days': holding_days,
            'market_regime': market_regime,
            'trend_strength': trend_strength,
            'atr': atr,
            'high_since_entry': high_since_entry
        }

    def _check_basic_stop(self, current_price, trend_strength):
        """高风险策略的止损 - 更宽松"""
        try:
            if not hasattr(self.strategy, 'entry_price'):
                return False

            entry_price = self.strategy.entry_price
            current_pnl = (current_price / entry_price - 1)

            # 对2025热门股票使用更宽松的止损
            trade_loss_pct = ParamAccessor.get_param(self, 'trade_loss_pct')

            # 根据趋势强度动态调整止损
            if trend_strength > 10:
                stop_loss_threshold = -0.25
            elif trend_strength > 0:
                stop_loss_threshold = -0.20
            else:
                stop_loss_threshold = -abs(trade_loss_pct)

            # 全局保底
            min_stop = -abs(ParamAccessor.get_param(self, 'trade_loss_pct'))
            stop_loss_threshold = max(stop_loss_threshold, min_stop)

            if current_pnl < stop_loss_threshold:
                self.strategy.log(
                    f"止损触发 - 当前价{current_price:.2f}, "
                    f"亏损{current_pnl:.1%} < 阈值{stop_loss_threshold:.1%}",
                    level="CRITICAL"
                )
                return True

            return False

        except Exception as e:
            self.strategy.log(f"止损计算错误: {str(e)}")
            return False

    def _check_trailing_stop(self, current_price, market_regime):
        """修复版追踪止损 - 避免过度敏感"""
        try:
            if not self.strategy.position or not hasattr(self.strategy, 'entry_price'):
                return False
            
            try:
                avg_entry = self.strategy.get_average_entry_price()
            except Exception:
                avg_entry = getattr(self.strategy, 'entry_price', 0)
            profit_pct = (current_price / avg_entry) - 1 if avg_entry and avg_entry > 0 else 0
            holding_days = self.strategy.get_holding_days()
            
            # 激活追踪止损
            if not self.trailing_activated:
                # 高风险策略：提高激活门槛
                if market_regime == "strong_uptrend":
                    activation_threshold = 0.18
                elif market_regime == "uptrend":
                    activation_threshold = 0.15
                else:
                    activation_threshold = 0.12
                
                if profit_pct > activation_threshold:
                    self.trailing_activated = True
                    self.highest_profit = profit_pct
                    self.highest_price = current_price
                    self.strategy.log(
                        f"追踪止损激活 - 盈利{profit_pct:.1%}，最高价{self.highest_price:.2f}",
                        level="INFO"
                    )
                    return False
                else:
                    return False
            
            # 更新最高点
            if self.highest_price is None or current_price > self.highest_price:
                self.highest_price = current_price
                self.highest_profit = (self.highest_price / self.strategy.entry_price) - 1
            
            # 计算从最高点的回撤
            drawdown_from_high = (self.highest_price - current_price) / self.highest_price
            
            # 高风险策略的动态止损线
            if self.highest_profit > 1.0:
                max_drawdown = 0.30
            elif self.highest_profit > 0.5:
                max_drawdown = 0.25
            elif self.highest_profit > 0.30:
                max_drawdown = 0.20
            else:
                return False
            
            # 持仓时间越长，允许的回撤越大
            if holding_days > 60:
                max_drawdown *= 1.2
            
            # 检查是否触发止损
            if drawdown_from_high > max_drawdown:
                self.strategy.log(
                    f"追踪止损触发 - 最高价{self.highest_price:.2f}，当前价{current_price:.2f}，"
                    f"最高盈利{self.highest_profit:.1%}，回撤{drawdown_from_high:.1%} > {max_drawdown:.1%}",
                    level="CRITICAL"
                )
                return True
            
            return False
            
        except Exception as e:
            self.strategy.log(f"追踪止损错误: {str(e)}")
            return False

    def _check_profit_protection(self, current_price, market_regime):
        """获利保护 - 优化版"""
        try:
            if not hasattr(self.strategy, 'entry_price'):
                return False

            try:
                avg_entry = self.strategy.get_average_entry_price()
            except Exception:
                avg_entry = getattr(self.strategy, 'entry_price', 0)
            profit_pct = (current_price / avg_entry) - 1 if avg_entry and avg_entry > 0 else 0

            # 使用ParamAccessor获取触发参数
            t1 = ParamAccessor.get_param(self, 'profit_trigger_1')
            t2 = ParamAccessor.get_param(self, 'profit_trigger_2')
            profit_lock_pct = ParamAccessor.get_param(self, 'profit_lock_pct')

            triggers = [t1, t2]

            # 如果尚未达到下一个触发点，检查当前是否突破
            if self.profit_target_hits < len(triggers) and profit_pct >= triggers[self.profit_target_hits]:
                self.profit_target_hits += 1
                self.strategy.log(f"达到获利目标{self.profit_target_hits}: {profit_pct:.1%}", level="INFO")

            # 如果已至少命中一个目标，检查保护性止盈
            if self.profit_target_hits > 0:
                current_trigger = triggers[self.profit_target_hits - 1]

                # 动态保护线（基于市场状态）
                if market_regime == "strong_uptrend":
                    protection_ratio = 0.7
                elif market_regime == "uptrend":
                    protection_ratio = 0.6
                else:
                    protection_ratio = 0.5

                lock_in = current_trigger * profit_lock_pct * protection_ratio
                base_entry = avg_entry if (avg_entry and avg_entry > 0) else getattr(self.strategy, 'entry_price', 0)
                protect_stop_price = base_entry * (1.0 + lock_in) if base_entry and base_entry > 0 else 0

                if current_price <= protect_stop_price:
                    position_size = self.strategy.position.size
                    if position_size <= 0:
                        return False
                    if self.profit_target_hits >= 2:
                        sell_ratio = 0.5
                    else:
                        sell_ratio = 0.25
                    size_to_sell = max(ParamAccessor.get_param(self, 'min_meaningful_shares'), int(position_size * sell_ratio))
                    size_to_sell = min(size_to_sell, position_size)
                    # 进一步限制单笔卖出规模，降低最大单笔亏损/回撤突刺
                    try:
                        max_single_exit_ratio = ParamAccessor.get_param(self, 'max_single_exit_ratio')
                    except Exception:
                        max_single_exit_ratio = 0.15
                    size_cap = max(ParamAccessor.get_param(self, 'min_meaningful_shares'), int(position_size * max_single_exit_ratio))
                    size_to_sell = min(size_to_sell, size_cap)
                    if size_to_sell > 0:
                        self.strategy.log(
                            f"保护性止盈触发 - 保护价{protect_stop_price:.2f}，当前{current_price:.2f}，卖出{size_to_sell}",
                            level="CRITICAL"
                        )
                        self.strategy._pending_action = {
                            'action': 'sell',
                            'size': size_to_sell,
                            'reason': 'Profit protection'
                        }
                        return True

            return False

        except Exception as e:
            self.strategy.log(f"获利保护检查错误: {str(e)}")
            raise

    def check_all_stops(self):
        """统一的止损检查 - 带优先级"""
        if not self.strategy.position or not hasattr(self.strategy, 'entry_price'):
            return False

        try:
            stop_info = self._get_stop_info()
            if not stop_info:
                return False

            current_price = stop_info['current_price']
            market_regime = stop_info['market_regime']
            trend_strength = stop_info['trend_strength']
            profit_pct = stop_info['profit_pct']
            atr = stop_info['atr']
            high_since_entry = stop_info['high_since_entry']

            # 极端亏损立即止损（使用参数 trade_loss_pct）
            if profit_pct < -abs(ParamAccessor.get_param(self, 'trade_loss_pct')):
                self.strategy.log("极端亏损止损 - 立即执行", level="CRITICAL")
                return True

            # 新仓位保护期
            protection_days = ParamAccessor.get_param(self, 'new_position_protection_days')
            if stop_info['holding_days'] <= protection_days:
                return False

            # ATR与峰值回撤双阈值止损
            if ParamAccessor.get_param(self, 'atr_drawdown_stop_enabled'):
                try:
                    peak_drawdown = (current_price / max(high_since_entry, stop_info['entry_price']) - 1.0)
                    atr_multiple = ParamAccessor.get_param(self, 'atr_multiple_stop')
                    atr_stop_price = stop_info['entry_price'] - atr_multiple * atr if atr and atr > 0 else None
                    peak_drawdown_limit = -abs(ParamAccessor.get_param(self, 'peak_drawdown_stop_pct'))

                    if atr_stop_price is not None:
                        # 采用"任一条件触发"更符合风险控制（避免过晚止损）
                        if (current_price <= atr_stop_price) or (peak_drawdown <= peak_drawdown_limit):
                            self.strategy.log(
                                f"ATR+峰值回撤联合止损触发 - 价{current_price:.2f}≤ATR止损价{atr_stop_price:.2f}, \n"
                                f"峰值回撤{peak_drawdown:.1%}≤阈值{peak_drawdown_limit:.1%}",
                                level="CRITICAL"
                            )
                            return True
                except Exception:
                    pass

            # 按优先级检查
            if self._check_basic_stop(current_price, trend_strength):
                return True

            if self._check_trailing_stop(current_price, market_regime):
                return True

            if self._check_profit_protection(current_price, market_regime):
                return True

            return False

        except Exception as e:
            self.strategy.log(f"止损检查错误: {str(e)}")
            return False
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

        self._position_cache = {}
        self._last_update = 0
        self._last_add_price = 0  # 最后加仓价格
        self.last_time_profit_day = 0  # 时间周期获利
        self._executed_profit_levels = set()  # 确保初始化
        # 新增：追踪最后买入价格
        self._last_buy_price = None  # 添加这行
        self._buy_price_history = []  # 添加这行，记录所有买入价格历史

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
        # 增加最小持仓规模，并随投资组合增长
        # 使用ParamAccessor获取参数 - 不提供默认值
        min_position_size = ParamAccessor.get_param(self, 'min_position_size')

        # 直接使用参数，不检查是否为0、负数或9999
        self.min_position_size = max(10, int(min_position_size))  # 删除portfolio_factor的影响

        # 获取batches_allowed参数 - 使用ParamAccessor无默认值
        self.batches_allowed = ParamAccessor.get_param(self, 'batches_allowed')

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
                self.strategy.log(f"平仓后重置批次计数: {self.added_batches} → 0", level="CRITICAL")  # 改为CRITICAL级别
            self.did_time_exit = False
            self.added_batches = 0  # 这行是正确的，空仓时应该重置
            self._last_sell_day = 0
            self._last_add_bar = None
            self._max_batches_logged = False
            self._executed_profit_levels = set()
            self._buy_price_history = []
            self._last_buy_price = None
            return True
        return False

    def _clean_expired_buy_history(self):
        """清理过期的买入价格历史"""
        if not self._buy_price_history:
            return

        current_bar = len(self.strategy)
        # 只保留最近100天的记录
        self._buy_price_history = [
            record for record in self._buy_price_history
            if current_bar - record.get('bar', 0) <= 365
        ]

    def _handle_small_position(self, current_position_size, min_meaningful_trade,
                          holding_days, trend_strength, profit_pct, current_day):
        """处理小仓位 - 使用统一的参数"""
        abs_position_size = abs(current_position_size)
        portfolio_value = self.strategy.broker.getvalue()
        position_value = abs_position_size * self.strategy.data.close[0]
        position_ratio = position_value / portfolio_value

        # 使用参数
        small_pos_ratio = ParamAccessor.get_param(self, 'small_position_ratio')
        micro_pos_ratio = ParamAccessor.get_param(self, 'micro_position_ratio')
        micro_pos_shares = ParamAccessor.get_param(self, 'micro_position_shares')

        # 极小仓位（股数或占比极小）：长期持有也要清理
        if (abs_position_size < micro_pos_shares or position_ratio < micro_pos_ratio) and holding_days > 180:
            self.strategy.log(
                f"清理极小仓位 - {abs_position_size}股(占比{position_ratio:.1%})，"
                f"持有{holding_days}天",
                level="INFO"
            )
            return {
                'action': 'sell',
                'size': abs_position_size,
                'reason': 'Micro position cleanup'
            }

        # 小仓位：有盈利就清理
        elif position_ratio < small_pos_ratio and profit_pct > 0.05:
            self.strategy.log(
                f"清理小仓位 - 持仓({abs_position_size}股，占比{position_ratio:.1%})，"
                f"盈利{profit_pct*100:.1f}%",
                level="INFO"
            )
            return {
                'action': 'sell',
                'size': abs_position_size,
                'reason': 'Small position cleanup'
            }

        return None

    def _check_position_concentration(self, position_ratio, portfolio_value,
                                     current_position_size, current_price,
                                     min_meaningful_trade, profit_pct, current_day):
        """检查仓位集中度风险"""
        # 动态阈值：强势上涨期放宽集中度阈值，避免过早减仓
        concentration_threshold = ParamAccessor.get_param(self, 'max_position_ratio_uptrend')
        profit_threshold = ParamAccessor.get_param(self, 'concentration_profit_threshold')
        try:
            market_regime = self.market_state.get_market_regime()
            if market_regime == 'strong_uptrend':
                concentration_threshold = max(
                    concentration_threshold,
                    ParamAccessor.get_param(self, 'dynamic_concentration_threshold_strong')
                )
                profit_threshold = max(
                    profit_threshold,
                    ParamAccessor.get_param(self, 'dynamic_concentration_profit_threshold_strong')
                )
        except Exception:
            pass
        if position_ratio > concentration_threshold and profit_pct > profit_threshold:
            self.strategy.log(f"风险控制 - 仓位过度集中({position_ratio:.1%})，执行减仓", level="INFO")

            # 强势上行保留更大核心仓位
            target_ratio = 0.75
            target_value = portfolio_value * target_ratio
            target_size = int(target_value / current_price)
            reduce_size = current_position_size - target_size

            # 确保减仓量有意义
            reduce_size = max(reduce_size, min_meaningful_trade)
            # 单笔卖出规模上限，平滑回撤/避免最大单笔亏损过大
            try:
                max_single_exit_ratio = ParamAccessor.get_param(self, 'max_single_exit_ratio')
            except Exception:
                max_single_exit_ratio = 0.15
            reduce_size_cap = max(min_meaningful_trade, int(current_position_size * max_single_exit_ratio))
            reduce_size = min(reduce_size, reduce_size_cap, current_position_size)

            if reduce_size >= min_meaningful_trade:
                self._last_sell_day = current_day
                return {
                    'action': 'sell',
                    'size': reduce_size,
                    'reason': 'Position concentration risk'
                }
        return None

    def _check_add_position(self, current_price, market_regime):
        """检查是否应该加仓 - 优化版"""
        
        # 先检查是否有持仓
        if not self.strategy.position or self.strategy.position.size <= 0:
            return None
        
        # 获取趋势强度
        trend_strength = self.market_state.get_trend_strength()
        
        # 检查批次限制
        batches_allowed = ParamAccessor.get_param(self, 'batches_allowed')
        if self.added_batches >= batches_allowed:
            if not self._max_batches_logged:
                self.strategy.log(f"达到最大加仓批次{batches_allowed}", level="INFO")
                self._max_batches_logged = True
            return None
        
        # 计算当前盈亏（使用平均成本）
        try:
            avg_entry = self.strategy.get_average_entry_price()
        except Exception:
            avg_entry = getattr(self.strategy, 'entry_price', 0)
        profit_pct = (current_price / avg_entry - 1) if avg_entry and avg_entry > 0 else 0
        
        # 获取参数（不使用硬编码）
        min_position_add_pct = ParamAccessor.get_param(self, 'position_min_add_pct')  # 0.15
        min_meaningful_shares = ParamAccessor.get_param(self, 'min_meaningful_shares')  # 50
        
        # 检查资金（修复：后半段因现金门槛过高导致长期不交易）
        portfolio_value = self.strategy.broker.getvalue()
        cash_available = self.strategy.broker.get_cash()
        # 使用“有意义交易规模”作为最低门槛的主约束；
        # 若仍希望保留部分现金比例要求，则打7折（显著降低阻断概率）
        min_cash_required = max(
            current_price * min_meaningful_shares,
            portfolio_value * min_position_add_pct * 0.3
        )
        if cash_available < min_cash_required:
            return None
        
        # 加仓条件判断（使用参数而非硬编码）
        should_add = False
        reason = ""
        
        # 条件1：盈利加仓（受 allow_profit_add 开关控制 + 止盈/盈利加仓冷却/当日互斥）
        profit_trigger = ParamAccessor.get_param(self, 'profit_trigger_1')  # 0.25（已提高门槛）
        if ParamAccessor.get_param(self, 'allow_profit_add') and profit_pct > profit_trigger:
            # 盈利加仓冷却：防止一天内连环追高
            try:
                if getattr(self, '_profit_add_day', None) == len(self.strategy):
                    return None
                profit_add_cooldown = ParamAccessor.get_param(self, 'profit_add_cooldown_bars')
                if hasattr(self, '_last_profit_add_bar') and (len(self.strategy) - getattr(self, '_last_profit_add_bar', 0) < profit_add_cooldown):
                    return None
            except Exception:
                pass
            # 额外过滤：避免在过热/过高溢价时盈利加仓
            try:
                rsi_ok = True
                ma_premium_ok = True
                strong_ok = True
                if hasattr(self.strategy, 'rsi') and len(self.strategy.rsi) > 0:
                    rsi_cap = ParamAccessor.get_param(self, 'profit_add_rsi_cap')
                    rsi_ok = self.strategy.rsi[0] <= rsi_cap
                if hasattr(self.strategy, 'ma_fast') and len(self.strategy.ma_fast) > 0:
                    premium_cap = ParamAccessor.get_param(self, 'profit_add_ma_fast_premium')
                    fast = float(self.strategy.ma_fast[0])
                    if fast > 0:
                        ma_premium_ok = (current_price - fast) / fast <= premium_cap
                if ParamAccessor.get_param(self, 'profit_add_only_strong'):
                    strong_ok = (market_regime == 'strong_uptrend')
                if not (rsi_ok and ma_premium_ok and strong_ok):
                    return None
            except Exception:
                pass
            # 若当日已发生止盈或仍处在止盈冷却期，则禁止盈利加仓
            cooldown = ParamAccessor.get_param(self, 'take_profit_cooldown_bars')
            if hasattr(self, '_last_take_profit_bar'):
                if (len(self.strategy) - getattr(self, '_last_take_profit_bar', -10**9)) < cooldown:
                    return None
            if getattr(self, '_take_profit_day', None) == len(self.strategy):
                return None
            should_add = True
            reason = f'Profit add at {profit_pct:.1%}'
            self.strategy.log(f"盈利加仓条件满足: {profit_pct:.1%} > {profit_trigger:.1%}", level="DEBUG")
        
        # 条件2：回调加仓（使用dip_base_pct参数）
        elif len(self.strategy.data.high) >= 2:
            # 使用参数化窗口并按日→bars换算（并做安全裁剪）
            window_days = int(ParamAccessor.get_param(self, 'breakout_period') or 24)
            intraday_mode = ParamAccessor.get_param(self, 'intraday_mode')
            if intraday_mode:
                mult = max(1, int(ParamAccessor.get_param(self, 'intraday_multiplier') or 26))
                desired = max(10, window_days * mult)
            else:
                desired = max(10, window_days)
            size = min(desired, len(self.strategy.data.high))
            if size >= 2:
                vals_high = list(self.strategy.data.high.get(size=size))
                vals_high = [v for v in vals_high if pd.notna(v)]
                recent_high = max(vals_high) if vals_high else 0.0
            else:
                recent_high = 0.0
            dip_pct = (recent_high - current_price) / recent_high if recent_high > 0 else 0.0
            dip_threshold = ParamAccessor.get_param(self, 'dip_base_pct')

            # 根据批次动态调整回调要求（百分比增量）
            dip_per_batch = ParamAccessor.get_param(self, 'dip_per_batch')
            required_dip = dip_threshold + (self.added_batches * dip_per_batch)

            # 长时间未加仓时允许小幅降低回调要求，提升中后期触发概率
            try:
                bars_since_last_add = len(self.strategy) - (self._last_add_bar or 0)
                if bars_since_last_add >= 80:
                    required_dip = max(0.06, required_dip - 0.02)
            except Exception:
                pass

            # 额外过滤（修复逻辑）：当价格同时低于中/慢均线时，必须出现RSI反弹信号才允许回调加仓
            try:
                need_ma_filter = ParamAccessor.get_param(self, 'dip_add_require_above_ma')
                need_rsi_bounce = ParamAccessor.get_param(self, 'dip_add_require_rsi_bounce')
                if need_ma_filter and hasattr(self.strategy, 'ma_mid') and hasattr(self.strategy, 'ma_slow'):
                    ma_mid_v = float(self.strategy.ma_mid[0]) if len(self.strategy.ma_mid) > 0 else 0.0
                    ma_slow_v = float(self.strategy.ma_slow[0]) if len(self.strategy.ma_slow) > 0 else 0.0
                    if ma_mid_v > 0 and ma_slow_v > 0 and current_price < ma_mid_v and current_price < ma_slow_v:
                        if need_rsi_bounce and hasattr(self.strategy, 'rsi') and len(self.strategy.rsi) > 1:
                            # 需要明确的反弹：上一bar<30 且 当前>35
                            if not (self.strategy.rsi[-1] < 30 and self.strategy.rsi[0] > 35):
                                return None
                        elif need_rsi_bounce:
                            return None
            except Exception:
                pass

            if dip_pct >= required_dip:
                should_add = True
                reason = f'Dip add at {required_dip:.1%}'
                self.strategy.log(f"回调加仓条件满足: 实际{dip_pct:.1%} ≥ 需求{required_dip:.1%}", level="DEBUG")
        
        # 条件3：强势趋势加仓
        elif trend_strength > 15 and market_regime in ["strong_uptrend", "uptrend"]:
            # 趋势加仓也受 allow_profit_add 控制（避免高位追涨）；与止盈互斥/冷却
            if ParamAccessor.get_param(self, 'allow_profit_add'):
                cooldown = ParamAccessor.get_param(self, 'take_profit_cooldown_bars')
                if hasattr(self, '_last_take_profit_bar'):
                    if (len(self.strategy) - getattr(self, '_last_take_profit_bar', -10**9)) < cooldown:
                        return None
                if getattr(self, '_take_profit_day', None) == len(self.strategy):
                    return None
                should_add = True
                reason = f'Trend add at strength {trend_strength:.1f}'
                self.strategy.log(f"趋势加仓条件满足: 趋势{trend_strength:.1f}", level="DEBUG")

        # 下跌市场禁止“回调加仓”与“趋势加仓”
        if market_regime == 'downtrend':
            # 允许在下跌市的极端超跌反弹中触发一次少量加仓（RSI<30回到>35且放量）
            try:
                if hasattr(self.strategy, 'rsi') and len(self.strategy.rsi) > 1 and self.strategy.rsi[-1] < 30 and self.strategy.rsi[0] > 35:
                    if hasattr(self.strategy, 'volume_ma') and self.strategy.data.volume[0] > self.strategy.volume_ma[0] * 1.5:
                        add_ratio = 0.12
                        target_value = cash_available * add_ratio
                        add_size = int(target_value / current_price)
                        if add_size >= ParamAccessor.get_param(self, 'min_meaningful_shares'):
                            self.strategy.log("下跌市反弹小幅加仓（一次性）", level="INFO")
                            return {
                                'action': 'buy',
                                'size': add_size,
                                'reason': 'Oversold rebound add'
                            }
            except Exception:
                pass
            return None
        
        if not should_add:
            return None
        
        # 计算加仓规模
        batch_decay = ParamAccessor.get_param(self, 'position_batch_decay')  # 0.10
        # 将基础加仓占比参数化
        base_ratio = ParamAccessor.get_param(self, 'position_add_base_pct')  # 使用参数代替硬编码
        add_ratio = base_ratio * (1.0 - batch_decay * self.added_batches)
        # 放缓后期衰减，避免后期加仓过小导致盈利能力不足
        add_ratio = max(add_ratio, base_ratio * 0.5, 0.12)
        
        target_value = cash_available * add_ratio
        add_size = int(target_value / current_price)
        
        # 确保有意义的规模
        if add_size < min_meaningful_shares:
            self.strategy.log(f"加仓规模过小: {add_size} < {min_meaningful_shares}", level="DEBUG")
            return None
        
        # 盈利加仓计数与冷却标记
        try:
            if 'Profit add' in reason:
                self._last_profit_add_bar = len(self.strategy)
                self._profit_add_day = len(self.strategy)
        except Exception:
            pass
        self.strategy.log(f"加仓决策通过 - 规模: {add_size}股, 原因: {reason}", level="CRITICAL")
        
        return {
            'action': 'buy',
            'size': add_size,
            'reason': reason
        }

    def _check_profit_taking(self, profit_pct, passive_mode, min_meaningful_trade,
                            current_position_size, current_day):
        """检查获利了结条件"""
        # 限制单日止盈次数
        if getattr(self, '_take_profit_day', None) != current_day:
            self._take_profit_day = current_day
            self._take_profit_actions_today = 0

        if self._take_profit_actions_today >= ParamAccessor.get_param(self, 'max_take_profit_actions_per_day'):
            return None

        if not passive_mode or profit_pct > 0.35:
            # 直接调用已经优化过的止盈逻辑
            take_profit_size = self._check_enhanced_profit_levels(profit_pct)

            if take_profit_size > 0:
                # 获取当前市场状态
                try:
                    market_regime = self.market_state.get_market_regime()
                except Exception:
                    market_regime = 'sideways'
                # 保留核心仓位（强势/上行市场提高核心比例）
                min_core_ratio = ParamAccessor.get_param(self, 'core_position_min_ratio')
                try:
                    if market_regime == 'strong_uptrend':
                        min_core_ratio = max(min_core_ratio, ParamAccessor.get_param(self, 'dynamic_core_min_ratio_strong'))
                    elif market_regime == 'uptrend':
                        min_core_ratio = max(min_core_ratio, ParamAccessor.get_param(self, 'dynamic_core_min_ratio_uptrend'))
                except Exception:
                    pass
                min_core = int(current_position_size * min_core_ratio)
                max_sellable = max(0, current_position_size - min_core)
                take_profit_size = min(take_profit_size, max_sellable)

                # 确保卖出规模有意义
                take_profit_size = max(take_profit_size, min_meaningful_trade)
                take_profit_size = min(take_profit_size, current_position_size)

                # 额外保护：若最近触发过集中度/回撤保护，则将止盈规模最多限制为现有仓位的40%
                try:
                    recent_protection = False
                    if hasattr(self.strategy.risk_manager, '_drawdown_triggered') and getattr(self.strategy.risk_manager, '_drawdown_triggered'):
                        recent_protection = True
                    # 这里可扩展更多标记
                    if recent_protection:
                        take_profit_size = min(take_profit_size, int(current_position_size * 0.4))
                except Exception:
                    pass

                if take_profit_size >= min_meaningful_trade:
                    self._last_sell_day = current_day
                    # 强势期限制每日止盈次数
                    try:
                        if market_regime == 'strong_uptrend':
                            if self._take_profit_actions_today >= ParamAccessor.get_param(self, 'tp_strong_up_max_actions_per_day'):
                                return None
                    except Exception:
                        pass
                    self._take_profit_actions_today += 1
                    return {
                        'action': 'sell',
                        'size': take_profit_size,
                        'reason': f'Take profit at {profit_pct:.1%}'
                    }

        return None

    def _check_stop_loss(self, *args, **kwargs):
        """已弃用：止损统一由 RiskManager.check_all_stops 管理"""
        return None

    def _get_position_info(self):
        """获取当前仓位信息 - 统一计算，避免重复"""
        current_price = self.strategy.data.close[0]
        position_size = self.strategy.position.size
        portfolio_value = self.strategy.broker.getvalue()

        # 计算盈亏（只计算一次）
        # 使用平均持仓成本计算盈亏
        try:
            avg_entry = self.strategy.get_average_entry_price()
        except Exception:
            avg_entry = getattr(self.strategy, 'entry_price', 0)
        profit_pct = (current_price / avg_entry - 1) if avg_entry and avg_entry > 0 else 0.0

        # 计算持仓天数（只计算一次）
        holding_days = self.strategy.get_holding_days()

        # 仓位价值和比例
        position_value = position_size * current_price
        position_ratio = position_value / portfolio_value if portfolio_value > 0 else 0

        # 当前日期
        current_day = len(self.strategy)

        return {
            'size': position_size,
            'price': current_price,
            'portfolio_value': portfolio_value,
            'position_value': position_value,
            'position_ratio': position_ratio,
            'profit_pct': profit_pct,
            'holding_days': holding_days,
            'current_day': current_day
        }

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

            passive_mode = False
            stop_loss_result = None
            profit_taking_result = None
            add_position_result = None
            concentration_result = None
            small_position_result = None
            sideways_result = None
            long_term_result = None

            # 更新缓存信息
            self._update_position_cache(current_price, market_regime, trend_strength)

            # 获取基础信息（统一计算一次）
            pos_info = self._get_position_info()
            current_day = pos_info['current_day']
            holding_days = pos_info['holding_days']
            profit_pct = pos_info['profit_pct']
            current_position_size = pos_info['size']
            portfolio_value = pos_info['portfolio_value']
            position_value = pos_info['position_value']
            position_ratio = pos_info['position_ratio']

            # 1. 检查并重置仓位状态
            if self._check_position_reset():
                return result

            # 1.1 批次计数“衰减复位”：过了N天自动释放一个批次，避免开局一次性用完导致中期无操作
            try:
                if ParamAccessor.get_param(self, 'enable_add_batches_decay') and self.added_batches > 0:
                    last_add = getattr(self, '_last_add_bar', None)
                    if last_add is not None:
                        decay_days = ParamAccessor.get_param(self, 'add_batches_decay_days')
                        if len(self.strategy) - last_add >= decay_days:
                            self.added_batches = max(0, self.added_batches - 1)
                            self._last_add_bar = len(self.strategy)
                            self.strategy.log(f"加仓批次自然衰减释放：剩余批次 {self.added_batches}", level="INFO")
            except Exception:
                pass

            # 新增：针对极小仓位的特殊处理
            micro_shares = ParamAccessor.get_param(self, 'micro_position_shares')
            micro_ratio = ParamAccessor.get_param(self, 'micro_position_ratio')


            if current_position_size > 0 and current_position_size < micro_shares:
                if position_ratio < micro_ratio * 0.4:  # 极小仓位的40%
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
            if self.strategy.position and current_position_size > 0:  # 添加条件检查
                # 当天与卖出后冷却限制（避免在单日形成加仓/老化减仓风暴）
                try:
                    if getattr(self, '_last_sell_bar', None) is not None:
                        if len(self.strategy) - self._last_sell_bar < ParamAccessor.get_param(self, 'post_sell_cooldown_bars'):
                            add_position_result = None
                        else:
                            add_position_result = self._check_add_position(current_price, market_regime)
                    else:
                        add_position_result = self._check_add_position(current_price, market_regime)
                except Exception:
                    add_position_result = self._check_add_position(current_price, market_regime)
                if add_position_result:
                    # 不要在这里增加批次计数，让notify_order处理
                    self._last_add_bar = len(self.strategy)
                    self.strategy.log(f"执行加仓 - {add_position_result['reason']}", level="CRITICAL")
                    return add_position_result

            # 5.1 老化亏损/期限退出（产生真实的亏损exit，释放仓位/现金，恢复后半段交易活性）
            try:
                per_lot_stop = -abs(ParamAccessor.get_param(self, 'per_lot_stop_loss_pct'))  # e.g., -0.18
                max_hold_bars = ParamAccessor.get_param(self, 'max_holding_bars_loss_exit')  # e.g., 200 bars
                # 若整体持仓亏损超过阈值，或持有时间过长仍为亏损，则减仓一部分
                if (profit_pct <= per_lot_stop) or (profit_pct < 0 and holding_days >= max_hold_bars):
                    # 单日老化退出次数限制
                    if getattr(self, '_aging_exit_day', None) != current_day:
                        self._aging_exit_day = current_day
                        self._aging_exit_count = 0
                    if self._aging_exit_count >= ParamAccessor.get_param(self, 'max_aging_exits_per_day'):
                        return None
                    # 老化退出最小bar间隔，避免同日/相邻bar连发
                    try:
                        min_space = ParamAccessor.get_param(self, 'aging_exit_min_spacing_bars')
                    except Exception:
                        min_space = 6
                    if getattr(self, '_last_aging_exit_bar', None) is not None:
                        if len(self.strategy) - self._last_aging_exit_bar < min_space:
                            return None
                    loss_trim_ratio = 0.20 if profit_pct <= per_lot_stop else 0.15
                    trim_size = max(min_meaningful_trade, int(current_position_size * loss_trim_ratio))
                    # 单笔退出规模上限
                    try:
                        max_single_exit_ratio = ParamAccessor.get_param(self, 'max_single_exit_ratio')
                    except Exception:
                        max_single_exit_ratio = 0.15
                    trim_cap = max(min_meaningful_trade, int(current_position_size * max_single_exit_ratio))
                    trim_size = min(trim_size, trim_cap)
                    if trim_size > 0:
                        self.strategy.log(
                            f"老化亏损退出 - 持有{holding_days}天, 盈亏{profit_pct:.1%}, 减仓{trim_size}",
                            level="CRITICAL"
                        )
                        self._aging_exit_count += 1
                        # 记录卖出bar，限制短期反向加仓
                        self._last_sell_bar = len(self.strategy)
                        self._last_aging_exit_bar = len(self.strategy)
                        return {
                            'action': 'sell',
                            'size': trim_size,
                            'reason': 'Aging loss exit'
                        }
            except Exception:
                pass

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

            # 8. 获利检查 - 确保这部分代码存在且条件正确
            if self.strategy.position and current_position_size > 0 and profit_pct > 0:  # 添加盈利检查
                # 强制启用获利检查，忽略passive_mode
                take_profit_size = self._check_enhanced_profit_levels(profit_pct)
                
                if take_profit_size > 0:
                    # 单日止盈次数限制（与冷却共同生效）
                    if getattr(self, '_take_profit_day', None) != current_day:
                        self._take_profit_day = current_day
                        self._take_profit_actions_today = 0
                    if self._take_profit_actions_today >= ParamAccessor.get_param(self, 'max_take_profit_actions_per_day'):
                        return None

                    # 止盈冷却：避免在同一趋势中连续多次止盈
                    cooldown = ParamAccessor.get_param(self, 'take_profit_cooldown_bars')
                    if hasattr(self, '_last_take_profit_bar') and (current_day - self._last_take_profit_bar) < cooldown:
                        return None
                    # 确保卖出规模有意义 + 单笔退出规模上限
                    take_profit_size = max(take_profit_size, min_meaningful_trade)
                    take_profit_size = min(take_profit_size, current_position_size)
                    try:
                        max_single_exit_ratio = ParamAccessor.get_param(self, 'max_single_exit_ratio')
                    except Exception:
                        max_single_exit_ratio = 0.15
                    size_cap = max(min_meaningful_trade, int(current_position_size * max_single_exit_ratio))
                    take_profit_size = min(take_profit_size, size_cap)

                    # 与上方一致的风险联动限制
                    try:
                        recent_protection = False
                        if hasattr(self.strategy.risk_manager, '_drawdown_triggered') and getattr(self.strategy.risk_manager, '_drawdown_triggered'):
                            recent_protection = True
                        if recent_protection:
                            take_profit_size = min(take_profit_size, int(current_position_size * 0.4))
                    except Exception:
                        pass
                    
                    if take_profit_size >= min_meaningful_trade:
                        self._last_sell_day = current_day
                        self._last_take_profit_bar = current_day
                        self._take_profit_actions_today += 1
                        result = {
                            'action': 'sell',
                            'size': take_profit_size,
                            'reason': f'Take profit at {profit_pct:.1%}'
                        }
                        self.strategy.log(f"获利止盈决策 - {result['reason']}, 数量: {take_profit_size}", level="CRITICAL")
                        return result

            return result

        except Exception as e:
            self.strategy.log(f"Position management error: {str(e)}", level='INFO')
            raise

    def _check_enhanced_profit_levels(self, profit_pct):
        """根据 TradingParameters 动态计算分级止盈规模
        逻辑说明：
        1. profit_trigger_1、profit_trigger_2 分别代表第一次、第二次主动止盈触发点
        2. 低于 trigger_1 不止盈；介于 trigger_1~trigger_2 按 25%-40% 分级；
           高于 trigger_2 以后最多一次性卖出 60% 仓位，保留核心仓位。
        3. 任何额外技术指标过滤依旧保留（如均线死叉、RSI 超买），但不再影响阈值本身。
        """
        # 获取动态阈值
        tp1 = ParamAccessor.get_param(self, 'profit_trigger_1', 0.30)  # 降低到30%
        tp2 = ParamAccessor.get_param(self, 'profit_trigger_2', 0.60)  # 降低到60%
        if profit_pct >= tp1:
            self.strategy.log(f"达到止盈点: {profit_pct:.1%}", level="INFO")

        # 新增：根据持仓时间调整
        holding_days = self.strategy.get_holding_days()
        if holding_days < 30:  # 短期持仓
            # 保持原有止盈
            pass
        elif holding_days < 90:  # 中期持仓
            tp1 *= 1.2  # 提高20%
            tp2 *= 1.2
        else:  # 长期持仓
            tp1 *= 1.5  # 提高50%
            tp2 *= 1.5
        # 日志
        self.strategy.log(f"检查止盈 - 当前盈利: {profit_pct:.1%} (阈值 {tp1:.0%}/{tp2:.0%})", level="INFO")

        # 首先，如果盈利低于第一阈值，直接返回 0
        if profit_pct < tp1:
            return 0

        current_position_size = self.strategy.position.size

        # --- 技术指标过滤（沿用原有逻辑） ---
        should_hold = True
        try:
            # 均线死叉
            if hasattr(self.strategy, 'ma_fast') and hasattr(self.strategy, 'ma_slow'):
                if len(self.strategy.ma_fast) > 0 and len(self.strategy.ma_slow) > 0:
                    if self.strategy.ma_fast[0] < self.strategy.ma_slow[0]:
                        should_hold = False
            # RSI 超买
            if should_hold and hasattr(self.strategy, 'rsi') and len(self.strategy.rsi) > 0:
                if self.strategy.rsi[0] > 75:
                    should_hold = False
        except Exception:
            pass

        if should_hold:
            # 如果技术面允许继续持有，则分级止盈
            if profit_pct >= tp2:
                return int(current_position_size * 0.60)
            # 在 tp1~tp2 区间线性映射 （tp1→25%，tp2→40%）
            scale = (profit_pct - tp1) / max(1e-9, (tp2 - tp1))
            pct = 0.25 + 0.15 * min(1, scale)  # 0.25~0.40
            return int(current_position_size * pct)
        else:
            # 技术指标不再支持持有，卖出 50%
            return int(current_position_size * 0.50)
            
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

            if self.strategy.position:
                # 1) 计算当前PnL%
                try:
                    avg_entry = self.strategy.get_average_entry_price()
                except Exception:
                    avg_entry = getattr(self.strategy, 'entry_price', 0)
                if not avg_entry or avg_entry == 0:
                    profit_pct = 0.0
                else:
                    profit_pct = (current_price / avg_entry) - 1

                # 2) 计算持仓比例
                position_value = self.strategy.position.size * current_price
                position_pct = position_value / portfolio_value

                # 直接更新缓存，不需要检查变化
                self._position_cache.update({
                    'profit_pct': profit_pct,
                    'position_value': position_value,
                    'position_pct': position_pct,
                    'market_regime': market_regime,
                    'trend_strength': trend_strength
                })

            self._last_update = current_bar

        except Exception as e:
            self.strategy.log(f"Cache update error: {str(e)}", level='INFO')
            raise
    def calculate_kelly_position(self):
        """
        使用凯利公式计算最优仓位比例 - 基于收益率版本（滚动窗口稳定化）
        """
        try:
            # 获取已完成的交易
            if not hasattr(self.strategy, 'trade_manager'):
                return 0.3  # 默认30%

            closed_trades = [t for t in self.strategy.trade_manager.executed_trades
                            if t.get('status') == 'closed' and t.get('pnl') is not None]

            # 需要足够的交易样本
            min_trades = 20
            if len(closed_trades) < min_trades:
                self.strategy.log(
                    f"凯利公式：交易样本不足({len(closed_trades)}<{min_trades})，使用默认仓位",
                    level="INFO"
                )
                return 0.3  # 默认30%

            # 使用最近的交易（滚动窗口）
            recent_trades = closed_trades[-100:] if len(closed_trades) > 100 else closed_trades

            # 计算基于收益率的盈亏
            returns = []
            for trade in recent_trades:
                if trade.get('entry_price', 0) > 0 and trade.get('size', 0) > 0:
                   # 计算该笔交易的收益率
                    entry_cost = trade['entry_price'] * trade['size']
                    if entry_cost > 0:
                        return_rate = trade['pnl'] / entry_cost
                        # 高风险策略：对大幅盈利给予更高权重
                        if return_rate > 0.20:  # 20%以上的收益
                            weight = 1.5
                        elif return_rate > 0.10:  # 10-20%的收益
                            weight = 1.25
                        else:
                            weight = 1.0
                        
                        # 将加权后的收益率加入列表
                        returns.extend([return_rate] * int(weight))
                        
                        # 记录超额收益
                        if return_rate > 0.30 and ParamAccessor.get_param(self, 'debug_mode'):
                            self.strategy.log(f"发现超额收益交易: {return_rate:.1%}", level="INFO")

            if not returns:
                return 0.3

            # 计算胜率
            winning_returns = [r for r in returns if r > 0]
            if len(winning_returns) == 0:
                return 0.15  # 全部亏损时使用15%

            win_rate = len(winning_returns) / len(returns)

            # 计算平均盈亏比（基于收益率）
            avg_win_return = np.mean(winning_returns)
            losing_returns = [r for r in returns if r <= 0]

            if len(losing_returns) == 0:
                # 全部盈利（罕见）
                win_loss_ratio = 2.0
            else:
                avg_loss_return = abs(np.mean(losing_returns))
                win_loss_ratio = avg_win_return / avg_loss_return if avg_loss_return > 0 else 1.5

            # 凯利公式：f = (p * b - q) / b
            kelly_fraction = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio

            # 使用更合理的部分凯利
            kelly_multiplier = ParamAccessor.get_param(self, 'kelly_multiplier') or 0.5
            conservative_kelly = kelly_fraction * kelly_multiplier

            # 根据账户表现调整
            current_value = self.strategy.broker.getvalue()
            initial_value = self.strategy.broker.startingcash
            account_return = (current_value / initial_value - 1)

            # 调整系数优化
            if account_return < -0.10:  # 亏损超过10%
                adjustment = 0.6
            elif account_return < 0:  # 小幅亏损
                adjustment = 0.8
            elif account_return > 0.50:  # 盈利超过50%
                adjustment = 1.4
            elif account_return > 0.20:  # 盈利20-50%
                adjustment = 1.2
            else:
                adjustment = 1.0

            # 应用调整
            final_kelly = conservative_kelly * adjustment

            # 仓位限制使用参数
            min_position = ParamAccessor.get_param(self, 'min_kelly_position') or 0.25
            max_position = ParamAccessor.get_param(self, 'max_kelly_position') or 0.70
            final_position = max(min_position, min(max_position, final_kelly))

            # 新增：对高质量信号提高仓位
            if self.market_state.get_trend_strength() > 10:
                final_position = min(final_position * 1.3, 0.8)  # 强趋势时提高30%

            # 记录凯利计算详情
            self.strategy.log(
                f"凯利公式(改进版) - 胜率: {win_rate:.1%}, 盈亏比: {win_loss_ratio:.2f}, "
                f"原始凯利: {kelly_fraction:.1%}, 保守凯利: {conservative_kelly:.1%}, "
                f"账户收益: {account_return:.1%}, 最终仓位: {final_position:.1%}",
                level="INFO"
            )

            return final_position

        except Exception as e:
            self.strategy.log(f"凯利公式计算错误: {str(e)}", level="ERROR")
            return 0.4  # 出错时返回40%而不是30%

    def _calculate_position_size(self):
        """统一的仓位计算 - 支持凯利公式和趋势跟踪"""
        try:
            cash_available = self.strategy.broker.get_cash()
            current_price = self.strategy.data.close[0]
            portfolio_value = self.strategy.broker.getvalue()

            # 获取仓位计算模式
            scaling_mode = ParamAccessor.get_param(self, 'position_scaling_mode')

            # 先判断是否为加仓场景
            is_adding_position = self.strategy.position and self.strategy.position.size > 0

            if scaling_mode == "kelly":
                kelly_fraction = self.calculate_kelly_position()

                if is_adding_position:
                    # 加仓场景：计算增量仓位
                    current_position_value = self.strategy.position.size * current_price
                    current_position_ratio = current_position_value / portfolio_value

                    # 如果当前仓位低于凯利建议，允许加仓
                    if current_position_ratio < kelly_fraction * 0.8:
                        # 计算加仓比例，考虑批次衰减
                        batches = self.added_batches
                        batch_decay = ParamAccessor.get_param(self, 'position_batch_decay')

                        # 提高加仓比例
                        remaining_space = kelly_fraction - current_position_ratio
                        add_ratio = remaining_space * (1.0 - batch_decay * batches * 0.5)  # 减缓衰减
                        add_ratio = max(add_ratio, 0.15)  # 从0.1提高到0.15

                        target_fraction = min(add_ratio, remaining_space)
                    else:
                        # 已接近目标仓位，不再加仓
                        self.strategy.log(f"仓位{current_position_ratio:.1%}接近目标{kelly_fraction:.1%}，跳过加仓", level="INFO")
                        return 0
                else:
                    # 新建仓位：直接使用凯利公式结果
                    target_fraction = min(kelly_fraction * 1.2, 0.8)  # 放大1.2倍，但不超过80%

            else:
                # 原有的趋势跟踪模式
                # ===== 趋势强度评估 =====
                trend_score = 0

                # 1. 均线排列
                if (self.strategy.ma_fast[0] > self.strategy.ma_slow[0] and
                    self.strategy.ma_slow[0] > self.strategy.ma_mid[0]):
                    trend_score += 2

                # 2. 价格动量
                if len(self.strategy.data.close) >= 20:
                    momentum_20d = (current_price / self.strategy.data.close[-20] - 1)
                    if momentum_20d > 0.15:  # 20天涨15%以上
                        trend_score += 2
                    elif momentum_20d > 0.08:
                        trend_score += 1

                # 3. 相对强度
                if hasattr(self.strategy, 'rsi') and self.strategy.rsi[0] > 60:
                    trend_score += 1

                # ===== 仓位决策 =====
                if self.strategy.position:
                    # 已有仓位：趋势跟踪加仓
                    current_position_value = self.strategy.position.size * current_price
                    current_position_ratio = current_position_value / portfolio_value

                    # 只在强趋势中加仓
                    if trend_score >= 3 and current_position_ratio < 0.6:
                        batches = getattr(self, 'added_batches', 0)
                        batch_decay = ParamAccessor.get_param(self, 'position_batch_decay')
                        add_ratio = 0.2 * (0.7 ** batches)  # 保持原逻辑
                        target_fraction = min(add_ratio, 0.8 - current_position_ratio)
                    else:
                        target_fraction = 0
                else:
                    # 新建仓位：根据趋势强度
                    if trend_score >= 4:
                        target_fraction = 0.8  # 强趋势80%
                    elif trend_score >= 2:
                        target_fraction = 0.6  # 中等趋势60%
                    else:
                        target_fraction = 0.5  # 弱趋势50%

            # ===== 计算股数 =====
            if target_fraction > 0:
                target_value = portfolio_value * target_fraction
                position_size = int(target_value / current_price)
            else:
                position_size = 0

            # ===== 确保有意义的交易规模 =====
            min_position_add_pct = ParamAccessor.get_param(self, 'position_min_add_pct')
            min_shares = max(50, int(portfolio_value * min_position_add_pct / current_price))

            if position_size < min_shares:
                return 0

            # 修正日志输出
            if scaling_mode == "kelly":
                self.strategy.log(
                    f"仓位计算[kelly] - 凯利比例: {kelly_fraction:.1%}, "
                    f"目标仓位: {target_fraction:.1%}, 股数: {position_size}",
                    level="CRITICAL"
                )
            else:
                self.strategy.log(
                    f"仓位计算[趋势] - 趋势得分: {trend_score}, "
                    f"目标仓位: {target_fraction:.1%}, 股数: {position_size}",
                    level="CRITICAL"
                )

            return position_size

        except Exception as e:
            self.strategy.log(f"仓位计算错误: {str(e)}", level='ERROR')
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

            # 1/2/3. 时间控制 + 批次控制 + 资金检查（完全内联）
            # 时间间隔
            if self.last_add_bar is not None:
                bars_since_last_add = len(self.strategy) - self.last_add_bar
                if bars_since_last_add < min_bars_between_trades:
                    return False
            # 批次限制
            if self.added_batches >= batches_allowed:
                return False
            # 资金检查：至少能买到最小交易单位
            cash_available = self.strategy.broker.get_cash()
            min_required_cash = current_price * min_position_size * 1.01
            if cash_available < min_required_cash:
                return False

            # 4. 根据是否有仓位分别处理
            if self.strategy.position:
                return self._check_position_dip(current_price, required_dip_pct)
            else:
                return self._check_new_position_dip(current_price, base_dip_pct)

        except Exception as e:
            self.strategy.log(f"回调买入逻辑错误: {str(e)}")
            return False

    def _check_position_dip(self, current_price, required_dip_pct):
        """检查持仓时的加仓回调条件 - 完全重写，修复逻辑错误"""

        # 获取市场状态
        market_regime = self.market_state.get_market_regime()
        trend_strength = self.market_state.get_trend_strength()
        holding_days = self.strategy.get_holding_days()

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

        if current_dip_pct < base_dip_pct * 0.8:  # 给予20%的宽容度
            # 如果市场状态良好，进一步放宽
            if market_regime in ["uptrend", "strong_uptrend", "sideways"] and trend_strength > 0:
                self.strategy.log(
                    f"市场状态良好({market_regime})，放宽新建仓要求",
                    level="INFO"
                )
                return True  # 直接允许建仓
            return False

        self.strategy.log(f"[参数影响] 新建仓位满足回调条件 {current_dip_pct:.4f} >= {base_dip_pct:.4f}", level="DEBUG")

        # 新建仓需要额外确认
        if not self._check_new_position_confirmation(market_regime, trend_strength):
            return False

        # 如果回调幅度很大，实施更激进的入场策略
        if current_dip_pct > base_dip_pct * 2.0:
            # 强势市场中的大幅回调，可能是最佳买入机会
            if market_regime in ["strong_uptrend", "uptrend"]:
                # 使用RSI确认超跌
                if hasattr(self.strategy, 'rsi') and self.strategy.rsi[0] < 30:
                    self.strategy.log(
                        f"强势市场超跌机会 - RSI:{self.strategy.rsi[0]:.1f}, "
                        f"回调幅度:{current_dip_pct:.1%}",
                        level="CRITICAL"
                    )
                    return True
            # 盘整市场中的大幅回调，需要更多确认
            elif market_regime == "sideways" and trend_strength > 0:
                # 使用MACD确认底部
                if (hasattr(self.strategy, 'macd') and 
                    self.strategy.macd.macd[0] > self.strategy.macd.signal[0]):
                    self.strategy.log(
                        f"盘整市场潜在反弹 - 回调:{current_dip_pct:.1%}, "
                        f"趋势强度:{trend_strength:.1f}, MACD金叉",
                        level="CRITICAL"
                    )
                    return True
            # 下跌市场中保持谨慎，需要很强的反转信号
            elif market_regime == "downtrend" and trend_strength > -5:
                # 需要同时满足RSI超跌和MACD金叉
                if (hasattr(self.strategy, 'rsi') and 
                    hasattr(self.strategy, 'macd') and 
                    self.strategy.rsi[0] < 25 and  # 极度超跌
                    self.strategy.macd.macd[0] > self.strategy.macd.signal[0]):
                    self.strategy.log(
                        f"下跌市场强势反转信号 - RSI:{self.strategy.rsi[0]:.1f}, "
                        f"回调:{current_dip_pct:.1%}, MACD金叉",
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
        self.last_trade_pnl = 0          # 记录最后一次交易的盈亏
        self.last_loss_day = 0           # 记录最后一次亏损的日期
        self.last_trade_bar = 0          # 记录最后一次交易的bar
        self.consecutive_losses = 0      # 记录连续亏损次数
        self.last_trade_was_loss = False # 记录最后一次交易是否亏损

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
                                          'stop_atr', 'profit_trigger_1']:
                            key_params[param_name] = value
                    except Exception as e:
                        print(f"设置参数 {param_name} 失败: {str(e)}")

            print(f"\n策略初始化 - {trading_params.symbol} | 资金: ${trading_params.initial_cash:,.0f}")

        # 设置symbol属性
        self.symbol = self.trading_params.symbol if self.trading_params else ""

        # ========== 新增：统一计算剩余持仓的加权平均成本 ==========
        def get_average_entry_price_inner():
            try:
                if hasattr(self, 'trade_manager'):
                    open_trades = [t for t in self.trade_manager.executed_trades
                                   if t.get('status') == 'open' and t.get('size', 0) > 0]
                    if open_trades:
                        total_cost = sum(t['entry_price'] * t['size'] for t in open_trades)
                        total_size = sum(t['size'] for t in open_trades)
                        if total_size > 0:
                            return total_cost / total_size
                # 回退使用策略级 entry_price
                return getattr(self, 'entry_price', 0)
            except Exception:
                return getattr(self, 'entry_price', 0)

        # 提供为实例方法
        self.get_average_entry_price = get_average_entry_price_inner

        # 初始化技术指标
        self._initialize_indicators()

        # 初始化管理器
        self.trade_manager = TradeManager()
        self.market_state = MarketState(self.trading_params if self.trading_params else self.params)
        self.risk_manager = RiskManager(self.trading_params if self.trading_params else self.params)
        # 已合并StopManager至RiskManager，不再保留stop_manager别名
        self.position_manager = PositionManager(self.trading_params if self.trading_params else self.params)

        # 为TradeManager设置参数
        self.trade_manager.params = self.trading_params if self.trading_params else self.params

        # 交易状态 - 初始化基本属性
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

        # 初始化参数缓存
        self._param_cache = {}
        if self.trading_params:
            for attr_name in dir(self.trading_params):
                if not attr_name.startswith('_') and not callable(getattr(self.trading_params, attr_name)):
                    self._param_cache[attr_name] = getattr(self.trading_params, attr_name)

        # 初始化通知器
        self.notifier = None
        if self.trading_params and (self.trading_params.serverchan_sendkey or self.trading_params.wecom_webhook_url):
            self.notifier = ServerChanNotifier(
                self.trading_params.serverchan_sendkey,
                wecom_webhook_url=self.trading_params.wecom_webhook_url,
                serverchan_channel=self.trading_params.serverchan_channel,
            )
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

    def _initialize_indicators(self):
        """使用优化参数初始化技术指标"""
        try:
            if not hasattr(self, 'trading_params') or self.trading_params is None:
                raise ValueError("无法访问trading_params，请确保已正确设置")
            
            # 修复：确保有足够的数据
            min_required = max(
                self.trading_params.long_ma,
                self.trading_params.macd_slow,
                self.trading_params.volatility_window,
                100  # 最少需要100个数据点
            )
            
            if len(self.data) < min_required:
                self.log(f"数据不足以初始化指标，需要{min_required}个数据点，当前只有{len(self.data)}个", level="WARNING")
                # 设置标志位，稍后重试
                self._indicators_initialized = False
                return
            
            # 获取基本参数值
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

            self.log(f"技术指标初始化 - ATR周期:{atr_period}, RSI周期:{rsi_period}, "
                    f"波动率窗口:{volatility_window}, 成交量调整:{volume_adjust}", level="INFO")

            # 移动平均线 - 使用获取的参数值
            self.ma_fast = bt.indicators.SMA(
                self.data.close,
                period=min(fast_ma, len(self.data.close)-1)
            )
            self.ma_slow = bt.indicators.SMA(
                self.data.close,
                period=min(slow_ma, len(self.data.close)-1)
            )
            self.ma_mid = bt.indicators.SMA(
                self.data.close,
                period=min(mid_ma, len(self.data.close)-1)
            )
            self.ma_long = bt.indicators.SMA(
                self.data.close,
                period=min(long_ma, len(self.data.close)-1)
            )

            # MACD指标
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
                period=atr_ma_period
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

            # 设置初始化完成标志
            self._indicators_initialized = True
            self.log(f"技术指标初始化完成", level="INFO")

        except Exception as e:
            self.log(f"指标初始化错误: {str(e)}")
            self._indicators_initialized = False
            raise

    def get_holding_days(self):
        """获取当前持仓天数的统一方法"""
        if not self.position or not hasattr(self, 'entry_time'):
            return 0

        # 日内模式下的特殊处理
        if hasattr(self.trading_params, 'intraday_mode') and self.trading_params.intraday_mode:
            bars_held = len(self) - self.entry_time

            # 根据数据间隔计算实际天数
            if self.trading_params.data_interval == "15m":
                days = bars_held / 26  # 每天26根15分钟K线
            elif self.trading_params.data_interval == "30m":
                days = bars_held / 13
            elif self.trading_params.data_interval == "5m":
                days = bars_held / 78
            elif self.trading_params.data_interval == "1h":
                days = bars_held / 7
            else:
                days = bars_held

            return max(1, int(days))  # 至少返回1天
        else:
            # 日线模式保持原有逻辑
            return len(self) - self.entry_time

    def _is_trading_hours(self, dt):
        """统一使用交易所日历判断（含半日市），与 SimpleLiveDataFeed 保持一致"""
        if not self.trading_params.intraday_mode:
            return True
        session, tradable = get_market_session(
            dt,
            include_prepost=getattr(self.trading_params, 'prepost', True)
        )
        # 如果显式排除盘前/盘后，则仅 regular 可交易
        if getattr(self.trading_params, 'exclude_premarket', False) or getattr(self.trading_params, 'exclude_afterhours', False):
            return session == 'regular'
        return tradable

    # 在 EnhancedStrategy 类中添加
    def get_min_meaningful_trade(self):
        """获取最小有意义的交易规模"""
        portfolio_value = self.broker.getvalue()
        portfolio_factor = max(1.0, portfolio_value / 100000)

        return max(15, int(15 * portfolio_factor))

    def check_stops(self):
        """统一到RiskManager（已合并）"""
        return self.risk_manager.check_all_stops()

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
                # 标记已进入真实实时数据阶段
                self._is_live = True
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
        # 激进策略：大幅降低最小数据要求
        min_required_bars = 24  # 激进：从100降到24
        
        if len(self) < min_required_bars:
            if len(self) % 10 == 0:
                self.log(f"数据积累中 - 当前: {len(self)}/{min_required_bars}", level="DEBUG")
            return

        # 实时模式：在数据未进入LIVE之前不执行交易，以免历史预热期间触发交易
        try:
            if getattr(self.trading_params, 'live_mode', False):
                if not hasattr(self, '_is_live'):
                    self._is_live = False
                if not self._is_live:
                    current_price = self.data.close[0]
                    self.risk_manager.update_tracking(current_price, self.broker.getvalue())
                    self.portfolio_values.append(self.broker.getvalue())
                    return
        except Exception:
            pass
        
        # ===== 关键修复：确保指标真正初始化 =====
        if not hasattr(self, '_indicators_initialized') or not self._indicators_initialized:
            # 尝试初始化指标
            if len(self.data) >= min_required_bars:
                try:
                    self._initialize_indicators()
                    # 验证指标是否真的初始化成功
                    if hasattr(self, 'ma_fast') and hasattr(self, 'ma_slow'):
                        self._indicators_initialized = True
                        self.log("指标动态初始化成功", level="CRITICAL")
                    else:
                        # 创建简化版指标作为备用
                        self.log("创建备用指标", level="CRITICAL")
                        self.ma_fast = bt.indicators.SMA(self.data.close, period=12)
                        self.ma_slow = bt.indicators.SMA(self.data.close, period=24)
                        self.ma_mid = bt.indicators.SMA(self.data.close, period=48)
                        self.ma_long = bt.indicators.SMA(self.data.close, period=96)
                        self.macd = bt.indicators.MACD(self.data.close)
                        self.atr = bt.indicators.ATR(self.data)
                        self.rsi = bt.indicators.RSI(self.data.close)
                        self.volume_ma = bt.indicators.SMA(self.data.volume, period=20)
                        self.volatility = bt.indicators.StdDev(self.data.close, period=20)
                        self._indicators_initialized = True
                except Exception as e:
                    self.log(f"指标初始化异常: {str(e)}，创建最小指标集", level="CRITICAL")
                    # 创建最小必需指标集
                    try:
                        self.ma_fast = bt.indicators.SMA(self.data.close, period=5)
                        self.ma_slow = bt.indicators.SMA(self.data.close, period=10)
                        self.ma_mid = bt.indicators.SMA(self.data.close, period=20)
                        self.ma_long = bt.indicators.SMA(self.data.close, period=50)
                        self.macd = bt.indicators.MACD(self.data.close, period_me1=8, period_me2=17, period_signal=6)
                        self.atr = bt.indicators.ATR(self.data, period=14)
                        self.rsi = bt.indicators.RSI(self.data.close, period=14)
                        self.volume_ma = bt.indicators.SMA(self.data.volume, period=10)
                        self.volatility = bt.indicators.StdDev(self.data.close, period=10)
                        self._indicators_initialized = True
                        self.log("最小指标集创建成功", level="CRITICAL")
                    except:
                        self.log("无法创建指标，跳过本轮", level="ERROR")
                        return
        
        # 添加额外的安全检查
        if not hasattr(self, 'ma_fast') or not hasattr(self, 'ma_slow'):
            self.log("指标仍未就绪，跳过", level="WARNING")
            return
        
        # 继续原有的调试日志
        self.log(f"next 执行 - 当前日期: {self.data.datetime.date(0)}", level="DEBUG")
        
        # ========== 1. 基础检查 ==========
        if len(self.data) < 5:
            return

        # ========== 新增：日内交易时段过滤 ==========
        if self.trading_params.intraday_mode:
            current_time = self.data.datetime.datetime(0)
            if not self._is_trading_hours(current_time):
                # 非交易时段，只更新追踪但不执行交易
                current_price = self.data.close[0]
                self.risk_manager.update_tracking(current_price, self.broker.getvalue())
                self.portfolio_values.append(self.broker.getvalue())
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

        # ========== 5. 定期账户状态输出（可关闭/按参数频率）==========
        if ParamAccessor.get_param(self, 'enable_periodic_account_log', False):
            interval_bars = max(1, ParamAccessor.get_param(self, 'periodic_log_interval_bars', 100))
            if len(self.data) % interval_bars == 0:
                total_value = self.broker.getvalue()
                cash = self.broker.get_cash()
                position_value = self.position.size * self.data.close[0] if self.position else 0

                self.log(
                    f"账户状态 - 总价值:${total_value:.2f}, 现金:${cash:.2f}, "
                    f"持仓价值:${position_value:.2f}, 持仓数量:{self.position.size if self.position else 0}",
                    level="CRITICAL"
                )

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

            # 当日亏损保护（仅日内模式）- 每个交易日最多触发一次，需在每日任意bar评估
            try:
                if self.trading_params.intraday_mode and hasattr(self, 'daily_value') and self.daily_value:
                    port_now = self.broker.getvalue()
                    day_ret = (port_now / self.daily_value - 1.0)
                    guard_pct = ParamAccessor.get_param(self, 'daily_loss_guard_pct')
                    if (day_ret < -abs(guard_pct)
                        and self.position and self.position.size > 0
                        and getattr(self, '_last_daily_guard_date', None) != current_date):
                        sell_ratio = ParamAccessor.get_param(self, 'daily_loss_guard_sell_ratio')
                        trim = max(ParamAccessor.get_param(self, 'min_meaningful_shares'), int(self.position.size * sell_ratio))
                        if trim > 0:
                            self.log(f"当日亏损保护触发：日内回撤{day_ret:.1%}，减仓{trim}", level="CRITICAL")
                            self.order = self.sell(size=trim)
                            self._last_daily_guard_date = current_date
                            return
            except Exception:
                pass

            # ========== 14. 更新市场状态 ==========
            market_regime = self.market_state.get_market_regime()
            trend_strength = self.market_state.get_trend_strength()

            # ========== 15. 判断是否为小仓位 ==========
            is_small_position = False
            if self.position and self.position.size > 0:
                portfolio_value = self.broker.getvalue()
                position_value = self.position.size * current_price
                position_ratio = position_value / portfolio_value

                # 使用参数而非硬编码
                small_pos_threshold = ParamAccessor.get_param(self, 'small_position_ratio')
                if position_ratio < small_pos_threshold:
                    is_small_position = True
                    self.log(f"检测到小仓位：{self.position.size}股(占比{position_ratio:.1%})，"
                            f"继续寻找买入机会", level="INFO")

            # ========== 16. 主要交易逻辑 ==========
            if self.position and not is_small_position:
                # === 正常持仓管理逻辑 ===
                self._was_in_position = True

                # 16.1 检查止损
                if self.risk_manager.check_all_stops():
                    self.log(f"止损触发 - 价格: {current_price:.2f}", level='INFO')
                    self.order = self.close()
                    return

                # 16.2 获取仓位管理建议
                position_action = self.position_manager.manage_position(
                    current_price,
                    market_regime,
                    trend_strength
                )

                # 16.3 若 StopManager 设定了保护性止盈的 pending action，优先执行
                if hasattr(self, '_pending_action') and self._pending_action:
                    pa = self._pending_action
                    self.log(f"【准备卖出】 - 原因: {pa.get('reason','pending')}, 数量: {pa['size']}股", level="INFO")
                    self.order = self.sell(size=pa['size'])
                    self._pending_action = None
                    return

                # 16.4 执行建议的操作（容错：position_action 可能为None）
                if not position_action:
                    # 无操作建议：若长时间无交易且处于可操作区间，允许触发微量再平衡，避免“后半段无交易”
                    try:
                        stagnation_bars = ParamAccessor.get_param(self, 'stagnation_relax_bars')
                        if hasattr(self, 'last_trade_bar') and (len(self) - self.last_trade_bar) >= stagnation_bars:
                            # 若趋势强且现金充足，尝试小额增持至目标（限额）
                            size_hint = int(self.broker.get_cash() / self.data.close[0] * 0.05)
                            if size_hint > 0:
                                self.log("久未交易微量再平衡：买入", level="INFO")
                                self.order = self.buy(size=size_hint)
                                self.last_trade_bar = len(self)
                                return
                    except Exception:
                        pass
                    return

                action = position_action.get('action')
                size = position_action.get('size', 0)

                if action == 'buy':
                    # 加仓
                    self.log(f"加仓决策 - 规模: {size}股, 原因: {position_action.get('reason','')}",
                            level="INFO")
                    if size and size > 0:
                        self.order = self.buy(size=size)
                    # 记录买入价格
                    self.position_manager._last_buy_price = current_price
                    self.position_manager._buy_price_history.append({
                        'price': current_price,
                        'size': size,
                        'bar': len(self)
                    })
                    self.position_manager.added_batches += 1
                    self.position_manager.last_add_bar = len(self)
                    self.log(f"加仓批次更新: {self.position_manager.added_batches}/{ParamAccessor.get_param(self.position_manager, 'batches_allowed')}", 
                            level="CRITICAL")  # 添加调试日志
                    return

                elif action == 'sell':
                    # 减仓或平仓
                    self.log(f"【准备卖出】 - 原因: {position_action.get('reason','')}, 数量: {size}股",
                            level="INFO")

                    # 检查是否为小额获利了结
                    if 'reason' in position_action and 'Take profit' in position_action['reason']:
                        profit_pct = (current_price / self.entry_price - 1) if self.entry_price > 0 else 0
                        if profit_pct < 0.10 and size < self.position.size * 0.2:
                            self.log(f"跳过小额获利了结 - 利润{profit_pct:.1%}，卖出比例过小", level="INFO")
                            return

                    if size and size > 0:
                        self.order = self.sell(size=size)
                    return

            else:
                # === 空仓或小仓位时的买入逻辑 ===

                # 16.4 快速重置（激进版）
                if hasattr(self, '_was_in_position') and self._was_in_position:
                    self._was_in_position = False

                # 16.7 检查买入信号（使用新的激进方法）
                base_buy = self.market_state.get_buy_signal()
                # 可选：定期打印买入信号检查
                if ParamAccessor.get_param(self, 'enable_periodic_signal_log', False):
                    interval_bars = max(1, ParamAccessor.get_param(self, 'periodic_log_interval_bars', 100))
                    if len(self) % interval_bars == 0:
                        self.log(f"买入信号检查 - base_buy: {base_buy}, 当前价: {current_price:.2f}", level="CRITICAL")

                # 16.8 检查市场恢复（保持原有逻辑）
                if hasattr(self, '_last_crash_bar'):
                    bars_since_crash = len(self) - self._last_crash_bar
                    # 这里可以添加市场恢复的判断逻辑

                # 16.9 检查回调条件
                dip_signal = self.position_manager._should_buy_the_dip(current_price)
                if ParamAccessor.get_param(self, 'enable_periodic_signal_log', False):
                    interval_bars = max(1, ParamAccessor.get_param(self, 'periodic_log_interval_bars', 100))
                    if len(self) % interval_bars == 0:
                        self.log(f"回调信号检查 - dip_signal: {dip_signal}", level="CRITICAL")

                # 16.11 构建信号来源
                signal_source = []
                if base_buy:
                    signal_source.append("激进买入信号")
                if dip_signal:
                    signal_source.append("回调信号")

                # 16.12 确定是否买入（激进版）
                should_buy = base_buy or (dip_signal and self.market_state.get_trend_strength() > 0)

                # 16.13 执行买入
                if should_buy:
                    # 检查最小交易间隔（激进版缩短）
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
                        self.log(f"【激进买入】 - 信号来源: {'+'.join(signal_source)}, 数量: {size}股", level='CRITICAL')
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

                # 获取当前交易日期与精确时间
                current_date = self.data.datetime.date(0)
                try:
                    current_dt = self.data.datetime.datetime(0)
                except Exception:
                    current_dt = pd.Timestamp(current_date).to_pydatetime()
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
                        'entry_dt': current_dt,
                        'entry_bar': len(self),
                        'entry_price': executed_price,
                        'exit_date': None,
                        'exit_price': None,
                        'size': executed_size,
                        'orig_size': executed_size,
                        'commission': executed_comm,
                        # 进场不计入负pnl，避免统计被误当作亏损交易
                        'pnl': 0.0,
                        'commission_remaining': executed_comm,
                        'type': 'entry',
                        'status': 'open'
                    }

                    # 添加到 trade_manager
                    self.trade_manager.add_trade(trade_dict)

                    # 追踪记录
                    if not hasattr(self, '_executed_trades'):
                        self._executed_trades = []
                    self._executed_trades.append(trade_dict.copy())

                    # 继续更新策略级别信息：用加权平均成本更新 entry_price
                    if hasattr(self, 'position') and self.position and self.position.size > 0:
                        # 计算新的加权平均入场价
                        try:
                            prev_size = max(0, self.position.size - executed_size)
                        except Exception:
                            prev_size = 0
                        if prev_size > 0 and hasattr(self, 'entry_price') and self.entry_price > 0:
                            weighted_cost = self.entry_price * prev_size + executed_price * executed_size
                            new_avg_entry = weighted_cost / (prev_size + executed_size)
                        else:
                            new_avg_entry = executed_price
                        self.entry_price = new_avg_entry
                    else:
                        self.entry_price = executed_price
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
                        # 使用本地 Asia/Shanghai 时区，避免误以为传入的是美东时区
                        now = datetime.datetime.now(ZoneInfo("Asia/Shanghai"))
                        if (now - current_data_time).total_seconds() < 86400:  # 24小时内
                            self.notifier.send_trade_alert(
                                action="买入",
                                symbol=self.symbol,
                                price=executed_price,
                                size=executed_size
                            )

                    # 重置追踪止损状态（不检查 stop_manager）
                    self.risk_manager.trailing_activated = False
                    self.risk_manager.highest_profit = 0

                # ============ SELL BRANCH ============
                else:
                    # 卖单执行
                    executed_size = abs(order.executed.size)
                    executed_price = order.executed.price
                    executed_comm = order.executed.comm

                    # 获取入场价格：优先使用平均成本
                    try:
                        entry_price = self.get_average_entry_price()
                    except Exception:
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

                            # 计算盈亏，按比例分摊进场手续费
                            entry_price_ot = open_trade['entry_price']
                            trade_pnl = (executed_price - entry_price_ot) * close_size
                            try:
                                commission_rem = float(open_trade.get('commission_remaining', open_trade.get('commission', 0.0)) or 0.0)
                                orig_size = float(open_trade.get('orig_size', open_trade['size']) or 1)
                                proportion = close_size / orig_size
                                entry_comm_share = min(commission_rem, (open_trade.get('commission', 0.0) or 0.0) * proportion)
                                trade_pnl -= entry_comm_share
                                open_trade['commission_remaining'] = max(0.0, commission_rem - entry_comm_share)
                            except Exception:
                                pass
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
                            'exit_dt': current_dt,
                            'exit_bar': len(self),
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
                            'exit_dt': current_dt,
                            'exit_bar': len(self),
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
                        # 重置追踪止损状态（不检查 stop_manager）
                        self.risk_manager.trailing_activated = False
                        self.risk_manager.highest_profit = 0
                        self.risk_manager.profit_target_hits = 0
                    else:
                        # 部分平仓后更新加权平均成本
                        try:
                            remaining_size = self.position.size
                            if remaining_size > 0:
                                # 基于 FIFO 剩余 open trades 的平均成本作为新的 entry_price
                                open_trades = [t for t in self.trade_manager.executed_trades if t.get('status') == 'open' and t.get('size', 0) > 0]
                                if open_trades:
                                    total_cost = sum(t['entry_price'] * t['size'] for t in open_trades)
                                    total_size = sum(t['size'] for t in open_trades)
                                    if total_size > 0:
                                        self.entry_price = total_cost / total_size
                        except Exception:
                            pass

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

    def __init__(self, sendkey: str, wecom_webhook_url: str = "", serverchan_channel: str = ""):
        """
        初始化通知器
        参数:
            sendkey: Server酱的SendKey
        """
        self.sendkey = sendkey
        self.base_url = "https://sctapi.ftqq.com"
        self.enabled = bool(sendkey) or bool(wecom_webhook_url)
        self.wecom_webhook_url = wecom_webhook_url
        self.serverchan_channel = serverchan_channel

    def send_message(self, title: str, content: str = "") -> bool:
        """发送消息：优先发企业微信；若失败再回退到Server酱，避免重复推送。
        - 这样企业微信不会再收到“带查看详情按钮”的Server酱转发。
        """
        if not self.enabled:
            return False

        # 1) 优先企业微信
        if self.wecom_webhook_url:
            ok = self.send_message_wecom(title, content)
            if ok:
                return True

        # 2) 回退到Server酱（可选指定channel）
        if self.sendkey:
            return self.send_message_serverchan(title, content, channel=self.serverchan_channel)

        return False

    def send_message_wecom(self, title: str, content: str = "") -> bool:
        """仅向企业微信群机器人发送一条消息。"""
        if not self.wecom_webhook_url:
            return False
        try:
            headers = {"Content-Type": "application/json"}
            payload = {
                "msgtype": "markdown",
                "markdown": {"content": f"**{title}**\n{content}"}
            }
            r = requests.post(self.wecom_webhook_url, headers=headers, data=json.dumps(payload), timeout=10)
            r.raise_for_status()
            print(f"[通知] 企业微信群机器人发送成功: {title}")
            return True
        except Exception as e:
            print(f"[通知] 企业微信群机器人发送失败: {str(e)}")
            return False

    def send_message_serverchan(self, title: str, content: str = "", channel: str = "") -> bool:
        """仅向Server酱服务号发送一条消息。
        参数 channel: 可选，覆盖默认的`serverchan_channel`，例如仅发到服务号'9'。
        """
        if not self.sendkey:
            return False
        try:
            url = f"{self.base_url}/{self.sendkey}.send"
            data = {"title": title, "desp": content}
            # 优先使用调用时传入的channel，否则使用默认初始化时的channel
            effective_channel = channel or self.serverchan_channel
            if effective_channel:
                data["channel"] = effective_channel
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

    lines = ('datetime', 'open', 'high', 'low', 'close', 'volume', 'openinterest',)

    params = (
        ('symbol', 'QQQ'),
        ('interval', 60),
        ('preload_bars', 100),
        ('history_days', 3650),
        ('external_data', None),
        ('fromdate', None),  # 添加开始日期参数
        ('todate', None),    # 添加结束日期参数
        ('tz', None),        # 时区参数
        ('tzinput', None),   # 时区输入参数
        ('sessionstart', None),  # 会话开始
        ('sessionend', None),    # 会话结束
    )

    def __init__(self):
        super().__init__()
        
        # 添加所有缺失的属性
        self._tzinput = self.p.tzinput
        self._tz = self.p.tz
        
        # 关键修复：将日期转换为backtrader的数字格式
        if self.p.fromdate is None:
            default_from = dt.datetime.now() - dt.timedelta(days=365)
            self.fromdate = bt.date2num(default_from)  # 转换为数字
        else:
            # 如果传入的是datetime，转换为数字
            if isinstance(self.p.fromdate, dt.datetime):
                self.fromdate = bt.date2num(self.p.fromdate)
            else:
                self.fromdate = self.p.fromdate
        
        if self.p.todate is None:
            default_to = dt.datetime.now()
            self.todate = bt.date2num(default_to)  # 转换为数字
        else:
            # 如果传入的是datetime，转换为数字
            if isinstance(self.p.todate, dt.datetime):
                self.todate = bt.date2num(self.p.todate)
            else:
                self.todate = self.p.todate
        
        self._sessionstart = self.p.sessionstart
        self._sessionend = self.p.sessionend
        
        # 添加更多backtrader可能需要的属性
        self._name = self.p.symbol
        self._compression = 1
        self._timeframe = bt.TimeFrame.Minutes
        
        # 原有代码继续...
        self.last_update = None
        self.current_data = None
        self.hist_data = []
        self.hist_index = 0
        self.trading_params = None
        self._last_valid_data = None
        self._last_dummy_time = None
        self._last_non_trading_msg = None
        self._started = False
        self._data_loaded = False
        
        # 优先使用外部数据
        if self.p.external_data is not None:
            self._use_external_data()
        else:
            self._load_initial_data()

    def start(self):
        """Backtrader会在开始前调用此方法"""
        super().start()
        self._started = True
        # 确保外部数据已加载
        if self.p.external_data is not None and len(self.hist_data) == 0:
            self._use_external_data()
        print(f"SimpleLiveDataFeed启动完成，数据点数: {len(self.hist_data)}")
        
    def _use_external_data(self):
        """使用外部传入的数据"""
        print(f"使用外部传入的历史数据...")
        if self.p.external_data and isinstance(self.p.external_data, list):
            self.hist_data = self.p.external_data.copy()  # 使用copy避免引用问题
            self.hist_index = 0
            self._data_loaded = True
            print(f"外部数据已加载，记录数: {len(self.hist_data)}")
        else:
            print(f"警告：外部数据无效或为空")
            self.hist_data = []
            self.hist_index = 0

    def _load_initial_data(self):
        """加载初始历史数据"""
        # 如果已有外部数据，跳过
        if self.hist_data:
            return
            
        try:
            end_date = datetime.datetime.now()
            start_date = end_date - datetime.timedelta(days=self.p.history_days)

            # 使用更稳定的方式获取数据
            import yfinance as yf
            ticker = yf.Ticker(self.p.symbol)
            hist_df = ticker.history(
                start=start_date,
                end=end_date,
                interval='1d'
            )

            if not hist_df.empty:
                self.hist_data = []
                for idx, row in hist_df.iterrows():
                    self.hist_data.append({
                        'datetime': idx.to_pydatetime(),
                        'open': float(row['Open']),
                        'high': float(row['High']),
                        'low': float(row['Low']),
                        'close': float(row['Close']),
                        'volume': float(row['Volume'])
                    })
                print(f"内部加载了 {len(self.hist_data)} 条历史数据")
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
        """加载下一个数据点 - Backtrader会重复调用此方法"""
        try:
            # 修正4: 确保数据已经加载
            if not self._started:
                return None
                
            # 如果没有历史数据，尝试重新加载
            if len(self.hist_data) == 0 and self.p.external_data:
                self._use_external_data()
                if len(self.hist_data) == 0:
                    print("错误：无法加载历史数据")
                    return None
            
            # 加载历史数据
            if self.hist_index < len(self.hist_data):
                data = self.hist_data[self.hist_index]
                self.hist_index += 1
                
                # 进度提示
                if self.hist_index % 500 == 0:
                    print(f"历史数据加载进度: {self.hist_index}/{len(self.hist_data)}")
                
                # 安全地设置数据
                if self._set_data_lines(data):
                    return True
                else:
                    # 如果设置失败，尝试下一个数据点
                    return self._load()
            
            # 历史数据加载完毕
            if self.hist_index == len(self.hist_data) and len(self.hist_data) > 0:
                print("历史数据加载完成，切换到实时模式...")
                self.hist_index += 1
            
            # 实时数据模式
            return self._load_realtime_data()
            
        except Exception as e:
            print(f"_load错误: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def _set_data_lines(self, data):
        """安全地设置数据线"""
        try:
            if not data:
                return False
                
            # 验证数据完整性
            required = ['datetime', 'open', 'high', 'low', 'close', 'volume']
            for field in required:
                if field not in data:
                    print(f"数据缺少字段: {field}")
                    return False
                    
            # 转换datetime
            dt = data['datetime']
            if not isinstance(dt, datetime.datetime):
                dt = pd.to_datetime(dt)
            
            # 关键修复：确保datetime是naive（无时区）的
            if hasattr(dt, 'tz') and dt.tz is not None:
                dt = dt.tz_localize(None)
            
            # 使用forward方法推进数据线
            self.lines.datetime[0] = bt.date2num(dt)
            self.lines.open[0] = float(data.get('open', 0))
            self.lines.high[0] = float(data.get('high', 0))
            self.lines.low[0] = float(data.get('low', 0))
            self.lines.close[0] = float(data.get('close', 0))
            self.lines.volume[0] = float(data.get('volume', 0))
            self.lines.openinterest[0] = 0.0
            
            # 保存有效数据
            self._last_valid_data = data.copy()
            
            return True
            
        except Exception as e:
            print(f"设置数据线错误: {str(e)}, 数据: {data}")
            import traceback
            traceback.print_exc()
            return False

    def _load_realtime_data(self):
        """加载实时数据 - 每小时获取新数据"""
        now = datetime.datetime.now()
        
        # 初始化时间变量
        if self._last_non_trading_msg is None:
            self._last_non_trading_msg = now
        if self._last_dummy_time is None:
            self._last_dummy_time = now
        
        # 每6分钟（360秒）检查一次新数据，而不是每60秒
        check_interval = 360  # 6分钟
        
        # 检查是否需要获取新数据
        if self.last_update is None or (now - self.last_update).seconds >= check_interval:
            if self._is_trading_hours(now):
                new_data = self._fetch_current_price()
                if new_data:
                    self.last_update = now
                    print(f"[{now.strftime('%H:%M:%S')}] 获取实时数据: {self.p.symbol} @ ${new_data['close']:.2f}")
                    
                    # 将新数据添加到历史数据中（用于策略分析）
                    self.hist_data.append(new_data)
                    
                    return self._set_data_lines(new_data)
        
        # 非交易时间处理
        if not self._is_trading_hours(now):
            # 北京时间晚上21:30-次日4:00是美股交易时间（考虑夏令时）
            # 调整为东八区时间
            hour = now.hour
            if hour < 4 or hour >= 21:  # 北京时间晚上9点后或凌晨4点前
                # 可能是美股交易时间
                if (now - self._last_non_trading_msg).seconds >= 300:  # 5分钟提醒一次
                    print(f"[{now.strftime('%H:%M:%S')}] 检查美股交易时间...")
                    self._last_non_trading_msg = now
                    
                    # 尝试获取数据
                    new_data = self._fetch_current_price()
                    if new_data:
                        print(f"美股交易中: {self.p.symbol} @ ${new_data['close']:.2f}")
                        return self._set_data_lines(new_data)
            else:
                if (now - self._last_non_trading_msg).seconds >= 300:
                    print(f"[{now.strftime('%H:%M:%S')}] 非交易时间...")
                    self._last_non_trading_msg = now
            
            # 返回虚拟数据保持系统活跃
            if self._last_valid_data and (now - self._last_dummy_time).seconds >= 60:
                dummy_data = self._last_valid_data.copy()
                dummy_data['datetime'] = now
                self._last_dummy_time = now
                return self._set_data_lines(dummy_data)
        
        return False  # 没有新数据

    def _is_trading_hours(self, dt_local):
        """检查是否在交易时间（含盘前/盘后可选）。"""
        session, tradable = get_market_session(dt_local, include_prepost=not (getattr(self.trading_params, 'exclude_premarket', False) and getattr(self.trading_params, 'exclude_afterhours', False)))
        # 若严格排除盘前/盘后，则只在regular返回True
        if getattr(self.trading_params, 'exclude_premarket', False) or getattr(self.trading_params, 'exclude_afterhours', False):
            return session == 'regular'
        return tradable

    def _fetch_current_price(self):
        """获取当前价格"""
        try:
            import yfinance as yf
            ticker = yf.Ticker(self.p.symbol)
            info = ticker.info
            
            price = info.get('regularMarketPrice', 0)
            if price > 0:
                return {
                    'datetime': datetime.datetime.now(),
                    'open': price,
                    'high': price,
                    'low': price,
                    'close': price,
                    'volume': info.get('regularMarketVolume', 0)
                }
        except:
            pass
        return None        

class HybridDataFeed:
    """混合数据源 - 结合历史数据和实时数据"""

    @staticmethod
    def create_feed(symbol, start_date, end_date, live_mode=False, api_key="ZYF7PH3KNEQX341D"):
        """创建数据源"""
        if live_mode:
            # 实时模式：先加载历史数据，再切换到实时
            print(f"创建实时数据源: {symbol}")

            # 先加载最近30天的历史数据作为基础
            hist_start = datetime.datetime.now() - datetime.timedelta(days=30)
            hist_df = yfinance_download(symbol, hist_start.strftime("%Y-%m-%d"),
                                      end_date, interval="1d", prepost=False)

            # 将历史数据转换为SimpleLiveDataFeed可以使用的格式
            hist_data = []
            if hist_df is not None and not hist_df.empty:
                for _, row in hist_df.iterrows():
                    hist_data.append({
                        'datetime': row.name,  # 索引作为日期
                        'open': row.get('Open', row.get('open', 0)),
                        'high': row.get('High', row.get('high', 0)),
                        'low': row.get('Low', row.get('low', 0)),
                        'close': row.get('Close', row.get('close', 0)),
                        'volume': row.get('Volume', row.get('volume', 0))
                    })

            # 创建实时数据源，使用SimpleLiveDataFeed
            feed = SimpleLiveDataFeed(
                symbol=symbol,
                interval=60,  # 更新间隔（秒）
                external_data=hist_data,  # 传入历史数据
                fromdate=datetime.datetime.strptime(start_date, "%Y-%m-%d") if start_date else None,
                todate=datetime.datetime.strptime(end_date, "%Y-%m-%d") if end_date else None
            )

            return feed
        else:
            # 回测模式：使用历史数据
            print(f"创建历史数据源: {symbol}")
            df = yfinance_download(symbol, start_date, end_date, interval="1d", prepost=False)

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
        if strategy is None:
            raise ValueError("策略对象为 None，无法计算统计数据")
        trade_manager = strategy.trades  # 修正为 trades

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
        # 使用日频年化：若已是小时级别曲线，则应先日聚合
        # 近似处理：用 bars→天 的换算来估算交易日数
        intraday_mode = getattr(strategy.trading_params, 'intraday_mode', False)
        intraday_multiplier = getattr(strategy.trading_params, 'intraday_multiplier', 26)
        if intraday_mode and days > 0:
            approx_days = max(1, int(days / max(1, intraday_multiplier)))
        else:
            approx_days = max(1, days)
        annual_return = ((1 + total_return) ** (252/approx_days) - 1) * 100 if approx_days > 0 else 0

        stats['收益统计'] = {
            '初始资金': initial_value,
            '最终资金': final_value,
            '净收益': final_value - initial_value,
            '收益率': total_return * 100,
            '年化收益率': annual_return,
            '回测交易天数': approx_days,
        }
        # 获取所有交易（包括未平仓）
        all_trades = trade_manager.executed_trades
        entry_trades = [t for t in all_trades if t.get('type') == 'entry']
        exit_trades = [t for t in all_trades if t.get('type') == 'exit']
        # 仅使用 exit 交易进行胜率/收益统计，避免将FIFO关闭的entry记录（pnl=0）计入
        closed_exits = exit_trades

        # 交易统计
        if exit_trades:
            # 只以有意义的平仓为样本（过滤掉极小成交/手续费近似为0的边缘交易）
            meaningful_exits = [t for t in closed_exits if abs(t.get('pnl', 0)) >= 1e-6 or (t.get('size', 0) or 0) >= 1]
            winning_trades = sum(1 for t in meaningful_exits if t.get('pnl', 0) > 0) if meaningful_exits else 0
            total_pnl = sum(t.get('pnl', 0) for t in meaningful_exits) if meaningful_exits else 0
            
            stats['交易统计'] = {
                '总交易次数': len(meaningful_exits),
                '已平仓次数': len(meaningful_exits),
                '未平仓次数': max(0, len(entry_trades) - len(meaningful_exits)),
                '胜率': (winning_trades / len(meaningful_exits) * 100) if meaningful_exits else 0,
                '平均每笔收益': total_pnl / len(meaningful_exits) if meaningful_exits else 0,
                '最大单笔收益': max((t.get('pnl', 0) for t in meaningful_exits), default=0),
                '最大单笔亏损': min((t.get('pnl', 0) for t in meaningful_exits), default=0),
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
        # 使用日频收益率估算波动/夏普
        if len(portfolio_values) > 1:
            returns_all = np.diff(portfolio_values) / portfolio_values[:-1]
            if intraday_mode:
                # 近似按天聚合：整天整组复合，其余不足一天的忽略
                mult = max(1, int(intraday_multiplier))
                steps = len(returns_all) // mult
                if steps >= 1:
                    clipped = returns_all[:steps * mult]
                    reshaped = clipped.reshape(steps, mult)
                    daily_returns = (1 + reshaped).prod(axis=1) - 1
                else:
                    # 数据太少，直接使用原序列
                    daily_returns = returns_all
            else:
                daily_returns = returns_all
            # 使用更稳健的波动率估计（样本标准差，偏度修正），并做winsorize裁剪极值
            try:
                dr = np.array(daily_returns)
                if len(dr) > 10:
                    low, high = np.percentile(dr, [1, 99])
                    dr = np.clip(dr, low, high)
                volatility = np.std(dr, ddof=1) * np.sqrt(252)
            except Exception:
                volatility = np.std(daily_returns) * np.sqrt(252)
            
            # 计算夏普比率、索提诺比率和最差单日收益（这些应该在try-except块之外）
            sharpe_ratio = TradeAnalyzer._calculate_sharpe_ratio(daily_returns)
            sortino_ratio = TradeAnalyzer._calculate_sortino_ratio(daily_returns)
            worst_daily_return_pct = float(np.min(daily_returns) * 100) if len(daily_returns) > 0 else 0.0
        else:
            volatility = 0
            sharpe_ratio = 0
            sortino_ratio = 0
            worst_daily_return_pct = 0.0

        # 计算最大回撤时，先做日度聚合，避免小时级噪音误差放大
        try:
            if intraday_mode and len(portfolio_values) > intraday_multiplier:
                mult = max(1, int(intraday_multiplier))
                steps = len(portfolio_values) // mult
                pv = portfolio_values[:steps * mult].reshape(steps, mult)
                pv_daily = pv[:, -1]
                tmp_strategy = strategy
                # 临时替换以复用函数
                saved = tmp_strategy.portfolio_values
                tmp_strategy.portfolio_values = pv_daily.tolist()
                max_drawdown = TradeAnalyzer._calculate_max_drawdown(tmp_strategy)
                tmp_strategy.portfolio_values = saved
            else:
                max_drawdown = TradeAnalyzer._calculate_max_drawdown(strategy)
        except Exception:
            max_drawdown = TradeAnalyzer._calculate_max_drawdown(strategy)

        stats['风险统计'] = {
            '最大回撤': max_drawdown * 100,
            '收益波动率': volatility * 100,
            '夏普比率': sharpe_ratio,
            'Sortino比率': sortino_ratio,
            '最差单日收益': worst_daily_return_pct,
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
        保持原始排版，仅修复无交易记录的处理
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
        dark_bg_page    = "#1F1F22"   # intended to unify the entire "page" area
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
            # 获取交易记录
            trade_records = self._process_trade_records()
            
            # 检查是否有实际交易（修复：处理空记录）
            has_trades = (
                trade_records and 
                len(trade_records) > 0 and 
                trade_records[0].get('Date', '') != '暂无交易'
            )

            # Add a "Transaction Records" label
            fig.add_annotation(
                text="<b>交易记录</b>",
                x=0.5,
                y=0.42,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=16, color="#CCCCCC"),
                xanchor='center',
                yanchor='bottom'
            )

            # Table of trades - 处理无交易情况
            if has_trades:
                # 有交易：使用原始英文列名保持一致性
                table = go.Table(
                    header=dict(
                        values=['Date', 'Type', 'Size', 'Price', 'Daily P&L', 'Total P&L'],
                        font=dict(size=12, color=table_header_font),
                        fill_color=table_header_bg,
                        align='left',
                        height=30,
                        line_color='rgba(255,255,255,0)'
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
                        line_color='rgba(255,255,255,0)'
                    )
                )
                
                # Add buy/sell markers
                self._add_trade_markers(fig, trade_records)
            else:
                # 无交易：显示空表格（保持英文列名）
                table = go.Table(
                    header=dict(
                        values=['Date', 'Type', 'Size', 'Price', 'Daily P&L', 'Total P&L'],
                        font=dict(size=12, color=table_header_font),
                        fill_color=table_header_bg,
                        align='left',
                        height=30,
                        line_color='rgba(255,255,255,0)'
                    ),
                    cells=dict(
                        values=[
                            ['暂无交易'],
                            [''],
                            [''],
                            [''],
                            [''],
                            ['']
                        ],
                        align='left',
                        font=dict(size=11, color="#AAAAAA"),
                        fill_color=table_cells_bg,
                        height=24,
                        line_color='rgba(255,255,255,0)'
                    )
                )
            
            fig.add_trace(table, row=2, col=1)

            # Stats table label
            fig.add_annotation(
                text="<b>回测结果</b>",
                x=0.5,
                y=0.18,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=16, color="#CCCCCC"),
                xanchor='center',
                yanchor='bottom'
            )

            # 获取当前价格（安全处理）
            current_price = 0.0
            if hasattr(self.strategy, 'data') and hasattr(self.strategy.data, 'close'):
                try:
                    if len(self.strategy.data.close) > 0:
                        current_price = self.strategy.data.close[0]
                except:
                    current_price = self.df[close_col].iloc[-1] if len(self.df) > 0 else 0.0
            
            stats_table = self._create_stats_table(current_price=current_price)
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

        # 7) Chart Title: main + second line（关键修复：使用HTML实体）
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
        # Partially transparent fill colors to mimic a "frosted glass" style
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
        处理执行的交易记录, 并生成 Buy/Sell 日志!
        """
        trade_records = []
        running_pnl = 0.0
        executed_trades = self.trade_manager.executed_trades or []
        # 修复：处理无交易的情况
        if not executed_trades or len(executed_trades) == 0:
            # 返回一条占位记    录，避免表格为空
            return [{
                'Date': '暂无交易',
                'Type': '-',
                'Size': 0,
                'Price': 0.0,
                'Daily P&L': 0.0,
                'Total P&L': 0.0
            }]
        
        for trade in executed_trades:
            if trade.get('type') == 'entry':
                b_date = trade.get('entry_date')
                if b_date is None:
                    continue
                buy_price = float(trade.get('entry_price', 0.0))
                buy_size = trade.get('orig_size', trade.get('size', 0))
                if buy_size <= 0 or buy_price <= 1.0:
                    continue
                if trade.get('status') == 'closed' and trade.get('size', 0) <= 0:
                    buy_size = trade.get('orig_size', 0)
                    if buy_size <= 0:
                        continue

                # 为保持“Total P&L”为已实现盈亏，买入行的 Daily P&L 置为 0（买入手续费在卖出时分摊计入）
                daily_pnl = 0.0
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
                # 以 exit 为主序，先按bar/时间排序，防止同日多次交易导致显示错序
                exit_dt = trade.get('exit_dt')
                exit_bar = trade.get('exit_bar', 10**12)
                realized_pnl = float(trade.get('pnl', 0.0))
                updated_pnl  = running_pnl + realized_pnl
                trade_records.append({
                    'Date': str(s_date),
                    'Type': 'S',
                    'Size': sell_size,
                    'Price': sell_price,
                    'Daily P&L': realized_pnl,
                    'Total P&L': updated_pnl,
                    '_exit_bar': exit_bar,
                    '_exit_dt': str(exit_dt) if exit_dt else ''
                })
                running_pnl = updated_pnl
        
        # 排序：优先 exit 的 bar 序，然后按日期
        def _sort_key(r):
            if r.get('Type') == 'S':
                return (0, r.get('_exit_bar', 10**12))
            # 买单按出现顺序靠前，避免“Total P&L”被后来的买单穿插显示为0
            return (1, pd.to_datetime(r['Date']))
        trade_records.sort(key=_sort_key)

        # 关键修复：在排序后重新累计 Total P&L（仅累加卖出行的 Daily P&L）
        running = 0.0
        for rec in trade_records:
            if rec.get('Type') == 'S':
                running += float(rec.get('Daily P&L', 0.0) or 0.0)
            rec['Total P&L'] = running
        
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

def yfinance_download(symbol: str, start: str, end: str, interval: str = "1d", prepost: bool = True) -> pd.DataFrame:
    try:
        print(f"使用YFinance获取 {symbol} 数据...")
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start, end=end, interval=interval, auto_adjust=True, prepost=prepost)
        if df.empty:
            raise Exception(f"YFinance未返回{symbol}的数据")
        df.reset_index(inplace=True)
        # 标准化列名
        df.columns = [str(col).title() for col in df.columns]
        required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        # 检查并重命名可能的日期列
        if 'Date' not in df.columns and 'Datetime' in df.columns:
            df.rename(columns={'Datetime': 'Date'}, inplace=True)
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise Exception(f"YFinance数据缺少必需列: {missing_cols}")
        print(f"YFinance成功获取 {len(df)} 条 {symbol} 数据记录")
        return df
    except Exception as e:
        print(f"YFinance数据获取失败: {str(e)}")
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
            print("使用Twelve Data demo密钥，功能有限")
            api_key = "4e06770f76fe42b9bc3b6760b14118f6"

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
            # 发送更详细的启动通知（企业微信优先，不回退Server酱）
            try:
                self._send_startup_notification()
                # 并向Server酱服务号发送一条“后台通知”，便于你在服务号看到启动
                try:
                    hb_channel = getattr(self.strategy_params, 'serverchan_heartbeat_channel', '')
                    self.notifier.send_message_serverchan(
                        "实时交易系统启动",
                        f"开始监控 {self.strategy_params.symbol}\n初始资金: ${self.strategy_params.initial_cash:,.2f}",
                        channel=hb_channel or self.strategy_params.serverchan_channel
                    )
                except Exception:
                    pass
            except Exception as e:
                print(f"发送启动通知失败: {str(e)}")
            # 按需：启动即发送一次“每日交易总结”，便于确认渠道和格式
            try:
                if getattr(self.strategy_params, 'send_daily_summary_on_start', True):
                    self._send_daily_summary()
            except Exception:
                pass

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

    def _send_startup_notification(self):
        """发送启动通知"""
        if self.notifier and self.is_running:
            # 去除前导空白的多行字符串，避免企业微信显示缩进空格
            startup_msg = "\n".join([
                "🚀 量化交易系统启动",
                "━━━━━━━━━━━━━━━",
                f"📊 标的: {self.strategy_params.symbol}",
                f"💰 初始资金: ${self.strategy_params.initial_cash:,.0f}",
                f"⏰ 启动时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
                "📈 策略类型: 高风险高回报",
                "━━━━━━━━━━━━━━━",
                "系统开始实时监控..."
            ])
            self.notifier.send_message("交易系统启动", startup_msg)

    def _send_daily_summary(self):
        """每日收盘后发送总结"""
        if not self.notifier or not hasattr(self.strategy, 'broker'):
            return
        
        try:
            portfolio_value = self.strategy.broker.getvalue()
            cash = self.strategy.broker.get_cash()
            position = self.strategy.position.size if self.strategy.position else 0
            
            # 计算当日盈亏
            daily_pnl = portfolio_value - getattr(self, '_last_portfolio_value', portfolio_value)
            daily_pnl_pct = daily_pnl / getattr(self, '_last_portfolio_value', portfolio_value) * 100
            
            # === KISS: 生成“去年今日-今天”的高保真可视化图（直接调用TradeVisualizer），并上传到 imgbb ===
            chart_url = ""
            backtest_stats = None
            try:
                # 时间范围
                tz = ZoneInfo("Asia/Shanghai")
                end_dt = datetime.datetime.now(tz)
                start_dt = end_dt - datetime.timedelta(days=365)
                start_str = start_dt.strftime('%Y-%m-%d')
                end_str = end_dt.strftime('%Y-%m-%d')
                print(f"[每日总结] 回测区间: {start_str} ~ {end_str}")

                # 直接用TradeVisualizer生成高保真图（与回测模式一致）
                out_png, stats_obj = generate_backtest_visual_png(
                    symbol=self.strategy_params.symbol,
                    start_str=start_str,
                    end_str=end_str,
                    interval=self.strategy_params.data_interval or '1h',
                    # 使用summary_backtest_initial_cash，保证有足够资金触发策略交易，从而产生买卖点
                    initial_cash=getattr(self.strategy_params, 'summary_backtest_initial_cash', 100000.0),
                    include_prepost=False
                )
                if out_png and os.path.exists(out_png):
                    print(f"[每日总结] 文件是否存在: True -> {out_png}")
                    print("[每日总结] 上传图像至 imgbb ...")
                    chart_url = upload_image_imgbb(out_png, api_key=self.strategy_params.imgbb_api_key)
                    if chart_url:
                        print(f"[每日总结] 上传成功: {chart_url}")
                    else:
                        print("[每日总结] 上传失败：未获得URL（请检查imgbb_api_key或配额）")
                else:
                    print("[每日总结] 未生成PNG（可能缺少kaleido），降级为简易统计，不附图")
            except Exception as e:
                print(f"[每日总结] 回测图与上传流程异常: {str(e)}")
                chart_url = ""

            # 去除多余空格的整洁消息体
            lines = [
                "📊 每日交易总结",
                "━━━━━━━━━━━━━━━",
                f"📅 日期: {datetime.datetime.now().strftime('%Y-%m-%d')}",
                f"💼 账户价值: ${portfolio_value:,.2f}",
                f"💵 可用现金: ${cash:,.2f}",
                f"📈 持仓数量: {position}股",
                "━━━━━━━━━━━━━━━",
                f"📊 当日盈亏: ${daily_pnl:+,.2f} ({daily_pnl_pct:+.1f}%)",
                "━━━━━━━━━━━━━━━",
            ]
            # 附加：简要回测统计（去年今日-今天）
            if backtest_stats is not None:
                lines.extend([
                    f"简易回测区间: { (datetime.datetime.now()-datetime.timedelta(days=365)).strftime('%Y-%m-%d') } ~ {datetime.datetime.now().strftime('%Y-%m-%d') }",
                    f"初始资金: ${self.strategy_params.initial_cash:,.2f}",
                    f"最终资金: ${backtest_stats['最终价值']:,.2f}",
                    f"总收益率: {backtest_stats['收益率']:.2f}%",
                ])
            if chart_url:
                lines.append(f"简易回测K线图: {chart_url} 请复制链接到默认浏览器打开。")
            summary_msg = "\n".join(lines)
            # 每日总结仅通过企业微信发送，避免服务号打扰
            self.notifier.send_message_wecom("每日交易总结", summary_msg)
            self._last_portfolio_value = portfolio_value
            
        except Exception as e:
            print(f"发送每日总结失败: {str(e)}")

    def _monitor_loop(self): 
        # 等待策略初始化
        init_wait_count = 0
        while not hasattr(self, 'strategy') or self.strategy is None:
            # KISS加速时钟：初始化等待使用time_scale
            time.sleep(1 / max(1.0, float(getattr(self.strategy_params, 'time_scale', 1.0))))
            init_wait_count += 1
            if init_wait_count > 10:
                print("策略初始化超时")
                return
        
        # 启动时通知由 start() 统一发送，避免重复

        check_interval = 30
        last_position_check = None
        error_count = 0  # 添加错误计数器
        max_errors = 3   # 最大错误次数

        while self.is_running:
            try:
                now = datetime.datetime.now(ZoneInfo("Asia/Shanghai"))

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

                        # 定期状态报告（按参数间隔）
                        if not hasattr(self, '_last_full_report') or (now - self._last_full_report).seconds >= int(getattr(self.strategy_params, 'heartbeat_interval_seconds', 1800)):
                            session, tradable = get_market_session(now, include_prepost=getattr(self.strategy_params, 'prepost', True))
                            session_map = {"pre": "盘前", "regular": "常规", "after": "盘后", "closed": "休市"}
                            session_human = session_map.get(session, session)
                            # 计算总收益/当日收益
                            try:
                                initial_cash = self.strategy_params.initial_cash
                                total_ret = portfolio_value / initial_cash - 1
                            except Exception:
                                total_ret = 0.0
                            try:
                                if hasattr(self.strategy_params, 'strategy') and hasattr(self.strategy_params.strategy, 'daily_value') and self.strategy_params.strategy.daily_value:
                                    day_ret = portfolio_value / self.strategy_params.strategy.daily_value - 1
                                else:
                                    day_ret = 0.0
                            except Exception:
                                day_ret = 0.0

                            status_msg = (
                                f"[{now.strftime('%Y-%m-%d %H:%M:%S %Z')}] 系统状态报告\n"
                                f"- 账户总值: ${portfolio_value:,.2f}\n"
                                f"- 可用现金: ${cash:,.2f}\n"
                                f"- 当前持仓: {current_position}股\n"
                                f"- 当前价格: ${current_price:.2f}\n"
                                f"- 当日: {day_ret*100:.2f}% | 总收益: {total_ret*100:.2f}%\n"
                                f"- 会话: {session_human} | 交易时间: {'是' if tradable else '否'}\n"
                                f"- 会话规则: {'包含盘前/盘后' if getattr(self.strategy_params, 'prepost', True) else '仅常规时段'}"
                            )
                            print(status_msg)
                            self._last_full_report = now

                    except Exception as e:
                        # 内部错误，不发送通知
                        if 'addindicator' not in str(e) and 'nonzero' not in str(e):  # 忽略特定错误
                            print(f"监控数据访问错误: {str(e)}")
                else:
                    print(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] 等待策略初始化...")

                # KISS加速时钟：监控循环节拍受time_scale影响
                time.sleep(max(0.1, check_interval / max(1.0, float(getattr(self.strategy_params, 'time_scale', 1.0)))))

            except Exception as e:
                error_count += 1
                print(f"监控错误 ({error_count}/{max_errors}): {str(e)}")

                # 只有在错误次数达到阈值时才发送通知
                if error_count >= max_errors and self.notifier:
                    self.notifier.send_message("监控异常", f"连续{error_count}次错误，最后错误: {str(e)}")
                    error_count = 0  # 重置计数器，避免重复发送

    def _is_trading_hours(self, dt_local):
        """检查是否在交易时间（含盘前/盘后可选）。"""
        session, tradable = get_market_session(dt_local, include_prepost=getattr(self.strategy_params, 'prepost', True))
        if getattr(self.strategy_params, 'exclude_premarket', False) or getattr(self.strategy_params, 'exclude_afterhours', False):
            return session == 'regular'
        return tradable

def preprocess_intraday_data(df: pd.DataFrame, trading_params: TradingParameters) -> pd.DataFrame:
    if not trading_params.intraday_mode:
        return df

    print(f"过滤非交易时段数据...")
    original_len = len(df)

    # 确保 'date' 列存在并转换为 datetime 类型
    if 'date' not in df.columns:
        raise KeyError("数据中缺少 'date' 列")

    df['date'] = pd.to_datetime(df['date'])

    # 移除时区信息
    if df['date'].dt.tz is not None:
        print("检测到时区信息，正在移除...")
        df['date'] = df['date'].dt.tz_localize(None)

    df['hour'] = df['date'].dt.hour
    df['minute'] = df['date'].dt.minute
    df['time_minutes'] = df['hour'] * 60 + df['minute']

    # 解析交易时段
    start_hour, start_minute = map(int, trading_params.trading_start_time.split(':'))
    end_hour, end_minute = map(int, trading_params.trading_end_time.split(':'))
    start_time = start_hour * 60 + start_minute
    end_time = end_hour * 60 + end_minute

    print(f"交易时段: {start_hour}:{start_minute:02d} - {end_hour}:{end_minute:02d}")

    # 标准过滤逻辑（仅按参数控制盘前/盘后）
    if trading_params.exclude_premarket and trading_params.exclude_afterhours:
        df = df[(df['time_minutes'] >= start_time) & (df['time_minutes'] < end_time)]

    # 过滤周末
    df = df[df['date'].dt.weekday < 5]

    # 排序并重置索引
    df = df.sort_values('date')
    df = df.drop(['hour', 'minute', 'time_minutes'], axis=1)
    df = df.reset_index(drop=True)

    filtered_len = len(df)
    if original_len > 0:
        print(f"过滤完成 - 原始数据: {original_len}条, 过滤后: {filtered_len}条, "
              f"过滤比例: {(original_len-filtered_len)/original_len*100:.1f}%")
    
    return df

def harmonize_intraday(df: pd.DataFrame, trading_params: TradingParameters, interval: str, provider: str = "") -> pd.DataFrame:
    """统一不同数据源的日内数据：
    - 统一时区为naive（已在预处理实现）
    - 统一交易会话 09:30-16:00（预处理已实现）
    - 若为1小时数据，锚定小时bar到 10:30、11:30、...、16:00（右侧对齐）
    - 统一列名与排序
    """
    try:
        if not trading_params.intraday_mode:
            return df

        if df is None or df.empty:
            return df

        if 'date' not in df.columns:
            raise KeyError("harmonize_intraday 需要 'date' 列")

        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')

        # 确保必要列存在
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col not in df.columns:
                raise KeyError(f"harmonize_intraday 缺少列: {col}")

        # 仅对小时级做聚合处理（更宽松，避免丢失过多bar）
        if interval in ["1h", "60m", "1H"]:
            original_count = len(df)
            tmp = df.set_index('date')
            # 将索引左移30分钟，再以1小时重采样，最后右移回去
            shifted = tmp.copy()
            shifted.index = shifted.index - pd.Timedelta(minutes=30)
            agg = {
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }
            resampled = shifted.resample('60min', label='right', closed='right').agg(agg)
            resampled.index = resampled.index + pd.Timedelta(minutes=30)

            # 会话内过滤（宽松）：仅保留工作日，且结束时间在 10:00-16:00 之间
            def in_session(ts):
                return (ts.weekday() < 5) and (ts.hour > 9 or (ts.hour == 9 and ts.minute >= 30)) and (ts.hour < 16 or (ts.hour == 16 and ts.minute == 0))
            resampled = resampled[resampled.index.map(in_session)]

            # Fallback：若条数异常偏少（低于预期的60%），改用不移位的直接1小时聚合
            expected = max(1, int(original_count / 4))
            if len(resampled) < expected * 0.6:
                direct = tmp.resample('60min').agg(agg)
                direct = direct[direct.index.map(lambda x: x.weekday() < 5)]
                if len(direct) > len(resampled):
                    resampled = direct

            resampled = resampled.dropna(subset=['open', 'high', 'low', 'close'])
            resampled = resampled.reset_index().rename(columns={'index': 'date'})
            df = resampled

        # 标注数据源（若提供）
        if provider:
            df['source'] = provider

        return df.reset_index(drop=True)

    except Exception as e:
        print(f"harmonize_intraday 错误: {str(e)}")
        return df
def main():
    """主执行函数"""
    try:
        # 获取交易参数，支持默认值
        symbol = input("输入股票代码（例如：NVDA/TSLA/AMD/COIN/PLTR） [默认: NVDA]: ").strip().upper() or "NVDA"
        # 放宽白名单：不再限制回测/实时，仅用于提示
        whitelist = {"NVDA", "TSLA", "AMD", "COIN", "PLTR"}
        # 新增：数据源选择
        print("\n选择数据源: 1=YFinance  2=Twelve Data  (回车自动)")
        provider_choice = (input("请输入数字选择数据源 (默认自动): ").strip() or "").lower()
        if provider_choice not in {"1", "2", ""}:
            provider_choice = ""
        if symbol in ['NVDA', 'AMD', 'TSLA', 'COIN', 'PLTR']:
            print(f"\n提示：{symbol} 是2025年热门股票，使用激进策略参数")

        # 新增：询问是否为实时模式
        live_mode_input = input("是否启用实时交易模式? y/n (默认n): ").strip().lower()
        live_mode = (live_mode_input == 'y')

        # 实时模式：固定初始资金10,000；回测模式：保留可输入，默认100,000
        if live_mode:
            initial_cash = 10000.0
        else:
            initial_cash_str = input("输入初始资金（例如：100000） [默认: 100000]: ").strip() or "100000"
            initial_cash = float(initial_cash_str)


        serverchan_key = "SCT119824TW3DsGlBjkhAV9lJWIOLohv1P"

        # API 密钥（仅保留 Twelve Data）
        TWELVE_DATA_API_KEY = "4e06770f76fe42b9bc3b6760b14118f6"

        # 日期设置
        generate_live_report = False  # 默认不在实时后生成回测报告
        if live_mode:
            # 实时模式：允许用户指定历史数据长度
            history_days_input = input("输入历史数据天数（默认365天）: ").strip() or "365"
            history_days = int(history_days_input)

            end_date = datetime.datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.datetime.now() - datetime.timedelta(days=history_days)).strftime("%Y-%m-%d")
            print(f"\n实时模式：加载从 {start_date} 到 {end_date} 的历史数据")

            # 实时模式不生成历史回测报告
            generate_live_report = False

            interval_input = "1h"
            include_prepost_input = input("是否包含盘前/盘后交易? y/n (默认y): ").strip().lower() or "y"
            include_prepost = (include_prepost_input == 'y')
        else:
            # 可选：指定日期范围
            start_input = input("输入开始日期(YYYY-MM-DD)，或者直接回车使用默认: ").strip()
            end_input = input("输入结束日期(YYYY-MM-DD)，或者直接回车使用默认: ").strip()

            # 可选：数据间隔及是否包含盘前/盘后数据（默认1d和y）
            interval_input = input("输入数据间隔(如'1d','1h','15m')，默认'1h': ").strip() or "1h"
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
            data_interval=interval_input,  # 传入数据间隔
            live_mode=live_mode,
            serverchan_sendkey=serverchan_key,
            enable_trade_notification=bool(serverchan_key),
            enable_risk_notification=bool(serverchan_key),
            generate_live_report=generate_live_report,
            exclude_premarket=not include_prepost,
            exclude_afterhours=not include_prepost,
            prepost=include_prepost
        )
        trading_params.validate()

        # 创建通知器（提前创建以便在数据下载阶段也能使用）
        notifier = None
        if trading_params.serverchan_sendkey:
            notifier = ServerChanNotifier(
                trading_params.serverchan_sendkey,
                wecom_webhook_url=trading_params.wecom_webhook_url,
                serverchan_channel=trading_params.serverchan_channel,
            )
            print("Server酱通知已启用")

        # ===== 数据获取部分 =====
        df = None
        df_feed = None

        if live_mode:
            # 实时模式：先获取历史数据，再创建数据源
            print(f"\n获取 {symbol} 的历史数据...")
            api_sources = [
                ("YFinance", lambda: yfinance_download(
                    symbol=trading_params.symbol,
                    start=trading_params.start_date,
                    end=trading_params.end_date,
                    interval=interval_input,
                    prepost=trading_params.prepost
                )),
                ("Twelve Data", lambda: twelvedata_download(
                    symbol=trading_params.symbol,
                    start=trading_params.start_date,
                    end=trading_params.end_date,
                    interval=interval_input,
                    api_key='4e06770f76fe42b9bc3b6760b14118f6'
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
                        print(f"  {source_name} 数据获取成功！")
                        break

                except Exception as e:
                    print(f"错误： {source_name} 失败: {str(e)}")
                    continue

            if df is None or df.empty:
                raise ValueError(f"未能获取 {trading_params.symbol} 的历史数据")
            
            # 数据处理 - 标准化列名
            df.reset_index(inplace=True)
            
            # 将所有列名转换为小写
            df.columns = [str(col).lower() for col in df.columns]
            
            # 确保date列存在（处理可能的Date或Datetime列）
            if 'date' not in df.columns:
                if 'datetime' in df.columns:
                    df.rename(columns={'datetime': 'date'}, inplace=True)
                elif 'index' in df.columns:
                    df.rename(columns={'index': 'date'}, inplace=True)
                else:
                    # 如果都没有，使用索引作为date
                    df['date'] = df.index
            
            # 转换数据格式供SimpleLiveDataFeed使用
            hist_data = []
            for idx, row in df.iterrows():
                # 确保所有字段都存在
                hist_data.append({
                    'datetime': pd.to_datetime(row['date']),
                    'open': float(row.get('open', 0)),
                    'high': float(row.get('high', 0)),
                    'low': float(row.get('low', 0)),
                    'close': float(row.get('close', 0)),
                    'volume': float(row.get('volume', 0))
                })
            
            print(f"准备传入 {len(hist_data)} 条历史数据到实时数据源")
            
            # 创建实时数据源，确保external_data正确传递
            # 准备日期参数（转换为数字格式）
            from_date = datetime.datetime.strptime(trading_params.start_date, "%Y-%m-%d")
            to_date = datetime.datetime.strptime(trading_params.end_date, "%Y-%m-%d")

            df_feed = SimpleLiveDataFeed(
                symbol=trading_params.symbol,
                interval=60,  # 每60秒检查一次新数据
                history_days=trading_params.history_days,
                external_data=hist_data,
                # 传入数字格式的日期
                fromdate=bt.date2num(from_date),
                todate=bt.date2num(to_date)
            )
            
            # 手动触发数据加载
            if hasattr(df_feed, '_use_external_data'):
                df_feed._use_external_data()
            
            df_feed.trading_params = trading_params
            print(f"实时数据源已创建，确认数据点数: {len(df_feed.hist_data)}")

            # 统一使用 SimpleLiveDataFeed 作为实时数据源，无需额外实时线程

        else:
            # ===== 使用多源API自动切换机制（支持用户优先级） =====
            print(f"\n下载 {symbol} 的历史数据...")

            # API源列表，按优先级排序（可被用户选择覆盖）
            api_sources_all = [
                ("YFinance", lambda: yfinance_download(
                    symbol=trading_params.symbol,
                    start=trading_params.start_date,
                    end=trading_params.end_date,
                    interval=interval_input,
                    prepost=trading_params.prepost
                )),
                ("Twelve Data", lambda: twelvedata_download(
                    symbol=trading_params.symbol,
                    start=trading_params.start_date,
                    end=trading_params.end_date,
                    interval=interval_input,
                    api_key='4e06770f76fe42b9bc3b6760b14118f6'
                ))
            ]
            if provider_choice == "1":
                api_sources = api_sources_all
            elif provider_choice == "2":
                api_sources = [api_sources_all[1], api_sources_all[0]]
            else:
                api_sources = api_sources_all

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
                        print(f" {source_name} 数据获取成功！")
                        break

                except Exception as e:
                    print(f"错误： {source_name} 失败: {str(e)}")
                    if "rate limit" in str(e).lower() or "too many requests" in str(e).lower():
                        print("  提示: API额度已用完，尝试下一个数据源...")
                    continue

        if df is None or df.empty:
            raise ValueError(f"未找到股票代码 {trading_params.symbol} 的数据")

        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"数据下载错误: 期望 pandas DataFrame，得到 {type(df)}")

        # 立即进行列名标准化（移到preprocess_intraday_data之前）
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
            #print("\n执行小时/分钟级别数据的时间修复...")

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

        else:
            # 如果是日级别数据，确保日期列是datetime类型
            df['date'] = pd.to_datetime(df['date'])

        # 只保留必要的列
        df = df[required]
        # 预处理与统一化日内数据（仅在此处执行一次）
        if trading_params.intraday_mode:
            df = preprocess_intraday_data(df, trading_params)
            df = harmonize_intraday(df, trading_params, interval_input, provider=successful_source if 'successful_source' in locals() else "")
            data_length = len(df)
            print(f"预处理后数据长度: {data_length}条")
            if data_length < 50:
                raise ValueError("数据量不足，请使用日线数据或调整日期范围")
            trading_params.adjust_periods_for_data_length(data_length)

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

        if df is not None and not df.empty:
            # 上方已完成预处理，这里仅做拷贝
            df_bt = df.copy()
            df_viz = df.copy()

        # 创建数据源
        if live_mode:
            # 创建实时数据源，确保external_data正确传递
            # 准备日期参数（转换为数字格式）
            from_date = datetime.datetime.strptime(trading_params.start_date, "%Y-%m-%d")
            to_date = datetime.datetime.strptime(trading_params.end_date, "%Y-%m-%d")

            df_feed = SimpleLiveDataFeed(
                symbol=trading_params.symbol,
                interval=60,  # 每60秒检查一次新数据
                history_days=trading_params.history_days,
                external_data=hist_data,
                # 传入数字格式的日期
                fromdate=bt.date2num(from_date),
                todate=bt.date2num(to_date)
            )
            df_feed.trading_params = trading_params
            # 注意：SimpleLiveDataFeed会自动加载历史数据
            print("实时数据源已创建")
        else:
            # 回测模式：使用历史数据（优先使用列名映射，避免列顺序造成的数据错配）
            cols_lower = [str(c).lower() for c in df_bt.columns]
            def has_cols(*names):
                return all(n in cols_lower for n in names)
            if has_cols('date', 'open', 'high', 'low', 'close', 'volume'):
                # 名称映射更稳健
                df_feed = bt.feeds.PandasData(
                    dataname=df_bt,
                    datetime='date',
                    open='open',
                    high='high',
                    low='low',
                    close='close',
                    volume='volume',
                    openinterest=-1,
                    timeframe=timeframe,
                    compression=compression
                )
            else:
                # 回退：索引映射（假设顺序为 date, open, high, low, close, volume）
                df_feed = bt.feeds.PandasData(
                    dataname=df_bt,
                    datetime=0,
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
                    # KISS加速时钟：启动延迟使用time_scale
                    time.sleep(2 / max(1.0, float(getattr(trading_params, 'time_scale', 1.0))))
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

                # 启动通知已由 Monitor.start() 统一发送

                # 实时模式：保持运行（仅主循环推送心跳，监控线程不重复推送）
                last_status_time = datetime.datetime.now()
                status_interval = int(trading_params.heartbeat_interval_seconds)
                last_serverchan_hb = datetime.datetime.now() - datetime.timedelta(hours=1)
                last_wecom_hb_date = None
                while True:
                    try:
                        now = datetime.datetime.now()

                        # 定期输出状态信息
                        if (now - last_status_time).seconds >= status_interval:
                            # 检查是否在交易时间，并输出会话类型（同时推送到通知渠道，便于远程查看）
                            session, tradable = get_market_session(now, include_prepost=trading_params.prepost)
                            is_trading = tradable if monitor else False
                            # 构造更友好的会话文案
                            session_map = {"pre": "盘前", "regular": "常规", "after": "盘后", "closed": "休市"}
                            session_human = session_map.get(session, session)
                            status_msg = f"[{now.strftime('%Y-%m-%d %H:%M:%S %Z')}] 系统运行中 - "
                            status_msg += ("交易时间" if is_trading else "非交易时间") + f"（会话: {session_human}）"

                            print(status_msg)
                            # 心跳推送（分渠道频率）
                            if notifier:
                                # 组装心跳内容
                                extra = f"会话规则: {'包含盘前/盘后' if trading_params.prepost else '仅常规时段'}"
                                if getattr(trading_params, 'heartbeat_include_returns', True):
                                    try:
                                        current_price = strategy.data.close[0]
                                        portfolio_value = strategy.broker.getvalue()
                                        initial_cash = trading_params.initial_cash
                                        total_ret = portfolio_value / initial_cash - 1
                                        if hasattr(strategy, 'daily_value') and strategy.daily_value:
                                            day_ret = portfolio_value / strategy.daily_value - 1
                                        elif len(strategy.portfolio_values) >= 2:
                                            pv = strategy.portfolio_values
                                            day_ret = pv[-1] / pv[-2] - 1
                                        else:
                                            day_ret = 0.0
                                        extra = (
                                            f"{extra}\n标的: {trading_params.symbol} | 价格: ${current_price:.2f}\n当日: {day_ret*100:.2f}% | 总收益: {total_ret*100:.2f}%"
                                        )
                                    except Exception:
                                        pass

                                # Server酱：每小时一次（仅服务号，不再连带企业微信）
                                if getattr(trading_params, 'enable_serverchan_hourly_heartbeat', True):
                                    sc_interval = int(getattr(trading_params, 'serverchan_heartbeat_seconds', 3600))
                                    if (now - last_serverchan_hb).total_seconds() >= sc_interval:
                                        # 使用心跳专用渠道覆盖，只发服务号，避免企业微信收到小时心跳
                                        hb_channel = getattr(trading_params, 'serverchan_heartbeat_channel', '')
                                        notifier.send_message_serverchan(
                                            "系统心跳",
                                            f"{status_msg}\n{extra}",
                                            channel=hb_channel
                                        )
                                        last_serverchan_hb = now

                                # 企业微信：每日一次（本地时区）
                                if getattr(trading_params, 'enable_wecom_daily_heartbeat', True):
                                    try:
                                        tz = ZoneInfo("Asia/Shanghai")
                                        now_cst = now.astimezone(tz)
                                        target_h = int(getattr(trading_params, 'wecom_daily_heartbeat_hour_local', 10))
                                        target_m = int(getattr(trading_params, 'wecom_daily_heartbeat_minute_local', 15))
                                        if (now_cst.hour == target_h and now_cst.minute >= target_m):
                                            if last_wecom_hb_date != now_cst.date():
                                                # 改为发送“每日交易总结”到企业微信
                                                try:
                                                    if monitor:
                                                        monitor._send_daily_summary()
                                                    else:
                                                        notifier.send_message_wecom("每日交易总结", f"{status_msg}\n{extra}")
                                                except Exception:
                                                    notifier.send_message_wecom("每日交易总结", f"{status_msg}\n{extra}")
                                                last_wecom_hb_date = now_cst.date()
                                    except Exception:
                                        pass
                            last_status_time = now

                        # 外部健康检查心跳（轻量、异常忽略）
                        try:
                            ping_url = getattr(trading_params, 'healthcheck_ping_url', '')
                            if ping_url:
                                requests.get(ping_url, timeout=5)
                        except Exception:
                            pass

                        # KISS加速时钟：主循环步进为60秒/scale
                        time.sleep(max(0.1, 60 / max(1.0, float(getattr(trading_params, 'time_scale', 1.0)))))

                    except KeyboardInterrupt:
                        raise  # 向上传递中断信号
                    except Exception as e:
                        print(f"主循环错误: {str(e)}")
                        # 错误后等待保留真实间隔，避免重试风暴
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
                    try:
                        notifier.send_message(
                            "实时交易系统停止",
                            "系统已安全关闭"
                        )
                    except Exception:
                        pass
                    # 同步向Server酱发送后台停止通知
                    try:
                        hb_channel = getattr(trading_params, 'serverchan_heartbeat_channel', '')
                        notifier.send_message_serverchan(
                            "实时交易系统停止",
                            "系统已安全关闭",
                            channel=hb_channel or trading_params.serverchan_channel
                        )
                    except Exception:
                        pass

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