#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高级日志压缩工具 - 将回测终端输出压缩为极小的JSON格式
"""

import sys
import json
import re
import os
from io import StringIO
from collections import defaultdict

def read_rtf_file(rtf_file):
    """读取RTF文件并转换为纯文本"""
    with open(rtf_file, 'r', encoding='utf-8', errors='ignore') as f:
        rtf_content = f.read()

    # 简单的RTF标记处理
    # 移除所有RTF标记
    plain_text = re.sub(r'\\[a-z]+[-]?[0-9]*', ' ', rtf_content)
    # 移除花括号
    plain_text = re.sub(r'[{}]', '', plain_text)
    # 移除控制字符
    plain_text = re.sub(r'[^\x20-\x7E\n]', '', plain_text)

    return plain_text

def compress_log(input_file=None, output_file="backtest_log.json", compression_level=2):
    """
    将回测日志压缩为优化的JSON格式

    Args:
        input_file (str): 包含终端输出的文本文件路径，默认为None (标准输入)
        output_file (str): 输出JSON文件的路径
        compression_level (int): 压缩级别（1-3，数字越大压缩越强）
    """
    # 日志数据结构
    log_data = {
        "v": 1,  # 版本号
        "m": [   # 元数据
            {"t": "cmd", "v": {"sp": ""}},  # 命令
            {"t": "cfg", "v": {}},          # 配置
            {"t": "prm", "v": {}}           # 参数
        ],
        "mgr": [],  # 管理器信息
        "tpl": [],  # 模板定义
        "log": [],  # 日志（按日期分组）
        "report": { # 报告
            "stats": {},     # 统计数据
            "position": {},  # 持仓信息
            "costs": {},     # 成本信息
            "benchmark": {}  # 基准对比
        }
    }

    # 读取输入
    content = ""
    if input_file and os.path.exists(input_file):
        if input_file.lower().endswith('.rtf'):
            content = read_rtf_file(input_file)
        else:
            with open(input_file, 'r', encoding='utf-8') as f:
                content = f.read()
    else:
        # 从标准输入读取
        content = sys.stdin.read()

    if not content:
        print("错误: 没有找到输入内容")
        return False

    # 1. 提取命令行
    cmd_match = re.search(r"python\s+([\w\.]+\.py)", content)
    if cmd_match:
        log_data["m"][0]["v"]["sp"] = f"python {cmd_match.group(1)}"

    # 2. 提取配置信息
    cfg = log_data["m"][1]["v"]

    # 提取股票代码
    symbol_match = re.search(r"输入股票代码.*\[默认:\s*(\w+)\]", content)
    if symbol_match:
        cfg["sy"] = symbol_match.group(1)

    # 提取初始资金
    cash_match = re.search(r"输入初始资金.*\[默认:\s*(\d+)\]", content)
    if cash_match:
        cfg["ic"] = int(cash_match.group(1))

    # 提取开始日期
    start_date_match = re.search(r"输入开始日期.*\s+(\d{4}-\d{2}-\d{2})", content)
    if start_date_match:
        cfg["sd"] = start_date_match.group(1)

    # 提取结束日期
    param_end_date = re.search(r"设置策略参数: end_date = (\d{4}-\d{2}-\d{2})", content)
    if param_end_date:
        cfg["ed"] = param_end_date.group(1)

    # 设置时间框架和盘前/盘后交易（默认值）
    cfg["tf"] = "1d"
    cfg["po"] = "y"

    # 3. 提取策略参数
    params = log_data["m"][2]["v"]
    for line in content.split('\n'):
        param_match = re.match(r"设置策略参数: (\w+) = (.*)", line)
        if param_match:
            name, value = param_match.group(1), param_match.group(2)

            # 类型转换
            if value == "True":
                value = True
            elif value == "False":
                value = False
            else:
                try:
                    if "." in value:
                        value = float(value)
                    else:
                        value = int(value)
                except ValueError:
                    pass  # 保持字符串

            # 跳过配置中已有的参数
            if name not in ["symbol", "start_date", "end_date"]:
                params[name] = value

    # 4. 提取管理器信息
    for line in content.split('\n'):
        if "追踪止损激活阈值" in line or "初始化仓位管理" in line:
            # 移除日期前缀
            cleaned_line = re.sub(r"^\d{4}-\d{2}-\d{2}\s+", "", line.strip())
            log_data["mgr"].append({"t": "i", "tx": cleaned_line})

    # 5. 创建增强版模板定义
    log_data["tpl"] = create_enhanced_templates()

    # 6. 提取按日期分组的日志，应用增强压缩
    extract_logs_with_compression(content, log_data, compression_level)

    # 7. 提取报告数据
    extract_report(content, log_data)

    # 8. 进行最终优化，删除空或冗余字段
    optimize_json_structure(log_data)

    # 保存为JSON文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, ensure_ascii=False)

    # 计算压缩率
    input_size = len(content)
    output_size = os.path.getsize(output_file)
    compression_ratio = (1 - (output_size / input_size)) * 100

    print(f"终端输出已保存为压缩JSON格式: {output_file}")
    print(f"原始大小: {input_size:,} 字节, 压缩后: {output_size:,} 字节")
    print(f"压缩率: {compression_ratio:.2f}%")

    return True

def create_enhanced_templates():
    """创建增强版日志模板，覆盖更多模式"""
    return [
        # 趋势和信号模板
        {"id":"t1","tx":"趋势计算 - 价格评分: $1, 动量评分: $2"},
        {"id":"t2","tx":"趋势强度计算使用的参数 - etf_bias_mult: $1, trend_price_up_score: $2, trend_price_accel_score: $3"},
        {"id":"s1","tx":"买入信号参数检查 - td_count=$1, 信号阈值=$2"},
        {"id":"s2","tx":"买入信号检测 - 关键参数: td_count=$1, 信号阈值=$2"},
        {"id":"s3","tx":"信号整合详情 - TD序列: $1 (权重=$2), 趋势: $3 (权重=$4), 动量: $5 (权重=$6), 形态: $7 (权重=$8), 总分: $9, 阈值: $10"},
        {"id":"s4","tx":"买入决策: $1 - 总分$2 vs 阈值$3"},
        {"id":"s5","tx":"买入决策 - 信号=$1, 使用参数: td_count=$2, 信号阈值=$3"},

        # 止损相关模板
        {"id":"sl1","tx":"止损标志详情 - 基础止损启用? $1, 追踪止损启用? $2, 趋势反转止损启用? $3"},
        {"id":"sl2","tx":"止损计算 - 直接使用stop_atr=$1，无默认值"},
        {"id":"sl3","tx":"ATR止损详细计算: current_price($1) - (atr($2) * stop_atr($3)) = $4"},
        {"id":"sl4","tx":"最终止损决策 - stop_atr=$1产生直接影响: 止损价=$2"},
        {"id":"sl5","tx":"检查获利保护: profit_trigger_1=$1, profit_trigger_2=$2, profit_lock_pct=$3, 当前盈利率=$4%"},

        # 回调相关模板
        {"id":"cb1","tx":"回调参数详细 - 基础回调: $1, 每批次增加: $2, 市场状态: $3, 趋势强度: $4"},
        {"id":"cb2","tx":"回调计算详细: base_dip_pct=$1, added_batches=$2, additional_pct_per_batch=$3, required_dip_pct计算=$4+($5*$6)=$7"},
        {"id":"cb3","tx":"最终回调阈值计算: dip_threshold = $1 - $2 = $3"},
        {"id":"cb4","tx":"回调条件检查: 当前价格($1) / 上次买入($2) = $3 <= 阈值($4)? $5"},

        # 执行交易相关模板
        {"id":"e1","tx":"执行买入 - 规模: $1, 价格: $2, $3"},
        {"id":"e2","tx":"ENTRY $1股 @ $2"},
        {"id":"e3","tx":"买入执行 - 价格: $1, 数量: $2, 手续费: $3, 日期: $4"},
        {"id":"e4","tx":"EXIT $1股 @ $2 | 平仓价 $3 P&L: $4"},
        {"id":"e5","tx":"$1平仓 - 卖出 $2股"},
        {"id":"e6","tx":"卖出执行 - 价格: $1, 数量: $2, 手续费: $3, 状态: $4"},

        # 警告模板
        {"id":"w1","tx":"警告: 为参数 '$1' 提供的默认值 $2 被忽略"},
        {"id":"w2","tx":"卖出冷却期 ($1/$2 天)"},

        # TD序列模板
        {"id":"td1","tx":"TD序列信号生效 - td_count=$1直接影响, td_up=$2"},

        # 其他常见模式
        {"id":"x1","tx":"确认基础止损信号 - 执行平仓"}
    ]

def match_log_to_enhanced_template(log_text):
    """增强版模板匹配，更全面地捕获日志模式"""

    # TD序列信号生效
    match = re.match(r"TD序列信号生效 - td_count=(\d+)直接影响, td_up=(\d+)", log_text)
    if match:
        return {"r": "td1", "p": [int(match.group(1)), int(match.group(2))]}

    # 获利保护检查
    match = re.match(r"检查获利保护: profit_trigger_1=([\d\.]+), profit_trigger_2=([\d\.]+), profit_lock_pct=([\d\.]+), 当前盈利率=([\d\.\-]+)%", log_text)
    if match:
        return {"r": "sl5", "p": [float(match.group(1)), float(match.group(2)), float(match.group(3)), float(match.group(4))]}

    # 警告 - 参数默认值被忽略
    match = re.match(r"警告: 为参数 '(\w+)' 提供的默认值 ([\d\.]+) 被忽略", log_text)
    if match:
        return {"r": "w1", "p": [match.group(1), float(match.group(2))]}

    # 卖出冷却期
    match = re.match(r"卖出冷却期 \((\d+)/(\d+) 天\)", log_text)
    if match:
        return {"r": "w2", "p": [int(match.group(1)), int(match.group(2))]}

    # 趋势计算
    match = re.match(r"趋势计算 - 价格评分: ([\d\.]+), 动量评分: ([\d\.]+)", log_text)
    if match:
        return {"r": "t1", "p": [float(match.group(1)), float(match.group(2))]}

    # 趋势强度参数
    match = re.match(r"趋势强度计算使用的参数 - etf_bias_mult: ([\d\.]+), trend_price_up_score: ([\d\.]+), trend_price_accel_score: ([\d\.]+)", log_text)
    if match:
        return {"r": "t2", "p": [float(match.group(1)), float(match.group(2)), float(match.group(3))]}

    # 买入信号参数检查
    match = re.match(r"买入信号参数检查 - td_count=(\d+), 信号阈值=([\d\.]+)", log_text)
    if match:
        return {"r": "s1", "p": [int(match.group(1)), float(match.group(2))]}

    # 买入信号检测
    match = re.match(r"买入信号检测 - 关键参数: td_count=(\d+), 信号阈值=([\d\.]+)", log_text)
    if match:
        return {"r": "s2", "p": [int(match.group(1)), float(match.group(2))]}

    # 信号整合详情
    match = re.match(r"信号整合详情 - TD序列: ([\d\.]+) \(权重=([\d\.]+)\), 趋势: ([\d\.]+) \(权重=([\d\.]+)\), 动量: ([\d\.]+) \(权重=([\d\.]+)\), 形态: ([\d\.]+) \(权重=([\d\.]+)\), 总分: ([\d\.]+), 阈值: ([\d\.]+)", log_text)
    if match:
        return {"r": "s3", "p": [
            float(match.group(1)), float(match.group(2)),
            float(match.group(3)), float(match.group(4)),
            float(match.group(5)), float(match.group(6)),
            float(match.group(7)), float(match.group(8)),
            float(match.group(9)), float(match.group(10))
        ]}

    # 买入决策
    match = re.match(r"买入决策: (是|否) - 总分([\d\.]+) vs 阈值([\d\.]+)", log_text)
    if match:
        return {"r": "s4", "p": [match.group(1), float(match.group(2)), float(match.group(3))]}

    # 买入决策信号
    match = re.match(r"买入决策 - 信号=(True|False), 使用参数: td_count=(\d+), 信号阈值=([\d\.]+)", log_text)
    if match:
        return {"r": "s5", "p": [match.group(1).lower() == "true", int(match.group(2)), float(match.group(3))]}

    # 止损标志详情
    match = re.match(r"止损标志详情 - 基础止损启用\? (True|False), 追踪止损启用\? (True|False), 趋势反转止损启用\? (True|False)", log_text)
    if match:
        return {"r": "sl1", "p": [
            match.group(1) == "True",
            match.group(2) == "True",
            match.group(3) == "True"
        ]}

    # 止损计算
    match = re.match(r"止损计算 - 直接使用stop_atr=([\d\.]+)，无默认值", log_text)
    if match:
        return {"r": "sl2", "p": [float(match.group(1))]}

    # ATR止损详细计算
    match = re.match(r"ATR止损详细计算: current_price\(([\d\.]+)\) - \(atr\(([\d\.]+)\) \* stop_atr\(([\d\.]+)\)\) = ([\d\.]+)", log_text)
    if match:
        return {"r": "sl3", "p": [float(match.group(1)), float(match.group(2)), float(match.group(3)), float(match.group(4))]}

    # 最终止损决策
    match = re.match(r"最终止损决策 - stop_atr=([\d\.]+)产生直接影响: 止损价=([\d\.]+)", log_text)
    if match:
        return {"r": "sl4", "p": [float(match.group(1)), float(match.group(2))]}

    # 执行买入
    match = re.match(r"执行买入 - 规模: (\d+), 价格: ([\d\.]+), (.*)", log_text)
    if match:
        return {"r": "e1", "p": [int(match.group(1)), float(match.group(2)), match.group(3)]}

    # 入场记录
    match = re.match(r"ENTRY (\d+)股 @ ([\d\.]+)", log_text)
    if match:
        return {"r": "e2", "p": [int(match.group(1)), float(match.group(2))]}

    # 买入执行
    match = re.match(r"买入执行 - 价格: ([\d\.]+), 数量: (\d+), 手续费: ([\d\.]+), 日期: (.*)", log_text)
    if match:
        return {"r": "e3", "p": [float(match.group(1)), int(match.group(2)), float(match.group(3)), match.group(4)]}

    # 平仓记录
    match = re.match(r"EXIT (\d+)股 @ ([\d\.]+) \| 平仓价 ([\d\.]+) P&L: ([+-]?[\d\.]+)", log_text)
    if match:
        return {"r": "e4", "p": [int(match.group(1)), float(match.group(2)), float(match.group(3)), float(match.group(4))]}

    # 平仓类型
    match = re.match(r"(部分|全部)平仓 - 卖出 (\d+)股", log_text)
    if match:
        return {"r": "e5", "p": [match.group(1), int(match.group(2))]}

    # 卖出执行
    match = re.match(r"卖出执行 - 价格: ([\d\.]+), 数量: (\d+), 手续费: ([\d\.]+), 状态: (\d+)", log_text)
    if match:
        return {"r": "e6", "p": [float(match.group(1)), int(match.group(2)), float(match.group(3)), int(match.group(4))]}

    # 回调参数详细
    match = re.match(r"回调参数详细 - 基础回调: ([\d\.]+), 每批次增加: ([\d\.]+), 市场状态: (\w+), 趋势强度: ([\d\.]+)", log_text)
    if match:
        return {"r": "cb1", "p": [float(match.group(1)), float(match.group(2)), match.group(3), float(match.group(4))]}

    # 回调计算详细
    match = re.match(r"回调计算详细: base_dip_pct=([\d\.]+), added_batches=(\d+), additional_pct_per_batch=([\d\.]+), required_dip_pct计算=([\d\.]+)\+\((\d+)\*([\d\.]+)\)=([\d\.]+)", log_text)
    if match:
        return {"r": "cb2", "p": [
            float(match.group(1)), int(match.group(2)), float(match.group(3)),
            float(match.group(4)), int(match.group(5)), float(match.group(6)), float(match.group(7))
        ]}

    # 最终回调阈值计算
    match = re.match(r"最终回调阈值计算: dip_threshold = ([\d\.]+) - ([\d\.]+) = ([\d\.]+)", log_text)
    if match:
        return {"r": "cb3", "p": [float(match.group(1)), float(match.group(2)), float(match.group(3))]}

    # 回调条件检查
    match = re.match(r"回调条件检查: 当前价格\(([\d\.]+)\) / 上次买入\(([\d\.]+)\) = ([\d\.]+) <= 阈值\(([\d\.]+)\)\? (True|False)", log_text)
    if match:
        return {"r": "cb4", "p": [
            float(match.group(1)), float(match.group(2)), float(match.group(3)),
            float(match.group(4)), match.group(5)
        ]}

    # 特殊事件 - 止损触发
    if "确认基础止损信号 - 执行平仓" in log_text:
        return {"r": "x1", "p": []}

    # 无模板匹配，返回原始文本
    return {"tx": log_text}

def extract_logs_with_compression(content, log_data, compression_level=2):
    """提取日志并应用高级压缩技术"""
    date_pattern = re.compile(r"(\d{4}-\d{2}-\d{2})\s+(.*)")

    # 追踪每日日志类型频率，用于高度压缩
    date_log_frequency = defaultdict(lambda: defaultdict(int))
    date_log_samples = defaultdict(dict)

    # 收集所有日志项目
    all_logs = []

    # 第一阶段：收集和分类日志
    for line in content.split('\n'):
        match = date_pattern.match(line)
        if match:
            date, log_text = match.group(1), match.group(2).strip()
            if log_text:
                # 匹配日志文本到模板
                action = match_log_to_enhanced_template(log_text)
                if action:
                    # 创建识别码，用于频率计数
                    if "r" in action:
                        log_id = f"r:{action['r']}"
                    else:
                        # 对于非模板文本，使用前20个字符作为ID
                        log_id = f"tx:{action['tx'][:20]}"

                    # 增加该类型日志的计数
                    date_log_frequency[date][log_id] += 1

                    # 保存样本（用于高频率日志）
                    if log_id not in date_log_samples[date]:
                        date_log_samples[date][log_id] = action

                    # 保存完整日志项目
                    all_logs.append((date, action))

    # 第二阶段：应用压缩策略（根据压缩级别）
    log_entries_by_date = defaultdict(list)

    # 定义各压缩级别的频率阈值
    # 当日志类型超过这个频率时，将被压缩
    frequency_thresholds = {
        1: 100,  # 轻度压缩
        2: 50,   # 中度压缩
        3: 20    # 高度压缩
    }
    threshold = frequency_thresholds.get(compression_level, 50)

    # 应用压缩策略
    for date, action in all_logs:
        # 识别此日志类型
        if "r" in action:
            log_id = f"r:{action['r']}"
        else:
            log_id = f"tx:{action['tx'][:20]}"

        # 检查是否为高频率日志
        freq = date_log_frequency[date][log_id]

        # 压缩策略：
        # 1. 对于重要日志类型（交易执行、止损等），始终保留
        # 2. 对于普通日志，如果频率小于阈值，保留
        # 3. 对于高频率普通日志，仅保留样本
        important_types = ["e1", "e2", "e3", "e4", "e5", "e6", "sl4", "x1", "td1"]

        # 检查是否为重要日志类型或频率低于阈值
        keep_log = False
        if "r" in action and action["r"] in important_types:
            keep_log = True  # 重要日志类型，始终保留
        elif freq <= threshold:
            keep_log = True  # 频率低于阈值，保留
        elif log_id in date_log_samples[date] and len(log_entries_by_date[date]) < 10:
            # 高频率日志，保留样本（每天最多10个样本）
            keep_log = True

        if keep_log:
            log_entries_by_date[date].append(action)

    # 初始化 merged_dates 集合，确保在所有压缩级别下都有定义
    merged_dates = set()

    # 压缩级别3：合并高频率相似天
    if compression_level >= 3:
        # 分析日期之间的相似度
        dates = sorted(log_entries_by_date.keys())
        merged_dates = set()

        for i in range(len(dates) - 1):
            if dates[i] in merged_dates:
                continue

            # 检查连续日期的相似度
            curr_day_logs = log_entries_by_date[dates[i]]
            next_day_logs = log_entries_by_date[dates[i+1]]

            # 简单相似度检查：如果两天日志类型和数量相近，则合并
            if abs(len(curr_day_logs) - len(next_day_logs)) <= 3:
                # 合并到当前日期
                merged_dates.add(dates[i+1])
                # 取两天中更有代表性的日志（通常是有更多交易活动的那天）
                if len(next_day_logs) > len(curr_day_logs):
                    log_entries_by_date[dates[i]] = next_day_logs

    # 构建最终的日志数据
    for date in sorted(log_entries_by_date.keys()):
        if date in merged_dates:
            continue  # 跳过已合并的日期

        log_data["log"].append({
            "d": date,
            "a": log_entries_by_date[date]
        })

def extract_report(content, log_data):
    """提取回测报告数据"""
    # 查找报告区域
    report_start = content.find("==================== 策略回测详细报告 ====================")
    if report_start == -1:
        return  # 未找到报告

    report_end = content.find("==================================================", report_start)
    if report_end == -1:
        report_end = len(content)

    report_text = content[report_start:report_end]

    # 提取统计数据
    stats = log_data["report"]["stats"]

    # 收益统计
    initial_match = re.search(r"初始资金: \$([\d,\.]+)", report_text)
    if initial_match:
        stats["initial"] = float(initial_match.group(1).replace(',', ''))

    final_match = re.search(r"最终资金: \$([\d,\.]+)", report_text)
    if final_match:
        stats["final"] = float(final_match.group(1).replace(',', ''))

    net_match = re.search(r"净收益: \$([\d,\.]+)", report_text)
    if net_match:
        stats["net"] = float(net_match.group(1).replace(',', ''))

    return_match = re.search(r"总收益率: ([\d\.]+)%", report_text)
    if return_match:
        stats["return"] = float(return_match.group(1))

    annual_match = re.search(r"年化收益率: ([\d\.]+)%", report_text)
    if annual_match:
        stats["annual"] = float(annual_match.group(1))

    # 交易统计
    trades_match = re.search(r"总交易次数: (\d+)", report_text)
    if trades_match:
        stats["trades"] = int(trades_match.group(1))

    winrate_match = re.search(r"胜率: ([\d\.]+)%", report_text)
    if winrate_match:
        stats["winrate"] = float(winrate_match.group(1))

    avg_match = re.search(r"平均每笔收益: \$([\d,\.]+)", report_text)
    if avg_match:
        stats["avg"] = float(avg_match.group(1).replace(',', ''))

    max_win_match = re.search(r"最大单笔收益: \$([\d,\.]+)", report_text)
    if max_win_match:
        stats["max_win"] = float(max_win_match.group(1).replace(',', ''))

    max_loss_match = re.search(r"最大单笔亏损: \$([+-]?[\d,\.]+)", report_text)
    if max_loss_match:
        stats["max_loss"] = float(max_loss_match.group(1).replace(',', ''))

    # 风险统计
    drawdown_match = re.search(r"最大回撤: ([\d\.]+)%", report_text)
    if drawdown_match:
        stats["drawdown"] = float(drawdown_match.group(1))

    volatility_match = re.search(r"收益波动率\(年化\): ([\d\.]+)%", report_text)
    if volatility_match:
        stats["volatility"] = float(volatility_match.group(1))

    sharpe_match = re.search(r"夏普比率: ([\d\.]+)", report_text)
    if sharpe_match:
        stats["sharpe"] = float(sharpe_match.group(1))

    sortino_match = re.search(r"索提诺比率: ([\d\.]+)", report_text)
    if sortino_match:
        stats["sortino"] = float(sortino_match.group(1))

    # 持仓信息
    position = log_data["report"]["position"]

    shares_match = re.search(r"持仓数量: (\d+)股", report_text)
    if shares_match:
        position["shares"] = int(shares_match.group(1))

    cost_match = re.search(r"持仓成本: \$([\d\.]+)", report_text)
    if cost_match:
        position["cost"] = float(cost_match.group(1))

    value_match = re.search(r"持仓市值: \$([\d,\.]+)", report_text)
    if value_match:
        position["value"] = float(value_match.group(1).replace(',', ''))

    # 成本统计
    costs = log_data["report"]["costs"]

    total_cost_match = re.search(r"总交易成本: \$([\d\.]+)", report_text)
    if total_cost_match:
        costs["total"] = float(total_cost_match.group(1))

    cost_pct_match = re.search(r"成本占初始资金比例: ([\d\.]+)%", report_text)
    if cost_pct_match:
        costs["pct"] = float(cost_pct_match.group(1))

    # 基准对比
    benchmark = log_data["report"]["benchmark"]

    bh_return_match = re.search(r"买入持有收益率: ([\d\.]+)%", report_text)
    if bh_return_match:
        benchmark["return"] = float(bh_return_match.group(1))

    excess_match = re.search(r"超额收益率: ([+-]?[\d\.]+)%", report_text)
    if excess_match:
        benchmark["excess"] = float(excess_match.group(1))

    bh_drawdown_match = re.search(r"买入持有最大回撤: ([\d\.]+)%", report_text)
    if bh_drawdown_match:
        benchmark["drawdown"] = float(bh_drawdown_match.group(1))

    # 提取浮动盈亏（如果有）
    pnl_match = re.search(r"浮动盈亏: \$([\d,\.]+)", content)
    if pnl_match:
        position["pnl"] = float(pnl_match.group(1).replace(',', ''))

def optimize_json_structure(log_data):
    """优化JSON结构，删除空字段和减少小数点位数"""

    # 优化函数：对数值截取小数点
    def optimize_number(value):
        if isinstance(value, float):
            # 对小数部分较长的数字进行截断
            return round(value, 3)
        return value

    # 递归处理字典中的数值
    def process_dict(d):
        if not isinstance(d, dict):
            return d

        result = {}
        for k, v in d.items():
            if isinstance(v, dict):
                processed = process_dict(v)
                if processed:  # 不保存空字典
                    result[k] = processed
            elif isinstance(v, list):
                processed = process_list(v)
                if processed:  # 不保存空列表
                    result[k] = processed
            elif v is not None:  # 不保存None值
                result[k] = optimize_number(v)
        return result

    # 递归处理列表中的数值
    def process_list(lst):
        return [
            process_dict(item) if isinstance(item, dict) else
            process_list(item) if isinstance(item, list) else
            optimize_number(item)
            for item in lst
        ]

    # 应用优化
    optimized_data = process_dict(log_data)

    # 更新原始数据
    log_data.clear()
    log_data.update(optimized_data)

if __name__ == "__main__":
    # 处理命令行参数
    import argparse
    parser = argparse.ArgumentParser(description='将回测日志压缩为优化的JSON格式')
    parser.add_argument('--input', '-i', help='输入的日志文件', default=None)
    parser.add_argument('--output', '-o', help='输出的JSON文件', default='backtest_log.json')
    parser.add_argument('--level', '-l', type=int, choices=[1, 2, 3], default=2,
                       help='压缩级别 (1=轻度, 2=中度, 3=高度)')
    args = parser.parse_args()

    # 调用函数
    compress_log(args.input, args.output, args.level)
