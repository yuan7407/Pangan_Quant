# 盘感 Quant - A Project from Trading Instinct

"盘感"项目为两个本人在用的量化策略：
1. 针对ETF类型的长期低风险控制回撤策略，比如QQQ胜率能到达70%的量化程序。基于 TD序列（神奇九转） 和 市场趋势判断，结合均线、MACD、RSI和成交量分析的综合性 ETF量化交易策略，通过server酱推送通知到绑定微信上。
2. 针对热门股票的短期高风险高爆发的股票的量化策略，比如NVDA/TSLA/PLTR等。基于 市场趋势判断，结合均线、MACD、RSI和成交量分析的高风险高回报量化交易策略，通过server酱推送通知到绑定微信上。 PLTR年化利率：319%，TSLA：47.7%。

## 目录结构
- `src/strategy/` 策略相关代码
- `src/backtest/` 回测相关代码
- `src/utils/` 工具脚本
- `logs/` 回测日志与输出
- `docs/` 项目文档

## 快速开始
1. 安装依赖：`pip install -r requirements.txt`
2. 运行主程序或回测脚本

## 依赖环境
- Python 3.8+
- 见 requirements.txt

## 贡献
欢迎提交PR或issue。
