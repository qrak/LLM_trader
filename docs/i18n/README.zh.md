# Semantic Signal — 一个把市场当作文本阅读、并记住已平仓交易的 LLM 交易智能体

[![Python 3.13](https://img.shields.io/badge/python-3.13-blue?logo=python&logoColor=white)](https://www.python.org)
[![测试](https://img.shields.io/badge/tests-3%2C043-brightgreen)](../../tests)
[![许可证: MIT](https://img.shields.io/badge/license-MIT-green)](../../LICENSE.md)
[![tests](https://github.com/qrak/LLM_trader/actions/workflows/tests.yml/badge.svg?branch=master)](https://github.com/qrak/LLM_trader/actions/workflows/tests.yml)
[![CodeQL](https://github.com/qrak/LLM_trader/actions/workflows/codeql.yml/badge.svg?branch=master)](https://github.com/qrak/LLM_trader/actions/workflows/codeql.yml)

**其他语言版本：** [English](../../README.md) · [Polski](README.pl.md) · **中文** · [日本語](README.ja.md) · [한국어](README.ko.md) · [Español](README.es.md) · [Deutsch](README.de.md) · [Português](README.pt.md) · [Русский](README.ru.md)

![实时仪表盘：账户自首笔交易以来的收益、当前决策与本次运行成本](../assets/dashboard-overview.webp)

**Semantic Signal** 是一个实验性交易智能体。每根 K 线收盘时，它把市场指标、新闻和已平仓交易的记忆发送给文本模型，然后用确定性的风险规则校验答案。图表只用于仪表盘和 Discord，不是模型输入。下方公开结果均为模拟结果；可选的订单转发需要单独的执行器。

[实时仪表盘](https://semanticsignal.qrak.org) · [交易账本](https://qrak.org/results/) · [一轮循环如何运作](https://qrak.org/story/) · [文档](../README.md)

## 结果

**窗口：** 2026-09-20 → 2026-11-25（67 天），BTC/USDC，4h K 线，10 000 USDC 模拟资金。
**来源：** 机器人自己的账本，冻结在 [`results/2026-11-25/`](../../results/2026-11-25)；`python scripts/results_report.py --data-dir results/2026-11-25` 可重新打印下面的每一个数字。

| 指标 | 机器人 | 买入并持有 |
|---|---|---|
| 收益（毛） | **+0.71%**（+$70.78） | **-27.91%** |
| 已平仓交易 | 23（计入 23 笔）+ 8 笔被排除（成交未核实或从未成交） | — |
| 胜率 | 52.2%（12 胜 / 11 负） | — |
| 盈利因子 | 1.91 | — |
| 夏普 / 索提诺（按单笔交易，未年化） | 0.31 / 0.74 | — |
| 最大回撤 | -0.25% | — |
| 最佳 / 最差交易 | +6.29% / -2.67% | — |
| 推理成本 | $7.64（输入 11.8M / 输出 5.2M tokens） | — |
| 扣除推理成本后的净额 | +$63.14 | — |
| 分析次数 | 224 个循环，中位数 259 秒，每次 $0.0093 | — |

![同一窗口内机器人与买入并持有的对比](../assets/results-vs-btc-2026-11-25.png)

在这 67 天里，机器人收益 +0.71%，而标的下跌了 27.91%。在 23 笔计入的交易上，
这相当于 28.6 个百分点——并不构成存在优势的证据。
收益未扣除交易手续费。统计排除了八组交易对：其成交无法与真实市场核对，
或其入场从未执行。上述数字来自那份冻结快照，
而非实时账本，因此不会随着机器人运行的每一个循环而变动。
方法、逐笔交易表和复现命令：
`python scripts/results_report.py --data-dir results/2026-11-25`。

> 本译文是一份冻结的副本，最新官方数字请以 [English README](../../README.md) 为准。

## 快速开始

```bash
git clone https://github.com/qrak/LLM_trader.git && cd LLM_trader
python -m venv .venv && source .venv/bin/activate  # Windows： .venv\Scripts\Activate.ps1
pip install --no-compile -r requirements.txt -c constraints.txt
cp config/config.ini.example config/config.ini  # 受版本管理的模板；如有需要请修改 provider
cp keys.env.example keys.env  # 或导出一个 provider 密钥；本地服务器上 `provider = custom` 不需要任何密钥
python start.py               # 仪表盘位于 http://localhost:8000
```

仅支持 Python **3.13**，约 3 GB 磁盘空间，一个 API 密钥（或任何兼容 OpenAI 或 Anthropic 的本地服务器）。各平台启动器、
Windows 注意事项和完整密钥表：[docs/SETUP.md](../SETUP.md)。

## 了解更多

- [功能与实现](../FEATURES.md) · [配置](../CONFIGURATION.md) · [本地设置面板](../SETTINGS_PANEL.md)
- [参与贡献](../../CONTRIBUTING.md) · [智能体指引](../../AGENTS.md) · [安全](../../SECURITY.md) · [更新日志](../../CHANGELOG.md)
```bash
python -m pytest tests -q  # 3,043 个测试，约 65 秒
```

## 许可证

[MIT](../../LICENSE.md)。实验性软件，非财务建议；模拟结果不能证明盈利能力。
