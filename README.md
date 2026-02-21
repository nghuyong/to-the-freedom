# To The Freedom

基于技术指标的多市场股票筛选工具，支持美股、A 股、港股。

## 筛选条件

1. **NX 指标** — 日线 / 周线蓝色均在黄色之上
2. **SMA200** — 股价位于 200 日均线之上
3. **CD 抄底 + 站稳蓝色梯子** — 日线或 4h 级别出现 CD 信号，且收盘价持续站上蓝色梯子下边缘

## 快速开始

```bash
pip install -r requirements.txt
```

### 单只股票检查

```bash
python stock_filter.py HOOD
python stock_filter.py AAPL NYSE
```

### 批量筛选

```bash
python run_batch_filter.py                          # 美股 Top 200
python run_batch_filter.py --market cn --top 200    # 沪深 Top 200
python run_batch_filter.py --market hk --top 100    # 港股 Top 100
python run_batch_filter.py --lookback 30            # 自定义回看 K 线数
python run_batch_filter.py -o my_result.json        # 指定输出文件名
```

### 单指标运行

```bash
python run_nx.py AAPL           # NX 指标
python run_ma.py AAPL           # 均线指标
python run_cd.py AAPL           # CD 指标
```

## 支持市场

| 市场 | 文件 | 交易所 |
|------|------|--------|
| 美股 | `stocks/us.csv` | NASDAQ / NYSE / AMEX |
| A 股 | `stocks/cn.csv` | SSE / SZSE |
| 港股 | `stocks/hk.csv` | HKEX |

## 项目结构

```
├── stock_filter.py        # 综合过滤器（3 个条件判断）
├── run_batch_filter.py    # 批量筛选入口
├── nx_indicator.py        # NX 指标计算
├── ma_indicator.py        # 均线指标计算
├── cd_indicator.py        # CD 指标计算
├── tv_market_cap.py       # TradingView 市值数据获取
├── run_nx.py / run_ma.py / run_cd.py   # 单指标运行脚本
├── stocks/                # 股票列表 CSV（Git LFS 追踪）
└── filter_results/        # 筛选结果输出（已 gitignore）
```
