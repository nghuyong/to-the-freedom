"""
单股过滤工具 — 对指定股票运行全部 3 个选股条件。

用法:
    python run_filter.py HOOD
    python run_filter.py AAPL NYSE
    python run_filter.py BABA NYSE 60    # 自定义 CD 回看根数
"""

import sys

from strategy.filter import run_filter


def main():
    ticker   = sys.argv[1] if len(sys.argv) > 1 else "HOOD"
    exchange = sys.argv[2] if len(sys.argv) > 2 else "NASDAQ"
    lookback = int(sys.argv[3]) if len(sys.argv) > 3 else 30

    run_filter(ticker, exchange, lookback_daily_bars=lookback)


if __name__ == "__main__":
    main()
