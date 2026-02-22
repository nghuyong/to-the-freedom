"""
MA 均线查看工具 — 获取股票行情并打印最新均线点位。

用法:
    python run_ma.py HOOD                  # 日线 SMA（默认，NASDAQ）
    python run_ma.py HOOD 4h               # 4 小时线
    python run_ma.py HOOD 1w               # 周线
    python run_ma.py HOOD 1d EMA           # 日线 EMA
    python run_ma.py CVX 1d SMA NYSE       # 指定交易所
"""

import sys
import pandas as pd

from data.feed import fetch_tv, INTERVAL_MAP, INTERVAL_LABEL
from indicators.ma import compute_ma


def main():
    ticker   = sys.argv[1] if len(sys.argv) > 1 else "HOOD"
    interval = sys.argv[2] if len(sys.argv) > 2 else "1d"
    ma_type  = sys.argv[3].upper() if len(sys.argv) > 3 else "SMA"
    exchange = sys.argv[4] if len(sys.argv) > 4 else "NASDAQ"

    if interval not in INTERVAL_MAP:
        print(f"不支持的周期: {interval}，可选: {', '.join(INTERVAL_MAP)}")
        sys.exit(1)

    if ma_type not in ("SMA", "EMA"):
        print(f"不支持的均线类型: {ma_type}，可选: SMA, EMA")
        sys.exit(1)

    print(f"正在获取 {ticker} 行情数据 ({interval}) ...")

    df = fetch_tv(ticker, interval, exchange=exchange)
    interval_label = INTERVAL_LABEL[interval]

    if df is None or df.empty:
        print(f"未获取到 {ticker} 的行情数据，请检查股票代码是否正确。")
        sys.exit(1)

    periods = [5, 10, 20, 50, 100, 200]
    result = compute_ma(df, periods=periods, ma_type=ma_type)
    latest = result.iloc[-1]
    date_str = str(result.index[-1])
    close = latest["Close"]
    prefix = "MA" if ma_type == "SMA" else "EMA"

    print(f"\n{'=' * 55}")
    print(f"  {ticker} {ma_type} 均线点位 — {date_str} ({interval_label})")
    print(f"{'=' * 55}")
    print(f"  当前价格: {close:.3f}")
    print(f"{'─' * 55}")

    for period in periods:
        col = f"{prefix}{period}"
        val = latest[col]
        if pd.isna(val):
            print(f"  {col:<8s}:  数据不足")
        else:
            diff_pct = (close - val) / val * 100
            position = "↑" if close > val else "↓"
            print(f"  {col:<8s}:  {val:>10.3f}   {position} {diff_pct:+.2f}%")

    print(f"{'=' * 55}")
    ma20 = latest.get(f"{prefix}20")
    ma50 = latest.get(f"{prefix}50")
    ma200 = latest.get(f"{prefix}200")

    if not pd.isna(ma20) and not pd.isna(ma50) and not pd.isna(ma200):
        if close > ma20 > ma50 > ma200:
            trend = "强多头排列 (价格 > MA20 > MA50 > MA200)"
        elif close > ma200:
            trend = "中期多头 (价格在 MA200 之上)"
        elif close < ma20 < ma50 < ma200:
            trend = "强空头排列 (价格 < MA20 < MA50 < MA200)"
        elif close < ma200:
            trend = "中期空头 (价格在 MA200 之下)"
        else:
            trend = "震荡整理"
        print(f"  趋势判断: {trend}")
    print()


if __name__ == "__main__":
    main()
