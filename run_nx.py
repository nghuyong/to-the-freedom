"""
NX 指标查看工具 — 获取股票行情并打印最新通道点位。

用法:
    python run_nx.py HOOD                  # 日线（默认，NASDAQ）
    python run_nx.py HOOD 4h               # 4 小时线
    python run_nx.py HOOD 1w               # 周线
    python run_nx.py CVX 1w NYSE           # 指定交易所
"""

import sys

from data.feed import fetch_tv, INTERVAL_MAP, INTERVAL_LABEL
from indicators.nx import compute_nx


def main():
    ticker   = sys.argv[1] if len(sys.argv) > 1 else "HOOD"
    interval = sys.argv[2] if len(sys.argv) > 2 else "1d"
    exchange = sys.argv[3] if len(sys.argv) > 3 else "NASDAQ"

    if interval not in INTERVAL_MAP:
        print(f"不支持的周期: {interval}，可选: {', '.join(INTERVAL_MAP)}")
        sys.exit(1)

    print(f"正在获取 {ticker} 行情数据 ({interval}) ...")

    df = fetch_tv(ticker, interval, exchange=exchange)
    interval_label = INTERVAL_LABEL[interval]

    if df is None or df.empty:
        print(f"未获取到 {ticker} 的行情数据，请检查股票代码是否正确。")
        sys.exit(1)

    result = compute_nx(df)
    latest = result.iloc[-1]
    date_str = str(result.index[-1])

    print(f"\n{'=' * 55}")
    print(f"  {ticker} NX 通道点位 — {date_str} ({interval_label})")
    print(f"{'=' * 55}")
    print(f"  蓝色上轨 (EMA24 High) :  {latest['BLUE_TOP']:.3f}")
    print(f"  蓝色下轨 (EMA23 Low)  :  {latest['BLUE_BOTTOM']:.3f}")
    print(f"  黄色上轨 (EMA89 High) :  {latest['YELLOW_TOP']:.3f}")
    print(f"  黄色下轨 (EMA90 Low)  :  {latest['YELLOW_BOTTOM']:.3f}")
    print(f"{'=' * 55}")

    if latest["BLUE_TOP"] > latest["YELLOW_TOP"]:
        trend = "多头排列 (蓝色通道在黄色上方)"
    elif latest["BLUE_TOP"] < latest["YELLOW_TOP"]:
        trend = "空头排列 (蓝色通道在黄色下方)"
    else:
        trend = "通道交织"

    print(f"  当前趋势: {trend}")

    if latest["LONG_SIGNAL"]:
        print("  *** 今日出现金叉信号 ***")
    if latest["SHORT_SIGNAL"]:
        print("  *** 今日出现死叉信号 ***")

    print()


if __name__ == "__main__":
    main()
