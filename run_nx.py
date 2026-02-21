"""
获取股票行情数据并打印最新 NX 通道点位。

用法:
    python run_nx.py HOOD          # 日线（默认）
    python run_nx.py HOOD 4h       # 4 小时线
    python run_nx.py HOOD 1w       # 周线

依赖: pip install ta tradingview-datafeed
"""

import sys

from tvDatafeed import TvDatafeed, Interval

from nx_indicator import compute_nx

_TV_NBARS = 5000

_INTERVAL_MAP = {
    "1d": Interval.in_daily,
    "4h": Interval.in_4_hour,
    "1w": Interval.in_weekly,
}

_INTERVAL_LABEL = {
    "1d": "日线",
    "4h": "4h",
    "1w": "周线",
}


def fetch_tv(ticker: str, interval: str, exchange: str = "NASDAQ"):
    """从 TradingView 获取 K线数据。"""
    tv = TvDatafeed()
    df = tv.get_hist(
        symbol=ticker,
        exchange=exchange,
        interval=_INTERVAL_MAP[interval],
        n_bars=_TV_NBARS,
    )
    df = df.rename(
        columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        }
    )
    return df


def main():
    ticker = sys.argv[1] if len(sys.argv) > 1 else "HOOD"
    interval = sys.argv[2] if len(sys.argv) > 2 else "1d"

    if interval not in _INTERVAL_MAP:
        print(f"不支持的周期: {interval}，可选: {', '.join(_INTERVAL_MAP)}")
        sys.exit(1)

    print(f"正在获取 {ticker} 行情数据 ({interval}) ...")

    df = fetch_tv(ticker, interval)
    interval_label = _INTERVAL_LABEL[interval]

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
