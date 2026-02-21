"""
获取股票行情数据并打印最新均线点位。

用法:
    python run_ma.py HOOD              # 日线 SMA（默认）
    python run_ma.py HOOD 4h           # 4 小时线
    python run_ma.py HOOD 1w           # 周线
    python run_ma.py HOOD 1d EMA       # 日线 EMA

依赖: pip install ta tradingview-datafeed
"""

import sys

from tvDatafeed import TvDatafeed, Interval

from ma_indicator import compute_ma

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
    ma_type = sys.argv[3].upper() if len(sys.argv) > 3 else "SMA"

    if interval not in _INTERVAL_MAP:
        print(f"不支持的周期: {interval}，可选: {', '.join(_INTERVAL_MAP)}")
        sys.exit(1)

    if ma_type not in ("SMA", "EMA"):
        print(f"不支持的均线类型: {ma_type}，可选: SMA, EMA")
        sys.exit(1)

    print(f"正在获取 {ticker} 行情数据 ({interval}) ...")

    df = fetch_tv(ticker, interval)
    interval_label = _INTERVAL_LABEL[interval]

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
            status = "数据不足"
            print(f"  {col:<8s}:  {status}")
        else:
            diff_pct = (close - val) / val * 100
            position = "↑" if close > val else "↓"
            print(f"  {col:<8s}:  {val:>10.3f}   {position} {diff_pct:+.2f}%")

    # Trend summary
    print(f"{'=' * 55}")
    ma20_col = f"{prefix}20"
    ma50_col = f"{prefix}50"
    ma200_col = f"{prefix}200"

    ma20 = latest.get(ma20_col)
    ma50 = latest.get(ma50_col)
    ma200 = latest.get(ma200_col)

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
    import pandas as pd

    main()
