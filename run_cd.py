"""
获取股票行情数据并计算 CD（MACD 背离）指标。

用法:
    python run_cd.py HOOD          # 日线（默认）
    python run_cd.py HOOD 4h       # 4 小时线
    python run_cd.py HOOD 1w       # 周线

依赖: pip install ta tvDatafeed
"""

import sys

from tvDatafeed import TvDatafeed, Interval

from cd_indicator import compute_cd

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

    result = compute_cd(df)

    # ---------- 打印最新状态 ----------
    latest = result.iloc[-1]
    date_str = str(result.index[-1])

    print(f"\n{'=' * 55}")
    print(f"  {ticker} CD 指标 — {date_str} ({interval_label})")
    print(f"{'=' * 55}")
    print(f"  DIFF  : {latest['DIFF']:.4f}")
    print(f"  DEA   : {latest['DEA']:.4f}")
    print(f"  MACD  : {latest['MACD_HIST']:.4f}")
    print(f"{'=' * 55}")

    if latest["BUY_SIGNAL"]:
        print("  *** 今日 抄底 信号 ***")
    if latest["SELL_SIGNAL"]:
        print("  *** 今日 卖出 信号 ***")
    if latest["BUY_CANCEL"]:
        print("  (底背离消失)")
    if latest["SELL_CANCEL"]:
        print("  (顶背离消失)")

    # ---------- 打印历史信号 ----------
    buys = result[result["BUY_SIGNAL"]]
    sells = result[result["SELL_SIGNAL"]]

    print(f"\n{'─' * 55}")
    print(f"  历史 抄底 信号 (共 {len(buys)} 次):")
    print(f"{'─' * 55}")
    if len(buys) == 0:
        print("  (无)")
    else:
        for idx, row in buys.iterrows():
            print(f"    {str(idx):20s}  Close={row['Close']:.2f}  DIFF={row['DIFF']:.4f}")

    print(f"\n{'─' * 55}")
    print(f"  历史 卖出 信号 (共 {len(sells)} 次):")
    print(f"{'─' * 55}")
    if len(sells) == 0:
        print("  (无)")
    else:
        for idx, row in sells.iterrows():
            print(f"    {str(idx):20s}  Close={row['Close']:.2f}  DIFF={row['DIFF']:.4f}")

    print()


if __name__ == "__main__":
    main()
