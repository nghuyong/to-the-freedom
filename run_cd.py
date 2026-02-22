"""
CD 指标查看工具 — 获取股票行情并计算 MACD 背离信号。

用法:
    python run_cd.py HOOD                  # 日线（默认，NASDAQ）
    python run_cd.py HOOD 4h               # 4 小时线
    python run_cd.py HOOD 1w               # 周线
    python run_cd.py CVX 1d NYSE           # 指定交易所
"""

import sys

from data.feed import fetch_tv, INTERVAL_MAP, INTERVAL_LABEL
from indicators.cd import compute_cd


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

    result = compute_cd(df)
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
