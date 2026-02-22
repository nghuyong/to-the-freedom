"""
策略回测工具 — 蓝色通道抄底策略（4h 级别）批量回测。

买入条件（同时满足）:
  C2. 周级别 NX: 蓝色上边缘 > 黄色上边缘
  C3. 日级别 NX: 蓝色上边缘 > 黄色下边缘
  C4. 股价 > 200 日均线（SMA200）
  C5. 近期出现4h级别 CD 抄底信号
  C6. 股价连续 ≥ MIN_HOLD_BARS 根4h K线收盘高于4h级别蓝色下边缘（首次站上后未跌破，默认 3 根）
  C7. BB_4H ≤ 收盘价 ≤ BB_4H × 1.05，4h 级别，以4h收盘价买入

卖出条件: 股价4h收盘跌破4h级别蓝色下边缘

用法:
    python run_backtest_4h.py                                      # 美股 Top200, 2026-01-01 至今
    python run_backtest_4h.py --start 2025-10-01                   # 自定义起始日期
    python run_backtest_4h.py --top 100                            # Top 100
    python run_backtest_4h.py --symbol BABA TSLA AAPL              # 指定美股
    python run_backtest_4h.py --market hk --top 50                 # 港股 Top50
    python run_backtest_4h.py --symbol 9992 --market hk            # 港股: 泡泡玛特
    python run_backtest_4h.py --symbol 700 9988 --market hk        # 港股: 腾讯 + 阿里
    python run_backtest_4h.py --debug                              # 开启逐 bar 日志
"""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from strategy.backtest_4h import backtest_single, SKIP_LABELS, MIN_HOLD_BARS
from strategy.stocks import NumpyEncoder, load_top_stocks, build_exchange_lookup, resolve_exchange

_BACKTEST_START = "2026-01-01"
_RESULTS_DIR = Path(__file__).parent / "backtest_results"


def _print_summary(all_trades: list[dict]) -> None:
    """打印交易汇总。"""
    print(f"\n{'═' * 70}")
    print("  回测汇总 (4h 级别策略)")
    print(f"{'═' * 70}")

    if not all_trades:
        print("\n  ⚠ 无满足条件的交易\n")
        return

    print(f"\n  共 {len(all_trades)} 笔交易:\n")
    header = (
        f"  {'股票':<8s} {'CD信号':<18s} {'首站上':<18s} {'买入时间':<18s} {'买入价':>10s}"
        f"  {'卖出时间':<18s} {'卖出价':>10s}"
        f"  {'收益%':>8s}  {'根数':>4s}  {'状态'}"
    )
    print(header)
    print("  " + "─" * (len(header) - 2))

    for t in sorted(all_trades, key=lambda x: x["buy_date"]):
        sell_dt = t["sell_date"] or "—"
        status = "持有中" if t["status"] == "holding" else "已平仓"
        sig_dt = t.get("signal_date") or "—"
        fa_dt = t.get("first_above_date") or "—"
        print(
            f"  {t['symbol']:<8s} {sig_dt:<18s} {fa_dt:<18s} {t['buy_date']:<18s} {t['buy_price']:>10.3f}"
            f"  {sell_dt:<18s} {t['sell_price']:>10.3f}"
            f"  {t['return_pct']:>+8.2f}  {t['hold_bars']:>4d}  {status}"
        )

    closed = [t for t in all_trades if t["status"] == "closed"]
    holding = [t for t in all_trades if t["status"] == "holding"]

    print(f"\n  {'─' * 50}")
    if closed:
        returns = [t["return_pct"] for t in closed]
        wins = [r for r in returns if r > 0]
        losses = [r for r in returns if r <= 0]
        avg_ret = sum(returns) / len(returns)
        win_rate = len(wins) / len(returns) * 100
        avg_win = sum(wins) / len(wins) if wins else 0
        avg_loss = sum(losses) / len(losses) if losses else 0
        print(f"  已平仓: {len(closed)} 笔")
        print(f"    平均收益: {avg_ret:+.2f}%")
        print(f"    胜率:     {win_rate:.1f}% ({len(wins)}胜 / {len(losses)}负)")
        print(f"    平均盈利: {avg_win:+.2f}%   平均亏损: {avg_loss:+.2f}%")

    if holding:
        avg_unr = sum(t["return_pct"] for t in holding) / len(holding)
        print(f"  持有中: {len(holding)} 笔, 平均浮盈: {avg_unr:+.2f}%")

    total_return = sum(t["return_pct"] for t in all_trades)
    print(f"  累计收益 (简单加总): {total_return:+.2f}%")
    print()


def _save_results(
    output_path: Path,
    meta: dict,
    all_trades: list[dict],
) -> None:
    """保存回测结果到 JSON。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    closed = [t for t in all_trades if t["status"] == "closed"]
    holding = [t for t in all_trades if t["status"] == "holding"]

    payload: dict = {
        "meta": meta,
        "trades": sorted(all_trades, key=lambda x: x["buy_date"]),
        "summary": {
            "total_trades": len(all_trades),
            "closed_trades": len(closed),
            "holding_trades": len(holding),
            "symbols_traded": sorted(set(t["symbol"] for t in all_trades)),
        },
    }

    if closed:
        returns = [t["return_pct"] for t in closed]
        wins = [r for r in returns if r > 0]
        payload["summary"]["closed_avg_return_pct"] = round(
            sum(returns) / len(returns), 2
        )
        payload["summary"]["win_rate_pct"] = round(
            len(wins) / len(returns) * 100, 1
        )

    tmp = output_path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
    tmp.replace(output_path)
    print(f"  结果已保存: {output_path}\n")


def main():
    ap = argparse.ArgumentParser(description="蓝色通道抄底策略回测 (4h 级别)")
    ap.add_argument("--symbol", nargs="+", metavar="SYM",
                    help="指定股票代码（可多个），如 BABA 或 BABA TSLA AAPL；"
                         "指定后忽略 --top")
    ap.add_argument("--exchange", default=None,
                    help="交易所（仅在 --symbol 时生效，如 NYSE / NASDAQ / HKEX）；"
                         "不指定则自动推断")
    ap.add_argument("--market", default="us", choices=["us", "cn", "hk"],
                    help="市场（默认 us）")
    ap.add_argument("--top", type=int, default=200,
                    help="取前 N 只股票（默认 200）；--symbol 存在时忽略")
    ap.add_argument("--start", default=_BACKTEST_START,
                    help=f"回测起始日期（默认 {_BACKTEST_START}）")
    ap.add_argument("-o", "--output", default=None,
                    help="输出文件路径（默认自动生成）")
    ap.add_argument("--min-hold", type=int, default=MIN_HOLD_BARS,
                    help=f"C6: 首次站上 BB_4H 后至少持续根数（默认 {MIN_HOLD_BARS}）")
    ap.add_argument("--debug", action="store_true",
                    help="开启逐 bar debug 日志（每只股票打印指标值与条件判断）")
    args = ap.parse_args()

    ts_short = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.symbol:
        exchange_lookup = build_exchange_lookup()
        stocks = []
        for sym in args.symbol:
            sym = sym.upper()
            if args.market == "hk":
                sym = sym.lstrip("0") or "0"
            exch = (
                args.exchange.upper()
                if args.exchange
                else resolve_exchange(sym, args.market, exchange_lookup)
            )
            stocks.append({"symbol": sym, "exchange": exch, "name": sym})
        mode_tag = "_".join(s["symbol"] for s in stocks)
        output_path = Path(args.output) if args.output else (
            _RESULTS_DIR / f"backtest_4h_{mode_tag}_{ts_short}.json"
        )
        header_desc = f"单股模式 (4h): {', '.join(s['symbol'] for s in stocks)}"
    else:
        stocks = load_top_stocks(args.market, args.top)
        output_path = Path(args.output) if args.output else (
            _RESULTS_DIR / f"backtest_4h_{args.market}_top{args.top}_{ts_short}.json"
        )
        header_desc = f"{args.market.upper()} Top {args.top} ({len(stocks)} 只) [4h级别]"

    total = len(stocks)

    print(f"\n{'═' * 70}")
    print(f"  策略回测: {header_desc}")
    print(f"  回测区间: {args.start} → 至今")
    print(f"  输出文件: {output_path}")
    print(f"{'═' * 70}\n")

    all_trades: list[dict] = []

    for i, s in enumerate(stocks, 1):
        sym, exch, name = s["symbol"], s["exchange"], s["name"]
        t0 = time.time()
        print(f"  [{i}/{total}] {sym:<8s} ({name}) ...", end="", flush=True)

        try:
            if args.debug:
                print()
                print(f"  ┌─ DEBUG: {sym} {'─' * 60}")
            trades, skip_counts = backtest_single(
                sym, exch, args.start,
                min_hold_bars=args.min_hold,
                debug=args.debug,
            )
            for t in trades:
                t["symbol"] = sym
                t["name"] = name
            all_trades.extend(trades)
            elapsed = time.time() - t0

            if trades:
                signs = " ".join(
                    f"{t['buy_date']}→{t['sell_date'] or '持有'}"
                    for t in trades
                )
                result_line = f"  ✅ {len(trades)} 笔 [{signs}]  ({elapsed:.1f}s)"
            elif skip_counts:
                top_skips = sorted(
                    skip_counts.items(), key=lambda x: x[1], reverse=True
                )[:2]
                reasons = "  |  ".join(
                    f"{SKIP_LABELS.get(k, k)}({v}根)"
                    for k, v in top_skips
                )
                result_line = f"  —  无买点: {reasons}  ({elapsed:.1f}s)"
            else:
                result_line = f"  —  无买点: 回测区间内无有效数据  ({elapsed:.1f}s)"

            if args.debug:
                print(f"  └─ 结果: {result_line.strip()}")
            else:
                print(result_line)
        except Exception as e:
            elapsed = time.time() - t0
            err_line = f"  ⚠ {e}  ({elapsed:.1f}s)"
            if args.debug:
                print(f"  └─ 结果: {err_line.strip()}")
            else:
                print(err_line)

    _print_summary(all_trades)

    meta = {
        "strategy": "4h_blue_channel",
        "market": args.market,
        "top_n": args.top,
        "total_stocks_scanned": total,
        "backtest_start": args.start,
        "max_distance_pct": 5.0,
        "min_hold_bars": args.min_hold,
        "run_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    _save_results(output_path, meta, all_trades)


if __name__ == "__main__":
    main()
