"""
批量过滤工具 — 从 stocks/ 目录读取 CSV，按热度取 Top N 并逐只运行选股条件。

结果保存到 filter_results/<market>_top<N>_<timestamp>.json

支持市场:
    us  — 美股 (stocks/us.csv)    交易所: NASDAQ / NYSE / AMEX
    cn  — 沪深 (stocks/cn.csv)    交易所: SSE (沪) / SZSE (深)
    hk  — 港股 (stocks/hk.csv)    交易所: HKEX

用法:
    python run_batch_filter.py                          # 美股 Top 200
    python run_batch_filter.py --market us --top 200    # 同上
    python run_batch_filter.py --market cn --top 200    # 沪深 Top 200
    python run_batch_filter.py --market hk --top 100    # 港股 Top 100
    python run_batch_filter.py --lookback 30            # 30 根日K 回看
    python run_batch_filter.py -o my_result.json        # 指定输出文件名
"""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from strategy.filter import (
    check_daily_weekly_nx,
    check_above_sma200,
    check_cd_breakout,
)
from strategy.stocks import NumpyEncoder, load_top_stocks

_RESULTS_DIR = Path(__file__).parent / "filter_results"


def _save_json(output_path: Path, payload: dict) -> None:
    """原子写入 JSON 文件。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
    tmp.replace(output_path)


def main():
    parser = argparse.ArgumentParser(description="批量股票过滤器")
    parser.add_argument("--market", default="us", choices=["us", "cn", "hk"],
                        help="市场: us=美股, cn=沪深, hk=港股 (默认 us)")
    parser.add_argument("--top", type=int, default=200,
                        help="取前 N 只，按 CSV 原始顺序（热度）(默认 200)")
    parser.add_argument("--lookback", type=int, default=30,
                        help="CD 回看日K根数 (默认 30)")
    parser.add_argument("-o", "--output", default=None,
                        help="输出 JSON 文件路径 (默认自动生成)")
    args = parser.parse_args()

    stocks = load_top_stocks(args.market, args.top)
    lookback_daily = args.lookback
    run_ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    run_ts_short = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_path = (
        Path(args.output) if args.output
        else _RESULTS_DIR / f"{args.market}_top{args.top}_{run_ts_short}.json"
    )

    payload = {
        "meta": {
            "market": args.market,
            "top_n": args.top,
            "total_stocks": len(stocks),
            "lookback_daily_bars": lookback_daily,
            "run_at": run_ts,
            "status": "running",
        },
        "stocks": [],
    }

    print(f"\n{'═' * 65}")
    print(f"  批量过滤: {args.market} Top {args.top} ({len(stocks)} 只)")
    print(f"  排序: 按热度 (CSV 原始顺序)")
    print(f"  CD 回看: {lookback_daily} 根日K")
    print(f"  结果文件: {output_path}")
    print(f"{'═' * 65}\n")

    condition_keys = ["c1_daily_weekly_nx", "c2_above_sma200", "c3_cd_breakout"]
    display_labels = ["1.日+周NX", "2.SMA200", "3.CD站稳"]
    total = len(stocks)

    for i, s in enumerate(stocks, 1):
        sym = s["symbol"]
        exch = s["exchange"]
        name = s["name"]
        tv_code = s["tv_code"]
        t0 = time.time()

        print(f"  [{i}/{total}] {sym} ({name}, {s['market_cap_b']}B) ...",
              end="", flush=True)

        stock_result = {
            "symbol": sym, "exchange": exch, "name": name,
            "tv_code": tv_code, "market_cap_b": s["market_cap_b"],
            "conditions": {k: None for k in condition_keys},
        }

        checks = [
            ("c1_daily_weekly_nx", lambda: check_daily_weekly_nx(sym, exch)),
            ("c2_above_sma200",    lambda: check_above_sma200(sym, exch)),
            ("c3_cd_breakout",     lambda: check_cd_breakout(
                sym, exch, lookback_daily_bars=lookback_daily)),
        ]
        for key, fn in checks:
            try:
                stock_result["conditions"][key] = fn()
            except Exception as e:
                stock_result["conditions"][key] = {
                    "passed": False, "detail": f"异常: {e}", "error": True}

        passed_count = sum(
            1 for c in stock_result["conditions"].values()
            if c and c.get("passed")
        )
        stock_result["passed_count"] = passed_count
        stock_result["check_time_s"] = round(time.time() - t0, 1)

        flags = "".join(
            "✅" if (stock_result["conditions"].get(k) or {}).get("passed") else "❌"
            for k in condition_keys
        )
        print(f"  {flags}  {passed_count}/3  ({stock_result['check_time_s']}s)")

        payload["stocks"].append(stock_result)
        _save_json(output_path, payload)

    # ── 汇总 ─────────────────────────────────────────────────────────
    all_results = payload["stocks"]
    all_results.sort(key=lambda r: r.get("passed_count", 0), reverse=True)

    payload["meta"]["status"] = "completed"
    payload["meta"]["completed_at"] = datetime.now(timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    summary = {"total": len(all_results)}
    for n in range(4):
        summary[f"passed_{n}"] = sum(
            1 for r in all_results if r.get("passed_count", 0) == n
        )
    summary["full_pass_symbols"] = [
        r["symbol"] for r in all_results if r.get("passed_count", 0) == 3
    ]
    payload["summary"] = summary
    _save_json(output_path, payload)

    print(f"\n{'═' * 65}")
    print("  汇总结果")
    print(f"{'═' * 65}\n")

    full_pass = [r for r in all_results if r.get("passed_count", 0) == 3]
    print(f"  ★ 全部 3 条件通过: {len(full_pass)} 只")
    if full_pass:
        for r in full_pass:
            print(f"    ✅ {r['symbol']:<8s} {r['name']:<30s} 市值 {r['market_cap_b']}B")
    else:
        print("    (无)")

    for target in (2, 1):
        group = [r for r in all_results if r.get("passed_count", 0) == target]
        if group:
            print(f"\n  ★ {target}/3 条件通过: {len(group)} 只")
            for r in group:
                missed = [
                    dl for dl, ck in zip(display_labels, condition_keys)
                    if not (r["conditions"].get(ck) or {}).get("passed")
                ]
                print(f"    ⚠️  {r['symbol']:<8s} {r['name']:<30s} "
                      f"缺: {', '.join(missed)}")

    print(f"\n  结果已保存: {output_path}\n")


if __name__ == "__main__":
    main()
