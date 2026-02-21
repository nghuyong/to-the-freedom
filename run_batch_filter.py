"""
批量运行股票过滤器 — 从 stocks/ 目录读取 CSV, 按市值取 Top N 并逐只检查。

CSV 按热度排序, 直接取前 N 只; 市值信息从 CSV 读取, 无需额外请求 API。
结果会保存到 filter_results/<market>_top<N>_<timestamp>.json

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

依赖: pip install ta tvDatafeed
"""

import argparse
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from stock_filter import (
    check_daily_weekly_nx,
    check_above_sma200,
    check_cd_breakout,
)

_STOCKS_DIR = Path(__file__).parent / "stocks"
_CONSTITUENTS_FILE = Path(__file__).parent / "index_constituents.json"
_RESULTS_DIR = Path(__file__).parent / "filter_results"

_MIN_CAP_BILLION = 10.0

_DEFAULT_EXCHANGE = {
    "us": "NASDAQ",
    "cn": "SSE",
    "hk": "HKEX",
}


def _resolve_exchange(symbol: str, market: str, exchange_lookup: dict) -> str:
    """根据市场和股票代码确定交易所。"""
    if market == "cn":
        if symbol.startswith("0") or symbol.startswith("3"):
            return "SZSE"
        return "SSE"
    if market == "hk":
        return "HKEX"
    return exchange_lookup.get(symbol, _DEFAULT_EXCHANGE.get(market, "NASDAQ"))


class _NumpyEncoder(json.JSONEncoder):
    """处理 numpy 类型的 JSON 编码器。"""

    def default(self, obj):
        if isinstance(obj, (np.bool_, np.bool)):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _build_exchange_lookup() -> dict[str, str]:
    """从 index_constituents.json 构建 symbol → exchange 映射。"""
    if not _CONSTITUENTS_FILE.exists():
        return {}
    with open(_CONSTITUENTS_FILE, encoding="utf-8") as f:
        data = json.load(f)
    lookup = {}
    for key, stocks in data.items():
        if key == "updated_at" or not isinstance(stocks, list):
            continue
        for s in stocks:
            sym = s.get("symbol", "")
            exch = s.get("exchange", "")
            if sym and exch:
                lookup[sym] = exch
    return lookup


def load_top_stocks(market: str, top_n: int) -> list[dict]:
    """从 stocks/{market}.csv 加载股票, 保持 CSV 原始顺序 (热度) 取前 N 只。"""
    csv_path = _STOCKS_DIR / f"{market}.csv"
    if not csv_path.exists():
        available = [f.stem for f in _STOCKS_DIR.glob("*.csv")]
        raise FileNotFoundError(f"未找到 {csv_path}, 可选市场: {available}")

    exchange_lookup = _build_exchange_lookup()

    stocks = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            symbol = row.get("代码", "").strip()
            name = row.get("名称", "").strip()
            cap_raw = row.get("总市值", "").strip()
            if not symbol or not cap_raw:
                continue
            try:
                cap = float(cap_raw)
            except ValueError:
                continue
            cap_b = round(cap / 1e9, 2)
            if cap_b < _MIN_CAP_BILLION:
                continue

            if market == "us" and "ADR" in name:
                continue

            if market == "hk":
                symbol = symbol.lstrip("0") or "0"

            exchange = _resolve_exchange(symbol, market, exchange_lookup)
            stocks.append({
                "symbol": symbol,
                "name": name,
                "exchange": exchange,
                "tv_code": f"{exchange}:{symbol}",
                "market_cap_b": cap_b,
            })

    return stocks[:top_n]


def _save_json(output_path: Path, payload: dict):
    """原子写入 JSON 文件。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, cls=_NumpyEncoder)
    tmp.replace(output_path)


def main():
    parser = argparse.ArgumentParser(description="批量股票过滤器")
    parser.add_argument("--market", default="us",
                        choices=["us", "cn", "hk"],
                        help="市场: us=美股, cn=沪深, hk=港股 (默认 us)")
    parser.add_argument("--top", type=int, default=200,
                        help="取前 N 只, 按 CSV 原始顺序即热度 (默认 200)")
    parser.add_argument("--lookback", type=int, default=30,
                        help="CD 回看日K根数 (默认 30)")
    parser.add_argument("-o", "--output", default=None,
                        help="输出 JSON 文件路径 (默认自动生成)")
    args = parser.parse_args()

    stocks = load_top_stocks(args.market, args.top)
    lookback_daily = args.lookback
    run_ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    run_ts_short = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = _RESULTS_DIR / f"{args.market}_top{args.top}_{run_ts_short}.json"

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

    condition_labels = [
        "c1_daily_weekly_nx",
        "c2_above_sma200",
        "c3_cd_breakout",
    ]
    display_labels = [
        "1.日+周NX",
        "2.SMA200",
        "3.CD站稳",
    ]

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
            "symbol": sym,
            "exchange": exch,
            "name": name,
            "tv_code": tv_code,
            "market_cap_b": s["market_cap_b"],
            "conditions": {
                "c1_daily_weekly_nx": None,
                "c2_above_sma200": None,
                "c3_cd_breakout": None,
            },
        }

        try:
            r1 = check_daily_weekly_nx(sym, exch)
            stock_result["conditions"]["c1_daily_weekly_nx"] = r1
        except Exception as e:
            stock_result["conditions"]["c1_daily_weekly_nx"] = {
                "passed": False, "detail": f"异常: {e}", "error": True}

        try:
            r2 = check_above_sma200(sym, exch)
            stock_result["conditions"]["c2_above_sma200"] = r2
        except Exception as e:
            stock_result["conditions"]["c2_above_sma200"] = {
                "passed": False, "detail": f"异常: {e}", "error": True}

        try:
            r3 = check_cd_breakout(
                sym, exch,
                lookback_daily_bars=lookback_daily,
            )
            stock_result["conditions"]["c3_cd_breakout"] = r3
        except Exception as e:
            stock_result["conditions"]["c3_cd_breakout"] = {
                "passed": False, "detail": f"异常: {e}", "error": True}

        passed_count = sum(
            1 for c in stock_result["conditions"].values()
            if c and c.get("passed")
        )
        stock_result["passed_count"] = passed_count

        elapsed = time.time() - t0
        stock_result["check_time_s"] = round(elapsed, 1)

        flags = "".join(
            "✅" if (stock_result["conditions"].get(k) or {}).get("passed") else "❌"
            for k in condition_labels
        )
        print(f"  {flags}  {passed_count}/3  ({elapsed:.1f}s)")

        payload["stocks"].append(stock_result)
        _save_json(output_path, payload)

    # ── 汇总 ──────────────────────────────────────────────────────────
    all_results = payload["stocks"]
    all_results.sort(key=lambda r: r.get("passed_count", 0), reverse=True)

    payload["meta"]["status"] = "completed"
    payload["meta"]["completed_at"] = datetime.now(timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )

    summary = {"total": len(all_results)}
    for n in range(4):
        group = [r for r in all_results if r.get("passed_count", 0) == n]
        summary[f"passed_{n}"] = len(group)
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
                    dl for dl, cl in zip(display_labels, condition_labels)
                    if not (r["conditions"].get(cl) or {}).get("passed")
                ]
                print(f"    ⚠️  {r['symbol']:<8s} {r['name']:<30s} "
                      f"缺: {', '.join(missed)}")

    print(f"\n  结果已保存: {output_path}\n")


if __name__ == "__main__":
    main()
