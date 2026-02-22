"""
选股过滤器 — 3 个独立条件判断。

条件:
  1. 日线 + 周线均蓝色下边缘高于黄色下边缘（NX 通道多头排列）
  2. 股价高于 200 日均线（SMA200）
  3. 日线或 4h 出现 CD 抄底信号，且收盘价站稳蓝色梯子 ≥ 3 根 K 线
"""

import pandas as pd

from data.feed import fetch_tv
from indicators.cd import compute_cd
from indicators.nx import compute_nx
from indicators.ma import compute_ma


# ── 条件 1 ────────────────────────────────────────────────────────────

def check_daily_weekly_nx(ticker: str, exchange: str = "NASDAQ") -> dict:
    """检查日线 + 周线 NX 通道：两个级别均蓝色下边缘 > 黄色下边缘。

    Returns
    -------
    dict
        passed (bool), weekly_*, daily_*, detail (str)
    """
    results = {}
    for tf, label in [("1w", "周线"), ("1d", "日线")]:
        df = fetch_tv(ticker, tf, exchange)
        nx = compute_nx(df)
        latest = nx.iloc[-1]
        bb = latest["BLUE_BOTTOM"]
        yb = latest["YELLOW_BOTTOM"]
        ok = bb > yb
        results[tf] = {
            "blue_bottom": round(bb, 3),
            "yellow_bottom": round(yb, 3),
            "passed": ok,
            "label": label,
        }

    weekly = results["1w"]
    daily = results["1d"]
    passed = weekly["passed"] and daily["passed"]

    def _fmt(r):
        op = ">" if r["passed"] else "≤"
        return f"{r['label']}蓝下 {r['blue_bottom']:.3f} {op} 黄下 {r['yellow_bottom']:.3f}"

    return {
        "passed": passed,
        "weekly_blue_bottom": weekly["blue_bottom"],
        "weekly_yellow_bottom": weekly["yellow_bottom"],
        "weekly_passed": weekly["passed"],
        "daily_blue_bottom": daily["blue_bottom"],
        "daily_yellow_bottom": daily["yellow_bottom"],
        "daily_passed": daily["passed"],
        "detail": f"{_fmt(weekly)}; {_fmt(daily)}",
    }


# ── 条件 2 ────────────────────────────────────────────────────────────

def check_above_sma200(ticker: str, exchange: str = "NASDAQ") -> dict:
    """检查日线收盘价是否高于 SMA200。

    Returns
    -------
    dict
        passed (bool), close, sma200, detail (str)
    """
    df = fetch_tv(ticker, "1d", exchange)
    result = compute_ma(df, periods=[200], ma_type="SMA")
    latest = result.iloc[-1]

    close = latest["Close"]
    sma200 = latest["MA200"]

    if pd.isna(sma200):
        return {
            "passed": False, "close": close, "sma200": None,
            "detail": "SMA200 数据不足，无法判断",
        }

    passed = close > sma200
    return {
        "passed": passed,
        "close": round(close, 3),
        "sma200": round(sma200, 3),
        "detail": (
            f"收盘价 {close:.3f} {'>' if passed else '≤'} SMA200 {sma200:.3f}"
        ),
    }


# ── 条件 3 ────────────────────────────────────────────────────────────

def _check_cd_hold_single_tf(
    ticker: str,
    exchange: str,
    tf: str,
    lookback_bars: int,
    min_hold_bars: int,
) -> dict | None:
    """单个级别检查：CD 信号 + 站稳蓝色梯子。通过或有信号返回 dict，无信号返回 None。"""
    df = fetch_tv(ticker, tf, exchange)
    cd = compute_cd(df)

    recent = cd.iloc[-lookback_bars:]
    buys = recent[recent["BUY_SIGNAL"]]
    signal_dates = [str(idx) for idx in buys.index]

    if buys.empty:
        return None

    nx = compute_nx(df)
    merged = cd.copy()
    merged["BLUE_BOTTOM"] = nx["BLUE_BOTTOM"]

    latest_buy_idx = buys.index[-1]
    after_signal = merged.loc[latest_buy_idx:]

    above = after_signal["Close"] > after_signal["BLUE_BOTTOM"]
    first_above_idx = None
    for idx, is_above in above.items():
        if is_above:
            first_above_idx = idx
            break

    latest = after_signal.iloc[-1]
    current_close = latest["Close"]
    current_bb = latest["BLUE_BOTTOM"]
    still_above = current_close > current_bb

    if first_above_idx is None:
        return {
            "passed": False,
            "tf": tf,
            "signals": signal_dates,
            "latest_signal": signal_dates[-1],
            "bars_above": 0,
            "min_hold_bars": min_hold_bars,
            "still_above": still_above,
            "current_close": round(current_close, 3),
            "current_blue_bottom": round(current_bb, 3),
            "detail": f"{tf} CD {signal_dates[-1]}, 尚未站上蓝色梯子",
        }

    since_break = after_signal.loc[first_above_idx:]
    all_above = (since_break["Close"] > since_break["BLUE_BOTTOM"]).all()
    bars_above = len(since_break)
    held_enough = bars_above >= min_hold_bars
    passed = all_above and held_enough

    return {
        "passed": passed,
        "tf": tf,
        "signals": signal_dates,
        "latest_signal": signal_dates[-1],
        "bars_above": bars_above,
        "min_hold_bars": min_hold_bars,
        "held_above": all_above,
        "still_above": still_above,
        "current_close": round(current_close, 3),
        "current_blue_bottom": round(current_bb, 3),
        "detail": (
            f"{tf} CD {signal_dates[-1]}, "
            f"站上蓝色梯子后 {bars_above} 根K"
            + (f" ≥ {min_hold_bars}" if held_enough else f" < {min_hold_bars}")
            + (", 持续保持" if all_above else ", 曾跌破")
            + f", 当前 {current_close:.3f} "
            f"{'>' if still_above else '≤'} "
            f"蓝下边缘 {current_bb:.3f}"
        ),
    }


def check_cd_breakout(
    ticker: str,
    exchange: str = "NASDAQ",
    lookback_daily_bars: int = 30,
    min_hold_bars: int = 3,
) -> dict:
    """条件 3：日线或 4h CD 抄底 + 站稳蓝色梯子。

    依次检查日线、4h，任一级别满足即通过:
      1. 最近 N 根 K 线中有 CD 抄底信号
      2. 信号后收盘价站上蓝色梯子下边缘
      3. 持续 ≥ min_hold_bars 根 K 线且至今未跌破

    日线优先；日线通过则不请求 4h 数据。
    """
    tf_checks = [
        ("1d", lookback_daily_bars),
        ("4h", lookback_daily_bars * 6),
    ]

    tf_results = {}
    for tf, lookback in tf_checks:
        r = _check_cd_hold_single_tf(ticker, exchange, tf, lookback, min_hold_bars)
        if r is not None:
            tf_results[tf] = r
            if r["passed"]:
                return r

    if not tf_results:
        return {
            "passed": False,
            "detail": f"最近日线 {lookback_daily_bars} 根K 和 4h 均无 CD 抄底信号",
        }

    best = max(tf_results.values(), key=lambda r: r.get("bars_above", 0))
    return best


# ── 综合过滤 ──────────────────────────────────────────────────────────

def run_filter(
    ticker: str,
    exchange: str = "NASDAQ",
    lookback_daily_bars: int = 30,
) -> dict:
    """对股票运行全部 3 个条件，返回每个条件的结果。

    Parameters
    ----------
    ticker : str
        股票代码。
    exchange : str
        交易所，默认 "NASDAQ"。
    lookback_daily_bars : int
        条件 3 回看的日 K 根数，默认 30。

    Returns
    -------
    dict
        {condition_name: result_dict, ...}
    """
    print(f"\n{'═' * 60}")
    print(f"  开始检查 {ticker} ({exchange})  [CD 回看 {lookback_daily_bars} 根日K]")
    print(f"{'═' * 60}\n")

    checks = [
        ("1. 日+周蓝色通道在黄色之上",
         lambda: check_daily_weekly_nx(ticker, exchange)),
        ("2. 股价高于 SMA200",
         lambda: check_above_sma200(ticker, exchange)),
        ("3. CD + 站稳蓝色梯子",
         lambda: check_cd_breakout(ticker, exchange,
                                   lookback_daily_bars=lookback_daily_bars)),
    ]

    results = {}
    for name, fn in checks:
        print(f"  检查: {name} ...")
        try:
            result = fn()
            results[name] = result
            status = "✅ 通过" if result["passed"] else "❌ 未通过"
            print(f"    {status} — {result['detail']}\n")
        except Exception as e:
            results[name] = {"passed": False, "detail": f"异常: {e}"}
            print(f"    ⚠️  异常: {e}\n")

    passed_count = sum(1 for r in results.values() if r["passed"])
    total = len(results)

    print(f"{'═' * 60}")
    print(f"  {ticker} 综合结果: {passed_count}/{total} 条件通过")
    print(f"{'═' * 60}")
    for name, result in results.items():
        flag = "✅" if result["passed"] else "❌"
        print(f"  {flag} {name}")
    print()

    return results
