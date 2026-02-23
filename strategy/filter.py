"""
选股过滤器 — 3 个独立条件判断。

条件:
  1. NX 通道多头排列
       周线: 蓝色上边缘(BLUE_TOP) > 黄色上边缘(YELLOW_TOP)
       日线: 蓝色上边缘(BLUE_TOP) > 黄色下边缘(YELLOW_BOTTOM)
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
    """检查 NX 通道多头排列：
      - 周线: 蓝色上边缘 > 黄色上边缘
      - 日线: 蓝色上边缘 > 黄色下边缘

    Returns
    -------
    dict
        passed (bool), weekly_*, daily_*, detail (str)
    """
    # 周线：BLUE_TOP > YELLOW_TOP
    df_w = fetch_tv(ticker, "1w", exchange)
    nx_w = compute_nx(df_w)
    w = nx_w.iloc[-1]
    weekly_bt = round(float(w["BLUE_TOP"]), 3)
    weekly_yt = round(float(w["YELLOW_TOP"]), 3)
    weekly_passed = weekly_bt > weekly_yt

    # 日线：BLUE_TOP > YELLOW_BOTTOM
    df_d = fetch_tv(ticker, "1d", exchange)
    nx_d = compute_nx(df_d)
    d = nx_d.iloc[-1]
    daily_bt = round(float(d["BLUE_TOP"]), 3)
    daily_yb = round(float(d["YELLOW_BOTTOM"]), 3)
    daily_passed = daily_bt > daily_yb

    passed = weekly_passed and daily_passed

    w_op = ">" if weekly_passed else "≤"
    d_op = ">" if daily_passed else "≤"
    detail = (
        f"周线蓝上 {weekly_bt:.3f} {w_op} 黄上 {weekly_yt:.3f}; "
        f"日线蓝上 {daily_bt:.3f} {d_op} 黄下 {daily_yb:.3f}"
    )

    return {
        "passed": passed,
        "weekly_blue_top": weekly_bt,
        "weekly_yellow_top": weekly_yt,
        "weekly_passed": weekly_passed,
        "daily_blue_top": daily_bt,
        "daily_yellow_bottom": daily_yb,
        "daily_passed": daily_passed,
        "detail": detail,
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

_RECOVERY_WINDOW = 10  # 跌破后允许重新站上的最大 K 线根数


def _check_cd_hold_single_tf(
    ticker: str,
    exchange: str,
    tf: str,
    lookback_bars: int,
    min_hold_bars: int,
) -> dict | None:
    """单个级别检查：CD 信号 + 站稳蓝色梯子。通过或有信号返回 dict，无信号返回 None。

    允许至多一次跌破蓝色梯子：跌破后须在 _RECOVERY_WINDOW 根 K 线内重新站上，
    此后不可再跌破。持续站上时间从「锚点」（首次站上 或 恢复后重站上）起算。
    """
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
    rows = list(after_signal.itertuples())  # 顺序列表，方便按整数索引访问

    latest = after_signal.iloc[-1]
    current_close = latest["Close"]
    current_bb = latest["BLUE_BOTTOM"]
    still_above = current_close > current_bb

    # 找第一次站上蓝色梯子的位置
    first_above_pos = None
    for pos, row in enumerate(rows):
        if not pd.isna(row.BLUE_BOTTOM) and row.Close >= row.BLUE_BOTTOM:
            first_above_pos = pos
            break

    if first_above_pos is None:
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

    # 找首次站上后第一次跌破的位置
    break_pos = None
    for pos in range(first_above_pos, len(rows)):
        row = rows[pos]
        if not pd.isna(row.BLUE_BOTTOM) and row.Close < row.BLUE_BOTTOM:
            break_pos = pos
            break

    anchor_pos = first_above_pos
    recovered = False
    second_break = False

    if break_pos is not None:
        # 在 _RECOVERY_WINDOW 根 K 线内寻找恢复
        recovery_pos = None
        for pos in range(break_pos + 1, min(break_pos + 1 + _RECOVERY_WINDOW, len(rows))):
            row = rows[pos]
            if not pd.isna(row.BLUE_BOTTOM) and row.Close >= row.BLUE_BOTTOM:
                recovery_pos = pos
                break

        if recovery_pos is None:
            # 跌破后未在窗口内恢复
            bars_above = first_above_pos  # 仅供参考
            return {
                "passed": False,
                "tf": tf,
                "signals": signal_dates,
                "latest_signal": signal_dates[-1],
                "bars_above": bars_above,
                "min_hold_bars": min_hold_bars,
                "still_above": still_above,
                "current_close": round(current_close, 3),
                "current_blue_bottom": round(current_bb, 3),
                "detail": (
                    f"{tf} CD {signal_dates[-1]}, "
                    f"跌破蓝色梯子后 {_RECOVERY_WINDOW} 根K内未恢复"
                    f", 当前 {current_close:.3f} "
                    f"{'>' if still_above else '≤'} 蓝下边缘 {current_bb:.3f}"
                ),
            }

        anchor_pos = recovery_pos
        recovered = True

        # 恢复后不可再跌破
        second_break = any(
            (not pd.isna(rows[pos].BLUE_BOTTOM))
            and rows[pos].Close < rows[pos].BLUE_BOTTOM
            for pos in range(recovery_pos, len(rows))
        )

    bars_above = len(rows) - anchor_pos
    held_enough = bars_above >= min_hold_bars
    passed = (not second_break) and held_enough

    detail_parts = [f"{tf} CD {signal_dates[-1]}"]
    if recovered:
        detail_parts.append(f"跌破后恢复")
    detail_parts.append(
        f"站上蓝色梯子后 {bars_above} 根K"
        + (f" ≥ {min_hold_bars}" if held_enough else f" < {min_hold_bars}")
    )
    if second_break:
        detail_parts.append("恢复后再次跌破")
    detail_parts.append(
        f"当前 {current_close:.3f} "
        f"{'>' if still_above else '≤'} 蓝下边缘 {current_bb:.3f}"
    )

    return {
        "passed": passed,
        "tf": tf,
        "signals": signal_dates,
        "latest_signal": signal_dates[-1],
        "bars_above": bars_above,
        "min_hold_bars": min_hold_bars,
        "recovered": recovered,
        "second_break": second_break,
        "still_above": still_above,
        "current_close": round(current_close, 3),
        "current_blue_bottom": round(current_bb, 3),
        "detail": ", ".join(detail_parts),
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
        ("4h", lookback_daily_bars * 2),
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
        ("1. NX 多头排列(周蓝上>黄上, 日蓝上>黄下)",
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
