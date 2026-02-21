"""
股票综合过滤器 — 3 个独立条件判断。

条件:
  1. 日级别/周级别均蓝色在黄色之上(蓝色的下边缘高于黄色下边缘)
  2. 股价高于 200 日均线 (SMA)
  3. 日级别或 4h 级别 CD + 站稳蓝色梯子:
     最近有日级别或者 4h 级别的 CD 抄底信号,
     且对应级别上收盘价站上蓝色梯子下边缘,
     持续至少 3 根 K 线且至今没有跌破。

用法:
    python stock_filter.py HOOD
    python stock_filter.py AAPL NYSE

依赖: pip install ta tvDatafeed requests
"""

import json
import sys
import time
import logging
from pathlib import Path

import pandas as pd
from tvDatafeed import TvDatafeed, Interval

from nx_indicator import compute_nx
from ma_indicator import compute_ma
from cd_indicator import compute_cd

logger = logging.getLogger(__name__)

_TV_NBARS = 5000
_MAX_RETRIES = 3
_RETRY_DELAY = 2
_EXCHANGE_CACHE_FILE = Path(__file__).parent / "stocks" / "exchange_cache.json"


# ═══════════════════════════════════════════════════════════════════════
# 公共: 获取行情数据
# ═══════════════════════════════════════════════════════════════════════

_INTERVAL_MAP = {
    "1d": Interval.in_daily,
    "4h": Interval.in_4_hour,
    "1w": Interval.in_weekly,
}


_shared_tv: TvDatafeed | None = None


def _get_tv(force_new: bool = False) -> TvDatafeed:
    """获取共享的 TvDatafeed 连接实例; force_new=True 强制重建连接。"""
    global _shared_tv
    if _shared_tv is None or force_new:
        _shared_tv = TvDatafeed()
    return _shared_tv


_US_EXCHANGES = ["NASDAQ", "NYSE", "AMEX"]
_CN_EXCHANGES = ["SSE", "SZSE"]

# symbol → 已知正确交易所的缓存, 避免重复探测 (持久化到文件)
_exchange_cache: dict[str, str] = {}
_exchange_cache_dirty = False


def _load_exchange_cache():
    """从文件加载交易所缓存。"""
    global _exchange_cache
    if _EXCHANGE_CACHE_FILE.exists():
        try:
            with open(_EXCHANGE_CACHE_FILE, encoding="utf-8") as f:
                _exchange_cache = json.load(f)
        except (json.JSONDecodeError, OSError):
            _exchange_cache = {}


def _save_exchange_cache():
    """将交易所缓存写入文件。"""
    global _exchange_cache_dirty
    if not _exchange_cache_dirty:
        return
    _EXCHANGE_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(_EXCHANGE_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(_exchange_cache, f, indent=2)
    _exchange_cache_dirty = False


def _cache_exchange(ticker: str, exchange: str):
    """记录 symbol → exchange 映射并标记为需要写入。"""
    global _exchange_cache_dirty
    if _exchange_cache.get(ticker) != exchange:
        _exchange_cache[ticker] = exchange
        _exchange_cache_dirty = True
        _save_exchange_cache()


_load_exchange_cache()


def _try_fetch_once(ticker: str, exchange: str, interval: str) -> pd.DataFrame | None:
    """单次尝试获取数据, 用于交易所探测。"""
    try:
        tv = _get_tv()
        df = tv.get_hist(
            symbol=ticker,
            exchange=exchange,
            interval=_INTERVAL_MAP[interval],
            n_bars=_TV_NBARS,
        )
        if df is not None and not df.empty:
            df = df.rename(columns={
                "open": "Open", "high": "High", "low": "Low",
                "close": "Close", "volume": "Volume",
            })
            return df
    except Exception:
        pass
    return None


def _try_fetch(ticker: str, exchange: str, interval: str) -> pd.DataFrame | None:
    """从指定交易所获取数据 (含重试), 失败返回 None。"""
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            tv = _get_tv(force_new=(attempt > 1))
            df = tv.get_hist(
                symbol=ticker,
                exchange=exchange,
                interval=_INTERVAL_MAP[interval],
                n_bars=_TV_NBARS,
            )
            if df is not None and not df.empty:
                df = df.rename(columns={
                    "open": "Open", "high": "High", "low": "Low",
                    "close": "Close", "volume": "Volume",
                })
                return df
        except Exception:
            pass
        if attempt < _MAX_RETRIES:
            time.sleep(_RETRY_DELAY)
    return None


def _fetch_tv(ticker: str, interval: str, exchange: str = "NASDAQ") -> pd.DataFrame:
    """从 TradingView 获取 K 线数据。

    对美股 (NASDAQ/NYSE/AMEX) 自动探测正确交易所并缓存结果。
    探测阶段每个交易所只试 1 次; 找到正确交易所后使用完整重试。
    """
    cached_exch = _exchange_cache.get(ticker)
    if cached_exch:
        df = _try_fetch(ticker, cached_exch, interval)
        if df is not None:
            return df

    # 探测阶段: 每个交易所只试 1 次, 快速定位
    exchanges_to_probe = [exchange]
    if exchange in _US_EXCHANGES:
        exchanges_to_probe = [exchange] + [e for e in _US_EXCHANGES if e != exchange]
    elif exchange in _CN_EXCHANGES:
        exchanges_to_probe = [exchange] + [e for e in _CN_EXCHANGES if e != exchange]

    for exch in exchanges_to_probe:
        df = _try_fetch_once(ticker, exch, interval)
        if df is not None:
            _cache_exchange(ticker, exch)
            if exch != exchange:
                logger.info("%s 交易所: %s → %s", ticker, exchange, exch)
            return df

    # 所有交易所探测失败, 用默认交易所做完整重试 (可能是网络临时问题)
    df = _try_fetch(ticker, exchange, interval)
    if df is not None:
        _cache_exchange(ticker, exchange)
        return df

    raise ValueError(
        f"未获取到 {ticker} ({interval}) 的行情数据 "
        f"(尝试交易所: {exchanges_to_probe})"
    )


# ═══════════════════════════════════════════════════════════════════════
# 条件 1: 日+周级别均蓝色在黄色之上 (蓝色下边缘 > 黄色下边缘)
# ═══════════════════════════════════════════════════════════════════════

def check_daily_weekly_nx(ticker: str, exchange: str = "NASDAQ") -> dict:
    """
    检查日线 + 周线 NX 通道: 两个级别均蓝色下边缘 > 黄色下边缘。

    Returns
    -------
    dict with keys: passed (bool), daily_*, weekly_*, detail (str)
    """
    results = {}
    for tf, label in [("1w", "周线"), ("1d", "日线")]:
        df = _fetch_tv(ticker, tf, exchange)
        nx = compute_nx(df)
        latest = nx.iloc[-1]
        bb = latest["BLUE_BOTTOM"]
        yb = latest["YELLOW_BOTTOM"]
        ok = bb > yb
        results[tf] = {"blue_bottom": round(bb, 3),
                        "yellow_bottom": round(yb, 3),
                        "passed": ok, "label": label}

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


# ═══════════════════════════════════════════════════════════════════════
# 条件 2: 股价高于 200 日均线 (SMA200)
# ═══════════════════════════════════════════════════════════════════════

def check_above_sma200(ticker: str, exchange: str = "NASDAQ") -> dict:
    """
    检查日线收盘价是否高于 SMA200。

    Returns
    -------
    dict with keys: passed (bool), close, sma200, detail (str)
    """
    df = _fetch_tv(ticker, "1d", exchange)
    result = compute_ma(df, periods=[200], ma_type="SMA")
    latest = result.iloc[-1]

    close = latest["Close"]
    sma200 = latest["MA200"]

    if pd.isna(sma200):
        return {"passed": False, "close": close, "sma200": None,
                "detail": "SMA200 数据不足，无法判断"}

    passed = close > sma200
    return {
        "passed": passed,
        "close": round(close, 3),
        "sma200": round(sma200, 3),
        "detail": (
            f"收盘价 {close:.3f} {'>' if passed else '≤'} SMA200 {sma200:.3f}"
        ),
    }


# ═══════════════════════════════════════════════════════════════════════
# 条件 3: 日 或 4h CD → 站稳蓝色梯子
#   在日线或 4h 中, 只要有一个级别满足:
#     1) 最近 N 根 K 线中出现 CD 抄底信号
#     2) 信号后收盘价站上蓝色梯子下边缘 (BLUE_BOTTOM)
#     3) 持续 ≥ min_hold_bars 根 K 线, 且至今没有跌破
# ═══════════════════════════════════════════════════════════════════════

def _check_cd_hold_single_tf(
    ticker: str,
    exchange: str,
    tf: str,
    lookback_bars: int,
    min_hold_bars: int,
) -> dict | None:
    """单个级别检查: CD 信号 + 站稳蓝色梯子。通过或有信号返回 dict, 无信号返回 None。"""
    df = _fetch_tv(ticker, tf, exchange)
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
    """
    条件 3: 日级别或 4h 级别 CD 抄底 + 站稳蓝色梯子。

    依次检查日线、4h, 任一级别满足即通过:
      1. 最近 N 根 K 线中有 CD 抄底信号
      2. 信号后收盘价站上蓝色梯子下边缘
      3. 持续 ≥ min_hold_bars 根 K 线且至今未跌破

    日线优先; 日线通过则不请求 4h 数据。
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


# ═══════════════════════════════════════════════════════════════════════
# 综合过滤
# ═══════════════════════════════════════════════════════════════════════

def run_filter(ticker: str, exchange: str = "NASDAQ",
               lookback_daily_bars: int = 30) -> dict:
    """
    对股票运行全部 3 个条件, 返回每个条件的结果。

    Parameters
    ----------
    ticker : str
        股票代码。
    exchange : str
        交易所, 默认 "NASDAQ"。
    lookback_daily_bars : int
        条件 3 回看的日 K 根数, 默认 30。

    Returns
    -------
    dict: {condition_name: result_dict, ...}
    """
    print(f"\n{'═' * 60}")
    print(f"  开始检查 {ticker} ({exchange})  [CD 回看 {lookback_daily_bars} 根日K]")
    print(f"{'═' * 60}\n")

    results = {}
    checks = [
        ("1. 日+周蓝色通道在黄色之上", lambda: check_daily_weekly_nx(ticker, exchange)),
        ("2. 股价高于 SMA200", lambda: check_above_sma200(ticker, exchange)),
        ("3. CD + 站稳蓝色梯子", lambda: check_cd_breakout(
            ticker, exchange,
            lookback_daily_bars=lookback_daily_bars,
        )),
    ]

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

    # 汇总
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


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    ticker = sys.argv[1] if len(sys.argv) > 1 else "HOOD"
    exchange = sys.argv[2] if len(sys.argv) > 2 else "NASDAQ"
    lookback = int(sys.argv[3]) if len(sys.argv) > 3 else 30
    run_filter(ticker, exchange, lookback_daily_bars=lookback)
