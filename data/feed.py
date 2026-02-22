"""
TradingView 数据源 — K 线获取，含交易所自动探测与本地缓存。

直接使用项目内 tvdatafeed/ 目录的源码，无需安装 tradingview-datafeed 包。
默认使用「调整股息」后的复权数据（adjustment="dividends"）。

公共接口
--------
TvDatafeed, Interval   — 原始类，供需要细粒度控制的模块使用
TV_NBARS               — 默认拉取 K 线根数（5000）
INTERVAL_MAP           — 字符串周期 → Interval 枚举映射
INTERVAL_LABEL         — 字符串周期 → 中文标签映射
fetch_tv(...)          — 获取 K 线，含交易所自动探测与重试
"""

import json
import logging
import sys
import time
from pathlib import Path

# 将本地 tvdatafeed 源码目录注入搜索路径（幂等）
_TVDF_SRC = str(Path(__file__).parent.parent / "tvdatafeed")
if _TVDF_SRC not in sys.path:
    sys.path.insert(0, _TVDF_SRC)

from tvDatafeed import TvDatafeed, Interval  # noqa: E402

import pandas as pd

logger = logging.getLogger(__name__)

__all__ = [
    "TvDatafeed",
    "Interval",
    "TV_NBARS",
    "INTERVAL_MAP",
    "INTERVAL_LABEL",
    "fetch_tv",
]

TV_NBARS = 5000

INTERVAL_MAP: dict[str, Interval] = {
    "1d": Interval.in_daily,
    "4h": Interval.in_4_hour,
    "1w": Interval.in_weekly,
}

INTERVAL_LABEL: dict[str, str] = {
    "1d": "日线",
    "4h": "4h",
    "1w": "周线",
}

# ── 交易所常量 ────────────────────────────────────────────────────────
_US_EXCHANGES = ["NASDAQ", "NYSE", "AMEX"]
_CN_EXCHANGES = ["SSE", "SZSE"]

_EXCHANGE_CACHE_FILE = Path(__file__).parent.parent / "stocks" / "exchange_cache.json"

_MAX_RETRIES = 3
_RETRY_DELAY = 2  # 秒

# ── 共享连接 ──────────────────────────────────────────────────────────
_shared_tv: TvDatafeed | None = None


def _get_tv(force_new: bool = False) -> TvDatafeed:
    """获取（或重建）共享的 TvDatafeed 连接实例。"""
    global _shared_tv
    if _shared_tv is None or force_new:
        _shared_tv = TvDatafeed()
    return _shared_tv


# ── 交易所缓存 ────────────────────────────────────────────────────────
_exchange_cache: dict[str, str] = {}
_exchange_cache_dirty = False


def _load_exchange_cache() -> None:
    global _exchange_cache
    if _EXCHANGE_CACHE_FILE.exists():
        try:
            with open(_EXCHANGE_CACHE_FILE, encoding="utf-8") as f:
                _exchange_cache = json.load(f)
        except (json.JSONDecodeError, OSError):
            _exchange_cache = {}


def _save_exchange_cache() -> None:
    global _exchange_cache_dirty
    if not _exchange_cache_dirty:
        return
    _EXCHANGE_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(_EXCHANGE_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(_exchange_cache, f, indent=2)
    _exchange_cache_dirty = False


def _cache_exchange(ticker: str, exchange: str) -> None:
    global _exchange_cache_dirty
    if _exchange_cache.get(ticker) != exchange:
        _exchange_cache[ticker] = exchange
        _exchange_cache_dirty = True
        _save_exchange_cache()


_load_exchange_cache()


# ── 内部获取函数 ──────────────────────────────────────────────────────

def _raw_fetch(
    ticker: str,
    exchange: str,
    interval: str,
    adjustment: str = "dividends",
    force_new_conn: bool = False,
) -> pd.DataFrame | None:
    """单次尝试从指定交易所获取数据，失败返回 None。"""
    try:
        tv = _get_tv(force_new=force_new_conn)
        df = tv.get_hist(
            symbol=ticker,
            exchange=exchange,
            interval=INTERVAL_MAP[interval],
            n_bars=TV_NBARS,
            adjustment=adjustment,
        )
        if df is not None and not df.empty:
            return df.rename(columns={
                "open": "Open", "high": "High", "low": "Low",
                "close": "Close", "volume": "Volume",
            })
    except Exception:
        pass
    return None


def _fetch_with_retry(
    ticker: str,
    exchange: str,
    interval: str,
    adjustment: str = "dividends",
) -> pd.DataFrame | None:
    """从指定交易所获取数据，含重试逻辑。"""
    for attempt in range(1, _MAX_RETRIES + 1):
        df = _raw_fetch(ticker, exchange, interval, adjustment,
                        force_new_conn=(attempt > 1))
        if df is not None:
            return df
        if attempt < _MAX_RETRIES:
            time.sleep(_RETRY_DELAY)
    return None


# ── 公共接口 ──────────────────────────────────────────────────────────

def fetch_tv(
    ticker: str,
    interval: str,
    exchange: str = "NASDAQ",
    adjustment: str = "dividends",
) -> pd.DataFrame:
    """从 TradingView 获取 K 线数据。

    对美股（NASDAQ/NYSE/AMEX）和 A 股（SSE/SZSE）自动探测正确交易所
    并将结果缓存到本地，避免重复探测。

    Parameters
    ----------
    ticker : str
        股票代码（如 "AAPL"、"600519"）。
    interval : str
        周期字符串，可选 "1d" / "4h" / "1w"。
    exchange : str
        首选交易所，默认 "NASDAQ"。对美股/A 股会自动尝试同市场其他交易所。
    adjustment : str
        复权方式：
          "dividends" — 分股 + 股息双重调整（默认）
          "splits"    — 仅分股调整
          ""          — 不复权

    Returns
    -------
    pd.DataFrame
        含 Open / High / Low / Close / Volume 列，以 datetime 为索引。

    Raises
    ------
    ValueError
        所有交易所尝试均失败时抛出。
    """
    if interval not in INTERVAL_MAP:
        raise ValueError(
            f"不支持的周期: {interval!r}，可选: {', '.join(INTERVAL_MAP)}"
        )

    # 优先使用缓存的交易所
    cached = _exchange_cache.get(ticker)
    if cached:
        df = _fetch_with_retry(ticker, cached, interval, adjustment)
        if df is not None:
            return df

    # 构建探测顺序：首选交易所 + 同市场其他交易所
    if exchange in _US_EXCHANGES:
        probe_list = [exchange] + [e for e in _US_EXCHANGES if e != exchange]
    elif exchange in _CN_EXCHANGES:
        probe_list = [exchange] + [e for e in _CN_EXCHANGES if e != exchange]
    else:
        probe_list = [exchange]

    for exch in probe_list:
        df = _raw_fetch(ticker, exch, interval, adjustment)
        if df is not None:
            _cache_exchange(ticker, exch)
            if exch != exchange:
                logger.info("%s 交易所: %s → %s", ticker, exchange, exch)
            return df

    # 探测全部失败，用默认交易所做完整重试（可能是瞬时网络问题）
    df = _fetch_with_retry(ticker, exchange, interval, adjustment)
    if df is not None:
        _cache_exchange(ticker, exchange)
        return df

    raise ValueError(
        f"未获取到 {ticker} ({interval}) 的行情数据 "
        f"（尝试交易所: {probe_list}）"
    )
