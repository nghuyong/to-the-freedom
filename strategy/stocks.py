"""
股票列表工具 — 从 stocks/ CSV 加载标的，解析交易所映射。

支持市场:
    us  — 美股 (stocks/us.csv)    交易所: NASDAQ / NYSE / AMEX
    cn  — 沪深 (stocks/cn.csv)    交易所: SSE (沪) / SZSE (深)
    hk  — 港股 (stocks/hk.csv)    交易所: HKEX
"""

import csv
import json
from pathlib import Path

import numpy as np

_STOCKS_DIR = Path(__file__).parent.parent / "stocks"
_CONSTITUENTS_FILE = Path(__file__).parent.parent / "index_constituents.json"

_MIN_CAP_BILLION = 10.0

_DEFAULT_EXCHANGE = {
    "us": "NASDAQ",
    "cn": "SSE",
    "hk": "HKEX",
}


class NumpyEncoder(json.JSONEncoder):
    """处理 numpy 类型的 JSON 编码器。"""

    def default(self, obj):
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def build_exchange_lookup() -> dict[str, str]:
    """从 index_constituents.json 构建 symbol → exchange 映射。"""
    if not _CONSTITUENTS_FILE.exists():
        return {}
    with open(_CONSTITUENTS_FILE, encoding="utf-8") as f:
        data = json.load(f)
    lookup: dict[str, str] = {}
    for key, stocks in data.items():
        if key == "updated_at" or not isinstance(stocks, list):
            continue
        for s in stocks:
            sym = s.get("symbol", "")
            exch = s.get("exchange", "")
            if sym and exch:
                lookup[sym] = exch
    return lookup


def resolve_exchange(
    symbol: str,
    market: str,
    exchange_lookup: dict[str, str],
) -> str:
    """根据市场和股票代码确定交易所。"""
    if market == "cn":
        if symbol.startswith("0") or symbol.startswith("3"):
            return "SZSE"
        return "SSE"
    if market == "hk":
        return "HKEX"
    return exchange_lookup.get(symbol, _DEFAULT_EXCHANGE.get(market, "NASDAQ"))


def load_top_stocks(market: str, top_n: int) -> list[dict]:
    """从 stocks/{market}.csv 加载股票，按 CSV 原始顺序（热度）取前 N 只。

    过滤规则:
      - 总市值 ≥ 10B
      - 美股排除 ADR
      - 港股代码去掉前导 0

    Parameters
    ----------
    market : str
        市场代码："us" / "cn" / "hk"。
    top_n : int
        返回前 N 只。

    Returns
    -------
    list[dict]
        每项含 symbol, name, exchange, tv_code, market_cap_b。
    """
    csv_path = _STOCKS_DIR / f"{market}.csv"
    if not csv_path.exists():
        available = [f.stem for f in _STOCKS_DIR.glob("*.csv")]
        raise FileNotFoundError(
            f"未找到 {csv_path}，可选市场: {available}"
        )

    exchange_lookup = build_exchange_lookup()
    stocks: list[dict] = []

    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            symbol = row.get("代码", "").strip()
            name = row.get("名称", "").strip()
            cap_raw = row.get("总市值", "").strip()
            if not symbol or not cap_raw:
                continue
            try:
                cap_b = round(float(cap_raw) / 1e9, 2)
            except ValueError:
                continue
            if cap_b < _MIN_CAP_BILLION:
                continue
            if market == "us" and "ADR" in name:
                continue
            if market == "hk":
                symbol = symbol.lstrip("0") or "0"

            exchange = resolve_exchange(symbol, market, exchange_lookup)
            stocks.append({
                "symbol": symbol,
                "name": name,
                "exchange": exchange,
                "tv_code": f"{exchange}:{symbol}",
                "market_cap_b": cap_b,
            })

    return stocks[:top_n]
