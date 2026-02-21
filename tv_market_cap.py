#!/usr/bin/env python3
"""
通过 TradingView Screener API 获取股票市值。

支持的交易所: NASDAQ, NYSE, AMEX, SSE, SZSE, HKEX

用法:
    from tv_market_cap import get_market_cap, batch_get_market_caps

    # 单只股票
    cap = get_market_cap("NASDAQ:AAPL")
    print(cap)  # e.g. 3512.34  (单位: 十亿美元)

    # 批量获取
    caps = batch_get_market_caps(["NASDAQ:AAPL", "HKEX:700", "SSE:600519"])
    print(caps)  # {"NASDAQ:AAPL": 3512.34, "HKEX:700": 512.78, ...}

依赖:
    pip install requests
"""

import requests

# ═══════════════════════════════════════════════════════════════════════
# Exchange → TradingView screener region mapping
# ═══════════════════════════════════════════════════════════════════════
_EXCHANGE_TO_REGION = {
    "NASDAQ": "america",
    "NYSE": "america",
    "AMEX": "america",
    "SSE": "china",
    "SZSE": "china",
    "HKEX": "hongkong",
}

# Currency → USD approximate conversion rates
_CURRENCY_TO_USD = {
    "USD": 1.0,
    "HKD": 1 / 7.8,
    "CNY": 1 / 7.25,
}

_SCREENER_URL = "https://scanner.tradingview.com/{region}/scan"


def _parse_tv_code(tv_code: str) -> tuple[str, str]:
    """Parse 'EXCHANGE:SYMBOL' → (exchange, symbol)."""
    parts = tv_code.split(":", 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid tv_code format: {tv_code!r}, expected 'EXCHANGE:SYMBOL'")
    return parts[0], parts[1]


def batch_get_market_caps(tv_codes: list[str]) -> dict[str, float | None]:
    """
    批量获取股票市值 (单位: 十亿美元 / billions USD)。

    Parameters
    ----------
    tv_codes : list[str]
        TradingView 代码列表, 如 ["NASDAQ:AAPL", "HKEX:700", "SSE:600519"]

    Returns
    -------
    dict[str, float | None]
        tv_code → 市值 (十亿美元)。获取失败返回 None。
    """
    # Group by screener region
    region_groups: dict[str, list[str]] = {}
    for code in tv_codes:
        exchange, _ = _parse_tv_code(code)
        region = _EXCHANGE_TO_REGION.get(exchange)
        if region is None:
            raise ValueError(
                f"Unsupported exchange {exchange!r} in {code!r}. "
                f"Supported: {list(_EXCHANGE_TO_REGION.keys())}"
            )
        region_groups.setdefault(region, []).append(code)

    results: dict[str, float | None] = {code: None for code in tv_codes}

    for region, codes in region_groups.items():
        url = _SCREENER_URL.format(region=region)
        payload = {
            "symbols": {"tickers": codes},
            "columns": ["market_cap_basic", "currency"],
        }
        try:
            resp = requests.post(url, json=payload, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"  ⚠ TradingView screener 请求失败 ({region}): {e}")
            continue

        for item in data.get("data", []):
            tv_code = item["s"]
            mcap_local = item["d"][0]  # market_cap_basic (local currency)
            currency = item["d"][1] or "USD"

            if mcap_local is None or mcap_local == 0:
                continue

            usd_rate = _CURRENCY_TO_USD.get(currency, 1.0)
            results[tv_code] = mcap_local * usd_rate / 1e9

    return results


def get_market_cap(tv_code: str) -> float | None:
    """
    获取单只股票市值 (单位: 十亿美元 / billions USD)。

    Parameters
    ----------
    tv_code : str
        TradingView 代码, 如 "NASDAQ:AAPL", "HKEX:700", "SSE:600519"

    Returns
    -------
    float | None
        市值 (十亿美元)。获取失败返回 None。

    Examples
    --------
    >>> get_market_cap("NASDAQ:AAPL")
    3512.34
    >>> get_market_cap("HKEX:700")
    512.78
    """
    result = batch_get_market_caps([tv_code])
    return result.get(tv_code)


# ═══════════════════════════════════════════════════════════════════════
# CLI demo
# ═══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import sys

    demo_codes = sys.argv[1:] or [
        "NASDAQ:AAPL",
        "NASDAQ:GOOGL",
        "NYSE:BRK.A",
        "HKEX:700",
        "HKEX:9988",
        "SSE:600519",
        "SZSE:000858",
    ]

    print("=" * 55)
    print("  TradingView 市值查询")
    print("=" * 55)

    caps = batch_get_market_caps(demo_codes)
    for code in demo_codes:
        cap = caps.get(code)
        if cap is not None:
            print(f"  {code:<20s}  {cap:>10.2f} B USD")
        else:
            print(f"  {code:<20s}  {'N/A':>10s}")

    print()
