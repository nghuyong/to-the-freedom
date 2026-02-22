"""
MA 均线指标 — 简单移动平均（SMA）和指数移动平均（EMA）。

默认计算周期: 5, 10, 20, 50, 100, 200
输出列名格式: "MA{period}"（SMA）或 "EMA{period}"（EMA）
"""

import pandas as pd
from ta.trend import EMAIndicator, SMAIndicator


def compute_ma(
    df: pd.DataFrame,
    periods: list[int] | None = None,
    ma_type: str = "SMA",
    source: str = "Close",
) -> pd.DataFrame:
    """计算均线指标。

    Parameters
    ----------
    df : pd.DataFrame
        必须包含 source 列（默认 'Close'）的行情数据。
    periods : list[int] | None
        均线周期列表，默认 [5, 10, 20, 50, 100, 200]。
    ma_type : str
        均线类型，"SMA" 或 "EMA"，默认 "SMA"。
    source : str
        计算均线的数据列名，默认 "Close"。

    Returns
    -------
    pd.DataFrame
        追加了均线列的 DataFrame，列名格式为 "MA{period}" 或 "EMA{period}"。
    """
    if periods is None:
        periods = [5, 10, 20, 50, 100, 200]

    ma_type = ma_type.upper()
    if ma_type not in ("SMA", "EMA"):
        raise ValueError(
            f"Unsupported ma_type: {ma_type!r}, expected 'SMA' or 'EMA'"
        )

    df = df.copy()
    prefix = "MA" if ma_type == "SMA" else "EMA"

    for period in sorted(periods):
        col_name = f"{prefix}{period}"
        if ma_type == "SMA":
            df[col_name] = SMAIndicator(
                close=df[source], window=period
            ).sma_indicator()
        else:
            df[col_name] = EMAIndicator(
                close=df[source], window=period
            ).ema_indicator()

    return df
