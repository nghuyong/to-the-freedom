"""
NX 指标 — 蓝黄双通道 EMA + 金叉死叉信号。

Futu 公式:
  A :EMA(HIGH,24),COLORBLUE;  对应月线（蓝色上轨）
  B :EMA(LOW, 23),COLORBLUE;  对应月线（蓝色下轨）
  A1:EMA(H,  89),COLORYELLOW; 对应季线（黄色上轨）
  B1:EMA(L,  90),COLORYELLOW; 对应季线（黄色下轨）

输出列:
  BLUE_TOP, BLUE_BOTTOM     — 蓝色通道上/下轨
  YELLOW_TOP, YELLOW_BOTTOM — 黄色通道上/下轨
  LONG_SIGNAL               — 金叉（蓝色上轨由下穿上越过黄色上轨）
  SHORT_SIGNAL              — 死叉（蓝色上轨由上穿下跌破黄色上轨）
"""

import pandas as pd
from ta.trend import EMAIndicator


def compute_nx(
    df: pd.DataFrame,
    blue_top_n: int = 24,
    blue_bot_n: int = 23,
    yellow_top_n: int = 89,
    yellow_bot_n: int = 90,
) -> pd.DataFrame:
    """计算 NX 指标：蓝黄双通道 + 金叉死叉信号。

    Parameters
    ----------
    df : pd.DataFrame
        必须包含 'High', 'Low', 'Close' 列的 OHLC 行情数据。
    blue_top_n : int
        蓝色上轨 EMA 周期，默认 24。
    blue_bot_n : int
        蓝色下轨 EMA 周期，默认 23。
    yellow_top_n : int
        黄色上轨 EMA 周期，默认 89。
    yellow_bot_n : int
        黄色下轨 EMA 周期，默认 90。

    Returns
    -------
    pd.DataFrame
        追加了通道轨道和交叉信号列的 DataFrame。
    """
    df = df.copy()

    df["BLUE_TOP"] = EMAIndicator(close=df["High"], window=blue_top_n).ema_indicator()
    df["BLUE_BOTTOM"] = EMAIndicator(close=df["Low"], window=blue_bot_n).ema_indicator()
    df["YELLOW_TOP"] = EMAIndicator(close=df["High"], window=yellow_top_n).ema_indicator()
    df["YELLOW_BOTTOM"] = EMAIndicator(close=df["Low"], window=yellow_bot_n).ema_indicator()

    above = df["BLUE_TOP"] > df["YELLOW_TOP"]
    prev_above = above.shift(1, fill_value=False)
    df["LONG_SIGNAL"] = above & (~prev_above)
    df["SHORT_SIGNAL"] = (~above) & prev_above

    return df
