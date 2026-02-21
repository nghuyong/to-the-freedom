"""
NX 指标 — Python 实现（基于 ta 量化库）

Futu 公式:
  A :EMA(HIGH,24),COLORBLUE; 对应月线
  B :EMA(LOW,23), COLORBLUE;
  A1:EMA(H,89),   COLORYELLOW; 对应季线
  B1:EMA(L,90),   COLORYELLOW;

注意：上下轨 EMA 周期不同！
  - 蓝色上轨: EMA(High, 24)   蓝色下轨: EMA(Low, 23)
  - 黄色上轨: EMA(High, 89)   黄色下轨: EMA(Low, 90)

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

    # 金叉: 蓝色上轨从下方上穿黄色上轨
    prev_below = df["BLUE_TOP"].shift(1) <= df["YELLOW_TOP"].shift(1)
    curr_above = df["BLUE_TOP"] > df["YELLOW_TOP"]
    df["LONG_SIGNAL"] = prev_below & curr_above

    # 死叉: 蓝色上轨从上方下穿黄色上轨
    prev_above = df["BLUE_TOP"].shift(1) >= df["YELLOW_TOP"].shift(1)
    curr_below = df["BLUE_TOP"] < df["YELLOW_TOP"]
    df["SHORT_SIGNAL"] = prev_above & curr_below

    return df
