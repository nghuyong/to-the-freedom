"""
CD 指标 — Python 实现（MACD 背离抄底/卖出信号）

基于 MACD 底背离和顶背离检测:
  - DXDX  (抄底): MACD 底背离确认信号
  - DBJGXC(卖出): MACD 顶背离确认信号

公式参数:
  S = 12  (DIFF 短期 EMA)
  P = 26  (DIFF 长期 EMA)
  M = 9   (DEA 信号线 EMA)

依赖: pip install ta
"""

import numpy as np
import pandas as pd
from ta.trend import EMAIndicator


# ---------------------------------------------------------------------------
# 辅助函数：对应通达信公式函数
# ---------------------------------------------------------------------------

def _barslast(cond: np.ndarray) -> np.ndarray:
    """BARSLAST: 条件最近一次为 True 距今的 bar 数（当日为 True 则返回 0）。"""
    n = len(cond)
    result = np.full(n, np.nan)
    last_true = -1
    for i in range(n):
        if cond[i]:
            last_true = i
        if last_true >= 0:
            result[i] = i - last_true
    return result


def _ref_fixed(arr: np.ndarray, offset: int) -> np.ndarray:
    """REF(arr, N): 固定偏移引用（N 根 bar 之前的值）。"""
    if offset <= 0:
        return arr.copy()
    result = np.full(len(arr), np.nan)
    if offset < len(arr):
        result[offset:] = arr[:-offset]
    return result


def _ref_dynamic(arr: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """REF(arr, N) 其中 N 为逐 bar 变化的动态偏移。"""
    n = len(arr)
    result = np.full(n, np.nan)
    for i in range(n):
        if not np.isnan(offsets[i]):
            j = i - int(offsets[i])
            if 0 <= j < n:
                result[i] = arr[j]
    return result


def _llv_dynamic(arr: np.ndarray, lookbacks: np.ndarray) -> np.ndarray:
    """LLV(arr, N): 动态窗口最低值。"""
    n = len(arr)
    result = np.full(n, np.nan)
    for i in range(n):
        if not np.isnan(lookbacks[i]):
            lb = int(lookbacks[i])
            start = max(0, i - lb + 1)
            seg = arr[start: i + 1]
            valid = seg[~np.isnan(seg)]
            if len(valid) > 0:
                result[i] = np.min(valid)
    return result


def _hhv_dynamic(arr: np.ndarray, lookbacks: np.ndarray) -> np.ndarray:
    """HHV(arr, N): 动态窗口最高值。"""
    n = len(arr)
    result = np.full(n, np.nan)
    for i in range(n):
        if not np.isnan(lookbacks[i]):
            lb = int(lookbacks[i])
            start = max(0, i - lb + 1)
            seg = arr[start: i + 1]
            valid = seg[~np.isnan(seg)]
            if len(valid) > 0:
                result[i] = np.max(valid)
    return result


def _count_fixed(cond: np.ndarray, window: int) -> np.ndarray:
    """COUNT(cond, N): 最近 N 根 bar 中条件为 True 的次数。"""
    n = len(cond)
    c = cond.astype(float)
    cs = np.cumsum(c)
    result = np.zeros(n)
    for i in range(n):
        start = max(0, i - window + 1)
        result[i] = cs[i] - (cs[start - 1] if start > 0 else 0.0)
    return result


def _to_bool(arr) -> np.ndarray:
    """将 float/bool 数组转为 bool，NaN 视为 False。"""
    a = np.asarray(arr, dtype=float)
    return np.where(np.isnan(a), False, a != 0)


# ---------------------------------------------------------------------------
# 核心计算
# ---------------------------------------------------------------------------

def compute_cd(
    df: pd.DataFrame,
    S: int = 12,
    P: int = 26,
    M: int = 9,
) -> pd.DataFrame:
    """计算 CD 指标（MACD 背离抄底/卖出信号）。

    Parameters
    ----------
    df : pd.DataFrame
        必须包含 'Close' 列的行情数据。
    S, P, M : int
        MACD 参数，默认 12 / 26 / 9。

    Returns
    -------
    pd.DataFrame
        追加了 DIFF, DEA, MACD, DXDX(抄底), DBJGXC(卖出) 等列。
    """
    df = df.copy()
    close = df["Close"].values.astype(float)
    n = len(close)

    # ==================== MACD ====================
    ema_s = EMAIndicator(close=df["Close"], window=S).ema_indicator().values
    ema_p = EMAIndicator(close=df["Close"], window=P).ema_indicator().values
    diff = ema_s - ema_p
    dea = pd.Series(diff).ewm(span=M, adjust=False).mean().values
    macd = (diff - dea) * 2

    df["DIFF"] = diff
    df["DEA"] = dea
    df["MACD_HIST"] = macd

    # ==================== BARSLAST ====================
    # N1: bars since MACD histogram crossed from >=0 to <0
    cond_n1 = np.zeros(n, dtype=bool)
    cond_mm1 = np.zeros(n, dtype=bool)
    for i in range(1, n):
        if not (np.isnan(macd[i - 1]) or np.isnan(macd[i])):
            cond_n1[i] = (macd[i - 1] >= 0) and (macd[i] < 0)
            cond_mm1[i] = (macd[i - 1] <= 0) and (macd[i] > 0)

    n1 = _barslast(cond_n1)   # bars since last death cross of histogram
    mm1 = _barslast(cond_mm1)  # bars since last golden cross of histogram

    # ==================== 底背离变量 ====================
    cc1 = _llv_dynamic(close, n1 + 1)
    cc2 = _ref_dynamic(cc1, mm1 + 1)
    cc3 = _ref_dynamic(cc2, mm1 + 1)

    difl1 = _llv_dynamic(diff, n1 + 1)
    difl2 = _ref_dynamic(difl1, mm1 + 1)
    difl3 = _ref_dynamic(difl2, mm1 + 1)

    # ==================== 顶背离变量 ====================
    ch1 = _hhv_dynamic(close, mm1 + 1)
    ch2 = _ref_dynamic(ch1, n1 + 1)
    ch3 = _ref_dynamic(ch2, n1 + 1)

    difh1 = _hhv_dynamic(diff, mm1 + 1)
    difh2 = _ref_dynamic(difh1, n1 + 1)
    difh3 = _ref_dynamic(difh2, n1 + 1)

    # ==================== 底背离信号 (抄底) ====================
    ref_macd_1 = _ref_fixed(macd, 1)
    ref_diff_1 = _ref_fixed(diff, 1)

    # AAA: 经典两周期底背离 — 价格新低但 DIFF 更高
    aaa = (
        _to_bool(cc1 < cc2)
        & _to_bool(difl1 > difl2)
        & _to_bool(ref_macd_1 < 0)
        & _to_bool(diff < 0)
    )

    # BBB: 三周期底背离 — 跳过中间周期
    bbb = (
        _to_bool(cc1 < cc3)
        & _to_bool(difl1 < difl2)
        & _to_bool(difl1 > difl3)
        & _to_bool(ref_macd_1 < 0)
        & _to_bool(diff < 0)
    )

    ccc = (aaa | bbb) & _to_bool(diff < 0)
    ref_ccc_1 = _to_bool(_ref_fixed(ccc.astype(float), 1))
    lll = (~ref_ccc_1) & ccc  # CCC 首次出现

    # XXX: 背离消失条件
    ref_aaa_1 = _to_bool(_ref_fixed(aaa.astype(float), 1))
    ref_bbb_1 = _to_bool(_ref_fixed(bbb.astype(float), 1))
    xxx = (
        (ref_aaa_1 & _to_bool(difl1 <= difl2) & _to_bool(diff < dea))
        | (ref_bbb_1 & _to_bool(difl1 <= difl3) & _to_bool(diff < dea))
    )

    # JJJ: 背离确认（|DIFF| 开始缩小 ≥1%）
    jjj = ref_ccc_1 & _to_bool(np.abs(ref_diff_1) >= np.abs(diff) * 1.01)

    # DXDX: JJJ 首次出现 → "抄底"
    ref_jjj_1 = _to_bool(_ref_fixed(jjj.astype(float), 1))
    dxdx = (~ref_jjj_1) & jjj

    # BLBL
    blbl = ref_jjj_1 & ccc & _to_bool(np.abs(ref_diff_1) * 1.01 <= np.abs(diff))

    # DJGXX: 补充抄底条件
    ref_jjj_mm1p1 = _to_bool(_ref_dynamic(jjj.astype(float), mm1 + 1))
    ref_jjj_mm1 = _to_bool(_ref_dynamic(jjj.astype(float), mm1))
    ref_lll_1 = _to_bool(_ref_fixed(lll.astype(float), 1))
    count_jjj_24 = _count_fixed(jjj, 24)
    djgxx = (
        (_to_bool(close < cc2) | _to_bool(close < cc1))
        & (ref_jjj_mm1p1 | ref_jjj_mm1)
        & (~ref_lll_1)
        & _to_bool(count_jjj_24 >= 1)
    )

    # DJXX: 去重后的 DJGXX
    ref_djgxx_1 = _ref_fixed(djgxx.astype(float), 1)
    count_ref_djgxx_1_2 = _count_fixed(_to_bool(ref_djgxx_1), 2)
    djxx = (~_to_bool(count_ref_djgxx_1_2 >= 1)) & djgxx

    # DXX: 底背离消失复合信号
    dxx = (xxx | djxx) & (~ccc)

    # ==================== 顶背离信号 (卖出) ====================
    # ZJDBL: 经典两周期顶背离 — 价格新高但 DIFF 更低
    zjdbl = (
        _to_bool(ch1 > ch2)
        & _to_bool(difh1 < difh2)
        & _to_bool(ref_macd_1 > 0)
        & _to_bool(diff > 0)
    )

    # GXDBL: 三周期顶背离
    gxdbl = (
        _to_bool(ch1 > ch3)
        & _to_bool(difh1 > difh2)
        & _to_bool(difh1 < difh3)
        & _to_bool(ref_macd_1 > 0)
        & _to_bool(diff > 0)
    )

    dbbl = (zjdbl | gxdbl) & _to_bool(diff > 0)
    ref_dbbl_1 = _to_bool(_ref_fixed(dbbl.astype(float), 1))
    dbl = (~ref_dbbl_1) & dbbl & _to_bool(diff > dea)  # DBBL 首次出现

    # DBLXS: 顶背离消失条件
    ref_zjdbl_1 = _to_bool(_ref_fixed(zjdbl.astype(float), 1))
    ref_gxdbl_1 = _to_bool(_ref_fixed(gxdbl.astype(float), 1))
    dblxs = (
        (ref_zjdbl_1 & _to_bool(difh1 >= difh2) & _to_bool(diff > dea))
        | (ref_gxdbl_1 & _to_bool(difh1 >= difh3) & _to_bool(diff > dea))
    )

    # DBJG: 顶背离确认（DIFF 开始下降 ≥1%）
    dbjg = ref_dbbl_1 & _to_bool(ref_diff_1 >= diff * 1.01)

    # DBJGXC: DBJG 首次出现 → "卖出"
    ref_dbjg_1 = _to_bool(_ref_fixed(dbjg.astype(float), 1))
    dbjgxc = (~ref_dbjg_1) & dbjg

    # DBJGBL
    dbjgbl = ref_dbjg_1 & dbbl & _to_bool(ref_diff_1 * 1.01 <= diff)

    # ZZZZZ: 补充卖出条件
    ref_dbjg_n1p1 = _to_bool(_ref_dynamic(dbjg.astype(float), n1 + 1))
    ref_dbjg_n1 = _to_bool(_ref_dynamic(dbjg.astype(float), n1))
    ref_dbl_1 = _to_bool(_ref_fixed(dbl.astype(float), 1))
    count_dbjg_23 = _count_fixed(dbjg, 23)
    zzzzz = (
        (_to_bool(close > ch2) | _to_bool(close > ch1))
        & (ref_dbjg_n1p1 | ref_dbjg_n1)
        & (~ref_dbl_1)
        & _to_bool(count_dbjg_23 >= 1)
    )

    # YYYYY: 去重后的 ZZZZZ
    ref_zzzzz_1 = _ref_fixed(zzzzz.astype(float), 1)
    count_ref_zzzzz_1_2 = _count_fixed(_to_bool(ref_zzzzz_1), 2)
    yyyyy = (~_to_bool(count_ref_zzzzz_1_2 >= 1)) & zzzzz

    # WWWWW: 顶背离消失复合信号
    wwwww = (dblxs | yyyyy) & (~dbbl)

    # ==================== 输出 ====================
    df["BUY_SIGNAL"] = dxdx      # 抄底
    df["SELL_SIGNAL"] = dbjgxc   # 卖出
    df["BUY_CANCEL"] = dxx       # 底背离消失
    df["SELL_CANCEL"] = wwwww    # 顶背离消失

    return df
