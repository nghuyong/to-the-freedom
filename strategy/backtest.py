"""
策略回测 — 蓝色通道抄底策略。

买入条件（同时满足）:
  C2. 周级别 NX: 蓝色上边缘 > 黄色上边缘
  C3. 日级别 NX: 蓝色上边缘 > 黄色下边缘
  C4. 股价 > 200 日均线（SMA200）
  C5. 近期出现日级别 CD 抄底信号
  C6. 股价连续 ≥ MIN_HOLD_BARS 天收盘高于日级别蓝色下边缘（首次站上后未跌破）
  C7. BB_D ≤ 收盘价 ≤ BB_D × (1 + MAX_DIST_PCT)

买入价格: 收盘价。
卖出条件: 股价收盘跌破日级别蓝色下边缘（BB_D），以收盘价卖出。
"""

import pandas as pd

from data.feed import fetch_tv
from indicators.cd import compute_cd
from indicators.nx import compute_nx
from indicators.ma import compute_ma

MAX_DIST_PCT = 0.05
MIN_HOLD_BARS = 3

SKIP_LABELS: dict[str, str] = {
    "nan_data":          "数据缺失(NaN)",
    "C2_weekly_nx":      "C2: 周BT_W≤YT_W(周NX未满足)",
    "C3_daily_nx":       "C3: 日BT_D≤YB_D(日NX未满足)",
    "C4_sma200":         "C4: 收盘价≤SMA200(低于均线)",
    "C5_no_cd":          "C5: 无CD抄底信号",
    "C6_no_first_above": "C6: CD信号后未站上BB_D",
    "C6_hold_bars":      f"C6: 站上BB_D不足{MIN_HOLD_BARS}天",
    "C6_broke":          "C6: 站上后再次跌破BB_D",
    "C7_dist":           f"C7: 盘中低点距BB_D超过{int(MAX_DIST_PCT * 100)}%",
}


def _flatten_index(df: pd.DataFrame) -> pd.DataFrame:
    """如果 DataFrame 是 MultiIndex，丢弃 symbol 层。"""
    if isinstance(df.index, pd.MultiIndex):
        df = df.droplevel(0)
    return df


def _align_weekly_to_daily(
    daily_index: pd.Index,
    nx_weekly: pd.DataFrame,
) -> pd.DataFrame:
    """将周线 NX 数据按日线日期对齐（forward-fill）。

    将每根周线的 bar 日期统一移到所在周的周一，使当周任意交易日
    都能取到本周（而非上周）的 NX 值。
    """
    weekly_cols = pd.DataFrame({
        "BT_W": nx_weekly["BLUE_TOP"],
        "YT_W": nx_weekly["YELLOW_TOP"],
    })
    weekly_cols.index = pd.DatetimeIndex(
        [d - pd.Timedelta(days=d.dayofweek) for d in weekly_cols.index]
    )
    weekly_cols = weekly_cols[~weekly_cols.index.duplicated(keep="last")]
    combined = daily_index.union(weekly_cols.index).sort_values().drop_duplicates()
    filled = weekly_cols.reindex(combined).ffill()
    return filled.reindex(daily_index)


def backtest_single(
    ticker: str,
    exchange: str,
    start_date: str = "2026-01-01",
    min_hold_bars: int = MIN_HOLD_BARS,
    debug: bool = False,
) -> tuple[list[dict], dict]:
    """对单只股票运行回测，返回 (交易列表, 跳过原因统计)。

    每笔交易包含:
      buy_date, buy_price, sell_date (None=持有中), sell_price,
      return_pct, hold_days, status ("closed"/"holding"),
      signal_date, first_above_date

    跳过原因统计: {原因key: 天数} — 仅统计「未持仓」时每天第一个不满足的条件。
    debug=True 时逐日打印指标值与条件判断详情。
    """
    df_daily = _flatten_index(fetch_tv(ticker, "1d", exchange))
    df_weekly = _flatten_index(fetch_tv(ticker, "1w", exchange))

    nx_d = compute_nx(df_daily)
    ma_d = compute_ma(df_daily, periods=[200], ma_type="SMA")
    cd_d = compute_cd(df_daily)
    nx_w = compute_nx(df_weekly)

    daily = df_daily[["Open", "High", "Low", "Close", "Volume"]].copy()
    daily["BB_D"] = nx_d["BLUE_BOTTOM"]
    daily["BT_D"] = nx_d["BLUE_TOP"]
    daily["YB_D"] = nx_d["YELLOW_BOTTOM"]
    daily["SMA200"] = ma_d["MA200"]
    daily["CD_BUY"] = cd_d["BUY_SIGNAL"]

    weekly_aligned = _align_weekly_to_daily(daily.index, nx_w)
    daily["BT_W"] = weekly_aligned["BT_W"]
    daily["YT_W"] = weekly_aligned["YT_W"]

    start_ts = pd.Timestamp(start_date)
    bt_mask = daily.index >= start_ts
    if not bt_mask.any():
        return [], {}
    bt_start = daily.index.get_loc(daily.index[bt_mask][0])

    trades: list[dict] = []
    skip_counts: dict[str, int] = {}
    position: dict | None = None
    last_sell_pos = -1

    def _skip(reason: str):
        skip_counts[reason] = skip_counts.get(reason, 0) + 1

    def _dfmt(v) -> str:
        return f"{v:8.3f}" if not pd.isna(v) else "     NaN"

    if debug:
        bt_end_str = str(daily.index[-1])[:10]
        bt_start_str = str(daily.index[bt_start])[:10]
        n_bars = len(daily) - bt_start
        print(f"  │  回测区间: {bt_start_str} → {bt_end_str}  ({n_bars} bars)")
        print(
            f"  │  {'日期':<12s} {'收盘':>8s} {'BB_D':>8s} {'BT_D':>8s} "
            f"{'YB_D':>8s} {'BT_W':>8s} {'YT_W':>8s} {'SMA200':>8s} {'CD':>3s}  结果"
        )
        print(f"  │  {'─' * 100}")

    for i in range(bt_start, len(daily)):
        row = daily.iloc[i]
        date_str = str(daily.index[i])[:10]
        close = row["Close"]
        bb_d = row["BB_D"]

        if (pd.isna(bb_d) or pd.isna(row["BT_D"])
                or pd.isna(row["SMA200"])
                or pd.isna(row["BT_W"])
                or pd.isna(row["YT_W"])):
            if position is None:
                _skip("nan_data")
            if debug:
                print(
                    f"  │  {date_str:<12s} {_dfmt(close)} {_dfmt(bb_d)} {_dfmt(row['BT_D'])} "
                    f"{_dfmt(row['YB_D'])} {_dfmt(row['BT_W'])} {_dfmt(row['YT_W'])} "
                    f"{_dfmt(row['SMA200'])} {'★' if row['CD_BUY'] else ' ':>3s}  "
                    f"⚠ 数据缺失(NaN), 跳过"
                )
            continue

        # ── 卖出判断 ──────────────────────────────────────────────────
        if position is not None:
            ret_pct = (close / position["price"] - 1) * 100
            if close < bb_d:
                trades.append({
                    "buy_date": position["date"],
                    "buy_price": round(position["price"], 3),
                    "sell_date": date_str,
                    "sell_price": round(close, 3),
                    "return_pct": round(ret_pct, 2),
                    "hold_days": i - position["pos"],
                    "status": "closed",
                    "signal_date": position.get("signal_date"),
                    "first_above_date": position.get("first_above_date"),
                })
                if debug:
                    print(
                        f"  │  {date_str:<12s} {_dfmt(close)} {_dfmt(bb_d)} {_dfmt(row['BT_D'])} "
                        f"{_dfmt(row['YB_D'])} {_dfmt(row['BT_W'])} {_dfmt(row['YT_W'])} "
                        f"{_dfmt(row['SMA200'])} {'★' if row['CD_BUY'] else ' ':>3s}  "
                        f"🔴 卖出 @ {close:.3f}  收益 {ret_pct:+.2f}%  "
                        f"(持仓 {i - position['pos']} 天, 买入 {position['date']} @ {position['price']:.3f})"
                    )
                last_sell_pos = i
                position = None
            else:
                if debug:
                    print(
                        f"  │  {date_str:<12s} {_dfmt(close)} {_dfmt(bb_d)} {_dfmt(row['BT_D'])} "
                        f"{_dfmt(row['YB_D'])} {_dfmt(row['BT_W'])} {_dfmt(row['YT_W'])} "
                        f"{_dfmt(row['SMA200'])} {'★' if row['CD_BUY'] else ' ':>3s}  "
                        f"  持仓中  浮盈 {ret_pct:+.2f}%  close={close:.3f} > BB_D={bb_d:.3f}"
                    )
            continue

        # ── 买入判断 ──────────────────────────────────────────────────
        cd_flag = "★" if row["CD_BUY"] else " "
        _base = (
            f"  │  {date_str:<12s} {_dfmt(close)} {_dfmt(bb_d)} {_dfmt(row['BT_D'])} "
            f"{_dfmt(row['YB_D'])} {_dfmt(row['BT_W'])} {_dfmt(row['YT_W'])} "
            f"{_dfmt(row['SMA200'])} {cd_flag:>3s}  "
        ) if debug else ""

        # C2: 周级别蓝色上边缘 > 黄色上边缘
        if not (row["BT_W"] > row["YT_W"]):
            _skip("C2_weekly_nx")
            if debug:
                print(_base + f"❌ C2: BT_W={row['BT_W']:.3f} ≤ YT_W={row['YT_W']:.3f}")
            continue

        # C3: 日级别蓝色上边缘 > 黄色下边缘
        if not (row["BT_D"] > row["YB_D"]):
            _skip("C3_daily_nx")
            if debug:
                print(_base + f"❌ C3: BT_D={row['BT_D']:.3f} ≤ YB_D={row['YB_D']:.3f}")
            continue

        # C4: 收盘价 > SMA200
        if close <= row["SMA200"]:
            _skip("C4_sma200")
            if debug:
                print(_base + f"❌ C4: close={close:.3f} ≤ SMA200={row['SMA200']:.3f}")
            continue

        # C5: 找回测起始日之后、上次卖出之后的最近一次 CD 抄底信号
        lb_start = max(bt_start, last_sell_pos + 1)
        signal_pos = None
        for j in range(i, lb_start - 1, -1):
            if daily.iloc[j]["CD_BUY"]:
                signal_pos = j
                break
        if signal_pos is None:
            _skip("C5_no_cd")
            if debug:
                print(_base + "❌ C5: 回测区间内无CD抄底信号")
            continue

        signal_date = str(daily.index[signal_pos])[:10]

        # C6: 找 CD 信号后股价第一次站上蓝色下边缘的位置
        first_above_pos = None
        for j in range(signal_pos, i + 1):
            row_j = daily.iloc[j]
            if not pd.isna(row_j["BB_D"]) and row_j["Close"] >= row_j["BB_D"]:
                first_above_pos = j
                break
        if first_above_pos is None:
            _skip("C6_no_first_above")
            if debug:
                print(_base + f"❌ C6: CD信号({signal_date})后收盘价未站上BB_D")
            continue

        first_above_date = str(daily.index[first_above_pos])[:10]

        hold_bars = i - first_above_pos + 1
        if hold_bars < min_hold_bars:
            _skip("C6_hold_bars")
            if debug:
                print(_base + f"❌ C6: 首站上({first_above_date})至今仅 {hold_bars} 天 < {min_hold_bars} 天")
            continue

        still_holding = all(
            (not pd.isna(daily.iloc[j]["BB_D"]))
            and daily.iloc[j]["Close"] >= daily.iloc[j]["BB_D"]
            for j in range(first_above_pos, i + 1)
        )
        if not still_holding:
            _skip("C6_broke")
            if debug:
                print(_base + f"❌ C6: 首站上({first_above_date})后期间曾跌破BB_D")
            continue

        # C7: BB_D ≤ 收盘价 ≤ BB_D × (1 + MAX_DIST_PCT)，以收盘价买入
        dist = (close - bb_d) / bb_d
        if not (bb_d <= close <= bb_d * (1 + MAX_DIST_PCT)):
            _skip("C7_dist")
            if debug:
                print(
                    _base +
                    f"❌ C7: 收盘价={close:.3f} 距BB_D={dist*100:.2f}%"
                    f"  (BB_D={bb_d:.3f}, 阈值={bb_d * (1 + MAX_DIST_PCT):.3f})"
                )
            continue

        # ── 全部条件满足 → 买入（收盘价）────────────────────────────────
        position = {
            "date": date_str, "price": close, "pos": i,
            "signal_date": signal_date,
            "first_above_date": first_above_date,
        }
        if debug:
            print(
                _base +
                f"✅ 买入 @ {close:.3f}  BB_D={bb_d:.3f}  距离={dist*100:.2f}%  "
                f"CD信号={signal_date}  首站上={first_above_date}  已站上{hold_bars}天"
            )

    # 未平仓 → 以最新收盘价计算浮盈
    if position is not None:
        last_row = daily.iloc[-1]
        trades.append({
            "buy_date": position["date"],
            "buy_price": round(position["price"], 3),
            "sell_date": None,
            "sell_price": round(last_row["Close"], 3),
            "return_pct": round(
                (last_row["Close"] / position["price"] - 1) * 100, 2
            ),
            "hold_days": len(daily) - 1 - position["pos"],
            "status": "holding",
            "signal_date": position.get("signal_date"),
            "first_above_date": position.get("first_above_date"),
        })

    if debug:
        print(f"  │  {'─' * 100}")

    return trades, skip_counts
