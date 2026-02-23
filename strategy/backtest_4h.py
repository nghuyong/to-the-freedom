"""
策略回测 — 蓝色通道抄底策略（4h 级别）。

买入条件（同时满足）:
  C2. 周级别 NX: 蓝色上边缘 > 黄色上边缘
  C3. 日级别 NX: 蓝色上边缘 > 黄色下边缘
  C4. 股价 > 200 日均线（SMA200）
  C5. 近期出现4h级别 CD 抄底信号
  C6. CD 信号后股价站上 4h 蓝色下边缘，允许至多一次跌破（须在 10 根K线内重新站上），
      重站上后持续 ≥ MIN_HOLD_BARS 根4h K线且不再跌破
  C7. BB_4H ≤ 收盘价 ≤ BB_4H × (1 + MAX_DIST_PCT)，4h 级别

买入价格: 4h 级别收盘价。
卖出条件: 股价收盘跌破4h级别蓝色下边缘（BB_4H），以收盘价卖出。
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
    "C5_no_cd":          "C5: 无4h CD抄底信号",
    "C6_no_first_above": "C6: CD信号后未站上BB_4H",
    "C6_hold_bars":      f"C6: 站上BB_4H不足{MIN_HOLD_BARS}根4h K线",
    "C6_broke":          "C6: 跌破后10根内未恢复 或 恢复后再次跌破BB_4H",
    "C7_dist":           f"C7: 盘中低点距BB_4H超过{int(MAX_DIST_PCT * 100)}%",
}


def _flatten_index(df: pd.DataFrame) -> pd.DataFrame:
    """如果 DataFrame 是 MultiIndex，丢弃 symbol 层。"""
    if isinstance(df.index, pd.MultiIndex):
        df = df.droplevel(0)
    return df


def _align_weekly_to_4h(
    idx_4h: pd.DatetimeIndex,
    nx_weekly: pd.DataFrame,
) -> pd.DataFrame:
    """将周线 NX 数据对齐到 4h 时间轴（forward-fill）。

    将每根周线的 bar 日期移到所在周的周一，使当周任意 4h bar
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
    combined = idx_4h.union(weekly_cols.index).sort_values().drop_duplicates()
    filled = weekly_cols.reindex(combined).ffill()
    return filled.reindex(idx_4h)


def _align_daily_to_4h(
    idx_4h: pd.DatetimeIndex,
    daily_df: pd.DataFrame,
    cols: list[str],
) -> pd.DataFrame:
    """将日线数据对齐到 4h 时间轴（forward-fill）。

    每根日线的值在当天所有 4h bar 上生效，直到下一根日线更新为止。
    """
    sub = daily_df[cols].copy()
    combined = idx_4h.union(sub.index).sort_values().drop_duplicates()
    filled = sub.reindex(combined).ffill()
    return filled.reindex(idx_4h)


def backtest_single(
    ticker: str,
    exchange: str,
    start_date: str = "2026-01-01",
    min_hold_bars: int = MIN_HOLD_BARS,
    debug: bool = False,
) -> tuple[list[dict], dict]:
    """对单只股票运行 4h 级别回测，返回 (交易列表, 跳过原因统计)。

    每笔交易包含:
      buy_date, buy_price, sell_date (None=持有中), sell_price,
      return_pct, hold_bars, status ("closed"/"holding"),
      signal_date, first_above_date

    跳过原因统计: {原因key: 根数} — 仅统计「未持仓」时每根 bar 第一个不满足的条件。
    debug=True 时逐 bar 打印指标值与条件判断详情。
    """
    df_4h = _flatten_index(fetch_tv(ticker, "4h", exchange))
    df_daily = _flatten_index(fetch_tv(ticker, "1d", exchange))
    df_weekly = _flatten_index(fetch_tv(ticker, "1w", exchange))

    nx_4h = compute_nx(df_4h)
    cd_4h = compute_cd(df_4h)
    nx_d = compute_nx(df_daily)
    ma_d = compute_ma(df_daily, periods=[200], ma_type="SMA")
    nx_w = compute_nx(df_weekly)

    bars = df_4h[["Open", "High", "Low", "Close", "Volume"]].copy()
    bars["BB_4H"] = nx_4h["BLUE_BOTTOM"]
    bars["BT_4H"] = nx_4h["BLUE_TOP"]
    bars["CD_BUY"] = cd_4h["BUY_SIGNAL"]

    # 日线指标对齐到 4h 轴
    daily_aligned = _align_daily_to_4h(
        bars.index,
        pd.DataFrame({"BT_D": nx_d["BLUE_TOP"], "YB_D": nx_d["YELLOW_BOTTOM"], "SMA200": ma_d["MA200"]}),
        ["BT_D", "YB_D", "SMA200"],
    )
    bars["BT_D"] = daily_aligned["BT_D"]
    bars["YB_D"] = daily_aligned["YB_D"]
    bars["SMA200"] = daily_aligned["SMA200"]

    # 周线 NX 对齐到 4h 轴
    weekly_aligned = _align_weekly_to_4h(bars.index, nx_w)
    bars["BT_W"] = weekly_aligned["BT_W"]
    bars["YT_W"] = weekly_aligned["YT_W"]

    start_ts = pd.Timestamp(start_date)
    bt_mask = bars.index >= start_ts
    if not bt_mask.any():
        return [], {}
    bt_start = bars.index.get_loc(bars.index[bt_mask][0])

    trades: list[dict] = []
    skip_counts: dict[str, int] = {}
    position: dict | None = None
    last_sell_pos = -1

    def _skip(reason: str):
        skip_counts[reason] = skip_counts.get(reason, 0) + 1

    def _dfmt(v) -> str:
        return f"{v:8.3f}" if not pd.isna(v) else "     NaN"

    if debug:
        bt_end_str = str(bars.index[-1])[:16]
        bt_start_str = str(bars.index[bt_start])[:16]
        n_bars = len(bars) - bt_start
        print(f"  │  回测区间: {bt_start_str} → {bt_end_str}  ({n_bars} bars/4h)")
        print(
            f"  │  {'时间':<18s} {'收盘':>8s} {'BB_4H':>8s} {'BT_4H':>8s} "
            f"{'BT_D':>8s} {'YB_D':>8s} {'BT_W':>8s} {'YT_W':>8s} {'SMA200':>8s} {'CD':>3s}  结果"
        )
        print(f"  │  {'─' * 110}")

    for i in range(bt_start, len(bars)):
        row = bars.iloc[i]
        date_str = str(bars.index[i])[:16]
        close = row["Close"]
        bb_4h = row["BB_4H"]

        if (pd.isna(bb_4h) or pd.isna(row["BT_4H"])
                or pd.isna(row["SMA200"])
                or pd.isna(row["BT_W"])
                or pd.isna(row["YT_W"])
                or pd.isna(row["BT_D"])
                or pd.isna(row["YB_D"])):
            if position is None:
                _skip("nan_data")
            if debug:
                print(
                    f"  │  {date_str:<18s} {_dfmt(close)} {_dfmt(bb_4h)} {_dfmt(row['BT_4H'])} "
                    f"{_dfmt(row['BT_D'])} {_dfmt(row['YB_D'])} {_dfmt(row['BT_W'])} {_dfmt(row['YT_W'])} "
                    f"{_dfmt(row['SMA200'])} {'★' if row['CD_BUY'] else ' ':>3s}  "
                    f"⚠ 数据缺失(NaN), 跳过"
                )
            continue

        # ── 卖出判断 ──────────────────────────────────────────────────
        if position is not None:
            ret_pct = (close / position["price"] - 1) * 100
            if close < bb_4h:
                trades.append({
                    "buy_date": position["date"],
                    "buy_price": round(position["price"], 3),
                    "sell_date": date_str,
                    "sell_price": round(close, 3),
                    "return_pct": round(ret_pct, 2),
                    "hold_bars": i - position["pos"],
                    "status": "closed",
                    "signal_date": position.get("signal_date"),
                    "first_above_date": position.get("first_above_date"),
                })
                if debug:
                    print(
                        f"  │  {date_str:<18s} {_dfmt(close)} {_dfmt(bb_4h)} {_dfmt(row['BT_4H'])} "
                        f"{_dfmt(row['BT_D'])} {_dfmt(row['YB_D'])} {_dfmt(row['BT_W'])} {_dfmt(row['YT_W'])} "
                        f"{_dfmt(row['SMA200'])} {'★' if row['CD_BUY'] else ' ':>3s}  "
                        f"🔴 卖出 @ {close:.3f}  收益 {ret_pct:+.2f}%  "
                        f"(持 {i - position['pos']} 根, 买入 {position['date']} @ {position['price']:.3f})"
                    )
                last_sell_pos = i
                position = None
            else:
                if debug:
                    print(
                        f"  │  {date_str:<18s} {_dfmt(close)} {_dfmt(bb_4h)} {_dfmt(row['BT_4H'])} "
                        f"{_dfmt(row['BT_D'])} {_dfmt(row['YB_D'])} {_dfmt(row['BT_W'])} {_dfmt(row['YT_W'])} "
                        f"{_dfmt(row['SMA200'])} {'★' if row['CD_BUY'] else ' ':>3s}  "
                        f"  持仓中  浮盈 {ret_pct:+.2f}%  close={close:.3f} > BB_4H={bb_4h:.3f}"
                    )
            continue

        # ── 买入判断 ──────────────────────────────────────────────────
        cd_flag = "★" if row["CD_BUY"] else " "
        _base = (
            f"  │  {date_str:<18s} {_dfmt(close)} {_dfmt(bb_4h)} {_dfmt(row['BT_4H'])} "
            f"{_dfmt(row['BT_D'])} {_dfmt(row['YB_D'])} {_dfmt(row['BT_W'])} {_dfmt(row['YT_W'])} "
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

        # C5: 找回测起始 bar 之后、上次卖出之后的最近一次 4h CD 抄底信号
        lb_start = max(bt_start, last_sell_pos + 1)
        signal_pos = None
        for j in range(i, lb_start - 1, -1):
            if bars.iloc[j]["CD_BUY"]:
                signal_pos = j
                break
        if signal_pos is None:
            _skip("C5_no_cd")
            if debug:
                print(_base + "❌ C5: 回测区间内无4h CD抄底信号")
            continue

        signal_date = str(bars.index[signal_pos])[:16]

        # C6: 找 CD 信号后股价第一次站上 4h 蓝色下边缘的位置
        first_above_pos = None
        for j in range(signal_pos, i + 1):
            row_j = bars.iloc[j]
            if not pd.isna(row_j["BB_4H"]) and row_j["Close"] >= row_j["BB_4H"]:
                first_above_pos = j
                break
        if first_above_pos is None:
            _skip("C6_no_first_above")
            if debug:
                print(_base + f"❌ C6: CD信号({signal_date})后收盘价未站上BB_4H")
            continue

        first_above_date = str(bars.index[first_above_pos])[:16]

        # 允许至多一次跌破：跌破后须在 10 根K线内重新站上，此后不可再跌破
        # anchor_pos 是最终"持续站上"的起始位置（首站上 或 恢复后重站上）
        anchor_pos = first_above_pos
        break_pos = None
        for j in range(first_above_pos, i + 1):
            row_j = bars.iloc[j]
            if not pd.isna(row_j["BB_4H"]) and row_j["Close"] < row_j["BB_4H"]:
                break_pos = j
                break

        if break_pos is not None:
            # 在 10 根 K 线内寻找恢复（重新站上）
            recovery_pos = None
            for j in range(break_pos + 1, min(break_pos + 11, i + 1)):
                row_j = bars.iloc[j]
                if not pd.isna(row_j["BB_4H"]) and row_j["Close"] >= row_j["BB_4H"]:
                    recovery_pos = j
                    break
            if recovery_pos is None:
                _skip("C6_broke")
                if debug:
                    print(
                        _base +
                        f"❌ C6: 跌破BB_4H后10根K线内未重新站上 "
                        f"(跌破@{str(bars.index[break_pos])[:16]})"
                    )
                continue
            anchor_pos = recovery_pos
            # 恢复后不可再跌破
            second_break = any(
                (not pd.isna(bars.iloc[j]["BB_4H"]))
                and bars.iloc[j]["Close"] < bars.iloc[j]["BB_4H"]
                for j in range(recovery_pos, i + 1)
            )
            if second_break:
                _skip("C6_broke")
                if debug:
                    print(
                        _base +
                        f"❌ C6: 重站上({str(bars.index[recovery_pos])[:16]})后再次跌破BB_4H"
                    )
                continue

        hold_bars_count = i - anchor_pos + 1
        if hold_bars_count < min_hold_bars:
            _skip("C6_hold_bars")
            if debug:
                print(
                    _base +
                    f"❌ C6: 锚点({str(bars.index[anchor_pos])[:16]})至今仅 "
                    f"{hold_bars_count} 根 < {min_hold_bars} 根"
                )
            continue

        # C7: BB_4H ≤ 收盘价 ≤ BB_4H × (1 + MAX_DIST_PCT)
        dist = (close - bb_4h) / bb_4h
        if not (bb_4h <= close <= bb_4h * (1 + MAX_DIST_PCT)):
            _skip("C7_dist")
            if debug:
                print(
                    _base +
                    f"❌ C7: 收盘价={close:.3f} 距BB_4H={dist*100:.2f}%"
                    f"  (BB_4H={bb_4h:.3f}, 阈值={bb_4h * (1 + MAX_DIST_PCT):.3f})"
                )
            continue

        # ── 全部条件满足 → 买入（4h 收盘价）────────────────────────────────
        position = {
            "date": date_str, "price": close, "pos": i,
            "signal_date": signal_date,
            "first_above_date": first_above_date,
        }
        if debug:
            print(
                _base +
                f"✅ 买入 @ {close:.3f}  BB_4H={bb_4h:.3f}  距离={dist*100:.2f}%  "
                f"CD信号={signal_date}  首站上={first_above_date}  "
                f"锚点={str(bars.index[anchor_pos])[:16]}  已站上{hold_bars_count}根"
            )

    # 未平仓 → 以最新收盘价计算浮盈
    if position is not None:
        last_row = bars.iloc[-1]
        trades.append({
            "buy_date": position["date"],
            "buy_price": round(position["price"], 3),
            "sell_date": None,
            "sell_price": round(last_row["Close"], 3),
            "return_pct": round(
                (last_row["Close"] / position["price"] - 1) * 100, 2
            ),
            "hold_bars": len(bars) - 1 - position["pos"],
            "status": "holding",
            "signal_date": position.get("signal_date"),
            "first_above_date": position.get("first_above_date"),
        })

    if debug:
        print(f"  │  {'─' * 110}")

    return trades, skip_counts
