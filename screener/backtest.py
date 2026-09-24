"""
バックテスト: 「底値圏 + 3ヶ月底打ち」のシグナルが過去に出た瞬間を全銘柄から拾い、
その後に買っていたらどうなったかを集計する。

やっていること(先読みしないように、各日付ではその日までのデータだけで判定):
  1. 各銘柄・各営業日について、その日時点の
       - 5年位置(その日までの最大5年・最低2年の値幅の中での位置)
       - 3ヶ月底打ち(signals.pyと同じ条件)
     を計算し、両方満たした日を「シグナル日」とする
  2. 同じ銘柄で連続して出るのを避けるため、一度シグナルが出たら60営業日は拾わない
  3. シグナル日の「翌営業日の終値」で買ったことにする(当日終値で判定→翌日買い)
  4. 集計:
     - 保有期間別(20/40/60/120/250営業日後に売った場合)の平均・中央値・勝率
       → 同じ期間に「適当な銘柄を適当な日に買った場合」と比較
     - 利確/損切りルール別の成績(終値ベース、最大250営業日で強制決済)
     - 保有60日間の最大含み損/含み益の分布(損切り幅を決める材料)
     - 年別の成績(相場環境に左右されていないか)

注意(結果を読むときに必ず意識すること):
  - 現在上場している銘柄だけで計算している(途中で上場廃止になった銘柄は含まれない)ので、
    実際より成績が良く出る方向の偏り(生存バイアス)がある
  - 売買手数料・スリッページは含めていない
  - 期間は直近約10年分の株価(シグナル判定には最低2年分の過去データが要るので、実際に拾えるのは約8年分)
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from screener import signals

HOLD_DAYS = [20, 40, 60, 120, 250]
TP_LIST = [0.10, 0.15, 0.20, 0.30, 0.50]
SL_LIST = [0.05, 0.08, 0.10, 0.15, 0.20, "floor"]  # "floor" = シグナル時の底値を終値で割ったら損切り
MAX_HOLD = 250
COOLDOWN = 60
POS_WINDOW = 1250  # 約5年
POS_MIN = 500  # 最低約2年分はないと5年位置を判定しない
JST = timezone(timedelta(hours=9))


def _signal_state(close: np.ndarray):
    """
    各営業日tについて「底値圏 + 3ヶ月底打ち」を満たすか(先読みなし)と、その時点の底値・底の日。
    戻り値: (sig[bool], low_at[t]=直近3ヶ月の最安値, low_idx_at[t]=その日, near[bool]=底値圏か) いずれも長さn。
    """
    n = len(close)
    W = signals.BOTTOM_LOOKBACK_DAYS
    sig = np.zeros(n, dtype=bool)
    near = np.zeros(n, dtype=bool)
    low_at = np.full(n, np.nan)
    low_idx_at = np.zeros(n, dtype=int)
    if n < POS_MIN + 2:
        return sig, low_at, low_idx_at, near
    s = pd.Series(close)
    lo = s.rolling(POS_WINDOW, min_periods=POS_MIN).min().to_numpy()
    hi = s.rolling(POS_WINDOW, min_periods=POS_MIN).max().to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        pos = (close - lo) / (hi - lo)
    near = pos <= signals.NEAR_LOW_THRESHOLD  # NaNはFalse

    win = sliding_window_view(close, W)  # win[k] = close[k : k+W], 判定日 t = k+W-1
    li = win.argmin(axis=1)
    rows = np.arange(len(win))
    low = win[rows, li]
    pre = np.maximum.accumulate(win, axis=1)[rows, li]
    cur = win[:, -1]
    with np.errstate(invalid="ignore", divide="ignore"):
        dec = (pre - low) / pre
        reb = (cur - low) / low
    dsl = W - 1 - li
    bottomed = (dec >= signals.MIN_DECLINE_PCT) & (reb >= signals.MIN_REBOUND_PCT) & (dsl >= signals.MIN_DAYS_SINCE_LOW)

    sig[W - 1 :] = bottomed & near[W - 1 :]
    low_at[W - 1 :] = low
    low_idx_at[W - 1 :] = rows + li
    return sig, low_at, low_idx_at, near


def _events(close: np.ndarray, state=None) -> list[tuple[int, float, int]]:
    """(シグナル日のインデックス, その時点の底値, 底をつけた日のインデックス)のリスト。初めて条件を満たした日。"""
    state = state if state is not None else _signal_state(close)
    return _pick(state[0], state)


def _pick(mask: np.ndarray, state) -> list[tuple[int, float, int]]:
    """maskがTrueの日を先頭から拾う(一度拾ったらCOOLDOWN日は拾わない。翌日に買えない最終日は除く)。"""
    _, low_at, low_idx_at, _ = state
    n = len(mask)
    out, last = [], -10**9
    for t in np.flatnonzero(mask):
        if t - last < COOLDOWN or t + 1 >= n:
            continue
        out.append((int(t), float(low_at[t]), int(low_idx_at[t])))
        last = t
    return out


def _post_sim_series(close: np.ndarray, state, ref_curves) -> tuple[np.ndarray, np.ndarray]:
    """(旧方式・比較用)条件を満たしている各日の「底からの上昇カーブ」類似度と比較日数。それ以外の日はNaN/0。"""
    sig, _, low_idx_at, _ = state
    sim = np.full(len(close), np.nan)
    ln = np.zeros(len(close), dtype=int)
    if ref_curves:
        for t in np.flatnonzero(sig):
            sim[t], ln[t] = _similarity_at(close, int(low_idx_at[t]), int(t), ref_curves)
    return sim, ln


def _ref_curves(refs: list[dict] | None) -> list[tuple[str, np.ndarray]]:
    """(旧方式・比較用)勝ちパターン銘柄の「底からSIMILARITY_WINDOW日分の上昇カーブ」(底=100)。"""
    out = []
    for r in refs or []:
        b = r["close"].loc[r["low"]:].iloc[: signals.SIMILARITY_WINDOW].to_numpy(dtype=float)
        if len(b) >= 5 and b[0]:
            out.append((r["name"], b / b[0] * 100))
    return out


def _similarity_at(close: np.ndarray, low_idx: int, t: int, ref_curves) -> tuple[float, int]:
    """
    (旧方式・比較用)シグナル日tの時点で分かるデータだけを使った「底からの上昇カーブ」の類似度。
    戻り値: (最大類似度, 比較に使えた日数)
    """
    a = close[low_idx : min(t + 1, low_idx + signals.SIMILARITY_WINDOW)]
    if len(a) < 5 or not a[0]:
        return 0.0, len(a)
    a = a / a[0] * 100
    best = 0.0
    for _, b in ref_curves:
        n = min(len(a), len(b))
        if n < 5:
            continue
        with np.errstate(invalid="ignore", divide="ignore"):
            c = np.corrcoef(a[:n], b[:n])[0, 1]
        if not np.isnan(c):
            best = max(best, float(c))
    return best, len(a)


def _market_proxy(closes: dict[str, pd.Series]) -> pd.Series | None:
    """全銘柄の日々の値動きの中央値をつないだ「相場全体」の指数(等ウェイト)。"""
    rets = {}
    for t, s in closes.items():
        if len(s) > 250:
            rets[t] = s.pct_change()
    if not rets:
        return None
    df = pd.DataFrame(rets)
    med = df.median(axis=1, skipna=True)
    cnt = df.notna().sum(axis=1)
    med = med[cnt >= 100].fillna(0)
    return (1 + med).cumprod()


def _simulate(path: np.ndarray, tp: float, sl, floor_ret: float) -> tuple[float, int]:
    """path = 買値からの損益率の推移(0日目=0)。先に到達した方で決済。"""
    stop = floor_ret if sl == "floor" else -sl
    hit_tp = np.flatnonzero(path >= tp)
    hit_sl = np.flatnonzero(path < stop) if sl == "floor" else np.flatnonzero(path <= stop)
    i_tp = hit_tp[0] if len(hit_tp) else None
    i_sl = hit_sl[0] if len(hit_sl) else None
    if i_tp is not None and (i_sl is None or i_tp <= i_sl):
        return float(path[i_tp]), int(i_tp)
    if i_sl is not None:
        return float(path[i_sl]), int(i_sl)
    return float(path[-1]), len(path) - 1


def _robust_mean(a: np.ndarray) -> float:
    """上下1%を端の値に丸めてから平均(一部の異常値や超大化け1銘柄に平均が引っ張られないように)。"""
    if len(a) < 20:
        return float(a.mean())
    lo, hi = np.percentile(a, [1, 99])
    return float(np.clip(a, lo, hi).mean())


def _stats(x: list[float]) -> dict:
    a = np.asarray(x, dtype=float)
    if not len(a):
        return {"n": 0}
    return {
        "n": int(len(a)),
        "mean": round(_robust_mean(a), 4),
        "median": round(float(np.median(a)), 4),
        "win": round(float((a > 0).mean()), 4),
        "p25": round(float(np.percentile(a, 25)), 4),
        "p75": round(float(np.percentile(a, 75)), 4),
    }


MARKET_TICKER = "1306.T"  # TOPIX連動ETF。相場全体の地合いの判定に使う


def _market_uptrend(market: pd.Series | None) -> pd.Series | None:
    """日付→「TOPIXが200日移動平均より上か」。"""
    if market is None or len(market) < 250:
        return None
    return (market > market.rolling(200).mean()).where(market.rolling(200).mean().notna())


KEY_RULES = [(0.50, 0.20), (0.30, 0.20), (0.15, "floor"), (0.10, "floor")]

# 類似度での比較(類似度=「上がる前の形」で比べる新方式。各日その日までの株価だけで計算)
#  - first_*: 通常シグナル(底打ち条件を初めて満たした日)を、その日の類似度で分けたもの
#  - hit_*:   類似度が基準を初めて超えた日に買ったもの
SIM_BUCKETS = [
    ("first_90", "通常シグナル×類似度90%以上", 0.90, 9.0),
    ("first_80", "通常シグナル×類似度80〜90%", 0.80, 0.90),
    ("first_70", "通常シグナル×類似度70〜80%", 0.70, 0.80),
    ("first_lo", "通常シグナル×類似度70%未満", -9.0, 0.70),
]
# (key, label, 類似度の基準, 前提条件: "near"=底値圏なら反発を待たない / "sig"=底打ち条件も満たす / "post"=旧方式)
SIM_HITS = [
    ("near_90", "底値圏で類似度90%超え→即買い(反発を待たない)", 0.90, "near"),
    ("near_95", "底値圏で類似度95%超え→即買い(反発を待たない)", 0.95, "near"),
    ("sig_90", "底打ち候補で類似度90%超え→買い", 0.90, "sig"),
    ("post_90", "【旧方式】上昇カーブ類似度90%超え(比較15日以上)", 0.90, "post"),
]


class _Acc:
    """1つの買い方(バケット)の成績を集める箱。"""

    def __init__(self):
        self.hold = {h: [] for h in HOLD_DAYS}
        self.rules = {k: [] for k in KEY_RULES}
        self.by_year: dict[int, list[float]] = {}
        self.sims: list[float] = []
        self.lens: list[int] = []
        self.stocks: set[str] = set()
        self.from_low: list[float] = []

    def add(self, ticker, close, dates, t, low, sim=None, ln=None):
        n = len(close)
        e = t + 1
        entry = close[e]
        self.stocks.add(ticker)
        if sim is not None and not np.isnan(sim):
            self.sims.append(sim)
        if ln is not None:
            self.lens.append(ln)
        if low and not np.isnan(low):
            self.from_low.append(entry / low - 1)  # 買った時点で直近3ヶ月の最安値から何%上がっていたか
        for h in HOLD_DAYS:
            if e + h < n:
                r = close[e + h] / entry - 1
                self.hold[h].append(r)
                if h == 60:
                    self.by_year.setdefault(dates[e].year, []).append(r)
        if e + MAX_HOLD < n:
            path = close[e : e + MAX_HOLD + 1] / entry - 1
            floor_ret = low / entry - 1
            for tp, sl in KEY_RULES:
                self.rules[(tp, sl)].append(_simulate(path, tp, sl, floor_ret))

    def out(self, key, label) -> dict:
        rules = []
        for (tp, sl), trades in self.rules.items():
            if not trades:
                rules.append({"tp": tp, "sl": sl, "n": 0})
                continue
            rets = np.array([r for r, _ in trades])
            days = np.array([d for _, d in trades])
            rules.append({"tp": tp, "sl": sl, "n": int(len(rets)), "mean": round(_robust_mean(rets), 4),
                          "median": round(float(np.median(rets)), 4), "win": round(float((rets > 0).mean()), 4),
                          "avg_days": round(float(days.mean()), 1)})
        return {
            "key": key,
            "label": label,
            "events": len(self.hold[HOLD_DAYS[0]]) if self.hold[HOLD_DAYS[0]] else 0,
            "stocks": len(self.stocks),
            "sim_med": round(float(np.median(self.sims)), 3) if self.sims else None,
            "len_med": int(np.median(self.lens)) if self.lens else None,
            "from_low_med": round(float(np.median(self.from_low)), 4) if self.from_low else None,
            "hold": [{"days": h, **_stats(self.hold[h])} for h in HOLD_DAYS],
            "rules": rules,
            "by_year": [{"year": y, **_stats(v)} for y, v in sorted(self.by_year.items())],
        }


def run(closes: dict[str, pd.Series], refs: list[dict] | None = None) -> dict:
    mkt = closes.get(MARKET_TICKER)
    mkt_label = "TOPIX"
    if mkt is None or len(mkt) < 250:
        mkt = _market_proxy(closes)  # TOPIX ETFが取れない時は全銘柄の中央値で代用
        mkt_label = "全銘柄の中央値指数"
    up = _market_uptrend(mkt)
    ref_curves = _ref_curves(refs)  # 旧方式(比較用)
    ref_tickers = {r["ticker"] for r in refs or []}
    tpl_all = [(r["ticker"], tp) for r in refs or [] for tp in r.get("templates", [])]
    sim_acc = {k: _Acc() for k, *_ in SIM_BUCKETS + SIM_HITS}
    sim_first_all = _Acc()  # 類似度の比較対象にできた通常シグナル全体(勝ちパターン銘柄自身は除く)

    regime = {"up": {60: [], 120: []}, "down": {60: [], 120: []}}
    hold = {h: [] for h in HOLD_DAYS}
    base = {h: [] for h in HOLD_DAYS}
    grid = {(tp, sl): [] for tp in TP_LIST for sl in SL_LIST}
    mae60, mfe60 = [], []
    by_year: dict[int, list[float]] = {}
    events_all = 0
    stocks_with = 0
    first_date, last_date = None, None

    rng = np.random.default_rng(0)
    for ticker, ser in closes.items():
        if ticker == MARKET_TICKER:
            continue
        close = ser.to_numpy(dtype=float)
        dates = ser.index
        n = len(close)
        if n < POS_MIN + 2:
            continue
        first_date = dates[0] if first_date is None or dates[0] < first_date else first_date
        last_date = dates[-1] if last_date is None or dates[-1] > last_date else last_date

        # 比較用: 同じ銘柄を(判定可能な期間の)ランダムな日に買った場合
        cand = np.arange(POS_MIN, n - 1)
        if len(cand):
            for d in rng.choice(cand, size=min(12, len(cand)), replace=False):
                for h in HOLD_DAYS:
                    if d + h < n:
                        base[h].append(close[d + h] / close[d] - 1)

        state = _signal_state(close)
        evs = _events(close, state)
        if evs:
            stocks_with += 1

        use_sim = bool(tpl_all) and ticker not in ref_tickers
        if use_sim:
            T = np.stack([tp["z"] for tk, tp in tpl_all if tk != ticker])
            sim, _ = signals.pre_similarity_all(close, T)
            sig, _, _, near = state
            with np.errstate(invalid="ignore"):
                for key, _, thr, cond in SIM_HITS:
                    if cond == "post":
                        if not ref_curves or not sig.any():
                            continue
                        psim, pln = _post_sim_series(close, state, ref_curves)
                        mask = sig & (pln >= 15) & (psim >= thr)
                        for t, low, _li in _pick(mask, state):
                            sim_acc[key].add(ticker, close, dates, t, low, float(psim[t]), int(pln[t]))
                        continue
                    mask = (near if cond == "near" else sig) & (sim >= thr)
                    for t, low, _li in _pick(mask, state):
                        sim_acc[key].add(ticker, close, dates, t, low, float(sim[t]))

        for t, low, low_idx in evs:
            e = t + 1
            entry = close[e]
            events_all += 1
            if use_sim and not np.isnan(sim[t]):
                sv = float(sim[t])
                sim_first_all.add(ticker, close, dates, t, low, sv)
                for key, _, lo_, hi_ in SIM_BUCKETS:
                    if lo_ <= sv < hi_:
                        sim_acc[key].add(ticker, close, dates, t, low, sv)
                        break
            reg = None
            if up is not None:
                v = up.asof(dates[t])
                reg = None if pd.isna(v) else ("up" if bool(v) else "down")
            for h in HOLD_DAYS:
                if e + h < n:
                    r = close[e + h] / entry - 1
                    hold[h].append(r)
                    if h == 60:
                        by_year.setdefault(dates[e].year, []).append(r)
                    if reg and h in (60, 120):
                        regime[reg][h].append(r)
            if e + 60 < n:
                p60 = close[e : e + 61] / entry - 1
                mae60.append(float(p60.min()))
                mfe60.append(float(p60.max()))
            if e + MAX_HOLD < n:  # 決済まで追える取引だけでルールを比較(途中のものは除外)
                path = close[e : e + MAX_HOLD + 1] / entry - 1
                floor_ret = low / entry - 1
                for tp in TP_LIST:
                    for sl in SL_LIST:
                        grid[(tp, sl)].append(_simulate(path, tp, sl, floor_ret))

    hold_rows = []
    for h in HOLD_DAYS:
        st, bs = _stats(hold[h]), _stats(base[h])
        hold_rows.append({"days": h, **st, "base_mean": bs.get("mean"), "base_median": bs.get("median"), "base_win": bs.get("win")})

    grid_rows = []
    for (tp, sl), trades in grid.items():
        if not trades:
            continue
        rets = np.array([r for r, _ in trades])
        days = np.array([d for _, d in trades])
        grid_rows.append({
            "tp": tp,
            "sl": sl,
            "n": int(len(rets)),
            "mean": round(_robust_mean(rets), 4),
            "median": round(float(np.median(rets)), 4),
            "win": round(float((rets > 0).mean()), 4),
            "avg_days": round(float(days.mean()), 1),
            # 1営業日あたりの平均リターン(資金効率の目安)
            "per_day": round(_robust_mean(rets) / max(float(days.mean()), 1), 5),
        })
    grid_rows.sort(key=lambda r: r["mean"], reverse=True)

    def pct(a, q):
        return round(float(np.percentile(a, q)), 4) if len(a) else None

    by_similarity = None
    if tpl_all:
        by_similarity = {
            "refs": [r["name"] for r in refs or [] if r.get("templates")],
            "ref_lows": {r["name"]: r["low"].strftime("%Y-%m-%d") for r in refs or [] if r.get("templates")},
            "window": signals.PRE_WINDOW,
            "all": sim_first_all.out("all", "通常シグナル全体(比較用)"),
            "first": [sim_acc[k].out(k, label) for k, label, *_ in SIM_BUCKETS],
            "hit": [sim_acc[k].out(k, label) for k, label, *_ in SIM_HITS],
            "note": "類似度は「その日までの直近60営業日の形」と「勝ちパターン銘柄が大きく上がる前(底〜底の20日後)の形」の相関。"
                    "各日その日までの株価だけで計算(アプリの表示と同じ)。勝ちパターン銘柄自身は除外。"
                    "『買値の底からの上昇』=買った時点で直近3ヶ月の最安値から既に何%上がっていたか(小さいほど早く乗れている)",
        }

    return {
        "generated_at": datetime.now(JST).isoformat(timespec="minutes"),
        "period": {
            "from": first_date.strftime("%Y-%m-%d") if first_date is not None else None,
            "to": last_date.strftime("%Y-%m-%d") if last_date is not None else None,
        },
        "stocks": len(closes),
        "events": events_all,
        "stocks_with_events": stocks_with,
        "rules": {
            "near_low": signals.NEAR_LOW_THRESHOLD,
            "min_decline": signals.MIN_DECLINE_PCT,
            "min_rebound": signals.MIN_REBOUND_PCT,
            "cooldown": COOLDOWN,
            "max_hold": MAX_HOLD,
            "entry": "シグナル翌営業日の終値",
        },
        "hold": hold_rows,
        "grid": grid_rows,
        "excursion60": {
            "n": len(mae60),
            "mae_p25": pct(mae60, 25), "mae_med": pct(mae60, 50), "mae_p75": pct(mae60, 75),
            "mfe_p25": pct(mfe60, 25), "mfe_med": pct(mfe60, 50), "mfe_p75": pct(mfe60, 75),
        },
        "by_year": [{"year": y, **_stats(v)} for y, v in sorted(by_year.items())],
        "by_regime": [
            {"regime": "up", "label": f"相場全体が上向き({mkt_label}が200日線より上)", "h60": _stats(regime["up"][60]), "h120": _stats(regime["up"][120])},
            {"regime": "down", "label": f"相場全体が下向き({mkt_label}が200日線より下)", "h60": _stats(regime["down"][60]), "h120": _stats(regime["down"][120])},
        ] if up is not None else [],
        "by_similarity": by_similarity,
        "caveats": [
            "現在上場中の銘柄だけで計算(上場廃止銘柄が入らない分、成績は実際より良く出やすい)",
            "手数料・スリッページは含まない",
            "約10年分の株価で検証(シグナルは約8年分)。この期間の相場環境に依存する",
            "利確・損切りは終値で判定(ザラ場の値動きは考慮しない)",
            "平均は上下1%の極端な値を丸めて計算(データ異常や一部の超大化けに引っ張られないように)",
            "勝ちパターンの形(比較対象のカーブ)は今の設定のものを過去にも当てはめている",
        ],
    }


def summary_markdown(bt: dict) -> str:
    """GitHub Actionsの実行結果ページに出す要約。"""
    f = lambda v: "-" if v is None else f"{v * 100:+.1f}%"
    w = lambda v: "-" if v is None else f"{v * 100:.0f}%"
    lines = [
        "## バックテスト結果",
        f"期間 {bt['period']['from']} 〜 {bt['period']['to']} / 対象 {bt['stocks']} 銘柄 / シグナル {bt['events']} 回({bt['stocks_with_events']} 銘柄)",
        "",
        "### 保有期間別(シグナル翌日終値で買い)",
        "| 保有 | 件数 | 平均 | 中央値 | 勝率 | ランダム平均 | ランダム中央値 | ランダム勝率 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in bt["hold"]:
        lines.append(f"| {r['days']}日 | {r.get('n', 0)} | {f(r.get('mean'))} | {f(r.get('median'))} | {w(r.get('win'))} | {f(r.get('base_mean'))} | {f(r.get('base_median'))} | {w(r.get('base_win'))} |")
    lines += ["", "### 利確/損切りルール別(全30通り・平均順、最大250営業日保有)", "| 利確 | 損切り | 件数 | 平均 | 中央値 | 勝率 | 平均保有日数 | 1日あたり |", "|---|---|---|---|---|---|---|---|"]
    for r in bt["grid"]:
        sl = "底値割れ" if r["sl"] == "floor" else f"-{r['sl'] * 100:.0f}%"
        lines.append(f"| +{r['tp'] * 100:.0f}% | {sl} | {r['n']} | {f(r['mean'])} | {f(r.get('median'))} | {w(r['win'])} | {r['avg_days']} | {r['per_day'] * 100:.3f}% |")
    ex = bt["excursion60"]
    lines += ["", "### 60日保有中の最大含み損/含み益", f"最大含み損: 25%点 {f(ex['mae_p25'])} / 中央値 {f(ex['mae_med'])} / 75%点 {f(ex['mae_p75'])}",
              f"最大含み益: 25%点 {f(ex['mfe_p25'])} / 中央値 {f(ex['mfe_med'])} / 75%点 {f(ex['mfe_p75'])}",
              "", "### 年別(60日保有)", "| 年 | 件数 | 平均 | 勝率 |", "|---|---|---|---|"]
    for r in bt["by_year"]:
        lines.append(f"| {r['year']} | {r.get('n', 0)} | {f(r.get('mean'))} | {w(r.get('win'))} |")
    if bt.get("by_regime"):
        lines += ["", "### 相場全体の地合い別", "| 地合い | 60日 件数 | 60日 平均 | 60日 中央値 | 60日 勝率 | 120日 平均 | 120日 中央値 | 120日 勝率 |", "|---|---|---|---|---|---|---|---|"]
        for r in bt["by_regime"]:
            a, b = r["h60"], r["h120"]
            lines.append(f"| {r['label']} | {a.get('n', 0)} | {f(a.get('mean'))} | {f(a.get('median'))} | {w(a.get('win'))} | {f(b.get('mean'))} | {f(b.get('median'))} | {w(b.get('win'))} |")
    bs = bt.get("by_similarity")
    if bs:
        lines += ["", "### 勝ちパターン類似度別(上がる前の形で比較)", "比較対象(底の日): " + ", ".join(f"{k} {v}" for k, v in bs.get("ref_lows", {}).items()), "",
                  "| 買い方 | 件数 | 類似度中央値 | 買値の底からの上昇 | 20日 平均/勝率 | 60日 平均/勝率 | 120日 平均/勝率 | 250日 平均/勝率 | +30/-20 平均/勝率 | +15/底値割れ 平均/勝率 |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for r in [bs["all"]] + bs["first"] + bs["hit"]:
            hd = {x["days"]: x for x in r["hold"]}
            rl = {(x["tp"], x["sl"]): x for x in r["rules"]}
            cell = lambda x: "-" if not x or not x.get("n") else f"{f(x.get('mean'))} / {w(x.get('win'))}"
            sm = "-" if r["sim_med"] is None else f"{r['sim_med']:.2f}"
            lines.append(f"| {r['label']} | {r['events']} | {sm} | {f(r.get('from_low_med'))} | "
                         f"{cell(hd.get(20))} | {cell(hd.get(60))} | {cell(hd.get(120))} | {cell(hd.get(250))} | "
                         f"{cell(rl.get((0.30, 0.20)))} | {cell(rl.get((0.15, 'floor')))} |")
        h90 = next((r for r in bs["hit"] if r["key"] == "near_90"), None)
        if h90 and h90["by_year"]:
            lines += ["", "#### 底値圏で類似度90%超え→即買い の年別(60日保有)", "| 年 | 件数 | 平均 | 勝率 |", "|---|---|---|---|"]
            for r in h90["by_year"]:
                lines.append(f"| {r['year']} | {r.get('n', 0)} | {f(r.get('mean'))} | {w(r.get('win'))} |")
    return "\n".join(lines) + "\n"
