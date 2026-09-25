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
    戻り値: (sig[bool], low_at[t]=直近3ヶ月の最安値, low_idx_at[t]=その日, near[bool]=底値圏か, feat{特徴量の配列}) いずれも長さn。
    """
    n = len(close)
    W = signals.BOTTOM_LOOKBACK_DAYS
    sig = np.zeros(n, dtype=bool)
    near = np.zeros(n, dtype=bool)
    low_at = np.full(n, np.nan)
    low_idx_at = np.zeros(n, dtype=int)
    feat = {k: np.full(n, np.nan) for k in ("pos", "dec", "reb", "dsl", "dd")}
    if n < POS_MIN + 2:
        return sig, low_at, low_idx_at, near, feat
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
    # 条件の効き目分析用の特徴量(その日時点で分かる値だけ)
    feat["pos"] = pos
    with np.errstate(invalid="ignore", divide="ignore"):
        feat["dd"] = 1 - close / hi  # 5年高値からの下落率
    feat["dec"][W - 1 :] = dec
    feat["reb"][W - 1 :] = reb
    feat["dsl"][W - 1 :] = dsl
    return sig, low_at, low_idx_at, near, feat


def _events(close: np.ndarray, state=None) -> list[tuple[int, float, int]]:
    """(シグナル日のインデックス, その時点の底値, 底をつけた日のインデックス)のリスト。初めて条件を満たした日。"""
    state = state if state is not None else _signal_state(close)
    return _pick(state[0], state)


def _pick(mask: np.ndarray, state) -> list[tuple[int, float, int]]:
    """maskがTrueの日を先頭から拾う(一度拾ったらCOOLDOWN日は拾わない。翌日に買えない最終日は除く)。"""
    low_at, low_idx_at = state[1], state[2]
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
    sig, low_idx_at = state[0], state[2]
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


MARKET_LABEL = "日本株全体(全銘柄の平均)"
MARKET_MA = 200


def market_index(closes: dict[str, pd.Series]) -> pd.Series | None:
    """
    「相場全体」の指数: 全銘柄の日々の騰落率の平均(等ウェイト)をつないだもの。
    1日±20%超の値動きはデータ異常の可能性があるので±20%に丸める。100銘柄以上そろっている日だけ使う。
    (バックテストの地合い判定と、アプリの「今の地合い」表示の両方でこれを使う)
    """
    rets = {t: s.pct_change() for t, s in closes.items() if t != MARKET_TICKER and len(s) > 250}
    if not rets:
        return None
    df = pd.DataFrame(rets).clip(-0.2, 0.2)
    cnt = df.notna().sum(axis=1)
    avg = df.mean(axis=1, skipna=True)[cnt >= 100].fillna(0)
    if avg.empty:
        return None
    return 100 * (1 + avg).cumprod()


def market_now(idx: pd.Series | None) -> dict | None:
    """アプリ用: 今日の地合い(指数が200日移動平均より上か下か)と、直近1年の推移。"""
    if idx is None or len(idx) < MARKET_MA + 5:
        return None
    ma = idx.rolling(MARKET_MA).mean()
    last = idx.iloc[-250:]
    lm = ma.iloc[-250:]
    base = last.index[0]
    return {
        "label": MARKET_LABEL,
        "up": bool(idx.iloc[-1] > ma.iloc[-1]),
        "gap": round(float(idx.iloc[-1] / ma.iloc[-1] - 1), 4),
        "date": idx.index[-1].strftime("%Y-%m-%d"),
        "chart": {
            "b": base.strftime("%Y-%m-%d"),
            "x": [int((d - base).days) for d in last.index],
            "y": [round(float(v), 2) for v in last],
            "ma": [None if pd.isna(v) else round(float(v), 2) for v in lm],
        },
    }


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


# 「統計上のベスト条件」の検証用: 買い方 × 買った日の地合い
COMBOS = [
    ("all_down", "底打ち候補 × 相場下向き"),
    ("all_up", "底打ち候補 × 相場上向き"),
    ("sig_90_down", "底打ち候補×類似度90%超え × 相場下向き"),
    ("sig_90_up", "底打ち候補×類似度90%超え × 相場上向き"),
    ("near_90_down", "底値圏×類似度90%超え即買い × 相場下向き"),
    ("near_90_up", "底値圏×類似度90%超え即買い × 相場上向き"),
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
        ex = [r for y, v in self.by_year.items() if y != 2020 for r in v]
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
            "h60_ex2020": _stats(ex),  # 2020年(コロナ後の急反発)を除いた60日保有
            "rules": rules,
            "by_year": [{"year": y, **_stats(v)} for y, v in sorted(self.by_year.items())],
        }


# ---- 条件の効き目分析 ------------------------------------------------------------
# 通常シグナルを「シグナル日に分かっていた条件」で分けて、どの条件だと成績が良かったかを見る。
# たまたま(過剰最適化)を避けるため、前半(〜2021年)と後半(2022年〜)の両方で全体平均を上回ったかも出す。
FACTORS = [
    ("dec", "3ヶ月の下落率", [(0.15, 0.20, "15〜20%"), (0.20, 0.30, "20〜30%"), (0.30, 0.40, "30〜40%"), (0.40, 9, "40%以上")]),
    ("reb", "底からの反発率(シグナル日)", [(0.10, 0.13, "10〜13%"), (0.13, 0.17, "13〜17%"), (0.17, 0.25, "17〜25%"), (0.25, 99, "25%以上")]),
    ("dsl", "底からの日数", [(3, 6, "3〜5日"), (6, 11, "6〜10日"), (11, 21, "11〜20日"), (21, 999, "21日以上")]),
    ("pos", "5年位置", [(-1, 0.10, "0〜10%"), (0.10, 0.20, "10〜20%"), (0.20, 0.31, "20〜30%")]),
    ("dd", "5年高値からの下落", [(0, 0.40, "40%未満"), (0.40, 0.55, "40〜55%"), (0.55, 0.70, "55〜70%"), (0.70, 1.01, "70%以上")]),
    ("vol", "値動きの荒さ(60日)", [(0, 0.02, "おとなしい(日2%未満)"), (0.02, 0.03, "ふつう(2〜3%)"), (0.03, 0.045, "荒い(3〜4.5%)"), (0.045, 9, "かなり荒い(4.5%以上)")]),
    ("px", "株価", [(0, 500, "500円未満"), (500, 1000, "500〜1000円"), (1000, 3000, "1000〜3000円"), (3000, 1e9, "3000円以上")]),
    ("sim", "勝ちパターン類似度", [(-1, 0.7, "0.70未満"), (0.7, 0.8, "0.70〜0.80"), (0.8, 0.9, "0.80〜0.90"), (0.9, 2, "0.90以上")]),
    ("regime", "相場全体の向き", [("down", None, "下向き"), ("up", None, "上向き")]),
    ("month", "買った月", [(m, None, f"{m}月") for m in range(1, 13)]),
    ("sector", "業種", None),  # 業種は値そのままで分ける(件数の多い順)
]
FACTOR_MIN_N = 150


def _bucket_label(spec, v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    for lo, hi, label in spec:
        if hi is None:
            if v == lo:
                return label
        elif lo <= v < hi:
            return label
    return None


def _factor_analysis(rows: list[dict]) -> dict | None:
    if not rows:
        return None
    df = pd.DataFrame(rows)
    df = df[df["r60"].notna()].copy()
    if df.empty:
        return None
    first = df["year"] <= 2021
    base = {
        "n": int(len(df)), "mean60": round(_robust_mean(df["r60"].to_numpy()), 4), "win60": round(float((df["r60"] > 0).mean()), 4),
        "h1": round(_robust_mean(df.loc[first, "r60"].to_numpy()), 4) if first.any() else None,
        "h2": round(_robust_mean(df.loc[~first, "r60"].to_numpy()), 4) if (~first).any() else None,
    }

    def summarize(sub: pd.DataFrame) -> dict:
        a = sub["r60"].to_numpy()
        f1, f2 = sub.loc[sub["year"] <= 2021, "r60"].to_numpy(), sub.loc[sub["year"] >= 2022, "r60"].to_numpy()
        ex = sub.loc[sub["year"] != 2020, "r60"].to_numpy()
        r15 = sub["r15"].dropna().to_numpy()
        h1 = round(_robust_mean(f1), 4) if len(f1) >= 30 else None
        h2 = round(_robust_mean(f2), 4) if len(f2) >= 30 else None
        return {
            "n": int(len(a)),
            "mean60": round(_robust_mean(a), 4),
            "med60": round(float(np.median(a)), 4),
            "win60": round(float((a > 0).mean()), 4),
            "ex2020": round(_robust_mean(ex), 4) if len(ex) >= 30 else None,
            "h1": h1, "h2": h2,
            # 前半・後半の両方で全体平均を上回った = 期間によらず効いていそう
            "stable": bool(h1 is not None and h2 is not None and base["h1"] is not None and base["h2"] is not None
                           and h1 > base["h1"] and h2 > base["h2"]),
            "r15": round(_robust_mean(r15), 4) if len(r15) >= 30 else None,
            "r15_win": round(float((r15 > 0).mean()), 4) if len(r15) >= 30 else None,
        }

    factors = []
    for key, label, spec in FACTORS:
        if key not in df:
            continue
        if spec is None:
            col = df[key].where(df[key].notna())
            order = col.value_counts().index.tolist()
        else:
            col = df[key].map(lambda v, sp=spec: _bucket_label(sp, v))
            order = [b[2] for b in spec]
        df[f"b_{key}"] = col
        buckets = []
        for b in order:
            sub = df[col == b]
            if len(sub) >= (FACTOR_MIN_N if spec is None else 30):
                buckets.append({"label": str(b), **summarize(sub)})
        if buckets:
            factors.append({"key": key, "label": label, "buckets": buckets})

    # 2つの条件の組み合わせで良かったもの(件数が少ないとたまたまが混じるので FACTOR_MIN_N 件以上、前半後半とも全体超え)
    keys = [f["key"] for f in factors if f["key"] not in ("month", "sector")]
    labels = {f["key"]: f["label"] for f in factors}
    pairs = []
    for i, k1 in enumerate(keys):
        for k2 in keys[i + 1 :]:
            g = df.groupby([f"b_{k1}", f"b_{k2}"])
            for (b1, b2), sub in g:
                if len(sub) < FACTOR_MIN_N:
                    continue
                st = summarize(sub)
                if st["stable"]:
                    pairs.append({"a": f"{labels[k1]}: {b1}", "b": f"{labels[k2]}: {b2}", **st})
    pairs.sort(key=lambda r: r["mean60"], reverse=True)
    # 同じ条件ばかり並ばないように、1つの条件が出てくるのは2回まで
    seen: dict[str, int] = {}
    picked = []
    for r in pairs:
        if seen.get(r["a"], 0) >= 2 or seen.get(r["b"], 0) >= 2:
            continue
        seen[r["a"]] = seen.get(r["a"], 0) + 1
        seen[r["b"]] = seen.get(r["b"], 0) + 1
        picked.append(r)
    pairs = picked

    return {"base": base, "factors": factors, "pairs": pairs[:12],
            "note": "通常シグナル(底値圏+3ヶ月底打ち)を、シグナル日に分かっていた条件で分けた60営業日後の成績。"
                    "『安定』=前半(〜2021年)と後半(2022年〜)の両方で全体平均を上回ったもの"}


def run(closes: dict[str, pd.Series], refs: list[dict] | None = None, sectors: dict[str, str] | None = None) -> dict:
    mkt = market_index(closes)
    mkt_label = MARKET_LABEL
    up = _market_uptrend(mkt)
    combo = {k: _Acc() for k, _ in COMBOS}
    ev_rows: list[dict] = []  # 条件の効き目分析用(通常シグナル1回=1行)
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
        feat = state[4]
        with np.errstate(invalid="ignore", divide="ignore"):
            lr = np.diff(np.log(close), prepend=np.nan)
        vol60 = pd.Series(lr).rolling(60, min_periods=40).std().to_numpy()
        if evs:
            stocks_with += 1

        # 各日の地合い("up"/"down"/None)
        if up is not None:
            upv = up.reindex(dates, method="ffill").to_numpy()
            regs = np.array([None if (v is None or (isinstance(v, float) and np.isnan(v))) else ("up" if v else "down") for v in upv], dtype=object)
        else:
            regs = np.full(n, None, dtype=object)

        use_sim = bool(tpl_all) and ticker not in ref_tickers
        if use_sim:
            T = np.stack([tp["z"] for tk, tp in tpl_all if tk != ticker])
            sim, _ = signals.pre_similarity_all(close, T)
            sig, near = state[0], state[3]
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
                        ck = f"{key}_{regs[t]}"
                        if ck in combo:
                            combo[ck].add(ticker, close, dates, t, low, float(sim[t]))

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
            reg = regs[t]
            if reg:
                combo[f"all_{reg}"].add(ticker, close, dates, t, low)
            row = {
                "dec": feat["dec"][t], "reb": feat["reb"][t], "pos": feat["pos"][t], "dsl": feat["dsl"][t], "dd": feat["dd"][t],
                "vol": vol60[t], "px": close[t], "month": dates[e].month, "year": dates[e].year,
                "sector": (sectors or {}).get(ticker), "regime": reg,
                "sim": float(sim[t]) if use_sim and not np.isnan(sim[t]) else np.nan,
                "r60": close[e + 60] / entry - 1 if e + 60 < n else np.nan,
                "r15": np.nan,
            }
            if e + MAX_HOLD < n:
                row["r15"] = _simulate(close[e : e + MAX_HOLD + 1] / entry - 1, 0.15, "floor", low / entry - 1)[0]
            ev_rows.append(row)
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
            "combo": [combo[k].out(k, label) for k, label in COMBOS],
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
        "factors": _factor_analysis(ev_rows),
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
    fa = bt.get("factors")
    if fa:
        b = fa["base"]
        lines += ["", "### 条件の効き目分析(60日保有)", f"全体: {b['n']}件 平均 {f(b['mean60'])} 勝率 {w(b['win60'])} / 前半 {f(b['h1'])} 後半 {f(b['h2'])}", "",
                  "| 条件 | 区分 | 件数 | 平均 | 勝率 | 2020除く | 前半 | 後半 | 安定 | +15/底値割れ |", "|---|---|---|---|---|---|---|---|---|---|"]
        for fc in fa["factors"]:
            for x in fc["buckets"]:
                lines.append(f"| {fc['label']} | {x['label']} | {x['n']} | {f(x['mean60'])} | {w(x['win60'])} | {f(x['ex2020'])} | {f(x['h1'])} | {f(x['h2'])} | {'◎' if x['stable'] else ''} | {f(x['r15'])} |")
        if fa.get("pairs"):
            lines += ["", "#### 良かった組み合わせ(前半・後半とも全体超え)", "| 条件1 | 条件2 | 件数 | 平均 | 勝率 | 2020除く |", "|---|---|---|---|---|---|"]
            for x in fa["pairs"]:
                lines.append(f"| {x['a']} | {x['b']} | {x['n']} | {f(x['mean60'])} | {w(x['win60'])} | {f(x['ex2020'])} |")
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
        if bs.get("combo"):
            lines += ["", "### 地合いとの組み合わせ(相場下向き/上向き = 買った日に日本株全体の指数が200日線より下/上)",
                      "| 買い方 | 件数 | 60日 平均/勝率 | 60日(2020年除く) 平均/勝率 | 120日 平均/勝率 | +15/底値割れ 平均/勝率/日数 | +30/-20 平均/勝率/日数 |",
                      "|---|---|---|---|---|---|---|"]
            for r in [bs["all"]] + [x for x in bs["hit"] if x["key"] == "sig_90"] + bs["combo"]:
                hd = {x["days"]: x for x in r["hold"]}
                rl = {(x["tp"], x["sl"]): x for x in r["rules"]}
                c2 = lambda x: "-" if not x or not x.get("n") else f"{f(x.get('mean'))} / {w(x.get('win'))}"
                c3 = lambda x: "-" if not x or not x.get("n") else f"{f(x.get('mean'))} / {w(x.get('win'))} / {x.get('avg_days')}日"
                lines.append(f"| {r['label']} | {r['events']} | {c2(hd.get(60))} | {c2(r.get('h60_ex2020'))} | {c2(hd.get(120))} | "
                             f"{c3(rl.get((0.15, 'floor')))} | {c3(rl.get((0.30, 0.20)))} |")
        h90 = next((r for r in bs["hit"] if r["key"] == "near_90"), None)
        if h90 and h90["by_year"]:
            lines += ["", "#### 底値圏で類似度90%超え→即買い の年別(60日保有)", "| 年 | 件数 | 平均 | 勝率 |", "|---|---|---|---|"]
            for r in h90["by_year"]:
                lines.append(f"| {r['year']} | {r.get('n', 0)} | {f(r.get('mean'))} | {w(r.get('win'))} |")
    return "\n".join(lines) + "\n"
