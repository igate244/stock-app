"""
出来高急増(ボリュームスパイク)の検出とバックテスト。

「普段の何倍もの出来高を伴って株価が上がった日」は、大口の資金が入ったサインとされる。
その日を全銘柄から拾い、翌営業日の終値で買っていたらどうなったかを過去10年で集計する。
(底打ちシグナルと同じく、各日その日までのデータだけで判定=先読みなし)

判定条件(今日の銘柄一覧もバックテストも同じ):
  - 出来高が直近20営業日の平均の VS_RATIO 倍以上
  - その日の株価が前日比 +VS_MIN_RET 以上
  - 直近20日の平均売買代金が VS_MIN_TURNOVER 円以上(ほとんど売買のない銘柄の急増はノイズなので除外)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from screener import backtest as bt

VS_RATIO = 3.0
VS_MIN_RET = 0.03
VS_MIN_TURNOVER = 30_000_000  # 3千万円/日
AVG_WIN = 20
SHOW_DAYS = 3  # 画面には直近3営業日以内の急増を出す
COOLDOWN = 20
HOLD_DAYS = [5, 20, 60, 120]
MAX_HOLD = 60
RULES = [(0.10, 0.05), (0.10, 0.10), (0.15, 0.08), (0.20, 0.10), (0.30, 0.15)]


def features(close: pd.Series, vol: pd.Series) -> pd.DataFrame:
    """各営業日の 出来高倍率・当日騰落・平均売買代金・52週高値更新・5年位置。"""
    vol = vol.reindex(close.index).fillna(0)
    avg = vol.shift(1).rolling(AVG_WIN, min_periods=15).mean()
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = vol / avg
    ret = close.pct_change()
    turnover = (close * vol).shift(1).rolling(AVG_WIN, min_periods=15).mean()
    hi52 = close >= close.shift(1).rolling(250, min_periods=120).max()
    lo = close.rolling(bt.POS_WINDOW, min_periods=bt.POS_MIN).min()
    hi = close.rolling(bt.POS_WINDOW, min_periods=bt.POS_MIN).max()
    with np.errstate(invalid="ignore", divide="ignore"):
        pos = (close - lo) / (hi - lo)
    ret20 = close / close.shift(20) - 1  # 急増の前の20日でどれだけ動いていたか
    return pd.DataFrame({"ratio": ratio, "ret": ret, "to": turnover, "hi52": hi52, "pos": pos, "pre20": ret20})


def signal_mask(f: pd.DataFrame) -> pd.Series:
    return (f["ratio"] >= VS_RATIO) & (f["ret"] >= VS_MIN_RET) & (f["to"] >= VS_MIN_TURNOVER)


def today(closes: dict[str, pd.Series], volumes: dict[str, pd.Series], tickers: list[str]) -> dict[str, dict]:
    """直近SHOW_DAYS営業日以内に急増した銘柄 → 画面用の情報。"""
    out = {}
    for t in tickers:
        c, v = closes.get(t), volumes.get(t)
        if c is None or v is None or len(c) < 60 or v.tail(5).sum() == 0:
            continue
        f = features(c, v)
        m = signal_mask(f).to_numpy()
        n = len(m)
        hits = [i for i in range(max(0, n - SHOW_DAYS), n) if m[i]]
        if not hits:
            continue
        i = hits[-1]
        r = f.iloc[i]
        out[t] = {
            "r": round(float(r["ratio"]), 1),          # 出来高倍率
            "d": round(float(r["ret"]), 4),            # その日の上昇率
            "to": int(r["to"] // 1_000_000),           # 平均売買代金(百万円)
            "hi": bool(r["hi52"]),                     # 52週高値を更新したか
            "pos": None if pd.isna(r["pos"]) else round(float(r["pos"]), 3),
            "pre": None if pd.isna(r["pre20"]) else round(float(r["pre20"]), 4),
            "age": int(n - 1 - i),                     # 何営業日前の急増か(0=今日)
            "date": c.index[i].strftime("%Y-%m-%d"),
            "since": round(float(c.iloc[-1] / c.iloc[i] - 1), 4),  # 急増した日から今日までの値動き
        }
    return out


VOL_FACTORS = [
    ("ratio", "出来高の倍率", [(3, 5, "3〜5倍"), (5, 10, "5〜10倍"), (10, 1e9, "10倍以上")]),
    ("ret", "その日の上昇率", [(0.03, 0.07, "+3〜7%"), (0.07, 0.15, "+7〜15%"), (0.15, 9, "+15%以上")]),
    ("pos", "5年位置", [(-1, 0.3, "底値圏(0〜30%)"), (0.3, 0.7, "中間(30〜70%)"), (0.7, 1.01, "高値圏(70%〜)")]),
    ("hi52", "52週高値", [(True, None, "更新した"), (False, None, "更新してない")]),
    ("pre20", "直前20日の値動き", [(-9, -0.1, "-10%以上下げてた"), (-0.1, 0.1, "横ばい(±10%)"), (0.1, 9, "+10%以上上げてた")]),
    ("to", "売買代金(20日平均)", [(0, 1e8, "1億円未満"), (1e8, 1e9, "1〜10億円"), (1e9, 1e14, "10億円以上")]),
    ("regime", "相場全体の向き", [("down", None, "下向き"), ("up", None, "上向き")]),
    ("px", "株価", [(0, 500, "500円未満"), (500, 1000, "500〜1000円"), (1000, 3000, "1000〜3000円"), (3000, 1e9, "3000円以上")]),
    ("month", "月", [(m, None, f"{m}月") for m in range(1, 13)]),
    ("sector", "業種", None),
]


def run_backtest(closes: dict[str, pd.Series], volumes: dict[str, pd.Series], sectors: dict[str, str] | None = None) -> dict | None:
    if not volumes:
        return None
    up = bt._market_uptrend(bt.market_index(closes))
    hold = {h: [] for h in HOLD_DAYS}
    base = {h: [] for h in HOLD_DAYS}
    rules = {k: [] for k in RULES}
    by_year: dict[int, list[float]] = {}
    rows: list[dict] = []
    rng = np.random.default_rng(1)
    n_events = 0
    stocks_with = 0
    for t, c in closes.items():
        v = volumes.get(t)
        if t == bt.MARKET_TICKER or v is None or len(c) < 300 or v.sum() == 0:
            continue
        f = features(c, v)
        m = signal_mask(f).to_numpy()
        close = c.to_numpy(dtype=float)
        dates = c.index
        n = len(close)
        for d in rng.choice(np.arange(60, n - 1), size=min(8, n - 61), replace=False) if n > 61 else []:
            for h in HOLD_DAYS:
                if d + h < n:
                    base[h].append(close[d + h] / close[d] - 1)
        regs = None
        if up is not None:
            upv = up.reindex(dates, method="ffill").to_numpy()
            regs = [None if (x is None or (isinstance(x, float) and np.isnan(x))) else ("up" if x else "down") for x in upv]
        last = -10**9
        got = False
        for i in np.flatnonzero(m):
            if i - last < COOLDOWN or i + 1 >= n:
                continue
            last = i
            got = True
            e = i + 1
            entry = close[e]
            n_events += 1
            r = f.iloc[i]
            row = {"ratio": r["ratio"], "ret": r["ret"], "pos": r["pos"], "hi52": bool(r["hi52"]), "pre20": r["pre20"], "to": r["to"],
                   "regime": regs[i] if regs else None, "px": close[i], "month": dates[e].month, "year": dates[e].year,
                   "sector": (sectors or {}).get(t), "r20": np.nan, "rr": np.nan}
            for h in HOLD_DAYS:
                if e + h < n:
                    x = close[e + h] / entry - 1
                    hold[h].append(x)
                    if h == 20:
                        row["r20"] = x
                        by_year.setdefault(dates[e].year, []).append(x)
            if e + MAX_HOLD < n:
                path = close[e : e + MAX_HOLD + 1] / entry - 1
                for tp, sl in RULES:
                    res = bt._simulate(path, tp, sl, 0.0)
                    rules[(tp, sl)].append(res)
                    if (tp, sl) == (0.15, 0.08):
                        row["rr"] = res[0]
            rows.append(row)
        stocks_with += got

    rule_rows = []
    for (tp, sl), trades in rules.items():
        if not trades:
            continue
        rets = np.array([x for x, _ in trades])
        days = np.array([d for _, d in trades])
        rule_rows.append({"tp": tp, "sl": sl, "n": int(len(rets)), "mean": round(bt._robust_mean(rets), 4),
                          "win": round(float((rets > 0).mean()), 4), "avg_days": round(float(days.mean()), 1)})
    rule_rows.sort(key=lambda r: r["mean"], reverse=True)

    fa = bt._factor_analysis(rows, factors=VOL_FACTORS, ret_key="r20", rule_key="rr", horizon=20,
                             note="出来高急増の日を、その日に分かっていた条件で分けた20営業日後の成績")
    return {
        "events": n_events,
        "stocks_with_events": stocks_with,
        "rules_def": {"ratio": VS_RATIO, "min_ret": VS_MIN_RET, "min_turnover": VS_MIN_TURNOVER, "cooldown": COOLDOWN},
        "hold": [{"days": h, **bt._stats(hold[h]), "base_mean": bt._stats(base[h]).get("mean"), "base_win": bt._stats(base[h]).get("win")}
                 for h in HOLD_DAYS],
        "rules": rule_rows,
        "by_year": [{"year": y, **bt._stats(v)} for y, v in sorted(by_year.items())],
        "factors": fa,
    }


def summary_markdown(vb: dict | None) -> str:
    if not vb:
        return ""
    f = lambda v: "-" if v is None else f"{v * 100:+.1f}%"
    w = lambda v: "-" if v is None else f"{v * 100:.0f}%"
    lines = ["", "## 出来高急増のバックテスト",
             f"条件: 出来高が20日平均の{VS_RATIO:g}倍以上 & 当日+{VS_MIN_RET * 100:.0f}%以上 & 売買代金{VS_MIN_TURNOVER / 1e6:.0f}百万円以上 / シグナル {vb['events']} 回",
             "", "| 保有 | 件数 | 平均 | 中央値 | 勝率 | ランダム平均 | ランダム勝率 |", "|---|---|---|---|---|---|---|"]
    for r in vb["hold"]:
        lines.append(f"| {r['days']}日 | {r.get('n', 0)} | {f(r.get('mean'))} | {f(r.get('median'))} | {w(r.get('win'))} | {f(r.get('base_mean'))} | {w(r.get('base_win'))} |")
    lines += ["", "| 利確 | 損切り | 件数 | 平均 | 勝率 | 平均日数 |", "|---|---|---|---|---|---|"]
    for r in vb["rules"]:
        lines.append(f"| +{r['tp'] * 100:.0f}% | -{r['sl'] * 100:.0f}% | {r['n']} | {f(r['mean'])} | {w(r['win'])} | {r['avg_days']} |")
    fa = vb.get("factors")
    if fa:
        lines += ["", "| 条件 | 区分 | 件数 | 20日平均 | 勝率 | 前半 | 後半 | 安定 |", "|---|---|---|---|---|---|---|---|"]
        for fc in fa["factors"]:
            for x in fc["buckets"]:
                lines.append(f"| {fc['label']} | {x['label']} | {x['n']} | {f(x['mean60'])} | {w(x['win60'])} | {f(x['h1'])} | {f(x['h2'])} | {'◎' if x['stable'] else ''} |")
    return "\n".join(lines) + "\n"
