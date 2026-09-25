"""
配当: 利回り・配当の時期・次の「権利付最終日」の目安と、配当前後の値動きの統計。

日本株の配当のもらい方(2019年7月以降):
  - 権利確定日(多くは月末)の2営業日前 = 「権利付最終日」。この日の大引けに株を持っていれば配当がもらえる
  - 翌営業日 = 「権利落ち日」。この日に売っても配当はもらえる(そのぶん株価は下がりやすい)
ここでは過去の権利落ち日の月から「次の権利確定月」を推定し、月末を権利確定日として日付を計算する
(20日締めなど月末以外の会社もあるので、正確な日付は会社の発表で確認)。

データ: Yahoo Finance の配当履歴(権利落ち日ごとの1株配当)と、配当調整なしの終値。
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

try:
    import jpholiday
except ImportError:  # 祝日ライブラリが無くても土日だけで計算する
    jpholiday = None

UPCOMING_DAYS = 60  # 画面の「もうすぐ権利付最終日」は60日先まで
PRE_DAYS = [5, 10, 20, 40]
POST_DAYS = [5, 20, 60]


# ---- 営業日の計算 -----------------------------------------------------------------
def is_business_day(d: date) -> bool:
    if d.weekday() >= 5:
        return False
    if (d.month == 12 and d.day == 31) or (d.month == 1 and d.day <= 3):  # 年末年始は休場
        return False
    if jpholiday is not None and jpholiday.is_holiday(d):
        return False
    return True


def add_business_days(d: date, n: int) -> date:
    step = 1 if n > 0 else -1
    while n:
        d += timedelta(days=step)
        if is_business_day(d):
            n -= step
    return d


def rights_dates(year: int, month: int) -> tuple[date, date, date]:
    """月末を権利確定日としたときの (権利付最終日, 権利落ち日, 権利確定日)。"""
    last = (date(year + (month == 12), month % 12 + 1, 1) - timedelta(days=1))
    rec = last
    while not is_business_day(rec):
        rec -= timedelta(days=1)
    return add_business_days(rec, -2), add_business_days(rec, -1), last


def next_rights(months: list[int], today: date) -> tuple[date, date, int] | None:
    """配当月のリストから、今日以降で一番近い (権利付最終日, 権利落ち日, 月)。"""
    if not months:
        return None
    for k in range(0, 14):
        y, m = today.year + (today.month - 1 + k) // 12, (today.month - 1 + k) % 12 + 1
        if m in months:
            cum, ex, _ = rights_dates(y, m)
            if cum >= today:
                return cum, ex, m
    return None


# ---- 銘柄ごとの配当情報 -------------------------------------------------------------
def info(raw_close: pd.Series | None, divs: pd.Series | None, today: date) -> dict | None:
    if divs is None or divs.empty or raw_close is None or raw_close.dropna().empty:
        return None
    last_date = divs.index[-1]
    if (pd.Timestamp(today) - last_date).days > 450:  # 1年以上配当がない=無配になった可能性
        return None
    px = float(raw_close.dropna().iloc[-1])
    ttm = float(divs[divs.index > pd.Timestamp(today) - pd.Timedelta(days=365)].sum())
    recent = divs[divs.index > pd.Timestamp(today) - pd.Timedelta(days=730)]
    months = sorted({int(d.month) for d in recent.index})
    per_year = divs.groupby(divs.index.year).sum()
    this_year = pd.Timestamp(today).year
    full = per_year[per_year.index < this_year].tail(4)  # 1年分そろっている年だけで増配/減配を見る
    trend = None
    if len(full) >= 3:
        diffs = np.diff(full.to_numpy())
        trend = "増配傾向" if (diffs >= -1e-9).all() and diffs.sum() > 0 else "減配あり" if (diffs < -1e-9).any() else "横ばい"
    nx = next_rights(months, today)
    return {
        "y": round(ttm / px, 4) if px > 0 else None,         # 配当利回り(直近1年の実績ベース)
        "ttm": round(ttm, 2),                                # 直近1年の1株配当
        "m": months,                                         # 配当の月(権利落ちの月)
        "amt": round(float(divs.iloc[-1]), 2),               # 前回の1株配当
        "last": last_date.strftime("%Y-%m-%d"),
        "tr": trend,
        "cum": nx[0].isoformat() if nx else None,            # 次の権利付最終日(推定)
        "ex": nx[1].isoformat() if nx else None,             # 次の権利落ち日(推定)
        "hist": [[d.strftime("%Y-%m-%d"), round(float(v), 2)] for d, v in divs.tail(8).items()],
    }


# ---- 配当前後の値動きの統計(過去10年) ------------------------------------------------
YIELD_BUCKETS = [(0, 0.01, "1%未満"), (0.01, 0.02, "1〜2%"), (0.02, 0.03, "2〜3%"), (0.03, 0.05, "3〜5%"), (0.05, 9, "5%以上")]


def _st(a: list[float]) -> dict:
    x = np.asarray(a, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 20:
        return {"n": int(len(x))}
    lo, hi = np.percentile(x, [1, 99])
    return {"n": int(len(x)), "mean": round(float(np.clip(x, lo, hi).mean()), 4),
            "med": round(float(np.median(x)), 4), "win": round(float((x > 0).mean()), 4)}


def run_backtest(raw: dict[str, pd.Series], divs: dict[str, pd.Series]) -> dict | None:
    """
    権利落ち日ごとに(配当調整なしの終値で):
      - 権利取り: 権利付最終日のk営業日前に買って、権利付最終日の終値で売る(配当はもらわない)
      - 配当取り: k営業日前に買って、権利落ち日の終値で売る(配当をもらう。税金は考えない)
      - 落ちてから買う: 権利落ち日の終値で買って、m営業日後に売る
    比較用に、同じ銘柄の適当な日にk営業日持った場合も出す。
    """
    if not divs:
        return None
    rng = np.random.default_rng(2)
    pre = {k: [] for k in PRE_DAYS}
    cap = {k: [] for k in PRE_DAYS}
    base = {k: [] for k in PRE_DAYS}
    post = {m: [] for m in POST_DAYS}
    drop, drop_ratio = [], []
    by_y = {lab: {"pre20": [], "cap20": [], "post20": []} for *_, lab in YIELD_BUCKETS}
    by_m = {m: [] for m in range(1, 13)}
    n_ev = 0
    for t, d in divs.items():
        c = raw.get(t)
        if c is None:
            continue
        c = c.dropna()
        if len(c) < 120:
            continue
        v = c.to_numpy(dtype=float)
        idx = c.index
        n = len(v)
        for dd in rng.choice(np.arange(45, n - 1), size=min(6, n - 46), replace=False) if n > 46 else []:
            for k in PRE_DAYS:
                base[k].append(v[dd] / v[dd - k] - 1)
        for ex_date, amt in d.items():
            e = int(idx.searchsorted(ex_date))  # 権利落ち日(以降で最初の取引日)
            if e < 41 or e >= n or amt <= 0:
                continue
            cum = e - 1  # 権利付最終日
            if abs((idx[e] - ex_date).days) > 4:
                continue
            y = amt / v[cum]
            if not (0 < y < 0.2):  # 異常値(分割の未調整など)は除外
                continue
            n_ev += 1
            for k in PRE_DAYS:
                pre[k].append(v[cum] / v[cum - k] - 1)
                cap[k].append((v[e] + amt) / v[cum - k] - 1)
            drop.append(v[e] / v[cum] - 1)
            drop_ratio.append((v[cum] - v[e]) / amt)
            for m in POST_DAYS:
                if e + m < n:
                    post[m].append(v[e + m] / v[e] - 1)
            # 年利回りの目安 = 1回の配当 × その時点までの1年間の配当回数
            cnt = int(((d.index > ex_date - pd.Timedelta(days=365)) & (d.index <= ex_date)).sum()) or 1
            ya = y * cnt
            lab = next((lab for lo, hi, lab in YIELD_BUCKETS if lo <= ya < hi), None)
            if lab:
                by_y[lab]["pre20"].append(v[cum] / v[cum - 20] - 1)
                by_y[lab]["cap20"].append((v[e] + amt) / v[cum - 20] - 1)
                if e + 20 < n:
                    by_y[lab]["post20"].append(v[e + 20] / v[e] - 1)
            by_m[int(idx[cum].month)].append(v[cum] / v[cum - 20] - 1)

    if not n_ev:
        return None
    return {
        "events": n_ev,
        "pre": [{"days": k, "kenri": _st(pre[k]), "haito": _st(cap[k]), "base": _st(base[k])} for k in PRE_DAYS],
        "post": [{"days": m, **_st(post[m])} for m in POST_DAYS],
        "drop": {**_st(drop), "ratio_med": round(float(np.median(drop_ratio)), 2) if drop_ratio else None},
        "by_yield": [{"label": lab, "pre20": _st(v["pre20"]), "cap20": _st(v["cap20"]), "post20": _st(v["post20"])} for lab, v in by_y.items()],
        "by_month": [{"month": m, **_st(v)} for m, v in by_m.items() if len(v) >= 20],
        "notes": [
            "配当調整なしの終値で計算。税金(約20%)と手数料は含まない",
            "権利落ち日はYahoo Financeの配当履歴の日付",
            "現在上場している銘柄だけ(上場廃止した銘柄は含まない)",
        ],
    }


def summary_markdown(db: dict | None) -> str:
    if not db:
        return ""
    f = lambda v: "-" if v is None else f"{v * 100:+.2f}%"
    w = lambda v: "-" if v is None else f"{v * 100:.0f}%"
    lines = ["", "## 配当前後の値動き", f"権利落ち {db['events']} 回",
             "", "| 何営業日前に買う | 権利取り(最終日に売る) | 配当取り(落ち日に売る・配当込み) | 適当な日に同じ日数 |", "|---|---|---|---|"]
    for r in db["pre"]:
        k, h, b = r["kenri"], r["haito"], r["base"]
        lines.append(f"| {r['days']}日前 | {f(k.get('mean'))} / {w(k.get('win'))} | {f(h.get('mean'))} / {w(h.get('win'))} | {f(b.get('mean'))} / {w(b.get('win'))} |")
    dr = db["drop"]
    lines += ["", f"権利落ち日の値下がり: 平均 {f(dr.get('mean'))}(配当額の{dr.get('ratio_med')}倍くらい下がる・中央値)",
              "", "| 落ちてから買って | 平均 | 勝率 |", "|---|---|---|"]
    for r in db["post"]:
        lines.append(f"| {r['days']}日後 | {f(r.get('mean'))} | {w(r.get('win'))} |")
    lines += ["", "| 年利回り | 権利取り20日 | 配当取り20日 | 落ち後20日 |", "|---|---|---|---|"]
    for r in db["by_yield"]:
        lines.append(f"| {r['label']} | {f(r['pre20'].get('mean'))} | {f(r['cap20'].get('mean'))} | {f(r['post20'].get('mean'))} |")
    return "\n".join(lines) + "\n"
