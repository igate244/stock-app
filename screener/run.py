"""
毎日の自動スクリーニング本体(GitHub Actionsから実行)。

全銘柄について STEP2(5年株価位置)・STEP3(3ヶ月底打ち)を計算し、
底打ち候補まで残った銘柄だけ STEP4(業績)・STEP5(類似度)・STEP6(ニュース)・STEP7(AI, 任意)を追加。
結果を1つのJSON(results.json)に書き出し、スマホ画面(web/index.html)がそれを読む。

テーマ選別(STEP1)はスマホ側で即時に行う(全銘柄分の結果を持っているので、
テーマを切り替えても再計算は不要)。

使い方:
    python -m screener.run --out site/data
    python -m screener.run --out site/data --limit 200   # 動作確認用に銘柄数を絞る
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import yaml

from screener import data, enrich, signals
from screener.themes import JPX_33_SECTORS, THEME_GENRES, tag_themes

CONFIG = Path(__file__).resolve().parent.parent / "config" / "win_patterns.yaml"
JST = timezone(timedelta(hours=9))


def _r(x: float | None, nd: int = 3) -> float | None:
    return None if x is None else round(float(x), nd)


def _series_payload(close: pd.Series) -> dict:
    """グラフ用: 基準日 + 日数オフセット + 値(有効数字4桁)で軽量化。"""
    base = close.index[0]
    return {
        "b": base.strftime("%Y-%m-%d"),
        "x": [int((d - base).days) for d in close.index],
        "y": [float(f"{v:.4g}") for v in close.to_numpy()],
    }


def load_references(closes: dict[str, pd.Series]) -> list[dict]:
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8")) or {}
    refs = []
    for wp in cfg.get("winning_stocks", []):
        t = (wp.get("ticker") or "").strip()
        if not t or t not in closes:
            continue
        low = signals.reference_low_date(closes[t], wp.get("pattern_start") or None)
        if low is not None:
            refs.append({"name": wp.get("name") or t, "ticker": t, "close": closes[t], "low": low})
    print(f"[refs] 勝ちパターン比較対象: {[r['name'] for r in refs]}")
    return refs


def screen(limit: int | None = None) -> dict:
    started = time.time()
    universe = data.load_universe()
    if limit:
        universe = universe.head(limit)

    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8")) or {}
    ref_tickers = [(wp.get("ticker") or "").strip() for wp in cfg.get("winning_stocks", [])]
    tickers = list(dict.fromkeys(universe["ticker"].tolist() + [t for t in ref_tickers if t]))

    closes = data.download_closes(tickers)
    refs = load_references(closes)

    stocks: list[dict] = []
    candidates: list[tuple[dict, pd.Series, signals.BottomSignal]] = []

    for row in universe.itertuples(index=False):
        base = {"t": row.ticker, "n": str(row.name), "s": str(row.sector), "th": tag_themes(str(row.name))}
        close = closes.get(row.ticker)
        if close is None:
            stocks.append({**base, "st": "nodata"})
            continue

        pos = signals.price_position(close)
        bottom = signals.detect_bottom(close)
        rec = {
            **base,
            "px": float(f"{close.iloc[-1]:.5g}"),
            "pos": _r(pos.position),
            "yrs": _r(pos.years_of_data, 1),
            "dec": _r(bottom.decline_pct),
            "reb": _r(bottom.rebound_pct),
            "dsl": bottom.days_since_low,
        }
        if not pos.near_low:
            rec["st"] = "high"
        elif not bottom.bottomed:
            rec["st"] = "nobottom"
        else:
            rec["st"] = "pass"
            candidates.append((rec, close, bottom))
        stocks.append(rec)

    print(f"[screen] 底打ち候補: {len(candidates)} 銘柄 → 類似度・業績・ニュースを収集")

    for i, (rec, close, bottom) in enumerate(candidates, 1):
        # STEP5 類似度(最も似ている勝ちパターン銘柄を記録)
        best, best_ref = 0.0, None
        for ref in refs:
            if ref["ticker"] == rec["t"]:
                continue
            sc = signals.similarity_score(close, bottom.low_date, ref["close"], ref["low"])
            if sc > best:
                best, best_ref = sc, ref
        rec["sim"] = _r(best)
        rec["sim_ref"] = best_ref["name"] if best_ref else None
        if best_ref is not None:
            a = close.loc[bottom.low_date:].iloc[: signals.SIMILARITY_WINDOW]
            b = best_ref["close"].loc[best_ref["low"]:].iloc[: signals.SIMILARITY_WINDOW]
            rec["cmp"] = {
                "a": [round(float(v / a.iloc[0] * 100), 1) for v in a],
                "b": [round(float(v / b.iloc[0] * 100), 1) for v in b],
                "b_from": best_ref["low"].strftime("%Y-%m-%d"),
            }

        # STEP4 業績 / STEP6 ニュース
        rec["earn"] = enrich.earnings(rec["t"])
        rec["news"] = enrich.news(rec["n"])

        # グラフ用データ(5年=週足、3ヶ月=日足)
        # 週足: 各週の最後の取引日の終値(日付は実際の取引日のまま)
        weekly = close.groupby(close.index.to_period("W-FRI")).tail(1)
        rec["ch"] = {"w": _series_payload(weekly), "d": _series_payload(close.tail(signals.BOTTOM_LOOKBACK_DAYS))}
        rec["low_d"] = bottom.low_date.strftime("%Y-%m-%d")

        if i % 10 == 0:
            print(f"[enrich] {i}/{len(candidates)}")
        time.sleep(0.3)

    # STEP7 AI(APIキーがある時だけ)
    enrich.ai_evaluate([c[0] for c in candidates])

    return {
        "generated_at": datetime.now(JST).isoformat(timespec="minutes"),
        "elapsed_min": round((time.time() - started) / 60, 1),
        "universe_count": len(universe),
        "params": {
            "near_low": signals.NEAR_LOW_THRESHOLD,
            "min_decline": signals.MIN_DECLINE_PCT,
            "min_rebound": signals.MIN_REBOUND_PCT,
            "lookback_days": signals.BOTTOM_LOOKBACK_DAYS,
        },
        "genres": THEME_GENRES,
        "sectors": JPX_33_SECTORS,
        "refs": [r["name"] for r in refs],
        "stocks": stocks,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="site/data")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    result = screen(args.limit)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(result, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    n_pass = sum(1 for s in result["stocks"] if s["st"] == "pass")
    print(f"[done] {len(result['stocks'])} 銘柄 / 候補 {n_pass} / {result['elapsed_min']}分 → {out / 'results.json'}")


if __name__ == "__main__":
    main()
