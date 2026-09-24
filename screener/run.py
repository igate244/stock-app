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
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import yaml

from screener import backtest, data, enrich, signals
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


EARLY_TOP = 40  # 反発前の銘柄のうちグラフを付ける数
EARLY_MIN_SIM = 0.85


def _charts(close: pd.Series) -> dict:
    # 5年=週足(各週の最後の取引日の終値)、3ヶ月=日足
    weekly = close.groupby(close.index.to_period("W-FRI")).tail(1)
    return {"w": _series_payload(weekly), "d": _series_payload(close.tail(signals.BOTTOM_LOOKBACK_DAYS))}


def load_references(closes: dict[str, pd.Series], closes_full: dict[str, pd.Series] | None = None) -> list[dict]:
    """
    勝ちパターン銘柄。底の日は直近5年(closes)で決め、「上がる前の形」は底より前のデータも要るので
    取れていれば10年分(closes_full)から切り出す。
    """
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8")) or {}
    refs = []
    for wp in cfg.get("winning_stocks", []):
        t = (wp.get("ticker") or "").strip()
        if not t or t not in closes:
            continue
        low = signals.reference_low_date(closes[t], wp.get("pattern_start") or None)
        if low is None:
            continue
        name = wp.get("name") or t
        full = (closes_full or {}).get(t, closes[t])
        refs.append({"name": name, "ticker": t, "close": closes[t], "full": full, "low": low,
                     "templates": signals.pre_templates(full, low, name)})
    print(f"[refs] 勝ちパターン比較対象: {[(r['name'], r['low'].strftime('%Y-%m-%d'), len(r['templates'])) for r in refs]}")
    return refs


def _templates_for(refs: list[dict], ticker: str) -> list[dict]:
    return [tp for r in refs if r["ticker"] != ticker for tp in r["templates"]]


def _cmp_payload(close: pd.Series, tpl: dict) -> dict:
    """比較グラフ用: 今日=100 にそろえた、この銘柄の直近60日 と 勝ちパターンの同じ形の区間+その後。"""
    W = signals.PRE_WINDOW
    a = close.iloc[-W:]
    ref = tpl["close"]
    end = tpl["end"]
    b = ref.iloc[end - W + 1 : end + 1 + signals.PRE_FUTURE]
    a0, b0 = float(a.iloc[-1]), float(ref.iloc[end])
    return {
        "a": [round(float(v) / a0 * 100, 1) for v in a],
        "b": [round(float(v) / b0 * 100, 1) for v in b],
        "b_at": ref.index[end].strftime("%Y-%m-%d"),
        "k": tpl["k"],
    }


def screen(limit: int | None = None) -> tuple[dict[str, pd.Series], list[dict], dict]:
    started = time.time()
    universe = data.load_universe()
    if limit:
        universe = universe.head(limit)

    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8")) or {}
    ref_tickers = [(wp.get("ticker") or "").strip() for wp in cfg.get("winning_stocks", [])]
    tickers = list(dict.fromkeys(universe["ticker"].tolist() + [t for t in ref_tickers if t] + [backtest.MARKET_TICKER]))

    # バックテスト用に10年分取得し、毎日の判定(5年位置など)には直近5年分だけを使う
    closes_full = data.download_closes(tickers, period="10y")
    closes = {}
    for t, ser in closes_full.items():
        cut = ser.index[-1] - pd.DateOffset(years=5)
        closes[t] = ser[ser.index >= cut]
    refs = load_references(closes, closes_full)

    stocks: list[dict] = []
    early: list[tuple] = []
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
        else:
            # STEP5 類似度(上がる前の形で比較)。底値圏の銘柄は全部計算する(まだ反発前の早い段階の銘柄も見つけるため)
            sim, tpl = signals.pre_similarity(close, _templates_for(refs, row.ticker))
            if tpl is not None:
                rec["sim"] = _r(sim)
                rec["sim_ref"] = tpl["name"]
            if not bottom.bottomed:
                rec["st"] = "nobottom"
                if tpl is not None:
                    early.append((sim, rec, close, tpl))
            else:
                rec["st"] = "pass"
                candidates.append((rec, close, bottom))
                if tpl is not None:
                    rec["cmp"] = _cmp_payload(close, tpl)
        stocks.append(rec)

    # まだ反発前でも形がよく似ている銘柄(上位のみ)は、グラフを見られるようにしておく
    early.sort(key=lambda x: x[0], reverse=True)
    for sim, rec, close, tpl in early[:EARLY_TOP]:
        if sim < EARLY_MIN_SIM:
            break
        rec["cmp"] = _cmp_payload(close, tpl)
        rec["ch"] = _charts(close)

    print(f"[screen] 底打ち候補: {len(candidates)} 銘柄 → 類似度・業績・ニュースを収集")

    for i, (rec, close, bottom) in enumerate(candidates, 1):
        # STEP4 業績 / STEP6 ニュース
        rec["earn"] = enrich.earnings(rec["t"])
        rec["news"] = enrich.news(rec["n"])

        # グラフ用データ(5年=週足、3ヶ月=日足)
        # 週足: 各週の最後の取引日の終値(日付は実際の取引日のまま)
        rec["ch"] = _charts(close)
        rec["low_d"] = bottom.low_date.strftime("%Y-%m-%d")

        if i % 10 == 0:
            print(f"[enrich] {i}/{len(candidates)}")
        time.sleep(0.3)

    # 今の地合い(日本株全体の指数が200日移動平均より上か下か)。バックテストと同じ指数・同じ判定
    market = None
    try:
        market = backtest.market_now(backtest.market_index(closes_full))
        if market:
            print(f"[market] {'上向き' if market['up'] else '下向き'}(200日線比 {market['gap'] * 100:+.1f}%)")
    except Exception as exc:  # noqa: BLE001
        print(f"[market] 計算失敗: {exc}")

    # STEP7 AI(APIキーがある時だけ)
    enrich.ai_evaluate([c[0] for c in candidates])

    return closes_full, refs, {
        "generated_at": datetime.now(JST).isoformat(timespec="minutes"),
        "elapsed_min": round((time.time() - started) / 60, 1),
        "universe_count": len(universe),
        "params": {
            "near_low": signals.NEAR_LOW_THRESHOLD,
            "min_decline": signals.MIN_DECLINE_PCT,
            "min_rebound": signals.MIN_REBOUND_PCT,
            "lookback_days": signals.BOTTOM_LOOKBACK_DAYS,
            "sim_window": signals.PRE_WINDOW,
            "sim_future": signals.PRE_FUTURE,
            "sim_method": "pre",
        },
        "genres": THEME_GENRES,
        "sectors": JPX_33_SECTORS,
        "market": market,
        "refs": [r["name"] for r in refs],
        "ref_lows": {r["name"]: r["low"].strftime("%Y-%m-%d") for r in refs},
        "stocks": stocks,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="site/data")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    closes, refs, result = screen(args.limit)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(result, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    n_pass = sum(1 for s in result["stocks"] if s["st"] == "pass")
    print(f"[done] {len(result['stocks'])} 銘柄 / 候補 {n_pass} / {result['elapsed_min']}分 → {out / 'results.json'}")

    # バックテスト(失敗しても毎日のスクリーニング結果の公開は止めない)
    try:
        t0 = time.time()
        bt = backtest.run(closes, refs)
        (out / "backtest.json").write_text(json.dumps(bt, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        md = backtest.summary_markdown(bt)
        print(md)
        summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
        if summary_path:
            with open(summary_path, "a", encoding="utf-8") as fh:
                fh.write(md)
        print(f"[backtest] シグナル {bt['events']} 回 / {round(time.time() - t0)}秒")
    except Exception as exc:  # noqa: BLE001
        print(f"[backtest] 失敗: {exc}")


if __name__ == "__main__":
    main()
