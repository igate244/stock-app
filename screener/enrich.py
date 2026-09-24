"""
底打ち候補まで残った銘柄だけに行う追加情報の収集(STEP4 業績 / STEP6 ニュース / STEP7 AI評価)。
1銘柄ずつ外部に問い合わせるので、全銘柄ではなく候補だけに絞って実行する。
"""

from __future__ import annotations

import os
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import quote

import pandas as pd
import requests
import yfinance as yf

# ---- STEP4: 業績 ---------------------------------------------------------------
_REVENUE_ROWS = ["Total Revenue", "TotalRevenue"]
_OP_ROWS = ["Operating Income", "OperatingIncome"]
_NET_ROWS = ["Net Income", "NetIncome", "Net Income Common Stockholders"]


def _row(df: pd.DataFrame, names: list[str]) -> pd.Series | None:
    for n in names:
        if n in df.index:
            r = df.loc[n]
            return r.iloc[0] if isinstance(r, pd.DataFrame) else r
    return None


def _growth(s: pd.Series | None, back: int) -> float | None:
    if s is None or len(s) <= back:
        return None
    latest, prior = s.iloc[-1], s.iloc[-1 - back]
    if pd.isna(latest) or pd.isna(prior) or prior == 0:
        return None
    return round(float((latest - prior) / abs(prior)), 4)


def earnings(ticker: str) -> dict:
    """
    売上・利益(営業利益優先、なければ純利益)のQoQ/YoY。
    yfinanceは日本の中小型株だと欠損が多いので、取れなければ has_data=False(判断材料なし)。
    フィルタには使わず「証拠」として画面とAIに渡す。
    """
    try:
        df = yf.Ticker(ticker).quarterly_financials
    except Exception:  # noqa: BLE001
        df = None
    if df is None or df.empty:
        return {"has_data": False, "trend": "データ不足"}

    df = df.reindex(sorted(df.columns), axis=1)  # 古い順に
    rev = _row(df, _REVENUE_ROWS)
    profit, metric = _row(df, _OP_ROWS), "営業利益"
    if profit is None:
        profit, metric = _row(df, _NET_ROWS), "純利益"
    if rev is None and profit is None:
        return {"has_data": False, "trend": "データ不足"}

    p_yoy, p_qoq = _growth(profit, 4), _growth(profit, 1)
    if p_yoy is not None and p_qoq is not None:
        trend = "改善" if (p_yoy > 0 and p_qoq > 0) else "悪化" if (p_yoy < 0 and p_qoq < 0) else "横ばい"
    elif p_qoq is not None:
        trend = "改善" if p_qoq > 0 else "悪化" if p_qoq < 0 else "横ばい"
    else:
        trend = "データ不足"

    return {
        "has_data": True,
        "trend": trend,
        "metric": metric if profit is not None else None,
        "rev_yoy": _growth(rev, 4),
        "rev_qoq": _growth(rev, 1),
        "profit_yoy": p_yoy,
        "profit_qoq": p_qoq,
    }


# ---- STEP6: ニュース(Google News RSS・無料) ----------------------------------------
def news(name: str, lookback_days: int = 14, max_items: int = 6) -> list[dict]:
    """銘柄名で検索した直近ニュースの見出し。ポジネガの判断はしない(AI/自分で読む)。"""
    url = f"https://news.google.com/rss/search?q={quote(name + ' 株価')}&hl=ja&gl=JP&ceid=JP:ja"
    try:
        resp = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
    except (requests.RequestException, ET.ParseError):
        return []

    cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    items: list[dict] = []
    for it in root.iter("item"):
        title = (it.findtext("title") or "").strip()
        if not title:
            continue
        published = None
        raw = it.findtext("pubDate")
        if raw:
            try:
                published = parsedate_to_datetime(raw)
            except (TypeError, ValueError):
                published = None
        if published is not None and published < cutoff:
            continue
        items.append(
            {
                "d": published.astimezone(timezone(timedelta(hours=9))).strftime("%m/%d") if published else "",
                "t": title,
                "u": (it.findtext("link") or "").strip(),
            }
        )
        if len(items) >= max_items:
            break
    return items


# ---- STEP7: AI最終評価(任意・APIキーがある時だけ) -------------------------------
_SYSTEM = """あなたは個人投資家の日本株スクリーニングを手伝うアナリストです。
このユーザーが過去に利益を出した『勝ちパターン』銘柄(三菱電機、オムロン、キーエンス、
ABEJA、Tier IV、円谷フィールズ等)に似た値動き・業績パターンを持つ銘柄を探すのが目的です。
候補銘柄の証拠(株価位置、底打ち、業績、類似度、直近ニュース)を踏まえ、0.0〜1.0のスコアと
日本語2〜3文のコメント(良い点と懸念点の両方)をsubmit_evaluationツールで提出してください。
投資助言ではなくスクリーニングの参考情報なので、断定的な買い/売り表現は避けてください。"""

_TOOL = {
    "name": "submit_evaluation",
    "description": "候補銘柄の最終評価を提出する",
    "input_schema": {
        "type": "object",
        "properties": {
            "score": {"type": "number", "minimum": 0, "maximum": 1},
            "comment": {"type": "string"},
        },
        "required": ["score", "comment"],
    },
}


def evidence_text(stock: dict) -> str:
    e = stock.get("earn") or {}
    earn_txt = "データなし" if not e.get("has_data") else (
        f"トレンド{e.get('trend')} / 売上YoY {e.get('rev_yoy')} / {e.get('metric')}YoY {e.get('profit_yoy')} / QoQ {e.get('profit_qoq')}"
    )
    news_txt = "\n".join(f"[{n['d']}] {n['t']}" for n in stock.get("news") or []) or "なし"
    return (
        f"銘柄: {stock['n']} ({stock['t']}) / 業種: {stock['s']}\n"
        f"5年株価位置: {stock['pos']:.0%}(0%=5年最安値)\n"
        f"3ヶ月: 下落{stock['dec']:.0%}→底から反発{stock['reb']:.0%}(底から{stock['dsl']}営業日)\n"
        f"業績: {earn_txt}\n"
        f"勝ちパターン類似度: {stock.get('sim', 0):.2f}(最も近い: {stock.get('sim_ref') or '-'})\n"
        f"直近ニュース:\n{news_txt}\n"
    )


def ai_evaluate(stocks: list[dict], limit: int = 20) -> None:
    """ANTHROPIC_API_KEY がある時だけ、類似度上位limit件にAIスコアとコメントを付ける(従量課金)。"""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key or not stocks:
        print("[ai] APIキー未設定のためAI評価はスキップ(スマホ画面の「Claudeに聞く」で代用可)")
        return
    import anthropic

    client = anthropic.Anthropic(api_key=key)
    model = os.environ.get("ANTHROPIC_MODEL") or "claude-sonnet-4-5"
    for s in sorted(stocks, key=lambda x: x.get("sim", 0), reverse=True)[:limit]:
        try:
            msg = client.messages.create(
                model=model,
                max_tokens=512,
                system=_SYSTEM,
                tools=[_TOOL],
                tool_choice={"type": "tool", "name": "submit_evaluation"},
                messages=[{"role": "user", "content": evidence_text(s)}],
            )
            for block in msg.content:
                if block.type == "tool_use":
                    s["ai"] = {"score": round(float(block.input["score"]), 3), "comment": str(block.input["comment"])}
        except Exception as exc:  # noqa: BLE001 — 1件の失敗で全体を止めない
            print(f"[ai] {s['t']} 失敗: {exc}")
        time.sleep(0.5)
