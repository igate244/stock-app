"""
イベント(自社株買い・株式分割・上方修正・増配)の簡易検出。

公式の適時開示(TDnet)は機械的な自動取得が禁止されているので、Googleニュースの検索結果(RSS)から
見出しを拾い、見出しに含まれる銘柄コード/銘柄名で銘柄を特定する。
  - 見出しの取りこぼし・誤判定はありうる(あくまで「気づくきっかけ」用)
  - 過去の履歴は公開中のページ(data/events.json)から毎回読み戻して積み上げる。
    たまってくると「イベント後に株価がどう動いたか」を集計できる
"""

from __future__ import annotations

import json
import os
import re
import unicodedata
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests

JST = timezone(timedelta(hours=9))
TYPES = {
    "buyback": {"label": "自社株買い", "q": ["自社株買い", "自己株式取得"]},
    "split": {"label": "株式分割", "q": ["株式分割"]},
    "upward": {"label": "上方修正", "q": ["上方修正"]},
    "dividend": {"label": "増配", "q": ["増配"]},
}
# 見出しに入っていたら別の意味になりやすい語(誤判定よけ)
NEGATIVE = {"buyback": ["終了", "結果"], "split": ["併合"], "upward": ["下方修正"], "dividend": ["減配"]}
KEEP_DAYS = 400
AFTER_DAYS = [1, 5, 20, 60]
_CODE_RE = re.compile(r"[<＜【(（\[\s]([0-9]{3}[0-9A-Z])[>＞】)）\]]")
_SUFFIXES = ["ホールディングス", "ＨＤ", "HD", "グループ", "株式会社", "(株)", "（株）", "製作所", "自動車"]


def _norm(s: str) -> str:
    return unicodedata.normalize("NFKC", s or "").replace(" ", "").replace("　", "")


def _name_variants(name: str) -> list[str]:
    n = _norm(name)
    out = {n}
    for suf in _SUFFIXES:
        suf = _norm(suf)
        if n.endswith(suf) and len(n) - len(suf) >= 2:
            out.add(n[: -len(suf)])
    return [v for v in out if len(v) >= 2]


def _fetch(query: str) -> list[dict]:
    url = f"https://news.google.com/rss/search?q={quote(query + ' when:7d')}&hl=ja&gl=JP&ceid=JP:ja"
    try:
        resp = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
    except (requests.RequestException, ET.ParseError) as exc:
        print(f"[events] 取得失敗 {query}: {exc}")
        return []
    items = []
    for it in root.iter("item"):
        title = (it.findtext("title") or "").strip()
        if not title:
            continue
        src = (it.findtext("source") or "").strip()
        if src and title.endswith(" - " + src):
            title = title[: -len(src) - 3]
        pub = None
        try:
            pub = parsedate_to_datetime(it.findtext("pubDate") or "")
        except (TypeError, ValueError):
            pass
        items.append({"title": title, "url": (it.findtext("link") or "").strip(), "src": src, "pub": pub})
    return items


def _match(title: str, codes: dict[str, str], names: list[tuple[str, str]]) -> str | None:
    """見出し → ティッカー。銘柄コード優先、なければ一番長く一致した銘柄名。"""
    for m in _CODE_RE.finditer(title):
        t = codes.get(m.group(1))
        if t:
            return t
    nt = _norm(title)
    best, best_len = None, 0
    for variant, t in names:
        if len(variant) > best_len and variant in nt:
            best, best_len = t, len(variant)
    return best


def _load_history() -> list[dict]:
    """公開中のページから、これまでに検出したイベントを読み戻す(なければ空)。"""
    url = os.environ.get("EVENTS_HISTORY_URL")
    if not url:
        repo = os.environ.get("GITHUB_REPOSITORY", "")
        if "/" in repo:
            owner, name = repo.split("/", 1)
            url = f"https://{owner}.github.io/{name}/data/events.json"
    if not url:
        return []
    try:
        resp = requests.get(url, timeout=20)
        if resp.status_code != 200:
            return []
        return list(resp.json().get("history", []))
    except (requests.RequestException, ValueError):
        return []


def collect(universe: pd.DataFrame, closes: dict[str, pd.Series]) -> dict:
    codes = {t.replace(".T", ""): t for t in universe["ticker"]}
    names = []
    for t, n in zip(universe["ticker"], universe["name"]):
        for v in _name_variants(str(n)):
            names.append((v, t))
    name_of = dict(zip(universe["ticker"], universe["name"].astype(str)))

    found: dict[str, dict] = {}
    for typ, spec in TYPES.items():
        for q in spec["q"]:
            for it in _fetch(q):
                if not any(k in it["title"] for k in spec["q"]):
                    continue
                if any(ng in it["title"] for ng in NEGATIVE.get(typ, [])):
                    continue
                t = _match(it["title"], codes, names)
                if not t:
                    continue
                d = (it["pub"] or datetime.now(timezone.utc)).astimezone(JST).strftime("%Y-%m-%d")
                key = f"{t}|{typ}|{d}"
                if key not in found:
                    found[key] = {"t": t, "n": unicodedata.normalize("NFKC", name_of.get(t, t)), "type": typ,
                                  "date": d, "title": it["title"], "url": it["url"], "src": it["src"]}
    print(f"[events] 今回の検出 {len(found)} 件")

    # 履歴とマージ(同じ銘柄・同じ種類で7日以内のものは同じイベントとみなす)
    history = _load_history()
    merged: dict[str, dict] = {}
    for e in history + list(found.values()):
        k = f"{e['t']}|{e['type']}|{e['date']}"
        merged.setdefault(k, e)
    events = sorted(merged.values(), key=lambda e: e["date"])
    dedup: list[dict] = []
    last_seen: dict[str, str] = {}
    for e in events:
        k = f"{e['t']}|{e['type']}"
        if k in last_seen and (pd.Timestamp(e["date"]) - pd.Timestamp(last_seen[k])).days <= 7:
            continue
        last_seen[k] = e["date"]
        dedup.append(e)
    cutoff = (datetime.now(JST) - timedelta(days=KEEP_DAYS)).strftime("%Y-%m-%d")
    events = [e for e in dedup if e["date"] >= cutoff]

    # イベント後の値動き(イベント日の翌営業日の終値で買ったとして)
    for e in events:
        c = closes.get(e["t"])
        e.pop("after", None)
        if c is None or c.empty:
            continue
        idx = c.index.searchsorted(pd.Timestamp(e["date"]), side="right")  # 翌営業日
        if idx >= len(c):
            continue
        entry = float(c.iloc[idx])
        e["entry"] = round(entry, 2)
        e["now"] = round(float(c.iloc[-1] / entry - 1), 4)
        e["after"] = {str(h): round(float(c.iloc[idx + h] / entry - 1), 4) for h in AFTER_DAYS if idx + h < len(c)}

    stats = []
    for typ, spec in TYPES.items():
        row = {"type": typ, "label": spec["label"], "n": sum(1 for e in events if e["type"] == typ)}
        for h in AFTER_DAYS:
            xs = np.array([e["after"][str(h)] for e in events if e["type"] == typ and str(h) in (e.get("after") or {})])
            if len(xs) >= 5:
                row[f"d{h}"] = {"n": int(len(xs)), "mean": round(float(xs.mean()), 4), "med": round(float(np.median(xs)), 4),
                                "win": round(float((xs > 0).mean()), 4)}
        stats.append(row)
    recent_cut = (datetime.now(JST) - timedelta(days=14)).strftime("%Y-%m-%d")
    return {
        "types": {k: v["label"] for k, v in TYPES.items()},
        "recent": [e for e in events if e["date"] >= recent_cut][::-1],
        "stats": stats,
        "history": events,
        "since": events[0]["date"] if events else None,
    }
