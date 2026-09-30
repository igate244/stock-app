"""
材料チェック(無料でできる範囲)。

  1. 足切り(risk): 危ない材料がある銘柄に印を付ける。画面では初期状態で一覧から外す
       - JPXの 監理銘柄・整理銘柄 / 特別注意銘柄 / 上場維持基準の改善期間・猶予期間
       - 株価100円未満、売買代金がほとんどない(20日平均1千万円未満)
       - 債務超過、直近4四半期ずっと赤字
       - 株を大量に刷る資金調達(MSワラント・第三者割当など)や不祥事のニュース見出し
  2. 材料スコア(score): 良い材料を足して悪い材料を引いた点数
       決算(増収・増益・黒字)、上方修正・増配・自社株買い、配当、過去の統計で効いていた形 など

注意: 決算データは直近1年分くらいしか取れないので、このスコアで絞った場合の過去10年の成績は検証できない。
あくまで「危ないものを避ける安全装置」。
"""

from __future__ import annotations

import re

import requests

JPX_ALERT_PAGES = {
    "https://www.jpx.co.jp/listing/market-alerts/supervision/index.html": "監理・整理銘柄",
    "https://www.jpx.co.jp/listing/measures/alert/index.html": "特別注意銘柄",
    "https://www.jpx.co.jp/listing/market-alerts/improvement-period/index.html": "上場維持基準の改善期間中",
    "https://www.jpx.co.jp/listing/market-alerts/grace-period/index.html": "上場維持基準の猶予期間中",
}
MIN_PRICE = 100
MIN_TURNOVER = 10_000_000  # 20日平均の売買代金(円)

# ニュース見出しのキーワード(銘柄名で検索した見出し+イベント検出の見出し)
NEG_NEWS = {
    "株の希薄化(MSワラント・増資など)": ["新株予約権", "MSワラント", "行使価額修正", "第三者割当", "公募増資", "新株式発行"],
    "不祥事・上場維持の懸念": ["不正", "不適切", "調査委員会", "決算発表の延期", "発表延期", "上場廃止", "特別注意", "監理銘柄",
                     "継続企業の前提", "疑義注記", "債務超過", "行政処分", "業務停止", "粉飾"],
    "下方修正・減配": ["下方修正", "減配", "無配"],
}
POS_NEWS = {
    "上方修正": ["上方修正"],
    "増配": ["増配", "復配"],
    "自社株買い": ["自社株買い", "自己株式取得"],
    "最高益・黒字転換": ["最高益", "黒字転換", "黒字化", "過去最高"],
}
RISK_NEWS = {"株の希薄化(MSワラント・増資など)", "不祥事・上場維持の懸念"}


def fetch_alert_codes() -> dict[str, str]:
    """JPXの注意喚起の一覧ページから銘柄コードを集める。取れなければ空(=このチェックは無しで進む)。"""
    out: dict[str, str] = {}
    for url, label in JPX_ALERT_PAGES.items():
        try:
            r = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
            r.raise_for_status()
            html = r.content.decode("utf-8", errors="ignore")
        except requests.RequestException as exc:
            print(f"[quality] {label} の取得失敗: {exc}")
            continue
        # 表の中の銘柄コード(4桁の数字、または数字+英字の新コード)
        body = html[html.find("<table") :] if "<table" in html else ""
        for code in re.findall(r">\s*([0-9]{3}[0-9A-Z])\s*<", body):
            out.setdefault(code, label)
    print(f"[quality] JPXの注意喚起銘柄 {len(out)} 件")
    return out


def cheap_risks(ticker: str, price: float | None, turnover: float | None, alerts: dict[str, str]) -> list[str]:
    """全銘柄に使える軽いチェック(株価データとJPXの一覧だけ)。"""
    out = []
    lab = alerts.get(ticker.replace(".T", ""))
    if lab:
        out.append(lab)
    if price is not None and price < MIN_PRICE:
        out.append(f"株価{MIN_PRICE}円未満")
    if turnover is not None and turnover < MIN_TURNOVER:
        out.append("売買がほとんどない")
    return out


def _hits(titles: list[str], table: dict[str, list[str]]) -> list[str]:
    found = []
    for label, kws in table.items():
        if any(k in t for t in titles for k in kws):
            found.append(label)
    return found


def assess(rec: dict, event_titles: list[str], event_types: set[str]) -> dict:
    """
    候補銘柄(決算・ニュースを取った銘柄)の材料チェック。
    戻り値: {"s": 点数, "p": [良い材料], "m": [悪い材料], "r": [足切り理由]}
    """
    plus: list[str] = []
    minus: list[str] = []
    risk: list[str] = list(rec.get("rk") or [])
    score = 0

    e = rec.get("earn") or {}
    if e.get("has_data"):
        ry, py = e.get("rev_yoy"), e.get("profit_yoy")
        if ry is not None:
            if ry > 0:
                plus.append("増収"); score += 1
            elif ry < -0.05:
                minus.append("減収"); score -= 1
        if py is not None:
            if py > 0:
                plus.append("増益"); score += 1
            elif py < 0:
                minus.append("減益"); score -= 1
        if e.get("loss_last") is False:
            plus.append("黒字"); score += 1
        elif e.get("loss_last"):
            if (e.get("loss_q") or 0) >= 4:
                minus.append("4四半期ずっと赤字"); score -= 2; risk.append("赤字が続いている")
            else:
                minus.append("直近の四半期が赤字"); score -= 1
        er = e.get("eq_ratio")
        if er is not None:
            if er < 0:
                minus.append("債務超過"); score -= 3; risk.append("債務超過")
            elif er < 0.2:
                minus.append(f"自己資本比率{round(er * 100)}%と低い"); score -= 1
            elif er >= 0.5:
                plus.append("財務が健全(自己資本比率50%以上)"); score += 1

    elif rec.get("fin"):
        # 四半期決算が取れない銘柄は、Yahoo Financeの会社情報(直近12ヶ月)で代わりに判定
        f = rec["fin"]
        if f.get("rg") is not None:
            if f["rg"] > 0:
                plus.append("増収"); score += 1
            elif f["rg"] < -0.05:
                minus.append("減収"); score -= 1
        if f.get("eg") is not None:
            if f["eg"] > 0:
                plus.append("増益"); score += 1
            elif f["eg"] < 0:
                minus.append("減益"); score -= 1
        if f.get("pm") is not None:
            if f["pm"] > 0:
                plus.append("黒字"); score += 1
            elif f["pm"] < 0:
                minus.append("赤字(直近12ヶ月)"); score -= 1

    titles = [n.get("t", "") for n in rec.get("news") or []] + list(event_titles)
    for lab in _hits(titles, NEG_NEWS):
        minus.append(f"ニュース: {lab}")
        score -= 3 if lab in RISK_NEWS else 2
        if lab in RISK_NEWS:
            risk.append(lab)
    pos = set(_hits(titles, POS_NEWS))
    pos |= {{"upward": "上方修正", "dividend": "増配", "buyback": "自社株買い"}.get(t) for t in event_types} - {None}
    for lab in ["上方修正", "増配", "自社株買い", "最高益・黒字転換"]:
        if lab in pos:
            plus.append(lab); score += 2 if lab == "上方修正" else 1

    dv = rec.get("dv")
    if dv and dv.get("y"):
        plus.append(f"配当あり({dv['y'] * 100:.1f}%)"); score += 1
        if dv.get("tr") == "増配傾向":
            plus.append("増配傾向"); score += 1

    # 過去の統計で効いていた形(条件の効き目分析より)
    if rec.get("dsl") is not None and rec["dsl"] <= 5:
        plus.append("底から5日以内(早く乗れる)"); score += 1
    if rec.get("pos") is not None and rec["pos"] <= 0.10:
        plus.append("5年の最安値圏(下から10%以内)"); score += 1

    return {"s": score, "p": plus, "m": minus, "r": sorted(set(risk))}
