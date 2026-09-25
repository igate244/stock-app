"""
データ取得: 銘柄ユニバース(JPX上場銘柄一覧)と株価(yfinance一括ダウンロード)。
GitHub Actions上で実行する前提(日本の証券系サイトに普通にアクセスできる環境)。
"""

from __future__ import annotations

import io
import time
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

JPX_URL = "https://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.xlsx"
FALLBACK_CSV = Path(__file__).resolve().parent.parent / "data" / "tickers.csv"


def load_universe() -> pd.DataFrame:
    """columns: ticker, name, sector。JPXから取れなければリポジトリ内のCSVで代用。"""
    try:
        resp = requests.get(JPX_URL, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        df = pd.read_excel(io.BytesIO(resp.content))
        df = df.rename(columns={"コード": "code", "銘柄名": "name", "33業種区分": "sector"})
        df = df[df["sector"] != "-"]
        df["ticker"] = df["code"].astype(str).str.strip() + ".T"
        universe = df[["ticker", "name", "sector"]]
        print(f"[universe] JPXから {len(universe)} 銘柄を取得")
    except Exception as exc:  # noqa: BLE001
        print(f"[universe] JPX取得に失敗({exc})。{FALLBACK_CSV.name} を使用")
        universe = pd.read_csv(FALLBACK_CSV, encoding="utf-8-sig")[["ticker", "name", "sector"]]

    # 社債型種類株式などの普通株以外は除外
    universe = universe[~universe["name"].astype(str).str.contains("種類株式")]
    return universe.dropna(subset=["ticker", "name"]).drop_duplicates("ticker").reset_index(drop=True)


def _field(raw: pd.DataFrame, name: str, tickers: list[str]) -> pd.DataFrame | None:
    """yf.downloadの戻り値から1項目(Close/Adj Close/Volume/Dividends…)を「列=銘柄」の表で取り出す(列の持ち方がバージョンで揺れる)。"""
    if isinstance(raw.columns, pd.MultiIndex):
        lv0, lv1 = raw.columns.get_level_values(0), raw.columns.get_level_values(1)
        if name in lv0:
            return raw[name]
        if name in lv1:
            return raw.xs(name, axis=1, level=1)
        return None
    if name in raw.columns:  # 単一銘柄・フラット列
        return raw[[name]].rename(columns={name: tickers[0]})
    return None


def _extract_closes(raw: pd.DataFrame, tickers: list[str], volumes: dict | None = None,
                    extras: dict | None = None) -> dict[str, pd.Series]:
    """
    銘柄ごとの終値Series(配当・分割調整済み)を取り出す。
    volumes(dict)を渡すと出来高を、extras={"raw": {}, "div": {}} を渡すと
    配当調整なしの終値(raw)と配当の履歴(div: 権利落ち日→1株配当)も、終値と同じ日付にそろえて入れる。
    """
    out: dict[str, pd.Series] = {}
    if raw is None or raw.empty:
        return out
    closes = _field(raw, "Adj Close", tickers)
    if closes is None:
        closes = _field(raw, "Close", tickers)
    if closes is None:
        return out
    vols = _field(raw, "Volume", tickers) if volumes is not None else None
    raws = _field(raw, "Close", tickers) if extras is not None else None
    divs = _field(raw, "Dividends", tickers) if extras is not None else None
    for t in closes.columns:
        s = closes[t].dropna()
        s = s[s > 0]
        s = _drop_glitches(s)
        if len(s) < 30:
            continue
        idx = pd.to_datetime(s.index).tz_localize(None)
        if vols is not None and t in vols.columns:
            v = vols[t].reindex(s.index).fillna(0).astype(float)
            v.index = idx
            volumes[str(t)] = v
        if raws is not None and t in raws.columns:
            r = raws[t].reindex(s.index).astype(float)
            r.index = idx
            extras.setdefault("raw", {})[str(t)] = r
        if divs is not None and t in divs.columns:
            d = divs[t].reindex(s.index).fillna(0).astype(float)
            d.index = idx
            d = d[d > 0]
            if len(d):
                extras.setdefault("div", {})[str(t)] = d
        s.index = idx
        out[str(t)] = s.astype(float)
    return out


def _drop_glitches(s: pd.Series) -> pd.Series:
    """
    1日で3倍超 / 3分の1未満になるような値動きは、日本株の値幅制限上ほぼあり得ないので
    データ異常(分割の未調整・誤データ)とみなし、最後の異常より後ろのデータだけ残す。
    (初回のバックテストで、こうした異常値のせいで平均リターンが+45,000%のように壊れたため)
    """
    if len(s) < 2:
        return s
    ratio = s / s.shift(1)
    bad = (ratio > 3) | (ratio < 1 / 3)
    if not bad.any():
        return s
    last_bad = bad[bad].index[-1]
    return s.loc[s.index >= last_bad]


def download_closes(tickers: list[str], period: str = "5y", batch_size: int = 150,
                    volumes: dict[str, pd.Series] | None = None, extras: dict | None = None) -> dict[str, pd.Series]:
    """
    全銘柄の終値(配当・分割調整済み)を一括取得。volumesを渡すと出来高、extrasを渡すと
    配当調整なしの終値と配当履歴も入れる。取れなかった銘柄は小さいバッチでリトライ。
    """
    result: dict[str, pd.Series] = {}

    def run(batch: list[str]) -> None:
        for attempt in range(3):
            try:
                raw = yf.download(
                    batch,
                    period=period,
                    interval="1d",
                    auto_adjust=extras is None,   # 配当情報が要る時は調整前/調整後の両方を取る
                    actions=extras is not None,
                    group_by="column",
                    threads=True,
                    progress=False,
                )
                result.update(_extract_closes(raw, batch, volumes, extras))
                return
            except Exception as exc:  # noqa: BLE001 — レート制限等は待って再試行
                wait = 30 * (attempt + 1)
                print(f"[prices] 取得エラー({exc})、{wait}秒待って再試行")
                time.sleep(wait)

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i : i + batch_size]
        run(batch)
        print(f"[prices] {min(i + batch_size, len(tickers))}/{len(tickers)} 取得済み(成功 {len(result)})")
        time.sleep(2)

    missing = [t for t in tickers if t not in result]
    if missing:
        print(f"[prices] 未取得 {len(missing)} 銘柄を再試行")
        for i in range(0, len(missing), 40):
            run(missing[i : i + 40])
            time.sleep(3)
    print(f"[prices] 最終: {len(result)}/{len(tickers)} 銘柄")
    return result
