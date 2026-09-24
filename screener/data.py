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


def _extract_closes(raw: pd.DataFrame, tickers: list[str]) -> dict[str, pd.Series]:
    """yf.downloadの戻り値(列の持ち方がバージョンで揺れる)から、銘柄ごとの終値Seriesを取り出す。"""
    out: dict[str, pd.Series] = {}
    if raw is None or raw.empty:
        return out
    if isinstance(raw.columns, pd.MultiIndex):
        lv0 = raw.columns.get_level_values(0)
        closes = raw["Close"] if "Close" in lv0 else raw.xs("Close", axis=1, level=1)
    else:  # 単一銘柄・フラット列
        closes = raw[["Close"]].rename(columns={"Close": tickers[0]})
    for t in closes.columns:
        s = closes[t].dropna()
        s = s[s > 0]
        if len(s) >= 30:
            s.index = pd.to_datetime(s.index).tz_localize(None)
            out[str(t)] = s.astype(float)
    return out


def download_closes(tickers: list[str], period: str = "5y", batch_size: int = 150) -> dict[str, pd.Series]:
    """全銘柄の終値を一括取得。取れなかった銘柄は小さいバッチで1回だけリトライ。"""
    result: dict[str, pd.Series] = {}

    def run(batch: list[str]) -> None:
        for attempt in range(3):
            try:
                raw = yf.download(
                    batch,
                    period=period,
                    interval="1d",
                    auto_adjust=True,
                    group_by="column",
                    threads=True,
                    progress=False,
                )
                result.update(_extract_closes(raw, batch))
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
