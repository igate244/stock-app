"""
株価系列から数値シグナルを作る(旧 price_position / bottom_pattern / pattern_similarity)。

すべて終値の系列(pd.Series、日付インデックス)だけを入力にする。
ボリンジャーバンドやMACDなどの派生指標は使わず、線グラフを見れば人間の目でも
追える「どの位置にいるか」「高値→安値→反発したか」「上昇の形が似ているか」だけで判定する。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# ---- 判定パラメータ(ここを変えれば判定の厳しさが変わる) -------------------------
NEAR_LOW_THRESHOLD = 0.30  # 5年レンジの下位30%以内を「底値圏」とみなす
MIN_YEARS_FOR_CONFIDENCE = 3.0  # これ未満のデータ期間は「5年位置」の信頼度が低い
BOTTOM_LOOKBACK_DAYS = 63  # 「3ヶ月」の判定窓(営業日)
MIN_DECLINE_PCT = 0.15  # 底に至るまでに最低これだけ下げていること
MIN_REBOUND_PCT = 0.10  # 底からこれだけ反発していること
MIN_DAYS_SINCE_LOW = 3  # 底から最低これだけ経過(直後の跳ね返りだけで判定しない)
SIMILARITY_WINDOW = 40  # (旧方式)底から何営業日分の上昇カーブを比較するか
PRE_WINDOW = 60  # 類似度: 「今日までの直近何営業日」の形を比べるか(下落→底→反発しはじめ)
PRE_MAX_OFFSET = 20  # 勝ちパターン側は「底の日〜底の20営業日後」までのどの時点を"今日"に当てても良い
PRE_FUTURE = 60  # 比較グラフで見せる「勝ちパターンのその後」の日数


# ---- STEP2: 5年株価位置 ---------------------------------------------------------
@dataclass
class PricePosition:
    position: float  # 0.0(期間内最安値)〜1.0(最高値)
    years_of_data: float
    sufficient_data: bool

    @property
    def near_low(self) -> bool:
        return self.position <= NEAR_LOW_THRESHOLD


def price_position(close: pd.Series) -> PricePosition:
    low, high, current = close.min(), close.max(), close.iloc[-1]
    position = 0.5 if high == low else float((current - low) / (high - low))
    years = (close.index[-1] - close.index[0]).days / 365.25 if len(close) > 1 else 0.0
    return PricePosition(position, years, years >= MIN_YEARS_FOR_CONFIDENCE)


# ---- STEP3: 3ヶ月底打ち --------------------------------------------------------
@dataclass
class BottomSignal:
    bottomed: bool
    low_price: float
    low_date: pd.Timestamp | None
    days_since_low: int
    decline_pct: float  # 底に至るまでの下落率
    rebound_pct: float  # 底からの反発率


def detect_bottom(close: pd.Series, lookback_days: int = BOTTOM_LOOKBACK_DAYS) -> BottomSignal:
    """
    直近lookback_days の中で「高値→安値→反発」が起きているか。
      1. 窓内の最安値を「底」候補とする
      2. 底に至るまでに MIN_DECLINE_PCT 以上下げている
      3. 底から MIN_REBOUND_PCT 以上反発している
      4. 底から MIN_DAYS_SINCE_LOW 営業日以上経過している
    """
    window = close.tail(lookback_days)
    if len(window) < 10:
        return BottomSignal(False, float("nan"), None, 0, 0.0, 0.0)

    low_date = window.idxmin()
    low_price = float(window.loc[low_date])
    pre_low_high = float(window.loc[:low_date].max())
    current = float(window.iloc[-1])
    days_since_low = len(window.loc[low_date:]) - 1

    decline = (pre_low_high - low_price) / pre_low_high if pre_low_high else 0.0
    rebound = (current - low_price) / low_price if low_price else 0.0
    bottomed = (
        decline >= MIN_DECLINE_PCT
        and rebound >= MIN_REBOUND_PCT
        and days_since_low >= MIN_DAYS_SINCE_LOW
    )
    return BottomSignal(bottomed, low_price, low_date, days_since_low, float(decline), float(rebound))


# ---- STEP5: 勝ちパターン類似度 -------------------------------------------------
def _post_bottom_curve(close: pd.Series, low_date: pd.Timestamp, window: int) -> np.ndarray | None:
    after = close.loc[low_date:].iloc[:window]
    if len(after) < 5 or not after.iloc[0]:
        return None
    return (after / after.iloc[0]).to_numpy() * 100


def reference_low_date(ref_close: pd.Series, pattern_start: str | None) -> pd.Timestamp | None:
    """
    勝ちパターン銘柄側の「底の日」。
    pattern_start 指定があればその日。なければ「期間内(5年)の最安値の日」を底とみなす
    (その後の上昇を比較できるよう、直近SIMILARITY_WINDOW日は探索対象から外す)。
    """
    if pattern_start:
        idx = ref_close.index[ref_close.index >= pd.Timestamp(pattern_start)]
        return idx[0] if len(idx) else None
    searchable = ref_close.iloc[:-SIMILARITY_WINDOW]
    return searchable.idxmin() if len(searchable) else None


def similarity_score(
    cand_close: pd.Series,
    cand_low_date: pd.Timestamp,
    ref_close: pd.Series,
    ref_low_date: pd.Timestamp,
    window: int = SIMILARITY_WINDOW,
) -> float:
    """
    底の日を起点(=100)に正規化した上昇カーブ同士の相関係数(0.0〜1.0)。
    注意: 相関ベースなので「どちらも右肩上がり」なだけで高めに出やすい。参考値。
    """
    a = _post_bottom_curve(cand_close, cand_low_date, window)
    b = _post_bottom_curve(ref_close, ref_low_date, window)
    if a is None or b is None:
        return 0.0
    n = min(len(a), len(b))
    if n < 5:
        return 0.0
    corr = np.corrcoef(a[:n], b[:n])[0, 1]
    return 0.0 if np.isnan(corr) else max(0.0, float(corr))


# ---- STEP5(新方式): 上がる前の形で比べる ----------------------------------------
# 旧方式は「底からの上昇カーブ」同士を比べていたので、形が似ていると分かる頃には
# もう上がったあと(買うには遅い)。新方式は「今日までの直近PRE_WINDOW日の形」
# (下げて→底をつけて→反発しはじめ)を、勝ちパターン銘柄が大きく上がる"前"の同じ区間と比べる。
# 勝ちパターン側は「底の日」〜「底のPRE_MAX_OFFSET日後」のどこを"今日"に当てるかを全部試して一番似ている所を採用
# (=底打ち直後でも、少し反発した後でも見つけられる)。

def _zrows(m: np.ndarray) -> np.ndarray:
    """各行(=1つの期間の対数株価)を平均0・標準偏差1にそろえる。相関を行列の掛け算で一気に出すため。"""
    mu = m.mean(axis=-1, keepdims=True)
    sd = m.std(axis=-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        z = (m - mu) / sd
    return np.nan_to_num(z)


def pre_templates(ref_close: pd.Series, ref_low: pd.Timestamp, name: str,
                  window: int = PRE_WINDOW, max_offset: int = PRE_MAX_OFFSET) -> list[dict]:
    """勝ちパターン銘柄の「上がる前の形」のお手本(底の日+k日を"今日"とした直近window日)。"""
    arr = ref_close.to_numpy(dtype=float)
    li = int(ref_close.index.get_indexer([ref_low])[0])
    out = []
    if li < 0:
        return out
    for k in range(0, max_offset + 1):
        end = li + k
        start = end - window + 1
        if start < 0 or end >= len(arr):
            continue
        seg = np.log(arr[start : end + 1])
        out.append({"name": name, "k": k, "end": end, "z": _zrows(seg[None, :])[0], "close": ref_close})
    return out


def pre_similarity(close: pd.Series, templates: list[dict], window: int = PRE_WINDOW) -> tuple[float, dict | None]:
    """今日までの直近window日の形と、お手本の形の相関(-1〜1、高いほど似ている)の最大値と、そのお手本。"""
    if len(close) < window or not templates:
        return 0.0, None
    z = _zrows(np.log(close.to_numpy(dtype=float)[-window:])[None, :])[0]
    T = np.stack([t["z"] for t in templates])
    c = T @ z / window
    i = int(np.argmax(c))
    return float(c[i]), templates[i]


def pre_similarity_all(close: np.ndarray, T: np.ndarray, window: int = PRE_WINDOW) -> tuple[np.ndarray, np.ndarray]:
    """
    バックテスト用: 全営業日について「その日までの直近window日」の類似度を一括計算。
    T = お手本のz配列を縦に積んだもの。戻り値 (類似度[n], 一番似たお手本の番号[n])。最初のwindow-1日はNaN。
    """
    from numpy.lib.stride_tricks import sliding_window_view

    n = len(close)
    sim = np.full(n, np.nan)
    arg = np.full(n, -1)
    if n < window or not len(T):
        return sim, arg
    Z = _zrows(sliding_window_view(np.log(close), window))
    C = Z @ T.T / window
    sim[window - 1 :] = C.max(axis=1)
    arg[window - 1 :] = C.argmax(axis=1)
    return sim, arg
