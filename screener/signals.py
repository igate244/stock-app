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
SIMILARITY_WINDOW = 40  # 底から何営業日分の上昇カーブを比較するか


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
