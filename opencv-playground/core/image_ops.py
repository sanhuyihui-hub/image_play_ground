"""画像処理の本体。Streamlit に依存しない純粋な関数だけを置く（テストしやすくするため）。

画像はすべて OpenCV の流儀に合わせて BGR / uint8 で受け渡す。
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# グレースケール変換
# ---------------------------------------------------------------------------

GRAY_METHODS = ("cvtColor", "単純平均", "加重平均（手計算）")


def to_gray(img: np.ndarray, method: str = "cvtColor") -> np.ndarray:
    if img.ndim == 2:
        return img.copy()
    if method == "cvtColor":
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    b, g, r = cv2.split(img.astype(np.float32))
    if method == "単純平均":
        gray = (b + g + r) / 3
    elif method == "加重平均（手計算）":
        # ITU-R BT.601 の係数。cv2.cvtColor と同じ重み
        gray = 0.299 * r + 0.587 * g + 0.114 * b
    else:
        raise ValueError(f"unknown method: {method}")
    return np.clip(np.round(gray), 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# 二値化
# ---------------------------------------------------------------------------

THRESH_TYPES = {
    "THRESH_BINARY": cv2.THRESH_BINARY,
    "THRESH_BINARY_INV": cv2.THRESH_BINARY_INV,
    "THRESH_TRUNC": cv2.THRESH_TRUNC,
    "THRESH_TOZERO": cv2.THRESH_TOZERO,
    "THRESH_TOZERO_INV": cv2.THRESH_TOZERO_INV,
}

ADAPTIVE_METHODS = {
    "ADAPTIVE_THRESH_MEAN_C": cv2.ADAPTIVE_THRESH_MEAN_C,
    "ADAPTIVE_THRESH_GAUSSIAN_C": cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
}


def threshold(
    gray: np.ndarray, thresh: int, maxval: int = 255, type_name: str = "THRESH_BINARY", otsu: bool = False
) -> tuple[float, np.ndarray]:
    flag = THRESH_TYPES[type_name]
    if otsu:
        flag |= cv2.THRESH_OTSU
    ret, binary = cv2.threshold(gray, thresh, maxval, flag)
    return ret, binary


def adaptive_threshold(
    gray: np.ndarray, method_name: str, block_size: int, c: int, invert: bool = False
) -> np.ndarray:
    if block_size % 2 == 0 or block_size < 3:
        raise ValueError("block_size は 3 以上の奇数にしてください")
    type_ = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    return cv2.adaptiveThreshold(gray, 255, ADAPTIVE_METHODS[method_name], type_, block_size, c)


# ---------------------------------------------------------------------------
# HSV 色抽出
# ---------------------------------------------------------------------------

# OpenCV の H は 0〜179（360° を半分にしたもの）
HSV_PRESETS = {
    "赤": ((170, 80, 60), (10, 255, 255)),
    "オレンジ": ((10, 100, 100), (22, 255, 255)),
    "黄": ((22, 80, 80), (35, 255, 255)),
    "緑": ((40, 60, 40), (85, 255, 255)),
    "青": ((95, 80, 40), (130, 255, 255)),
    "紫": ((130, 60, 40), (165, 255, 255)),
}


def hsv_mask(img: np.ndarray, lower: tuple[int, int, int], upper: tuple[int, int, int]) -> np.ndarray:
    """H の下限が上限より大きいときは、赤のように 0/179 をまたぐ範囲として扱う。"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_lo, s_lo, v_lo = lower
    h_hi, s_hi, v_hi = upper
    if h_lo <= h_hi:
        return cv2.inRange(hsv, np.array(lower), np.array(upper))
    mask1 = cv2.inRange(hsv, np.array((h_lo, s_lo, v_lo)), np.array((179, s_hi, v_hi)))
    mask2 = cv2.inRange(hsv, np.array((0, s_lo, v_lo)), np.array((h_hi, s_hi, v_hi)))
    return cv2.bitwise_or(mask1, mask2)


def apply_mask(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return cv2.bitwise_and(img, img, mask=mask)


# ---------------------------------------------------------------------------
# エッジ検出
# ---------------------------------------------------------------------------

EDGE_METHODS = ("Canny", "Sobel", "Laplacian")


def blur(gray: np.ndarray, ksize: int) -> np.ndarray:
    if ksize <= 1:
        return gray
    return cv2.GaussianBlur(gray, (ksize, ksize), 0)


def canny(gray: np.ndarray, t1: int, t2: int, aperture: int = 3, l2: bool = False) -> np.ndarray:
    return cv2.Canny(gray, t1, t2, apertureSize=aperture, L2gradient=l2)


def sobel(gray: np.ndarray, ksize: int = 3, direction: str = "両方") -> np.ndarray:
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
    if direction == "x方向":
        mag = np.abs(gx)
    elif direction == "y方向":
        mag = np.abs(gy)
    else:
        mag = cv2.magnitude(gx, gy)
    return cv2.convertScaleAbs(mag)


def laplacian(gray: np.ndarray, ksize: int = 3) -> np.ndarray:
    return cv2.convertScaleAbs(cv2.Laplacian(gray, cv2.CV_64F, ksize=ksize))


# ---------------------------------------------------------------------------
# 輪郭抽出
# ---------------------------------------------------------------------------

RETRIEVAL_MODES = {
    "RETR_EXTERNAL": cv2.RETR_EXTERNAL,
    "RETR_LIST": cv2.RETR_LIST,
    "RETR_TREE": cv2.RETR_TREE,
}

APPROX_METHODS = {
    "CHAIN_APPROX_SIMPLE": cv2.CHAIN_APPROX_SIMPLE,
    "CHAIN_APPROX_NONE": cv2.CHAIN_APPROX_NONE,
}


@dataclass
class ContourInfo:
    index: int
    area: float
    perimeter: float
    x: int
    y: int
    w: int
    h: int
    vertices: int


def find_contours(
    binary: np.ndarray, mode_name: str = "RETR_EXTERNAL", approx_name: str = "CHAIN_APPROX_SIMPLE", min_area: float = 0
) -> tuple[list[np.ndarray], list[ContourInfo]]:
    contours, _ = cv2.findContours(binary, RETRIEVAL_MODES[mode_name], APPROX_METHODS[approx_name])
    kept, infos = [], []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        perimeter = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
        x, y, w, h = cv2.boundingRect(cnt)
        kept.append(cnt)
        infos.append(ContourInfo(len(kept), area, perimeter, x, y, w, h, len(approx)))
    return kept, infos


def draw_contours(img: np.ndarray, contours: list[np.ndarray], draw_bbox: bool = False, label: bool = False) -> np.ndarray:
    out = img.copy() if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(out, contours, -1, (0, 255, 0), 2)
    for i, cnt in enumerate(contours, start=1):
        x, y, w, h = cv2.boundingRect(cnt)
        if draw_bbox:
            cv2.rectangle(out, (x, y), (x + w, y + h), (0, 0, 255), 2)
        if label:
            cv2.putText(out, str(i), (x, max(y - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    return out


# ---------------------------------------------------------------------------
# LUT（ルックアップテーブル）
# ---------------------------------------------------------------------------

LUT_KINDS = ("ガンマ補正", "コントラスト（S字）", "ネガポジ反転", "ポスタリゼーション")


def make_lut(kind: str, param: float) -> np.ndarray:
    x = np.arange(256, dtype=np.float64)
    if kind == "ガンマ補正":
        y = 255 * (x / 255) ** (1 / param)
    elif kind == "コントラスト（S字）":
        # シグモイドで中間調を強調。param が大きいほど強くなる
        y = 255 / (1 + np.exp(-param * (x - 127.5) / 255))
        y = (y - y.min()) / (y.max() - y.min()) * 255
    elif kind == "ネガポジ反転":
        y = 255 - x
    elif kind == "ポスタリゼーション":
        levels = int(param)
        step = 256 / levels
        y = np.floor(x / step) * (255 / (levels - 1))
    else:
        raise ValueError(f"unknown kind: {kind}")
    return np.clip(np.round(y), 0, 255).astype(np.uint8)


def apply_lut(img: np.ndarray, lut: np.ndarray) -> np.ndarray:
    return cv2.LUT(img, lut)


# ---------------------------------------------------------------------------
# ヒストグラム / ヒストグラムマッチング
# ---------------------------------------------------------------------------


def histogram(channel: np.ndarray) -> np.ndarray:
    return cv2.calcHist([channel], [0], None, [256], [0, 256]).ravel()


def cdf(channel: np.ndarray) -> np.ndarray:
    """0〜1 に正規化した累積分布関数。"""
    hist = histogram(channel)
    c = hist.cumsum()
    return c / c[-1]


def match_channel(src: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """src の各画素値 v を、CDF の値が最も近くなる ref 側の画素値へ置き換える。"""
    src_cdf = cdf(src)
    ref_cdf = cdf(ref)
    lut = np.searchsorted(ref_cdf, src_cdf, side="left").clip(0, 255).astype(np.uint8)
    return cv2.LUT(src, lut)


def match_histograms(src: np.ndarray, ref: np.ndarray) -> np.ndarray:
    if src.ndim == 2:
        return match_channel(src, to_gray(ref))
    if ref.ndim == 2:
        ref = cv2.cvtColor(ref, cv2.COLOR_GRAY2BGR)
    return cv2.merge([match_channel(s, r) for s, r in zip(cv2.split(src), cv2.split(ref))])
