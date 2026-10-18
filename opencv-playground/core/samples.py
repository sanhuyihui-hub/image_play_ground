"""サンプル画像をコードで生成する（著作権を気にせず記事のスクリーンショットに使えるように）。"""

from __future__ import annotations

import cv2
import numpy as np

W, H = 640, 480


def shapes() -> np.ndarray:
    """色・形の異なる図形。HSV 抽出、輪郭抽出、エッジ検出向け。"""
    img = np.full((H, W, 3), 235, np.uint8)
    # 背景にうっすら横グラデーション
    img = (img.astype(np.float32) * np.linspace(0.85, 1.0, W)[None, :, None]).astype(np.uint8)
    cv2.circle(img, (110, 120), 70, (40, 40, 220), -1)  # 赤
    cv2.rectangle(img, (230, 50), (370, 190), (60, 180, 60), -1)  # 緑
    cv2.fillPoly(img, [np.array([(500, 40), (600, 200), (400, 200)])], (200, 120, 30))  # 青
    cv2.ellipse(img, (120, 350), (90, 50), 30, 0, 360, (30, 200, 230), -1)  # 黄
    cv2.fillPoly(img, [np.array([(300, 270), (370, 300), (360, 380), (280, 410), (240, 330)])], (170, 60, 150))  # 紫
    cv2.circle(img, (520, 340), 75, (30, 120, 240), -1)  # オレンジ
    cv2.circle(img, (520, 340), 35, (235, 235, 235), -1)  # 穴（RETR_TREE の確認用）
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 6, img.shape)
    return np.clip(img + noise, 0, 255).astype(np.uint8)


def document() -> np.ndarray:
    """照明ムラのある文書風画像。固定しきい値と適応的二値化の違いを見る用。"""
    img = np.full((H, W), 220, np.float32)
    lines = [
        "OpenCV Playground",
        "Adaptive thresholding works well",
        "when the lighting is uneven.",
        "cv2.adaptiveThreshold(gray, 255,",
        "    ADAPTIVE_THRESH_GAUSSIAN_C,",
        "    THRESH_BINARY, 31, 10)",
        "Otsu picks one global threshold.",
    ]
    for i, text in enumerate(lines):
        cv2.putText(img, text, (30, 60 + i * 58), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 40, 2, cv2.LINE_AA)
    # 左上から右下へ暗くなる照明ムラ
    yy, xx = np.mgrid[0:H, 0:W]
    shade = 1.0 - 0.65 * ((xx / W) * 0.6 + (yy / H) * 0.4)
    img = img * shade
    rng = np.random.default_rng(1)
    img += rng.normal(0, 4, img.shape)
    gray = np.clip(img, 0, 255).astype(np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def landscape(dark: bool = True) -> np.ndarray:
    """合成の風景。dark=True で暗く低コントラスト（LUT・ヒストグラム向け）。"""
    img = np.zeros((H, W, 3), np.float32)
    sky_top, sky_bottom = np.array([200, 140, 60]), np.array([250, 220, 180])
    t = np.linspace(0, 1, H // 2)[:, None, None]
    img[: H // 2] = sky_top * (1 - t) + sky_bottom * t
    img[H // 2 :] = (60, 130, 70)
    cv2.circle(img, (480, 120), 45, (120, 230, 255), -1)
    for x in range(40, W, 90):
        cv2.fillPoly(img, [np.array([(x, H // 2 + 60), (x + 30, H // 2 - 70), (x + 60, H // 2 + 60)])], (30, 90, 40))
    cv2.rectangle(img, (0, H - 80), (W, H), (90, 110, 140), -1)
    if dark:
        img = img * 0.35 + 20
    else:
        # 夕焼け風の暖色。ヒストグラムマッチングの参照画像に使う
        img = img * np.array([0.55, 0.85, 1.25]) + np.array([0, 10, 30])
    rng = np.random.default_rng(2)
    img += rng.normal(0, 3, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


SAMPLES = {
    "図形（カラー）": shapes,
    "文書（照明ムラあり）": document,
    "風景（暗め）": lambda: landscape(dark=True),
    "風景（夕焼け）": lambda: landscape(dark=False),
}
