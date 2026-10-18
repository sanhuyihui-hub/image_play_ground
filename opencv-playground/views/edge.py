import numpy as np
import streamlit as st

from core import image_ops
from views import common

ARTICLE = "https://qiita.com/sanhuyihui/items/82abf5a190be1d21ddec"


def render(img: np.ndarray) -> None:
    common.header("エッジ検出", "明るさが急に変わる場所（物体の境界）を取り出します。", ARTICLE)
    gray = image_ops.to_gray(img)

    method = st.radio("手法", image_ops.EDGE_METHODS, horizontal=True)
    ksize_blur = common.odd_slider("前処理のガウシアンぼかし（1 = なし）", 1, 15, 5, help="ノイズによる細かいエッジを減らします")
    blurred = image_ops.blur(gray, ksize_blur)
    blur_code = f"gray = cv2.GaussianBlur(gray, ({ksize_blur}, {ksize_blur}), 0)\n" if ksize_blur > 1 else ""

    if method == "Canny":
        t1, t2 = st.slider("しきい値（下限, 上限）", 0, 500, (50, 150), help="上限以上は確実にエッジ、下限〜上限は強いエッジにつながる場合だけ残す")
        c1, c2 = st.columns(2)
        aperture = c1.select_slider("Sobel のカーネルサイズ", [3, 5, 7])
        l2 = c2.toggle("L2gradient（より正確な勾配）")
        edges = image_ops.canny(blurred, t1, t2, aperture, l2)
        body = f"edges = cv2.Canny(gray, {t1}, {t2}, apertureSize={aperture}, L2gradient={l2})"
    elif method == "Sobel":
        c1, c2 = st.columns(2)
        direction = c1.radio("方向", ["両方", "x方向", "y方向"], horizontal=True)
        ksize = c2.select_slider("カーネルサイズ", [1, 3, 5, 7], value=3)
        edges = image_ops.sobel(blurred, ksize, direction)
        if direction == "x方向":
            body = f"edges = cv2.convertScaleAbs(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize={ksize}))"
        elif direction == "y方向":
            body = f"edges = cv2.convertScaleAbs(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize={ksize}))"
        else:
            body = f"""gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize={ksize})
gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize={ksize})
edges = cv2.convertScaleAbs(cv2.magnitude(gx, gy))"""
    else:
        ksize = st.select_slider("カーネルサイズ", [1, 3, 5, 7], value=3)
        edges = image_ops.laplacian(blurred, ksize)
        body = f"edges = cv2.convertScaleAbs(cv2.Laplacian(gray, cv2.CV_64F, ksize={ksize}))"

    common.show_pair(blurred, edges, "前処理後のグレースケール", f"{method} の結果")

    with st.expander("3つの手法を並べて比較"):
        cols = st.columns(3)
        cols[0].image(image_ops.canny(blurred, 50, 150), caption="Canny (50, 150)", width="stretch")
        cols[1].image(image_ops.sobel(blurred, 3), caption="Sobel (ksize=3)", width="stretch")
        cols[2].image(image_ops.laplacian(blurred, 3), caption="Laplacian (ksize=3)", width="stretch")

    common.show_code(f"""
import cv2

img = cv2.imread("sample.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
{blur_code}{body}
""")
    common.download(edges, f"edge_{method.lower()}.png")
