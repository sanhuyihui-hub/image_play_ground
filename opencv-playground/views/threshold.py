import numpy as np
import streamlit as st

from core import image_ops
from views import common

ARTICLE = "https://qiita.com/sanhuyihui/items/2513e1cf8ed9a6d1d816"


def render(img: np.ndarray) -> None:
    common.header("二値化（Threshold）", "しきい値を境に画素を白と黒に分けます。照明ムラがあるなら「適応的二値化」を試してください。", ARTICLE)
    gray = image_ops.to_gray(img)

    tab_global, tab_adaptive, tab_compare = st.tabs(["固定しきい値 / 大津の二値化", "適応的二値化", "3つを比較"])

    with tab_global:
        c1, c2 = st.columns(2)
        type_name = c1.selectbox("種類", list(image_ops.THRESH_TYPES))
        otsu = c2.toggle("大津の二値化で自動決定", help="しきい値をヒストグラムから自動で決めます")
        thresh = st.slider("しきい値", 0, 255, 127, disabled=otsu)
        ret, binary = image_ops.threshold(gray, thresh, 255, type_name, otsu)
        common.show_pair(gray, binary, "グレースケール", f"結果（使われたしきい値: {ret:.0f}）")
        common.histogram_chart(gray, f"ヒストグラム（しきい値 = {ret:.0f}）")
        flag = f"cv2.{type_name}" + (" + cv2.THRESH_OTSU" if otsu else "")
        common.show_code(f"""
import cv2

img = cv2.imread("sample.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, binary = cv2.threshold(gray, {0 if otsu else thresh}, 255, {flag})
print(ret)  # 実際に使われたしきい値
""")
        common.download(binary, "threshold.png")

    with tab_adaptive:
        c1, c2, c3 = st.columns(3)
        method = c1.selectbox("方法", list(image_ops.ADAPTIVE_METHODS))
        block = common.odd_slider("ブロックサイズ", 3, 101, 31, help="周囲何ピクセルを見てしきい値を決めるか（奇数）")
        c = c3.slider("C（しきい値から引く値）", -20, 40, 10)
        invert = c2.toggle("白黒反転", key="adaptive_invert")
        adaptive = image_ops.adaptive_threshold(gray, method, block, c, invert)
        common.show_pair(gray, adaptive, "グレースケール", "適応的二値化")
        type_ = "cv2.THRESH_BINARY_INV" if invert else "cv2.THRESH_BINARY"
        common.show_code(f"""
import cv2

img = cv2.imread("sample.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
binary = cv2.adaptiveThreshold(gray, 255, cv2.{method}, {type_}, {block}, {c})
""")
        common.download(adaptive, "adaptive_threshold.png")

    with tab_compare:
        st.caption("サンプル「文書（照明ムラあり）」で見ると違いがよく分かります。")
        _, fixed = image_ops.threshold(gray, 127)
        ret, otsu_img = image_ops.threshold(gray, 0, otsu=True)
        adaptive = image_ops.adaptive_threshold(gray, "ADAPTIVE_THRESH_GAUSSIAN_C", 31, 10)
        cols = st.columns(3)
        cols[0].image(fixed, caption="固定しきい値（127）", width="stretch")
        cols[1].image(otsu_img, caption=f"大津の二値化（{ret:.0f}）", width="stretch")
        cols[2].image(adaptive, caption="適応的二値化（Gaussian, 31, 10）", width="stretch")
