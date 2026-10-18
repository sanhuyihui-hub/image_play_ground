from dataclasses import asdict

import numpy as np
import pandas as pd
import streamlit as st

from core import image_ops
from views import common

ARTICLE = "https://qiita.com/sanhuyihui/items/a2d7e7a5832e10f53e29"


def render(img: np.ndarray) -> None:
    common.header("輪郭線抽出", "二値化した画像から物体の輪郭を見つけ、面積や外接矩形を求めます。", ARTICLE)
    gray = image_ops.to_gray(img)

    st.markdown("**1. 二値化**")
    c1, c2, c3 = st.columns(3)
    otsu = c1.toggle("大津の二値化", value=True)
    invert = c2.toggle("白黒反転", value=True, help="findContours は白い領域を物体とみなします。背景が明るい画像では反転が必要です")
    thresh = c3.slider("しきい値", 0, 255, 127, disabled=otsu)
    type_name = "THRESH_BINARY_INV" if invert else "THRESH_BINARY"
    ret, binary = image_ops.threshold(gray, thresh, 255, type_name, otsu)

    st.markdown("**2. 輪郭の検出**")
    c1, c2, c3 = st.columns(3)
    mode = c1.selectbox("取得モード", list(image_ops.RETRIEVAL_MODES), help="RETR_TREE / RETR_LIST にすると穴の輪郭も取れます")
    approx = c2.selectbox("近似方法", list(image_ops.APPROX_METHODS))
    min_area = c3.number_input("最小面積（これ未満は除外）", 0, 100000, 200, step=50)
    c1, c2 = st.columns(2)
    draw_bbox = c1.toggle("外接矩形を描く", value=True)
    label = c2.toggle("番号を描く", value=True)

    contours, infos = image_ops.find_contours(binary, mode, approx, min_area)
    drawn = image_ops.draw_contours(img, contours, draw_bbox, label)

    common.show_pair(binary, drawn, f"二値化（しきい値 {ret:.0f}）", f"輪郭 {len(contours)} 個")
    st.caption(
        "💡 サンプル「図形」では黄色の楕円が見つかりません。黄色は明るさが背景に近く、グレースケールにすると区別できないためです。"
        "色で分けたいときは「HSV色抽出」でマスクを作ってから輪郭を取ります。"
    )
    if infos:
        df = pd.DataFrame([asdict(i) for i in infos]).rename(
            columns={"index": "番号", "area": "面積", "perimeter": "周囲長", "w": "幅", "h": "高さ", "vertices": "近似頂点数"}
        )
        st.dataframe(df.round(1), hide_index=True, width="stretch")
        st.caption("近似頂点数は cv2.approxPolyDP(周囲長 × 0.02) で求めたもの。3 なら三角形、4 なら四角形の目安になります。")

    flag = f"cv2.{type_name}" + (" + cv2.THRESH_OTSU" if otsu else "")
    common.show_code(f"""
import cv2

img = cv2.imread("sample.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, {0 if otsu else thresh}, 255, {flag})

contours, hierarchy = cv2.findContours(binary, cv2.{mode}, cv2.{approx})
contours = [c for c in contours if cv2.contourArea(c) >= {min_area}]

result = img.copy()
cv2.drawContours(result, contours, -1, (0, 255, 0), 2)
for c in contours:
    x, y, w, h = cv2.boundingRect(c)
    cv2.rectangle(result, (x, y), (x + w, y + h), (0, 0, 255), 2)
print(len(contours), "個の輪郭")
""")
    common.download(drawn, "contours.png")
