import numpy as np
import pandas as pd
import streamlit as st

from core import image_ops
from views import common

ARTICLE = "https://qiita.com/sanhuyihui/items/15693ebbd47da94f383a"


def render(img: np.ndarray) -> None:
    common.header("LUT（ルックアップテーブル）", "「入力の画素値 → 出力の画素値」の対応表を先に作り、cv2.LUT で一気に変換します。", ARTICLE)

    kind = st.radio("変換の種類", image_ops.LUT_KINDS, horizontal=True)
    if kind == "ガンマ補正":
        param = st.slider("γ（1より大きいと明るく、小さいと暗く）", 0.1, 3.0, 1.8, 0.05)
        lut_code = f"lut = (255 * (x / 255) ** (1 / {param})).round()"
    elif kind == "コントラスト（S字）":
        param = st.slider("強さ", 1.0, 20.0, 8.0, 0.5)
        lut_code = f"""y = 255 / (1 + np.exp(-{param} * (x - 127.5) / 255))
lut = ((y - y.min()) / (y.max() - y.min()) * 255).round()"""
    elif kind == "ネガポジ反転":
        param = 0
        lut_code = "lut = 255 - x"
    else:
        param = st.slider("階調数", 2, 16, 4)
        lut_code = f"lut = np.floor(x / (256 / {param})) * (255 / ({param} - 1))"

    lut = image_ops.make_lut(kind, param)
    out = image_ops.apply_lut(img, lut)
    common.show_pair(img, out, after_label=f"{kind} の結果")

    left, right = st.columns([1, 1])
    with left:
        st.markdown("**変換カーブ（横: 入力 / 縦: 出力）**")
        st.line_chart(pd.DataFrame({"LUT": lut, "変換なし": np.arange(256)}), height=220, color=["#d84b3b", "#bbbbbb"])
    with right:
        common.histogram_chart(image_ops.to_gray(out), "変換後のヒストグラム（輝度）")

    with st.expander("LUT の中身（先頭と末尾）"):
        idx = [*range(0, 8), *range(248, 256)]
        st.dataframe(pd.DataFrame({"入力": idx, "出力": lut[idx]}).T, width="stretch")

    common.show_code(f"""
import cv2
import numpy as np

img = cv2.imread("sample.jpg")
x = np.arange(256, dtype=np.float64)
{lut_code}
lut = np.clip(lut, 0, 255).astype(np.uint8)
result = cv2.LUT(img, lut)
""")
    common.download(out, "lut.png")
