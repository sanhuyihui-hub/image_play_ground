import cv2
import numpy as np
import pandas as pd
import streamlit as st

from core import image_ops, samples
from views import common

ARTICLE = "https://qiita.com/sanhuyihui/items/06762d3c2e85270514b7"


def _load_reference() -> np.ndarray:
    choice = st.radio("参照画像", ["サンプル「風景（夕焼け）」", "アップロード"], horizontal=True)
    if choice == "アップロード":
        file = st.file_uploader("参照画像を選択", type=["png", "jpg", "jpeg", "bmp", "webp"], key="ref_upload")
        if file is not None:
            ref = cv2.imdecode(np.frombuffer(file.getvalue(), np.uint8), cv2.IMREAD_COLOR)
            if ref is not None:
                return ref
            st.error("画像として読み込めませんでした。サンプルを使います。")
    return samples.landscape(dark=False)


def render(img: np.ndarray) -> None:
    common.header(
        "ヒストグラムマッチング",
        "入力画像の明るさの分布を、参照画像に近づけます。CDF（累積分布関数）同士を対応させるのがポイントです。",
        ARTICLE,
    )
    ref = _load_reference()
    per_channel = st.toggle("BGR のチャンネルごとに合わせる（オフ = グレースケールで合わせる）", value=True)

    src = img if per_channel else image_ops.to_gray(img)
    matched = image_ops.match_histograms(src, ref)

    cols = st.columns(3)
    cols[0].image(common.to_rgb(src), caption="入力", width="stretch")
    cols[1].image(common.to_rgb(ref), caption="参照", width="stretch")
    cols[2].image(common.to_rgb(matched), caption="マッチング後", width="stretch")

    st.subheader("CDF で見る対応関係（輝度）")
    st.caption("マッチング後の CDF（赤）が参照画像の CDF（青）にほぼ重なれば成功です。")
    df = pd.DataFrame({
        "入力": image_ops.cdf(image_ops.to_gray(src)),
        "参照": image_ops.cdf(image_ops.to_gray(ref)),
        "マッチング後": image_ops.cdf(image_ops.to_gray(matched)),
    })
    st.line_chart(df, height=240, color=["#888888", "#3b6fd8", "#d84b3b"])

    c1, c2 = st.columns(2)
    with c1:
        common.histogram_chart(image_ops.to_gray(src), "入力のヒストグラム")
    with c2:
        common.histogram_chart(image_ops.to_gray(matched), "マッチング後のヒストグラム")

    common.show_code("""
import cv2
import numpy as np

def cdf(ch):
    hist = cv2.calcHist([ch], [0], None, [256], [0, 256]).ravel()
    c = hist.cumsum()
    return c / c[-1]

def match_channel(src, ref):
    # src の画素値ごとに、CDF の値が同じになる ref 側の画素値を探して LUT にする
    lut = np.searchsorted(cdf(ref), cdf(src)).clip(0, 255).astype(np.uint8)
    return cv2.LUT(src, lut)

src = cv2.imread("input.jpg")
ref = cv2.imread("reference.jpg")
result = cv2.merge([match_channel(s, r) for s, r in zip(cv2.split(src), cv2.split(ref))])
""")
    common.download(matched, "histogram_matching.png")
