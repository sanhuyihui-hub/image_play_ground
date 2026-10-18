"""OpenCV 画像処理プレイグラウンド

起動: streamlit run app.py
"""

import cv2
import numpy as np
import streamlit as st

from core import samples
from views import contour, edge, grayscale, histmatch, hsv, lut, threshold

PAGES = {
    "グレースケール変換": grayscale,
    "二値化（Threshold）": threshold,
    "HSV色抽出": hsv,
    "エッジ検出": edge,
    "輪郭線抽出": contour,
    "LUT（ルックアップテーブル）": lut,
    "ヒストグラムマッチング": histmatch,
}

# 各ページで使いやすいサンプルの初期値
DEFAULT_SAMPLE = {
    "二値化（Threshold）": "文書（照明ムラあり）",
    "LUT（ルックアップテーブル）": "風景（暗め）",
    "ヒストグラムマッチング": "風景（暗め）",
}

MAX_SIDE = 1024  # 大きすぎる画像は縮小してスライダー操作を軽くする

st.set_page_config(page_title="OpenCV Playground", page_icon="🎛️", layout="wide")


@st.cache_data
def load_sample(name: str) -> np.ndarray:
    return samples.SAMPLES[name]()


@st.cache_data
def decode_upload(data: bytes) -> np.ndarray | None:
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None
    h, w = img.shape[:2]
    scale = MAX_SIDE / max(h, w)
    if scale < 1:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img


def sidebar() -> tuple[str, np.ndarray]:
    st.sidebar.title("🎛️ OpenCV Playground")
    st.sidebar.caption("パラメータを動かして、画像処理の結果をその場で確かめられます。")
    page = st.sidebar.radio("処理を選ぶ", list(PAGES))

    st.sidebar.divider()
    source = st.sidebar.radio("入力画像", ["サンプル", "アップロード"], horizontal=True)
    img = None
    if source == "アップロード":
        file = st.sidebar.file_uploader("画像ファイル", type=["png", "jpg", "jpeg", "bmp", "webp"])
        if file is not None:
            img = decode_upload(file.getvalue())
            if img is None:
                st.sidebar.error("画像として読み込めませんでした。")
    if img is None:
        names = list(samples.SAMPLES)
        default = names.index(DEFAULT_SAMPLE.get(page, names[0]))
        # ページごとに key を分け、ページを切り替えたときに初期値が効くようにする
        name = st.sidebar.selectbox("サンプル画像", names, index=default, key=f"sample_{page}")
        img = load_sample(name)

    h, w = img.shape[:2]
    st.sidebar.caption(f"サイズ: {w} × {h}")
    st.sidebar.divider()
    st.sidebar.markdown("[Qiita: @sanhuyihui](https://qiita.com/sanhuyihui)")
    return page, img


def main() -> None:
    page, img = sidebar()
    PAGES[page].render(img)


main()
