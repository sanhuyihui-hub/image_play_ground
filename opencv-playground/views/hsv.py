import cv2
import numpy as np
import streamlit as st

from core import image_ops
from views import common

ARTICLE = "https://qiita.com/sanhuyihui/items/d9c651d48d653265628b"


def _hue_bar(h_lo: int, h_hi: int) -> np.ndarray:
    """H の範囲を色帯で表示する。選択外は暗くする。"""
    h = np.tile(np.arange(180, dtype=np.uint8), (20, 1))
    bar = cv2.cvtColor(cv2.merge([h, np.full_like(h, 255), np.full_like(h, 255)]), cv2.COLOR_HSV2BGR)
    x = np.arange(180)
    inside = (x >= h_lo) & (x <= h_hi) if h_lo <= h_hi else (x >= h_lo) | (x <= h_hi)
    bar[:, ~inside] = (bar[:, ~inside] * 0.25).astype(np.uint8)
    return cv2.resize(bar, (720, 24), interpolation=cv2.INTER_NEAREST)


def _apply_preset() -> None:
    name = st.session_state["hsv_preset"]
    if name in image_ops.HSV_PRESETS:
        (h1, s1, v1), (h2, s2, v2) = image_ops.HSV_PRESETS[name]
        st.session_state.update(h_lo=h1, h_hi=h2, s_range=(s1, s2), v_range=(v1, v2))


def render(img: np.ndarray) -> None:
    common.header("HSV色空間で色を抽出", "色相（H）・彩度（S）・明度（V）の範囲で、特定の色だけを取り出します。", ARTICLE)

    st.session_state.setdefault("h_lo", 170)
    st.session_state.setdefault("h_hi", 10)
    st.session_state.setdefault("s_range", (80, 255))
    st.session_state.setdefault("v_range", (60, 255))

    st.selectbox("プリセット", ["（手動）", *image_ops.HSV_PRESETS], key="hsv_preset", on_change=_apply_preset)
    c1, c2 = st.columns(2)
    h_lo = c1.slider("H 下限", 0, 179, key="h_lo")
    h_hi = c2.slider("H 上限", 0, 179, key="h_hi")
    st.image(common.to_rgb(_hue_bar(h_lo, h_hi)), width="stretch")
    if h_lo > h_hi:
        st.info("H の下限 > 上限なので、赤のように 179 と 0 をまたぐ範囲として扱っています（マスクを2つ作って OR）。")
    s_lo, s_hi = st.slider("S（彩度）", 0, 255, key="s_range")
    v_lo, v_hi = st.slider("V（明度）", 0, 255, key="v_range")

    lower, upper = (h_lo, s_lo, v_lo), (h_hi, s_hi, v_hi)
    mask = image_ops.hsv_mask(img, lower, upper)
    extracted = image_ops.apply_mask(img, mask)

    cols = st.columns(3)
    cols[0].image(common.to_rgb(img), caption="入力", width="stretch")
    cols[1].image(mask, caption="マスク", width="stretch")
    cols[2].image(common.to_rgb(extracted), caption="抽出結果", width="stretch")
    st.metric("抽出された画素の割合", f"{(mask > 0).mean() * 100:.1f} %")

    if h_lo <= h_hi:
        mask_code = f"mask = cv2.inRange(hsv, np.array({list(lower)}), np.array({list(upper)}))"
    else:
        mask_code = f"""# H が 179 と 0 をまたぐので2つに分ける
mask1 = cv2.inRange(hsv, np.array([{h_lo}, {s_lo}, {v_lo}]), np.array([179, {s_hi}, {v_hi}]))
mask2 = cv2.inRange(hsv, np.array([0, {s_lo}, {v_lo}]), np.array([{h_hi}, {s_hi}, {v_hi}]))
mask = cv2.bitwise_or(mask1, mask2)"""
    common.show_code(f"""
import cv2
import numpy as np

img = cv2.imread("sample.jpg")
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
{mask_code}
result = cv2.bitwise_and(img, img, mask=mask)
""")
    common.download(extracted, "hsv_extract.png")
