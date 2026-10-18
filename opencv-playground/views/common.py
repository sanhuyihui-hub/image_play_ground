"""各ページで共通して使う表示部品。"""

from __future__ import annotations

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from core import image_ops


def to_rgb(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def header(title: str, summary: str, article_url: str | None = None) -> None:
    st.header(title)
    st.caption(summary)
    if article_url:
        st.markdown(f"📝 解説記事: [{article_url}]({article_url})")


def show_pair(before: np.ndarray, after: np.ndarray, before_label: str = "入力", after_label: str = "結果") -> None:
    left, right = st.columns(2)
    left.image(to_rgb(before), caption=before_label, width="stretch")
    right.image(to_rgb(after), caption=after_label, width="stretch")


def show_code(code: str) -> None:
    with st.expander("🧑‍💻 今のパラメータで書いたコード", expanded=False):
        st.code(code.strip(), language="python")


def download(img: np.ndarray, filename: str) -> None:
    ok, buf = cv2.imencode(".png", img)
    if ok:
        st.download_button("結果をPNGで保存", buf.tobytes(), file_name=filename, mime="image/png")


def histogram_chart(img: np.ndarray, title: str = "ヒストグラム") -> None:
    st.markdown(f"**{title}**")
    if img.ndim == 2:
        df = pd.DataFrame({"gray": image_ops.histogram(img)})
        st.line_chart(df, height=180, color=["#888888"])
    else:
        b, g, r = cv2.split(img)
        df = pd.DataFrame({"B": image_ops.histogram(b), "G": image_ops.histogram(g), "R": image_ops.histogram(r)})
        st.line_chart(df, height=180, color=["#3b6fd8", "#3aa655", "#d84b3b"])


def odd_slider(label: str, min_value: int, max_value: int, value: int, key: str | None = None, help: str | None = None) -> int:
    """カーネルサイズのように奇数しか取れない値用のスライダー。"""
    return st.slider(label, min_value, max_value, value, step=2, key=key, help=help)
