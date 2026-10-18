import numpy as np
import streamlit as st

from core import image_ops
from views import common

ARTICLE = "https://qiita.com/sanhuyihui/items/c04bb0879553e28ee212"


def render(img: np.ndarray) -> None:
    common.header("グレースケール変換", "カラー画像を明るさだけの1チャンネル画像にします。多くの処理の最初の一歩です。", ARTICLE)

    method = st.radio("変換方法", image_ops.GRAY_METHODS, horizontal=True)
    gray = image_ops.to_gray(img, method)
    common.show_pair(img, gray, after_label=f"グレースケール（{method}）")

    st.subheader("方法による違い")
    st.caption("cv2.cvtColor と加重平均はほぼ同じ結果になり、単純平均は緑が暗く・青が明るく出やすくなります。")
    ref = image_ops.to_gray(img, "cvtColor").astype(np.int16)
    cols = st.columns(len(image_ops.GRAY_METHODS))
    for col, m in zip(cols, image_ops.GRAY_METHODS):
        g = image_ops.to_gray(img, m)
        diff = np.abs(g.astype(np.int16) - ref).mean()
        col.image(g, caption=m, width="stretch")
        col.metric("cvtColor との平均差", f"{diff:.2f}")

    common.histogram_chart(gray)

    if method == "cvtColor":
        body = "gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)"
    elif method == "単純平均":
        body = "gray = img.mean(axis=2).round().astype(np.uint8)"
    else:
        body = """b, g, r = cv2.split(img.astype(np.float32))
gray = (0.299 * r + 0.587 * g + 0.114 * b).round().astype(np.uint8)"""
    common.show_code(f"""
import cv2
import numpy as np

img = cv2.imread("sample.jpg")
{body}
cv2.imwrite("gray.png", gray)
""")
    common.download(gray, "gray.png")
