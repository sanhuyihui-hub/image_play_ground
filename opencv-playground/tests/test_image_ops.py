import cv2
import numpy as np
import pytest

from core import image_ops, samples


@pytest.fixture
def color_img():
    return samples.shapes()


def test_manual_weighted_gray_matches_cvtcolor(color_img):
    a = image_ops.to_gray(color_img, "cvtColor").astype(int)
    b = image_ops.to_gray(color_img, "加重平均（手計算）").astype(int)
    assert np.abs(a - b).max() <= 1


def test_threshold_binary_values():
    gray = np.array([[50, 80, 120, 180, 220]], np.uint8)
    _, binary = image_ops.threshold(gray, 127)
    assert binary.tolist() == [[0, 0, 0, 255, 255]]


def test_otsu_returns_threshold_between_two_peaks():
    gray = np.concatenate([np.full(500, 40), np.full(500, 200)]).astype(np.uint8).reshape(10, 100)
    ret, _ = image_ops.threshold(gray, 0, otsu=True)
    assert 40 <= ret < 200


def test_adaptive_threshold_rejects_even_block():
    with pytest.raises(ValueError):
        image_ops.adaptive_threshold(np.zeros((10, 10), np.uint8), "ADAPTIVE_THRESH_MEAN_C", 4, 2)


def test_hsv_mask_red_wraps_around():
    img = np.zeros((1, 3, 3), np.uint8)
    img[0, 0] = (0, 0, 255)  # 赤 (H=0)
    img[0, 1] = (40, 0, 255)  # 赤寄りのマゼンタ (H≈175)
    img[0, 2] = (255, 0, 0)  # 青
    lower, upper = image_ops.HSV_PRESETS["赤"]
    mask = image_ops.hsv_mask(img, lower, upper)
    assert mask[0].tolist() == [255, 255, 0]


def test_find_contours_counts_shapes(color_img):
    gray = image_ops.to_gray(color_img)
    _, binary = image_ops.threshold(gray, 0, type_name="THRESH_BINARY_INV", otsu=True)
    contours, infos = image_ops.find_contours(binary, "RETR_EXTERNAL", min_area=500)
    assert len(contours) == len(infos) >= 5


def test_lut_gamma_one_is_identity():
    assert np.array_equal(image_ops.make_lut("ガンマ補正", 1.0), np.arange(256, dtype=np.uint8))


@pytest.mark.parametrize("levels", [2, 4, 8])
def test_lut_posterize_level_count(levels):
    assert len(np.unique(image_ops.make_lut("ポスタリゼーション", levels))) == levels


def test_histogram_matching_moves_cdf_toward_reference():
    src = samples.landscape(dark=True)
    ref = samples.landscape(dark=False)
    out = image_ops.match_histograms(src, ref)
    g = lambda x: image_ops.to_gray(x)
    before = np.abs(image_ops.cdf(g(src)) - image_ops.cdf(g(ref))).mean()
    after = np.abs(image_ops.cdf(g(out)) - image_ops.cdf(g(ref))).mean()
    assert after < before * 0.3


@pytest.mark.parametrize("name", list(samples.SAMPLES))
def test_samples_are_bgr_uint8(name):
    img = samples.SAMPLES[name]()
    assert img.shape == (samples.H, samples.W, 3) and img.dtype == np.uint8


def test_edges_return_uint8(color_img):
    gray = image_ops.to_gray(color_img)
    for out in (image_ops.canny(gray, 50, 150), image_ops.sobel(gray), image_ops.laplacian(gray)):
        assert out.dtype == np.uint8 and out.shape == gray.shape
    assert cv2.countNonZero(image_ops.canny(gray, 50, 150)) > 0
