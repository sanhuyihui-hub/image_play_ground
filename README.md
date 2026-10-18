# image_play_ground

# Qiita 連動プロジェクト

Qiita（[@sanhuyihui](https://qiita.com/sanhuyihui)）で書いている画像処理・物体検出の記事と連動して作ったシステムをまとめたリポジトリです。

記事では仕組みや考え方を解説し、このリポジトリでは**実際に動くもの**を公開しています。

## プロジェクト一覧

| プロジェクト | 概要 | 主な技術 |
|---|---|---|
| [opencv-playground](./opencv-playground/) | スライダーでパラメータを動かすと、OpenCV の画像処理の結果がその場で変わる Web アプリ | Python / OpenCV / Streamlit |

---

## opencv-playground

Qiita の「OpenCVで学ぶ画像処理入門」シリーズを、**読むだけでなく触って確かめられる**ようにした Web アプリです。

### 作った理由

画像処理の記事では「しきい値を 127 にすると…」のように説明しますが、値を変えたら結果がどう変わるのかは、文章やコードだけでは伝わりにくいです。
そこで、パラメータをスライダーで動かしながら結果を見比べられるアプリを作りました。記事の内容を自分で手を動かして確かめる、という目的もあります。

### 特徴

- **7種類の処理を1つのアプリで**：グレースケール変換、二値化、HSV色抽出、エッジ検出、輪郭線抽出、LUT、ヒストグラムマッチング
- **今のパラメータで書いたコードを表示**：画面で試した設定を、そのまま自分のプログラムに貼って使える
- **手法を並べて比較**：固定しきい値・大津の二値化・適応的二値化や、Canny・Sobel・Laplacian などを並べて違いを確認できる
- **サンプル画像はコードで生成**：著作権を気にせず、記事のスクリーンショットにも使える

### 対応する Qiita 記事

| ページ | 記事 |
|---|---|
| グレースケール変換 | [OpenCVで学ぶ画像処理入門：グレースケール変換の基礎と実装方法](https://qiita.com/sanhuyihui/items/c04bb0879553e28ee212) |
| 二値化 | [OpenCVで学ぶ画像処理入門：二値化（Threshold）の基礎と実装方法](https://qiita.com/sanhuyihui/items/2513e1cf8ed9a6d1d816) |
| HSV色抽出 | [OpenCVで学ぶ画像処理入門：HSV色空間の基礎と色抽出の実装方法](https://qiita.com/sanhuyihui/items/d9c651d48d653265628b) |
| エッジ検出 | [OpenCVで学ぶ画像処理入門：エッジ検出（Edge Detection）の基礎と実装方法](https://qiita.com/sanhuyihui/items/82abf5a190be1d21ddec) |
| 輪郭線抽出 | [OpenCVで学ぶ画像処理入門：輪郭線抽出（Contour Detection）の基礎と実装方法](https://qiita.com/sanhuyihui/items/a2d7e7a5832e10f53e29) |
| LUT | [LUT（Look-Up Table）による画素値変換をわかりやすく解説](https://qiita.com/sanhuyihui/items/15693ebbd47da94f383a) |
| ヒストグラムマッチング | [ヒストグラムマッチングでCDF（累積分布関数）を使う理由をわかりやすく解説](https://qiita.com/sanhuyihui/items/06762d3c2e85270514b7) |

### 設計で工夫したこと

- **画像処理と画面を分離**：処理の本体（`core/`）は Streamlit に依存しない関数として書き、画面（`views/`）から呼び出す形にしました。処理だけを単体でテストでき、別のアプリからも再利用できます。
- **テストで記事の内容を確認**：「手計算の加重平均は `cv2.cvtColor` と一致する」「赤の抽出では H が 179 と 0 をまたぐ範囲も拾える」など、記事で説明している性質を pytest で確かめています。
- **実際につまずく点をアプリに反映**：たとえば輪郭抽出では、黄色の図形はグレースケールにすると背景と区別できず検出されません。こうした「うまくいかない例」もそのまま見せ、解決方法（HSV でマスクを作る）を案内しています。

### 動かし方

Windows / Python 3.11 で動作を確認しています。

```bash
cd opencv-playground
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m streamlit run app.py
```

ブラウザで http://localhost:8501 が開きます。詳しくは [opencv-playground/README.md](./opencv-playground/README.md) を見てください。

---

## リポジトリ構成

```
Qiita/
├── README.md              # このファイル（全体の紹介）
└── opencv-playground/     # OpenCV 画像処理プレイグラウンド
    ├── README.md
    ├── app.py
    ├── core/              # 画像処理の本体
    ├── views/             # 各ページの画面
    └── tests/
```

## 作者

- Qiita: [@sanhuyihui](https://qiita.com/sanhuyihui)
