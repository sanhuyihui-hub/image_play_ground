# 🎛️ OpenCV Playground

スライダーでパラメータを動かすと、OpenCV の画像処理の結果がその場で変わる Web アプリです。
Qiita で書いている「OpenCVで学ぶ画像処理入門」シリーズの、**実際に触って確かめられる版**として作りました。

<!-- TODO: 動作の GIF を docs/demo.gif に置いてここに貼る -->
<!-- ![demo](docs/demo.gif) -->

## できること

| ページ | 内容 | 解説記事 |
|---|---|---|
| グレースケール変換 | cvtColor / 単純平均 / 加重平均の比較 | [Qiita](https://qiita.com/sanhuyihui/items/c04bb0879553e28ee212) |
| 二値化 | 固定しきい値・大津の二値化・適応的二値化の比較 | [Qiita](https://qiita.com/sanhuyihui/items/2513e1cf8ed9a6d1d816) |
| HSV色抽出 | H/S/V の範囲指定、赤のような 0/179 をまたぐ色にも対応 | [Qiita](https://qiita.com/sanhuyihui/items/d9c651d48d653265628b) |
| エッジ検出 | Canny / Sobel / Laplacian、前処理のぼかしの効果 | [Qiita](https://qiita.com/sanhuyihui/items/82abf5a190be1d21ddec) |
| 輪郭線抽出 | 取得モードの違い、面積・外接矩形・近似頂点数の一覧 | [Qiita](https://qiita.com/sanhuyihui/items/a2d7e7a5832e10f53e29) |
| LUT | ガンマ補正・S字コントラスト・ネガポジ・ポスタリゼーションと変換カーブ | [Qiita](https://qiita.com/sanhuyihui/items/15693ebbd47da94f383a) |
| ヒストグラムマッチング | CDF を使った変換と、CDF の重なり具合の可視化 | [Qiita](https://qiita.com/sanhuyihui/items/06762d3c2e85270514b7) |

どのページにも次の機能があります。

- **今のパラメータで書いたコード**を表示（コピーしてそのまま使える）
- 結果を PNG で保存
- 自分の画像をアップロードして試せる（サンプル画像はコードで生成しているので、著作権を気にせず使えます）

## 動かし方

Python 3.11 で動作を確認しています。

```bash
python -m venv .venv
.venv\Scripts\activate        # macOS / Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

ブラウザで http://localhost:8501 が開きます。

### テスト

```bash
pip install pytest
pytest
```

## 構成

```
opencv-playground/
├── app.py              # 入口。サイドバー（ページ選択・入力画像）
├── core/
│   ├── image_ops.py    # 画像処理の本体（Streamlit に依存しない純粋な関数）
│   └── samples.py      # サンプル画像の生成
├── views/              # 各ページの UI
│   ├── common.py       # 画像の並列表示・ヒストグラム・コード表示などの共通部品
│   ├── grayscale.py  threshold.py  hsv.py  edge.py
│   └── contour.py  lut.py  histmatch.py
└── tests/              # core のユニットテスト
```

画像処理（`core/`）と画面（`views/`）を分けているので、処理だけをテストしたり、別のアプリから再利用したりできます。

## 技術スタック

- Python 3.11
- OpenCV 4.x（`opencv-python-headless`）
- Streamlit
- NumPy / pandas
- pytest

## 今後やりたいこと

- [ ] Streamlit Community Cloud で公開する
- [ ] モルフォロジー演算（膨張・収縮）、平滑化フィルタのページを追加
- [ ] 処理を順につなげる「パイプライン」モード（例: グレースケール → ぼかし → 二値化 → 輪郭）

## 作者

- Qiita: [@sanhuyihui](https://qiita.com/sanhuyihui)
