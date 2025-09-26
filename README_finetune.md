# YOLOv8 ファインチューニングガイド

## 概要
このスクリプトは、YOLOv8モデルをカスタムデータセットでファインチューニングするためのツールです。

## 必要なファイル

### 1. メインスクリプト
- `yolo_finetune.py` - ファインチューニングメインスクリプト
- `yolo_detection.py` - 推論用スクリプト

### 2. データセット構造
アノテーションデータは以下のYOLO形式で準備してください：

```
dataset/
├── train/
│   ├── images/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── labels/
│       ├── image1.txt
│       ├── image2.txt
│       └── ...
├── val/
│   ├── images/
│   │   ├── val_image1.jpg
│   │   └── ...
│   └── labels/
│       ├── val_image1.txt
│       └── ...
└── test/
    ├── images/
    │   ├── test_image1.jpg
    │   └── ...
    └── labels/
        ├── test_image1.txt
        └── ...
```

### 3. ラベルファイル形式
各ラベルファイル（.txt）は以下の形式で記述してください：

```
class_id center_x center_y width height
```

例：
```
0 0.5 0.5 0.3 0.4
1 0.2 0.3 0.1 0.2
```

- `class_id`: クラスID（0から開始）
- `center_x, center_y`: バウンディングボックスの中心座標（画像サイズで正規化、0-1）
- `width, height`: バウンディングボックスの幅と高さ（画像サイズで正規化、0-1）

## 使用方法

### 1. 基本的な使用方法
```bash
cd /Users/rinotsuka/code/papers/RTMdet/0926
source venv/bin/activate
python yolo_finetune.py
```

### 2. スクリプト内での設定
スクリプト実行時に以下の設定が可能です：

- **ベースモデル選択**: n/s/m/l/x
- **プロジェクト名**: 出力ディレクトリ名
- **データセットパス**: アノテーションデータのパス
- **トレーニング設定**:
  - エポック数（デフォルト: 100）
  - バッチサイズ（デフォルト: 16）
  - 学習率（デフォルト: 0.01）

### 3. プログラムでの使用方法
```python
from yolo_finetune import YOLOFinetuner

# ファインチューナーを初期化
finetuner = YOLOFinetuner(
    base_model='yolov8n.pt',  # ベースモデル
    project_name='my_project'  # プロジェクト名
)

# データセットをセットアップ
finetuner.setup_dataset('/path/to/dataset')

# トレーニング実行
results = finetuner.train(
    epochs=100,
    batch_size=16,
    lr0=0.01
)

# モデル検証
val_results = finetuner.validate()

# 推論実行
predictions = finetuner.predict('/path/to/test/images')
```

## 出力ファイル

トレーニング完了後、以下のファイルが生成されます：

```
project_name/
├── dataset.yaml          # データセット設定ファイル
├── train/               # トレーニング結果
│   ├── weights/
│   │   ├── best.pt      # 最良のモデル
│   │   └── last.pt      # 最終モデル
│   ├── results.png      # トレーニング結果グラフ
│   ├── confusion_matrix.png
│   └── ...
└── predict/             # 推論結果（predict実行時）
    └── ...
```

## トラブルシューティング

### 1. データセットエラー
- データセット構造が正しいか確認
- ラベルファイルの形式が正しいか確認
- 画像ファイルとラベルファイルの対応が正しいか確認

### 2. メモリエラー
- バッチサイズを小さくする
- 画像サイズを小さくする（imgszパラメータ）

### 3. CUDAエラー
- GPUが利用可能か確認
- PyTorchのCUDA対応版がインストールされているか確認

## 推論実行

トレーニング完了後、以下のコマンドで推論を実行できます：

```bash
python yolo_detection.py
```

または、プログラム内で：

```python
from ultralytics import YOLO

# トレーニング済みモデルをロード
model = YOLO('project_name/train/weights/best.pt')

# 推論実行
results = model('/path/to/image.jpg')
```

## 注意事項

1. **データセットサイズ**: 少なくとも数百枚の画像を用意することを推奨
2. **クラスバランス**: 各クラスの画像数をできるだけ均等にする
3. **データ品質**: 高品質なアノテーションを心がける
4. **検証データ**: トレーニングと独立した検証データを用意する
5. **GPU使用**: 可能であればGPUを使用してトレーニング時間を短縮する
