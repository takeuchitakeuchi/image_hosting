#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOv8 ファインチューニングスクリプト
- プログラム名: YOLOv8 ファインチューニングプログラム
- 特徴技術名: YOLOv8 Fine-tuning
- 特徴機能: カスタムデータセットでのYOLOv8モデルのファインチューニング
- 前提条件: YOLO形式のアノテーションデータが準備済み
"""

import os
import yaml
import shutil
from pathlib import Path
from ultralytics import YOLO
import torch
import warnings

# 警告を抑制
warnings.filterwarnings("ignore")

class YOLOFinetuner:
    def __init__(self, base_model='yolov8n.pt', project_name='yolo_finetune'):
        """
        YOLOv8ファインチューニングクラス
        
        Args:
            base_model (str): ベースモデル（yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt）
            project_name (str): プロジェクト名
        """
        self.base_model = base_model
        self.project_name = project_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'デバイス: {self.device}')
        
        # プロジェクトディレクトリを作成
        self.project_dir = Path(project_name)
        self.project_dir.mkdir(exist_ok=True)
        
        # データセットディレクトリ
        self.dataset_dir = self.project_dir / 'dataset'
        self.train_dir = self.dataset_dir / 'train'
        self.val_dir = self.dataset_dir / 'val'
        self.test_dir = self.dataset_dir / 'test'
        
        # 各ディレクトリを作成
        for dir_path in [self.train_dir, self.val_dir, self.test_dir]:
            (dir_path / 'images').mkdir(parents=True, exist_ok=True)
            (dir_path / 'labels').mkdir(parents=True, exist_ok=True)
    
    def setup_dataset(self, data_path, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
        """
        データセットをYOLO形式でセットアップ
        
        Args:
            data_path (str): アノテーションデータのパス
            train_ratio (float): トレーニングデータの割合
            val_ratio (float): 検証データの割合
            test_ratio (float): テストデータの割合
        """
        print("データセットをセットアップ中...")
        
        # データパスをPathオブジェクトに変換
        data_path = Path(data_path)
        
        if not data_path.exists():
            raise ValueError(f"データパスが存在しません: {data_path}")
        
        # 既存のYOLO形式データセットの場合
        if (data_path / 'train' / 'images').exists() and (data_path / 'train' / 'labels').exists():
            print("既存のYOLO形式データセットをコピー中...")
            self._copy_yolo_dataset(data_path)
        else:
            print("データセット構造を確認してください。YOLO形式のディレクトリ構造が必要です。")
            print("期待される構造:")
            print("  dataset/")
            print("    ├── train/")
            print("    │   ├── images/")
            print("    │   └── labels/")
            print("    ├── val/")
            print("    │   ├── images/")
            print("    │   └── labels/")
            print("    └── test/")
            print("        ├── images/")
            print("        └── labels/")
            return False
        
        # データセット設定ファイルを作成
        self.create_dataset_yaml()
        return True
    
    def _copy_yolo_dataset(self, source_path):
        """YOLO形式データセットをコピー"""
        source_path = Path(source_path)
        
        for split in ['train', 'val', 'test']:
            src_split = source_path / split
            if src_split.exists():
                dst_split = self.dataset_dir / split
                if dst_split.exists():
                    shutil.rmtree(dst_split)
                shutil.copytree(src_split, dst_split)
                print(f"{split} データをコピーしました")
    
    def create_dataset_yaml(self):
        """データセット設定YAMLファイルを作成"""
        # クラス名を取得（labelsディレクトリから）
        class_names = self._get_class_names()
        
        dataset_config = {
            'path': str(self.dataset_dir.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'nc': len(class_names),
            'names': class_names
        }
        
        yaml_path = self.project_dir / 'dataset.yaml'
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(dataset_config, f, default_flow_style=False, allow_unicode=True)
        
        print(f"データセット設定ファイルを作成しました: {yaml_path}")
        return yaml_path
    
    def _get_class_names(self):
        """ラベルファイルからクラス名を取得"""
        class_names = set()
        
        for split in ['train', 'val', 'test']:
            labels_dir = self.dataset_dir / split / 'labels'
            if labels_dir.exists():
                for label_file in labels_dir.glob('*.txt'):
                    with open(label_file, 'r') as f:
                        for line in f:
                            if line.strip():
                                class_id = int(line.split()[0])
                                class_names.add(f'class_{class_id}')
        
        # クラス名をソート
        sorted_classes = sorted(list(class_names))
        if not sorted_classes:
            # デフォルトクラス名
            sorted_classes = ['class_0', 'class_1', 'class_2']
        
        print(f"検出されたクラス数: {len(sorted_classes)}")
        print(f"クラス名: {sorted_classes}")
        return sorted_classes
    
    def train(self, epochs=100, imgsz=640, batch_size=16, lr0=0.01, 
              patience=50, save_period=10, device=None, workers=8):
        """
        ファインチューニングを実行
        
        Args:
            epochs (int): エポック数
            imgsz (int): 画像サイズ
            batch_size (int): バッチサイズ
            lr0 (float): 初期学習率
            patience (int): Early stopping patience
            save_period (int): モデル保存間隔
            device (str): デバイス ('cpu', 'cuda', '0', '1', etc.)
            workers (int): データローダーワーカー数
        """
        print("ファインチューニングを開始...")
        
        # デバイス設定
        if device is None:
            device = self.device
        
        # モデルをロード
        model = YOLO(self.base_model)
        print(f"ベースモデル: {self.base_model}")
        
        # データセット設定ファイルのパス
        dataset_yaml = self.project_dir / 'dataset.yaml'
        
        # トレーニング実行
        results = model.train(
            data=str(dataset_yaml),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch_size,
            lr0=lr0,
            patience=patience,
            save_period=save_period,
            device=device,
            workers=workers,
            project=str(self.project_dir),
            name='train',
            exist_ok=True,
            pretrained=True,
            optimizer='AdamW',
            cos_lr=True,
            close_mosaic=10,
            resume=False,
            amp=True,
            fraction=1.0,
            profile=False,
            freeze=None,
            multi_scale=False,
            overlap_mask=True,
            mask_ratio=4,
            drop_path=0.0,
            verbose=True
        )
        
        print("ファインチューニング完了!")
        return results
    
    def validate(self, model_path=None):
        """モデルを検証"""
        if model_path is None:
            # 最新のモデルを取得
            model_path = self.project_dir / 'train' / 'weights' / 'best.pt'
        
        if not Path(model_path).exists():
            print(f"モデルファイルが見つかりません: {model_path}")
            return None
        
        model = YOLO(model_path)
        dataset_yaml = self.project_dir / 'dataset.yaml'
        
        print("モデル検証中...")
        results = model.val(data=str(dataset_yaml))
        return results
    
    def predict(self, source, model_path=None, conf=0.25, iou=0.7, save=True):
        """推論を実行"""
        if model_path is None:
            model_path = self.project_dir / 'train' / 'weights' / 'best.pt'
        
        if not Path(model_path).exists():
            print(f"モデルファイルが見つかりません: {model_path}")
            return None
        
        model = YOLO(model_path)
        
        print(f"推論実行中: {source}")
        results = model.predict(
            source=source,
            conf=conf,
            iou=iou,
            save=save,
            project=str(self.project_dir),
            name='predict',
            exist_ok=True
        )
        
        return results

def main():
    """メイン関数"""
    print("=== YOLOv8 ファインチューニングスクリプト ===")
    print("使用方法:")
    print("1. データセットをYOLO形式で準備")
    print("2. このスクリプトを実行")
    print("3. トレーニングが完了すると、モデルが保存されます")
    print()
    
    # ベースモデル選択
    print("ベースモデルを選択してください:")
    print("n: YOLOv8n (nano) - 最速、軽量")
    print("s: YOLOv8s (small) - バランス型")
    print("m: YOLOv8m (medium) - 中程度の精度")
    print("l: YOLOv8l (large) - 高精度")
    print("x: YOLOv8x (xlarge) - 最高精度")
    
    model_choice = input("モデル選択 (n/s/m/l/x): ").lower().strip()
    model_map = {'n': 'yolov8n.pt', 's': 'yolov8s.pt', 'm': 'yolov8m.pt', 
                 'l': 'yolov8l.pt', 'x': 'yolov8x.pt'}
    
    if model_choice not in model_map:
        print("無効な選択。YOLOv8nを使用します")
        model_choice = 'n'
    
    base_model = model_map[model_choice]
    
    # プロジェクト名
    project_name = input("プロジェクト名を入力してください (デフォルト: yolo_finetune): ").strip()
    if not project_name:
        project_name = 'yolo_finetune'
    
    # ファインチューナーを初期化
    finetuner = YOLOFinetuner(base_model=base_model, project_name=project_name)
    
    # データセットパス
    dataset_path = input("データセットのパスを入力してください: ").strip()
    if not dataset_path:
        print("データセットパスが指定されていません")
        return
    
    # データセットセットアップ
    if not finetuner.setup_dataset(dataset_path):
        print("データセットのセットアップに失敗しました")
        return
    
    # トレーニング設定
    print("\nトレーニング設定:")
    epochs = input("エポック数 (デフォルト: 100): ").strip()
    epochs = int(epochs) if epochs else 100
    
    batch_size = input("バッチサイズ (デフォルト: 16): ").strip()
    batch_size = int(batch_size) if batch_size else 16
    
    lr = input("学習率 (デフォルト: 0.01): ").strip()
    lr = float(lr) if lr else 0.01
    
    # トレーニング実行
    try:
        results = finetuner.train(
            epochs=epochs,
            batch_size=batch_size,
            lr0=lr
        )
        
        print("\n=== トレーニング完了 ===")
        print(f"プロジェクトディレクトリ: {finetuner.project_dir}")
        print(f"最良のモデル: {finetuner.project_dir}/train/weights/best.pt")
        print(f"最終モデル: {finetuner.project_dir}/train/weights/last.pt")
        
        # 検証実行
        print("\nモデル検証中...")
        val_results = finetuner.validate()
        
        if val_results:
            print("検証完了!")
            print(f"mAP50: {val_results.box.map50:.3f}")
            print(f"mAP50-95: {val_results.box.map:.3f}")
        
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
