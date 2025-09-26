#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOv8による物体検出プログラム
- プログラム名: YOLOv8 による物体検出プログラム
- 特徴技術名: YOLOv8
- 特徴機能: リアルタイム物体検出、高精度な物体認識
- 学習済みモデル: YOLOv8n/s/m/l/x - COCO 2017データセットで事前学習済み
"""

import os
import cv2
import time
import torch
import urllib.request
import ssl
import numpy as np
import sys
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
import warnings

# tqdmプログレスバーを無効化
os.environ["TQDM_DISABLE"] = "1"

# SSL証明書検証を無効化（モデルダウンロード用）
ssl._create_default_https_context = ssl._create_unverified_context

# 重要でないUserWarningを最小限に抑制
warnings.filterwarnings("ignore", category=UserWarning)

# 設定定数
MODEL_CONFIGS = {
    'n': 'yolov8n.pt',
    's': 'yolov8s.pt', 
    'm': 'yolov8m.pt',
    'l': 'yolov8l.pt',
    'x': 'yolov8x.pt'
}
PRED_SCORE_THR = 0.3
FONT_SIZE = 20
SAMPLE_URL = 'https://raw.githubusercontent.com/opencv/opencv/master/samples/data/vtest.avi'

# GPU/CPU自動選択
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'デバイス: {str(device)}', flush=True)

# 変数
frame_count = 0
results_log = []
model = None

def get_font():
    """フォントを取得"""
    # macOS用のフォントパスを試す
    font_paths = [
        '/System/Library/Fonts/Helvetica.ttc',
        '/System/Library/Fonts/Arial.ttf',
        '/Library/Fonts/Arial.ttf',
        '/System/Library/Fonts/HelveticaNeue.ttc'
    ]
    
    for font_path in font_paths:
        try:
            return ImageFont.truetype(font_path, FONT_SIZE)
        except OSError:
            continue
    
    # フォールバック
    return ImageFont.load_default()

def video_frame_processing(frame):
    """1フレームを推論し、可視化フレームと検出結果文字列リスト、タイムスタンプを返す"""
    global frame_count
    current_time = time.time()
    frame_count += 1

    # YOLOv8で推論
    results = model(frame, verbose=False)
    result = results[0]
    
    # 検出結果を取得
    boxes = result.boxes
    obj_lines = []
    
    if boxes is not None and len(boxes) > 0:
        # バウンディングボックス、スコア、クラスを取得
        for box in boxes:
            # スコアが閾値以上のもののみ処理
            if box.conf.item() >= PRED_SCORE_THR:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                class_id = int(box.cls.item())
                score = box.conf.item()
                class_name = model.names[class_id]
                
                obj_lines.append(
                    f"{class_name} ({score:.2f}), x1={x1:.0f}, y1={y1:.0f}, x2={x2:.0f}, y2={y2:.0f}"
                )
    
    # 結果を描画
    vis_frame = result.plot()
    
    # 画面に日本語テキスト描画（検出数）
    font = get_font()
    img_pil = Image.fromarray(cv2.cvtColor(vis_frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    draw.text((10, 30), f"検出物体数: {len(obj_lines)}", font=font, fill=(0, 255, 0))
    vis_frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    return vis_frame, obj_lines, current_time

def init_model(model_choice_key):
    """モデルを初期化"""
    selected_model = MODEL_CONFIGS[model_choice_key]
    print(f"選択されたモデル: YOLOv8-{model_choice_key.upper()} ({selected_model})", flush=True)
    print("モデルを初期化中...", flush=True)
    yolo_model = YOLO(selected_model)
    print("YOLOv8モデルの初期化が完了した", flush=True)
    return yolo_model

# ガイダンス表示
print("概要: YOLOv8で物体検出を行う。各物体ごとに1行で出力する。", flush=True)
print("操作方法:", flush=True)
print("  1) モデルを選択する（n/s/m/l/x）", flush=True)
print("  2) 入力を選択する（0:動画ファイル, 1:カメラ, 2:サンプル動画）", flush=True)
print("  3) OpenCVウィンドウで結果を確認し、q キーで終了", flush=True)
print("注意事項: 初回実行時はモデルを自動ダウンロードする場合がある", flush=True)

# モデル選択
print("\nYOLOv8モデルを選択してください:", flush=True)
print("n: YOLOv8n (nano) - 最速、軽量", flush=True)
print("s: YOLOv8s (small) - バランス型", flush=True)
print("m: YOLOv8m (medium) - 中程度の精度", flush=True)
print("l: YOLOv8l (large) - 高精度", flush=True)
print("x: YOLOv8x (xlarge) - 最高精度", flush=True)
model_choice = input("モデル選択 (n/s/m/l/x): ").lower().strip()
if model_choice not in MODEL_CONFIGS:
    print("無効な選択。YOLOv8nを使用する", flush=True)
    model_choice = 'n'
model = init_model(model_choice)

# 入力選択
print("0: 動画ファイル", flush=True)
print("1: カメラ", flush=True)
print("2: サンプル動画", flush=True)
choice = input("選択: ").strip()

temp_file = None
if choice == '0':
    print("動画ファイルのパスを入力してください:", flush=True)
    path = input("ファイルパス: ").strip()
    if not path or not os.path.exists(path):
        print("有効な動画ファイルが指定されなかったため終了する", flush=True)
        raise SystemExit
    cap = cv2.VideoCapture(path)
elif choice == '1':
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
else:
    print("サンプル動画をダウンロード中...", flush=True)
    SAMPLE_FILE = 'vtest.avi'
    urllib.request.urlretrieve(SAMPLE_URL, SAMPLE_FILE)
    temp_file = SAMPLE_FILE
    cap = cv2.VideoCapture(SAMPLE_FILE)

if not cap.isOpened():
    print('動画ファイル・カメラを開けなかった', flush=True)
    if temp_file and os.path.exists(temp_file):
        os.remove(temp_file)
    raise SystemExit

# メイン処理
MAIN_FUNC_DESC = "YOLOv8 物体検出"
print('\n=== 動画処理開始 ===', flush=True)
print('操作方法:', flush=True)
print('  q キー: プログラム終了', flush=True)
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame, obj_lines, current_time = video_frame_processing(frame)
        cv2.imshow(MAIN_FUNC_DESC, processed_frame)

        # 物体単位1行。カメラ=日付を含む現地時刻(YYYY-MM-DD HH:MM:SS.mmm)、動画=フレーム番号(1開始)。
        if choice == '1':  # カメラ
            ts = datetime.fromtimestamp(current_time).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            for line in obj_lines:
                print(ts, line, flush=True)
                results_log.append(f"{ts} {line}")
        else:  # 動画ファイル/サンプル動画
            for line in obj_lines:
                print(frame_count, line, flush=True)
                results_log.append(f"{frame_count} {line}")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
    if results_log:
        with open('result.txt', 'w', encoding='utf-8') as f:
            f.write('=== 結果 ===\n')
            f.write(f'処理フレーム数: {frame_count}\n')
            f.write(f'使用デバイス: {str(device).upper()}\n')
            if device.type == 'cuda':
                f.write(f'GPU: {torch.cuda.get_device_name(0)}\n')
            f.write('\n')
            f.write('\n'.join(results_log))
        print('処理結果をresult.txtに保存しました', flush=True)
    if temp_file and os.path.exists(temp_file):
        os.remove(temp_file)
    print('\n=== プログラム終了 ===', flush=True)
