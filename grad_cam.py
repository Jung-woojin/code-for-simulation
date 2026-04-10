# -*- coding: utf-8 -*-
"""
Grad-CAM Viewer (폴더 이미지 순회)
- 지정된 폴더 내 모든 이미지 파일을 순회
- 원본 + Grad-CAM 좌우 표시 (원본에만 파일명/클래스/확률 오버레이)
- 키: Space/N=다음, P=이전, ESC=종료
"""

import os
import glob
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import Xception
from tensorflow.keras.applications.xception import preprocess_input as xception_preprocess

# =========================
# 하드코딩 설정
# =========================
WEIGHTS_PATH   = r"C:\Users\ust21\label\final_test\Pohang_DL.h5"
IMAGE_FOLDER   = r"C:\Users\ust21\label\final_test\Pohang\auto_0\0"  # 이미지 폴더

INPUT_SHAPE    = (576, 704, 3)       # (H, W, C)  => (576,704)로 리사이즈 후 추론/표시
NUM_CLASSES    = 3
USE_XCEPTION_PREPROCESS = False      # False: 1/255, True: xception_preprocess
TARGET_LAYER_NAME = "block14_sepconv2"
CLASS_NAMES    = ["class0", "class1", "class2"]  # 실제 라벨명으로 교체 가능

WINDOW_NAME    = "Original (Left) | Grad-CAM (Right)"
SCREEN_W, SCREEN_H = 1920, 1080      # 화면 해상도 맞게 조절

# 지원하는 이미지 확장자
IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']

# =========================
# 모델 정의 (Xception 백본 + 헤드)
# =========================
def build_model(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES):
    base = Xception(include_top=False, weights=None, input_shape=input_shape)
    x = layers.GlobalAveragePooling2D(name="gap")(base.output)
    x = layers.Dropout(0.2, name="dropout")(x)
    out = layers.Dense(num_classes, activation='softmax',
                       kernel_initializer='zeros', name='dense')(x)
    model = models.Model(base.input, out, name="Xception_custom")
    return model

# =========================
# 폴더 내 이미지 파일 수집
# =========================
def get_image_files(folder_path):
    """폴더 내 모든 이미지 파일 경로 리스트 반환"""
    image_files = []
    for ext in IMAGE_EXTENSIONS:
        pattern = os.path.join(folder_path, f"*{ext}")
        image_files.extend(glob.glob(pattern))
        pattern = os.path.join(folder_path, f"*{ext.upper()}")
        image_files.extend(glob.glob(pattern))
    
    # 중복 제거 및 정렬
    image_files = sorted(list(set(image_files)))
    return image_files

# =========================
# 안전 로더
# =========================
def safe_imread(path: str):
    """이미지 안전하게 읽기"""
    try:
        data = np.fromfile(path, dtype=np.uint8)
        if data.size == 0:
            return None
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None

# =========================
# 전처리
# =========================
def prep_for_model(img_bgr):
    """(BGR, 임의크기) -> (RGB float 모델입력)"""
    img_for_model = cv2.resize(img_bgr, (INPUT_SHAPE[1], INPUT_SHAPE[0]), interpolation=cv2.INTER_LINEAR)
    img_rgb = cv2.cvtColor(img_for_model, cv2.COLOR_BGR2RGB).astype(np.float32)
    x = xception_preprocess(img_rgb.copy()) if USE_XCEPTION_PREPROCESS else (img_rgb / 255.0)
    x = np.expand_dims(x, axis=0)  # (1,H,W,3)
    return img_for_model, x

# =========================
# Grad-CAM
# =========================
def make_gradcam(model, inputs, target_layer_name=TARGET_LAYER_NAME, class_index=None):
    target_layer = model.get_layer(target_layer_name)
    grad_model = tf.keras.models.Model([model.inputs], [target_layer.output, model.output])

    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(inputs)  # conv_out: (1, Hc, Wc, C), preds: (1, num_classes)
        if class_index is None:
            class_index = int(tf.argmax(preds[0]).numpy())
        score = preds[:, class_index]

    grads = tape.gradient(score, conv_out)            # (1, Hc, Wc, C)
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))  # (C,)

    conv_out = conv_out[0]                             # (Hc, Wc, C)
    heatmap = tf.reduce_sum(conv_out * pooled_grads, axis=-1)  # (Hc, Wc)
    heatmap = tf.nn.relu(heatmap)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy(), class_index, preds.numpy()

def overlay_heatmap_on_image(heatmap, img_bgr, alpha=0.4):
    H, W = img_bgr.shape[:2]
    heat = cv2.resize(heatmap, (W, H))
    heat = np.uint8(255 * heat)
    heat_color = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_bgr, 1.0 - alpha, heat_color, alpha, 0)
    return overlay

# =========================
# 텍스트(원본에만)
# =========================
def put_centered_text(img_bgr, text, y, font=cv2.FONT_HERSHEY_DUPLEX,
                      font_scale=0.8, color=(0,0,155), thickness=2):
    H, W = img_bgr.shape[:2]
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    x = (W - tw) // 2
    cv2.putText(img_bgr, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)

# =========================
# 표시 (키는 외부에서 받음)
# =========================
def show_side_by_side(original_bgr, overlay_bgr, window_name=WINDOW_NAME):
    if original_bgr.shape[:2] != overlay_bgr.shape[:2]:
        H = min(original_bgr.shape[0], overlay_bgr.shape[0])
        W = min(original_bgr.shape[1], overlay_bgr.shape[1])
        original_bgr = cv2.resize(original_bgr, (W, H))
        overlay_bgr  = cv2.resize(overlay_bgr, (W, H))
    combined = np.hstack([original_bgr, overlay_bgr])

    scale = min(SCREEN_W / combined.shape[1], SCREEN_H / combined.shape[0], 1.0)
    if scale < 1.0:
        combined_show = cv2.resize(combined,
                                   (int(combined.shape[1]*scale), int(combined.shape[0]*scale)))
    else:
        combined_show = combined

    cv2.imshow(window_name, combined_show)
    return combined

def wait_for_valid_key(valid_keys):
    """유효 키가 눌릴 때까지 대기하고 해당 키코드 반환"""
    while True:
        k = cv2.waitKey(0) & 0xFF
        if k in valid_keys:
            return k

# =========================
# MAIN
# =========================
def main():
    # 1) 모델
    print(f"모델 로드 중: {WEIGHTS_PATH}")
    model = build_model()
    model.load_weights(WEIGHTS_PATH)
    print("모델 로드 완료")

    # 2) 이미지 파일 수집
    print(f"\n이미지 폴더: {IMAGE_FOLDER}")
    image_files = get_image_files(IMAGE_FOLDER)
    
    if not image_files:
        print("폴더에 이미지 파일이 없습니다.")
        return
    
    print(f"총 {len(image_files)}개의 이미지 파일 발견\n")
    print("[키 안내] Space/N: 다음, P: 이전, ESC: 종료")

    idx = 0
    while 0 <= idx < len(image_files):
        img_path = image_files[idx]
        filename = os.path.basename(img_path)
        
        print(f"[{idx+1}/{len(image_files)}] {filename}")
        
        # 이미지 로드
        img = safe_imread(img_path)
        
        if img is None:
            # 로드 실패: 안내 화면
            canvas = np.zeros((480, 960, 3), dtype=np.uint8)
            msg1 = "Failed to load image"
            msg2 = filename
            cv2.putText(canvas, msg1, (30, 80), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0,0,255), 2, cv2.LINE_AA)
            cv2.putText(canvas, msg2,  (30, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1, cv2.LINE_AA)
            cv2.putText(canvas, "[Space/N]=Next  [P]=Prev  [ESC]=Quit",
                        (30, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230,230,230), 1, cv2.LINE_AA)
            cv2.imshow(WINDOW_NAME, canvas)

            key = wait_for_valid_key({ord('n'), ord('N'), 32, ord('p'), ord('P'), 27})
            if key in (ord('n'), ord('N'), 32):   # 다음
                idx += 1
            elif key in (ord('p'), ord('P')):     # 이전
                idx = max(0, idx - 1)
            elif key == 27:                        # ESC
                break
            cv2.destroyAllWindows()
            continue

        # 3) 추론 준비/수행
        img_for_model, x = prep_for_model(img)
        heatmap, cls_idx, preds = make_gradcam(model, x, TARGET_LAYER_NAME, class_index=None)
        overlay = overlay_heatmap_on_image(heatmap, img_for_model, alpha=0.4)

        # 4) 원본에만 텍스트(파일명/클래스/확률)
        prob = float(preds[0][cls_idx])
        cls_name = CLASS_NAMES[cls_idx] if 0 <= cls_idx < len(CLASS_NAMES) else str(cls_idx)
        put_centered_text(img_for_model, filename, y=35, font_scale=0.8, color=(60,60,60), thickness=2)
        put_centered_text(img_for_model, f"class : {cls_name} ({cls_idx})", y=70, color=(0,0,155))
        put_centered_text(img_for_model, f"prob  : {prob:.4f}", y=100, color=(0,0,155))

        # 5) 표시
        show_side_by_side(img_for_model, overlay)

        # 6) 허용키만 입력
        key = wait_for_valid_key({ord('n'), ord('N'), 32, ord('p'), ord('P'), 27})
        if key in (ord('n'), ord('N'), 32):   # 다음
            idx += 1
        elif key in (ord('p'), ord('P')):     # 이전
            idx = max(0, idx - 1)
        elif key == 27:                        # ESC
            break

        cv2.destroyAllWindows()

    cv2.destroyAllWindows()
    print("종료")

if __name__ == "__main__":
    main()