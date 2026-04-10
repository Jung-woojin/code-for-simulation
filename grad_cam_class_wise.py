# -*- coding: utf-8 -*-
"""
Grad-CAM Viewer (전체 클래스별)
- 원본 + Class 0, 1, 2 Grad-CAM을 4개 나란히 표시
- 각 클래스별로 모델이 어디를 보는지 비교
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
WEIGHTS_PATH   = r"C:\Users\ust21\label\final_test\PTDJ_DL.h5"
IMAGE_FOLDER   = r"C:\Users\ust21\label\final_test\PTDJ\auto_2\1"

INPUT_SHAPE    = (576, 704, 3)
NUM_CLASSES    = 3
USE_XCEPTION_PREPROCESS = False
TARGET_LAYER_NAME = "block14_sepconv2"
CLASS_NAMES    = ["class0", "class1", "class2"]

WINDOW_NAME    = "Original | Class0 CAM | Class1 CAM | Class2 CAM"
SCREEN_W, SCREEN_H = 1920, 1080

IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']

# =========================
# 모델 정의
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
# 이미지 수집
# =========================
def get_image_files(folder_path):
    image_files = []
    for ext in IMAGE_EXTENSIONS:
        pattern = os.path.join(folder_path, f"*{ext}")
        image_files.extend(glob.glob(pattern))
        pattern = os.path.join(folder_path, f"*{ext.upper()}")
        image_files.extend(glob.glob(pattern))
    return sorted(list(set(image_files)))

def safe_imread(path: str):
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
    img_for_model = cv2.resize(img_bgr, (INPUT_SHAPE[1], INPUT_SHAPE[0]), 
                               interpolation=cv2.INTER_LINEAR)
    img_rgb = cv2.cvtColor(img_for_model, cv2.COLOR_BGR2RGB).astype(np.float32)
    x = xception_preprocess(img_rgb.copy()) if USE_XCEPTION_PREPROCESS else (img_rgb / 255.0)
    x = np.expand_dims(x, axis=0)
    return img_for_model, x

# =========================
# Grad-CAM
# =========================
def make_gradcam(model, inputs, target_layer_name=TARGET_LAYER_NAME, class_index=None):
    target_layer = model.get_layer(target_layer_name)
    grad_model = tf.keras.models.Model([model.inputs], [target_layer.output, model.output])

    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(inputs)
        if class_index is None:
            class_index = int(tf.argmax(preds[0]).numpy())
        score = preds[:, class_index]

    grads = tape.gradient(score, conv_out)
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))

    conv_out = conv_out[0]
    heatmap = tf.reduce_sum(conv_out * pooled_grads, axis=-1)
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
# 텍스트
# =========================
def put_centered_text(img_bgr, text, y, font=cv2.FONT_HERSHEY_DUPLEX,
                      font_scale=0.8, color=(0,0,155), thickness=2):
    H, W = img_bgr.shape[:2]
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    x = (W - tw) // 2
    cv2.putText(img_bgr, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)

def put_text_top_left(img_bgr, text, y, font=cv2.FONT_HERSHEY_SIMPLEX,
                     font_scale=0.6, color=(255,255,255), thickness=1):
    cv2.putText(img_bgr, text, (10, y), font, font_scale, color, thickness, cv2.LINE_AA)

# =========================
# 표시
# =========================
def show_all_classes(original_bgr, cam_images, filename, preds, predicted_class):
    """
    원본 + 3개 클래스 CAM을 4개 나란히 표시
    cam_images: [cam_class0, cam_class1, cam_class2]
    """
    # 원본에 정보 추가
    img_with_info = original_bgr.copy()
    put_centered_text(img_with_info, filename, y=30, font_scale=0.7, 
                     color=(60,60,60), thickness=2)
    
    # 예측 정보
    pred_text = f"Predicted: {CLASS_NAMES[predicted_class]} ({preds[0][predicted_class]:.3f})"
    put_centered_text(img_with_info, pred_text, y=60, font_scale=0.6, 
                     color=(0,0,200), thickness=2)
    
    # 각 클래스 CAM에 라벨 추가
    labeled_cams = []
    for i, cam_img in enumerate(cam_images):
        cam_labeled = cam_img.copy()
        class_name = CLASS_NAMES[i]
        prob = preds[0][i]
        
        # 클래스명 + 확률
        title = f"{class_name}"
        prob_text = f"prob: {prob:.3f}"
        
        # 예측된 클래스면 강조
        if i == predicted_class:
            title_color = (0, 255, 0)  # 초록
            thickness = 3
        else:
            title_color = (255, 255, 255)
            thickness = 2
        
        put_centered_text(cam_labeled, title, y=30, font_scale=0.7, 
                         color=title_color, thickness=thickness)
        put_centered_text(cam_labeled, prob_text, y=60, font_scale=0.5, 
                         color=(255,255,255), thickness=1)
        labeled_cams.append(cam_labeled)
    
    # 4개 이미지 가로 배치
    combined = np.hstack([img_with_info] + labeled_cams)
    
    # 화면 크기에 맞게 조정
    scale = min(SCREEN_W / combined.shape[1], SCREEN_H / combined.shape[0], 1.0)
    if scale < 1.0:
        combined_show = cv2.resize(combined,
                                   (int(combined.shape[1]*scale), 
                                    int(combined.shape[0]*scale)))
    else:
        combined_show = combined
    
    cv2.imshow(WINDOW_NAME, combined_show)
    return combined

def wait_for_valid_key(valid_keys):
    """유효 키가 눌릴 때까지 대기"""
    while True:
        k = cv2.waitKey(0) & 0xFF
        if k in valid_keys:
            return k

# =========================
# MAIN
# =========================
def main():
    # 1) 모델 로드
    print(f"모델 로드: {WEIGHTS_PATH}")
    model = build_model()
    model.load_weights(WEIGHTS_PATH)
    print("모델 로드 완료\n")

    # 2) 이미지 수집
    print(f"이미지 폴더: {IMAGE_FOLDER}")
    image_files = get_image_files(IMAGE_FOLDER)
    
    if not image_files:
        print("폴더에 이미지가 없습니다.")
        return
    
    print(f"총 {len(image_files)}개 이미지\n")
    print("[키] Space/N=다음, P=이전, S=저장, ESC=종료\n")

    idx = 0
    while 0 <= idx < len(image_files):
        img_path = image_files[idx]
        filename = os.path.basename(img_path)
        
        print(f"[{idx+1}/{len(image_files)}] {filename}")
        
        # 이미지 로드
        img = safe_imread(img_path)
        if img is None:
            print("  ⚠ 로드 실패")
            idx += 1
            continue
        
        # 전처리
        img_display, x_input = prep_for_model(img)
        
        # 추론
        preds = model.predict(x_input, verbose=0)
        predicted_class = int(np.argmax(preds[0]))
        
        print(f"  예측: {CLASS_NAMES[predicted_class]} ({preds[0][predicted_class]:.3f})")
        print(f"  전체 확률: class0={preds[0][0]:.3f}, class1={preds[0][1]:.3f}, class2={preds[0][2]:.3f}")
        
        # 각 클래스별 Grad-CAM 생성
        cam_images = []
        for class_idx in range(NUM_CLASSES):
            heatmap, _, _ = make_gradcam(model, x_input, TARGET_LAYER_NAME, class_index=class_idx)
            overlay = overlay_heatmap_on_image(heatmap, img_display.copy(), alpha=0.4)
            cam_images.append(overlay)
        
        # 4개 이미지 표시
        combined = show_all_classes(img_display, cam_images, filename, preds, predicted_class)
        
        # 키 입력
        while True:
            k = wait_for_valid_key({ord('n'), ord('N'), 32, ord('p'), ord('P'), 
                                   ord('s'), ord('S'), 27})
            if k in (ord('n'), ord('N'), 32):   # 다음
                idx += 1
                break
            elif k in (ord('p'), ord('P')):     # 이전
                idx = max(0, idx - 1)
                break
            elif k in (ord('s'), ord('S')):     # 저장
                save_path = img_path.replace(os.path.splitext(img_path)[1], 
                                            f"_all_classes_cam.png")
                cv2.imwrite(save_path, combined)
                print(f"  💾 저장: {save_path}")
            elif k == 27:                        # ESC
                cv2.destroyAllWindows()
                print("\n종료")
                return
        
        cv2.destroyAllWindows()
    
    print("\n모든 이미지 처리 완료")

if __name__ == "__main__":
    main()