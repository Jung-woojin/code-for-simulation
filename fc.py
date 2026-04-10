# fc_vs_cnn_full_demo_tf.py
# pip install tensorflow pillow matplotlib

import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


# =========================
# 1. 네가 바꿀 부분
# =========================
IMAGE_PATH = r"C:\Users\ust21\Desktop\dog.png"   # <- 여기만 바꿔
IMG_SIZE = 32                   # flatten 시각화 때문에 32 추천


# =========================
# 2. 이미지 로드
# =========================
img = Image.open(IMAGE_PATH).convert("RGB")
img = img.resize((IMG_SIZE, IMG_SIZE))

img_arr = np.array(img) / 255.0         # [H, W, C]
x = np.expand_dims(img_arr, axis=0)     # [1, H, W, C]

print("원본 shape:", x.shape)


# =========================
# 3. FC용 flatten
# =========================
flat = img_arr.reshape(-1)

print("Flatten shape:", flat.shape)


# =========================
# 4. FC 모델
# =========================
fc_model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(10)
])


# =========================
# 5. CNN 모델
# =========================
inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

conv1 = tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu')(inputs)
pool1 = tf.keras.layers.MaxPooling2D()(conv1)

conv2 = tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu')(pool1)
pool2 = tf.keras.layers.MaxPooling2D()(conv2)

flat_cnn = tf.keras.layers.Flatten()(pool2)
outputs = tf.keras.layers.Dense(10)(flat_cnn)

cnn_model = tf.keras.Model(inputs, outputs)

# feature map 확인용
feature_model = tf.keras.Model(inputs, [conv1, conv2])


# =========================
# 6. 파라미터 수 비교
# =========================
print("\n[파라미터 수]")
print("FC  :", fc_model.count_params())
print("CNN :", cnn_model.count_params())


# =========================
# 7. forward
# =========================
fc_out = fc_model(x)
cnn_out = cnn_model(x)
fmap1, fmap2 = feature_model(x)

print("\n[출력 shape]")
print("FC :", fc_out.shape)
print("CNN:", cnn_out.shape)

print("\n[중간 표현]")
print("Flatten:", flat.shape)
print("Conv1:", fmap1.shape)
print("Conv2:", fmap2.shape)


# =========================
# 8. 시각화
# =========================

# (1) 원본 이미지
plt.figure(figsize=(4, 4))
plt.imshow(img_arr)
plt.title("Original Image")
plt.axis("off")
plt.show()


# (2) FC가 보는 방식 (엿가락)
plt.figure(figsize=(15, 3))
plt.plot(flat)
plt.title("FC view: flattened 1D vector")
plt.xlabel("Index")
plt.ylabel("Pixel value")
plt.show()

plt.figure(figsize=(15, 1.5))
plt.imshow(flat[np.newaxis, :], aspect="auto", cmap="gray")
plt.title("Flattened image (long strip)")
plt.yticks([])
plt.xlabel("Index")
plt.show()


# (3) CNN feature map
def show_feature_maps(feature_map, title, max_channels=6):
    fmap = feature_map[0]
    num_channels = min(fmap.shape[-1], max_channels)

    plt.figure(figsize=(12, 3))
    for i in range(num_channels):
        plt.subplot(1, num_channels, i+1)
        plt.imshow(fmap[:, :, i], cmap='gray')
        plt.axis("off")
        plt.title(f"{title}\nch {i}")
    plt.tight_layout()
    plt.show()


show_feature_maps(fmap1.numpy(), "Conv1")
show_feature_maps(fmap2.numpy(), "Conv2")


# =========================
# 9. 핵심 비교 출력
# =========================
print("\n" + "="*60)
print("🔥 핵심 차이")
print("="*60)
print("FC  : 이미지를 1차원 숫자열로 본다 (구조 없음)")
print("CNN : 이미지를 2D 구조 그대로 본다 (패턴 인식)")
print()
print("FC  : 파라미터 많음")
print("CNN : 필터 공유로 효율적")
print("="*60)