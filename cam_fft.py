import os
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt

# 이미지 폴더
IMAGE_FOLDER = r"C:\Users\ust21\label\final_test\Busan\auto_2\2"

# 처리할 확장자
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def fft_image(img: np.ndarray):
    """
    입력 이미지를 grayscale로 받아 2D FFT 수행
    return:
        fshift      : shift된 복소 FFT 결과
        magnitude   : 로그 스케일 magnitude spectrum
    """
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude = 20 * np.log(np.abs(fshift) + 1)  # log scale
    return fshift, magnitude

def main():
    image_dir = Path(IMAGE_FOLDER)

    if not image_dir.exists():
        print(f"폴더가 존재하지 않습니다: {image_dir}")
        return

    image_files = sorted(
        [p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS]
    )

    if not image_files:
        print("이미지 파일이 없습니다.")
        return

    for img_path in image_files:
        print(f"처리 중: {img_path.name}")

        # grayscale로 읽기
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

        if img is None:
            print(f"이미지 로드 실패: {img_path.name}")
            continue

        _, magnitude = fft_image(img)

        # 결과 표시
        plt.figure(figsize=(10, 4))

        plt.subplot(1, 2, 1)
        plt.imshow(img, cmap="gray")
        plt.title(f"Original\n{img_path.name}")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(magnitude, cmap="gray")
        plt.title("FFT Magnitude Spectrum")
        plt.axis("off")

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()