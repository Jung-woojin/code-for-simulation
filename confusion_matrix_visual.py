import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

model_name = "Qwen2_5_VL_72B"
save_svg = f"{model_name}_cm.svg"

# rows=Pred, cols=GT
cm = np.array([
    [363,   0,    2,   35,    0],
    [  3,  93,    0,    1,    4],
    [  0,   0,   100,   0,    0],
    [  0,   0,    0,  100,    0],
    [  0,   1,    0,    0,   99],
], dtype=float)

labels = ["Normal", "Collision", "Fire", "Seafog", "Sinking"]
n = len(labels)

# GT(열) 기준 정규화 (색상용)
col_sums = cm.sum(axis=0, keepdims=True)
cm_norm = cm / np.clip(col_sums, 1e-12, None)

# -------------------------
# ✅ Figure / GridSpec
#   width_ratios로 컬러바 폭만 아주 얇게
# -------------------------
fig = plt.figure(figsize=(9.2, 8.2))  # 정사각 + 컬러바 고려한 적당한 크기
fig.subplots_adjust(bottom=0.18, top=0.92)  # bottom만 올려도 OK

gs = GridSpec(1, 2, width_ratios=[1, 0.04], wspace=0.02)

ax = fig.add_subplot(gs[0, 0])
ax_cbar = fig.add_subplot(gs[0, 1])

# -------------------------
# ✅ Confusion matrix (정사각형 강제)
# -------------------------
im = ax.imshow(
    cm_norm,
    interpolation="nearest",
    cmap=plt.cm.Blues,
    vmin=0, vmax=1,
    aspect="equal"      # 데이터 aspect도 1:1
)

# ✅ 박스 자체가 정사각형 되게 강제 (이게 핵심)
ax.set_box_aspect(1)

# 축/라벨
ax.set_xticks(np.arange(n))
ax.set_yticks(np.arange(n))
ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=12)
ax.set_yticklabels(labels, fontsize=12)

ax.set_xlabel("Predicted", fontsize=13)
ax.set_ylabel("Ground Truth", fontsize=13)
ax.set_title(f"{model_name} | Normalized Confusion Matrix", fontsize=14)

ax.tick_params(axis='both', which='major', labelsize=12, pad=2)

# 셀 텍스트: 정규화값 + (카운트)
thresh = 0.5
for i in range(n):
    for j in range(n):
        txt = f"{cm_norm[i, j]:.2f}\n({int(cm[i, j])})"
        ax.text(
            j, i, txt,
            ha="center", va="center",
            color="white" if cm_norm[i, j] > thresh else "black",
            fontsize=11,
            linespacing=1.25
        )

# -------------------------
# ✅ Colorbar (별도 축에 할당)
# -------------------------
cbar = plt.colorbar(im, cax=ax_cbar)
cbar.ax.tick_params(labelsize=11)

# -------------------------
# ✅ 저장: 박스 바깥 영역 타이트 크롭
# -------------------------
plt.savefig(save_svg, format="svg", bbox_inches="tight", pad_inches=0.03)
plt.show()

print(f"Saved: {save_svg}")
