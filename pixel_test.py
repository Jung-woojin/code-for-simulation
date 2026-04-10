import cv2

# 이미지 경로
IMAGE_PATH = r"C:\Users\ust21\Desktop\seafogbusan_cctv_20240122111000.jpg"

img = cv2.imread(IMAGE_PATH)
if img is None:
    raise FileNotFoundError(f"이미지를 불러올 수 없습니다: {IMAGE_PATH}")

orig = img.copy()
display = img.copy()

drawing = False
start_point = None
end_point = None


def draw_box_preview(base_img, pt1, pt2):
    x1, y1 = pt1
    x2, y2 = pt2

    left = min(x1, x2)
    top = min(y1, y2)
    right = max(x1, x2)
    bottom = max(y1, y2)

    w = right - left
    h = bottom - top

    canvas = base_img.copy()

    # 박스
    cv2.rectangle(canvas, (left, top), (right, bottom), (0, 255, 0), 2)

    # 시작점, 끝점 표시
    cv2.circle(canvas, (x1, y1), 4, (0, 0, 255), -1)
    cv2.circle(canvas, (x2, y2), 4, (255, 0, 0), -1)

    padding = 10

    tx = right + padding
    ty = top + 20

    text1 = f"Start: ({x1}, {y1})"
    text2 = f"End: ({x2}, {y2})"
    text3 = f"Size: {w} x {h} px"

    cv2.putText(canvas, text1, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(canvas, text2, (tx, ty + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    cv2.putText(canvas, text3, (tx, ty + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return canvas


def mouse_callback(event, x, y, flags, param):
    global drawing, start_point, end_point, display

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_point = (x, y)
        end_point = (x, y)

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            end_point = (x, y)
            display = draw_box_preview(orig, start_point, end_point)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        end_point = (x, y)
        display = draw_box_preview(orig, start_point, end_point)

        x1, y1 = start_point
        x2, y2 = end_point
        left = min(x1, x2)
        top = min(y1, y2)
        right = max(x1, x2)
        bottom = max(y1, y2)
        w = right - left
        h = bottom - top

        print(f"Box: ({left}, {top}) -> ({right}, {bottom}), size = {w} x {h} px")


cv2.namedWindow("Box Drawer")
cv2.setMouseCallback("Box Drawer", mouse_callback)

print("사용법:")
print("- 마우스 왼쪽 버튼 누르고 드래그: 박스 그리기")
print("- r: 초기화")
print("- q 또는 ESC: 종료")

while True:
    cv2.imshow("Box Drawer", display)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("r"):
        display = orig.copy()
        start_point = None
        end_point = None

    elif key == ord("q") or key == 27:
        break

cv2.destroyAllWindows()