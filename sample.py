import cv2
import numpy as np

def draw_ground_truth_polygons(img, label_file, color=(0, 0, 255), thickness=2):
    h, w = img.shape[:2]

    with open(label_file, 'r') as f:
        for line in f:
            parts = list(map(float, line.strip().split()))
            if len(parts) != 9:
                continue  # skip nếu không đủ 8 tọa độ

            # Bỏ class_id, lấy 8 giá trị tiếp theo
            coords = parts[1:]  # x1 y1 x2 y2 x3 y3 x4 y4 (normalized)
            points = np.array([
                [int(coords[0] * w), int(coords[1] * h)],
                [int(coords[2] * w), int(coords[3] * h)],
                [int(coords[4] * w), int(coords[5] * h)],
                [int(coords[6] * w), int(coords[7] * h)]
            ], dtype=np.int32).reshape((-1, 1, 2))

            cv2.polylines(img, [points], isClosed=True, color=color, thickness=thickness)

    return img

img_path = "dataset/train/images/15-04-2022__10-02-35AM_jpg.rf.201a97f02e33c31b348ea628a958db5e.jpg"
label_path = "dataset/train/labels/15-04-2022__10-02-35AM_jpg.rf.201a97f02e33c31b348ea628a958db5e.txt"

img = cv2.imread(img_path)
img_with_labels = draw_ground_truth_polygons(img, label_path)

cv2.imshow("Ground Truth", img_with_labels)
cv2.waitKey(0)
cv2.destroyAllWindows()