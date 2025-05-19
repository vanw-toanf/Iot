import cv2
import os
import numpy as np

# Đường dẫn ảnh và nhãn
image_path = 'dataset/train/images/000004_jpg.rf.b88f487bbc8f0e0c17350ee3c04a45b5.jpg'
label_path = 'dataset/train/labels/000004_jpg.rf.b88f487bbc8f0e0c17350ee3c04a45b5.txt'

# Đọc ảnh
img = cv2.imread(image_path)
h, w = img.shape[:2]

# Đọc file label
with open(label_path, 'r') as f:
    for line in f:
        parts = list(map(float, line.strip().split()))
        class_id = int(parts[0])
        points = parts[1:]

        # Chuyển điểm chuẩn hóa -> pixel
        pts = []
        for i in range(0, len(points), 2):
            x = int(points[i] * w)
            y = int(points[i + 1] * h)
            pts.append((x, y))

        # Vẽ polygon
        pts_np = cv2.convexHull(np.array(pts)).reshape((-1, 1, 2))
        cv2.polylines(img, [pts_np], isClosed=True, color=(0, 255, 0), thickness=2)
        cv2.putText(img, str(class_id), pts[0], cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 1)

# Hiển thị ảnh
cv2.imshow("Polygon Bounding Box", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
