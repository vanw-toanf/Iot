import numpy as np
import cv2
from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf
from model import Tiny_model
import os

# Load custom metric nếu có dùng khi save model
# from tensorflow_addons.metrics import F1Score


MODEL_PATH = "tinymodel.h5"
IMAGE_DIR = "dataset/test/images"  
OUTPUT_DIR = "output"  
IMG_SIZE = 96
MAX_OBJECTS = 12

def load_image(path, size=IMG_SIZE):
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Không load được ảnh: {path}")
    img_resized = cv2.resize(img, (size, size))
    img_norm = img_resized.astype(np.float32) / 255.0
    return img, img_norm  # trả cả ảnh gốc và ảnh chuẩn hóa

# ----------- Vẽ polygon ----------
def draw_polygons(img, coords_list, color=(0, 255, 0)):
    h, w = img.shape[:2]
    for coords in coords_list:
        pts = np.array([
            [int(coords[0] * w), int(coords[1] * h)],
            [int(coords[2] * w), int(coords[3] * h)],
            [int(coords[4] * w), int(coords[5] * h)],
            [int(coords[6] * w), int(coords[7] * h)]
        ], dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(img, [pts], isClosed=True, color=color, thickness=2)

os.makedirs(OUTPUT_DIR, exist_ok=True)

model = load_model(MODEL_PATH)

for img_name in os.listdir(IMAGE_DIR):
    if not img_name.lower().endswith(('.jpg', '.png')):
        continue

    img_path = os.path.join(IMAGE_DIR, img_name)
    try:
        img_original, img_input = load_image(img_path)
    except Exception as e:
        print(f"Bỏ qua ảnh lỗi: {img_path}")
        continue
    
    img_input = np.expand_dims(img_input, axis=0)
    pred = model.predict(img_input)[0]  
    polygons = pred.reshape(MAX_OBJECTS, 8)

    # print(f"Ảnh: {img_name}")
    # for i, poly in enumerate(polygons):
    #     print(f"  Polygon {i+1}: {np.round(poly, 4).tolist()}")


    # Vẽ lên ảnh gốc
    draw_polygons(img_original, polygons)

    # Lưu kết quả
    output_path = os.path.join(OUTPUT_DIR, img_name)
    cv2.imwrite(output_path, img_original)
    print(f"Đã lưu kết quả: {output_path}")