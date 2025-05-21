import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf
import sys
from model import Tiny_model, F1Score

# Load custom metric nếu có dùng khi save model
# from tensorflow_addons.metrics import F1Score


def test():
    IMAGE_PATH = "1.jpeg"
    OUTPUT_PATH = 'output.jpg'
    INPUT_SHAPE = (96, 96)
    GRID_SIZE = 3
    MODEL_PATH = "best.h5"
    THRESHOLD = 0.5

    model = load_model(MODEL_PATH, custom_objects={
        'F1Score': F1Score,
    })

    # Load ảnh và tiền xử lý
    img = load_img(IMAGE_PATH, target_size=INPUT_SHAPE)
    img_array = img_to_array(img) / 255.0  # Chuẩn hóa về [0,1]
    img_array = np.expand_dims(img_array, axis=0)  # thêm batch dimension

    # Dự đoán
    pred = model.predict(img_array)[0]  # Lấy kết quả batch đầu tiên

    # Xử lý output
    pred_binary = (pred > 0.5).astype(int)
    num_boxes = np.sum(pred_binary)

    # In kết quả
    print(f'Dự đoán có {num_boxes} thùng hàng trong ảnh.')
    print(f'Chi tiết từng ô (grid {GRID_SIZE}x{GRID_SIZE}):')
    print(pred_binary.reshape((GRID_SIZE, GRID_SIZE)))

    # Vẽ ảnh
    original_img = cv2.imread(IMAGE_PATH)
    original_img = cv2.resize(original_img, INPUT_SHAPE)

    h, w = INPUT_SHAPE
    cell_w, cell_h = w // GRID_SIZE, h // GRID_SIZE

    # Vẽ lưới
    for i in range(1, GRID_SIZE):
        # kẻ đường ngang
        cv2.line(original_img, (0, i * cell_h), (w, i * cell_h), (200, 200, 200), 1)
        # kẻ đường dọc
        cv2.line(original_img, (i * cell_w, 0), (i * cell_w, h), (200, 200, 200), 1)

    # Vẽ bbox đỏ vào ô có hàng
    for idx, val in enumerate(pred_binary):
        if val == 1:
            row = idx // GRID_SIZE
            col = idx % GRID_SIZE
            x1 = col * cell_w
            y1 = row * cell_h
            x2 = x1 + cell_w
            y2 = y1 + cell_h
            cv2.rectangle(original_img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # --- Lưu kết quả ---
    cv2.imwrite(OUTPUT_PATH, original_img)


    # model = Tiny_model()
    # model.model.load_weights("best.h5")

if __name__ == "__main__":
    test()