import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from model import Tiny_model
# from model import Tiny_model, F1Score
from data_preprocessing import labels_to_polygons
import cv2


# Define constants value
INPUT_SHAPE = (96, 96, 3)
NUM_CELLS = 3
NUM_CLASSES = 1
BATCH_SIZE = 32
train_dir = "dataset/train/images"
train_label_dir = "dataset/train/labels"
val_dir = "dataset/valid/images"
val_label_dir = "dataset/valid/labels"
MAX_OBJECTS = 12
# def load_data(img_dir, label_dir, input_shape):
#     X = []
#     y = []

#     # Tạo dict: {filename.txt: label_vector}
#     label_dict = labels_to_grid(label_dir)

#     for file_name in os.listdir(label_dir):
#         if not file_name.endswith('.txt'):
#             continue

#         img_name = file_name.replace('.txt', '.jpg')
#         img_path = os.path.join(img_dir, img_name)

#         if not os.path.exists(img_path):
#             print(f"[WARNING] Missing image file: {img_path}")
#             continue
        
#         img = load_img(img_path, target_size=input_shape)
#         img_array = img_to_array(img) / 255.0

#         X.append(img_array)
#         y.append(label_dict[file_name])
 
#     return np.array(X), np.array(y)

def load_and_resize_image(img_name, folder, size=96):
    path = os.path.join(folder, img_name)
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Không đọc được ảnh: {path}")
    img = cv2.resize(img, (size, size))
    img = img.astype(np.float32) / 255.0  # normalize về [0, 1]
    return img

# def prepare_data(label_dict, image_folder, max_objects=3):
#     X, y = [], []
#     for img_name, polygons in label_dict.items():
#         try:
#             img = load_and_resize_image(img_name, image_folder)
#         except Exception as e:
#             print(f"Bỏ qua ảnh lỗi: {img_name}")
#             continue

#         padded = np.zeros((max_objects, 8), dtype=np.float32)
#         count = min(len(polygons), max_objects)
#         padded[:count] = polygons[:count]
#         X.append(img)
#         y.append(padded.flatten())

#     return np.array(X), np.array(y)

def prepare_data(label_dict, image_folder, max_objects=MAX_OBJECTS):
    X, y = [], []
    for img_name, polygons in label_dict.items():
        try:
            img = load_and_resize_image(img_name, image_folder)
        except Exception as e:
            print(f"Bỏ qua ảnh lỗi: {img_name}")
            continue

        if polygons.shape[1] != 8:
            print(f"{img_name}: không đúng polygon 8 điểm. Shape: {polygons.shape}")
            continue

        padded = np.zeros((max_objects, 8), dtype=np.float32)
        count = min(len(polygons), max_objects)
        padded[:count] = polygons[:count]

        X.append(img)
        y.append(padded.flatten())

    X_arr = np.array(X)
    y_arr = np.array(y)

    print("✅ Tập dữ liệu đã chuẩn bị:")
    print(" - X shape:", X_arr.shape)
    print(" - y shape:", y_arr.shape)
    print(" - Ví dụ y[0]:", y_arr[0])

    return X_arr, y_arr

def draw_boxes_from_X_y(X, y, max_objects=MAX_OBJECTS, output_dir='output/gt/', img_names=None):
    os.makedirs(output_dir, exist_ok=True)

    for idx, (img_array, y_vector) in enumerate(zip(X, y)):
        # Resize ảnh từ 96x96 về 480x480 để dễ nhìn (nếu cần)
        img_vis = (img_array * 255).astype(np.uint8)
        img_vis = cv2.resize(img_vis, (480, 480))

        h, w = img_vis.shape[:2]
        polygons = y_vector.reshape(max_objects, 8)

        for poly in polygons:
            # Nếu box toàn 0 → bỏ qua (padding)
            if np.max(poly) < 1e-3:
                continue

            pts = np.array([
                [int(poly[0] * w), int(poly[1] * h)],
                [int(poly[2] * w), int(poly[3] * h)],
                [int(poly[4] * w), int(poly[5] * h)],
                [int(poly[6] * w), int(poly[7] * h)]
            ], dtype=np.int32).reshape((-1, 1, 2))

            cv2.polylines(img_vis, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

        name = img_names[idx] if img_names else f"sample_{idx}.jpg"
        cv2.imwrite(os.path.join(output_dir, name), img_vis)

if __name__ == "__main__":
    # Load label dictionaries
    train_label_dict = labels_to_polygons(train_label_dir)
    val_label_dict = labels_to_polygons(val_label_dir)

    X_train, y_train = prepare_data(train_label_dict, train_dir, max_objects=MAX_OBJECTS)
    X_val, y_val = prepare_data(val_label_dict, val_dir, max_objects=MAX_OBJECTS)

    # draw_boxes_from_X_y(X_train, y_train, max_objects=MAX_OBJECTS, output_dir='output/gt_train/', img_names=list(train_label_dict.keys()))

    # Khởi tạo model
    model = Tiny_model(max_objects=MAX_OBJECTS)

    # Huấn luyện
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=32,
        epochs=30
    )

    # Lưu model
    model.save_model("tinymodel.h5")
