import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from model import Tiny_model, F1Score
from data_preprocessing import labels_to_grid


# Define constants value
INPUT_SHAPE = (96, 96, 3)
NUM_CELLS = 3
NUM_CLASSES = 1
BATCH_SIZE = 32
EPOCHS = 20

def load_data(img_dir, label_dir, input_shape):
    X = []
    y = []

    # Tạo dict: {filename.txt: label_vector}
    label_dict = labels_to_grid(label_dir)

    for file_name in os.listdir(label_dir):
        if not file_name.endswith('.txt'):
            continue

        img_name = file_name.replace('.txt', '.jpg')
        img_path = os.path.join(img_dir, img_name)

        if not os.path.exists(img_path):
            print(f"[WARNING] Missing image file: {img_path}")
            continue
        
        img = load_img(img_path, target_size=input_shape)
        img_array = img_to_array(img) / 255.0

        X.append(img_array)
        y.append(label_dict[file_name])

    return np.array(X), np.array(y)


train_dir = "dataset/train/images"
train_label_dir = "dataset/train/labels"
val_dir = "dataset/valid/images"
val_label_dir = "dataset/valid/labels"

X_train, y_train = load_data(train_dir, train_label_dir, INPUT_SHAPE)
X_val, y_val = load_data(val_dir, val_label_dir, INPUT_SHAPE)

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

checkpoint = ModelCheckpoint("best.h5", monitor="val_loss", save_best_only=True, verbose = 1)

history = Tiny_model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    batch_size=BATCH_SIZE,
    epochs = EPOCHS,
    callbacks=[checkpoint]
)

# #Save the model
# Tiny_model.save("model.h5")

print("Model saved as model.h5")
