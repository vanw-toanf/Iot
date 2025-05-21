import numpy as np
import cv2 
import os

def polygon_to_center(points):
    # change from polygon to center
    x_coords = points[0::2]
    y_coords = points[1::2]
    cx = np.mean(x_coords)
    cy = np.mean(y_coords)
    return cx, cy

# # Get annotation 
# def labels_to_grid(label_folder, grid_size=3):
#     label_dict = {}
#     for file_name in os.listdir(label_folder):
#         if not file_name.endswith('.txt'):
#             continue
    
#         path = os.path.join(label_folder, file_name)
#         image_name = file_name.replace('.txt', '.jpg')
#         label_vector = np.zeros( grid_size*grid_size , dtype=np.float32)
        
#         with open(path, 'r') as f:
#             for line in f:
#                 parts = list(map(float, line.strip().split()))
#                 if len(parts) != 9:
#                     continue
                
#                 # chuyen toa do ve pixel
#                 cx, cy = polygon_to_center(parts)
                
#                 # Tính xem trung tâm nằm trong ô nào
#                 col = int(cx * grid_size)
#                 row = int(cy * grid_size)
#                 col = min(col, grid_size - 1)  # tránh ra ngoài
#                 row = min(row, grid_size - 1)
#                 cell_id = row * grid_size + col
#                 label_vector[cell_id] = 1.0  # co vat
                
#         label_dict[file_name] = label_vector
#     return label_dict

def labels_to_polygons(label_folder):
    label_dict = {}
    for file_name in os.listdir(label_folder):
        if not file_name.endswith('.txt'):
            continue

        path = os.path.join(label_folder, file_name)
        image_name = file_name.replace('.txt', '.jpg')
        polygons = []

        with open(path, 'r') as f:
            for line in f:
                parts = list(map(float, line.strip().split()))
                if len(parts) != 9:
                    continue  # Bỏ qua dòng lỗi

                class_id = int(parts[0])
                coords = parts[1:]  # [x1, y1, x2, y2, ..., x4, y4]
                polygons.append(coords)

        if polygons:
            label_dict[image_name] = np.array(polygons, dtype=np.float32)
    
    return label_dict

# label_dict = labels_to_polygons("dataset/train/labels/")
# print(label_dict["15-04-2022__10-06-50AM_jpg.rf.beb03f2ed6d508c77ad22bbac488bb25.jpg"])

