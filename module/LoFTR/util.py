import cv2
import numpy as np
from sklearn.cluster import DBSCAN
import os
import shutil
import random


def addWhiteBlock(x1,y1,x2,y2,orginalFile,fileName,pth):

    image = cv2.imread(orginalFile)
    image = cv2.resize(image, (640, 480))


    rectangle = cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), -1)
    fileName = pth + fileName
    if ( not os.path.exists(pth)):
        os.makedirs(pth)
    cv2.imwrite(fileName, rectangle)

def generate_random_rectangle(image_width, image_height):
    # Ensure the rectangle is less than 25% of the image area
    max_area = 0.25 * image_width * image_height

    while True:
        # Generate two random points ensuring x1 < x2 and y1 < y2
        x1, x2 = sorted(random.sample(range(image_width), 2))
        y1, y2 = sorted(random.sample(range(image_height), 2))

        # Calculate the width and height of the rectangle
        rect_width = x2 - x1
        rect_height = y2 - y1

        # Calculate the area of the rectangle
        area = rect_width * rect_height

        if area <= max_area:
            return (x1, y1, x2, y2)
def makeBlackImg(x1,y1,x2,y2, fileName):
    width = 512
    height = 512

    black_image = np.zeros((height, width, 3), dtype='uint8')

    black_image[:] = 0

    cv_image = black_image

    sx = 0.8
    sy = 1.0667

    # 計算新座標
    x1 = int(x1 * sx)
    y1 = int(y1 * sy)
    x2 = int(x2 * sx)
    y2 = int(y2 * sy)

    print(x1,y1,x2,y2)

    rectangle = cv2.rectangle(cv_image, (x1, y1), (x2, y2), (255, 255, 255), -1)

    #print("----", fileName)
    cv2.imwrite(fileName, rectangle)


def block_region(mkpts0):
  # 应用 DBSCAN 算法
  dbscan = DBSCAN(eps=30, min_samples=10)  # 这里的参数可以根据你的实际数据调整
  dbscan.fit(mkpts0)

  # 获取聚类结果
  cluster_labels = dbscan.labels_

  # 查看聚类标签
  print("Cluster labels:", cluster_labels)

  # 找出最大的簇
  labels, counts = np.unique(cluster_labels[cluster_labels >= 0], return_counts=True)
  if ( labels.any() == False):
    return None
  largest_cluster_label = labels[np.argmax(counts)]
  largest_cluster_points = mkpts0[cluster_labels == largest_cluster_label]
  # indices = np.where(cluster_labels == largest_cluster_label)[0]
  # 计算矩形边界
  x_min, y_min = np.min(largest_cluster_points, axis=0)
  x_max, y_max = np.max(largest_cluster_points, axis=0)

  # 确保坐标是整数
  x_min, y_min, x_max, y_max = map(int, [x_min, y_min, x_max, y_max])
  return (x_min, y_min, x_max, y_max)   
    
def returnBaseName(path):
    if path[-1] == "/":
        path = path.rstrip("/")
    pth = os.path.basename(path)
    return pth
def copy_file(source_path, destination_path, newName):

    try:
        if not os.path.exists(destination_path):
            os.mkdir(destination_path)
        # 复制文件
        shutil.copy(source_path, destination_path+newName)
        
    except Exception as e:
        print("copy error", e)
