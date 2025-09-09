import os
import tensorflow as tf
import cv2
from tqdm import tqdm

import cv2
import tensorflow as tf
import os
import matplotlib.pyplot as plt


def create_dataset(dir_path, image_size, batch_size):
    dataset = load_data(dir_path, image_size[0:2], batch_size)
    return dataset

def preprocess_data(data):
    """
    預處理函數，將資料集中每個元素的遮罩圖像轉換為二值圖像。
    
    參數:
        data: dict, 包含原始圖像、遮罩圖像和修復後圖像的字典。
    
    返回:
        dict, 預處理過的字典，遮罩圖像已經被轉換為二值圖像。
    """
    data['original_images'] = data['original_images'] / 127.5 - 1.0
    data['masks'] = convert_mask(data['masks'])  # Apply convert_mask to the masks
    data['fixed_images'] = data['fixed_images'] / 127.5 - 1.0
    return data

def load_data(image_path, image_size = (512, 512), batch_size = 4, shuffle = True):
    origin_path = os.path.join(image_path, 'origin')
    mask_path = os.path.join(image_path, 'masks')
    fixed_path = os.path.join(image_path, 'fixed')

    origin = tf.keras.preprocessing.image_dataset_from_directory(
        origin_path,
        labels = None,
        label_mode = None,
        image_size = image_size,
        batch_size = None,
        shuffle = False,
    )
    mask = tf.keras.preprocessing.image_dataset_from_directory(
        mask_path,
        labels = None,
        label_mode = None,
        image_size = image_size,
        batch_size = None,
        shuffle = False,
    )
    fixed = tf.keras.preprocessing.image_dataset_from_directory(
        fixed_path,
        labels = None,
        label_mode = None,
        image_size = image_size,
        batch_size = None,
        shuffle = False,
    )
    dataset = tf.data.Dataset.zip((origin, mask, fixed))
    dataset = dataset.map(lambda orig, mask, fixed: {'original_images': orig, 'masks': mask, 'fixed_images': fixed})
    dataset = dataset.map(preprocess_data)
    dataset = dataset.prefetch(buffer_size = tf.data.experimental.AUTOTUNE)
    if shuffle:
        dataset = dataset.shuffle(buffer_size = 4000)
    dataset = dataset.batch(batch_size, drop_remainder = True)
    return dataset


def preprocess_data(data):
    """
    預處理函數，將資料集中每個元素的遮罩圖像轉換為二值圖像。
    
    參數:
        data: dict, 包含原始圖像、遮罩圖像和修復後圖像的字典。
    
    返回:
        dict, 預處理過的字典，遮罩圖像已經被轉換為二值圖像。
    """
    data['original_images'] = data['original_images'] / 127.5 - 1.0
    data['masks'] = convert_mask(data['masks'])  # Apply convert_mask to the masks
    data['fixed_images'] = data['fixed_images'] / 127.5 - 1.0
    return data

def convert_mask(mask):
    """
    將 RGB 的 mask 轉換為單通道的二值 mask，白色部分為 1，黑色部分為 0。

    參數:
        mask: Tensor, 大小為 (512, 512, 3) 的 RGB mask 圖像。

    返回:
        Tensor, 大小為 (512, 512, 1) 的二值 mask。
    """
    # 計算灰度值，這是由於 RGB 到灰度的轉換通常按這樣的權重：0.299 R + 0.587 G + 0.114 B
    gray_mask = tf.reduce_mean(mask, axis=-1, keepdims=True)

    # 假設像素值已被正規化在 [0, 1]，即白色為 1，黑色為 0
    binary_mask = tf.where(gray_mask > 0.5, 1.0, 0.0)

    return binary_mask
