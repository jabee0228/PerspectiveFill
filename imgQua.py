import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np


# 加载图像并转换为 Tensor
def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_image(img, channels=3)  # 假设是RGB图像
    img = tf.image.convert_image_dtype(img, dtype=tf.float32)  # 转换为浮点数格式
    return img

# 自定义区域裁剪函数
def crop_image(img, x1, y1, x2, y2):
    return img[y1:y2, x1:x2]

# 讀取圖片並預處理
def load_and_preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(512, 512))  # 調整到適合的尺寸
    img = image.img_to_array(img)  # 將圖片轉換為數組
    img = img.astype('float32')  # 轉換數據類型
    img /= 255.0  # 正規化到 [0, 1]
    return img





def process(filename, image1, image2, x1, y1, x2, y2 ):
    '''
    :filename: str, the name of file which is going to be stored
    :image1: image1
    :image2: image2
    '''

    sx = 0.8
    sy = 1.0667

    # 計算新座標
    x1 = int(x1 * sx)
    y1 = int(y1 * sy)
    x2 = int(x2 * sx)
    y2 = int(y2 * sy)

    # image1 = load_and_preprocess_image(image_path1)
    # image2 = load_and_preprocess_image(image_path2)
    image1 = (image1+1.0)/2.0 # from -1.0~1.0 to 0~1.0
    image2 = (image2+1.0)/2.0 # from -1.0~1.0 to 0~1.0

    image1_cropped = crop_image(image1, x1, y1, x2, y2)
    image2_cropped = crop_image(image2, x1, y1, x2, y2)
    print(image1_cropped.shape)
    print(image2_cropped.shape)

    psnr_value = tf.image.psnr(image1_cropped, image2_cropped, max_val=1.0)
    print(f"PSNR: {psnr_value.numpy()} dB")

    # 计算 SSIM
    ssim_value = tf.image.ssim(image1_cropped, image2_cropped, max_val=1.0)
    print(f"SSIM: {ssim_value.numpy()}")

    return float(psnr_value), float(ssim_value)


