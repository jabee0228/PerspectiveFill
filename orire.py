from resize import resize_images
import random
import os
import cv2
import numpy as np

def jpg_files(directory):
    jpg_filesre = [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith('.jpg')]

    return jpg_filesre

def makeBlackImg(x1,y1,x2,y2, fileName ,pth):
    width = 512
    height = 512
    fileName = fileName[18:]
    fileName = pth + fileName

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


def generate_random_blocks(image_width, image_height):

    block_width = random.randint(250, 500)
    block_height = random.randint(250, 500)

    x1 = random.randint(10, image_width-block_width-1)
    y1 = random.randint(10, image_height-block_height-1)

    x2 = x1 + block_width
    y2 = y1 + block_height

    print(x1 ,y1, x2, y2)
    assert x1 >= 0 and y1 >= 0 and x2 < image_width and y2 < image_height
    return [x1, y1, x2, y2]


resize_images("testData","./images/reshaped")


re = jpg_files("./images/reshaped")

for f in re:
    blocksXY = generate_random_blocks(512, 512)
    makeBlackImg(blocksXY[0],blocksXY[1],blocksXY[2], blocksXY[3],  f,"./images/masks/" )