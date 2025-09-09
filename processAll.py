
import os
import csv
import json

#from backend.resize import resize_image
from resize import resize_images
from resize import resize_imagea
from module.LoFTR.loftr import loftrGenerate, loftrGenerate_area
from module.LoFTR.findJpg import returnPair, checkDirectory, getDirs
from module.MyVersion.load_data import load_data
from module.MyVersion.util import InferModel
from module.MyVersion.train import load_yml
import imgQua
import random
import tensorflow as tf

# important! -> need check reuse.

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

def generate_random_blocks2(image_width, image_height):
    block_width = random.randint(100, 250)
    block_height = random.randint(100, 250)

    x1 = random.randint(30, image_width-block_width-30)
    y1 = random.randint(30, image_height-block_height-30)

    x2 = x1 + block_width
    y2 = y1 + block_height

    print(x1 ,y1, x2, y2)
    assert x1 >= 0 and y1 >= 0 and x2 < image_width and y2 < image_height
    return [x1, y1, x2, y2]


input_folder = "testData"
resize_images("testData","./images051301/reshaped")
pairs = returnPair("./images051301/reshaped")
config = load_yml('./module/MyVersion/config1.yml')

PSNR = []
SSIM = []
reuse = False
maskDir = "./images051301/masks/" # input mask of the model
loftrDir = "./images051301/fixed/" # input fixed image of the model
outDir = "./images051301/output/" # output of the model
if os.path.exists(outDir):
    os.makedirs(outDir, exist_ok=True)

batch_size = 4
image_num = 9999999
config.BATCH_SIZE = batch_size
blockList = []
nameList = []






with open('rectangles_20250608_181707.json', 'r') as file:
    data = json.load(file)

# 取得所有照片名稱
testData = list(data.keys())
directory = "./testData2/"
file_list = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]


i = 0

# 遍歷每一張照片
for a in testData:
    blocksXY = [int(data[a]["x1"]), int(data[a]["y1"]), int(data[a]["x2"]), int(data[a]["y2"])]
    a = "./testData2/" + a
    print(blocksXY)
    if blocksXY == [0, 0, 0, 0]:
        continue
    # 遍歷其他照片並排除自己
    for b in file_list:
        b = "./testData2/" + b

        if a != b:
            try:
                newName = loftrGenerate_area(a, b, blocksXY)
                newName = "./images060802/fixed/" + newName
            except ValueError as e:
                continue
            resize_imagea(newName)
        i = i + 1
        if i == image_num:
            exit()






if not reuse:
    i = 0
    for pair in pairs:

        # get masks
        blocksXY = generate_random_blocks2(640, 480)

        blockList.append(blocksXY)
        # get fixed
        newName = loftrGenerate(pair[0], pair[1], blocksXY)
        nameList.append(newName)
        print("New name: ", newName)
        i = i + 1
        if i == image_num:
            break
else:
    # get namelist
    i = 0
    with open('./output/record.csv', newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            nameList.append(row['filename'])
            blockList.append([int(row['x1']),int(row['y1']), int(row['x2']), int(row['y2'])])
            i = i + 1
            if i == image_num:
                break

# input collection is in the directory
# do inference
infer_model = InferModel(config)
dataset = load_data("./images", (512, 512), batch_size=batch_size ,shuffle=False)
nameList.sort()

i = 0


# setup csv table
table = [['filename', 'PSNR', 'SSIM', 'x1', 'y1', 'x2', 'y2']]

for data in dataset:
    predictions, origins = infer_model.inference_batch(data, batch_size=batch_size)
    # calculate the quality of the image
    predictions = tf.split(predictions, batch_size, axis=0)
    origins = tf.split(origins, batch_size, axis=0)
    for predict, origin in zip(predictions, origins):
        # store the image


        fName = nameList[i]
        predict = predict[0] # remove batch axis
        origin = origin[0]
        #img = tf.reverse(predict, axis=[-1])
        print("Writing to file: ", outDir+fName)
        tf.io.write_file(outDir + fName, tf.image.encode_jpeg(tf.cast((predict+1.0)*127.5, tf.uint8)))
        psnr, ssim = imgQua.process(fName, predict, origin, blockList[i][0], blockList[i][1], blockList[i][2], blockList[i][3])
        table.append([fName, psnr, ssim, blockList[i][0], blockList[i][1], blockList[i][2], blockList[i][3]])
        PSNR.append(psnr)
        SSIM.append(ssim)
        i += 1
        if i == image_num:
            break
    if i == image_num:
        break


print("PSNR_avg:",end='')
print(sum(PSNR)/len(PSNR))
print("SSIM_avg:", end='')
print(sum(SSIM)/len(SSIM))
# write the table to a csv file
with open('./output/record.csv', 'w', newline='') as file:
    print("Writing to csv file")
    writer = csv.writer(file)
    writer.writerows(table)
