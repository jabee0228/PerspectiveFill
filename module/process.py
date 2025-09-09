from module.LoFTR.loftr import loftrGenerate
# from module.LoFTR.findJpg import returnPair
from module.MyVersion.util import inference
import cv2

def process_image(img1, img2, img3, blocksXY):
    # Save the img into the uploads folder
    img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
    img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)
    img3 = cv2.cvtColor(img3, cv2.COLOR_RGB2BGR)
    cv2.imwrite("./uploads/tmp1.jpg", img1)
    cv2.imwrite("./uploads/tmp2.jpg", img2)
    cv2.imwrite("./uploads/tmp3.jpg", img3)
    
    # Process the image
    dir = "./uploads"
    # blocksXY = [0,0,0,0]
    # blocksXY[0] = 150 # x1
    # blocksXY[1] = 200 # x2
    # blocksXY[2] =  250# y1
    # blocksXY[3] = 300 # y2
    loftrGenerate("./uploads/tmp1.jpg", "./uploads/tmp2.jpg", dir ,blocksXY)
    # Save the processed image
    inference(dir)


# dir = input("Please input file dir: ")
# pair = returnPair(dir)

# blocksXY = [0,0,0,0]

# blocksXY[0] = int(input("x1: "))
# blocksXY[1] = int(input("x2: "))
# blocksXY[2] = int(input("y1: "))
# blocksXY[3] = int(input("y2: "))
# loftrGenerate(pair[0][0], pair[0][1], dir ,blocksXY)

# inference(dir)