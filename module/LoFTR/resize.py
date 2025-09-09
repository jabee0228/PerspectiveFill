import cv2
import numpy as np
from sklearn.cluster import DBSCAN
import os
import shutil
import random

image = cv2.imread("/storage/imageTest1.png")
image = cv2.resize(image, (512, 512))
cv2.imwrite("/storage/imageTest1or.jpg", image)
