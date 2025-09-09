from loftr import loftrGenerate
from findJpg import returnPair, checkDirectory, getDirs
from util import returnBaseName
import random

def run(dir):
    pair = returnPair(dir)
    blocksXY[0] = int(input("x1: "))
    blocksXY[1] = int(input("x2: "))
    blocksXY[2] = int(input("y1: "))
    blocksXY[3] = int(input("y2: "))
    loftrGenerate(pair[0][0], pair[0][1], returnBaseName(dir),blocksXY)

