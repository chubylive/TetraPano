from polFace import polVertex
from polFace import polFace
import polFace as pf
import numpy as np
import math
from math import pi
import imageio as im
import matplotlib.pyplot as plt
from PIL import Image
from datetime import datetime

def makeNet(filePath, fileName, res):
    pf.setRes(res)
    degrees = 180 / pi
    asin1_3 = math.asin(1 / 3)
    vertices = [[0, 90],
            [-180, asin1_3 * degrees],
            [-60, asin1_3 * degrees],
            [60, asin1_3 * degrees]]

        # centroidLat = -(asin1_3 * degrees)

        # centers = [[0, 90],
        #           [-120, centroidLat],
        #           [0,centroidLat],
        #           [120, centroidLat]]

    f1 = polFace([polVertex(vertices[1]), polVertex(vertices[2]), polVertex(vertices[3])], "TETRA","A")
    f2 = polFace([polVertex(vertices[0]), polVertex(vertices[2]), polVertex(vertices[3])], "TETRA","B")
    f3 = polFace([polVertex(vertices[0]), polVertex(vertices[1]), polVertex(vertices[2])], "TETRA","C")
    f4 = polFace([polVertex(vertices[0]), polVertex(vertices[3]), polVertex(vertices[1])], "TETRA","D")
    
    img = im.imread(filePath + fileName)
    # img = im.imread('images/test2.jpg')
    imageOut =  Image.new("RGB", (int(polFace.upClass*850*5),int(polFace.upClass*850*4)),"white")
    faceMap = {}

    f1.projectOnToPlace(img,faceMap,imageOut)
    f2.projectOnToPlace(img,faceMap,imageOut)
    f3.projectOnToPlace(img,faceMap,imageOut)
    f4.projectOnToPlace(img,faceMap,imageOut)

    # f2.projectOnToPlace(img)
    # f3.projectOnToPlace(img)
    # f4.projectOnToPlace(img)
    # f5.projectOnToPlace(img)
    # f6.projectOnToPlace(img)

    # plt.show()`
    now = datetime.now()
    current_time = now.strftime("%H_%M_%S")   
    imageOut.save(filePath + "tetra_" + current_time+ ".jpg")