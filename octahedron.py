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


    # octahedron
    octahedronFaces = [[0, 90], [-90, 0], [0, 0], [90, 0], [180, 0], [0, -90]]
    # octahedronFaces = [[0, 2, 1], [0, 3, 2], [5, 1, 2], [5, 2, 3], [0, 1, 4], [0, 4, 3], [5, 4, 1], [5, 3, 4]]
    octahedronParent = [-1, 0, 0, 1, 0, 1, 4, 5]
    f1 = polFace([polVertex(octahedronFaces[0]), polVertex(octahedronFaces[1]), polVertex(octahedronFaces[2])], "OCTA","A")
    f2 = polFace([polVertex(octahedronFaces[0]), polVertex(octahedronFaces[3]), polVertex(octahedronFaces[2])], "OCTA","B")
    f3 = polFace([polVertex(octahedronFaces[5]), polVertex(octahedronFaces[1]), polVertex(octahedronFaces[2])], "OCTA","C")
    f4 = polFace([polVertex(octahedronFaces[5]), polVertex(octahedronFaces[2]), polVertex(octahedronFaces[3])], "OCTA","D")
    f5 = polFace([polVertex(octahedronFaces[0]), polVertex(octahedronFaces[1]), polVertex(octahedronFaces[4])], "OCTA","E")
    f6 = polFace([polVertex(octahedronFaces[0]), polVertex(octahedronFaces[4]), polVertex(octahedronFaces[3])], "OCTA","F")
    f7 = polFace([polVertex(octahedronFaces[5]), polVertex(octahedronFaces[4]), polVertex(octahedronFaces[1])], "OCTA","G")
    f8 = polFace([polVertex(octahedronFaces[5]), polVertex(octahedronFaces[3]), polVertex(octahedronFaces[4])], "OCTA","H")
            
    img = im.imread(filePath + fileName)
    # img = im.imread('images/test2.jpg')
    # img = im.imread('images/atlas1.jpg')

    imageOut =  Image.new("RGB", (int(polFace.upClass*850*4),int(polFace.upClass*850*3)),"white")
    faceMap = {}

    f1.projectOnToPlace(img,faceMap,imageOut)
    f2.projectOnToPlace(img,faceMap,imageOut)
    f3.projectOnToPlace(img,faceMap,imageOut)
    f4.projectOnToPlace(img,faceMap,imageOut)
    f5.projectOnToPlace(img,faceMap,imageOut)
    f6.projectOnToPlace(img,faceMap,imageOut)
    f7.projectOnToPlace(img,faceMap,imageOut)
    f8.projectOnToPlace(img,faceMap,imageOut)
    now = datetime.now()
    current_time = now.strftime("%H_%M_%S")   
    imageOut.save(filePath + "octa_" + current_time+ ".jpg")