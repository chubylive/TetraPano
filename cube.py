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
      asin1_3 = math.asin(1/3)
      phi1 = math.atan(math.sqrt(1/2)) * degrees
      cube = [[0+45, phi1], [90+45, phi1], [180+45, phi1], [-90+45, phi1],
            [0+45, -phi1], [90+45, -phi1], [180+45, -phi1], [-90+45, -phi1]]


      f1 = polFace([polVertex(cube[0]), polVertex(cube[3]), polVertex(cube[2]), polVertex(cube[1])], "CUBE","A")
      f2 = polFace([polVertex(cube[0]), polVertex(cube[1]), polVertex(cube[5]), polVertex(cube[4])], "CUBE","B")
      f3 = polFace([polVertex(cube[1]), polVertex(cube[2]), polVertex(cube[6]), polVertex(cube[5])], "CUBE","E")
      f4 = polFace([polVertex(cube[2]), polVertex(cube[3]), polVertex(cube[7]), polVertex(cube[6])], "CUBE","D")
      f5 = polFace([polVertex(cube[3]), polVertex(cube[0]), polVertex(cube[4]), polVertex(cube[7])], "CUBE","C")
      f6 = polFace([polVertex(cube[4]), polVertex(cube[5]), polVertex(cube[6]), polVertex(cube[7])], "CUBE","F")

      img = im.imread(filePath + fileName)
      imageOut =  Image.new("RGB", (int(polFace.upClass*850*5),int(polFace.upClass*850*4)),"white")

      faceMap = {}
      f1.projectOnToPlace(img,faceMap,imageOut)
      f2.projectOnToPlace(img,faceMap,imageOut)
      f3.projectOnToPlace(img,faceMap,imageOut)
      f5.projectOnToPlace(img,faceMap,imageOut)
      f6.projectOnToPlace(img,faceMap,imageOut)
      f4.projectOnToPlace(img,faceMap,imageOut)
      now = datetime.now()
      current_time = now.strftime("%H_%M_%S")   
      imageOut.save(filePath + "cube_" + current_time+ ".jpg")
      
makeNet("images/" , "test.jpg", 4)