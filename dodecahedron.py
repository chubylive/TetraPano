from polFace import polVertex
from polFace import polFace
import numpy as np
import math
from math import pi
import imageio as im
import matplotlib.pyplot as plt
from PIL import Image


degrees = 180 / pi


# dodecahedron

degrees = 180 / pi
A0 = math.asin(1/math.sqrt(3)) * degrees
A1 = math.acos((math.sqrt(5) - 1) / math.sqrt(3) / 2) * degrees
A2 = 90 - A1
A3 = math.acos(-(1 + math.sqrt(5)) / math.sqrt(3) / 2) * degrees
dodecahedronVert = [[[45,A0],[0,A1],[180,A1],[135,A0],[90,A2]],         #1
                    [[45,A0],[A2,0],[-A2,0],[-45,A0],[0,A1]],           #2
                    [[45,A0],[90,A2],[90,-A2],[45,-A0],[A2,0]],         #3
                    [[0,A1],[ -45,A0],[-90,A2],[-135,A0],[180,A1]],     #4
                    [[A2,0],[45,-A0],[0,-A1],[-45,-A0],[-A2,0]],        #5
                    [[90,A2],[135,A0],[A3,0],[135,-A0],[90,-A2]],       #6
                    [[45,-A0],[90,-A2],[135,-A0],[180,-A1],[0,-A1]],    #7
                    [[135,A0],[180,A1],[-135,A0],[-A3,0],[A3,0]],       #8
                    [[-45,A0],[-A2,0],[-45,-A0],[-90,-A2],[-90,A2]],    #9
                    [[-45,-A0],[0,-A1],[180,-A1],[-135,-A0],[-90,-A2]], #10
                    [[135,-A0],[A3,0],[-A3,0],[-135,-A0],[180,-A1]],    #11
                    [[-135,A0],[-90,A2],[-90,-A2],[-135,-A0],[-A3,0]]]  #12

f1 = polFace([polVertex(dodecahedronVert[0][0]), polVertex(dodecahedronVert[0][1]), polVertex(dodecahedronVert[0][2]), 
        polVertex(dodecahedronVert[0][3]), polVertex(dodecahedronVert[0][4])], "DODE","A")
f2 = polFace([polVertex(dodecahedronVert[1][0]), polVertex(dodecahedronVert[1][1]), polVertex(dodecahedronVert[1][2]), 
        polVertex(dodecahedronVert[1][3]), polVertex(dodecahedronVert[1][4])], "DODE","B")
f3 = polFace([polVertex(dodecahedronVert[2][0]), polVertex(dodecahedronVert[2][1]), polVertex(dodecahedronVert[2][2]), 
        polVertex(dodecahedronVert[2][3]), polVertex(dodecahedronVert[2][4])], "DODE","C")
f4 = polFace([polVertex(dodecahedronVert[3][0]), polVertex(dodecahedronVert[3][1]), polVertex(dodecahedronVert[3][2]), 
        polVertex(dodecahedronVert[3][3]), polVertex(dodecahedronVert[3][4])], "DODE","D")
f5 = polFace([polVertex(dodecahedronVert[4][0]), polVertex(dodecahedronVert[4][1]), polVertex(dodecahedronVert[4][2]), 
        polVertex(dodecahedronVert[4][3]), polVertex(dodecahedronVert[4][4])], "DODE","E")                            
f6 = polFace([polVertex(dodecahedronVert[5][0]), polVertex(dodecahedronVert[5][1]), polVertex(dodecahedronVert[5][2]), 
        polVertex(dodecahedronVert[5][3]), polVertex(dodecahedronVert[5][4])], "DODE","F")
f7 = polFace([polVertex(dodecahedronVert[6][0]), polVertex(dodecahedronVert[6][1]), polVertex(dodecahedronVert[6][2]), 
        polVertex(dodecahedronVert[6][3]), polVertex(dodecahedronVert[6][4])], "DODE","G")
f8 = polFace([polVertex(dodecahedronVert[7][0]), polVertex(dodecahedronVert[7][1]), polVertex(dodecahedronVert[7][2]), 
        polVertex(dodecahedronVert[7][3]), polVertex(dodecahedronVert[7][4])], "DODE","H")
f9 = polFace([polVertex(dodecahedronVert[8][0]), polVertex(dodecahedronVert[8][1]), polVertex(dodecahedronVert[8][2]), 
        polVertex(dodecahedronVert[8][3]), polVertex(dodecahedronVert[8][4])], "DODE","I")
f10 = polFace([polVertex(dodecahedronVert[9][0]), polVertex(dodecahedronVert[9][1]), polVertex(dodecahedronVert[9][2]), 
        polVertex(dodecahedronVert[9][3]), polVertex(dodecahedronVert[9][4])], "DODE","J")
f11 = polFace([polVertex(dodecahedronVert[10][0]), polVertex(dodecahedronVert[10][1]), polVertex(dodecahedronVert[10][2]), 
        polVertex(dodecahedronVert[10][3]), polVertex(dodecahedronVert[10][4])], "DODE","K")
f12 = polFace([polVertex(dodecahedronVert[11][0]), polVertex(dodecahedronVert[11][1]), polVertex(dodecahedronVert[11][2]), 
        polVertex(dodecahedronVert[11][3]), polVertex(dodecahedronVert[11][4])], "DODE","L")


img = im.imread('images/test3.jpg')
imageOut =  Image.new("RGB", (int(polFace.upClass*850*4),int(polFace.upClass*850*4)),"white")
imageOut1 =  Image.new("RGB", (int(polFace.upClass*850*4),int(polFace.upClass*850*4)),"white")
imageOut12 =  Image.new("RGB", (int(polFace.upClass*850*4),int(polFace.upClass*850*4)),"white")

faceMap = {}


f1.projectOnToPlace(img,faceMap,imageOut)
f8.projectOnToPlace(img,faceMap,imageOut)
f4.projectOnToPlace(img,faceMap,imageOut)
f2.projectOnToPlace(img,faceMap,imageOut)
f3.projectOnToPlace(img,faceMap,imageOut)
f6.projectOnToPlace(img,faceMap,imageOut)

f10.projectOnToPlace(img,faceMap,imageOut1)
f11.projectOnToPlace(img,faceMap,imageOut1)
f7.projectOnToPlace(img,faceMap,imageOut1)
f9.projectOnToPlace(img,faceMap,imageOut1)
f5.projectOnToPlace(img,faceMap,imageOut1)
f12.projectOnToPlace(img,faceMap,imageOut1)
imageOut.show()
imageOut1.show()

    

dodecahedronParent = [-1,0,4,8,1,2,2,3,1,8,6,3]

