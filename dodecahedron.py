from polFace import polVertex
from polFace import polFace
import numpy as np
import math
from math import pi
import imageio as im
import matplotlib.pyplot as plt

degrees = 180 / pi


# dodecahedron

degrees = 180 / pi
A0 = math.asin(1/math.sqrt(3)) * degrees
A1 = math.acos((math.sqrt(5) - 1) / math.sqrt(3) / 2) * degrees
A2 = 90 - A1
A3 = math.acos(-(1 + math.sqrt(5)) / math.sqrt(3) / 2) * degrees
dodecahedronVert = [[[45,A0],[0,A1],[180,A1],[135,A0],[90,A2]],
                    [[45,A0],[A2,0],[-A2,0],[-45,A0],[0,A1]],
                    [[45,A0],[90,A2],[90,-A2],[45,-A0],[A2,0]],
                    [[0,A1],[ -45,A0],[-90,A2],[-135,A0],[180,A1]],
                    [[A2,0],[45,-A0],[0,-A1],[-45,-A0],[-A2,0]],
                    [[90,A2],[135,A0],[A3,0],[135,-A0],[90,-A2]],
                    [[45,-A0],[90,-A2],[135,-A0],[180,-A1],[0,-A1]],
                    [[135,A0],[180,A1],[-135,A0],[-A3,0],[A3,0]],
                    [[-45,A0],[-A2,0],[-45,-A0],[-90,-A2],[-90,A2]],
                    [[-45,-A0],[0,-A1],[180,-A1],[-135,-A0],[-90,-A2]],
                    [[135,-A0],[A3,0],[-A3,0],[-135,-A0],[180,-A1]],
                    [[-135,A0],[-90,A2],[-90,-A2],[-135,-A0],[-A3,0]]]

f1 = polFace([polVertex(dodecahedronVert[0][0]), polVertex(dodecahedronVert[0][1]), polVertex(dodecahedronVert[0][2]), 
        polVertex(dodecahedronVert[0][3]), polVertex(dodecahedronVert[0][4])], "DODE")
f2 = polFace([polVertex(dodecahedronVert[1][0]), polVertex(dodecahedronVert[1][1]), polVertex(dodecahedronVert[1][2]), 
        polVertex(dodecahedronVert[1][3]), polVertex(dodecahedronVert[1][4])], "DODE")
f3 = polFace([polVertex(dodecahedronVert[2][0]), polVertex(dodecahedronVert[2][1]), polVertex(dodecahedronVert[2][2]), 
        polVertex(dodecahedronVert[2][3]), polVertex(dodecahedronVert[2][4])], "DODE")
f4 = polFace([polVertex(dodecahedronVert[3][0]), polVertex(dodecahedronVert[3][1]), polVertex(dodecahedronVert[3][2]), 
        polVertex(dodecahedronVert[3][3]), polVertex(dodecahedronVert[3][4])], "DODE")
f5 = polFace([polVertex(dodecahedronVert[4][0]), polVertex(dodecahedronVert[4][1]), polVertex(dodecahedronVert[4][2]), 
        polVertex(dodecahedronVert[4][3]), polVertex(dodecahedronVert[4][4])], "DODE")                            
f6 = polFace([polVertex(dodecahedronVert[5][0]), polVertex(dodecahedronVert[5][1]), polVertex(dodecahedronVert[5][2]), 
        polVertex(dodecahedronVert[5][3]), polVertex(dodecahedronVert[5][4])], "DODE")
f7 = polFace([polVertex(dodecahedronVert[6][0]), polVertex(dodecahedronVert[6][1]), polVertex(dodecahedronVert[6][2]), 
        polVertex(dodecahedronVert[6][3]), polVertex(dodecahedronVert[6][4])], "DODE")
f8 = polFace([polVertex(dodecahedronVert[7][0]), polVertex(dodecahedronVert[7][1]), polVertex(dodecahedronVert[7][2]), 
        polVertex(dodecahedronVert[7][3]), polVertex(dodecahedronVert[7][4])], "DODE")
f9 = polFace([polVertex(dodecahedronVert[8][0]), polVertex(dodecahedronVert[8][1]), polVertex(dodecahedronVert[8][2]), 
        polVertex(dodecahedronVert[8][3]), polVertex(dodecahedronVert[8][4])], "DODE")
f10 = polFace([polVertex(dodecahedronVert[9][0]), polVertex(dodecahedronVert[9][1]), polVertex(dodecahedronVert[9][2]), 
        polVertex(dodecahedronVert[9][3]), polVertex(dodecahedronVert[9][4])], "DODE")


img = im.imread('images/test4.jpg')

f1.projectOnToPlace(img)
f2.projectOnToPlace(img)
f3.projectOnToPlace(img)
f4.projectOnToPlace(img)
f5.projectOnToPlace(img)
f6.projectOnToPlace(img)
f7.projectOnToPlace(img)
f8.projectOnToPlace(img)
f9.projectOnToPlace(img)
f10.projectOnToPlace(img)
    

dodecahedronParent = [-1,0,4,8,1,2,2,3,1,8,6,3]

