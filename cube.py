from polFace import polVertex
from polFace import polFace
import numpy as np
import math
from math import pi
import imageio as im
import matplotlib.pyplot as plt

degrees = 180 / pi
asin1_3 = math.asin(1/3)
phi1 = math.atan(math.sqrt(1/2)) * degrees
cube = [[0, phi1], [90, phi1], [180, phi1], [-90, phi1],
  [0, -phi1], [90, -phi1], [180, -phi1], [-90, -phi1]]


f1 = polFace([polVertex(cube[0]), polVertex(cube[3]), polVertex(cube[2]), polVertex(cube[1])])
f2 = polFace([polVertex(cube[0]), polVertex(cube[1]), polVertex(cube[5]), polVertex(cube[4])])
f3 = polFace([polVertex(cube[1]), polVertex(cube[2]), polVertex(cube[6]), polVertex(cube[5])])
f4 = polFace([polVertex(cube[2]), polVertex(cube[3]), polVertex(cube[7]), polVertex(cube[6])])
f5 = polFace([polVertex(cube[3]), polVertex(cube[0]), polVertex(cube[4]), polVertex(cube[7])])
f6 = polFace([polVertex(cube[4]), polVertex(cube[5]), polVertex(cube[6]), polVertex(cube[7])])

img = im.imread('images/test2.jpg')

f1.projectOnToPlace(img)
f2.projectOnToPlace(img)
f3.projectOnToPlace(img)
f4.projectOnToPlace(img)
f5.projectOnToPlace(img)
f6.projectOnToPlace(img)

# plt.show()