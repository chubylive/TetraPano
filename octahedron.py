from polFace import polVertex
from polFace import polFace
import numpy as np
import math
from math import pi
import imageio as im
import matplotlib.pyplot as plt

degrees = 180 / pi


# octahedron
octahedronFaces = [[0, 90], [-90, 0], [0, 0], [90, 0], [180, 0], [0, -90]]
# octahedronFaces = [[0, 2, 1], [0, 3, 2], [5, 1, 2], [5, 2, 3], [0, 1, 4], [0, 4, 3], [5, 4, 1], [5, 3, 4]]
octahedronParent = [-1, 0, 0, 1, 0, 1, 4, 5]
f1 = polFace([polVertex(octahedronFaces[0]), polVertex(octahedronFaces[2]), polVertex(octahedronFaces[1])], "OCTA")
f2 = polFace([polVertex(octahedronFaces[0]), polVertex(octahedronFaces[3]), polVertex(octahedronFaces[2])], "OCTA")
f3 = polFace([polVertex(octahedronFaces[5]), polVertex(octahedronFaces[1]), polVertex(octahedronFaces[2])], "OCTA")
f4 = polFace([polVertex(octahedronFaces[5]), polVertex(octahedronFaces[2]), polVertex(octahedronFaces[3])], "OCTA")
f5 = polFace([polVertex(octahedronFaces[0]), polVertex(octahedronFaces[1]), polVertex(octahedronFaces[4])], "OCTA")
f6 = polFace([polVertex(octahedronFaces[0]), polVertex(octahedronFaces[4]), polVertex(octahedronFaces[3])], "OCTA")
f7 = polFace([polVertex(octahedronFaces[5]), polVertex(octahedronFaces[4]), polVertex(octahedronFaces[1])], "OCTA")
f8 = polFace([polVertex(octahedronFaces[5]), polVertex(octahedronFaces[3]), polVertex(octahedronFaces[4])], "OCTA")
   
img = im.imread('images/test4.jpg')

f1.projectOnToPlace(img)
f2.projectOnToPlace(img)
f3.projectOnToPlace(img)
f4.projectOnToPlace(img)
f5.projectOnToPlace(img)
f6.projectOnToPlace(img)
f7.projectOnToPlace(img)
f8.projectOnToPlace(img)
