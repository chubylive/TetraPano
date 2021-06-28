from polFace import polVertex
from polFace import polFace
import numpy as np
import math
from math import pi
import imageio as im
import matplotlib.pyplot as plt

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

f1 = polFace([polVertex(vertices[1]), polVertex(vertices[2]), polVertex(vertices[3])], "TETRA")
f2 = polFace([polVertex(vertices[0]), polVertex(vertices[2]), polVertex(vertices[3])], "TETRA")
f3 = polFace([polVertex(vertices[0]), polVertex(vertices[1]), polVertex(vertices[2])], "TETRA")
f4 = polFace([polVertex(vertices[0]), polVertex(vertices[3]), polVertex(vertices[1])], "TETRA")

img = im.imread('images/test2.jpg')

f1.projectOnToPlace(img)
# f2.projectOnToPlace(img)
# f3.projectOnToPlace(img)
# f4.projectOnToPlace(img)
# f5.projectOnToPlace(img)
# f6.projectOnToPlace(img)

# plt.show()