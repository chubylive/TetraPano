from polFace import polVertex
from polFace import polFace
import numpy as np
import math
from math import pi
import imageio as im
import matplotlib.pyplot as plt

 # # # Icosohedron
sign = lambda x: math.copysign(1, x)

def cartesianToLatLon(X,Y,Z):
        phi = np.arctan2(np.sqrt((X**2)+(Y**2)), Z)
        theta = np.arctan2(Y,X)
        lat = 90 - phi * (180/pi)
        lon = theta * (180/pi)
        return [lon, lat]

def longLatToCartesian(lon, lat):
        R=1
        phi = ((90 - lat)* pi)/180
        theta = (lon)* pi/180

        X = R * math.sin(phi) * math.cos(theta)
        Y = R * math.sin(phi) * math.sin(theta)
        Z = R * math.cos(phi)
        return X,Y,Z
def arctan2(Y, X):
        if (X > 0):
            atan2 = math.atan(Y / X)
        elif( X < 0):
            if (Y ==0 ):
                atan2 = (pi - math.atan(math.abs(Y / X)))
            else:
                atan2 = sign(Y) * (pi - math.atan(math.abs(Y / X)))
         
        elif (X == 0):
            if (Y == 0):
                atan2 = 0
            else:
                atan2 = sign(Y) * pi / 2
        return atan2 
def distancePoint(x1,y1,z1,x2,y2,z2):
        return math.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2 )
    
sqrt5 = math.sqrt(5)
polIcos = []
cartIcos = [[1,0,0,],
[(1/5)*sqrt5,(2/5)*sqrt5,0,],
[(1/5)*sqrt5,(1/10) * (5-sqrt5),(1/10) * math.sqrt(50 + 10*sqrt5),],
[1/sqrt5,(-5-sqrt5)/10,np.sqrt((5 - sqrt5)/10)],
[1/sqrt5,(-5-sqrt5)/10,-(np.sqrt((5 - sqrt5)/10))],
[1/sqrt5,(5-sqrt5)/10,-(np.sqrt((5 + sqrt5)/10))],
[-1,0,0],
[-1/sqrt5,-2/sqrt5,0],
[-1/sqrt5,(-5+sqrt5)/10,-(np.sqrt((5 + sqrt5)/10))],
[-1/sqrt5,(5+sqrt5)/10,-(np.sqrt((5 - sqrt5)/10))],
[-1/sqrt5,(5+sqrt5)/10,(np.sqrt((5 - sqrt5)/10))],
[-1/sqrt5,(-5+sqrt5)/10,(np.sqrt((5 + sqrt5)/10))]]

for elm in cartIcos:
        polIcos.append(cartesianToLatLon(elm[1], elm[2], elm[0]))

print(polIcos)

icosahedronFacesPointUp = [ [0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 5], [0, 5, 1],
[1, 7, 2], [2, 8, 3], [3, 9, 4], [4, 10, 5], [5, 11, 1], [7, 2, 8], [8, 3, 9], [9, 4, 10], [10, 5, 11], [11, 1, 7], 
[6, 7, 8], [6, 8, 9], [6, 9, 10], [6, 10, 11], [6, 11, 7]]

# f1 = polFace([polVertex(polIcos[0]), polVertex(polIcos[1]), polVertex(polIcos[2])],"ICOS")
# f2 = polFace([polVertex(polIcos[0]), polVertex(polIcos[2]), polVertex(polIcos[3])],"ICOS")
# f3 = polFace([polVertex(polIcos[0]), polVertex(polIcos[3]), polVertex(polIcos[4])],"ICOS")
# f4 = polFace([polVertex(polIcos[0]), polVertex(polIcos[4]), polVertex(polIcos[5])],"ICOS")
# f5 = polFace([polVertex(polIcos[0]), polVertex(polIcos[5]), polVertex(polIcos[1])],"ICOS")
# f6 = polFace([polVertex(polIcos[1]), polVertex(polIcos[7]), polVertex(polIcos[2])],"ICOS")
# f7 = polFace([polVertex(polIcos[2]), polVertex(polIcos[8]), polVertex(polIcos[3])],"ICOS")
# f8 = polFace([polVertex(polIcos[3]), polVertex(polIcos[9]), polVertex(polIcos[4])],"ICOS")
# f9 = polFace([polVertex(polIcos[4]), polVertex(polIcos[10]), polVertex(polIcos[5])],"ICOS")
# f10 = polFace([polVertex(polIcos[5]), polVertex(polIcos[11]), polVertex(polIcos[1])],"ICOS")
# f11 = polFace([polVertex(polIcos[7]), polVertex(polIcos[2]), polVertex(polIcos[8])],"ICOS")
# f12 = polFace([polVertex(polIcos[8]), polVertex(polIcos[3]), polVertex(polIcos[9])],"ICOS")
# f13 = polFace([polVertex(polIcos[9]), polVertex(polIcos[4]), polVertex(polIcos[10])],"ICOS")
# f14 = polFace([polVertex(polIcos[10]), polVertex(polIcos[5]), polVertex(polIcos[11])],"ICOS") 
# f15 = polFace([polVertex(polIcos[11]), polVertex(polIcos[1]), polVertex(polIcos[7])],"ICOS")
# f16 = polFace([polVertex(polIcos[6]), polVertex(polIcos[7]), polVertex(polIcos[8])],"ICOS")
# f17 = polFace([polVertex(polIcos[6]), polVertex(polIcos[8]), polVertex(polIcos[9])],"ICOS")
# f18 = polFace([polVertex(polIcos[6]), polVertex(polIcos[9]), polVertex(polIcos[10])],"ICOS")
# f19 = polFace([polVertex(polIcos[6]), polVertex(polIcos[10]), polVertex(polIcos[11])],"ICOS") 
# f20 = polFace([polVertex(polIcos[6]), polVertex(polIcos[11]), polVertex(polIcos[7])],"ICOS")

degrees = 180 / pi
icosahedronVert = [[0, 90], [0, -90]]
for i in range(0,10):
        theta = math.atan(0.5) * degrees;
        phi = (i * 36 + 180) % 360 - 180
        icosahedronVert.append([phi,(theta if i & 1 else -theta)])

f1 = polFace([polVertex(icosahedronVert[0]), polVertex(icosahedronVert[3]), polVertex(icosahedronVert[11])],"ICOS")
f2 = polFace([polVertex(icosahedronVert[0]), polVertex(icosahedronVert[5]), polVertex(icosahedronVert[3])],"ICOS")
f3 = polFace([polVertex(icosahedronVert[0]), polVertex(icosahedronVert[7]), polVertex(icosahedronVert[5])],"ICOS")
f4 = polFace([polVertex(icosahedronVert[0]), polVertex(icosahedronVert[9]), polVertex(icosahedronVert[7])],"ICOS")
f5 = polFace([polVertex(icosahedronVert[0]), polVertex(icosahedronVert[11]), polVertex(icosahedronVert[9])],"ICOS")
f6 = polFace([polVertex(icosahedronVert[2]), polVertex(icosahedronVert[11]), polVertex(icosahedronVert[3])],"ICOS")
f7 = polFace([polVertex(icosahedronVert[3]), polVertex(icosahedronVert[4]), polVertex(icosahedronVert[2])],"ICOS")
f8 = polFace([polVertex(icosahedronVert[4]), polVertex(icosahedronVert[3]), polVertex(icosahedronVert[5])],"ICOS")
f9 = polFace([polVertex(icosahedronVert[5]), polVertex(icosahedronVert[6]), polVertex(icosahedronVert[4])],"ICOS")
f10 =polFace([polVertex(icosahedronVert[6]), polVertex(icosahedronVert[5]), polVertex(icosahedronVert[7])],"ICOS")
f11 =polFace([polVertex(icosahedronVert[7]), polVertex(icosahedronVert[8]), polVertex(icosahedronVert[6])],"ICOS")
f12 =polFace([polVertex(icosahedronVert[8]), polVertex(icosahedronVert[7]), polVertex(icosahedronVert[9])],"ICOS")
f13 =polFace([polVertex(icosahedronVert[9]), polVertex(icosahedronVert[10]), polVertex(icosahedronVert[8])],"ICOS")
f14 =polFace([polVertex(icosahedronVert[10]), polVertex(icosahedronVert[9]), polVertex(icosahedronVert[11])],"ICOS")
f15 =polFace([polVertex(icosahedronVert[11]), polVertex(icosahedronVert[2]), polVertex(icosahedronVert[10])],"ICOS")
f16 =polFace([polVertex(icosahedronVert[1]), polVertex(icosahedronVert[2]), polVertex(icosahedronVert[4])],"ICOS")
f17 =polFace([polVertex(icosahedronVert[1]), polVertex(icosahedronVert[4]), polVertex(icosahedronVert[6])],"ICOS")
f18 =polFace([polVertex(icosahedronVert[1]), polVertex(icosahedronVert[6]), polVertex(icosahedronVert[8])],"ICOS")
f19 =polFace([polVertex(icosahedronVert[1]), polVertex(icosahedronVert[8]), polVertex(icosahedronVert[10])],"ICOS")
f20 =polFace([polVertex(icosahedronVert[1]), polVertex(icosahedronVert[10]), polVertex(icosahedronVert[2])],"ICOS")

img = im.imread('images/atlas1.jpg')

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
f11.projectOnToPlace(img)
f12.projectOnToPlace(img)
f13.projectOnToPlace(img)
f14.projectOnToPlace(img)
f15.projectOnToPlace(img)
f16.projectOnToPlace(img)
f17.projectOnToPlace(img)
f18.projectOnToPlace(img)
f19.projectOnToPlace(img)   
f20.projectOnToPlace(img)    


icosahedronFaces = [ [0, 3, 11], [0, 5, 3], [0, 7, 5], [0, 9, 7], [0, 11, 9], # North
[2, 11, 3], [3, 4, 2], [4, 3, 5], [5, 6, 4], [6, 5, 7], [7, 8, 6], [8, 7, 9], [9, 10, 8], [10, 9, 11], [11, 2, 10], # Equator 
[1, 2, 4], [1, 4, 6], [1, 6, 8], [1, 8, 10], [1, 10, 2]] # South 


icosahedronParents = [-1,0,1,11,3,0,7,1,7,8,9,10,11,12,13,6,8,10,19,15]
