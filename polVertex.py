import numpy as np
import math
import sys
from math import pi
def longLatToCartesian(lon, lat):
        R=1
        phi = ((90 - lat)* pi)/180
        theta = (lon)* pi/180

        X = R * math.sin(phi) * math.cos(theta)
        Y = R * math.sin(phi) * math.sin(theta)
        Z = R * math.cos(phi)
        return X,Y,Z

def cartesianToLatLon(X,Y,Z):
    phi = np.arctan2((math.sqrt(X*X)+math.sqrt(Y*Y)), Z)
    theta = np.arctan2(Y,X)
    lat = 90 - phi * (180/pi)
    lon = theta * (180/pi)
    return lon, lat

class polVertex ():
    def __init__(self, lon, lat):
        self.vertex=np.array([lon,lat])
        self.lon = lon
        self.lat = lat

class polFace():
    """docstring for polFace"""
    def __init__(self, polVertexList):
        self.vertexList = polVertexList
        self.center = self.calcCenter()


    def calcCenter(self):
        #first convert to cartesian
        xAcc = 0
        yAcc = 0
        zAcc = 0
        div = 0
        for vert in self.vertexList:
            X,Y,Z = longLatToCartesian(vert.lon, vert.lat)
            print("lon: " , str(vert.lon) , " lat: " , str(vert.lat) ,"xyz:  ",X,Y,Z)
            xAcc = xAcc + X
            yAcc = yAcc + Y
            zAcc = zAcc + Z
            div = div + 1

        xCtrd = xAcc/div
        yCtrd = yAcc/div
        zCtrd = zAcc/div
        print("inSideSurf: " , xCtrd,yCtrd,zCtrd)
        ctrdVector  = math.sqrt((xCtrd * xCtrd) + (yCtrd * yCtrd) + (zCtrd * zCtrd))
        sufX = round(1 * xCtrd/ctrdVector,15)
        sufY = round(1 * yCtrd/ctrdVector,15)
        sufZ = round(1 * zCtrd/ctrdVector,15)
        print("surf: " , sufX,sufY,sufZ)
        lon , lat = cartesianToLatLon(sufX,sufY,sufZ)
        return polVertex(lon, lat)

    

degrees = 180 / pi
asin1_3 = math.asin(1 / 3)
phi1 = math.atan(math.sqrt(1/2)) * degrees
cube = [[0, phi1], [90, phi1], [180, phi1], [-90, phi1],
  [0, -phi1], [90, -phi1], [180, -phi1], [-90, -phi1]]

f1 = polFace([polVertex(0, -phi1),
              polVertex(90, -phi1),
              polVertex(180, -phi1),
              polVertex( -90, -phi1)])

print(f1.center.lon, f1.center.lat)
centroidLat = -(asin1_3 * degrees)

x,y,z = longLatToCartesian(0,centroidLat)
print("xyz: ", x,y,z)
l, ln = cartesianToLatLon(x,y,z)
print("lonLat: " ,l,ln)