# Copyright 2017 Nitish Mutha (nitishmutha.com)

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from math import pi
import math
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from polFace import polVertex
from polFace import polFace
fovTetraherdron =  [104/2, 104]
focOctahedron = [52/2, 52]
fovCube = [37/2, 37]
dodecahedron = [28/2,28]
Icosohedron = [28/2,28]
# plt.ion()
fig,ax = plt.subplots()
class NFOV():
    def __init__(self, height=850, width=850):
        #Field of view(90, 180)
        #30,90
        self.FOV = [37/2, 37]
        # self.FOV = [90/2, 90] 
        self.PI = pi
        self.PI_2 = pi * 0.5
        self.PI2 = pi * 2.0
        self.height = height
        self.width = width
        self.screen_points = self._get_screen_img()
        self.cp = [0,0]

    def _get_coord_rad(self, isCenterPt, center_point=None):
        # print("center: ",isCenterPt, center_point)
        # print("screen points: ", self.screen_points )
        if isCenterPt:
            return (center_point * 2 - 1) * np.array([self.PI, self.PI_2]) 
        else:
            print("why is this: ", self.screen_points)

            return (self.screen_points * 2 - 1) * np.array([self.PI, self.PI_2]) * (np.ones(self.screen_points.shape) * self.FOV * 1/(180 / pi))
            pass

    def _get_screen_img(self):
        xx, yy = np.meshgrid(np.linspace(0, 1, self.width), np.linspace(0, 1, self.height))
        return np.array([xx.ravel(), yy.ravel()]).T

    def _calcSphericaltoGnomonic(self, convertedScreenCoord):
        x = convertedScreenCoord.T[0]
        y = convertedScreenCoord.T[1]
        print("what is this: ", convertedScreenCoord.T[0])
        plt.plot(x,y,'b.')
        rou = np.sqrt(x ** 2 + y ** 2)
        c = np.arctan(rou)
        sin_c = np.sin(c)
        cos_c = np.cos(c)

        lat = np.arcsin(cos_c * np.sin(self.cp[1]) + (y * sin_c * np.cos(self.cp[1])) / rou)
        # print(lat)
        lon = self.cp[0] + np.arctan2(x * sin_c, rou * np.cos(self.cp[1]) * cos_c - y * np.sin(self.cp[1]) * sin_c)

        lat = (lat / self.PI_2 + 1.) * 0.5
        lon = (lon / self.PI  + 1.) * 0.5
        # print(lat)
        # print("what is this: ", convertedScreenCoord)

        return np.array([lon, lat]).T

    def getTriangleVertex(self, lon1, lat1):
        lat = lat1 * np.pi/180
        lon = lon1 * np.pi/180
        lon1 = self.cp[0]
        lat1 = self.cp[1] 
        print("center degrees ", str(lon1), lat1, lon, lat)       
        # cos_c = (np.sin(lat1) * np.sin(lat)) + (np.cos(lat1) * np.cos(lat) * np.cos(lon - lon1))
        # x = (np.cos(lat) * np.sin(lon - lon1))/cos_c
        # y = ((np.cos(lon1) * np.sin(lon)) - (np.sin(lon1) * np.cos(lon) * np.cos(lat - lat1)))/cos_c

        cos_c = (np.sin(lat1) * np.sin(lat)) + (np.cos(lat1) * np.cos(lat) * np.cos(lon - lon1))
        x = (np.cos(lat) * np.sin(lon - lon1))/cos_c
        y = ((np.cos(lat1) * np.sin(lat)) - (np.sin(lat1) * np.cos(lat) * np.cos(lon - lon1)))/cos_c
        
        return x,y 

    def setCp (self, polVertex):
        self.cp[0] = polVertex.lon  * pi/180
        self.cp[1] = polVertex.lat * pi/180

    def getTriangleVertexArr(self, lonlat):
        lat = lonlat[1] * np.pi/180
        lon = lonlat[0] * np.pi/180
        lon1 = self.cp[0]
        lat1 = self.cp[1] 
        print("lon, lat:" + str(lon1), lat1, lon, lat)       
        # cos_c = (np.sin(lat1) * np.sin(lat)) + (np.cos(lat1) * np.cos(lat) * np.cos(lon - lon1))
        # x = (np.cos(lat) * np.sin(lon - lon1))/cos_c
        # y = ((np.cos(lon1) * np.sin(lon)) - (np.sin(lon1) * np.cos(lon) * np.cos(lat - lat1)))/cos_c

        cos_c = (np.sin(lat1) * np.sin(lat)) + (np.cos(lat1) * np.cos(lat) * np.cos(lon - lon1))
        x = (np.cos(lat) * np.sin(lon - lon1))/cos_c
        y = ((np.cos(lat1) * np.sin(lat)) - (np.sin(lat1) * np.cos(lat) * np.cos(lon - lon1)))/cos_c
        
        return x,y     
    
    def _bilinear_interpolation(self, screen_coord):
        uf = np.mod(screen_coord.T[0],1) * self.frame_width  # long - width
        vf = np.mod(screen_coord.T[1],1) * self.frame_height  # lat - height

        x0 = np.floor(uf).astype(int)  # coord of pixel to bottom left
        y0 = np.floor(vf).astype(int)
        x2 = np.add(x0, np.ones(uf.shape).astype(int))  # coords of pixel to top right
        y2 = np.add(y0, np.ones(vf.shape).astype(int))

        base_y0 = np.multiply(y0, self.frame_width)
        base_y2 = np.multiply(y2, self.frame_width)

        A_idx = np.add(base_y0, x0)
        B_idx = np.add(base_y2, x0)
        lr = []
        count = 0
        for x in B_idx:

            if x >= 2097152:
                lr.append(count)
            count = count + 1
        for idx in lr:
            B_idx[idx] = 2097151
        C_idx = np.add(base_y0, x2)

        D_idx = np.add(base_y2, x2)
        lr = []
        count = 0
        for x in D_idx:

            if x >= 2097152:
                lr.append(count)
            count = count + 1
        for idx in lr:
            D_idx[idx] = 2097151
        # print(B_idx)
        # if B_idx >= 2097152:
        #     B_idx = 2097151
    
        flat_img = np.reshape(self.frame, [-1, self.frame_channel])

        A = np.take(flat_img, A_idx, axis=0)
        B = np.take(flat_img, B_idx, axis=0)
        C = np.take(flat_img, C_idx, axis=0)
        D = np.take(flat_img, D_idx, axis=0)

        wa = np.multiply(x2 - uf, y2 - vf)
        wb = np.multiply(x2 - uf, vf - y0)
        wc = np.multiply(uf - x0, y2 - vf)
        wd = np.multiply(uf - x0, vf - y0)

        # interpolate
        AA = np.multiply(A, np.array([wa, wa, wa]).T)
        BB = np.multiply(B, np.array([wb, wb, wb]).T)
        CC = np.multiply(C, np.array([wc, wc, wc]).T)
        DD = np.multiply(D, np.array([wd, wd, wd]).T)
        nfov = np.reshape(np.round(AA + BB + CC + DD).astype(np.uint8), [self.height, self.width, 3])
        
        return ax.imshow(nfov,animated =True)
        # plt.show()
         # nfov

    def toNFOV(self, frame, center_point):
        # print(frame)
        self.frame = frame
        self.frame_height = frame.shape[0]
        self.frame_width = frame.shape[1]
        self.frame_channel = frame.shape[2]
        self.cp = self._get_coord_rad(center_point=center_point, isCenterPt=True)
        print("center in rad: ", self.cp, "  ", center_point, "    ", self.cp * 180/pi)

        convertedScreenCoord = self._get_coord_rad(isCenterPt=False)
        # print("what is this: ", convertedScreenCoord)
        ax.set_title(str(center_point))
        spericalCoord = self._calcSphericaltoGnomonic(convertedScreenCoord)
        return self._bilinear_interpolation(spericalCoord)

    def toNFOVLatLon(self, frame, center_point):
        # print(frame)
        self.frame = frame
        self.frame_height = frame.shape[0]
        self.frame_width = frame.shape[1]
        self.frame_channel = frame.shape[2]
        self.cp = center_point * pi/180
        print("center in rad: ", self.cp, "    ", self.cp * 180/pi)

        convertedScreenCoord = self._get_coord_rad(isCenterPt=False)
        # print("what is this: ", convertedScreenCoord)
        ax.set_title(str(center_point))
        spericalCoord = self._calcSphericaltoGnomonic(convertedScreenCoord)
        return self._bilinear_interpolation(spericalCoord)

# test the class
if __name__ == '__main__':
    import imageio as im
    # img = im.imread('images/test3.jpg')
    nfov = NFOV()
    # Tetrahedron
    # degrees = 180 / pi
    # asin1_3 = math.asin(1 / 3)
    # vertices = [[0, 90],
    #           [-180, -asin1_3 * degrees],
    #           [-60, -asin1_3 * degrees],
    #           [60, -asin1_3 * degrees]]

    # centroidLat = ((90 + (asin1_3 * degrees))/2) - (asin1_3 * degrees)

    # centers = [[0, -90],
    #           [-120, centroidLat],
    #           [0,centroidLat],
    #           [120, centroidLat]]
    # center_point = np.array([0, -90])
    
    # f1 = polFace([polVertex(vertices[0]), polVertex(vertices[1]), polVertex(vertices[2])])
    # f2 = polFace([polVertex(vertices[0]), polVertex(vertices[2]), polVertex(vertices[3])])
    # f3 = polFace([polVertex(vertices[0]), polVertex(vertices[3]), polVertex(vertices[1])])
    # f4 = polFace([polVertex(vertices[1]), polVertex(vertices[2]), polVertex(vertices[3])])

    # nfov.toNFOVLatLon(img,center_point)
    # x, y  = nfov.getTriangleVertex(vertices[1][0],vertices[1][1])
    # print(x,", "+str(y))
    # plt.plot(x, y, 'r.')
    # x, y  = nfov.getTriangleVertex(vertices[2][0],vertices[2][1])
    # print(x,", "+str(y))
    # plt.plot(x, y, 'r.')
    # x, y  = nfov.getTriangleVertex(vertices[3][0],vertices[3][1])
    # print(x,", "+str(y))
    # plt.plot(x, y, 'r.')
    # x, y  = nfov.getTriangleVertex(center_point[0],center_point[1])
    # print(x,", "+str(y))
    # plt.plot(x, y, 'y.')









    # # cube
    # degrees = 180 / pi
    # asin1_3 = math.asin(1 / 3)
    # phi1 = math.atan(math.sqrt(1/2)) * degrees
    # cube = [[0+45, phi1], [90+45, phi1], [180+45, phi1], [-90+45, phi1],
    #   [0+45, -phi1], [90+45, -phi1], [180+45, -phi1], [-90+45, -phi1]]

    # [ [0, 3, 2, 1], # N
    #   [0, 1, 5, 4],
    #   [1, 2, 6, 5],
    #   [2, 3, 7, 6],
    #   [3, 0, 4, 7],
    #   [4, 5, 6, 7]]  # S

    # f1 = polFace([polVertex(cube[0]), polVertex(cube[3]), polVertex(cube[2]), polVertex(cube[1])])
    # f2 = polFace([polVertex(cube[0]), polVertex(cube[1]), polVertex(cube[5]), polVertex(cube[4])])
    # f3 = polFace([polVertex(cube[1]), polVertex(cube[2]), polVertex(cube[6]), polVertex(cube[5])])
    # f4 = polFace([polVertex(cube[2]), polVertex(cube[3]), polVertex(cube[7]), polVertex(cube[6])])
    # f5 = polFace([polVertex(cube[3]), polVertex(cube[0]), polVertex(cube[4]), polVertex(cube[7])])
    # f6 = polFace([polVertex(cube[4]), polVertex(cube[5]), polVertex(cube[6]), polVertex(cube[7])])

    # f1 = polFace([polVertex(cube[0]), polVertex(cube[3]), polVertex(cube[2]), polVertex(cube[1])])
    # f2 = polFace([polVertex(cube[0]), polVertex(cube[1]), polVertex(cube[5]), polVertex(cube[4])])
    # nfov.setCp(f1.center)
    # img = im.imread('images/test4.jpg')

    # f1.projectOnToPlace(img)
    # x, y  = nfov.getTriangleVertexArr([f1.center.lon,f1.center.lat])
    # print("center ", x,", "+str(y))
    # plt.plot(x, y, 'y.')
    # x, y  = nfov.getTriangleVertexArr(cube[0])
    # print("v1 ", x,", "+str(y))
    # plt.plot(x, y, 'y.')
    # x, y  = nfov.getTriangleVertexArr(cube[3])
    # print("v2 ",x,", "+str(y))
    # plt.plot(x, y, 'r.')
    # x, y  = nfov.getTriangleVertexArr(cube[2])
    # print("v3 ",x,", "+str(y))
    # plt.plot(x, y, 'g.')
    # x, y  = nfov.getTriangleVertexArr(cube[1])
    # print("v4 ",x,", "+str(y))
    # plt.plot(x, y, 'r*')
    










   ## octahedron
    # octahedronFaces = [[0, 90], [-90, 0], [0, 0], [90, 0], [180, 0], [0, -90]]
    # octahedronFaces = [[0, 2, 1], [0, 3, 2], [5, 1, 2], [5, 2, 3], [0, 1, 4], [0, 4, 3], [5, 4, 1], [5, 3, 4]]
    # octahedronParent = [-1, 0, 0, 1, 0, 1, 4, 5]
    # f1 = polFace([polVertex(octahedronFaces[0]), polVertex(octahedronFaces[2]), polVertex(octahedronFaces[1])])
    # f2 = polFace([polVertex(octahedronFaces[0]), polVertex(octahedronFaces[3]), polVertex(octahedronFaces[2])])
    # f3 = polFace([polVertex(octahedronFaces[5]), polVertex(octahedronFaces[1]), polVertex(octahedronFaces[2])])
    # f4 = polFace([polVertex(octahedronFaces[5]), polVertex(octahedronFaces[2]), polVertex(octahedronFaces[3])])
    # f5 = polFace([polVertex(octahedronFaces[0]), polVertex(octahedronFaces[1]), polVertex(octahedronFaces[4])])
    # f6 = polFace([polVertex(octahedronFaces[0]), polVertex(octahedronFaces[4]), polVertex(octahedronFaces[3])])
    # f7 = polFace([polVertex(octahedronFaces[5]), polVertex(octahedronFaces[4]), polVertex(octahedronFaces[1])])
    # f8 = polFace([polVertex(octahedronFaces[5]), polVertex(octahedronFaces[3]), polVertex(octahedronFaces[4])])
   
    # nfov.setCp(f1.center)
    
    # img = im.imread('images/test4.jpg')

    # f1.projectOnToPlace(img)
    # x, y  = nfov.getTriangleVertexArr([f1.center.lon,f1.center.lat])
    # print("center ", x,", "+str(y))
    # plt.plot(x, y, 'y.')
    # x, y  = nfov.getTriangleVertexArr(octahedronFaces[0])
    # print("v1 ", x,", "+str(y))
    # plt.plot(x, y, 'y.')
    # x, y  = nfov.getTriangleVertexArr(octahedronFaces[2])
    # print("v2 ",x,", "+str(y))
    # plt.plot(x, y, 'r.')
    # x, y  = nfov.getTriangleVertexArr(octahedronFaces[1])
    # print("v3 ",x,", "+str(y))
    # plt.plot(x, y, 'g.')









    # # dodecahedron

    # degrees = 180 / pi
    # A0 = math.asin(1/math.sqrt(3)) * degrees
    # A1 = math.acos((math.sqrt(5) - 1) / math.sqrt(3) / 2) * degrees
    # A2 = 90 - A1
    # A3 = math.acos(-(1 + math.sqrt(5)) / math.sqrt(3) / 2) * degrees
    # dodecahedronVert = [[[45,A0],[0,A1],[180,A1],[135,A0],[90,A2]],
    #                     [[45,A0],[A2,0],[-A2,0],[-45,A0],[0,A1]],
    #                     [[45,A0],[90,A2],[90,-A2],[45,-A0],[A2,0]],
    #                     [[0,A1],[ -45,A0],[-90,A2],[-135,A0],[180,A1]],
    #                     [[A2,0],[45,-A0],[0,-A1],[-45,-A0],[-A2,0]],
    #                     [[90,A2],[135,A0],[A3,0],[135,-A0],[90,-A2]],
    #                     [[45,-A0],[90,-A2],[135,-A0],[180,-A1],[0,-A1]],
    #                     [[135,A0],[180,A1],[-135,A0],[-A3,0],[A3,0]],
    #                     [[-45,A0],[-A2,0],[-45,-A0],[-90,-A2],[-90,A2]],
    #                     [[-45,-A0],[0,-A1],[180,-A1],[-135,-A0],[-90,-A2]],
    #                     [[135,-A0],[A3,0],[-A3,0],[-135,-A0],[180,-A1]],
    #                     [[-135,A0],[-90,A2],[-90,-A2],[-135,-A0],[-A3,0]]]

    # f1 = polFace([polVertex(dodecahedronVert[0][0]), polVertex(dodecahedronVert[0][1]), polVertex(dodecahedronVert[0][2]), 
    #         polVertex(dodecahedronVert[0][3]), polVertex(dodecahedronVert[0][4])])
    # f2 = polFace([polVertex(dodecahedronVert[1][0]), polVertex(dodecahedronVert[1][1]), polVertex(dodecahedronVert[1][2]), 
    #         polVertex(dodecahedronVert[1][3]), polVertex(dodecahedronVert[1][4])])
    # f3 = polFace([polVertex(dodecahedronVert[2][0]), polVertex(dodecahedronVert[2][1]), polVertex(dodecahedronVert[2][2]), 
    #         polVertex(dodecahedronVert[2][3]), polVertex(dodecahedronVert[2][4])])
    # f4 = polFace([polVertex(dodecahedronVert[3][0]), polVertex(dodecahedronVert[3][1]), polVertex(dodecahedronVert[3][2]), 
    #         polVertex(dodecahedronVert[3][3]), polVertex(dodecahedronVert[3][4])])
    # f5 = polFace([polVertex(dodecahedronVert[4][0]), polVertex(dodecahedronVert[4][1]), polVertex(dodecahedronVert[4][2]), 
    #         polVertex(dodecahedronVert[4][3]), polVertex(dodecahedronVert[4][4])])                            
    # f6 = polFace([polVertex(dodecahedronVert[5][0]), polVertex(dodecahedronVert[5][1]), polVertex(dodecahedronVert[5][2]), 
    #         polVertex(dodecahedronVert[5][3]), polVertex(dodecahedronVert[5][4])])
    # f7 = polFace([polVertex(dodecahedronVert[6][0]), polVertex(dodecahedronVert[6][1]), polVertex(dodecahedronVert[6][2]), 
    #         polVertex(dodecahedronVert[6][3]), polVertex(dodecahedronVert[6][4])])
    # f8 = polFace([polVertex(dodecahedronVert[7][0]), polVertex(dodecahedronVert[7][1]), polVertex(dodecahedronVert[7][2]), 
    #         polVertex(dodecahedronVert[7][3]), polVertex(dodecahedronVert[7][4])])
    # f9 = polFace([polVertex(dodecahedronVert[8][0]), polVertex(dodecahedronVert[8][1]), polVertex(dodecahedronVert[8][2]), 
    #         polVertex(dodecahedronVert[8][3]), polVertex(dodecahedronVert[8][4])])
    # f10 = polFace([polVertex(dodecahedronVert[9][0]), polVertex(dodecahedronVert[9][1]), polVertex(dodecahedronVert[9][2]), 
    #         polVertex(dodecahedronVert[9][3]), polVertex(dodecahedronVert[9][4])])

    # nfov.setCp(f1.center)
    
    # img = im.imread('images/test4.jpg')

    # f1.projectOnToPlace(img)
    # x, y  = nfov.getTriangleVertexArr([f1.center.lon,f1.center.lat])
    # print("center ", x,", "+str(y))
    # plt.plot(x, y, 'y.')
    # x, y  = nfov.getTriangleVertexArr(dodecahedronVert[0][0])
    # print("v1 ", x,", "+str(y))
    # plt.plot(x, y, 'y.')
    # x, y  = nfov.getTriangleVertexArr(dodecahedronVert[0][1])
    # print("v2 ",x,", "+str(y))
    # plt.plot(x, y, 'r.')
    # x, y  = nfov.getTriangleVertexArr(dodecahedronVert[0][2])
    # print("v3 ",x,", "+str(y))
    # plt.plot(x, y, 'g.')
    # x, y  = nfov.getTriangleVertexArr(dodecahedronVert[0][3])
    # print("v2 ",x,", "+str(y))
    # plt.plot(x, y, 'r.')
    # x, y  = nfov.getTriangleVertexArr(dodecahedronVert[0][4])
    # print("v3 ",x,", "+str(y))
    # plt.plot(x, y, 'g.')

    # dodecahedronParent = [-1,0,4,8,1,2,2,3,1,8,6,3]

    # # # Icosohedron

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
    sign = lambda x: math.copysign(1, x)
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

    f1 = polFace([polVertex(polIcos[0]), polVertex(polIcos[1]), polVertex(polIcos[2])])
    f2 = polFace([polVertex(polIcos[0]), polVertex(polIcos[2]), polVertex(polIcos[3])])
    f3 = polFace([polVertex(polIcos[0]), polVertex(polIcos[3]), polVertex(polIcos[4])])
    f4 = polFace([polVertex(polIcos[0]), polVertex(polIcos[4]), polVertex(polIcos[5])])
    f5 = polFace([polVertex(polIcos[0]), polVertex(polIcos[5]), polVertex(polIcos[1])])
    f6 = polFace([polVertex(polIcos[1]), polVertex(polIcos[7]), polVertex(polIcos[2])])
    f7 = polFace([polVertex(polIcos[2]), polVertex(polIcos[8]), polVertex(polIcos[3])])
    f8 = polFace([polVertex(polIcos[3]), polVertex(polIcos[9]), polVertex(polIcos[4])])
    f9 = polFace([polVertex(polIcos[4]), polVertex(polIcos[10]), polVertex(polIcos[5])])
    f10 = polFace([polVertex(polIcos[5]), polVertex(polIcos[11]), polVertex(polIcos[1])])
    f11 = polFace([polVertex(polIcos[7]), polVertex(polIcos[2]), polVertex(polIcos[8])])
    f12 = polFace([polVertex(polIcos[8]), polVertex(polIcos[3]), polVertex(polIcos[9])])
    f13 = polFace([polVertex(polIcos[9]), polVertex(polIcos[4]), polVertex(polIcos[10])])
    f14 = polFace([polVertex(polIcos[10]), polVertex(polIcos[5]), polVertex(polIcos[11])]) 
    f15 = polFace([polVertex(polIcos[11]), polVertex(polIcos[1]), polVertex(polIcos[7])])
    f16 = polFace([polVertex(polIcos[6]), polVertex(polIcos[7]), polVertex(polIcos[8])])
    f17 = polFace([polVertex(polIcos[6]), polVertex(polIcos[8]), polVertex(polIcos[9])])
    f18 = polFace([polVertex(polIcos[6]), polVertex(polIcos[9]), polVertex(polIcos[10])])
    f19 = polFace([polVertex(polIcos[6]), polVertex(polIcos[10]), polVertex(polIcos[11])]) 
    f20 = polFace([polVertex(polIcos[6]), polVertex(polIcos[11]), polVertex(polIcos[7])])

    degrees = 180 / pi
    icosahedronVert = [[0, 90], [0, -90]]
    for i in range(0,10):
        theta = math.atan(0.5) * degrees;
        phi = (i * 36 + 180) % 360 - 180
        icosahedronVert.append([phi,(theta if i & 1 else -theta)])

    f1 = polFace([polVertex(icosahedronVert[0]), polVertex(icosahedronVert[3]), polVertex(icosahedronVert[11])])
    f2 = polFace([polVertex(icosahedronVert[0]), polVertex(icosahedronVert[5]), polVertex(icosahedronVert[3])])
    f3 = polFace([polVertex(icosahedronVert[0]), polVertex(icosahedronVert[7]), polVertex(icosahedronVert[5])])
    f4 = polFace([polVertex(icosahedronVert[0]), polVertex(icosahedronVert[9]), polVertex(icosahedronVert[7])])
    f5 = polFace([polVertex(icosahedronVert[0]), polVertex(icosahedronVert[11]), polVertex(icosahedronVert[9])])
    f6 = polFace([polVertex(icosahedronVert[2]), polVertex(icosahedronVert[11]), polVertex(icosahedronVert[3])])
    f7 = polFace([polVertex(icosahedronVert[3]), polVertex(icosahedronVert[4]), polVertex(icosahedronVert[2])])
    f8 = polFace([polVertex(icosahedronVert[4]), polVertex(icosahedronVert[3]), polVertex(icosahedronVert[5])])
    f9 = polFace([polVertex(icosahedronVert[5]), polVertex(icosahedronVert[6]), polVertex(icosahedronVert[4])])
    f10 =polFace([polVertex(icosahedronVert[6]), polVertex(icosahedronVert[5]), polVertex(icosahedronVert[7])])
    f11 =polFace([polVertex(icosahedronVert[7]), polVertex(icosahedronVert[8]), polVertex(icosahedronVert[6])])
    f12 =polFace([polVertex(icosahedronVert[8]), polVertex(icosahedronVert[7]), polVertex(icosahedronVert[9])])
    f13 =polFace([polVertex(icosahedronVert[9]), polVertex(icosahedronVert[10]), polVertex(icosahedronVert[8])])
    f14 =polFace([polVertex(icosahedronVert[10]), polVertex(icosahedronVert[9]), polVertex(icosahedronVert[11])])
    f15 =polFace([polVertex(icosahedronVert[11]), polVertex(icosahedronVert[2]), polVertex(icosahedronVert[10])])
    f16 =polFace([polVertex(icosahedronVert[1]), polVertex(icosahedronVert[2]), polVertex(icosahedronVert[4])])
    f17 =polFace([polVertex(icosahedronVert[1]), polVertex(icosahedronVert[4]), polVertex(icosahedronVert[6])])
    f18 =polFace([polVertex(icosahedronVert[1]), polVertex(icosahedronVert[6]), polVertex(icosahedronVert[8])])
    f19 =polFace([polVertex(icosahedronVert[1]), polVertex(icosahedronVert[8]), polVertex(icosahedronVert[10])])
    f20 =polFace([polVertex(icosahedronVert[1]), polVertex(icosahedronVert[10]), polVertex(icosahedronVert[2])])

    
    

    # f1 = polFace([polVertex(icosahedronVert[0]), polVertex(icosahedronVert[3]), polVertex(icosahedronVert[11])])
    # nfov.setCp(f1.center)

    # img = im.imread('images/test5.jpg')

    # f1.projectOnToPlace(img)
    # x, y  = nfov.getTriangleVertexArr([f1.center.lon,f1.center.lat])
    # print("center ", x,", "+str(y))
    # plt.plot(x, y, 'y.')
    # x, y  = nfov.getTriangleVertexArr(icosahedronVert[0])
    # print("v1 ", x,", "+str(y))
    # plt.plot(x, y, 'y.')
    # x, y  = nfov.getTriangleVertexArr(icosahedronVert[3])
    # print("v2 ",x,", "+str(y))
    # plt.plot(x, y, 'r.')
    # x, y  = nfov.getTriangleVertexArr(icosahedronVert[11])
    # print("v3 ",x,", "+str(y))
    # plt.plot(x, y, 'g.')

    # icosahedronFaces = [ [0, 3, 11], [0, 5, 3], [0, 7, 5], [0, 9, 7], [0, 11, 9], // North
    # [2, 11, 3], [3, 4, 2], [4, 3, 5], [5, 6, 4], [6, 5, 7], [7, 8, 6], [8, 7, 9], [9, 10, 8], [10, 9, 11], [11, 2, 10], // Equator 
    # [1, 2, 4], [1, 4, 6], [1, 6, 8], [1, 8, 10], [1, 10, 2] // South 
    # ]

    # icosahedronParents = [-1,0,1,11,3,0,7,1,7,8,9,10,11,12,13,6,8,10,19,15]

    plt.show()