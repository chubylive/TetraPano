import numpy as np
import polVertex
import math
import sys
import matplotlib.pyplot as plt
from math import pi

fig,ax = plt.subplots()
pi_2 = pi * 0.5
def longLatToCartesian(lon, lat):
        R=1
        phi = ((90 - lat)* pi)/180
        theta = (lon)* pi/180

        X = R * math.sin(phi) * math.cos(theta)
        Y = R * math.sin(phi) * math.sin(theta)
        Z = R * math.cos(phi)
        return X,Y,Z

def cartesianToLatLon(X,Y,Z):
    phi = np.arctan2(np.sqrt((X**2)+(Y**2)), Z)
    theta = np.arctan2(Y,X)
    lat = 90 - phi * (180/pi)
    lon = theta * (180/pi)
    return lon, lat

def distancePoint(x1,y1,z1,x2,y2,z2):
   return math.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2 )
class polVertex ():
    def __init__(self,lonlat):
        self.vertex=np.array([lonlat[0], lonlat[1]])
        self.lon = lonlat[0]
        self.lat = lonlat[1]
class polFace():
    """docstring for polFace"""
    def __init__(self, polVertexList=[]):
        self.vertexList = polVertexList
        self.center = self.calcCenter()
        self.height = 850
        self.width = 850
        self.FOV = [28/2, 28]
        self.screen_points = self.get_screen_img()
        # cx,cy,cz = longLatToCartesian(self.center.lon, self.center.lat)
        # vx,vy,vz = longLatToCartesian(self.vertexList[0].lon, self.vertexList[0].lat)
        # vx1,vy1,vz1 = longLatToCartesian(self.vertexList[3].lon, self.vertexList[3].lat)

        # toCenter = distancePoint(0,0,0,cx,cy,cz)
        # # print(cx,cy,cz, "   ",toCenter)
        # toVetex = distancePoint(vx1,vy1,vz1,vx,vy,vz)
        # # print(vx,vy,vz, "  vertex ",toVetex)
        # fov = np.degrees(np.arctan((toVetex/2)/1) * 2 )
        # print(fov)
        # sys.exit()


    def calcCenter(self):
        #first convert to cartesian
        xAcc = 0
        yAcc = 0
        zAcc = 0
        div = 0
        for vert in self.vertexList:
            X,Y,Z = longLatToCartesian(vert.lon, vert.lat)
            # print("lon: " , str(vert.lon) , " lat: " , str(vert.lat) ,"xyz:  ",X,Y,Z)
            xAcc = xAcc + X
            yAcc = yAcc + Y
            zAcc = zAcc + Z
            div = div + 1

        xCtrd = xAcc/div
        yCtrd = yAcc/div
        zCtrd = zAcc/div
        # print("inSideSurf: " , xCtrd,yCtrd,zCtrd)
        ctrdVector  = math.sqrt((xCtrd * xCtrd) + (yCtrd * yCtrd) + (zCtrd * zCtrd))
        sufX = round(1 * xCtrd/ctrdVector,16)
        sufY = round(1 * yCtrd/ctrdVector,16)
        sufZ = round(1 * zCtrd/ctrdVector,16)
        # print("surf: " , sufX,sufY,sufZ)
        lon , lat = cartesianToLatLon(sufX,sufY,sufZ)
        return polVertex([lon, lat])

    def setFOV(self,val1, val2):
        self.FOV = [val1, val2]

    def setframe(self,height, width):
        self.height = height
        self.width = width

    def get_screen_img(self):
        xx, yy = np.meshgrid(np.linspace(0, 1, self.width), np.linspace(0, 1, self.height))
        return np.array([xx.ravel(), yy.ravel()]).T

    def get_coord_rad(self, isCenterPt, center_point=None):
        if isCenterPt:
            return (center_point *2 -1) * np.array([pi, pi_2]) 
        else:
            return (self.screen_points *2 -1) * np.array([pi, pi_2]) * (np.ones(self.screen_points.shape) * self.FOV * 1/(180 / pi))
            pass

    def getSphericalCordofGnomonic(self, convertedScreenCoord):
        x = convertedScreenCoord.T[0]
        y = convertedScreenCoord.T[1]
        plt.plot(x,y,'b.')
        rou = np.sqrt(x ** 2 + y ** 2)
        c = np.arctan(rou)
        sin_c = np.sin(c)
        cos_c = np.cos(c)

        lat = np.arcsin(cos_c * np.sin(self.cp[1]) + (y * sin_c * np.cos(self.cp[1])) / rou)
        # print(lat)
        lon = self.cp[0] + np.arctan2(x * sin_c, rou * np.cos(self.cp[1]) * cos_c - y * np.sin(self.cp[1]) * sin_c)

        lat = (lat / pi_2 + 1.) * 0.5
        lon = (lon / pi + 1.) * 0.5
        # print(lat)
        # print("what is this: ", convertedScreenCoord)

        return np.array([lon, lat]).T

    def getSphericalCordofGnomonicXY(self, x, y):
        print("what is this: ", convertedScreenCoord.T[0])
        plt.plot(x,y,'b.')
        rou = np.sqrt(x ** 2 + y ** 2)
        c = np.arctan(rou)
        sin_c = np.sin(c)
        cos_c = np.cos(c)

        lat = np.arcsin(cos_c * np.sin(self.cp[1]) + (y * sin_c * np.cos(self.cp[1])) / rou)
        # print(lat)
        lon = self.cp[0] + np.arctan2(x * sin_c, rou * np.cos(self.cp[1]) * cos_c - y * np.sin(self.cp[1]) * sin_c)

        lat = (lat / pi_2 + 1.) * 0.5
        lon = (lon / pi + 1.) * 0.5
        # print(lat)
        # print("what is this: ", convertedScreenCoord)

        return lon, lat

    def getGnomonicCordofSpherical(self, lon1, lat1):
        lat = lat1 * np.pi/180
        lon = lon1 * np.pi/180
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

    def bilinear_interpolation(self, screen_coord):
        uf = np.mod(screen_coord.T[0],1) * self.frame_width  # long - width
        vf = np.mod(screen_coord.T[1],1) * self.frame_height  # lat - height
        print(np.shape(vf), self.frame_width, np.shape(screen_coord.T[0]), np.shape(screen_coord[0]))
        x0 = np.floor(uf).astype(int)  # coord of pixel to bottom left
        y0 = np.floor(vf).astype(int)
        x2 = np.add(x0, np.ones(uf.shape).astype(int))  # coords of pixel to top right
        y2 = np.add(y0, np.ones(vf.shape).astype(int))

        base_y0 = np.multiply(y0, self.frame_width)
        base_y2 = np.multiply(y2, self.frame_width)

        A_idx = np.add(base_y0, x0)
        B_idx = np.add(base_y2, x0)
        # lr = []
        # count = 0
        # for x in B_idx:

        #     if x >= 2097152:
        #         lr.append(count)
        #     count = count + 1
        # for idx in lr:
        #     B_idx[idx] = 2097151
        C_idx = np.add(base_y0, x2)

        D_idx = np.add(base_y2, x2)
        # lr = []
        # count = 0
        # for x in D_idx:

        #     if x >= 2097152:
        #         lr.append(count)
        #     count = count + 1
        # for idx in lr:
        #     D_idx[idx] = 2097151
        # print(B_idx)
        # if B_idx >= 2097152:
        #     B_idx = 2097151
    
        flat_img = np.reshape(self.frame, [-1, self.frame_channel])
        print(len(flat_img), np.shape(A_idx), np.shape(B_idx), np.shape(C_idx), np.shape(D_idx))

        for idx in range(0,len(A_idx)):
            if A_idx[idx] >= len(flat_img) :
                A_idx[idx] = len(flat_img) - 1

        for idx in range(0,len(B_idx)):
            if B_idx[idx] >= len(flat_img) :
                B_idx[idx] = len(flat_img) - 1

        for idx in range(0,len(C_idx)):
            if C_idx[idx] >= len(flat_img) :
                C_idx[idx] = len(flat_img) - 1   

        for idx in range(0,len(D_idx)):
            if D_idx[idx] >= len(flat_img) :
                D_idx[idx] = len(flat_img) - 1

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

    def projectOnToPlace(self, frame):
        # print(frame)
        self.frame = frame
        self.frame_height = frame.shape[0]
        self.frame_width = frame.shape[1]
        self.frame_channel = frame.shape[2]
        self.cp = [self.center.lon * pi/180, self.center.lat * pi/180]

        convertedScreenCoord = self.get_coord_rad(isCenterPt=False)
        ax.set_title(str(self.cp))
        spericalCoord = self.getSphericalCordofGnomonic(convertedScreenCoord)
        return self.bilinear_interpolation(spericalCoord)


degrees = 180 / pi
asin1_3 = math.asin(1 / 3)
phi1 = math.atan(math.sqrt(1/2)) * degrees
cube = [[0, phi1], [90, phi1], [180, phi1], [-90, phi1],
  [0, -phi1], [90, -phi1], [180, -phi1], [-90, -phi1]]

[ [0, 3, 2, 1], # N
  [0, 1, 5, 4],
  [1, 2, 6, 5],
  [2, 3, 7, 6],
  [3, 0, 4, 7],
  [4, 5, 6, 7]]  # S

f1 = polFace([polVertex(cube[0]), polVertex(cube[3]), polVertex(cube[2]), polVertex(cube[1])])