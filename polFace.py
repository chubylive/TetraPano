import numpy as np
import polVertex
import math
import sys
import matplotlib.pyplot as plt
from math import pi
from PIL import Image
from PIL import ImageDraw

fig,ax = plt.subplots()
pi_2 = pi * 0.5
up = 5
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
def distancePoint(x1,y1,x2,y2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
def distancePointToLine(x0,y0,px1,py1,px2,py2):
    numer = abs((px2 - px1)*(py1 - y0) - (px1 - x0)*(py2 - py1))
    denum = math.sqrt(((px2 - px1)**2) + ((py2 - py1)**2))
    return numer/denum
def rotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy

def drawTabUpLeft(side, imageOut):
    thicknesH = 50 * 2 * up
    thicknesH1 = 57 * 2 * up
    thicknesA = 30 * 2 * up
    thicknesB = 40 * 2 * up
    (t11,t12), (t31,t32) = side
    lineDraw = ImageDraw.Draw(imageOut)
    lineDraw.line((t11,t12) +  (t11 - thicknesH, t12 + thicknesA),fill=178, width=4)
    lineDraw.line((t31,t32) +  (t31, t32 - thicknesH1),fill=178, width=4)
    lineDraw.line((t11 - thicknesH, t12 + thicknesA) +  (t31, t32 - thicknesH1),fill=178, width=4)

def drawTabLeft(side, imageOut):
    thicknesH = 50 * 2 * up
    thicknesH1 = 57 * 2 * up
    thicknesA = 30 * 2 * up
    thicknesB = 40 * 2 * up
    (t11,t12), (t31,t32) = side
    lineDraw = ImageDraw.Draw(imageOut)
    lineDraw.line((t11,t12) +  (t11 - thicknesH, t12 - thicknesA),fill=178, width=4)
    lineDraw.line((t31,t32) +  (t31, t32 + thicknesH1),fill=178, width=4)
    lineDraw.line((t11 - thicknesH, t12 - thicknesA) +  (t31, t32 + thicknesH1),fill=178, width=4)


def drawTabRight(side, imageOut):
    thicknesH = 50 * 2 * up
    thicknesH1 = 57 * 2 * up
    thicknesA = 30 * 2 * up
    thicknesB = 40 * 2 * up
    (t11,t12), (t21,t22) = side
    lineDraw = ImageDraw.Draw(imageOut)
    lineDraw.line((t11,t12) +  (t11 + thicknesH, t12 + thicknesA),fill=178, width=4)
    lineDraw.line((t21,t22) +  (t21, t22 - thicknesH1),fill=178, width=4)
    lineDraw.line((t11 + thicknesH, t12 + thicknesA) +  (t21, t22 - thicknesH1),fill=178, width=4)
    # lineDraw.line((xx1 + xt + thicknesH, yx1 + yt) +  (xx3 + xt + thicknesA, yx3 + yt - thicknesB),fill=178, width=4)
    # lineDraw.line((xx3 + xt, yx3+ yt) +  (xx3 + xt + thicknesA, yx3 + yt - thicknesB),fill=178, width=4)


    # tabDraw = ImageDraw.Draw(imageOut)

def drawHelpLine(side, imageOut):
    (t11,t12), (t21,t22) = side
    lineDraw = ImageDraw.Draw(imageOut)
    lineDraw.line((t11, t12) + (t21,t22),fill=118, width=3)


class polVertex ():
    
    def __init__(self,lonlat):
        self.vertex=np.array([lonlat[0], lonlat[1]])
        self.lon = lonlat[0]
        self.lat = lonlat[1]
class polFace():

    upClass = up
    """docstring for polFace"""
    def __init__(self, polVertexList=[], faceType="TETRA", faceId = ""):
        self.vertexList = polVertexList
        self.center = self.calcCenter()
        self.height = 850  * up
        self.width = 850 * up
        self.faceType = faceType
        self.faceId = faceId
        self.trans = 0
        self.scale = 1
        self.faceMap = {}
        if(faceType == "CUBE"):
            self.FOV = [37/2, 37]
        elif(faceType == "TETRA"):
            self.FOV = [104/2, 104]
        elif(faceType == "OCTA"):
            self.FOV = [52/2, 52]
        elif(faceType == "DODE"):
            self.FOV = [28/2, 28]
        elif(faceType == "ICOS"):
            self.FOV = [28/2, 28]
        else:
            self.FOV = [104/2, 104]

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
        # print(np.max(convertedScreenCoord.T[0]))
        # print(np.max(convertedScreenCoord.T[0]))
        self.trans = np.max(convertedScreenCoord.T[0])
        self.scale = (self.height/2) * (1/self.trans) 
        # print("trans: ", self.trans, "scale: ", self.scale)
        # plt.plot(x,y,'b.')
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
        # print("what is this: ", convertedScreenCoord.T[0])
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
        # print("lon, lat:" + str(lon1), lat1, lon, lat)       
        # cos_c = (np.sin(lat1) * np.sin(lat)) + (np.cos(lat1) * np.cos(lat) * np.cos(lon - lon1))
        # x = (np.cos(lat) * np.sin(lon - lon1))/cos_c
        # y = ((np.cos(lon1) * np.sin(lon)) - (np.sin(lon1) * np.cos(lon) * np.cos(lat - lat1)))/cos_c

        cos_c = (np.sin(lat1) * np.sin(lat)) + (np.cos(lat1) * np.cos(lat) * np.cos(lon - lon1))
        x = (np.cos(lat) * np.sin(lon - lon1))/cos_c
        y = ((np.cos(lat1) * np.sin(lat)) - (np.sin(lat1) * np.cos(lat) * np.cos(lon - lon1)))/cos_c
        
        return np.array([x,y])

    def bilinear_interpolation(self, screen_coord):
        uf = np.mod(screen_coord.T[0],1) * self.frame_width  # long - width
        vf = np.mod(screen_coord.T[1],1) * self.frame_height  # lat - height
        # print(np.shape(vf), self.frame_width, np.shape(screen_coord.T[0]), np.shape(screen_coord[0]))
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
        
        imgPrj = Image.fromarray(nfov,'RGB')
        # ax.imshow(nfov,animated =True)
        return imgPrj
        # plt.show()
         # nfov

    def projectOnToPlace(self, frame,faceMapIn,imageOut):
        self.faceMap = faceMapIn
        # print(frame)
        # imageOut = Image.new("RGB", (int(up*850*3),int(up*850*2)),"white")

        self.frame = frame
        self.frame_height = frame.shape[0]
        self.frame_width = frame.shape[1]
        self.frame_channel = frame.shape[2]
        self.cp = [self.center.lon * pi/180, self.center.lat * pi/180]

        convertedScreenCoord = self.get_coord_rad(isCenterPt=False)
        ax.set_title(str(self.cp))
        spericalCoord = self.getSphericalCordofGnomonic(convertedScreenCoord)
        prjImage =  self.bilinear_interpolation(spericalCoord)
        xyPoints = []
        for vert in self.vertexList:
            xyPoints.append(self.getGnomonicCordofSpherical(vert.lon, vert.lat))
        center_pointXY = self.getGnomonicCordofSpherical(self.center.lon, self.center.lat)
        
        
        # exit()
        if(self.faceType == "TETRA" ):
            faceVert = [(((xyPoints[0])[0] + self.trans) * self.scale,((xyPoints[0])[1] + self.trans) * self.scale),
            (((xyPoints[1])[0] + self.trans) * self.scale,((xyPoints[1])[1] + self.trans) * self.scale),
            (((xyPoints[2])[0] + self.trans) * self.scale,((xyPoints[2])[1] + self.trans) * self.scale)]
            self.transformTri(faceVert,faceVert,prjImage,imageOut)
        elif(self.faceType == "ICOS"):
            faceVert = [(((xyPoints[0])[0] + self.trans) * self.scale,((xyPoints[0])[1] + self.trans) * self.scale),
            (((xyPoints[1])[0] + self.trans) * self.scale,((xyPoints[1])[1] + self.trans) * self.scale),
            (((xyPoints[2])[0] + self.trans) * self.scale,((xyPoints[2])[1] + self.trans) * self.scale)]
            ((x11,x12), (x21,x22), (x31,x32)) = faceVert
            xt = up * (850 )
            yt = up * (850 )

            if (self.faceId == "G"):
                
                faceVertMove = ((x11 + xt, x12 + yt), (x21 + xt, x22 + yt), (x31 + xt, x32 + yt)) 
            elif (self.faceId == "B"):
                gFace = self.faceMap["G"]
                cx,cy = ((center_pointXY[0] + self.trans) * self.scale ),((center_pointXY[1] + self.trans) * self.scale )
                (z11,z12), (z21,z22), (z31,z32) = gFace
                xt = z11 - x31
                yt = z12 - x32
                
                faceVertMove = ((x11 + xt, x12 + yt), (x21 + xt, x22 + yt), (x31 + xt, x32 + yt)) 

                (t11,t12), (t21,t22), (t31,t32) = faceVertMove
                side = (t11,t12), (t31,t32)
                side1 = (t11,t12), (t21,t22)
                drawTabLeft(side, imageOut)
                drawHelpLine(side1, imageOut)
                drawHelpLine(side, imageOut)                                
            elif (self.faceId == "C"): 
                gFace = self.faceMap["L"]
                cx,cy = ((center_pointXY[0] + self.trans) * self.scale ),((center_pointXY[1] + self.trans) * self.scale )
                (z11,z12), (z21,z22), (z31,z32) = gFace
                xt = z21 - x21
                yt = z22 - x22
                
                faceVertMove = ((x11 + xt, x12 + yt), (x21 + xt, x22 + yt), (x31 + xt, x32 + yt)) 
                
                (t11,t12), (t21,t22), (t31,t32) = faceVertMove
                side = (t11,t12), (t31,t32)
                side1 = (t11,t12), (t21,t22)
                drawTabLeft(side, imageOut)
                drawHelpLine(side1, imageOut)
                drawHelpLine(side, imageOut)                                
            elif (self.faceId == "D"):     
                gFace = self.faceMap["L"]
                cx,cy = ((center_pointXY[0] + self.trans) * self.scale ),((center_pointXY[1] + self.trans) * self.scale )
                (z11,z12), (z21,z22), (z31,z32) = gFace
                xt = z21 - x31
                yt = z22 - x32
                
                faceVertMove = ((x11 + xt, x12 + yt), (x21 + xt, x22 + yt), (x31 + xt, x32 + yt)) 

                (t11,t12), (t21,t22), (t31,t32) = faceVertMove
                side = (t11,t12), (t31,t32)
                side1 = (t11,t12), (t21,t22)
                drawTabLeft(side, imageOut)
                drawHelpLine(side1, imageOut)
                drawHelpLine(side, imageOut)                                
            elif (self.faceId == "E"):     
                gFace = self.faceMap["L"]
                cx,cy = ((center_pointXY[0] + self.trans) * self.scale ),((center_pointXY[1] + self.trans) * self.scale )
                (z11,z12), (z21,z22), (z31,z32) = gFace
                xt = z31 - x31
                yt = z32 - x32
                
                faceVertMove = ((x11 + xt, x12 + yt), (x21 + xt, x22 + yt), (x31 + xt, x32 + yt)) 

                (t11,t12), (t21,t22), (t31,t32) = faceVertMove
                side = (t11,t12), (t31,t32)
                side1 = (t11,t12), (t21,t22)
                drawTabLeft(side, imageOut)
                drawHelpLine(side1, imageOut)
                drawHelpLine(side, imageOut)                                
            elif (self.faceId == "F"):     
                gFace = self.faceMap["G"]
                cx,cy = ((center_pointXY[0] + self.trans) * self.scale ),((center_pointXY[1] + self.trans) * self.scale )
                (z11,z12), (z21,z22), (z31,z32) = gFace
                xt = z11 - x31
                yt = z12 - x32
                
                faceVertMove = ((x11 + xt, x12 + yt), (x21 + xt, x22 + yt), (x31 + xt, x32 + yt)) 

            elif (self.faceId == "A"):     
                gFace = self.faceMap["G"]
                cx,cy = ((center_pointXY[0] + self.trans) * self.scale ),((center_pointXY[1] + self.trans) * self.scale )
                (z11,z12), (z21,z22), (z31,z32) = gFace
                xt = z11 - x21
                yt = z12 - x22
                
                faceVertMove = ((x11 + xt, x12 + yt), (x21 + xt, x22 + yt), (x31 + xt, x32 + yt)) 
                
                (t11,t12), (t21,t22), (t31,t32) = faceVertMove
                side = (t11,t12), (t31,t32)
                side1 = (t11,t12), (t21,t22)
                drawTabLeft(side, imageOut)
                drawHelpLine(side1, imageOut)
                drawHelpLine(side, imageOut)                
            elif (self.faceId == "H"):  
                gFace = self.faceMap["G"]
                cx,cy = ((center_pointXY[0] + self.trans) * self.scale ),((center_pointXY[1] + self.trans) * self.scale )
                (z11,z12), (z21,z22), (z31,z32) = gFace
                xt = z11 - x21
                yt = z12 - x22
                
                faceVertMove = ((x11 + xt, x12 + yt), (x21 + xt, x22 + yt), (x31 + xt, x32 + yt)) 

            elif (self.faceId == "I"):
                gFace = self.faceMap["H"]
                cx,cy = ((center_pointXY[0] + self.trans) * self.scale ),((center_pointXY[1] + self.trans) * self.scale )
                (z11,z12), (z21,z22), (z31,z32) = gFace
                xt = z11 - x31
                yt = z12 - x32
                
                faceVertMove = ((x11 + xt, x12 + yt), (x21 + xt, x22 + yt), (x31 + xt, x32 + yt)) 

                (t11,t12), (t21,t22), (t31,t32) = faceVertMove
                side1 = (t11,t12), (t21,t22)
                drawHelpLine(side1, imageOut)
            elif (self.faceId == "J"):
                gFace = self.faceMap["C"]
                cx,cy = ((center_pointXY[0] + self.trans) * self.scale ),((center_pointXY[1] + self.trans) * self.scale )
                (z11,z12), (z21,z22), (z31,z32) = gFace
                xt = z21 - x31
                yt = z22 - x32
                
                faceVertMove = ((x11 + xt, x12 + yt), (x21 + xt, x22 + yt), (x31 + xt, x32 + yt)) 

                (t11,t12), (t21,t22), (t31,t32) = faceVertMove
                side = (t11,t12), (t21,t22)
                drawTabUpLeft(side, imageOut)
            elif (self.faceId == "K"):
                gFace = self.faceMap["L"]
                cx,cy = ((center_pointXY[0] + self.trans) * self.scale ),((center_pointXY[1] + self.trans) * self.scale )
                (z11,z12), (z21,z22), (z31,z32) = gFace
                xt = z21 - x11
                yt = z22 - x12
                
                faceVertMove = ((x11 + xt, x12 + yt), (x21 + xt, x22 + yt), (x31 + xt, x32 + yt)) 

            elif (self.faceId == "L"):
                faceVertMove = ((x11 + xt, x12 + yt), (x21 + xt, x22 + yt), (x31 + xt, x32 + yt)) 

            elif (self.faceId == "M"):
                gFace = self.faceMap["L"]
                cx,cy = ((center_pointXY[0] + self.trans) * self.scale ),((center_pointXY[1] + self.trans) * self.scale )
                (z11,z12), (z21,z22), (z31,z32) = gFace
                xt = z11 - x31
                yt = z12 - x32
                
                faceVertMove = ((x11 + xt, x12 + yt), (x21 + xt, x22 + yt), (x31 + xt, x32 + yt)) 
          
            elif (self.faceId == "N"):
                gFace = self.faceMap["E"]
                cx,cy = ((center_pointXY[0] + self.trans) * self.scale ),((center_pointXY[1] + self.trans) * self.scale )
                (z11,z12), (z21,z22), (z31,z32) = gFace
                xt = z21 - x31
                yt = z22 - x32
                
                faceVertMove = ((x11 + xt, x12 + yt), (x21 + xt, x22 + yt), (x31 + xt, x32 + yt)) 


                (t11,t12), (t21,t22), (t31,t32) = faceVertMove
                side = (t11,t12), (t31,t32)
                drawHelpLine(side, imageOut)
            elif (self.faceId == "O"):
                gFace = self.faceMap["F"]
                cx,cy = ((center_pointXY[0] + self.trans) * self.scale ),((center_pointXY[1] + self.trans) * self.scale )
                (z11,z12), (z21,z22), (z31,z32) = gFace
                xt = z11 - x21
                yt = z12 - x22
                
                faceVertMove = ((x11 + xt, x12 + yt), (x21 + xt, x22 + yt), (x31 + xt, x32 + yt)) 
                (t11,t12), (t21,t22), (t31,t32) = faceVertMove
                side = (t11,t12), (t31,t32)
                drawTabLeft(side, imageOut)

            elif (self.faceId == "P"):
                gFace = self.faceMap["G"]
                cx,cy = ((center_pointXY[0] + self.trans) * self.scale ),((center_pointXY[1] + self.trans) * self.scale )
                (z11,z12), (z21,z22), (z31,z32) = gFace
                xt = z31 - x21
                yt = z32 - x22
                
                faceVertMove = ((x11 + xt, x12 + yt), (x21 + xt, x22 + yt), (x31 + xt, x32 + yt)) 

                (t11,t12), (t21,t22), (t31,t32) = faceVertMove
                side = (t11,t12), (t31,t32)
                side1 = (t11,t12), (t21,t22)
                drawTabRight(side, imageOut)
                drawHelpLine(side1, imageOut)
                drawHelpLine(side, imageOut)

            elif (self.faceId == "Q"):
                gFace = self.faceMap["G"]
                cx,cy = ((center_pointXY[0] + self.trans) * self.scale ),((center_pointXY[1] + self.trans) * self.scale )
                (z11,z12), (z21,z22), (z31,z32) = gFace
                xt = z21 - x21
                yt = z22 - x22
                
                faceVertMove = ((x11 + xt, x12 + yt), (x21 + xt, x22 + yt), (x31 + xt, x32 + yt)) 

                (t11,t12), (t21,t22), (t31,t32) = faceVertMove
                side = (t11,t12), (t31,t32)
                side1 = (t11,t12), (t21,t22)
                drawTabRight(side, imageOut)
                drawHelpLine(side1, imageOut)
                drawHelpLine(side, imageOut)                

            elif (self.faceId == "R"):
                gFace = self.faceMap["L"]
                cx,cy = ((center_pointXY[0] + self.trans) * self.scale ),((center_pointXY[1] + self.trans) * self.scale )
                (z11,z12), (z21,z22), (z31,z32) = gFace
                xt = z11 - x31
                yt = z12 - x32
                
                faceVertMove = ((x11 + xt, x12 + yt), (x21 + xt, x22 + yt), (x31 + xt, x32 + yt)) 

                (t11,t12), (t21,t22), (t31,t32) = faceVertMove
                side = (t11,t12), (t31,t32)
                side1 = (t11,t12), (t21,t22)
                drawTabRight(side, imageOut)
                drawHelpLine(side1, imageOut)
                drawHelpLine(side, imageOut)                

            elif (self.faceId == "S"):
                gFace = self.faceMap["L"]
                cx,cy = ((center_pointXY[0] + self.trans) * self.scale ),((center_pointXY[1] + self.trans) * self.scale )
                (z11,z12), (z21,z22), (z31,z32) = gFace
                xt = z11 - x21
                yt = z12 - x22
                
                faceVertMove = ((x11 + xt, x12 + yt), (x21 + xt, x22 + yt), (x31 + xt, x32 + yt)) 

                (t11,t12), (t21,t22), (t31,t32) = faceVertMove
                side = (t11,t12), (t31,t32)
                side1 = (t11,t12), (t21,t22)
                drawTabRight(side, imageOut)
                drawHelpLine(side1, imageOut)
                drawHelpLine(side, imageOut)                

            elif (self.faceId == "T"):
                gFace = self.faceMap["O"]
                cx,cy = ((center_pointXY[0] + self.trans) * self.scale ),((center_pointXY[1] + self.trans) * self.scale )
                (z11,z12), (z21,z22), (z31,z32) = gFace
                xt = z21 - x31
                yt = z22 - x32
                
                faceVertMove = ((x11 + xt, x12 + yt), (x21 + xt, x22 + yt), (x31 + xt, x32 + yt)) 
                
                (t11,t12), (t21,t22), (t31,t32) = faceVertMove
                side = (t11,t12), (t31,t32)
                side1 = (t11,t12), (t21,t22)
                drawTabRight(side, imageOut)
                drawHelpLine(side1, imageOut)
                drawHelpLine(side, imageOut)


            self.transformTri(faceVert,faceVertMove,prjImage,imageOut)
            self.faceMap[self.faceId] = faceVertMove

        elif(self.faceType == "OCTA"):
            faceVert = [(((xyPoints[0])[0] + self.trans) * self.scale,((xyPoints[0])[1] + self.trans) * self.scale),
            (((xyPoints[1])[0] + self.trans) * self.scale,((xyPoints[1])[1] + self.trans) * self.scale),
            (((xyPoints[2])[0] + self.trans) * self.scale,((xyPoints[2])[1] + self.trans) * self.scale)]
            ((x11,x12), (x21,x22), (x31,x32)) = faceVert
            if (self.faceId == "A"):
                xt = up * (850 + 425 + 425/2)
                yt = up * (850-100)
                faceVertMove = ((x11 + xt, x12 + yt), (x21 + xt, x22 + yt), (x31 + xt, x32 + yt)) 
            elif (self.faceId == "B"):
                aFace = self.faceMap["A"]
                (z11,z12), (z21,z22), (z31,z32) = aFace
                xt = distancePoint(z11,z12, z21,z22)
                faceVertMove = (z11,z12), (z11 + xt,z12), (z31,z32)
            elif (self.faceId == "C"): 
                aFace = self.faceMap["A"]
                (z11,z12), (z21,z22), (z31,z32) = aFace
                yt = distancePointToLine(z11,z12, z21,z22,z31,z32)
                faceVertMove = (z11,z12 - yt * 2), (z21,z22),(z31,z32)
            elif (self.faceId == "D"):     
                bFace = self.faceMap["B"]
                (z11,z12), (z21,z22), (z31,z32) = bFace
                xt = distancePoint(z11,z12, z21,z22)
                faceVertMove = (z31 + xt,z32), (z31,z32), (z21,z22)
            elif (self.faceId == "E"):     
                aFace = self.faceMap["A"]
                (z11,z12), (z21,z22), (z31,z32) = aFace
                xt = distancePoint(z11, z12, z21, z22)
                faceVertMove = (z11,z12), (z21,z22), (z11 - xt,z12)
            elif (self.faceId == "F"):     
                eFace = self.faceMap["E"]
                (z11,z12), (z21,z22), (z31,z32) = eFace
                yt = distancePointToLine(z21,z22, z11,z12,z31,z32)
                faceVertMove = (z11,z12), (z31,z32), (z21 ,z22 + yt*2)
            elif (self.faceId == "G"):     
                eFace = self.faceMap["E"]
                (z11,z12), (z21,z22), (z31,z32) = eFace
                xt = distancePoint(z11, z12, z21, z22)
                faceVertMove = (z21 - xt,z22), (z31,z32), (z21 ,z22)
            elif (self.faceId == "H"):     
                gFace = self.faceMap["G"]
                (z11,z12), (z21,z22), (z31,z32) = gFace
                xt = distancePoint(z11, z12, z21, z22)
                faceVertMove = (z11, z12), (z21 - xt ,z22), (z21 ,z22)

            self.transformTri(faceVert,faceVertMove,prjImage,imageOut)
            self.faceMap[self.faceId] = faceVertMove

            # if :
        elif(self.faceType == "CUBE"):
            # scale = 850/2
            # trans = 1
            faceVert = [(((xyPoints[0])[0] + self.trans) * self.scale,((xyPoints[0])[1] + self.trans) * self.scale),
            (((xyPoints[1])[0] + self.trans) * self.scale,((xyPoints[1])[1] + self.trans) * self.scale),
            (((xyPoints[2])[0] + self.trans) * self.scale,((xyPoints[2])[1] + self.trans) * self.scale),
            (((xyPoints[3])[0] + self.trans) * self.scale,((xyPoints[3])[1] + self.trans) * self.scale)]

            ((x11,x12), (x21,x22), (x31,x32), (x41,x42)) = faceVert
            if (self.faceId == "A"):
                xt  = up * (850 * 2 + 425)
                yt  = up * (850/2)

                faceVertMove = ((x41 + xt, x42 + yt), (x11 + xt, x12 + yt), (x21 + xt, x22 + yt), (x31 + xt, x32 + yt))

            elif (self.faceId == "B"):
                aFace = self.faceMap["A"]
                yt = self.scale * 2
                (z41,z42), (z11,z12), (z21,z22), (z31,z32) = aFace

                faceVertMove = (z11,z12 + yt), (z21,z22 + yt), (z31,z32 + yt), (z41,z42 + yt)
            elif (self.faceId == "F"):
                bFace = self.faceMap["B"]
                yt = self.scale * 2
                (z11,z12), (z21,z22), (z31,z32), (z41,z42) = bFace

                faceVertMove = (z11,z12 + yt), (z21,z22 + yt), (z31,z32 + yt), (z41,z42 + yt)
            elif (self.faceId == "C"):
                bFace = self.faceMap["B"]
                xt = self.scale *2
                (z11,z12), (z21,z22), (z31,z32), (z41,z42) = bFace
                    
                faceVertMove = (z11 + xt, z12), (z21 + xt, z22), (z31 + xt, z32), (z41 + xt, z42)
            elif (self.faceId == "E"):
                bFace = self.faceMap["B"]
                xt = self.scale *2
                (z11,z12), (z21,z22), (z31,z32), (z41,z42) = bFace
                    
                faceVertMove = (z11 - xt, z12), (z21 - xt, z22), (z31 - xt, z32), (z41 - xt, z42)
            elif (self.faceId == "D"):
                eFace = self.faceMap["E"]
                xt = self.scale *2
                (z11,z12), (z21,z22), (z31,z32), (z41,z42) = eFace
                    
                faceVertMove = (z11 - xt, z12), (z21 - xt, z22), (z31 - xt, z32), (z41 - xt, z42)

            self.transformSqr(faceVert,faceVertMove,prjImage,imageOut)
        elif(self.faceType == "DODE"):

            faceVert = [(((xyPoints[0])[0] + self.trans) * self.scale,((xyPoints[0])[1] + self.trans) * self.scale),
            (((xyPoints[1])[0] + self.trans) * self.scale,((xyPoints[1])[1] + self.trans) * self.scale),
            (((xyPoints[2])[0] + self.trans) * self.scale,((xyPoints[2])[1] + self.trans) * self.scale),
            (((xyPoints[3])[0] + self.trans) * self.scale,((xyPoints[3])[1] + self.trans) * self.scale),
            (((xyPoints[4])[0] + self.trans) * self.scale,((xyPoints[4])[1] + self.trans) * self.scale)]
            xt = up * ((850 + 425 ))
            yt = up * (850 + 425/2)
            ((x11,x12), (x21,x22), (x31,x32), (x41,x42), (x51,x52)) = faceVert
            if (self.faceId == "A"):
                faceVertMove = ((x11 + xt, x12 + yt), (x21 + xt, x22 + yt), (x31 + xt, x32 + yt),(x41 + xt, x42 + yt),(x51 + xt, x52 + yt)) 
            elif (self.faceId == "H"):
                aFace = self.faceMap["A"]
                cx,cy = ((center_pointXY[0] + self.trans) * self.scale ),((center_pointXY[1] + self.trans) * self.scale )
                (z11,z12), (z21,z22), (z31,z32), (z41,z42), (z51,z52) = aFace
                (x11,x12) = rotate((cx,cy),(x11,x12),1.25664)
                (x21,x22) = rotate((cx,cy),(x21,x22),1.25664)
                (x31,x32) = rotate((cx,cy),(x31,x32),1.25664)
                (x41,x42) = rotate((cx,cy),(x41,x42),1.25664)
                (x51,x52) = rotate((cx,cy),(x51,x52),1.25664)
                ((x11, x12), (x21, x22), (x31, x32),(x41, x42),(x51, x52)) = ((x11 + xt, x12 + yt), (x21 + xt, x22 + yt), (x31 + xt, x32 + yt),(x41 + xt, x42 + yt),(x51 + xt, x52 + yt)) 
                xt = z31 - x21
                yt = z32 - x22
                
                faceVertMove = ((x11 + xt, x12 + yt), (x21 + xt, x22 + yt), (x31 + xt, x32 + yt),(x41 + xt, x42 + yt),(x51 + xt, x52 + yt)) 
            elif (self.faceId ==  "D"):
                aFace = self.faceMap["A"]
                (z11,z12), (z21,z22), (z31,z32), (z41,z42), (z51,z52) = aFace

                cx,cy = ((center_pointXY[0] + self.trans) * self.scale ),((center_pointXY[1] + self.trans) * self.scale )
                (x11,x12) = rotate((cx,cy),(x11,x12),pi)
                (x21,x22) = rotate((cx,cy),(x21,x22),pi)
                (x31,x32) = rotate((cx,cy),(x31,x32),pi)
                (x41,x42) = rotate((cx,cy),(x41,x42),pi)
                (x51,x52) = rotate((cx,cy),(x51,x52),pi)
                ((x11, x12), (x21, x22), (x31, x32),(x41, x42),(x51, x52)) = ((x11 + xt, x12 + yt), (x21 + xt, x22 + yt), (x31 + xt, x32 + yt),(x41 + xt, x42 + yt),(x51 + xt, x52 + yt)) 
                xt = 0
                yt = z22 - x12
                faceVertMove = ((x11 + xt, x12 + yt), (x21 + xt, x22 + yt), (x31 + xt, x32 + yt),(x41 + xt, x42 + yt),(x51 + xt, x52 + yt)) 

            elif (self.faceId ==  "B"):
                aFace = self.faceMap["A"]
                (z11,z12), (z21,z22), (z31,z32), (z41,z42), (z51,z52) = aFace
                cx,cy = ((center_pointXY[0] + self.trans) * self.scale ),((center_pointXY[1] + self.trans) * self.scale )
                (x11,x12) = rotate((cx,cy),(x11,x12),-2*pi/5 )
                (x21,x22) = rotate((cx,cy),(x21,x22),-2*pi/5 )
                (x31,x32) = rotate((cx,cy),(x31,x32),-2*pi/5 )
                (x41,x42) = rotate((cx,cy),(x41,x42),-2*pi/5 )
                (x51,x52) = rotate((cx,cy),(x51,x52),-2*pi/5 )
                ((x11, x12), (x21, x22), (x31, x32),(x41, x42),(x51, x52)) = ((x11 + xt, x12 + yt), (x21 + xt, x22 + yt), (x31 + xt, x32 + yt),(x41 + xt, x42 + yt),(x51 + xt, x52 + yt)) 

                xt = z11 - x11 
                yt = z12 - x12
                faceVertMove = ((x11 + xt, x12 + yt), (x21 + xt, x22 + yt), (x31 + xt, x32 + yt),(x41 + xt, x42 + yt),(x51 + xt, x52 + yt)) 
            elif (self.faceId ==  "C"):
                aFace = self.faceMap["A"]
                (z11,z12), (z21,z22), (z31,z32), (z41,z42), (z51,z52) = aFace
                cx,cy = ((center_pointXY[0] + self.trans) * self.scale ),((center_pointXY[1] + self.trans) * self.scale )
                (x11,x12) = rotate((cx,cy),(x11,x12),-1*pi/10 )
                (x21,x22) = rotate((cx,cy),(x21,x22),-1*pi/10 )
                (x31,x32) = rotate((cx,cy),(x31,x32),-1*pi/10 )
                (x41,x42) = rotate((cx,cy),(x41,x42),-1*pi/10 )
                (x51,x52) = rotate((cx,cy),(x51,x52),-1*pi/10 )
                ((x11, x12), (x21, x22), (x31, x32),(x41, x42),(x51, x52)) = ((x11 + xt, x12 + yt), (x21 + xt, x22 + yt), (x31 + xt, x32 + yt),(x41 + xt, x42 + yt),(x51 + xt, x52 + yt)) 

                xt = z51 - x21  
                yt = z52 - x22
                faceVertMove = ((x11 + xt, x12 + yt), (x21 + xt, x22 + yt), (x31 + xt, x32 + yt),(x41 + xt, x42 + yt),(x51 + xt, x52 + yt)) 
            elif (self.faceId ==  "F"):
                aFace = self.faceMap["A"]
                (z11,z12), (z21,z22), (z31,z32), (z41,z42), (z51,z52) = aFace
                cx,cy = ((center_pointXY[0] + self.trans) * self.scale ),((center_pointXY[1] + self.trans) * self.scale )
                (x11,x12) = rotate((cx,cy),(x11,x12),pi/10 )
                (x21,x22) = rotate((cx,cy),(x21,x22),pi/10 )
                (x31,x32) = rotate((cx,cy),(x31,x32),pi/10 )
                (x41,x42) = rotate((cx,cy),(x41,x42),pi/10 )
                (x51,x52) = rotate((cx,cy),(x51,x52),pi/10 )
                ((x11, x12), (x21, x22), (x31, x32),(x41, x42),(x51, x52)) = ((x11 + xt, x12 + yt), (x21 + xt, x22 + yt), (x31 + xt, x32 + yt),(x41 + xt, x42 + yt),(x51 + xt, x52 + yt)) 

                xt = z41 - x21
                yt = z42 - x22
                faceVertMove = ((x11 + xt, x12 + yt), (x21 + xt, x22 + yt), (x31 + xt, x32 + yt),(x41 + xt, x42 + yt),(x51 + xt, x52 + yt)) 

            elif (self.faceId ==  "G"):
                aFace = self.faceMap["J"]
                (z11,z12), (z21,z22), (z31,z32), (z41,z42), (z51,z52) = aFace
                cx,cy = ((center_pointXY[0] + self.trans) * self.scale ),((center_pointXY[1] + self.trans) * self.scale )
                ((x11, x12), (x21, x22), (x31, x32),(x41, x42),(x51, x52)) = ((x11 + xt, x12 + yt), (x21 + xt, x22 + yt), (x31 + xt, x32 + yt),(x41 + xt, x42 + yt),(x51 + xt, x52 + yt)) 

                xt = z21 - x51
                yt = z22 - x52
                faceVertMove = ((x11 + xt, x12 + yt), (x21 + xt, x22 + yt), (x31 + xt, x32 + yt),(x41 + xt, x42 + yt),(x51 + xt, x52 + yt)) 
            elif (self.faceId ==  "I"):
                aFace = self.faceMap["J"]
                (z11,z12), (z21,z22), (z31,z32), (z41,z42), (z51,z52) = aFace
                cx,cy = ((center_pointXY[0] + self.trans) * self.scale ),((center_pointXY[1] + self.trans) * self.scale )
                (x11,x12) = rotate((cx,cy),(x11,x12),-1*pi/10 - 1*pi/5 - 4*pi/5)
                (x21,x22) = rotate((cx,cy),(x21,x22),-1*pi/10 - 1*pi/5 - 4*pi/5)
                (x31,x32) = rotate((cx,cy),(x31,x32),-1*pi/10 - 1*pi/5 - 4*pi/5)
                (x41,x42) = rotate((cx,cy),(x41,x42),-1*pi/10 - 1*pi/5 - 4*pi/5)
                (x51,x52) = rotate((cx,cy),(x51,x52),-1*pi/10 - 1*pi/5 - 4*pi/5)
                ((x11, x12), (x21, x22), (x31, x32),(x41, x42),(x51, x52)) = ((x11 + xt, x12 + yt), (x21 + xt, x22 + yt), (x31 + xt, x32 + yt),(x41 + xt, x42 + yt),(x51 + xt, x52 + yt)) 

                xt = z51 - x41
                yt = z52 - x42
                # print(x12,x22,x32,x42,x52)
                faceVertMove = ((x11 + xt, x12 + yt), (x21 + xt, x22 + yt), (x31 + xt, x32 + yt),(x41 + xt, x42 + yt),(x51 + xt, x52 + yt)) 
            elif (self.faceId ==  "J"):
                aFace = self.faceMap["A"]
                (z11,z12), (z21,z22), (z31,z32), (z41,z42), (z51,z52) = aFace
                cx,cy = ((center_pointXY[0] + self.trans) * self.scale ),((center_pointXY[1] + self.trans) * self.scale )
                (x11,x12) = rotate((cx,cy),(x11,x12),-1*pi/5 -4*pi/5 )
                (x21,x22) = rotate((cx,cy),(x21,x22),-1*pi/5 -4*pi/5 )
                (x31,x32) = rotate((cx,cy),(x31,x32),-1*pi/5 -4*pi/5 )
                (x41,x42) = rotate((cx,cy),(x41,x42),-1*pi/5 -4*pi/5 )
                (x51,x52) = rotate((cx,cy),(x51,x52),-1*pi/5 -4*pi/5 )
    
                ((x11, x12), (x21, x22), (x31, x32),(x41, x42),(x51, x52)) = ((x11 + xt, x12 + yt), (x21 + xt, x22 + yt), (x31 + xt, x32 + yt),(x41 + xt, x42 + yt),(x51 + xt, x52 + yt)) 

                xt = 0
                yt = 0
                faceVertMove = ((x11 + xt, x12 + yt), (x21 + xt, x22 + yt), (x31 + xt, x32 + yt),(x41 + xt, x42 + yt),(x51 + xt, x52 + yt)) 
            elif (self.faceId ==  "K"):
                aFace = self.faceMap["J"]
                (z11,z12), (z21,z22), (z31,z32), (z41,z42), (z51,z52) = aFace
                cx,cy = ((center_pointXY[0] + self.trans) * self.scale ),((center_pointXY[1] + self.trans) * self.scale )
                (x11,x12) = rotate((cx,cy),(x11,x12),pi )
                (x21,x22) = rotate((cx,cy),(x21,x22),pi )
                (x31,x32) = rotate((cx,cy),(x31,x32),pi )
                (x41,x42) = rotate((cx,cy),(x41,x42),pi )
                (x51,x52) = rotate((cx,cy),(x51,x52),pi )

                (x11,x12) = rotate((cx,cy),(x11,x12),2*pi/5 )
                (x21,x22) = rotate((cx,cy),(x21,x22),2*pi/5 )
                (x31,x32) = rotate((cx,cy),(x31,x32),2*pi/5 )
                (x41,x42) = rotate((cx,cy),(x41,x42),2*pi/5 )
                (x51,x52) = rotate((cx,cy),(x51,x52),2*pi/5 )
                ((x11, x12), (x21, x22), (x31, x32),(x41, x42),(x51, x52)) = ((x11 + xt, x12 + yt), (x21 + xt, x22 + yt), (x31 + xt, x32 + yt),(x41 + xt, x42 + yt),(x51 + xt, x52 + yt)) 

                xt = z31 - x51
                yt = z32 - x52

                faceVertMove = ((x11 + xt, x12 + yt), (x21 + xt, x22 + yt), (x31 + xt, x32 + yt),(x41 + xt, x42 + yt),(x51 + xt, x52 + yt)) 
            elif (self.faceId ==  "L"):
                aFace = self.faceMap["J"]
                (z11,z12), (z21,z22), (z31,z32), (z41,z42), (z51,z52) = aFace
                cx,cy = ((center_pointXY[0] + self.trans) * self.scale ),((center_pointXY[1] + self.trans) * self.scale )
                (x11,x12) = rotate((cx,cy),(x11,x12),pi/10 - pi)
                (x21,x22) = rotate((cx,cy),(x21,x22),pi/10 - pi)
                (x31,x32) = rotate((cx,cy),(x31,x32),pi/10 - pi)
                (x41,x42) = rotate((cx,cy),(x41,x42),pi/10 - pi)
                (x51,x52) = rotate((cx,cy),(x51,x52),pi/10 - pi)
                ((x11, x12), (x21, x22), (x31, x32),(x41, x42),(x51, x52)) = ((x11 + xt, x12 + yt), (x21 + xt, x22 + yt), (x31 + xt, x32 + yt),(x41 + xt, x42 + yt),(x51 + xt, x52 + yt)) 

                xt = z51 - x31
                yt = z52 - x32
                # print(x12,x22,x32,x42,x52)
                faceVertMove = ((x11 + xt, x12 + yt), (x21 + xt, x22 + yt), (x31 + xt, x32 + yt),(x41 + xt, x42 + yt),(x51 + xt, x52 + yt)) 
            elif (self.faceId ==  "E"):
                aFace = self.faceMap["J"]
                (z11,z12), (z21,z22), (z31,z32), (z41,z42), (z51,z52) = aFace
                cx,cy = ((center_pointXY[0] + self.trans) * self.scale ),((center_pointXY[1] + self.trans) * self.scale )
                (x11,x12) = rotate((cx,cy),(x11,x12),1*pi/5 + 2*pi/5)
                (x21,x22) = rotate((cx,cy),(x21,x22),1*pi/5 + 2*pi/5)
                (x31,x32) = rotate((cx,cy),(x31,x32),1*pi/5 + 2*pi/5)
                (x41,x42) = rotate((cx,cy),(x41,x42),1*pi/5 + 2*pi/5)
                (x51,x52) = rotate((cx,cy),(x51,x52),1*pi/5 + 2*pi/5)
                ((x11, x12), (x21, x22), (x31, x32),(x41, x42),(x51, x52)) = ((x11 + xt, x12 + yt), (x21 + xt, x22 + yt), (x31 + xt, x32 + yt),(x41 + xt, x42 + yt),(x51 + xt, x52 + yt)) 

                xt = z21 - x31
                yt = z22 - x32
                # print(x12,x22,x32,x42,x52)
                faceVertMove = ((x11 + xt, x12 + yt), (x21 + xt, x22 + yt), (x31 + xt, x32 + yt),(x41 + xt, x42 + yt),(x51 + xt, x52 + yt)) 

            self.transformPent(faceVert,faceVertMove,prjImage,imageOut)
            self.faceMap[self.faceId] = faceVertMove

        else:
            print("not exist")
        # imageOut.save("images/test11.jpg")

    def getFaceMap(self):
        return self.faceMap

    def setFaceMap(self,faceMapIn):
        self.faceMap = faceMapIn

    def transformTri(self, src_tri, dst_tri, src_img, dst_img):
        # print("tranforming trianglar face")
        ((x11,x12), (x21,x22), (x31,x32)) = src_tri
        ((y11,y12), (y21,y22), (y31,y32)) = dst_tri
        # print(src_tri)
        # print(dst_tri)
        # print("EndParam")

        M = np.array([
                         [y11, y12, 1, 0, 0, 0],
                         [y21, y22, 1, 0, 0, 0],
                         [y31, y32, 1, 0, 0, 0],
                         [0, 0, 0, y11, y12, 1],
                         [0, 0, 0, y21, y22, 1],
                         [0, 0, 0, y31, y32, 1]
                    ])

        y = np.array([x11, x21, x31, x12, x22, x32])
        A = np.linalg.solve(M, y)
        src_copy = src_img.copy()
        srcdraw = ImageDraw.Draw(src_copy)
        srcdraw.polygon(src_tri)
        # src_copy.show()
        transformed = src_img.transform(dst_img.size, Image.AFFINE, A)
        # transformed.show()
        # exit()
        mask = Image.new('1', dst_img.size)
        maskdraw = ImageDraw.Draw(mask)
        maskdraw.polygon(dst_tri, fill=255)

        dstdraw = ImageDraw.Draw(dst_img)
        dstdraw.polygon(dst_tri, fill=(255,255,255))
        # dst_img.show()
        dst_img.paste(transformed, mask=mask)
        # dst_img.show()

    
    def transformSqr(self, src_Sqr, dst_Sqr, src_img, dst_img):
        # print("transforming square face")
        # print(src_Sqr)
        # print(dst_Sqr);
        # print("EndParam")
        ((x11,x12), (x21,x22), (x31,x32), (x41,x42)) = src_Sqr
        ((y11,y12), (y21,y22), (y31,y32), (y41,y42)) = dst_Sqr
        triInSqr = ((x11,x12), (x21,x22), (x31,x32))
        triInSqr1 = ((x11,x12), (x41,x42), (x31,x32))
        triInSqrMove = ((y11,y12), (y21,y22), (y31,y32))
        triInSqr1Move = ((y11,y12), (y41,y42), (y31,y32))
        

        
        self.transformTri(triInSqr, triInSqrMove, src_img, dst_img)
        self.transformTri(triInSqr1, triInSqr1Move, src_img, dst_img)    
        self.faceMap[self.faceId] = dst_Sqr


        

    def transformPent(self, src_Sqr, dst_Sqr, src_img, dst_img):
        # print("transforming pentagon face")
        ((x11,x12), (x21,x22), (x31,x32), (x41,x42), (x51,x52)) = src_Sqr
        ((y11,y12), (y21,y22), (y31,y32), (y41,y42), (y51,y52)) = dst_Sqr
        triA = ((x11,x12), (x21,x22), (x31,x32))
        triB = ((x11,x12), (x31,x32), (x41,x42))
        triC = ((x11,x12), (x41,x42), (x51,x52))
        triA_Move = ((y11,y12), (y21,y22), (y31,y32))
        triB_Move = ((y11,y12), (y31,y32), (y41,y42))
        triC_Move = ((y11,y12), (y41,y42), (y51,y52))
        self.transformTri(triA, triA_Move, src_img, dst_img)
        self.transformTri(triB, triB_Move, src_img, dst_img) 
        self.transformTri(triC, triC_Move, src_img, dst_img)