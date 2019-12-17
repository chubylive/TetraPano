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
from PIL import Image
from PIL import ImageDraw
import math
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys
# plt.ion()
# fig,ax = plt.subplots()
up = 6
class NFOV():
    def __init__(self, height=up*850, width=up*850):
        #Field of view(90, 180)
        #30,90
        self.FOV = [104/2, 104]
        # self.FOV = [90/2, 90]
        self.PI = pi
        self.PI_2 = pi * 0.5
        self.PI2 = pi * 2.0
        self.height = height
        self.width = width
        self.screen_points = self._get_screen_img()

    def _get_coord_rad(self, isCenterPt, center_point=None):
        # print("center: ",isCenterPt, center_point)
        # print("screen points: ", self.screen_points )
        if isCenterPt:
            return (center_point * 2 - 1) * np.array([self.PI, self.PI_2])
        else:
            return (self.screen_points * 2 - 1) * np.array([self.PI, self.PI_2]) * (np.ones(self.screen_points.shape) * self.FOV * 1/(180 / pi))
            pass

    def _get_screen_img(self):
        xx, yy = np.meshgrid(np.linspace(0, 1, self.width), np.linspace(0, 1, self.height))
        return np.array([xx.ravel(), yy.ravel()]).T
    def setFOV(self,val1, val2):
        self.FOV = [val1, val2]
    def setframe(self,height, width):
        self.height = height
        self.width = width
    def _calcSphericaltoGnomonic(self, convertedScreenCoord):
        # print(convertedScreenCoord.shape)
        x = convertedScreenCoord.T[0]
        y = convertedScreenCoord.T[1]
        rou = np.sqrt(x ** 2 + y ** 2)
        c = np.arctan(rou)
        sin_c = np.sin(c)
        cos_c = np.cos(c)

        lat = np.arcsin(cos_c * np.sin(self.cp[1]) + (y * sin_c * np.cos(self.cp[1])) / rou)
        lon = self.cp[0] + np.arctan2(x * sin_c, rou * np.cos(self.cp[1]) * cos_c - y * np.sin(self.cp[1]) * sin_c)

        lat = (lat / self.PI_2 + 1.) * 0.5
        lon = (lon / self.PI + 1.) * 0.5

        return np.array([lon, lat]).T

    def getTriangleVertex(self, lon1, lat1):
        lat = lat1 * np.pi/180
        lon = lon1 * np.pi/180
        lon1 = self.cp[0]
        lat1 = self.cp[1]
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
        C_idx = np.add(base_y0, x2)
        D_idx = np.add(base_y2, x2)
        lr = []
        # count = 0
        # for x in B_idx:

        #     if x >= 2097152:
        #         lr.append(count)
        #     count = count + 1
        # for idx in lr:
        #     B_idx[idx] = 2097151

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
        # print (flat_img.shape)
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
        # print (np.round(AA + BB + CC + DD).astype(np.uint8).shape)
        # print (nfov.shape)
        # print(np.round(AA + BB + CC + DD).astype(np.uint8))
        # sys.exit()
        imgPrj = Image.fromarray(nfov,'RGB')
        # ax.imshow(nfov,animated =True)
        return imgPrj
        # plt.show()
         # nfov

    def toNFOV(self, frame, center_point):
        # print(frame)
        self.frame = frame
        self.frame_height = frame.shape[0]
        self.frame_width = frame.shape[1]
        self.frame_channel = frame.shape[2]
        self.cp = self._get_coord_rad(center_point=center_point, isCenterPt=True)
        # print("center in rad: ", self.cp, "  ", center_point, "    ", self.cp * 180/pi)

        convertedScreenCoord = self._get_coord_rad(isCenterPt=False)
        # print("what is this: ", convertedScreenCoord)
        ax.set_title(str(center_point))
        spericalCoord = self._calcSphericaltoGnomonic(convertedScreenCoord)
        x1, y1  = nfov.getTriangleVertex(vertices[1][0],vertices[1][1])
        x2, y2  = nfov.getTriangleVertex(vertices[2][0],vertices[2][1])
        x3, y3  = nfov.getTriangleVertex(vertices[3][0],vertices[3][1])
        spx1 = nfov._calcSphericaltoGnomonic(np.array([[x1], [y1]]).T)
        spx2 = nfov._calcSphericaltoGnomonic(np.array([[x2], [y2]]).T)
        spx3 = nfov._calcSphericaltoGnomonic(np.array([[x3], [y3]]).T)
        self._bilinear_interpolation(np.array([spx1,spx2,spx3]))
        return self._bilinear_interpolation(spericalCoord)

    def toNFOVLatLon(self, frame, center_point):
        # print(frame)
        self.frame = frame
        self.frame_height = frame.shape[0]
        self.frame_width = frame.shape[1]
        self.frame_channel = frame.shape[2]
        self.cp = center_point * pi/180
        # print("center in rad: ", self.cp, "    ", self.cp * 180/pi)

        convertedScreenCoord = self._get_coord_rad(isCenterPt=False)
        # print("what is this: ", convertedScreenCoord)
        # ax.set_title(str(center_point))
        spericalCoord = self._calcSphericaltoGnomonic(convertedScreenCoord)
        imgProj =  self._bilinear_interpolation(spericalCoord)
        return imgProj
def transform(src_tri, dst_tri, src_img, dst_img):
    ((x11,x12), (x21,x22), (x31,x32)) = src_tri
    ((y11,y12), (y21,y22), (y31,y32)) = dst_tri

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

    mask = Image.new('1', dst_img.size)
    maskdraw = ImageDraw.Draw(mask)
    maskdraw.polygon(dst_tri, fill=255)

    dstdraw = ImageDraw.Draw(dst_img)
    dstdraw.polygon(dst_tri, fill=(255,255,255))
    # dst_img.show()
    dst_img.paste(transformed, mask=mask)
    # dst_img.show()

def projectFace(vertex1, vertex2, vertex3, centroid, faceLetter):
    xtrans = 2.8512190492035923
    xscale =  up*149.059049012
    if faceLetter == "A":
        xt = up*850
        yt = up*425

    elif faceLetter == "B":
        xt = up*850
        yt = up*3.3973425855874


    elif faceLetter == "C":
        xt = up*484.8813883760911
        yt = up*635.8013287063926

    elif faceLetter == "D":
        xt = up*1215.1186116239088
        yt = up*635.8013287063926

        # draw tab1
    center_point = np.array(centroid)
    imgProj = nfov.toNFOVLatLon(img,center_point)
    x1, y1  = nfov.getTriangleVertex(vertex1[0],vertex1[1])
    x2, y2  = nfov.getTriangleVertex(vertex2[0],vertex2[1])
    x3, y3  = nfov.getTriangleVertex(vertex3[0],vertex3[1])
    cx,cy = nfov.getTriangleVertex(centroid[0],centroid[1])
    xx1 = (xtrans +  x1)* xscale
    xx2 = (xtrans +  x2)* xscale
    xx3 = (xtrans +  x3)* xscale
    yx1 = (2.8512190492035923 +  y1)* xscale
    yx2 = (2.8512190492035923 +  y2)* xscale
    yx3 = (2.8512190492035923 +  y3)* xscale
    cxx = (cx + 2.8512190492035923) * xscale
    cyx = (cy + 2.8512190492035923) * xscale

    if faceLetter == "A":
        tri1 = [(xx1, yx1), (xx2, yx2), (xx3, yx3)]
    elif faceLetter == "B":
        tri1 = [(xx1, yx1), (xx2, yx2), (xx3, yx3)]
    elif faceLetter == "C":
        tri1 = [(xx3, yx3), (xx1, yx1), (xx2, yx2)]
    elif faceLetter == "D":
        tri1 = [(xx2, yx2), (xx3, yx3), (xx1, yx1)]

    tri2 = [(xx1 + xt, yx1 + yt), (xx2 + xt, yx2 + yt), (xx3 + xt, yx3 + yt)]
    thicknesH = 50 * 2 * up
    thicknesA = 30 * 2 * up
    thicknesB = 40 * 2 * up
    if faceLetter == "B":
        tabDraw = ImageDraw.Draw(imageOut)
        tabDraw.line((xx1 + xt, yx1+ yt) + (xx2 + xt, yx2+ yt),fill=118, width=3)
        tabDraw.line((xx1 + xt, yx1+ yt) + (xx3 + xt, yx3+ yt),fill=118, width=2)
    if faceLetter == "C":
        # draw line
        tabDraw = ImageDraw.Draw(imageOut)
        tabDraw.line(tri2[0] +  (xx1 + xt - thicknesH, yx1 + yt),fill=178, width=4)
        tabDraw.line((xx1 + xt - thicknesH, yx1 + yt) + (xx2 + xt - thicknesA, yx2 + yt - thicknesB),fill=178, width=4)
        # tabDraw.line((xx2 + xt - thicknesH, yx2 + yt) +  (xx2 + xt, yx2 + yt),fill=178)
        tabDraw.line((xx2 + xt + thicknesA, yx2 + yt + thicknesB) +  (xx2 + xt - thicknesA, yx2 + yt - thicknesB),fill=178, width=4)
        tabDraw.line((xx3 + xt - thicknesA, yx3 + yt + thicknesB) +  (xx3 + xt, yx3 + yt),fill=178, width=4)
        tabDraw.line((xx2 + xt + thicknesA, yx2 + yt + thicknesB)+  (xx3 + xt - thicknesA, yx3 + yt + thicknesB),fill=178, width=4)

    elif faceLetter == "D":

        tabDraw = ImageDraw.Draw(imageOut)
        tabDraw.line(tri2[0] +  (xx1 + xt + thicknesH, yx1 + yt),fill=178, width=4)
        tabDraw.line((xx1 + xt + thicknesH, yx1 + yt) +  (xx3 + xt + thicknesA, yx3 + yt - thicknesB),fill=178, width=4)
        tabDraw.line((xx3 + xt, yx3+ yt) +  (xx3 + xt + thicknesA, yx3 + yt - thicknesB),fill=178, width=4)
        tabDraw.line((xx3 + xt, yx3+ yt) + (xx2 + xt, yx2+ yt),fill=118, width=3)

    # print (tri1)
    # print (tri2)

    transform(tri1, tri2, imgProj, imageOut)


# test the class
if __name__ == '__main__':
    import imageio as im
    img = im.imread('images/workBuddies.jpg')
    # img = im.imread('images/test4.jpg')
    # img = im.imread('images/atlas1.jpg')
    inSize = img.size

    imageOut = Image.new("RGB", (int(up*850*3),int(up*850*2)),"white")
    nfov = NFOV()

    degrees = 180 / pi
    asin1_3 = math.asin(1 / 3)
    vertices = [[0, 90],
              [-180, asin1_3 * degrees],
              [-60, asin1_3 * degrees],
              [60, asin1_3 * degrees]]

    centroidLat = -(asin1_3 * degrees)

    centers = [[0, 90],
              [-120, centroidLat],
              [0,centroidLat],
              [120, centroidLat]]

    projectFace(vertices[1], vertices[2], vertices[3], centers[0], "A")
    projectFace(vertices[0], vertices[2], vertices[3], centers[2], "B")
    projectFace(vertices[0], vertices[1], vertices[2], centers[1], "C")
    projectFace(vertices[0], vertices[3], vertices[1], centers[3], "D")
    # imageOut.show()
    imageOut.save("images/test11.jpg")

