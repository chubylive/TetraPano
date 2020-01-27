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
# plt.ion()
fig,ax = plt.subplots()
class NFOV():
    def __init__(self, height=850, width=850):
        #Field of view(90, 180)
        #30,90
        self.FOV = [45/2, 45]
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
        
        # return ax.imshow(nfov,animated =True)
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
    img = im.imread('images/test3.jpg')
    nfov = NFOV()
    # print(np.linspace(0,1,4))
    #center point (longitude, lattitude)
    # center_point = np.array([0.3333333*2, 0])  # camera center point (valid range [0,1])
    # center_point = np.array([1, 0.3])
    # center_point = np.array([1, -0.3])
    
    # print(img)
    

    # nfov.toNFOV(img, center_point)
    # time.sleep(10)
    # center_point = np.array([1, 0])
    # nfov.toNFOV(img, center_point)

    # ims = []
    # x = np.linspace(0,1,3)
    # y = np.linspace(0.6,0.8,20)
    # for yin in y:
    #     center_point = np.array([0,0.3])
    #     ims.append([nfov.toNFOV(img,center_point)])
    degrees = 180 / pi
    asin1_3 = math.asin(1 / 3)
    vertices = [[0, 90],
              [-180, -asin1_3 * degrees],
              [-60, -asin1_3 * degrees],
              [60, -asin1_3 * degrees]]

    centroidLat = ((90 + (asin1_3 * degrees))/2) - (asin1_3 * degrees)

    centers = [[0, -90],
              [-120, centroidLat],
              [0,centroidLat],
              [120, centroidLat]]
    center_point = np.array([0, -90])
    # nfov.toNFOV(img,center_point)

    # center_point = np.array([0.25, 0.723])
    # nfov.toNFOV(img,center_point)

    # center_point = np.array([0.5, 0.3])
    # nfov.toNFOV(img,center_point)

    # center_point = np.array([0.75, 0.7])
    # nfov.toNFOV(img,center_point)

    # center_point = np.array(centers[0])
    # center_point = np.array([90,35.2644])
    # fig = plt.figure()

    nfov.toNFOVLatLon(img,center_point)
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



    # n1 = nfov.toNFOV(img,center_point)
    # center_point = np.array([1, y[0]])
    # n1 = nfov.toNFOV(img,center_point)
    # center_point = np.array([1, y[1]])
    # n2 = nfov.toNFOV(img,center_point)
    # center_point = np.array([1, y[2]])
    # n3 = nfov.toNFOV(img,center_point)
    # ims.append([n1])
    # ims.append([n2])
    # ims.append([n3])
    # ani = animation.ArtistAnimation(fig, ims, interval=600, blit=True,
    #                                 repeat_delay=10)
    # ani.save('dynamic_images.mp4')
    plt.show()