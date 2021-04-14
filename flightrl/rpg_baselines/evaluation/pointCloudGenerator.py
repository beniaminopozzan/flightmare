#!/usr/bin/env python3

import numpy as np
import torch

class PointCloudGenerator():
    def __init__(self,
                frame_height : int = 128,
                frame_width : int = 128,
                camera_FOV : int = 90):
        self._frame_height = frame_height # number of vertical pixels
        self._frame_width = frame_width # number of horizontal pixels
        self._camera_FOV = camera_FOV * np.pi / 180 # field of view of the camera

        self.fl = 0.5 / np.tan(self._camera_FOV/2) # focal distance
        # the (x,y) coordinates in the image plane are mapped in the range (-0.5, 0.5)
        # therefore the focal distance results as 0.5 / tan(FoV/2)
        
        self.mask = np.ones((self._frame_height*self._frame_width,3))
        # the mask is a (TotNumberOfPixel x 3) array where each row represent a point
        # the first column is the x-coordinate, the second colum is the y-coordinate
        # and the third column is the z-coodinate normalized to 1
        # the actual point is computed multiplying each row by the point actual depth.
        ind = 0
        for px in list(range(self._frame_width)):
            x = (px/self._frame_width) - 0.5 # maps to (-0.5, 0.5)
            for py in list(range(self._frame_height)):
                y = -(py/self._frame_height) + 0.5 # maps to (-0.5, 0.5) with inversion
                self.mask[ind,0] = x / self.fl
                self.mask[ind,1] = y / self.fl
                ind = ind + 1

    # given a depthMap compute the pointCloud
    def generatePointCloudFromDepthMap(self, images):
        pointCloud = self.mask * images.reshape((self._frame_height*self._frame_width,1),order='F')
        # save estrapolated pointCloud for debugging
        #np.savetxt("pointCloud.dat",pointCloud)
        return torch.from_numpy(pointCloud)