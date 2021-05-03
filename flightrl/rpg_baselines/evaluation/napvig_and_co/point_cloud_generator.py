#!/usr/bin/env python3

import numpy as np
from scipy.spatial.transform import Rotation as R

class PointCloudGenerator():
    def __init__(self,
                frame_height,
                frame_width,
                camera_FOV,
                camera_rot,
                camera_pos):
        self._frame_height = frame_height # number of vertical pixels
        self._frame_width = frame_width # number of horizontal pixels
        self._camera_FOV = camera_FOV * np.pi / 180 # field of view of the camera
        self._camera_pos = camera_pos # position of the camera w.r.t. the drone

        self.fl = 0.5 / np.tan(self._camera_FOV/2) # focal distance
        # the (x,y) coordinates in the image plane are mapped in the range (-0.5, 0.5)
        # therefore the focal distance results as 0.5 / tan(FoV/2)
        
        self.mask = np.ones((self._frame_height*self._frame_width,3))
        # the mask is a (TotNumberOfPixel x 3) array where each row represent a point
        # the first column is the x-coordinate, the second colum is the y-coordinate
        # and the third column is the z-coodinate
        # the actual point is computed multiplying each row by the point actual depth.
        # The points are in the body reference frame
        ind = 0
        for px in list(range(self._frame_width)):
            x = (px/self._frame_width) - 0.5 # maps to (-0.5, 0.5)
            for py in list(range(self._frame_height)):
                y = (py/self._frame_height) - 0.5 # maps to (-0.5, 0.5)
                self.mask[ind,0] = x / self.fl # the x axis in the image plane is the x axis in the camera frame
                self.mask[ind,2] = y / self.fl # the y axis in the image plane is the inverse of the z axis in the camera frame
                # the depth axis of the image plane is the y axis in the camera frame: self.mask[ind,1] = 1
                ind = ind + 1
        
        # rotate camera mask
        r = (R.from_euler('ZYX', camera_rot, degrees=True)).as_matrix()
        self.mask = np.transpose( np.dot(r, self.mask.transpose()) )

    # given a depthMap compute the point cloud
    def generate_point_cloud_from_depthmap(self, images):
        # apply mask
        point_cloud = self.mask * images.reshape((self._frame_height*self._frame_width,1),order='F')
        # move camera frame origin w.r.t. quadrotor frame origin
        point_cloud = point_cloud + self._camera_pos
        
        return point_cloud