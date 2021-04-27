#!/usr/bin/env python3

import numpy as np
import math
from typing import List

from point_cloud_generator import PointCloudGenerator
from quadrotor_PID_controller import QuadrotorPIDcontroller

import torch

class ObstacleAvoidanceAgent():
  def __init__(self, 
               num_envs : int = 1,
               num_acts : int = 4,
               ):
    self._num_envs = num_envs
    self._num_acts = num_acts
    self._point_cloud_generator = PointCloudGenerator()
    self._quadrotor_controller = QuadrotorPIDcontroller()
    
    # initialization
        
  def getActions(self, obs, done, images, current_goal_position):
    image = images[0,:,:]

    # save current depthMap for debugging
    #np.savetxt("depthImage.dat",image)

    point_cloud = self._point_cloud_generator.generate_point_cloud_from_depthmap(image)
    # save estrapolated point cloud for debugging
    #np.savetxt("pointCloud.dat",point_cloud)


    action = np.zeros([self._num_envs,self._num_acts], dtype=np.float32)
    action[0,0] += 0.1
    action[0,1] -= 0.1
    action[0,2] += 0.1
    action[0,3] -= 0.1

    action[0,:] = self._quadrotor_controller.apply(obs[0,:12],[0,4,0,0]).reshape(-1)
    #print(action)
    action = (action - 0.73*9.81/4) / (0.73*9.81/2)

    # ignorantissima pausa per veder che succede
    
    # print([obs[0,0], obs[0,4], obs[0,10]])
    # input("ciao")

    
    return action