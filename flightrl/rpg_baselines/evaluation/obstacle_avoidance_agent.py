#!/usr/bin/env python3

import numpy as np
import math
from typing import List

from pointCloudGenerator import PointCloudGenerator
import torch

class ObstacleAvoidanceAgent():
  def __init__(self, 
               num_envs : int = 1,
               num_acts : int = 4,
               ):
    self._num_envs = num_envs
    self._num_acts = num_acts
    self._pointCloudGenerator = PointCloudGenerator()
    
    # initialization
        
  def getActions(self, obs, done, images, current_goal_position):
    image = images[0,:,:]

    # save current depthMap for debugging
    #np.savetxt("depthImage.dat",image)

    pointCloud = self._pointCloudGenerator.generatePointCloudFromDepthMap(image)
    

    # ignorantissima pausa per veder che succede
    #input("ciao")

    action = np.zeros([self._num_envs,self._num_acts], dtype=np.float32)
    action[0,0] += -0.01
    return action