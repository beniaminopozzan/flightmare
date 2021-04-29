#!/usr/bin/env python3

import numpy as np
import math
from typing import List

from napvig_and_co.load_configs import loadParameters
from napvig_and_co.point_cloud_generator import PointCloudGenerator
from napvig_and_co.quadrotor_PID_controller import QuadrotorPIDcontroller
from napvig_and_co.mid_level_direction_controller import MidLevelDirectionController
from scipy.spatial.transform import Rotation as R

import torch

class ObstacleAvoidanceAgent():
  def __init__(self, 
               num_envs : int = 1,
               num_acts : int = 4,
               ):
    self._num_envs = num_envs
    self._num_acts = num_acts

    quat_par, contr_par =  loadParameters()
    #print(contr_par)


    self._point_cloud_generator = PointCloudGenerator()
    self._quadrotor_controller = QuadrotorPIDcontroller()
    self._mid_level_dir_controlelr = MidLevelDirectionController()
    
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

    des_vel_global = np.array([-2, 1, 0])
    des_vel_local = R.from_euler('Z',obs[0,3]).apply(des_vel_global, inverse=True)


    quadrotor_refs = self._mid_level_dir_controlelr.genQuatrotorReferences(des_vel_local)

    action[0,:] = self._quadrotor_controller.apply(obs[0,:12], quadrotor_refs).reshape(-1)
    #print(action)
    action = (action - 0.73*9.81/4) / (0.73*9.81/2)

    # ignorantissima pausa per veder che succede
    
    # print([obs[0,0], obs[0,4], obs[0,10]])
    # input("ciao")

    
    return action