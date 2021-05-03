#!/usr/bin/env python3

import numpy as np
import math
from typing import List

from napvig_and_co.load_configs import loadParameters, load_napvig_params
from napvig_and_co.point_cloud_generator import PointCloudGenerator
from napvig_and_co.quadrotor_PID_controller import QuadrotorPIDcontroller
from napvig_and_co.mid_level_direction_controller import MidLevelDirectionController
from napvig_and_co.napvig import Napvig, State
from scipy.spatial.transform import Rotation as R

import torch

class ObstacleAvoidanceAgent():
  def __init__(self, 
               num_envs : int = 1,
               num_acts : int = 4,
               ):
    self._num_envs = num_envs
    self._num_acts = num_acts

    # retrieve parameters
    point_cloud_gen_params, PID_control_params, mid_level_control_params =  loadParameters()

    # initialize controllers
    self._point_cloud_generator = PointCloudGenerator(**point_cloud_gen_params)
    self._quadrotor_controller = QuadrotorPIDcontroller(PID_control_params)
    self._mid_level_dir_controlelr = MidLevelDirectionController(**mid_level_control_params)
    
    # setup napvig
    napvig_params, landscape_params = load_napvig_params ()

    self.napvig = Napvig (napvig_params, landscape_params)

  
  def reset(self):
    self._quadrotor_controller.reset()

  def getActions(self, obs, done, images, current_goal_position):
    image = images[0,:,:]
    obs = obs[0,:]

    # save current depthMap for debugging
    #np.savetxt("depthImage.dat",image)

    point_cloud = self._point_cloud_generator.generate_point_cloud_from_depthmap(image)
    # save estrapolated point cloud for debugging
    #np.savetxt("pointCloud.dat",point_cloud)
    # Update napvig info


    # Generate trajectory direction


    pos_error = current_goal_position - obs[:3]
    
    angles = R.from_euler('ZYX',obs[3:6]).as_euler('ZYX')
    goal_local = R.from_euler('ZYX',obs[3:6]).apply(pos_error, inverse=True)
    print (point_cloud.shape[0])
    q = State ()
    self.napvig.set_goal (goal_local)
    self.napvig.landscape.set_measures (point_cloud)
    target_state_local = self.napvig.compute (q)
    print (target_state_local.position)

    vel_target = target_state_local.position.numpy() / np.linalg.norm (target_state_local.position.numpy())
    #vel_world = R.from_euler('ZYX', angles)
    quadrotor_refs = self._mid_level_dir_controlelr.genQuatrotorReferences(vel_target)

    action = np.zeros([self._num_envs,self._num_acts], dtype=np.float32)
    action[0,:] = self._quadrotor_controller.apply(obs[:12], quadrotor_refs).reshape(-1)
    
    return action


if __name__=='__main__':
  o = ObstacleAvoidanceAgent()