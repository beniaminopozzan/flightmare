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

class AgentState:
  def __init__ (self, obs = np.zeros ((1,12),dtype=np.float)):
    self.set_obs (obs)

  def set_obs (self, obs):
    self.position = obs[0,0:3]
    self.rpy = R.from_euler('ZYX',obs[0,3:6]).as_euler('ZYX')
    self.rotation = R.from_euler ('ZYX',self.rpy)
    self.linear_velocity = obs[0,6:9]
    self.angular_velocity = obs[0,9:12]

class ObstacleAvoidanceAgent():
  def __init__(self, 
               num_envs : int = 1,
               num_acts : int = 4,
               ):
    self._num_envs = num_envs
    self._num_acts = num_acts
    self.state = AgentState ()

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

    # Pointcloud generation
    point_cloud = self._point_cloud_generator.generate_point_cloud_from_depthmap(image)

    # Read agent state
    self.state.set_obs (obs)

    # Convert goal to body frame
    goal_body = self.state.rotation.apply (current_goal_position - self.state.position, inverse=True) 

    # Setup NAPVIG
    q = State ()
    self.napvig.set_goal (goal_body)
    self.napvig.landscape.set_measures (point_cloud)

    # Get trajectory sample
    target_state_body = self.napvig.compute (q)
    target_pos_body = target_state_body.position.numpy()
    vel_target_body = target_pos_body / np.linalg.norm (target_pos_body)

    # Convert to controller frame
    vel_target_world = self.state.rotation.apply (vel_target_body)
    vel_target_ctl = R.from_euler ("Z", self.state.rpy[0]).apply (vel_target_world, inverse=True)

    quadrotor_refs = self._mid_level_dir_controlelr.genQuatrotorReferences(vel_target_ctl)

    action = np.zeros([self._num_envs,self._num_acts], dtype=np.float32)
    action[0,:] = self._quadrotor_controller.apply(obs[0,:12], quadrotor_refs).reshape(-1)
    
    return action


if __name__=='__main__':
  o = ObstacleAvoidanceAgent()