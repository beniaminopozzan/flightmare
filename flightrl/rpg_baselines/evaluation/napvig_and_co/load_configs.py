import os
from ruamel.yaml import YAML
from napvig_and_co.napvig import NapvigParams
from napvig_and_co.landscape import LandscapeParams

def loadParameters():
    quad_params = YAML().load(open(os.environ["FLIGHTMARE_PATH"] +
                "/flightlib/configs/quadrotor_env.yaml", 'r'))
    control_params = YAML().load(open(os.environ["FLIGHTMARE_PATH"] +
                "/flightrl/rpg_baselines/evaluation/napvig_and_co/configs/controllers.yaml", 'r'))
    
    # pointCloud generator parameters
    point_cloud_params = control_params['pointCloud_gen']
    point_cloud_params['camera_rot'] = quad_params['quadrotor_env']['camera_rot']
    point_cloud_params['camera_pos'] = quad_params['quadrotor_env']['camera_pos']

    # control parameters used by the PID controllers
    PID_control_params = control_params['PID_control_params']
    PID_control_params['T_s'] = quad_params['quadrotor_env']['sim_dt']
    PID_control_params['mass'] = quad_params['quadrotor_dynamics']['mass']
    PID_control_params['kappa'] = quad_params['quadrotor_dynamics']['kappa']
    PID_control_params['arm_l'] = quad_params['quadrotor_dynamics']['arm_l']

    # control parameters used by the mid level controller
    mid_level_control_params = control_params['mid_level_controller']
    
    return point_cloud_params, PID_control_params, mid_level_control_params

def load_napvig_params ():
    napvig_params = NapvigParams ()
    landscape_params = LandscapeParams ()

    config_yaml = YAML().load(open(os.environ["FLIGHTMARE_PATH"] +
                "/flightrl/rpg_baselines/evaluation/napvig_and_co/configs/napvig.yaml", 'r'))
    
    napvig_params.gradient_step_size = config_yaml["napvig"]["gradient_step_size"]
    napvig_params.termination_distance = config_yaml["napvig"]["termination_distance"]
    napvig_params.termination_count = config_yaml["napvig"]["termination_count"]
    napvig_params.step_ahead_size = config_yaml["napvig"]["step_ahead_size"]
    napvig_params.method = config_yaml["napvig"]["method"]

    landscape_params.measure_radius = config_yaml["landscape"]["measure_radius"]
    landscape_params.smooth_radius = config_yaml["landscape"]["smooth_radius"]
    landscape_params.precision = config_yaml["landscape"]["precision"]
    landscape_params.dim = config_yaml["landscape"]["dimensions"]
    landscape_params.decimation = config_yaml["landscape"]["decimation"]

    return napvig_params, landscape_params

if __name__=='__main__':
    loadParameters()