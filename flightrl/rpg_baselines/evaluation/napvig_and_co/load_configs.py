import os
from ruamel.yaml import YAML

def loadParameters():
    quad_params = YAML().load(open(os.environ["FLIGHTMARE_PATH"] +
                "/flightlib/configs/quadrotor_env.yaml", 'r'))
    control_params = YAML().load(open(os.environ["FLIGHTMARE_PATH"] +
                "/flightrl/rpg_baselines/evaluation/napvig_and_co/configs/controllers.yaml", 'r'))
    return quad_params, control_params
