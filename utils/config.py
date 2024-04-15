# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import sys
import yaml

from isaacgym import gymapi
from isaacgym import gymutil

import numpy as np
import random
import torch
import argparse

import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def set_np_formatting():
    np.set_printoptions(edgeitems=30, infstr='inf',
                        linewidth=4000, nanstr='nan', precision=2,
                        suppress=False, threshold=10000, formatter=None)
    
def set_seed(seed, torch_deterministic=False):
    if seed == -1 and torch_deterministic:
        seed = 42
    elif seed == -1:
        seed = np.random.randint(0, 10000)
    print("Setting seed: {}".format(seed))

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if torch_deterministic:
        # refer to https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        #torch.set_deterministic(True)
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    return seed

def load_cfg(args):
    train_cfg = yaml.safe_load(open(os.path.join(BASE_DIR,"../cfg/train/{}/{}_{}.yaml".format(args.algo, args.algo, args.task)))) 
    env_cfg = yaml.safe_load(open(os.path.join(BASE_DIR,"../cfg/task/{}.yaml".format(args.task)))) 

    return env_cfg, train_cfg

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--task',type=str, default="ur5cabinet_door", help='choose a task')
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--algo', default='sac', type=str)
    parser.add_argument('--use_gpu', default=True, type=bool) #for physicX and isaac
    parser.add_argument('--sim_device', type=str, default="cuda:0", help='Physics Device in PyTorch-like syntax')
    parser.add_argument('--device_id', default=0, type=int)
    parser.add_argument('--printlog', type=bool, default=True)


    args = parser.parse_args()
    args.physics_engine = gymapi.SIM_PHYSX
    if args.use_gpu:
        args.device_type = "cuda"
    else:
        args.device_type = "cpu"
    return args

def parse_sim_params(args, env_cfg):
    # initialize sim
    sim_params = gymapi.SimParams()
    sim_params.dt = 1./60.
    #sim_params.num_client_threads = args.slices

    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
    sim_params.substeps = 2

    # set simulation parameters (Only support PhysX engine)
    sim_params.physx.rest_offset = 0.0
    sim_params.physx.contact_offset = 0.001
    sim_params.physx.friction_offset_threshold = 0.001
    sim_params.physx.friction_correlation_distance = 0.0005

    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 8
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.num_threads = 4

    #sim_params.physx.max_gpu_contact_pairs = 8 * 1024

    sim_params.use_gpu_pipeline = args.use_gpu
    sim_params.physx.use_gpu = args.use_gpu

    # if sim options are provided in cfg, parse them and update/override above:
    if "sim" in env_cfg:
        gymutil.parse_sim_config(env_cfg["sim"], sim_params)


    return sim_params