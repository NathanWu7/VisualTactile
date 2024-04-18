       
from tasks.ur5lift import Ur5lift
from tasks.ur5pickandplace import Ur5pickandplace
from tasks.ur5cabinet import Ur5cabinet
from tasks.ur5cabinet_door import Ur5cabinet_door
#from tasks.franka_cube_stack import FrankaCubeStack
from tasks.base.vec_task import VecTaskCPU, VecTaskGPU, VecTaskPython, VecTaskPythonArm
from RL.ppo.ppo import PPO
#from RL.ppo_pc.ppopc import PPOPC
from RL.sac.sac import SAC
from RL.td3.td3 import TD3
from RL.test.test import TEST
from RL.pc_vtsyne.vtsyne import vtsyne
from RL.pc_vtafford.vtafford import vtafford
from RL.pc_vtafford.vtpolicy import vtpolicy

def parse_task(args, env_cfg, train_cfg, sim_params):
    device_id = args.device_id
    rl_device = args.sim_device
    if args.task == "ur5lift":
        env = Ur5lift(env_cfg,sim_params, args.physics_engine, args.device_type, args.device_id, args.headless)
        vec_env = VecTaskPython(env, rl_device)
    elif args.task == "ur5pickandplace":
        env = Ur5pickandplace(env_cfg,sim_params, args.physics_engine, args.device_type, args.device_id, args.headless)
        vec_env = VecTaskPython(env, rl_device)
    elif args.task == "ur5cabinet":
        env = Ur5cabinet(env_cfg,sim_params, args.physics_engine, args.device_type, args.device_id, args.headless)
        vec_env = VecTaskPython(env, rl_device)
    elif args.task == "ur5cabinet_door":
        env = Ur5cabinet_door(env_cfg,sim_params, args.physics_engine, args.device_type, args.device_id, args.headless)
        vec_env = VecTaskPython(env, rl_device)
    
    if args.algo == 'ppo':
        task = PPO( vec_env,
                    train_cfg,
                    device=rl_device,
                    sampler='sequential',
                    log_dir='run',
                    is_testing=args.test,
                    print_log=args.printlog,
                    apply_reset=False,
                    asymmetric=False)
        
    elif args.algo == 'ppopc':
        task = PPOPC( vec_env,
                    train_cfg,
                    device=rl_device,
                    sampler='sequential',
                    log_dir='run',
                    is_testing=args.test,
                    print_log=args.printlog,
                    apply_reset=False,
                    asymmetric=False)
        
    elif args.algo == 'vts':
        task = vtsyne(vec_env,
                         train_cfg,
                         log_dir='run',
                         is_testing = args.test,
                         device=rl_device)
        
    elif args.algo == 'vta':
        task = vtafford(vec_env,
                         train_cfg,
                         log_dir='run',
                         is_testing = args.test,
                         device=rl_device)
    elif args.algo == 'vtp':
        task = vtpolicy(vec_env,
                         train_cfg,
                         log_dir='run',
                         is_testing = args.test,
                         device=rl_device)
        
    elif args.algo == 'sac':
        task = SAC(vec_env,
                   train_cfg,
                #  actor_critic = MLPActorCritic,
                #  ac_kwargs=dict(),
                #  num_transitions_per_env=8,
                #  num_learning_epochs=5,
                #  num_mini_batches=100,
                #  replay_size=100000,
                #  gamma=0.99,
                #  polyak=0.99,
                #  learning_rate=1e-3,
                #  max_grad_norm =0.5,
                #  entropy_coef=0.2,
                #  use_clipped_value_loss=True,
                #  reward_scale=1,
                #  batch_size=32,
                 device=rl_device,
                 sampler='random',
                 log_dir='run',
                 is_testing=args.test,
                 print_log=True,
                 apply_reset=False,
                 asymmetric=False
                 )
    elif args.algo == 'td3':
        task = TD3(vec_env,
                 train_cfg,
                 device=rl_device,
                 sampler='random',
                 log_dir='run',
                 is_testing=args.test,
                 print_log=True,
                 apply_reset=False,
                 asymmetric=False)
    elif args.algo == 'test':
        task = TEST(vec_env, 
                    device=rl_device)

    return task