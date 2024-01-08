       
from tasks.ur5pickup import Ur5pickup
from tasks.base.vec_task import VecTaskCPU, VecTaskGPU, VecTaskPython, VecTaskPythonArm
from RL.ppo.ppo import PPO
from RL.ppo_pc.ppopc import PPOPC
from RL.sac.sac import SAC
from RL.td3.td3 import TD3
from RL.trpo.trpo import TRPO
from RL.test.test import TEST

def parse_task(args, env_cfg, train_cfg, sim_params):
    device_id = args.device_id
    rl_device = args.sim_device
    if args.task == "cartpole":
        env = Cartpole(env_cfg,sim_params, args.physics_engine, args.device_type, args.device_id, args.headless)   #vec_gpu
        vec_env = VecTaskPython(env, rl_device)
    elif args.task == "ur5pickup":
        env = Ur5pickup(env_cfg,sim_params, args.physics_engine, args.device_type, args.device_id, args.headless)
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
                 is_testing=False,
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
                 is_testing=False,
                 print_log=Trself.vec_envue,
                 apply_reset=False,
                 asymmetric=False)
    elif args.algo == 'trpo':
        task = TRPO(vec_env,
                 train_cfg,
                 device=rl_device,
                 sampler='random',
                 log_dir='run',
                 is_testing=False,
                 print_log=True,
                 apply_reset=False,
                 asymmetric=False)
    elif args.algo == 'test':
        task = TEST(vec_env, 
                    device=rl_device)

    return task