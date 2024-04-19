import warnings
warnings.filterwarnings('ignore')   #if you don't want to check

from utils.config import set_np_formatting, set_seed, get_args, parse_sim_params, load_cfg
from utils.parse_task import parse_task
from utils.runner import Runner


# task 
# ur5cabinet_door
# ur5cabinet
# ur5pickup
# ur5pickandplace

def train():
    print("Algorithm: ", args.algo)

    if args.algo in ["ppo", "sac", "td3","test","rla","vts","vta","vtp"]: 
        task = parse_task(args, env_cfg, train_cfg, sim_params)
        runner = Runner(task, train_cfg)
        if args.test == True:
            runner.eval()
        else:
            runner.run()

    else:
        print("Unrecognized algorithm!")


if __name__ == '__main__':
    set_np_formatting()
    args = get_args()
    env_cfg, train_cfg = load_cfg(args)
    sim_params = parse_sim_params(args, env_cfg)
    set_seed(train_cfg.get("seed", -1), train_cfg.get("torch_deterministic", False))
    train()

