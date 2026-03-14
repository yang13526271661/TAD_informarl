#!/usr/bin/env python
import argparse
from distutils.util import strtobool
import wandb
import socket
import setproctitle
import numpy as np
from pathlib import Path
import torch
from onpolicy import global_var as glv
import os, sys
import warnings
# 忽略掉烦人的 FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.append(os.path.abspath(os.getcwd()))

from utils.utils import print_args, print_box, connected_to_internet
# ================= 核心修复 ================= #
from onpolicy.config import get_config, graph_config  # 这里将 graph_config 导入进来
# ============================================ #
from multiagent.MPE_env import MPEEnv, GraphMPEEnv
from onpolicy.envs.env_wrappers import (
    SubprocVecEnv,
    DummyVecEnv,
    GraphSubprocVecEnv,
    GraphDummyVecEnv,
)

"""Train script for MPEs."""

glv._init()
glv.set_value('CL_ratio',0.0)

def make_train_env(all_args: argparse.Namespace):
    def get_env_fn(rank: int):
        def init_env():
            if all_args.env_name == "MPE":
                env = MPEEnv(all_args)
            elif all_args.env_name == "GraphMPE":
                env = GraphMPEEnv(all_args)
            else:
                print(f"Can not support the {all_args.env_name} environment")
                raise NotImplementedError
            env.seed(all_args.seed + rank * 1000)
            return env

        return init_env

    if all_args.n_rollout_threads == 1:
        if all_args.env_name == "GraphMPE":
            return GraphDummyVecEnv([get_env_fn(0)])
        return DummyVecEnv([get_env_fn(0)])
    else:
        if all_args.env_name == "GraphMPE":
            return GraphSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])
        return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])

def make_eval_env(all_args: argparse.Namespace):
    def get_env_fn(rank: int):
        def init_env():
            if all_args.env_name == "MPE":
                env = MPEEnv(all_args)
            elif all_args.env_name == "GraphMPE":
                env = GraphMPEEnv(all_args)
            else:
                print(f"Can not support the {all_args.env_name} environment")
                raise NotImplementedError
            env.seed(all_args.seed * 50000 + rank * 10000)
            return env

        return init_env

    if all_args.n_eval_rollout_threads == 1:
        if all_args.env_name == "GraphMPE":
            return GraphDummyVecEnv([get_env_fn(0)])
        return DummyVecEnv([get_env_fn(0)])
    else:
        if all_args.env_name == "GraphMPE":
            return GraphSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])
        return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])


def main(args):
    parser = get_config()
    all_args = parser.parse_known_args(args)[0]

    # ================= 核心修复：从你原本的 config.py 完美加载所有 GNN 参数 ================= #
    if all_args.env_name == "GraphMPE":
        all_args, parser = graph_config(args, parser)
    # ======================================================================================== #

    if all_args.algorithm_name == "rmappo":
        assert all_args.use_recurrent_policy or all_args.use_naive_recurrent_policy, "check recurrent policy!"
    elif all_args.algorithm_name == "mappo":
        assert all_args.use_recurrent_policy == False and all_args.use_naive_recurrent_policy == False, "check recurrent policy!"
    else:
        raise NotImplementedError

    # cuda
    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    # ================= 核心修复：强制固定日志与模型路径，禁止生成 run 文件夹 ================= #
    run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0] + "/results") / all_args.env_name / all_args.scenario_name / all_args.algorithm_name / all_args.experiment_name
    if not run_dir.exists():
        os.makedirs(str(run_dir))
    # ===================================================================================== #

    setproctitle.setproctitle(str(all_args.algorithm_name) + "-" + str(all_args.env_name) + "-" + str(all_args.experiment_name) + "@" + str(all_args.user_name))

    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # wandb init
    if all_args.use_wandb:
        run = wandb.init(config=all_args,
                         project=all_args.project_name,
                         entity=all_args.user_name,
                         notes=socket.gethostname(),
                         name=str(all_args.algorithm_name) + "_" +
                         str(all_args.experiment_name) +
                         "_seed" + str(all_args.seed),
                         group=all_args.scenario_name,
                         dir=str(run_dir),
                         job_type="training",
                         reinit=True)

    # env init
    envs = make_train_env(all_args)
    eval_envs = make_eval_env(all_args) if all_args.use_eval else None
    
    # ================= 同步底层核心维度 (支持动态切换训练主体) ================= #
    
    current_train_mode = os.environ.get('TRAIN_MODE', 'attacker')
    
    if current_train_mode == 'attacker':
        num_agents = all_args.num_attacker
    else:
        num_agents = all_args.num_defender
    all_args.num_agents = num_agents
    # ======================================================================= #

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir,
    }

    # run experiments
    if all_args.share_policy:
        if all_args.env_name == "GraphMPE":
            from onpolicy.runner.shared.graph_mpe_runner import GMPERunner as Runner
        else:
            from onpolicy.runner.shared.mpe_runner import MPERunner as Runner
    else:
        if all_args.env_name == "GraphMPE":
            raise NotImplementedError("Graph policy currently does not support separated runner out of the box.")
        from onpolicy.runner.separated.mpe_runner import MPERunner as Runner

    runner = Runner(config)
    print("model_dir:", all_args.model_dir)
    # ================= 核心修复：终极模型继承机制与路径追踪 ================= #
    if hasattr(all_args, 'model_dir') and all_args.model_dir is not None and all_args.model_dir != "":
        model_path = str(all_args.model_dir)
        print(f"\n" + "="*60)
        print(f"[DEBUG] 接收到模型加载指令！")
        print(f"[DEBUG] 正在核实绝对路径: {os.path.abspath(model_path)}")
        
        if os.path.exists(model_path):
            try:
                actor_state = torch.load(os.path.join(model_path, 'actor.pt'), map_location=device)
                runner.trainer.policy.actor.load_state_dict(actor_state)
                critic_state = torch.load(os.path.join(model_path, 'critic.pt'), map_location=device)
                runner.trainer.policy.critic.load_state_dict(critic_state)
                print(f"[SUCCESS] 历史模型继承成功！站在巨人的肩膀上继续进化！")
            except Exception as e:
                print(f"[ERROR] 路径存在，但 PyTorch 模型加载失败: {e}")
        else:
            print(f"[WARNING] 哎呀！文件夹不存在，被迫退回随机初始化！")
        print("="*60 + "\n")
    # ========================================================================= #

    runner.run()

    # post process
    envs.close()
    if all_args.use_eval and eval_envs is not envs:
        eval_envs.close()
    if all_args.use_wandb:
        wandb.finish()

if __name__ == "__main__":
    main(sys.argv[1:])