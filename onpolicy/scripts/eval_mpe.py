#!/usr/bin/env python
import argparse
import sys
import os
import torch
import numpy as np
from pathlib import Path

sys.path.append(os.path.abspath(os.getcwd()))

from onpolicy.config import get_config, graph_config
from multiagent.MPE_env import MPEEnv, GraphMPEEnv
from onpolicy.envs.env_wrappers import SubprocVecEnv, DummyVecEnv, GraphSubprocVecEnv, GraphDummyVecEnv
from onpolicy import global_var as glv

glv._init()
glv.set_value('CL_ratio', 1.0)

def make_render_env(all_args):
    def get_env_fn(rank):
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

def main(args):
    parser = get_config()
    all_args = parser.parse_known_args(args)[0]

    # ================= 核心修复：加载 Graph 网络配置 ================= #
    if all_args.env_name == "GraphMPE":
        all_args, parser = graph_config(args, parser)
    # ============================================================== #

    if all_args.algorithm_name == "rmappo":
        assert all_args.use_recurrent_policy or all_args.use_naive_recurrent_policy, "check recurrent policy!"
    elif all_args.algorithm_name == "mappo":
        assert all_args.use_recurrent_policy == False and all_args.use_naive_recurrent_policy == False, "check recurrent policy!"
    else:
        raise NotImplementedError

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

    run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0] + "/results") / all_args.env_name / all_args.scenario_name / all_args.algorithm_name / all_args.experiment_name
    
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # env init
    envs = make_render_env(all_args)

    # ================= 同步底层核心维度 (支持动态切换评估主体) ================= #
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
        "eval_envs": envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir
    }

    if all_args.share_policy:
        if all_args.env_name == "GraphMPE":
            from onpolicy.runner.shared.graph_mpe_runner import GMPERunner as Runner
        else:
            from onpolicy.runner.shared.mpe_runner import MPERunner as Runner
    else:
        raise NotImplementedError

    runner = Runner(config)
    # ================= 核心保障：测试模型加载确认 ================= #
    print(f"\n" + "="*60)
    print(f"[DEBUG] 渲染模式启动！目标路径: {all_args.model_dir}")
    if hasattr(all_args, 'model_dir') and all_args.model_dir is not None and all_args.model_dir != "":
        target_pt = os.path.join(all_args.model_dir, 'actor.pt')
        if os.path.exists(target_pt):
            print(f"[SUCCESS] 发现主脑模型: {target_pt}")
            print(f"[SUCCESS] 底层 base_runner 已将其加载入 GPU/CPU，准备绘图！")
        else:
            print(f"[ERROR] 致命错误：找不到 actor.pt！你现在看到的将是乱动的随机策略！")
    print("="*60 + "\n")
    # ============================================================== #
    # 执行渲染
    runner.render()
    
    envs.close()

if __name__ == "__main__":
    main(sys.argv[1:])