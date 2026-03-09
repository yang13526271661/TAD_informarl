#!/bin/bash
export PYTHONPATH=../:$PYTHONPATH

# 确保调用的是 render_mpe.py (请根据你仓库的实际位置调整，可能是 ../onpolicy/scripts/render_mpe.py)
CUDA_VISIBLE_DEVICES='0' python ../onpolicy/scripts/eval_mpe.py \
    --use_valuenorm \
    --project_name "AUV_Graph_MADRL" \
    --env_name "GraphMPE" \
    --algorithm_name "rmappo" \
    --experiment_name "auv_gnn_attacker_training" \
    --scenario_name "graph_auv_adg" \
    --model_dir "/data/yangxiaodi_space/TAD-informarl/InforMARL-main/onpolicy/results/GraphMPE/graph_auv_adg/rmappo/auv_gnn_attacker_training/wandb/run-20260309_130621-r0yrj5y4/files/" \
    --num_target 1 \
    --num_attacker 1 \
    --num_defender 2 \
    --num_landmarks 0 \
    --n_training_threads 1 \
    --n_rollout_threads 1 \
    --episode_length 500 \
    --save_gifs "True" \
    --use_render "True" \
    --render_episodes 5 \
    --hidden_size 64 \
    --layer_N 2 \
    --use_att_gnn "True" \
    --use_cent_obs "True"