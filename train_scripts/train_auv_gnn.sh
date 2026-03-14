#!/bin/bash
export PYTHONPATH=../:$PYTHONPATH
export WANDB_MODE=offline

TOTAL_ITERATIONS=10       
STEPS_PER_PHASE=3000000    

for (( iter=1; iter<=$TOTAL_ITERATIONS; iter++ ))
do
    echo "================================================================="
    echo "                 开始第 ${iter} 轮自我对弈交替训练                 "
    echo "================================================================="

    # ---------------- 阶段 A：训练防守者 ----------------

    echo ">>>> [阶段 A] 正在训练防守者 (Defender)..."
    export TRAIN_MODE='defender'
    export CURRENT_ITER=$iter

    DEFENDER_ARGS=(
        --use_valuenorm --use_popart
        --project_name "AUV_Graph_MADRL"
        --user_name "2043778278-lanzhou-university"
        --env_name "GraphMPE"
        --algorithm_name "rmappo"
        --seed 1
        --experiment_name "SelfPlay_Iter${iter}_Defender"
        --scenario_name "graph_auv_adg"
        --num_target 1 --num_attacker 1 --num_defender 2
        --n_training_threads 1 --n_rollout_threads 32 --num_mini_batch 1
        --episode_length 500
        --num_env_steps ${STEPS_PER_PHASE}
        --ppo_epoch 15
        --hidden_size 64
        --layer_N 2
        --save_data "True"
        --reward_file_name "auv_gnn_rewards_Iter${iter}_Defender"
        --use_train_render "False"
        --use_policy "False"
        --use_curriculum "True"
        --use_wandb "False"
        --save_interval 1   # <==== 强制每轮保存
    )

    if [ $iter -gt 1 ]; then
        PREV_ITER=$((iter-1))
        CUR_DIR=$(pwd)
        MODEL_PATH="${CUR_DIR}/../onpolicy/results/GraphMPE/graph_auv_adg/rmappo/SelfPlay_Iter${PREV_ITER}_Defender/models"
        
        if [ -f "${MODEL_PATH}/actor.pt" ]; then
            DEFENDER_ARGS+=("--model_dir" "${MODEL_PATH}")
            echo "[DEBUG] 找到上一轮防守者模型，准备继承进化。"
        else
            echo "[WARNING] 找不到 ${MODEL_PATH}/actor.pt，被迫从头训练！"
        fi
    fi

    CUDA_VISIBLE_DEVICES='3' python ../onpolicy/scripts/train_mpe.py "${DEFENDER_ARGS[@]}"

    export TRAIN_MODE='attacker'    
        # 使用数组管理参数，整洁且防报错
    ATTACKER_ARGS=(
        --use_valuenorm --use_popart
        --project_name "AUV_Graph_MADRL"
        --user_name "2043778278-lanzhou-university"
        --env_name "GraphMPE"
        --algorithm_name "rmappo"
        --seed 1
        --experiment_name "SelfPlay_Iter${iter}_Attacker"
        --scenario_name "graph_auv_adg"
        --num_target 1 --num_attacker 1 --num_defender 2
        --n_training_threads 1 --n_rollout_threads 32 --num_mini_batch 1
        --episode_length 500
        --num_env_steps ${STEPS_PER_PHASE}
        --ppo_epoch 15
        --hidden_size 64
        --layer_N 2
        --save_data "True"
        --reward_file_name "auv_gnn_rewards_Iter${iter}_Attacker"
        --use_train_render "False"
        --use_policy "False"
        --use_curriculum "True"
        --use_wandb "False"
        --save_interval 1   # <==== 第一道防线：强制每轮保存，防止丢失
    )

    if [ $iter -gt 1 ]; then
        PREV_ITER=$((iter-1))
        CUR_DIR=$(pwd)
        MODEL_PATH="${CUR_DIR}/../onpolicy/results/GraphMPE/graph_auv_adg/rmappo/SelfPlay_Iter${PREV_ITER}_Attacker/models"
        
        # <==== 第二道防线：文件物理存在性绝对校验
        if [ -f "${MODEL_PATH}/actor.pt" ]; then
            ATTACKER_ARGS+=("--model_dir" "${MODEL_PATH}")
            echo "[DEBUG] 找到上一轮攻击者模型，准备继承进化。"
        else
            echo "[WARNING] 找不到 ${MODEL_PATH}/actor.pt，被迫从头训练！"
        fi
    fi

    CUDA_VISIBLE_DEVICES='2' python ../onpolicy/scripts/train_mpe.py "${ATTACKER_ARGS[@]}"

done
