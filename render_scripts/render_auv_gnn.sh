#!/bin/bash
export PYTHONPATH=../:$PYTHONPATH
export WANDB_MODE=offline

# ================= 1. 先配置你想查看的比赛 ================= #
TARGET_ITER=2
# EVAL_MODE='attacker'  
EVAL_MODE='defender' 

# ================= 2. 再将配置导出为环境变量 ================= #
export TRAIN_MODE=$EVAL_MODE
export CURRENT_ITER=$TARGET_ITER
export EXPERIMENT_NAME="Render_SelfPlay_Iter${TARGET_ITER}_${EVAL_MODE}"

echo ">>>>>>>> 正在准备渲染 第 ${TARGET_ITER} 轮的 ${EVAL_MODE} <<<<<<<<"

# ================= 3. 路径判断 ================= #
if [ "$EVAL_MODE" = "attacker" ]; then
    MODEL_PATH="../onpolicy/results/GraphMPE/graph_auv_adg/rmappo/SelfPlay_Iter${TARGET_ITER}_Attacker/models"
else
    MODEL_PATH="../onpolicy/results/GraphMPE/graph_auv_adg/rmappo/SelfPlay_Iter${TARGET_ITER}_Defender/models"
fi

# ================= 4. 运行评估脚本 ================= #
CUDA_VISIBLE_DEVICES='3' python ../onpolicy/scripts/eval_mpe.py \
    --use_valuenorm \
    --project_name "AUV_Graph_MADRL" \
    --env_name "GraphMPE" \
    --algorithm_name "rmappo" \
    --experiment_name "${EXPERIMENT_NAME}" \
    --scenario_name "graph_auv_adg" \
    --model_dir "${MODEL_PATH}" \
    --num_target 1 \
    --num_attacker 1 \
    --num_defender 2 \
    --n_training_threads 1 \
    --n_rollout_threads 1 \
    --episode_length 500 \
    --save_gifs True \
    --use_render True \
    --render_episodes 5 \
    --hidden_size 64 \
    --layer_N 2