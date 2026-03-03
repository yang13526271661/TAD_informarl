#!/bin/bash

# Run the script
seed_max=1
n_agents=4
# graph_feat_types=("global" "global" "relative" "relative")
# cent_obs=("True" "False" "True" "False")
ep_lens=200

for seed in `seq ${seed_max}`;
do
# seed=`expr ${seed} + 1`
echo "seed: ${seed}"
# execute the script with different params
CUDA_VISIBLE_DEVICES='2' python  ../onpolicy/scripts/train_mpe.py \
--use_valuenorm --use_popart \
--project_name "GP_Graph_ABL" \
--env_name "GraphMPE" \
--algorithm_name "rmappo" \
--seed ${seed} \
--experiment_name "check" \
--scenario_name "graph_formation_4agts" \
--clip_param 0.15 --gamma 0.985 \
--hid_size 64 --layer_N 1 \
--num_target 0 --num_agents 4 --num_obstacle 4 --num_dynamic_obs 4 \
--gp_type "formation" \
--save_data "True" \
--reward_file_name "r_formation_4agts-01-1" \
--use_policy "False" \
--use_curriculum "True" \
--guide_cp 0.1 --cp 0.1 --js_ratio 0.8 \
--use_wandb "True" \
--n_training_threads 16 --n_rollout_threads 32 \
--use_lstm "True" \
--episode_length ${ep_lens} \
--num_env_steps 6000000 \
--data_chunk_length 20 \
--ppo_epoch 15 --use_ReLU --gain 0.01 --lr 2e-4 --critic_lr 2e-4 \
--user_name "finleygou" \
--use_cent_obs "False" \
--graph_feat_type "relative" \
--use_att_gnn "False" \
--split_batch "True" --max_batch_size 512 \
--auto_mini_batch_size "True" --target_mini_batch_size 512
done