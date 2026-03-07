#!/bin/bash
export PYTHONPATH=../:$PYTHONPATH
# Run the script
seed_max=1
# n_agents=3
# graph_feat_types=("global" "global" "relative" "relative")
# cent_obs=("True" "False" "True" "False")
ep_lens=200

for seed in `seq ${seed_max}`;
do
# seed=`expr ${seed} + 1`
echo "seed: ${seed}"
# execute the script with different params
CUDA_VISIBLE_DEVICES='1' python  ../onpolicy/scripts/train_mpe.py \
--use_valuenorm --use_popart \
--project_name "GP_Graph_NT" \
--env_name "MPE" \
--algorithm_name "rmappo" \
--seed ${seed} \
--experiment_name "check" \
--scenario_name "graph_TAD_rand_2t1a1d" \
--max_edge_dist 0.0 \
--clip_param 0.15 --gamma 0.995 \
--hidden_size 32 --layer_N 2 \
--num_target 2 --num_agents 1 --num_attacker 1 --num_defender 1 --num_landmarks 0 \
--gp_type "encirclement" \
--save_data "True" \
--use_train_render "True" \
--reward_file_name "reward1_cmp1" \
--use_policy "False" \
--use_curriculum "True" \
--max_grad_norm 10.0 \
--use_wandb "True" \
--n_training_threads 16 --n_rollout_threads 32 \
--use_lstm "False" \
--episode_length ${ep_lens} \
--num_env_steps 6000000 \
--data_chunk_length 20 \
--ppo_epoch 15 --use_ReLU False --gain 0.01 --lr 4e-4 --critic_lr 4e-4 \
--user_name "2043778278-lanzhou-university" \
--use_cent_obs "True" \
--graph_feat_type "relative" \
--use_att_gnn "True" \
--split_batch "True" --max_batch_size 512 \
--num_mini_batch 32 \
--auto_mini_batch_size "False" --target_mini_batch_size 512
done