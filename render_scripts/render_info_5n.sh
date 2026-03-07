#!/bin/bash

logs_folder="out_informarl5"
mkdir -p $logs_folder
# Run the script
seed_max=1
n_agents=5
# graph_feat_types=("global" "global" "relative" "relative")
# cent_obs=("True" "False" "True" "False")
ep_lens=30
save_data=False
save_gifs="False"

for seed in `seq ${seed_max}`;
do
# seed=`expr ${seed} + 1`
echo "seed: ${seed}"
# execute the script with different params
CUDA_VISIBLE_DEVICES='2' python  ../onpolicy/scripts/eval_mpe.py --use_valuenorm --use_popart \
--project_name "GraphMPE" \
--env_name "GraphMPE" \
--algorithm_name "rmappo" \
--seed ${seed} \
--experiment_name "check" \
--scenario_name "navigation_graph" \
--use_wandb "False" \
--save_gifs ${save_gifs} \
--use_render "True" \
--num_agents=${n_agents} \
--collision_rew 5 \
--n_rollout_threads 1 \
--use_lstm "False" \
--episode_length ${ep_lens} \
--render_episodes 5 \
--ppo_epoch 15 --use_ReLU --gain 0.01 \
--user_name "finleygou" \
--use_cent_obs "False" \
--graph_feat_type "relative" \
--use_att_gnn "True" \
--model_dir "/data/goufandi_space/Projects/InforMARL/onpolicy/results/GraphMPE/navigation_graph/rmappo/check/wandb/run-20240919_194341-motpgj37/files/" \
--auto_mini_batch_size "True" --target_mini_batch_size 2048
done

# &> $logs_folder/out_${ep_lens}_${seed} \
# --num_mini_batch 64 \