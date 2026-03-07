#!/bin/sh
export PYTHONPATH=../:$PYTHONPATH
env="MPE"
scenario="graph_TAD_rand_2t1a1d"
num_landmarks=0
num_agents=1
num_target=2
num_attacker=1
num_defender=1
algo="rmappo"
exp="check"
seed_max=1
use_Relu=False
layer_N=2
hidden_size=32
save_data=True
save_gifs=True

echo "env is ${env}"
for seed in `seq ${seed_max}`
do
    CUDA_VISIBLE_DEVICES=0 python  ../onpolicy/scripts/eval_mpe.py --env_name ${env} --algorithm_name ${algo} \
    --experiment_name ${exp} --scenario_name ${scenario} --num_agents ${num_agents} --num_landmarks ${num_landmarks} --seed ${seed} \
    --use_Relu ${use_Relu} --layer_N ${layer_N} --hidden_size ${hidden_size} \
    --max_edge_dist 0.0 --gp_type "encirclement" \
    --graph_feat_type "relative" --use_att_gnn True --use_cent_obs True \
    --use_valuenorm True --use_popart True --use_curriculum True --use_policy False \
    --clip_param 0.15 --gamma 0.995 --gain 0.01 --lr 4e-4 --critic_lr 4e-4 \
    --use_lstm False --split_batch True --max_batch_size 512 --num_mini_batch 32 \
    --n_training_threads 1 --n_rollout_threads 1 --use_render True --episode_length 200 --render_episodes 5 \
    --num_attacker ${num_attacker} --num_target ${num_target} --num_defender ${num_defender} \
    --model_dir "/data/yangxiaodi_space/TAD-informarl/InforMARL-main/onpolicy/results/MPE/graph_TAD_rand_2t1a1d/rmappo/check/wandb/run-20260305_225453-kbtz71uh/files/" \
    --save_data ${save_data} --save_gifs ${save_gifs}
done
# run-20240906_105448-xu4omp79