#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=$1
# export CUDA_LAUNCH_BLOCKING=1

# python train_transe_model.py --dataset ${dataset_name}
# python train_agent.py --dataset ${dataset_name}
# python test_agent.py --dataset ${dataset_name} --run_path True --run_eval True
export PYTHONPATH="../"

# MODEL_TYPE=('cell_core')
# # MODEL_TYPE=('beauty_core')
# # MODEL_TYPE=('cd_core')

if [ $3 = "cell" ]
then
    MODEL_TYPE=('cell_core')
elif [ $3 = "bu" ]
then
    MODEL_TYPE=('beauty_core')
elif [ $3 = "cd" ]
then
    MODEL_TYPE=('cd_core')
elif [ $3 = "cl" ]
then
    MODEL_TYPE=('cloth_core')
elif [ $3 = "az" ]
then
    MODEL_TYPE=('amazon-book_20core')
elif [ $3 = "mv" ]
then
    MODEL_TYPE=('MovieLens-1M_core')
elif [ $3 = "la" ]
then
    MODEL_TYPE=('LAST_FM_20core')
elif [ $3 = "bc" ]
then
    MODEL_TYPE=('BOOKC_20core')
fi

user_o=0
h0_embbed=0
test_lstm_up=1
env_meta_path=0
reward_hybrid=0
batch_size=32
gpu=$1
p_hop=$2
embed_size=$4
n_memory=64
epochs=150
# lr=1.5e-4
lr=1.2e-4
sub_batch_size=1

# exp_name="em${embed_size}_mf_ep"
# use_men='state_history_no_emb'
# # use_men='ls_hitsta'

# # no_shu_qy_uam_800_15_500_lr_0.0001_bs_32_sb_1_ma_50_p12_em16_mf_ep_g_aiu_0_0_6000_qy_uam_800_15_500

# for dataset_name in ${MODEL_TYPE[@]}
# do

#     cmd="python3 ../train_mem/${train_file} --sub_batch_size ${sub_batch_size} --name ${exp_name} --env_meta_path ${env_meta_path} \
#       --lr ${lr} --embed_size ${embed_size} --gp_setting ${gp_setting} --epochs ${epochs} --n_memory ${n_memory} --user_o ${user_o} --gpu ${gpu} \
#        --reward_hybrid ${reward_hybrid} --batch_size ${batch_size} --h0_embbed ${h0_embbed} --p_hop ${p_hop} --use_men ${use_men} --dataset ${dataset_name}"
#     echo "Executing $cmd"
#     $cmd

#     cmd="python3 ../train_mem/${test_file} --name ${exp_name} --sub_batch_size ${sub_batch_size} --env_meta_path ${env_meta_path}\
#       --user_o ${user_o} --gpu ${gpu} --gp_setting ${gp_setting}  --h0_embbed ${h0_embbed} --use_men ${use_men} --dataset ${dataset_name} \
#       --reward_hybrid ${reward_hybrid} --batch_size ${batch_size} --lr ${lr} --p_hop ${p_hop} --embed_size ${embed_size} --n_memory ${n_memory} --run_path True --run_eval True --test_lstm_up ${test_lstm_up}"
#     echo "Executing $cmd"
#     $cmd

# done

# no_shu_qy_uam_100_15_500_lr_0.00012_bs_32_sb_1_ma_50_p12_db_em16_mf_ep_300_g_aiu_0_0_6000_qy_uam_100_15_500

# exp_name="mq_rn${reasoning_step}_h1${p_hop}_nmem${n_memory}_em${embed_size}_ep${epochs}"
# exp_name="db_em${embed_size}_mf_ep_${epochs}"
# use_men='state_history'
# use_men='ls_hitsta'

exp_name="db_em16_mf_ep_300"
use_men='state_history'

for dataset_name in ${MODEL_TYPE[@]}
do

    # cmd="python3 ../train_mem/${train_file} --sub_batch_size ${sub_batch_size} --name ${exp_name} --env_meta_path ${env_meta_path} \
    #   --lr ${lr} --embed_size ${embed_size} --gp_setting ${gp_setting} --epochs ${epochs} --n_memory ${n_memory} --user_o ${user_o} --gpu ${gpu} \
    #    --reward_hybrid ${reward_hybrid} --batch_size ${batch_size} --h0_embbed ${h0_embbed} --p_hop ${p_hop} --use_men ${use_men} --dataset ${dataset_name}"
    # echo "Executing $cmd"
    # $cmd

    cmd="python3 ../train_mem/${test_file} --name ${exp_name} --sub_batch_size ${sub_batch_size} --env_meta_path ${env_meta_path}\
      --user_o ${user_o} --gpu ${gpu} --gp_setting ${gp_setting}  --h0_embbed ${h0_embbed} --use_men ${use_men} --dataset ${dataset_name} \
      --reward_hybrid ${reward_hybrid} --batch_size ${batch_size} --lr ${lr} --p_hop ${p_hop} --embed_size ${embed_size} --n_memory ${n_memory} --run_path True --run_eval True --test_lstm_up ${test_lstm_up}"
    echo "Executing $cmd"
    $cmd

done


# exp_name="mq_rn${reasoning_step}_h1${p_hop}_nmem${n_memory}_em${embed_size}_ep${epochs}"
# exp_name="db_em${embed_size}_mf_ep_${epochs}"
# use_men='state_history_no_grad'
# # use_men='ls_hitsta'

# for dataset_name in ${MODEL_TYPE[@]}
# do

#     cmd="python3 ../train_mem/${train_file} --sub_batch_size ${sub_batch_size} --name ${exp_name} --env_meta_path ${env_meta_path} \
#       --lr ${lr} --embed_size ${embed_size} --gp_setting ${gp_setting} --epochs ${epochs} --n_memory ${n_memory} --user_o ${user_o} --gpu ${gpu} \
#        --reward_hybrid ${reward_hybrid} --batch_size ${batch_size} --h0_embbed ${h0_embbed} --p_hop ${p_hop} --use_men ${use_men} --dataset ${dataset_name}"
#     echo "Executing $cmd"
#     $cmd

#     cmd="python3 ../train_mem/${test_file} --name ${exp_name} --sub_batch_size ${sub_batch_size} --env_meta_path ${env_meta_path}\
#       --user_o ${user_o} --gpu ${gpu} --gp_setting ${gp_setting}  --h0_embbed ${h0_embbed} --use_men ${use_men} --dataset ${dataset_name} \
#       --reward_hybrid ${reward_hybrid} --batch_size ${batch_size} --lr ${lr} --p_hop ${p_hop} --embed_size ${embed_size} --n_memory ${n_memory} --run_path True --run_eval True --test_lstm_up ${test_lstm_up}"
#     echo "Executing $cmd"
#     $cmd

# done
