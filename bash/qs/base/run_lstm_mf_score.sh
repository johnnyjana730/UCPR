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

load_pretrain_model=0
pretrained_st_epoch=0
epochs=150

batch_size=32
gpu=$1
p_hop=$2
embed_size=$4
n_memory=16
# epochs=100
# lr=2e-4
sub_batch_size=1

# reward_rh=no
reward_rh=hybrid

lr_step=(1.5e-4 1.3e-4 1e-4 2e-4)

for dataset_name in ${MODEL_TYPE[@]}
do
    for lr in ${lr_step[@]}
    do
        exp_name="nogv_lstm_em${embed_size}_mf"
        use_men='lstm_mf'

        cmd="python3 ../train_mem/${test_file} --name ${exp_name} --sub_batch_size ${sub_batch_size} --env_meta_path ${env_meta_path}\
          --user_o ${user_o} --gpu ${gpu} --gp_setting ${gp_setting} --h0_embbed ${h0_embbed} --reward_rh ${reward_rh} --use_men ${use_men} --dataset ${dataset_name} \
          --reward_hybrid ${reward_hybrid} --batch_size ${batch_size} --load_pretrain_model ${load_pretrain_model} --pretrained_st_epoch ${pretrained_st_epoch} --lr ${lr} --p_hop ${p_hop} --embed_size ${embed_size} --n_memory ${n_memory} --run_path True --run_eval True --test_lstm_up ${test_lstm_up}"
        echo "Executing $cmd"
        $cmd
    done
done
