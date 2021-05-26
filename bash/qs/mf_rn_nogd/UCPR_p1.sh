#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=$1
export PYTHONPATH="../"

if [ $2 = "cell" ]
then
    dataset_name='cell_core'
    model='UCPR'
    gp_setting="6_800_15_500_50"
    lr=1e-04
    lambda_num=2.0
    n_memory=64
    p_hop=2
    reasoning_step=2
    embed_size=16
elif [ $2 = "bu" ]
then
    dataset_name='beauty_core'
    model='UCPR'
    gp_setting="6_800_15_500_50"
    lr=1e-04
    lambda_num=0.5
    n_memory=64
    p_hop=2
    reasoning_step=2
    embed_size=16
elif [ $2 = "cl" ]
then
    dataset_name='cloth_core'
    model='UCPR'
    gp_setting="6_800_15_500_50"
    lr=1e-04
    lambda_num=0.3
    n_memory=32
    p_hop=2
    reasoning_step=2
    embed_size=16
fi

epochs=20
KGE_pretrained=1
kg_emb_grad=0

test_lstm_up=1
user_core_th_setting=1
load_pretrain_model=0

tri_wd_rm=1
tri_pro_rm=0

batch_size=1024

# exp_name=ld${lambda_num}_rn${reasoning_step}_h1${p_hop}_nmem${n_memory}_em${embed_size}
exp_name=wdrm${tri_wd_rm}

# cmd="python3 ../src/${train_file} --reasoning_step ${reasoning_step} --batch_size ${batch_size} --name ${exp_name}  \
#    --lr ${lr}  --embed_size ${embed_size} --n_memory ${n_memory} \
#    --load_pretrain_model ${load_pretrain_model} --tri_wd_rm ${tri_wd_rm} --tri_pro_rm ${tri_pro_rm}  --gp_setting ${gp_setting} --epochs ${epochs} --KGE_pretrained ${KGE_pretrained} \
#     --lambda_num ${lambda_num} --kg_emb_grad ${kg_emb_grad} --p_hop ${p_hop} --reasoning_step ${reasoning_step} --model ${model} --dataset ${dataset_name}"
# echo "Executing $cmd"
# $cmd

cmd="python3 ../src/${test_file} --name ${exp_name} --batch_size ${batch_size} \
  --gp_setting ${gp_setting}  --model lstm --dataset ${dataset_name} \
  --lambda_num ${lambda_num}  --kg_emb_grad ${kg_emb_grad}  --tri_wd_rm ${tri_wd_rm}
  --tri_pro_rm ${tri_pro_rm} --lr ${lr} --p_hop ${p_hop} --reasoning_step ${reasoning_step} --embed_size ${embed_size} --n_memory ${n_memory}  --test_lstm_up ${test_lstm_up}"
echo "Executing $cmd"
$cmd