#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=$1
export PYTHONPATH="../"

if [ $2 = "cell" ]
then
    dataset_name='cell_core'
    gp_setting="6_800_15_500_50"
    lr=1.0e-04
    lambda_num=2.0
    n_memory=64
    p_hop=2
    reasoning_step=2
    embed_size=16
    kg_emb_grad=0
    epochs=100
elif [ $2 = "bu" ]
then
    dataset_name='beauty_core'
    gp_setting="6_800_15_500_50"
    lr=1.0e-04
    lambda_num=0.5
    n_memory=64
    p_hop=2
    reasoning_step=2
    embed_size=16
    kg_emb_grad=0
    epochs=100
elif [ $2 = "cl" ]
then
    dataset_name='cloth_core'
    gp_setting="6_800_15_500_50"
    lr=1.0e-04
    lambda_num=0.3
    n_memory=32
    p_hop=2
    reasoning_step=2
    embed_size=16
    kg_emb_grad=0
    epochs=100
elif [ $2 = "az" ]
then
    dataset_name='amazon-book_20core'
    gp_setting="6000_800_15_500_50"
    lr=2.0e-04
    lambda_num=2.0
    n_memory=64
    p_hop=2
    reasoning_step=4
    embed_size=32
    kg_emb_grad=1
    epochs=400
elif [ $2 = "mv" ]
then
    dataset_name='MovieLens-1M_core'
    gp_setting="6000_800_15_500_50"
    lr=2.0e-04
    lambda_num=0.2
    n_memory=64
    p_hop=2
    reasoning_step=3
    embed_size=32
    kg_emb_grad=1
    epochs=400
fi

batch_size=1024
KGE_pretrained=1
env_old=0
grad_check=1
load_pretrain_model=0
save_pretrain_model=0

exp_name=lm_${epochs}_${lr}_gc${grad_check}

cmd="python3 ../src/${train_file} --reasoning_step ${reasoning_step} --batch_size ${batch_size} --name ${exp_name}  \
   --lr ${lr}  --embed_size ${embed_size} --n_memory ${n_memory} \
   --load_pretrain_model ${load_pretrain_model} --epochs ${epochs} --KGE_pretrained ${KGE_pretrained} \
    --lambda_num ${lambda_num} --env_old ${env_old} --grad_check ${grad_check} --kg_emb_grad ${kg_emb_grad} --p_hop ${p_hop} --reasoning_step ${reasoning_step} --model lstm --dataset ${dataset_name}"
echo "Executing $cmd"
$cmd

cmd="python3 ../src/${test_file} --name ${exp_name} --batch_size ${batch_size} \
  --model lstm --dataset ${dataset_name} --save_pretrain_model ${save_pretrain_model} \
  --lambda_num ${lambda_num} --env_old ${env_old}  --grad_check ${grad_check}  --kg_emb_grad ${kg_emb_grad}  --lr ${lr} --p_hop ${p_hop} --reasoning_step ${reasoning_step} --embed_size ${embed_size} --n_memory ${n_memory}"
echo "Executing $cmd"
$cmd