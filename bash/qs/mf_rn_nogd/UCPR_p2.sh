#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=$1
export PYTHONPATH="../"

if [ $2 = "az" ]
then
    dataset_name='amazon-book_20core'
    model='UCPR'
    gp_setting="6000_800_15_500_50"
    lr=1e-04
    lambda_num=2.0
    n_memory=64
    p_hop=2
    reasoning_step=4
    embed_size=32
elif [ $2 = "mv" ]
then
    dataset_name='MovieLens-1M_core'
    model='UCPR'
    gp_setting="6000_800_15_500_50"
    lr=1e-04
    lambda_num=0.2
    n_memory=64
    p_hop=2
    reasoning_step=3
    embed_size=32
fi

epochs=400
KGE_pretrained=1
kg_emb_grad=1

load_pretrain_model=0
save_pretrain_model=1
batch_size=32

exp_name=ld${lambda_num}_rn${reasoning_step}_h1${p_hop}_nmem${n_memory}_em${embed_size}

cmd="python3 ../src/${train_file} --reasoning_step ${reasoning_step} --batch_size ${batch_size} --name ${exp_name} \
   --lr ${lr} --embed_size ${embed_size} --n_memory ${n_memory}  \
   --load_pretrain_model ${load_pretrain_model}  --gp_setting ${gp_setting} --epochs ${epochs} --KGE_pretrained ${KGE_pretrained} \
    --lambda_num ${lambda_num} --kg_emb_grad ${kg_emb_grad} --p_hop ${p_hop} --reasoning_step ${reasoning_step} --model lstm --dataset ${dataset_name}"
echo "Executing $cmd"
$cmd

cmd="python3 ../src/${test_file} --name ${exp_name} --batch_size ${batch_size}\
  --gp_setting ${gp_setting}  --model lstm --dataset ${dataset_name} --save_pretrain_model ${save_pretrain_model}  \
  --lambda_num ${lambda_num}  --kg_emb_grad ${kg_emb_grad} --lr ${lr} --p_hop ${p_hop} --reasoning_step ${reasoning_step} --embed_size ${embed_size} --n_memory ${n_memory}"
echo "Executing $cmd"
$cmd

load_pretrain_model=1

cmd="python3 ../src/${train_file} --reasoning_step ${reasoning_step} --batch_size ${batch_size} --name ${exp_name} \
   --lr ${lr}  --embed_size ${embed_size} --n_memory ${n_memory} --KGE_pretrained ${KGE_pretrained} \
   --load_pretrain_model ${load_pretrain_model} --gp_setting ${gp_setting} --epochs ${epochs} \
    --lambda_num ${lambda_num} --kg_emb_grad ${kg_emb_grad}  --p_hop ${p_hop} --reasoning_step ${reasoning_step} --model ${model} --dataset ${dataset_name}"
echo "Executing $cmd"
$cmd

cmd="python3 ../src/${test_file} --name ${exp_name} --batch_size ${batch_size}\
    --gp_setting ${gp_setting} --model ${model} --dataset ${dataset_name} \
    --lambda_num ${lambda_num}  --kg_emb_grad ${kg_emb_grad} --lr ${lr} --p_hop ${p_hop} --reasoning_step ${reasoning_step} --embed_size ${embed_size} --n_memory ${n_memory}"
echo "Executing $cmd"
$cmd

