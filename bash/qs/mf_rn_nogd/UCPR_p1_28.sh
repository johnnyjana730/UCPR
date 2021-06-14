#!/usr/bin/env 
# bash

export CUDA_VISIBLE_DEVICES=$1
export PYTHONPATH="../"

if [ $2 = "cell" ]
then
    dataset_name='cell_core'
    model='UCPR'
    gp_setting="6_800_15_500_50"
    lr=1.0e-04
    lambda_num=1.0
    n_memory=64
    p_hop=2
    reasoning_step=2
    embed_size=16
elif [ $2 = "bu" ]
then
    dataset_name='beauty_core'
    model='UCPR'
    gp_setting="6_800_15_500_50"
    lr=1.0e-04
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
    lr=1.0e-04
    lambda_num=0.3
    n_memory=32
    p_hop=2
    reasoning_step=2
    embed_size=16
fi

model='UCPR_28'
lr=1e-04

epochs=100
KGE_pretrained=1
kg_emb_grad=0
batch_size=1024

grad_check=1
tri_wd_rm=1
tri_pro_rm=0
load_pretrain_model=0
exp_name=${epochs}_nopre_rm_w${tri_wd_rm}_p${tri_pro_rm}_${lr}_gc${grad_check}

# exp_name=100_nopre_rm_w1_p0_3e-04

cmd="python3 ../src/${train_file} --reasoning_step ${reasoning_step} --batch_size ${batch_size} --name ${exp_name} \
    --lr ${lr}  --embed_size ${embed_size} --n_memory ${n_memory}  --tri_wd_rm ${tri_wd_rm} --tri_pro_rm ${tri_pro_rm} --KGE_pretrained ${KGE_pretrained} \
    --load_pretrain_model ${load_pretrain_model} --gp_setting ${gp_setting} --epochs ${epochs} \
    --lambda_num ${lambda_num} --kg_emb_grad ${kg_emb_grad} --grad_check ${grad_check}  --p_hop ${p_hop} --model ${model} --dataset ${dataset_name}"
echo "Executing $cmd"
$cmd

cmd="python3 ../src/${test_file} --name ${exp_name} --batch_size ${batch_size} \
    --gp_setting ${gp_setting} --model ${model} --dataset ${dataset_name}  --tri_wd_rm ${tri_wd_rm} --tri_pro_rm ${tri_pro_rm}  \
    --lambda_num ${lambda_num}  --kg_emb_grad ${kg_emb_grad} --grad_check ${grad_check}  --lr ${lr} --p_hop ${p_hop} --reasoning_step ${reasoning_step} --embed_size ${embed_size} --n_memory ${n_memory}"
echo "Executing $cmd"
$cmd
