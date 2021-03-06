#!/usr/bin/env 

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


epochs=100
KGE_pretrained=1
kg_emb_grad=0

batch_size=1024

load_pretrain_model=0
save_pretrain_model=1

exp_name=pre_lm_${epochs}_${lr}_svemb_${embed_size}

cmd="python3 ../src/${train_file} --reasoning_step ${reasoning_step} --batch_size ${batch_size} --name ${exp_name}  \
   --lr ${lr}  --embed_size ${embed_size} --n_memory ${n_memory} \
   --load_pretrain_model ${load_pretrain_model}  --gp_setting ${gp_setting} --epochs ${epochs} --KGE_pretrained ${KGE_pretrained} \
    --lambda_num ${lambda_num} --kg_emb_grad ${kg_emb_grad} --p_hop ${p_hop} --reasoning_step ${reasoning_step} --model lstm --dataset ${dataset_name}"
echo "Executing $cmd"
$cmd

cmd="python3 ../src/${test_file} --name ${exp_name} --batch_size ${batch_size} \
  --gp_setting ${gp_setting}  --model lstm --dataset ${dataset_name} --save_pretrain_model ${save_pretrain_model} \
  --lambda_num ${lambda_num}  --kg_emb_grad ${kg_emb_grad}  --lr ${lr} --p_hop ${p_hop} --reasoning_step ${reasoning_step} --embed_size ${embed_size} --n_memory ${n_memory}"
echo "Executing $cmd"
$cmd


tri_wd_rm=0
tri_pro_rm=0
load_pretrain_model=1
exp_name=pre_${epochs}_grad_rm_w${tri_wd_rm}_p${tri_pro_rm}_${lr}

cmd="python3 ../src/${train_file} --reasoning_step ${reasoning_step} --batch_size ${batch_size} --name ${exp_name} \
    --lr ${lr}  --embed_size ${embed_size} --n_memory ${n_memory}  --tri_wd_rm ${tri_wd_rm} --tri_pro_rm ${tri_pro_rm} --KGE_pretrained ${KGE_pretrained} \
    --load_pretrain_model ${load_pretrain_model} --gp_setting ${gp_setting} --epochs ${epochs} \
    --lambda_num ${lambda_num} --kg_emb_grad ${kg_emb_grad}  --p_hop ${p_hop} --model ${model} --dataset ${dataset_name}"
echo "Executing $cmd"
$cmd

cmd="python3 ../src/${test_file} --name ${exp_name} --batch_size ${batch_size} \
    --gp_setting ${gp_setting} --model ${model} --dataset ${dataset_name}  --tri_wd_rm ${tri_wd_rm} --tri_pro_rm ${tri_pro_rm}  \
    --lambda_num ${lambda_num}  --kg_emb_grad ${kg_emb_grad} --lr ${lr} --p_hop ${p_hop} --reasoning_step ${reasoning_step} --embed_size ${embed_size} --n_memory ${n_memory}"
echo "Executing $cmd"
$cmd

tri_wd_rm=1
tri_pro_rm=0
load_pretrain_model=1
exp_name=pre_${epochs}_grad_rm_w${tri_wd_rm}_p${tri_pro_rm}_${lr}

cmd="python3 ../src/${train_file} --reasoning_step ${reasoning_step} --batch_size ${batch_size} --name ${exp_name} \
    --lr ${lr}  --embed_size ${embed_size} --n_memory ${n_memory}  --tri_wd_rm ${tri_wd_rm} --tri_pro_rm ${tri_pro_rm} --KGE_pretrained ${KGE_pretrained} \
    --load_pretrain_model ${load_pretrain_model} --gp_setting ${gp_setting} --epochs ${epochs} \
    --lambda_num ${lambda_num} --kg_emb_grad ${kg_emb_grad}  --p_hop ${p_hop} --model ${model} --dataset ${dataset_name}"
echo "Executing $cmd"
$cmd

cmd="python3 ../src/${test_file} --name ${exp_name} --batch_size ${batch_size} \
    --gp_setting ${gp_setting} --model ${model} --dataset ${dataset_name}  --tri_wd_rm ${tri_wd_rm} --tri_pro_rm ${tri_pro_rm}  \
    --lambda_num ${lambda_num}  --kg_emb_grad ${kg_emb_grad} --lr ${lr} --p_hop ${p_hop} --reasoning_step ${reasoning_step} --embed_size ${embed_size} --n_memory ${n_memory}"
echo "Executing $cmd"
$cmd


