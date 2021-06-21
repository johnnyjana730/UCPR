#!/usr/bin/env 

export CUDA_VISIBLE_DEVICES=$1
export PYTHONPATH="../"

if [ $2 = "az" ]
then
    dataset_name='amazon-book_20core'
    model='UCPR'
    gp_setting="6000_800_15_500_50"
    lr=2e-04
    lambda_num=0.3
    n_memory=64
    p_hop=2
    reasoning_step=3
    embed_size=32
    kg_emb_grad=0
    batch_size=32
    epochs=300
elif [ $2 = "mv" ]
then
    dataset_name='MovieLens-1M_core'
    model='UCPR'
    gp_setting="6000_800_15_500_50"
    lr=2e-04
    lambda_num=2.0
    n_memory=64
    p_hop=2
    reasoning_step=4
    embed_size=32
    kg_emb_grad=0
    batch_size=1024
    epochs=800
fi



KGE_pretrained=1

load_pretrain_model=0
save_pretrain_model=1

exp_name=3_ba${batch_size}lstm_${epochs}${lr}_sveb${embed_size}_kge${kg_emb_grad}

cmd="python3 ../src/${train_file} --reasoning_step ${reasoning_step} --batch_size ${batch_size} --name ${exp_name} \
   --lr ${lr} --embed_size ${embed_size} --n_memory ${n_memory}  \
   --load_pretrain_model ${load_pretrain_model}  --gp_setting ${gp_setting} --epochs ${epochs} --KGE_pretrained ${KGE_pretrained} \
    --lambda_num ${lambda_num} --kg_emb_grad ${kg_emb_grad} --p_hop ${p_hop} --reasoning_step ${reasoning_step} --model lstm --dataset ${dataset_name}"
$cmd

cmd="python3 ../src/${test_file} --name ${exp_name} --batch_size ${batch_size} \
  --gp_setting ${gp_setting}  --model lstm --dataset ${dataset_name} --save_pretrain_model ${save_pretrain_model}  \
  --lambda_num ${lambda_num} --kg_emb_grad ${kg_emb_grad} --lr ${lr} --p_hop ${p_hop} --reasoning_step ${reasoning_step} --embed_size ${embed_size} --n_memory ${n_memory} "
$cmd


epochs=200

reasoning_step=5
batch_size=32

exp_name=3_ba${batch_size}_${lambda_num}_rn${reasoning_step}_${lr}

load_pretrain_model=1

cmd="python3 ../src/${train_file} --reasoning_step ${reasoning_step} --batch_size ${batch_size} --name ${exp_name} \
   --lr ${lr}  --embed_size ${embed_size} --n_memory ${n_memory} --KGE_pretrained ${KGE_pretrained} \
   --load_pretrain_model ${load_pretrain_model} --gp_setting ${gp_setting} --epochs ${epochs} \
    --lambda_num ${lambda_num}  --kg_emb_grad ${kg_emb_grad}  --p_hop ${p_hop} --reasoning_step ${reasoning_step} --model ${model} --dataset ${dataset_name} "
$cmd


cmd="python3 ../src/${test_file} --name ${exp_name} --batch_size ${batch_size} \
    --gp_setting ${gp_setting} --model ${model} --dataset ${dataset_name} \
    --lambda_num ${lambda_num}  --kg_emb_grad ${kg_emb_grad} --lr ${lr} --p_hop ${p_hop} --reasoning_step ${reasoning_step} --embed_size ${embed_size} --n_memory ${n_memory} "
$cmd


epochs=200

reasoning_step=4
batch_size=32

exp_name=3_ba${batch_size}_${lambda_num}_rn${reasoning_step}_${lr}

load_pretrain_model=1

cmd="python3 ../src/${train_file} --reasoning_step ${reasoning_step} --batch_size ${batch_size} --name ${exp_name} \
   --lr ${lr}  --embed_size ${embed_size} --n_memory ${n_memory} --KGE_pretrained ${KGE_pretrained} \
   --load_pretrain_model ${load_pretrain_model} --gp_setting ${gp_setting} --epochs ${epochs} \
    --lambda_num ${lambda_num} --kg_emb_grad ${kg_emb_grad}  --p_hop ${p_hop} --reasoning_step ${reasoning_step} --model ${model} --dataset ${dataset_name} "
$cmd


cmd="python3 ../src/${test_file} --name ${exp_name} --batch_size ${batch_size} \
    --gp_setting ${gp_setting} --model ${model} --dataset ${dataset_name} \
    --lambda_num ${lambda_num} --kg_emb_grad ${kg_emb_grad} --lr ${lr} --p_hop ${p_hop} --reasoning_step ${reasoning_step} --embed_size ${embed_size} --n_memory ${n_memory} "
$cmd
