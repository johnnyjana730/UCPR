#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=$1
export PYTHONPATH="../"

if [ $2 = "az" ]
then
    dataset_name='amazon-book_20core'
    model='UCPR'
    gp_setting="6000_800_15_500_50"
    lr=2e-04

    # no_shu_qy_uam_800_15_500_lr_8e-05_bs_32_sb_1_ma_50_p12_ald_areg2e-6_lda0.25rn3h2m64e32_lm0_g_aiu_0_0_6000_qy_uam_800_15_500

    lambda_num=1.0
    n_memory=32
    p_hop=2
    reasoning_step=3
    embed_size=32
    kg_emb_grad=1
    batch_size=32
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
fi

epochs=1500
KGE_pretrained=1

load_pretrain_model=0
save_pretrain_model=1

# lstm_400_2e-04_save_emb_32_g_aiu_0_0_6000

# exp_name=ba${batch_size}lstm_${epochs}_${lr}_save_emb_${embed_size}_kge${kg_emb_grad}

# exp_name=dbkge_ba${batch_size}lstm_${epochs}${lr}_sveb${embed_size}kge${kg_emb_grad}

# cmd="python3 ../src/${train_file} --reasoning_step ${reasoning_step} --batch_size ${batch_size} --name ${exp_name} \
#    --lr ${lr} --embed_size ${embed_size} --n_memory ${n_memory}  \
#    --load_pretrain_model ${load_pretrain_model}  --gp_setting ${gp_setting} --epochs ${epochs} --KGE_pretrained ${KGE_pretrained} \
#     --lambda_num ${lambda_num} --kg_emb_grad ${kg_emb_grad} --p_hop ${p_hop} --reasoning_step ${reasoning_step} --model lstm --dataset ${dataset_name}"
# echo "Executing $cmd"
# $cmd

# cmd="python3 ../src/${test_file} --name ${exp_name} --batch_size ${batch_size} \
#   --gp_setting ${gp_setting}  --model lstm --dataset ${dataset_name} --save_pretrain_model ${save_pretrain_model}  \
#   --lambda_num ${lambda_num}  --kg_emb_grad ${kg_emb_grad} --lr ${lr} --p_hop ${p_hop} --reasoning_step ${reasoning_step} --embed_size ${embed_size} --n_memory ${n_memory}"
# echo "Executing $cmd"
# $cmd

# lstm_600_2e-04_save_emb_32_kge0_g_aiu_0_0_6000

epochs=150

lr=1.2e-04
# exp_name=dbkge_ba${batch_size}${lambda_num}_rn${reasoning_step}_h1${p_hop}_nmem${n_memory}_${lr}_em${embed_size}_kge${kg_emb_grad}_10101
# exp_name=dbkge_ba${batch_size}${lambda_num}_rn${reasoning_step}_h1${p_hop}_nmem${n_memory}_${lr}_em${embed_size}_kge${kg_emb_grad}
# exp_name=dbkge_kga_ba${batch_size}${lambda_num}_rn${reasoning_step}_h1${p_hop}_nmem${n_memory}_${lr}_em${embed_size}_kge${kg_emb_grad}

# exp_name=dbkge_ba320.25_rn2_h12_nmem32_1.0e-04_em32_kge1
# dbkge_ba320.1_rn2_h12_nmem32_0.5e-04_em32_kge1_10101_g_aiu_0_0_6000
# dbkge_ba320.5_rn2_h12_nmem32_1.0e-04_em32_kge1_g_aiu_0_0_6000
# exp_name=dbkge_ba320.5_rn2_h12_nmem32_1.0e-04_em32_kge1
exp_name=dbkge_ba10242.0_rn4_h12_nmem64_1.0e-04_em32_kge0
# dbkge_ba320.2_rn3_h12_nmem64_0.5e-04_em32_kge1_10101_g_aiu_0_0_6000
load_pretrain_model=1

# export train_file="train_2.py"

# cmd="python3 ../src/${train_file} --reasoning_step ${reasoning_step} --batch_size ${batch_size} --name ${exp_name} \
#    --lr ${lr}  --embed_size ${embed_size} --n_memory ${n_memory} --KGE_pretrained ${KGE_pretrained} \
#    --load_pretrain_model ${load_pretrain_model} --gp_setting ${gp_setting} --epochs ${epochs} \
#     --lambda_num ${lambda_num} --kg_emb_grad ${kg_emb_grad}  --p_hop ${p_hop} --reasoning_step ${reasoning_step} --model ${model} --dataset ${dataset_name}"
# echo "Executing $cmd"
# $cmd

cmd="python3 ../src/${test_file} --name ${exp_name} --batch_size ${batch_size}\
    --gp_setting ${gp_setting} --model ${model} --dataset ${dataset_name} \
    --lambda_num ${lambda_num}  --kg_emb_grad ${kg_emb_grad} --lr ${lr} --p_hop ${p_hop} --reasoning_step ${reasoning_step} --embed_size ${embed_size} --n_memory ${n_memory}"
echo "Executing $cmd"
$cmd
