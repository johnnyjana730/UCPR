#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=$1
export PYTHONPATH="../"

if [ $2 = "az" ]
then
    dataset_name='amazon-book_20core'
    model='UCPR'
    gp_setting="6000_800_15_500_50"
    lr=2e-04

    # no_shu_qy_uam_800_15_500_lr_8e-05_bs_32_sb_1_ma_50_p12_ald_areg2e-6_lda0.25rn3h2m64e32_lm0_qy_uam_800_15_500

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
    batch_size=32
fi

lr=1.2e-04
epochs=400
KGE_pretrained=1


load_pretrain_model=1
save_pretrain_model=0

# exp_name=lstm_pre_ba${batch_size}lstm_${epochs}${lr}_sveb${embed_size}kge${kg_emb_grad}l${load_pretrain_model}

# exp_name=lstm_pre_ba32lstm_4001.2e-04_sveb32kge0l1

# cmd="python3 ../src/${train_file} --reasoning_step ${reasoning_step} --batch_size ${batch_size} --name ${exp_name} \
#    --lr ${lr} --embed_size ${embed_size} --n_memory ${n_memory}  \
#    --load_pretrain_model ${load_pretrain_model}  --gp_setting ${gp_setting} --epochs ${epochs} --KGE_pretrained ${KGE_pretrained} \
#     --lambda_num ${lambda_num} --kg_emb_grad ${kg_emb_grad} --p_hop ${p_hop} --reasoning_step ${reasoning_step} --model lstm --dataset ${dataset_name}"
# echo "Executing $cmd"
# $cmd

cmd="python3 ../src/${test_file} --name ${exp_name} --batch_size ${batch_size} \
  --gp_setting ${gp_setting}  --model lstm --dataset ${dataset_name} --save_pretrain_model ${save_pretrain_model}  \
  --lambda_num ${lambda_num}  --kg_emb_grad ${kg_emb_grad} --lr ${lr} --p_hop ${p_hop} --reasoning_step ${reasoning_step} --embed_size ${embed_size} --n_memory ${n_memory}"
echo "Executing $cmd"
$cmd
