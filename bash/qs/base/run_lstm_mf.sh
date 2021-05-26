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
load_pt_emb_size=1
load_pretrain_model=0
pretrained_st_epoch=0
epochs=200

batch_size=1024
gpu=$1
p_hop=$2
embed_size=$4
n_memory=16

sub_batch_size=1

reward_rh=no

# no_shu_qy_uam_800_15_500_lr_0.00012_bs_32_sb_1_ma_50_p12_nogv_lstm_em16_mf_g_aiu_0_0_6000_qy_uam_800_15_500
# no_shu_qy_uam_800_15_500_lr_0.00013_bs_32_sb_1_ma_50_p12_nogv_lstm_em16_mf_g_aiu_0_0_6000_qy_uam_800_15_500_old
# no_shu_qy_uam_800_15_500_lr_0.00013_bs_32_sb_1_ma_50_p12_nogv_lstm_em16_mf_g_aiu_0_0_6000_qy_uam_800_15_500
# lr=1.2e-4
# exp_name="mqaluru_rn${reasoning_step}_h1${p_hop}_nmem${n_memory}_em${embed_size}"
# exp_name="nogv_lstm_em${embed_size}_mf"
# use_men='lstm_mf'

# for dataset_name in ${MODEL_TYPE[@]}
# do

#     cmd="python3 ../train_mem/${train_file} --sub_batch_size ${sub_batch_size} --name ${exp_name} --env_meta_path ${env_meta_path} \
#       --lr ${lr} --embed_size ${embed_size} --gp_setting ${gp_setting} --reward_rh ${reward_rh} --epochs ${epochs} --n_memory ${n_memory} --user_o ${user_o} --gpu ${gpu} \
#        --reward_hybrid ${reward_hybrid} --batch_size ${batch_size} --load_pretrain_model ${load_pretrain_model} --pretrained_st_epoch ${pretrained_st_epoch} --h0_embbed ${h0_embbed} --p_hop ${p_hop} --use_men ${use_men} --dataset ${dataset_name}"
#     echo "Executing $cmd"
#     $cmd

#     cmd="python3 ../train_mem/${test_file} --name ${exp_name} --sub_batch_size ${sub_batch_size} --env_meta_path ${env_meta_path}\
#       --user_o ${user_o} --gpu ${gpu} --gp_setting ${gp_setting} --h0_embbed ${h0_embbed} --reward_rh ${reward_rh} --use_men ${use_men} --dataset ${dataset_name} \
#       --reward_hybrid ${reward_hybrid} --batch_size ${batch_size} --load_pretrain_model ${load_pretrain_model} --pretrained_st_epoch ${pretrained_st_epoch} --lr ${lr} --p_hop ${p_hop} --embed_size ${embed_size} --n_memory ${n_memory} --run_path True --run_eval True --test_lstm_up ${test_lstm_up}"
#     echo "Executing $cmd"
#     $cmd

# done

# lr=1.2e-4
# exp_name="nogv_lstm_em${embed_size}_mf"
# use_men='lstm_mf_dummy'

# TIME_LIST=(1 2 3)

# for dataset_name in ${MODEL_TYPE[@]}
# do
#     for time in ${TIME_LIST[@]}
#     do
#         exp_name="nogv_lstm_em${embed_size}_mf_${time}_p2"
#         cmd="python3 ../train_mem/${train_file} --sub_batch_size ${sub_batch_size} --name ${exp_name} --env_meta_path ${env_meta_path} \
#           --lr ${lr} --embed_size ${embed_size} --gp_setting ${gp_setting} --reward_rh ${reward_rh} --epochs ${epochs} --n_memory ${n_memory} --user_o ${user_o} --gpu ${gpu} \
#            --reward_hybrid ${reward_hybrid} --batch_size ${batch_size} --load_pretrain_model ${load_pretrain_model} --pretrained_st_epoch ${pretrained_st_epoch} --h0_embbed ${h0_embbed} --p_hop ${p_hop} --use_men ${use_men} --dataset ${dataset_name}"
#         echo "Executing $cmd"
#         $cmd

#         cmd="python3 ../train_mem/${test_file} --name ${exp_name} --sub_batch_size ${sub_batch_size} --env_meta_path ${env_meta_path}\
#           --user_o ${user_o} --gpu ${gpu} --gp_setting ${gp_setting} --h0_embbed ${h0_embbed} --reward_rh ${reward_rh} --use_men ${use_men} --dataset ${dataset_name} \
#           --reward_hybrid ${reward_hybrid} --batch_size ${batch_size} --load_pretrain_model ${load_pretrain_model} --pretrained_st_epoch ${pretrained_st_epoch} --lr ${lr} --p_hop ${p_hop} --embed_size ${embed_size} --n_memory ${n_memory} --run_path True --run_eval True --test_lstm_up ${test_lstm_up}"
#         echo "Executing $cmd"
#         $cmd
#     done
# done

# mv
# no_shu_qy_uam_100_15_500_lr_0.0002_bs_1024_sb_1_ma_50_p12_nogv_lstm_em16_mf_1_ldemb1_p3_g_aiu_0_0_6000_qy_uam_100_15_500

# az
# no_shu_qy_uam_100_15_500_lr_0.0002_bs_1024_sb_1_ma_50_p12_nogv_lstm_em32_mf_1_ldemb1_g_aiu_0_0_6000_qy_uam_100_15_500

# lr=2e-4
lr=0.0002
embed_size=32
load_pretrain_model=0
pretrained_st_epoch=0
epochs=400

l2_weight=1e-5

TIME_LIST=(2)

for dataset_name in ${MODEL_TYPE[@]}
do
    for time in ${TIME_LIST[@]}
    do

        # exp_name="nogv_lstm_arg${l2_weight}em${embed_size}mf${time}lmb${load_pt_emb_size}"
        exp_name=nogv_lstm_em32_mf_1_ldemb1
        use_men='lstm_mf_dummy_no_grad'

        # cmd="python3 ../train_mem/${train_file} --l2_weight ${l2_weight} --sub_batch_size ${sub_batch_size} --name ${exp_name} --env_meta_path ${env_meta_path} --load_pt_emb_size ${load_pt_emb_size} \
        #   --lr ${lr} --embed_size ${embed_size} --gp_setting ${gp_setting} --reward_rh ${reward_rh} --epochs ${epochs} --n_memory ${n_memory} --user_o ${user_o} --gpu ${gpu} \
        #    --reward_hybrid ${reward_hybrid} --batch_size ${batch_size} --load_pretrain_model ${load_pretrain_model} --pretrained_st_epoch ${pretrained_st_epoch} --h0_embbed ${h0_embbed} --p_hop ${p_hop} --use_men ${use_men} --dataset ${dataset_name}"
        # echo "Executing $cmd"
        # $cmd

        cmd="python3 ../train_mem/${test_file} --l2_weight ${l2_weight} --name ${exp_name} --sub_batch_size ${sub_batch_size} --env_meta_path ${env_meta_path}  --load_pt_emb_size ${load_pt_emb_size} \
          --user_o ${user_o} --gpu ${gpu} --gp_setting ${gp_setting} --h0_embbed ${h0_embbed} --reward_rh ${reward_rh} --use_men ${use_men} --dataset ${dataset_name} \
          --reward_hybrid ${reward_hybrid} --batch_size ${batch_size} --load_pretrain_model ${load_pretrain_model} --pretrained_st_epoch ${pretrained_st_epoch} --lr ${lr} --p_hop ${p_hop} --embed_size ${embed_size} --n_memory ${n_memory} --run_path True --run_eval True --test_lstm_up ${test_lstm_up}"
        echo "Executing $cmd"
        $cmd

    done
done

# cd save_model_debug/amazon-book_20core/lstm_mf_dummy_no_grad_rh_no/

# no_shu_qy_uam_100_15_500_lr_0.0002_bs_1024_sb_1_ma_50_p12_nogv_lstm_em32_mf_1_ldemb1_g_aiu_0_0_6000_qy_uam_100_15_500

# mv
# no_shu_qy_uam_100_15_500_lr_0.0002_bs_1024_sb_1_ma_50_p12_nogv_lstm_em32_mf_1_ldemb1_p3_g_aiu_0_0_6000_qy_uam_100_15_500

# batch_size=32
# lr=2e-4
# embed_size=32
# load_pretrain_model=1
# pretrained_st_epoch=0
# epochs=300

# # TIME_LIST=(1)
# TIME_LIST=(1)

# for dataset_name in ${MODEL_TYPE[@]}
# do
#     for time in ${TIME_LIST[@]}
#     do

#         # exp_name="nogv_lstm_em${embed_size}_mf_${time}_ldemb${load_pt_emb_size}"
#         exp_name="nogv_lstm_em32_mf_1_ldemb1_p3"
#         use_men='lstm_mf_dummy_grad'

#         # cmd="python3 ../train_mem/${train_file} --sub_batch_size ${sub_batch_size} --name ${exp_name} --env_meta_path ${env_meta_path} --load_pt_emb_size ${load_pt_emb_size} \
#         #   --lr ${lr} --embed_size ${embed_size} --gp_setting ${gp_setting} --reward_rh ${reward_rh} --epochs ${epochs} --n_memory ${n_memory} --user_o ${user_o} --gpu ${gpu} \
#         #    --reward_hybrid ${reward_hybrid} --batch_size ${batch_size} --load_pretrain_model ${load_pretrain_model} --pretrained_st_epoch ${pretrained_st_epoch} --h0_embbed ${h0_embbed} --p_hop ${p_hop} --use_men ${use_men} --dataset ${dataset_name}"
#         # echo "Executing $cmd"
#         # $cmd

#         cmd="python3 ../train_mem/${test_file} --name ${exp_name} --sub_batch_size ${sub_batch_size} --env_meta_path ${env_meta_path}  --load_pt_emb_size ${load_pt_emb_size} \
#           --user_o ${user_o} --gpu ${gpu} --gp_setting ${gp_setting} --h0_embbed ${h0_embbed} --reward_rh ${reward_rh} --use_men ${use_men} --dataset ${dataset_name} \
#           --reward_hybrid ${reward_hybrid} --batch_size ${batch_size} --load_pretrain_model ${load_pretrain_model} --pretrained_st_epoch ${pretrained_st_epoch} --lr ${lr} --p_hop ${p_hop} --embed_size ${embed_size} --n_memory ${n_memory} --run_path True --run_eval True --test_lstm_up ${test_lstm_up}"
#         echo "Executing $cmd"
#         $cmd

#     done
# done

