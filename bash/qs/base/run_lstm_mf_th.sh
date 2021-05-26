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
user_core_th_setting=1

batch_size=1024
gpu=$1
p_hop=$2
embed_size=$4
n_memory=64
epochs=50
lr=1e-4
sub_batch_size=1

reward_rh=no

l2_weight=1e-5

lr=0.8e-4

# TIME_LIST=(2)
# TIME_LIST=(1 2 3)
# batch_size=32
# for dataset_name in ${MODEL_TYPE[@]}
# do
#     for time in ${TIME_LIST[@]}
#     do

#         # exp_name="nogv_lstm_em16_mf_2_p5"

#         exp_name="nogv_lstm_reg_${l2_weight}_em${embed_size}_mf_${time}"
#         use_men='lstm_mf_dummy'

#         cmd="python3 ../train_mem/${train_file} --sub_batch_size ${sub_batch_size} --name ${exp_name} --env_meta_path ${env_meta_path} \
#           --lr ${lr} --user_core_th_setting ${user_core_th_setting} --reward_rh ${reward_rh} --embed_size ${embed_size} --gp_setting ${gp_setting} --epochs ${epochs} --n_memory ${n_memory} --user_o ${user_o} --gpu ${gpu} \
#            --reward_hybrid ${reward_hybrid} --l2_weight ${l2_weight} --batch_size ${batch_size} --h0_embbed ${h0_embbed} --p_hop ${p_hop} --use_men ${use_men} --dataset ${dataset_name}"
#         echo "Executing $cmd"
#         $cmd

#         cmd="python3 ../train_mem/${test_file} --name ${exp_name} --sub_batch_size ${sub_batch_size} --env_meta_path ${env_meta_path}\
#           --user_o ${user_o} --user_core_th_setting ${user_core_th_setting} --reward_rh ${reward_rh} --gpu ${gpu} --gp_setting ${gp_setting}  --h0_embbed ${h0_embbed} --use_men ${use_men} --dataset ${dataset_name} \
#           --reward_hybrid ${reward_hybrid} --l2_weight ${l2_weight} --batch_size ${batch_size} --lr ${lr} --p_hop ${p_hop} --embed_size ${embed_size} --n_memory ${n_memory} --run_path True --run_eval True --test_lstm_up ${test_lstm_up}"
#         echo "Executing $cmd"
#         $cmd
#     done
# done



# bu
# th_qu_800_15_500_lr_0.0001_bs_1024_ma_50_p12_nogv_lstm_em16_mf_2_ldemd1_p2_g_aiu_0_0_6

# cloth
# th_qu_800_15_500_lr_0.0001_bs_1024_ma_50_p12_nogv_lstm_em32_mf_1_g_aiu_0_0_6

# cell
# th_qu_800_15_500_lr_0.0001_bs_1024_ma_50_p12_nogv_lstm_em32_mf_1_g_aiu_0_0_6

load_pretrain_model=0
pretrained_st_epoch=0
epochs=110

lr=1.0e-4

TIME_LIST=(1)
# TIME_LIST=(1 2 3)
# th_qu_800_15_500_lr_0.0001_bs_1024_ma_50_p12_nogv_lstm_em16_mf_3_ldemd1_p3_g_aiu_0_0_6

batch_size=1024

for dataset_name in ${MODEL_TYPE[@]}
do
    for time in ${TIME_LIST[@]}
    do

        # exp_name="nogv_lstm_em32_mf_1"
        # exp_name="nogv_lstm_em${embed_size}_mf_${time}"
        exp_name='nogv_lstm_em16_mf_3_ldemd1'
                # exp_name='nogv_lstm_em16_mf_bfs3'
        # use_men='lstm_mf_dummy'
        use_men='lstm_mf_dummy_no_grad'

        # cmd="python3 ../train_mem/${train_file} --sub_batch_size ${sub_batch_size} --name ${exp_name} --env_meta_path ${env_meta_path} \
        #   --load_pretrain_model ${load_pretrain_model} --pretrained_st_epoch ${pretrained_st_epoch} --lr ${lr} --user_core_th_setting ${user_core_th_setting} --reward_rh ${reward_rh} --embed_size ${embed_size} --gp_setting ${gp_setting} --epochs ${epochs} --n_memory ${n_memory} --user_o ${user_o} --gpu ${gpu} \
        #    --reward_hybrid ${reward_hybrid} --batch_size ${batch_size} --h0_embbed ${h0_embbed} --p_hop ${p_hop} --use_men ${use_men} --dataset ${dataset_name}"
        # echo "Executing $cmd"
        # $cmd

        cmd="python3 ../train_mem/${test_file} --name ${exp_name} --sub_batch_size ${sub_batch_size} --env_meta_path ${env_meta_path}\
          --load_pretrain_model ${load_pretrain_model} --pretrained_st_epoch ${pretrained_st_epoch} --user_o ${user_o} --user_core_th_setting ${user_core_th_setting} --reward_rh ${reward_rh} --gpu ${gpu} --gp_setting ${gp_setting}  --h0_embbed ${h0_embbed} --use_men ${use_men} --dataset ${dataset_name} \
          --reward_hybrid ${reward_hybrid} --batch_size ${batch_size} --lr ${lr} --p_hop ${p_hop} --embed_size ${embed_size} --n_memory ${n_memory} --run_path True --run_eval True --test_lstm_up ${test_lstm_up}"
        echo "Executing $cmd"
        $cmd
    done
done

# load_pretrain_model=0
# pretrained_st_epoch=0
# epochs=100

# lr=1.0e-4

# TIME_LIST=(1)
# # TIME_LIST=(1 2 3)

# batch_size=32

# for dataset_name in ${MODEL_TYPE[@]}
# do
#     for time in ${TIME_LIST[@]}
#     do
# # 
#         # exp_name="nogv_lstm_em32_mf_1"
#         exp_name="nogv_lstm_em${embed_size}_mf_${time}"
#         # use_men='lstm_mf_dummy'
#         use_men='lstm_mf_dummy_grad'

#         cmd="python3 ../train_mem/${train_file} --sub_batch_size ${sub_batch_size} --name ${exp_name} --env_meta_path ${env_meta_path} \
#           --load_pretrain_model ${load_pretrain_model} --pretrained_st_epoch ${pretrained_st_epoch} --lr ${lr} --user_core_th_setting ${user_core_th_setting} --reward_rh ${reward_rh} --embed_size ${embed_size} --gp_setting ${gp_setting} --epochs ${epochs} --n_memory ${n_memory} --user_o ${user_o} --gpu ${gpu} \
#            --reward_hybrid ${reward_hybrid} --batch_size ${batch_size} --h0_embbed ${h0_embbed} --p_hop ${p_hop} --use_men ${use_men} --dataset ${dataset_name}"
#         echo "Executing $cmd"
#         $cmd

#         cmd="python3 ../train_mem/${test_file} --name ${exp_name} --sub_batch_size ${sub_batch_size} --env_meta_path ${env_meta_path}\
#           --load_pretrain_model ${load_pretrain_model} --pretrained_st_epoch ${pretrained_st_epoch} --user_o ${user_o} --user_core_th_setting ${user_core_th_setting} --reward_rh ${reward_rh} --gpu ${gpu} --gp_setting ${gp_setting}  --h0_embbed ${h0_embbed} --use_men ${use_men} --dataset ${dataset_name} \
#           --reward_hybrid ${reward_hybrid} --batch_size ${batch_size} --lr ${lr} --p_hop ${p_hop} --embed_size ${embed_size} --n_memory ${n_memory} --run_path True --run_eval True --test_lstm_up ${test_lstm_up}"
#         echo "Executing $cmd"
#         $cmd
#     done
# done


# bu
# th_qu_800_15_500_lr_0.0001_bs_1024_ma_50_p12_nogv_lstm_em16_mf_3_ldemd1_p3_g_aiu_0_0_6

# cell
# th_qu_800_15_500_lr_0.0001_bs_1024_ma_50_p12_nogv_lstm_em16_mf_3_ldemd1_g_aiu_0_0_6

# cl
# th_qu_800_15_500_lr_0.0001_bs_1024_ma_50_p12_nogv_lstm_em16_mf_2_ldemd1_p3_g_aiu_0_0_6

# if [ $3 = "cell" ]
# then
#     MODEL_TYPE=('cell_core')
#     exp_name=nogv_lstm_em16_mf_3_ldemd1
# elif [ $3 = "bu" ]
# then
#     MODEL_TYPE=('beauty_core')
#     exp_name=nogv_lstm_em16_mf_3_ldemd1_p3
# elif [ $3 = "cl" ]
# then
#     MODEL_TYPE=('cloth_core')
#     exp_name=nogv_lstm_em16_mf_2_ldemd1_p
# fi

# lr=1.0e-4
# load_pt_emb_size=1

# TIME_LIST=(1)

# for dataset_name in ${MODEL_TYPE[@]}
# do
#     for time in ${TIME_LIST[@]}
#     do

#         # exp_name="nogv_lstm_em${embed_size}_mf_${time}_ba_${batch_size}"
#         # exp_name="nogv_lstm_em${embed_size}_mf_${time}_ldemd${load_pt_emb_size}_p5"
#         # use_men='lstm_mf_dummy'
#         use_men='lstm_mf_dummy_no_grad'

#         cmd="python3 ../train_mem/${train_file} --sub_batch_size ${sub_batch_size} --name ${exp_name} --env_meta_path ${env_meta_path} --load_pt_emb_size ${load_pt_emb_size} \
#           --lr ${lr} --user_core_th_setting ${user_core_th_setting} --reward_rh ${reward_rh} --embed_size ${embed_size} --gp_setting ${gp_setting} --epochs ${epochs} --n_memory ${n_memory} --user_o ${user_o} --gpu ${gpu} \
#            --reward_hybrid ${reward_hybrid} --batch_size ${batch_size} --h0_embbed ${h0_embbed} --p_hop ${p_hop} --use_men ${use_men} --dataset ${dataset_name}"
#         echo "Executing $cmd"
#         $cmd

#         cmd="python3 ../train_mem/${test_file} --name ${exp_name} --sub_batch_size ${sub_batch_size} --env_meta_path ${env_meta_path} --load_pt_emb_size ${load_pt_emb_size} \
#           --user_o ${user_o} --user_core_th_setting ${user_core_th_setting} --reward_rh ${reward_rh} --gpu ${gpu} --gp_setting ${gp_setting}  --h0_embbed ${h0_embbed} --use_men ${use_men} --dataset ${dataset_name} \
#           --reward_hybrid ${reward_hybrid} --batch_size ${batch_size} --lr ${lr} --p_hop ${p_hop} --embed_size ${embed_size} --n_memory ${n_memory} --run_path True --run_eval True --test_lstm_up ${test_lstm_up}"
#         echo "Executing $cmd"
#         $cmd
#     done
# done
