#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=$1
export PYTHONPATH="../"


if [ $2 = "cell" ]
then
    dataset_name='cell_core'
    lr=1e-04
    embed_size=16
elif [ $2 = "bu" ]
then
    dataset_name='beauty_core'
    lr=1e-04
    embed_size=16
elif [ $2 = "cl" ]
then
    dataset_name='cloth_core'
    lr=1e-04
    embed_size=16
elif [ $2 = "az" ]
then
    dataset_name='amazon-book_20core'
    lr=1e-04
    embed_size=32
elif [ $2 = "mv" ]
then
    dataset_name='MovieLens-1M_core'
    lr=1e-04
    embed_size=32
fi

batch_size=1024
model=baseline


epochs=500
KGE_pretrained=1
kg_emb_grad=0
exp_name=pre_baseline_lr${lr}

cmd="python3 ../src/${train_file} --batch_size ${batch_size} --name ${exp_name}  \
   --lr ${lr}  --embed_size ${embed_size}  --epochs ${epochs} --KGE_pretrained ${KGE_pretrained} \
   --kg_emb_grad ${kg_emb_grad} --model ${model} --dataset ${dataset_name}"
echo "Executing $cmd"
$cmd

cmd="python3 ../src/${test_file} --name ${exp_name} --batch_size ${batch_size} \
 --dataset ${dataset_name} --model ${model}  --kg_emb_grad ${kg_emb_grad}  --lr ${lr} --embed_size ${embed_size}"
echo "Executing $cmd"
$cmd


lr=3.0e-04

epochs=500
KGE_pretrained=1
kg_emb_grad=0
exp_name=pre_baseline_lr${lr}

cmd="python3 ../src/${train_file} --batch_size ${batch_size} --name ${exp_name}  \
   --lr ${lr}  --embed_size ${embed_size}  --epochs ${epochs} --KGE_pretrained ${KGE_pretrained} \
   --kg_emb_grad ${kg_emb_grad} --model ${model} --dataset ${dataset_name}"
echo "Executing $cmd"
$cmd

cmd="python3 ../src/${test_file} --name ${exp_name} --batch_size ${batch_size} \
 --dataset ${dataset_name} --model ${model}  --kg_emb_grad ${kg_emb_grad}  --lr ${lr} --embed_size ${embed_size}"
echo "Executing $cmd"
$cmd
