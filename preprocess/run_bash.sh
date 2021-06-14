#!/usr/bin/env 
export CUDA_VISIBLE_DEVICES=$1
# export PYTHONPATH="./"
export PYTHONPATH="../"

dataset=beauty_core

cmd="python ./preprocess.py --dataset ${dataset}"
echo "Executing $cmd"
$cmd &

# cmd="python ./train_transe_rw.py --dataset ${dataset}"
# echo "Executing $cmd"
# $cmd 

dataset=cell_core

cmd="python ./preprocess.py --dataset ${dataset}"
echo "Executing $cmd"
$cmd &

# cmd="python ./train_transe_rw.py --dataset ${dataset}"
# echo "Executing $cmd"
# $cmd 


dataset=cloth_core

cmd="python ./preprocess.py --dataset ${dataset}"
echo "Executing $cmd"
$cmd &

# cmd="python ./train_transe_rw.py --dataset ${dataset}"
# echo "Executing $cmd"
# $cmd 


dataset=MovieLens-1M_core

cmd="python ./preprocess.py --dataset ${dataset}"
echo "Executing $cmd"
$cmd & 

# # cmd="python ./train_transe_kg.py --dataset ${dataset}"
# # echo "Executing $cmd"
# # $cmd 


dataset=amazon-book_20core

cmd="python ./preprocess.py --dataset ${dataset}"
echo "Executing $cmd"
$cmd &

# cmd="python ./train_transe_kg.py --dataset ${dataset}"
# echo "Executing $cmd"
# $cmd 
