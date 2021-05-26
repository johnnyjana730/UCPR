#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=$1
export PYTHONPATH="."


export train_file="train.py"
export test_file="test.py"


file_name="UCPR_p1.sh"

# data=bu
# bash qs/mf_rn_nogd/${file_name} $1 $data &

# data=cl
# bash qs/mf_rn_nogd/${file_name} $1 $data &

# data=cell
# bash qs/mf_rn_nogd/${file_name} $1 $data &

file_name="UCPR_p2.sh"

data=az
bash qs/mf_rn_nogd/${file_name} $1 $data &

data=mv
bash qs/mf_rn_nogd/${file_name} $1 $data &