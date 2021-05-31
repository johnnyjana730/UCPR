#!/usr/bin/env bash

export train_file="train.py"
export test_file="test.py"

file_name="lstm.sh"

# data=bu
# bash qs/baseline/${file_name} $1 $data &

# data=cl
# bash qs/baseline/${file_name} $1 $data &

# data=cell
# bash qs/baseline/${file_name} $1 $data &

data=az
bash qs/baseline/${file_name} $1 $data &

data=mv
bash qs/baseline/${file_name} $1 $data &

