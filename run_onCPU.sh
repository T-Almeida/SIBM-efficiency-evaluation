#!/bin/bash

PYTHON=$(pwd)/tf2/bin/python

if [ "$1" == "" ]; then
    echo "Give file name as a program argument"
    exit 1
fi


#echo "model_type,batch_size,avg_time,std_time,median_time" > $1


#for batch_size in 16 32 64 128 256 512 1024; do
for batch_size in 2048; do
    echo RUNNING old $batch_size
    CUDA_VISIBLE_DEVICES="" $PYTHON old_model_inference_test.py $batch_size -o $1
    echo RUNNING new $batch_size
    CUDA_VISIBLE_DEVICES="" $PYTHON new_model_inference_test.py $batch_size -o $1
done
