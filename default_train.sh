#!/bin/bash

src=$1
tgt=$2
tok=$3
epochs=$4
batch=$5
output_path=$6

python train.py ${src}-${tgt}-train.txt ${src}-${tgt}-val.txt ${src}-${tgt}-test.txt ${epochs} ${batch} ${tok} data/${src}-${tgt}/${src}-tok.txt data/${src}-${tgt}/${tgt}-tok.txt --data_path data/${src}-${tgt} --d_model 512 --nhead 8 --num_encoder_layers 6 --num_decoder_layers 6 --learning_rate 0.0001 --dropout 0.1 --checkpoint_file ${output_path}/${src}-${en}-${tok}.ck --log_file ${output_path}/${src}-${tgt}-${tok}.log