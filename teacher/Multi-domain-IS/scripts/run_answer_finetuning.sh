#!/bin/bash
#SBATCH --gres=gpu:2
#SBATCH --mem=3G
#SBATCH --job-name=f10

python -u run_response_warmup_accelerate.py \
  --task_name=responsewarmup \
  --data_dir=datasets/MANtIS/answer_data_5 \
  --model_type=bert \
  --model_name_or_path=prev_trained_model/bert-base-uncased \
  --output_dir=outputs_cross_20220912_5e-5_MANtIS_5 \
  --overwrite_output_dir \
  --logging_steps=10 \
  --save_steps=500 \
  --learning_rate=5e-5 \
  --per_gpu_train_batch_size=16 \
  --per_gpu_eval_batch_size=64 \
  --num_train_epochs=2000 \
  --logging_epochs=5 \
  --do_lower_case \
  --do_train
