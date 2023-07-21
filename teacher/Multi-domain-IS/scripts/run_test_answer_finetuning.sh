#!/bin/bash
#SBATCH --job-name=mt5
#SBATCH --gres=gpu:2
#SBATCH --mem=3G

python -u run_response_warmup_accelerate.py \
  --task_name=responsewarmup \
  --data_dir=datasets/MANtIS/valid \
  --model_type=bert \
  --model_name_or_path=prev_trained_model/bert-base-uncased \
  --output_dir=outputs_answer_selection_20220912_5e-5_bce_MANtIS_5 \
  --overwrite_output_dir \
  --logging_steps=100 \
  --save_steps=500 \
  --learning_rate=5e-5 \
  --per_gpu_train_batch_size=30 \
  --per_gpu_eval_batch_size=64 \
  --num_train_epochs=1000 \
  --logging_epochs=-1 \
  --do_lower_case \
  --do_predict

# outputs_answer_20220704_3e-5_bce
# outputs_msdialog_300_20220703_3e-5_bce

