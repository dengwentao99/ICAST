python -u run_response_warmup_accelerate.py \
  --task_name=responsewarmup \
  --data_dir=labeled_data_dir \
  --model_type=bert \
  --model_name_or_path=pre_model_path \
  --output_dir=output_dir \
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

