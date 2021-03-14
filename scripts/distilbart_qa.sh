#!/bin/sh

train_distilbart_qa --use_hf_model False \
                    --output_dir ./model_outputs/distilbart_outputs/squad/ \
                    --model_name valhalla/bart-large-finetuned-squadv1 \
                    --tokenizer_name valhalla/bart-large-finetuned-squadv1 \
                    --task question-answering \
                    --dataset_name squad \
                    --use_v2 False \
                    --seed 696 \
                    --max_length 512 \
                    --per_device_train_batch_size 32 \
                    --per_device_eval_batch_size 32 \
                    --do_eval \
                    --max_val_samples 1 \
                    --num_train_epochs 5 \
                    --evaluation_strategy steps \
                    --save_total_limit 5 \
                    --load_best_model_at_end


