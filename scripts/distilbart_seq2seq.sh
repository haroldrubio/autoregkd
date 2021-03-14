#!/bin/sh

train_distilbart_seq2seq --use_hf_model False \
                         --output_dir ./model_outputs/distilbart_outputs/xsum/ \
                         --model_name sshleifer/distilbart-xsum-12-3 \
                         --tokenizer_name facebook/bart-large-xsum \
                         --task summarization \
                         --dataset_name xsum \
                         --seed 696 \
                         --max_source_length 1024 \
                         --max_target_length 256 \
                         --per_device_train_batch_size 32 \
                         --per_device_eval_batch_size 32 \
                         --do_train \
                         --do_eval \
                         --do_predict \
                         --num_beams 6 \
                         --num_train_epochs 5 \
                         --evaluation_strategy steps \
                         --save_total_limit 5 \
                         --load_best_model_at_end \
                         --predict_with_generate

