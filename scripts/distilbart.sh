#!/bin/sh

train_distilbart --use_hf_model False \
                 --output_dir /tmp/distilbart_outputs/xsum/ \
                 --model_name facebook/bart-large \
                 --task summarization \
                 --dataset_name xsum \
                 --remove_unused_columns True \
                 --do_train True \
                 --do_eval True \
                 --do_predict True \
                 --num_train_epochs 5 \
                 --evaluation_strategy steps \
                 --load_best_model_at_end True \
                 --predict_with_generate True


