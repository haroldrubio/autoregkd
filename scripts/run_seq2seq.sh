#!/bin/sh

run_seq2seq --model_type interpolation \
            --output_dir ./model_outputs/interpolation_outputs/xsum/ \
            --overwrite_output_dir True \
            --model_name facebook/bart-large-xsum \
            --tokenizer_name facebook/bart-large-xsum \
            --task summarization \
            --dataset_name xsum \
            --fp16 True \
            --fp16_opt_level O1 \
            --learning_rate 3e-4 \
            --gradient_accumulation_steps 2 \
            --use_kd_loss False \
            --alpha_data 1.0 \
            --alpha_logits 0.8 \
            --alpha_hidden 3.0 \
            --interpolation_p 0.0 \
            --max_prob 1.0 \
            --per_level_annealing_duration 0.3 \
            --step_size 2 \
            --seed 21 \
            --max_source_length 1024 \
            --max_target_length 256 \
            --per_device_train_batch_size 32 \
            --per_device_eval_batch_size 32 \
            --do_train \
            --do_eval \
            --do_predict \
            --num_beams 6 \
            --num_interpolation_epochs 5 \
            --num_train_epochs 8 \
            --evaluation_strategy steps \
            --num_evals_per_epoch 4 \
            --save_total_limit 5 \
            --predict_with_generate \
            --load_best_model_at_end True


