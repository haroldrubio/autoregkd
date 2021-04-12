#!/bin/sh

run_qa --model_type interpolation \
       --output_dir ./model_outputs/interpolation_outputs/ \
       --overwrite_output_dir True \
       --model_name Primer/bart-squad2 \
       --tokenizer_name Primer/bart-squad2 \
       --task question-answering \
       --dataset_name squad \
       --use_v2 True \
       --fp16 False \
       --fp16_opt_level O1 \
       --learning_rate 5e-5 \
       --warmup_steps 500 \
       --gradient_accumulation_steps 4 \
       --use_kd_loss False \
       --alpha_data 1.0 \
       --alpha_logits 0.8 \
       --alpha_hidden 3.0 \
       --interpolation_p 0.0 \
       --max_prob 1.0 \
       --per_level_annealing_duration 0.3 \
       --step_size 2 \
       --freeze_embedding True \
       --freeze_encoder True \
       --freeze_qa_head True \
       --seed 696 \
       --max_seq_length 384 \
       --doc_stride 128 \
       --n_best_size 20 \
       --max_answer_length 30 \
       --null_score_diff_threshold 0.0 \
       --per_device_train_batch_size 16 \
       --per_device_eval_batch_size 4 \
       --do_train \
       --do_eval \
       --num_interpolation_epochs 2 \
       --num_train_epochs 5 \
       --evaluation_strategy steps \
       --num_evals_per_epoch 4 \
       --save_total_limit 5 \
       --load_best_model_at_end \
       --metric_for_best_model f1


