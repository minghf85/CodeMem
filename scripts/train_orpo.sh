#!/bin/bash
# train_orpo_qwen3_4b.sh

export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

MODEL=/root/autodl-tmp/Train/model/Qwen3-8B   # 或 Qwen/Qwen3-4B
DATA=/root/autodl-tmp/EvoAtomMem/data/orpo/atom_dpo_standard.jsonl
OUT=/root/autodl-tmp/EvoAtomMem/output/qwen3-8b-orpo

swift rlhf \
    --rlhf_type orpo \
    --model "$MODEL" \
    --tuner_type lora \
    --lora_rank 32 \
    --lora_alpha 64 \
    --target_modules all-linear \
    --dataset "$DATA" \
    --torch_dtype bfloat16 \
    --bf16 true \
    --gradient_checkpointing true \
    --num_train_epochs 2 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 3e-5 \
    --beta 0.1 \
    --max_length 2048 \
    --warmup_ratio 0.05 \
    --logging_steps 5 \
    --eval_steps 200 \
    --save_steps 200 \
    --save_total_limit 3 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --split_dataset_ratio 0.02 \
    --output_dir "$OUT" \
    --report_to swanlab \
    --swanlab_project EvoAtomMem \
    --swanlab_exp_name qwen3-8b-orpo-gpt5.6-luna-vs-qwen3-4b \
    --swanlab_mode cloud \
    --system "Extract atomic facts from the conversation message."