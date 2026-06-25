#!/usr/bin/env bash
# ============================================================
# MisBind 使用示例
# ============================================================

# --- 示例 1: 基础用法 (仅图像扰动 + prompt 优化) ---
python -m misbind.core \
  --image_dir ./data/person1/set_B \
  --model sd1.5 \
  --lr 0.01 \
  --lr_p 0.01 \
  --timesteps 200 \
  --grad_accum 10 \
  --prompt_timesteps 200 \
  --prompt_grad_accum 10 \
  --original_word "sks person" \
  --seed 42

# --- 示例 2: 批量处理多个人物 ---
python -m misbind.batch \
  --data_dir ./data/cleandata \
  --output_dir ./outputs/MisBind \
  --model sd1.5 \
  --persons "person1,person2,person3" \
  --lr 0.008 \
  --lr_p 0.1 \
  --timesteps 200 \
  --grad_accum 10 \
  --prompt_timesteps 200 \
  --prompt_grad_accum 20 \
  --original_word "sks person" \
  --surrogate_class_dir ./data/A_class \
  --seed 42

# --- 示例 3: 仅图像模式 (跳过 prompt 优化) ---
python -m misbind.core \
  --image_dir ./data/cat \
  --model sd1.5 \
  --lr 0.01 \
  --timesteps 1000 \
  --grad_accum 5 \
  --image_only \
  --device cuda:0 \
  --seed 42

# --- 示例 4: 带 DreamBooth 的完整流程 ---
python -m misbind.core \
  --image_dir ./data/cat \
  --model sd1.5 \
  --lr 0.01 \
  --lr_p 0.01 \
  --timesteps 200 \
  --grad_accum 10 \
  --prompt_timesteps 200 \
  --prompt_grad_accum 10 \
  --original_word "sks cat" \
  --seed 42 \
  --run_dreambooth \
  --dreambooth_class_dir ./data/A_class/sd15_cat \
  --dreambooth_instance_prompt "a photo of sks cat" \
  --dreambooth_class_prompt "a photo of cat" \
  --dreambooth_prior_loss_weight 1.0 \
  --dreambooth_train_batch_size 1 \
  --dreambooth_gradient_accumulation_steps 1 \
  --dreambooth_learning_rate 5e-6 \
  --dreambooth_lr_scheduler constant \
  --dreambooth_lr_warmup_steps 0 \
  --dreambooth_num_class_images 200 \
  --dreambooth_max_train_steps 800
