#!/usr/bin/env python
"""
MisBind 批量处理脚本 (ASPL-style surrogate UNet)
对 cleandata 每个人的 set_B 图像添加扰动
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from misbind.core import optimize_folder_images_with_prompt_first


def main() -> None:
    parser = argparse.ArgumentParser(description="MisBind 批量处理")
    parser.add_argument("--data_dir", required=True, help="Path to dataset directory containing person folders")
    parser.add_argument("--output_dir", default="./outputs/MisBind", help="Output directory for perturbed images")
    parser.add_argument("--model", default="sd1.5")
    parser.add_argument("--persons", default="person1,person2,person3,person4,person5,person6,person7,person8")
    # Stage 1
    parser.add_argument("--pseudo_init_mode", default="hybrid", choices=["hybrid", "full_random"])
    parser.add_argument("--num_pseudo_tokens", type=int, default=4)
    parser.add_argument("--lr_p", type=float, default=0.1)
    parser.add_argument("--prompt_timesteps", type=int, default=200)
    parser.add_argument("--prompt_grad_accum", type=int, default=20)
    parser.add_argument("--attn_intensity_weight", type=float, default=0.5)
    parser.add_argument("--prompt_reg_weight", type=float, default=1e-3)
    # Surrogate
    parser.add_argument("--surrogate_train_steps", type=int, default=50)
    parser.add_argument("--surrogate_lr", type=float, default=5e-6)
    parser.add_argument("--surrogate_class_dir", default=None, help="Path to class images for surrogate prior preservation")
    parser.add_argument("--surrogate_class_prompt", default="a photo of a person")
    parser.add_argument("--surrogate_prior_loss_weight", type=float, default=1.0)
    parser.add_argument("--surrogate_num_class_images", type=int, default=200)
    # ASPL outer loop
    parser.add_argument("--outer_loop_steps", type=int, default=1, help="ASPL 外层迭代次数")
    parser.add_argument("--outer_main_train_steps", type=int, default=20, help="每次迭代用攻击图像训练原始 UNet 步数")
    # Stage 2
    parser.add_argument("--original_word", default="sks person")
    parser.add_argument("--lr", type=float, default=0.008)
    parser.add_argument("--timesteps", type=int, default=200)
    parser.add_argument("--grad_accum", type=int, default=10)
    parser.add_argument("--image_batch_size", type=int, default=4)
    parser.add_argument("--stage2_gamma", type=float, default=2.0)
    parser.add_argument("--pgd_eps", type=float, default=0.05)
    # Other
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image_only", action="store_true")
    parser.add_argument("--skip_surrogate", action="store_true", help="跳过 surrogate 训练，直接用原始 UNet")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_base = Path(args.output_dir)
    persons = [p.strip() for p in args.persons.split(",") if p.strip()]

    print("=" * 60)
    print("  MisBind 批量处理")
    print(f"  数据集: {data_dir}")
    print(f"  人物:   {persons}")
    print(f"  输出:   {output_base}")
    print(f"  伪词:   {args.pseudo_init_mode} (N={args.num_pseudo_tokens})")
    print(f"  Stage 1: pt={args.prompt_timesteps} lr_p={args.lr_p} λ={args.attn_intensity_weight}")
    print(f"  Stage 2: ts={args.timesteps} lr={args.lr} γ={args.stage2_gamma} batch={args.image_batch_size}")
    print(f"  Surrogate: steps={args.surrogate_train_steps} lr={args.surrogate_lr}")
    print("=" * 60)

    for person in persons:
        image_dir = data_dir / person / "set_B"
        surrogate_train_dir = data_dir / person / "set_A"

        if not image_dir.is_dir():
            print(f"\n[SKIP] {image_dir} 不存在")
            continue
        if not args.skip_surrogate and not surrogate_train_dir.is_dir():
            print(f"\n[SKIP] {surrogate_train_dir} 不存在")
            continue

        print(f"\n--- [{datetime.now():%H:%M:%S}] {person} "
              f"(set_B: {len(list(image_dir.iterdir()))}, set_A: {len(list(surrogate_train_dir.iterdir()))}) ---")

        try:
            optimize_folder_images_with_prompt_first(
                image_dir=str(image_dir),
                model_name=args.model,
                lr=args.lr,
                lr_p=args.lr_p if not args.image_only else None,
                original_word=args.original_word,
                timesteps=args.timesteps,
                image_batch_size=args.image_batch_size,
                grad_accum=args.grad_accum,
                prompt_timesteps=args.prompt_timesteps if not args.image_only else None,
                prompt_grad_accum=args.prompt_grad_accum if not args.image_only else None,
                pseudo_init_mode=args.pseudo_init_mode,
                num_pseudo_tokens=args.num_pseudo_tokens,
                image_only=args.image_only,
                prompt_reg_weight=args.prompt_reg_weight,
                attn_intensity_weight=args.attn_intensity_weight,
                stage2_gamma=args.stage2_gamma,
                pgd_eps=args.pgd_eps,
                surrogate_train_dir=str(surrogate_train_dir) if not args.skip_surrogate else None,
                surrogate_train_steps=args.surrogate_train_steps,
                surrogate_learning_rate=args.surrogate_lr,
                surrogate_class_dir=args.surrogate_class_dir,
                surrogate_class_prompt=args.surrogate_class_prompt,
                surrogate_prior_loss_weight=args.surrogate_prior_loss_weight,
                surrogate_num_class_images=args.surrogate_num_class_images,
                outer_loop_steps=args.outer_loop_steps,
                outer_main_train_steps=args.outer_main_train_steps,
                output_base_dir=str(output_base),
                seed=args.seed,
            )
        except Exception as e:
            print(f"[ERROR] {person}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n===== 完成，输出: {output_base} =====")


if __name__ == "__main__":
    main()
