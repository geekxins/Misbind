from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import os
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from diffusers import DDPMScheduler, StableDiffusionPipeline
from diffusers.models.attention_processor import Attention

try:
    from diffusers import StableDiffusionXLPipeline
except ImportError:
    StableDiffusionXLPipeline = None


# ============================================================
# 1. Prompt 数据结构
# ============================================================
@dataclass
class PromptSpec:
    """
    统一的 prompt 描述结构。

    字段说明：
    - prompt:
        主 prompt。所有模型都使用它。
    - prompt_2:
        主要给 SDXL 预留。SDXL 有两个文本编码器，因此可以额外传一个第二 prompt。
    - negative_prompt / negative_prompt_2:
        当前脚本不启用 classifier-free guidance，
        所以 negative prompt 不参与损失计算。
        这里保留字段，主要是为了输入格式统一、方便后续扩展。
    """
    prompt: str
    prompt_2: Optional[str] = None
    negative_prompt: Optional[str] = None
    negative_prompt_2: Optional[str] = None


# ============================================================
# 2. 模型注册表
# ============================================================
MODEL_REGISTRY: Dict[str, Dict[str, object]] = {
    "sd1.5": {
        "repo_id": "stable-diffusion-v1-5/stable-diffusion-v1-5",
        "pipeline_cls": StableDiffusionPipeline,
        "size": 512,
        "is_sdxl": False,
        "output_tag": "sd15",
    },
    "2.1base": {
        "repo_id": "Manojb/stable-diffusion-2-1-base",
        "pipeline_cls": StableDiffusionPipeline,
        "size": 512,
        "is_sdxl": False,
        "output_tag": "sd21base",
    },
    "xlbase-1.0": {
        "repo_id": "stabilityai/stable-diffusion-xl-base-1.0",
        "pipeline_cls": StableDiffusionXLPipeline,
        "size": 1024,
        "is_sdxl": True,
        "output_tag": "sdxlbase10",
    },
}

ALLOWED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_PROMPT = "a photo of a"
DEFAULT_NUM_PSEUDO_TOKENS = 4
CROSS_ATTN_INTENSITY_LAYERS = [
    "down_blocks.0.attentions.0",
    "mid_block.attentions.0",
    "up_blocks.2.attentions.0",
]
DEFAULT_ATTN_LAYER_FILTER = "attn2"
DEFAULT_ATTN_SAVE_SIZE = 512


# ============================================================
# 3. 命令行参数解析
# ============================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="先用 Adam 更新 prompt embeddings 以最大化损失，再用带 momentum 的图像优化以最小化损失。"
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        required=True,
        help="输入图像文件夹路径。"
    )

    parser.add_argument(
        "--pseudo_init_mode",
        type=str,
        default="hybrid",
        choices=["hybrid"],
        help="Stage I 伪词初始化模式：hybrid (base prompt + 随机向量) / full_random (全部随机)。",
    )
    parser.add_argument(
        "--num_pseudo_tokens",
        type=int,
        default=DEFAULT_NUM_PSEUDO_TOKENS,
        help="hybrid 模式下随机初始化的伪词 token 数量。",
    )
    parser.add_argument(
        "--attn_intensity_weight",
        type=float,
        default=0.0,
        help="Stage I/II attention intensity 权重，对整句 A(x,t) 的惩罚系数。",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=list(MODEL_REGISTRY.keys()),
        help="模型名：sd1.5 / 2.1base / xlbase-1.0"
    )
    parser.add_argument(
        "--lr",
        type=float,
        required=True,
        help="图像阶段 PGD 更新步长 (alpha)。"
    )
    parser.add_argument(
        "--lr_p",
        type=float,
        default=None,
        help="prompt embeddings 更新学习率。prompt 阶段使用 Adam 执行梯度上升。"
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        required=True,
        help=(
            "图像阶段总采样次数。这里指[图像优化阶段总共采样多少次扩散时间步 t],"
            "不是 scheduler.config.num_train_timesteps。"
        )
    )
    parser.add_argument(
        "--pgd_eps",
        type=float,
        default=0.05,
        help="PGD 扰动预算 epsilon，‖x - x_orig‖_∞ ≤ eps。",
    )
    parser.add_argument(
        "--image_batch_size",
        type=int,
        default=None,
        help="图像与 prompt 联合优化时的 batch size。默认等于 image_dir 中的图像数量。"
    )
    parser.add_argument(
        "--grad_accum",
        type=int,
        required=True,
        help="图像阶段梯度累积次数。必须满足 timesteps 是 grad_accum 的正整数倍。"
    )
    parser.add_argument(
        "--prompt_timesteps",
        type=int,
        default=None,
        help=(
            "prompt 阶段总采样次数。这里指[prompt 优化阶段总共采样多少次扩散时间步 t],"
            "不是 scheduler.config.num_train_timesteps。"
        )
    )
    parser.add_argument(
        "--prompt_grad_accum",
        type=int,
        default=None,
        help="prompt 阶段梯度累积次数。必须满足 prompt_timesteps 是 prompt_grad_accum 的正整数倍。"
    )
    parser.add_argument(
        "--image_only",
        action="store_true",
        help="若开启，则跳过 prompt 优化阶段，只优化图像。"
    )
    parser.add_argument(
        "--prompt_reg_weight",
        type=float,
        default=1e-3,
        help="prompt embedding 正则权重，用于约束其不要偏离初始 embedding 太远。"
    )
    parser.add_argument(
        "--original_word",
        type=str,
        default=None,
        help="Stage II 中用于原始类别的词元 e_0。若不提供 image_only，则必须指定。",
    )
    parser.add_argument(
        "--stage2_gamma",
        type=float,
        default=1.0,
        help="Stage II 中平衡原始类别扰乱与伪词语义对齐的权重系数 γ。",
    )
    parser.add_argument(
        "--surrogate_train_dir",
        type=str,
        default=None,
        help="ASPL 风格 surrogate 训练所需的 set_A 图像目录。"
    )
    parser.add_argument(
        "--surrogate_train_steps",
        type=int,
        default=20,
        help="surrogate UNet 训练步数。"
    )
    parser.add_argument(
        "--surrogate_learning_rate",
        type=float,
        default=5e-6,
        help="surrogate UNet 训练学习率。"
    )
    parser.add_argument(
        "--surrogate_class_dir",
        type=str,
        default=None,
        help="surrogate 训练的 class 图像目录 (prior preservation)。"
    )
    parser.add_argument(
        "--surrogate_class_prompt",
        type=str,
        default=None,
        help="class prompt，如 'a photo of a person'。"
    )
    parser.add_argument(
        "--surrogate_prior_loss_weight",
        type=float,
        default=1.0,
        help="surrogate 训练 prior loss 权重。"
    )
    parser.add_argument(
        "--surrogate_num_class_images",
        type=int,
        default=200,
        help="使用的 class 图像数量上限。"
    )
    parser.add_argument(
        "--outer_loop_steps",
        type=int,
        default=1,
        help="ASPL 外层迭代次数。每次迭代: surrogate训练 → PGD攻击 → 用攻击图像训练原始UNet。",
    )
    parser.add_argument(
        "--outer_main_train_steps",
        type=int,
        default=20,
        help="每次外层迭代中，用攻击后图像训练原始 UNet 的步数。",
    )
    parser.add_argument(
        "--attn_layer_filters",
        type=str,
        default=DEFAULT_ATTN_LAYER_FILTER,
        help=(
            "用逗号分隔的 cross-attention 层名子串过滤器。"
            "默认 attn2，会匹配 U-Net 中的 cross-attention 层。"
        ),
    )
    parser.add_argument(
        "--attn_save_size",
        type=int,
        default=DEFAULT_ATTN_SAVE_SIZE,
        help="保存 attention 热图时上采样到的正方形分辨率。"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="可选。指定输出目录。默认为 outputs/<dataset>_<model>_<timestamp>/。",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="例如 cuda / cuda:0 / cpu。默认自动选择。"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子。"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="可选。默认使用模型推荐分辨率。"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="可选。默认使用模型推荐分辨率。"
    )
    parser.add_argument(
        "--run_dreambooth",
        action="store_true",
        help="若开启，则在图像优化结束后自动执行 DreamBooth 训练。"
    )
    parser.add_argument(
        "--dreambooth_class_dir",
        type=str,
        default=None,
        help="DreamBooth 的 class 图像目录。仅在 --run_dreambooth 开启时需要。"
    )
    parser.add_argument(
        "--dreambooth_output_subdir",
        type=str,
        default="dreambooth",
        help="DreamBooth checkpoint 输出到当前输出目录下的哪个子文件夹。"
    )
    parser.add_argument(
        "--dreambooth_instance_prompt",
        type=str,
        default=None,
        help="DreamBooth 的 instance prompt。开启 --run_dreambooth 时必须显式指定。"
    )
    parser.add_argument(
        "--dreambooth_class_prompt",
        type=str,
        default=None,
        help="DreamBooth 的 class prompt。开启 --run_dreambooth 时必须显式指定。"
    )
    parser.add_argument(
        "--dreambooth_prior_loss_weight",
        type=float,
        default=None,
        help="DreamBooth 的 prior loss weight。开启 --run_dreambooth 时必须显式指定。"
    )
    parser.add_argument(
        "--dreambooth_train_batch_size",
        type=int,
        default=None,
        help="DreamBooth 的 train_batch_size。开启 --run_dreambooth 时必须显式指定。"
    )
    parser.add_argument(
        "--dreambooth_gradient_accumulation_steps",
        type=int,
        default=None,
        help="DreamBooth 的 gradient_accumulation_steps。开启 --run_dreambooth 时必须显式指定。"
    )
    parser.add_argument(
        "--dreambooth_learning_rate",
        type=float,
        default=None,
        help="DreamBooth 的 learning_rate。开启 --run_dreambooth 时必须显式指定。"
    )
    parser.add_argument(
        "--dreambooth_lr_scheduler",
        type=str,
        default=None,
        help="DreamBooth 的 lr_scheduler。开启 --run_dreambooth 时必须显式指定。"
    )
    parser.add_argument(
        "--dreambooth_lr_warmup_steps",
        type=int,
        default=None,
        help="DreamBooth 的 lr_warmup_steps。开启 --run_dreambooth 时必须显式指定。"
    )
    parser.add_argument(
        "--dreambooth_num_class_images",
        type=int,
        default=None,
        help="DreamBooth 的 num_class_images。开启 --run_dreambooth 时必须显式指定。"
    )
    parser.add_argument(
        "--dreambooth_max_train_steps",
        type=int,
        default=None,
        help="DreamBooth 的 max_train_steps。开启 --run_dreambooth 时必须显式指定。"
    )
    return parser.parse_args()


# ============================================================
# 4. 基础工具函数
# ============================================================
def resolve_device(device_arg: Optional[str]) -> torch.device:
    """
    解析最终使用的 device。
    如果用户未指定，则默认优先用 CUDA，否则用 CPU。
    """
    if device_arg is not None:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def list_images(image_dir: Path) -> List[Path]:
    """
    列出文件夹中的所有支持图像文件，并按文件名排序。
    """
    image_paths = [
        p for p in sorted(image_dir.iterdir())
        if p.is_file() and p.suffix.lower() in ALLOWED_IMAGE_EXTS
    ]
    if not image_paths:
        raise ValueError(f"在文件夹 {image_dir} 中没有找到支持的图像文件。")
    return image_paths


def chunk_list(items: List[Path], chunk_size: int) -> List[List[Path]]:
    """
    按固定 chunk_size 切分列表，用于 batch 处理。
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size 必须 > 0")
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]


def make_unique_output_dir(base_dir: Path) -> Path:
    """
    为输出目录生成一个不冲突的路径。

    规则：
    - 默认使用 目录名 + 时间戳
    - 若极少数情况下同一秒内重名，则继续追加 _1、_2、...
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = base_dir.parent / f"{base_dir.name}_{timestamp}"
    if not candidate.exists():
        return candidate

    suffix_idx = 1
    while True:
        suffixed_candidate = base_dir.parent / f"{base_dir.name}_{timestamp}_{suffix_idx}"
        if not suffixed_candidate.exists():
            return suffixed_candidate
        suffix_idx += 1


def build_output_stem(image_dir_name: str, model_name: str) -> str:
    """
    构造输出目录前缀，例如：cat_sd15。
    """
    model_tag = str(MODEL_REGISTRY[model_name]["output_tag"])
    return f"{image_dir_name}_{model_tag}"


def infer_dreambooth_cuda_visible_devices(device_arg: Optional[str]) -> Optional[str]:
    """
    从 --device 推导 DreamBooth 使用的 CUDA_VISIBLE_DEVICES。

    规则：
    - cuda:0 -> 0
    - cuda:1 -> 1
    - cuda   -> 保持当前环境，不额外设置
    - cpu    -> 返回 None
    """
    if device_arg is None:
        return None

    normalized = device_arg.strip().lower()
    if normalized == "cpu":
        return None
    if normalized == "cuda":
        return None
    if normalized.startswith("cuda:"):
        return normalized.split(":", 1)[1]
    return None


def validate_dreambooth_config(
    model_name: str,
    class_dir: Optional[str],
    output_subdir: str,
    instance_prompt: Optional[str],
    class_prompt: Optional[str],
    prior_loss_weight: Optional[float],
    train_batch_size: Optional[int],
    gradient_accumulation_steps: Optional[int],
    learning_rate: Optional[float],
    lr_scheduler: Optional[str],
    lr_warmup_steps: Optional[int],
    num_class_images: Optional[int],
    max_train_steps: Optional[int],
) -> dict:
    """
    检查自动 DreamBooth 所需配置是否完整。
    """
    if MODEL_REGISTRY[model_name]["is_sdxl"]:
        raise ValueError("自动 DreamBooth 目前仅支持非 SDXL 模型。")

    if class_dir is None:
        raise ValueError("开启 --run_dreambooth 时，必须提供 --dreambooth_class_dir。")

    class_dir_path = Path(class_dir)
    if not class_dir_path.exists() or not class_dir_path.is_dir():
        raise ValueError(f"dreambooth_class_dir 不存在或不是文件夹：{class_dir_path}")

    if not output_subdir.strip():
        raise ValueError("dreambooth_output_subdir 不能为空。")

    if shutil.which("accelerate") is None:
        raise ValueError("未找到 accelerate 命令，无法自动执行 DreamBooth。")

    train_script = PROJECT_ROOT / "scripts/train_dreambooth.py"
    if not train_script.exists():
        raise ValueError(f"未找到 DreamBooth 训练脚本：{train_script}")

    required_args = {
        "--dreambooth_instance_prompt": instance_prompt,
        "--dreambooth_class_prompt": class_prompt,
        "--dreambooth_prior_loss_weight": prior_loss_weight,
        "--dreambooth_train_batch_size": train_batch_size,
        "--dreambooth_gradient_accumulation_steps": gradient_accumulation_steps,
        "--dreambooth_learning_rate": learning_rate,
        "--dreambooth_lr_scheduler": lr_scheduler,
        "--dreambooth_lr_warmup_steps": lr_warmup_steps,
        "--dreambooth_num_class_images": num_class_images,
        "--dreambooth_max_train_steps": max_train_steps,
    }
    missing_args = [arg_name for arg_name, value in required_args.items() if value is None]
    if missing_args:
        raise ValueError(
            "开启 --run_dreambooth 时，必须显式提供以下参数："
            + " ".join(missing_args)
        )

    if not str(instance_prompt).strip():
        raise ValueError("--dreambooth_instance_prompt 不能为空。")
    if not str(class_prompt).strip():
        raise ValueError("--dreambooth_class_prompt 不能为空。")
    if not str(lr_scheduler).strip():
        raise ValueError("--dreambooth_lr_scheduler 不能为空。")

    return {
        "class_dir_path": class_dir_path,
        "output_subdir": output_subdir.strip(),
        "instance_prompt": str(instance_prompt),
        "class_prompt": str(class_prompt),
        "prior_loss_weight": float(prior_loss_weight),
        "train_batch_size": int(train_batch_size),
        "gradient_accumulation_steps": int(gradient_accumulation_steps),
        "learning_rate": float(learning_rate),
        "lr_scheduler": str(lr_scheduler),
        "lr_warmup_steps": int(lr_warmup_steps),
        "num_class_images": int(num_class_images),
        "max_train_steps": int(max_train_steps),
    }


def run_dreambooth_training(
    model_name: str,
    instance_dir: Path,
    class_dir: Path,
    checkpoint_dir: Path,
    resolution: int,
    device_arg: Optional[str],
    instance_prompt: str,
    class_prompt: str,
    prior_loss_weight: float,
    train_batch_size: int,
    gradient_accumulation_steps: int,
    learning_rate: float,
    lr_scheduler: str,
    lr_warmup_steps: int,
    num_class_images: int,
    max_train_steps: int,
) -> Path:
    """
    在图像优化结束后自动启动 DreamBooth 训练。
    """
    train_script = PROJECT_ROOT / "scripts/train_dreambooth.py"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    command = [
        "accelerate",
        "launch",
        str(train_script),
        f"--pretrained_model_name_or_path={MODEL_REGISTRY[model_name]['repo_id']}",
        f"--instance_data_dir={instance_dir}",
        f"--class_data_dir={class_dir}",
        f"--output_dir={checkpoint_dir}",
        "--with_prior_preservation",
        f"--prior_loss_weight={prior_loss_weight}",
        f"--instance_prompt={instance_prompt}",
        f"--class_prompt={class_prompt}",
        f"--resolution={resolution}",
        f"--train_batch_size={train_batch_size}",
        f"--gradient_accumulation_steps={gradient_accumulation_steps}",
        f"--learning_rate={learning_rate}",
        f"--lr_scheduler={lr_scheduler}",
        f"--lr_warmup_steps={lr_warmup_steps}",
        f"--num_class_images={num_class_images}",
        f"--max_train_steps={max_train_steps}",
    ]

    env = os.environ.copy()
    cuda_visible_devices = infer_dreambooth_cuda_visible_devices(device_arg)
    if cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

    print("[INFO] DreamBooth 已开启，开始执行训练命令：")
    if cuda_visible_devices is not None:
        print(f"[INFO] CUDA_VISIBLE_DEVICES={cuda_visible_devices} {shlex.join(command)}")
    else:
        print(f"[INFO] {shlex.join(command)}")

    subprocess.run(
        command,
        check=True,
        cwd=str(PROJECT_ROOT),
        env=env,
    )

    print(f"[INFO] DreamBooth 训练完成，checkpoint 输出目录：{checkpoint_dir}")
    return checkpoint_dir


def run_sd15_inference_sampling(
    model_dir: Path,
    output_dir: Path,
    subject_prompt: str,
    device_arg: Optional[str],
    seed: int,
) -> None:
    """
    调用 projects/DreamBooth/sd15_inference.py 对 DreamBooth checkpoint 做采样。
    """
    inference_script = PROJECT_ROOT / "scripts/sd15_inference.py"
    if not inference_script.exists():
        raise ValueError(f"未找到 SD1.5 推理脚本：{inference_script}")

    command = [
        "python",
        str(inference_script),
        f"--model_dir={model_dir}",
        f"--output_dir={output_dir}",
        f"--subject_prompt={subject_prompt}",
        f"--seed={seed}",
    ]
    if device_arg is not None:
        command.append(f"--device={device_arg}")

    print("[INFO] 开始执行 DreamBooth 采样：")
    print(f"[INFO] {shlex.join(command)}")
    subprocess.run(
        command,
        check=True,
        cwd=str(PROJECT_ROOT),
    )
    print(f"[INFO] DreamBooth 采样完成，结果目录：{output_dir}")


def release_cuda_memory() -> None:
    """
    尽量释放 Python 与 PyTorch 持有的临时显存。
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()


def save_run_config(config: dict, save_path: Path) -> None:
    """
    将本次运行配置保存为 JSON，方便复现与追踪。
    """
    save_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def append_jsonl_record(save_path: Path, record: dict) -> None:
    """
    以 JSONL 形式追加一条日志，便于后续统计每轮优化指标。
    """
    with save_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _normalize_single_map(attention_map: torch.Tensor) -> torch.Tensor:
    flat = attention_map.reshape(-1)
    min_val = flat.min()
    max_val = flat.max()
    denom = (max_val - min_val).clamp(min=1e-12)
    return (attention_map - min_val) / denom


def save_attention_map_as_image(attention_map: torch.Tensor, save_path: Path) -> None:
    """
    将单张 attention map 保存为伪彩色热图。
    """
    if attention_map.ndim == 3:
        if attention_map.shape[0] != 1:
            raise ValueError("save_attention_map_as_image 仅支持单张 attention map。")
        attention_map = attention_map[0]

    attention_map = _normalize_single_map(attention_map.detach().float().cpu())
    red = attention_map
    green = (1.0 - (attention_map * 2.0 - 1.0).abs()).clamp(0.0, 1.0)
    blue = 1.0 - attention_map
    rgb = torch.stack([red, green, blue], dim=-1).numpy()
    rgb = (rgb * 255.0).round().clip(0, 255).astype(np.uint8)
    Image.fromarray(rgb).save(save_path)


def _infer_spatial_shape(num_positions: int, target_height: int, target_width: int) -> Tuple[int, int]:
    """
    根据 query token 个数和目标长宽比，恢复 attention map 的二维形状。
    """
    if num_positions <= 0:
        raise ValueError("num_positions 必须 > 0")

    ratio = float(target_height) / float(max(target_width, 1))
    best_hw = (int(math.sqrt(num_positions)), int(math.sqrt(num_positions)))
    best_score = float("inf")

    for h in range(1, int(math.sqrt(num_positions)) + 1):
        if num_positions % h != 0:
            continue
        w = num_positions // h
        candidates = [(h, w), (w, h)]
        for cand_h, cand_w in candidates:
            score = abs((cand_h / float(max(cand_w, 1))) - ratio)
            if score < best_score:
                best_score = score
                best_hw = (cand_h, cand_w)

    if best_hw[0] * best_hw[1] != num_positions:
        raise ValueError(f"无法将 {num_positions} 个空间位置恢复成二维布局。")

    return best_hw


class RecordingAttnProcessor:
    """
    轻量版 cross-attention processor。

    作用：
    - 复用 diffusers Attention 模块自身的 q/k/v 与 score 计算
    - 在 cross-attention 路径上缓存 attention_probs
    - 输出行为与默认 AttnProcessor 保持一致
    """

    def __init__(self, recorder: "CrossAttentionMapRecorder", layer_name: str):
        self.recorder = recorder
        self.layer_name = layer_name

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        del args, kwargs

        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        else:
            batch_size = hidden_states.shape[0]
            channel = -1
            height = -1
            width = -1

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        is_cross_attention = encoder_hidden_states is not None
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)

        if is_cross_attention:
            self.recorder.record(
                layer_name=self.layer_name,
                attention_probs=attention_probs,
                batch_size=batch_size,
                query_tokens=attention_probs.shape[1],
                fallback_height=height,
                fallback_width=width,
            )

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states


class CrossAttentionMapRecorder:
    """
    负责在 U-Net cross-attention 层中提取 token-level 响应并聚合成热图。
    """

    def __init__(self, unet, layer_filters: Sequence[str]):
        self.unet = unet
        self.layer_filters = [item.strip() for item in layer_filters if item.strip()]
        self.target_height = 0
        self.target_width = 0
        self.records: List[dict] = []
        self.original_attn_processors = dict(unet.attn_processors)
        self.selected_layer_names = [
            name
            for name in self.original_attn_processors.keys()
            if "attn2" in name and self._match_layer(name)
        ]
        if not self.selected_layer_names:
            raise ValueError("未找到符合 attn_layer_filters 的 cross-attention 层。")

        processor_map = {}
        for name, processor in self.original_attn_processors.items():
            if name in self.selected_layer_names:
                processor_map[name] = RecordingAttnProcessor(self, name)
            else:
                processor_map[name] = processor
        self.unet.set_attn_processor(processor_map)

    def _match_layer(self, layer_name: str) -> bool:
        if not self.layer_filters:
            return True
        return any(layer_filter in layer_name for layer_filter in self.layer_filters)

    def close(self) -> None:
        self.unet.set_attn_processor(self.original_attn_processors)

    def set_target_size(self, height: int, width: int) -> None:
        self.target_height = int(height)
        self.target_width = int(width)

    def clear(self) -> None:
        self.records = []

    def record(
        self,
        layer_name: str,
        attention_probs: torch.Tensor,
        batch_size: int,
        query_tokens: int,
        fallback_height: int,
        fallback_width: int,
    ) -> None:
        if query_tokens <= 0:
            return

        if fallback_height > 0 and fallback_width > 0:
            height, width = fallback_height, fallback_width
        else:
            height, width = _infer_spatial_shape(
                query_tokens,
                target_height=max(self.target_height, 1),
                target_width=max(self.target_width, 1),
            )

        heads = max(attention_probs.shape[0] // batch_size, 1)
        reshaped = attention_probs.reshape(batch_size, heads, query_tokens, attention_probs.shape[-1])
        self.records.append(
            {
                "layer_name": layer_name,
                "attention_probs": reshaped,
                "height": height,
                "width": width,
            }
        )

    def aggregate_attention_maps(
        self,
        token_indices: torch.Tensor,
        output_size: Tuple[int, int],
    ) -> torch.Tensor:
        if not self.records:
            raise RuntimeError("当前 forward 未采集到任何 cross-attention 记录。")

        valid_token_indices = token_indices.detach().long()
        aggregated = None

        for record in self.records:
            attention_probs = record["attention_probs"]
            max_token_count = attention_probs.shape[-1]
            masked_indices = valid_token_indices[valid_token_indices < max_token_count]
            if masked_indices.numel() == 0:
                continue

            token_response = attention_probs[:, :, :, masked_indices].mean(dim=-1)
            token_response = token_response.reshape(
                attention_probs.shape[0],
                attention_probs.shape[1],
                record["height"],
                record["width"],
            )
            token_response = token_response.mean(dim=1, keepdim=True)
            token_response = F.interpolate(
                token_response,
                size=output_size,
                mode="bilinear",
                align_corners=False,
            )

            if aggregated is None:
                aggregated = token_response
            else:
                aggregated = aggregated + token_response

        if aggregated is None:
            raise RuntimeError("指定 token_indices 在采集到的 attention map 中均越界。")

        aggregated = aggregated / float(len(self.records))
        flat = aggregated.flatten(2)
        min_vals = flat.min(dim=-1, keepdim=True).values.unsqueeze(-1)
        max_vals = flat.max(dim=-1, keepdim=True).values.unsqueeze(-1)
        denom = (max_vals - min_vals).clamp(min=1e-12)
        normalized = (aggregated - min_vals) / denom
        return normalized[:, 0]


def evaluate_current_loss(
    pipe,
    x: torch.Tensor,
    prompt_state: dict,
    scheduler_train_steps: int,
    is_sdxl: bool,
) -> float:
    """
    评估[当前状态下]的一次损失。

    说明：
    - 这里会按同样的分层策略采样一个时间步 t
    - 并重新采样一次噪声
    - 返回的是更新完成后、当前 x 和当前 prompt_state 下的损失

    注意：
    - 这是[更新后的当前损失]
    - 由于 t 和噪声仍然是随机的，所以数值会有波动
    """
    with torch.no_grad():
        t = sample_stratified_timesteps(
            num_samples=1,
            scheduler_train_steps=scheduler_train_steps,
            device=x.device,
        )

        loss = compute_noise_prediction_loss(
            pipe=pipe,
            x=x.detach(),
            t=t,
            prompt_state=prompt_state,
            is_sdxl=is_sdxl,
        )
    return float(loss)


def resize_keep_aspect_and_pad(
    image: Image.Image,
    target_height: int,
    target_width: int,
    pad_value: int = 127,
) -> Tuple[Image.Image, Dict[str, int]]:
    """
    保持原图长宽比缩放，并居中 padding 到目标尺寸。

    参数：
    - image:
        输入 PIL 图像
    - target_height, target_width:
        目标尺寸
    - pad_value:
        padding 颜色，默认 127（中灰）。
        因为后续会把图像归一化到 [-1, 1]，127 对应接近 0，
        相比纯黑或纯白更[中性]。

    返回：
    - padded_image:
        处理后的目标尺寸图像
    - meta:
        记录缩放和 padding 的元信息，方便调试
    """
    image = image.convert("RGB")
    orig_width, orig_height = image.size

    scale = min(target_width / orig_width, target_height / orig_height)
    new_width = max(1, int(round(orig_width * scale)))
    new_height = max(1, int(round(orig_height * scale)))

    resized = image.resize((new_width, new_height), resample=Image.BICUBIC)

    canvas = Image.new("RGB", (target_width, target_height), (pad_value, pad_value, pad_value))
    left = (target_width - new_width) // 2
    top = (target_height - new_height) // 2
    canvas.paste(resized, (left, top))

    meta = {
        "orig_width": orig_width,
        "orig_height": orig_height,
        "new_width": new_width,
        "new_height": new_height,
        "left": left,
        "top": top,
        "target_width": target_width,
        "target_height": target_height,
    }
    return canvas, meta


def load_image_as_tensor(
    image_path: Path,
    height: int,
    width: int,
    device: torch.device
) -> torch.Tensor:
    """
    读取图像并转成模型输入张量。

    处理流程：
    1. 读取图像并转 RGB
    2. 保持原图长宽比缩放
    3. 居中 padding 到指定宽高
    4. 转成 [1, 3, H, W] 的 float32 张量
    5. 从 [0, 1] 映射到 [-1, 1]
    """
    image = Image.open(image_path).convert("RGB")
    padded_image, meta = resize_keep_aspect_and_pad(
        image=image,
        target_height=height,
        target_width=width,
        pad_value=127,
    )

    arr = np.asarray(padded_image).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    tensor = tensor * 2.0 - 1.0
    return tensor.to(device=device, dtype=torch.float32)


def save_tensor_as_image(image_tensor: torch.Tensor, save_path: Path) -> None:
    """
    将 [-1, 1] 范围的图像张量保存为普通 RGB 图像。

    注意：
    - 保存出来的是[模型输入分辨率下的 padding 后图像]
    - 也就是说，输出图会保留居中 padding 的边框
    """
    tensor = image_tensor.detach().float().cpu().clamp(-1.0, 1.0)
    tensor = (tensor + 1.0) / 2.0
    arr = tensor.squeeze(0).permute(1, 2, 0).numpy()
    arr = (arr * 255.0).round().clip(0, 255).astype(np.uint8)
    Image.fromarray(arr).save(save_path)


def sample_stratified_timesteps(
    num_samples: int,
    scheduler_train_steps: int,
    device: torch.device,
) -> torch.Tensor:
    """
    按固定时间步分布做分层采样。

    规则：
    - 将 [0, scheduler_train_steps) 切成 num_samples 个等宽分桶
    - 每个桶里随机取 1 个时间步
    - 返回一组形状为 [num_samples] 的 t

    这样做的好处是：
    - 每轮梯度累积都会覆盖整个时间轴，而不是纯随机集中在某些区间
    - 保持一定随机性，但采样分布更稳定
    """
    if num_samples <= 0:
        raise ValueError("num_samples 必须 > 0")
    if scheduler_train_steps <= 0:
        raise ValueError("scheduler_train_steps 必须 > 0")

    # 如果采样数大于时间步数，后面的分桶会退化成重复/单点采样，
    # 但仍能保持[分层覆盖]的语义。
    boundaries = torch.linspace(
        0,
        float(scheduler_train_steps),
        steps=num_samples + 1,
        device=device,
    )

    sampled_t = []
    for idx in range(num_samples):
        low = int(torch.floor(boundaries[idx]).item())
        high = int(torch.ceil(boundaries[idx + 1]).item()) - 1

        low = max(0, min(low, scheduler_train_steps - 1))
        high = max(low, min(high, scheduler_train_steps - 1))

        if high == low:
            t = low
        else:
            t = int(
                torch.randint(
                    low=low,
                    high=high + 1,
                    size=(1,),
                    device=device,
                    dtype=torch.long,
                ).item()
            )
        sampled_t.append(t)

    return torch.tensor(sampled_t, device=device, dtype=torch.long)


# ============================================================
# 5. Prompt 解析
# ============================================================
def _normalize_prompt_value(value) -> PromptSpec:
    """
    把不同格式的 prompt 值统一转成 PromptSpec。
    """
    if isinstance(value, str):
        return PromptSpec(prompt=value)

    if isinstance(value, dict):
        if "prompt" not in value:
            raise ValueError("prompt JSON 对象中必须至少包含 'prompt' 字段。")
        return PromptSpec(
            prompt=value["prompt"],
            prompt_2=value.get("prompt_2"),
            negative_prompt=value.get("negative_prompt"),
            negative_prompt_2=value.get("negative_prompt_2"),
        )

    raise ValueError("不支持的 prompt 格式。")


def resolve_prompt_spec(prompt_input: str, image_path: Optional[Path] = None) -> PromptSpec:
    """
    更通用的 prompt 输入解析器。

    支持以下几种方式：

    1) 直接字符串
       --prompt "a photo of a cat"

    2) .txt 文件
       整个文件内容作为共享 prompt

    3) .json 文件
       支持：
       (a) 共享 prompt：
           {"prompt": "a photo of a cat"}

       (b) 按文件名映射：
           {
             "a.jpg": {"prompt": "..."},
             "b.png": {"prompt": "..."},
             "default": {"prompt": "..."}
           }

       (c) 用 "*" 兜底：
           {
             "*": {"prompt": "..."}
           }

    4) .jsonl 文件
       每行一个 JSON 对象，例如：
       {"file": "a.jpg", "prompt": "..."}
       {"default": true, "prompt": "..."}
    """
    prompt_path = Path(prompt_input)

    if not prompt_path.exists():
        return PromptSpec(prompt=prompt_input)

    if prompt_path.suffix.lower() == ".txt":
        text = prompt_path.read_text(encoding="utf-8").strip()
        return PromptSpec(prompt=text)

    if prompt_path.suffix.lower() == ".json":
        data = json.loads(prompt_path.read_text(encoding="utf-8"))

        if isinstance(data, str):
            return PromptSpec(prompt=data)

        if isinstance(data, dict) and "prompt" in data:
            return _normalize_prompt_value(data)

        if image_path is None:
            raise ValueError("当 JSON 使用按文件名映射格式时，必须传入 image_path。")

        key_candidates = [image_path.name, image_path.stem, "default", "*"]
        for key in key_candidates:
            if isinstance(data, dict) and key in data:
                return _normalize_prompt_value(data[key])

        raise ValueError(f"在 JSON prompt 文件中找不到图像 {image_path.name} 对应的 prompt。")

    if prompt_path.suffix.lower() == ".jsonl":
        if image_path is None:
            raise ValueError("当 JSONL 使用按文件名映射格式时，必须传入 image_path。")

        default_item = None
        with prompt_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)

                if item.get("file") in {image_path.name, image_path.stem}:
                    return _normalize_prompt_value(item)

                if item.get("default", False):
                    default_item = item

        if default_item is not None:
            return _normalize_prompt_value(default_item)

        raise ValueError(f"在 JSONL prompt 文件中找不到图像 {image_path.name} 对应的 prompt。")

    raise ValueError("目前只支持字面量字符串、.txt、.json、.jsonl 作为 prompt 输入。")


# ============================================================
# 6. 模型加载
# ============================================================
def load_diffusion_pipeline(model_name: str, device: torch.device):
    """
    根据 model_name 加载对应的 diffusers pipeline。

    说明：
    - 这里只借用 pipeline 的组件（VAE / UNet / text encoder / scheduler）
      来构造训练损失。
    - 不更新模型参数，只更新 prompt embeddings 和图像。
    - 模型参数全部冻结。
    """
    cfg = MODEL_REGISTRY[model_name]
    pipeline_cls = cfg["pipeline_cls"]
    repo_id = cfg["repo_id"]

    torch_dtype = torch.float16 if device.type == "cuda" else torch.float32

    pipe = pipeline_cls.from_pretrained(
        repo_id,
        torch_dtype=torch_dtype,
        safety_checker=None,
        local_files_only=True,
    )
    pipe = pipe.to(device)
    if hasattr(pipe, "safety_checker"):
        pipe.safety_checker = None
    if hasattr(pipe, "set_progress_bar_config"):
        pipe.set_progress_bar_config(disable=True)
    if device.type == "cuda":
        if hasattr(pipe, "enable_attention_slicing"):
            pipe.enable_attention_slicing()
        if hasattr(pipe, "enable_vae_slicing"):
            pipe.enable_vae_slicing()

    pipe.unet.eval()
    pipe.vae.eval()
    if hasattr(pipe, "text_encoder") and pipe.text_encoder is not None:
        pipe.text_encoder.eval()
    if hasattr(pipe, "text_encoder_2") and pipe.text_encoder_2 is not None:
        pipe.text_encoder_2.eval()

    for module_name in ["unet", "vae", "text_encoder", "text_encoder_2"]:
        module = getattr(pipe, module_name, None)
        if module is not None:
            module.requires_grad_(False)

    # 显式换成 DDPMScheduler，用于训练侧 add_noise + 噪声预测损失
    pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)

    return pipe


# ============================================================
# 7. Pseudo-word 初始化
# ============================================================
def init_pseudo_state_hybrid(
    pipe,
    num_pseudo_tokens: int,
    is_sdxl: bool,
    device: torch.device,
    height: int,
    width: int,
) -> dict:
    """hybrid 模式：BASE_PROMPT 经 text_encoder 编码 + N 个随机向量"""
    with torch.no_grad():
        base_spec = PromptSpec(prompt=BASE_PROMPT, prompt_2=BASE_PROMPT)
        if is_sdxl:
            prompt_embeds, _, pooled_prompt_embeds, _ = pipe.encode_prompt(
                prompt=base_spec.prompt, prompt_2=base_spec.prompt_2,
                device=device, num_images_per_prompt=1,
                do_classifier_free_guidance=False,
            )
            add_time_ids = pipe._get_add_time_ids(
                original_size=(height, width), crops_coords_top_left=(0, 0),
                target_size=(height, width), dtype=prompt_embeds.dtype,
                text_encoder_projection_dim=pooled_prompt_embeds.shape[-1],
            ).to(device)
        else:
            prompt_embeds, _ = pipe.encode_prompt(
                prompt=base_spec.prompt, device=device,
                num_images_per_prompt=1, do_classifier_free_guidance=False,
            )

    # 找到 base 文本的有效 token 数 (去掉 BOS/padding)
    tokenizer = pipe.tokenizer
    base_ids = tokenizer(BASE_PROMPT, add_special_tokens=False).input_ids
    base_len = len(base_ids)  # token 数，不含 BOS/padding
    # BOS 在位置 0，base 文本从位置 1 开始
    base_end = 1 + base_len  # BOS + base tokens

    dim = prompt_embeds.shape[-1]
    pseudo_embeds = torch.randn(1, num_pseudo_tokens, dim, device=device) * 0.02

    # 构造完整 prompt_embeds: [BOS, base, pseudo, padding]
    full_len = prompt_embeds.shape[1]
    pad_start = base_end + num_pseudo_tokens
    prompt_embeds_out = prompt_embeds.detach().float().clone()
    prompt_embeds_out[:, base_end:pad_start, :] = pseudo_embeds
    # padding 之后的部分保持 text_encoder 的原始 padding 值

    learnable_token_indices = torch.arange(base_end, pad_start, device=device, dtype=torch.long)

    state = {
        "prompt_embeds": prompt_embeds_out,
        "prompt_text": BASE_PROMPT,
        "learnable_token_indices": learnable_token_indices,
        "pseudo_init_mode": "hybrid",
        "num_pseudo_tokens": num_pseudo_tokens,
    }
    if is_sdxl:
        state["pooled_prompt_embeds"] = pooled_prompt_embeds.detach().float().clone()
        state["add_time_ids"] = add_time_ids.detach().float().clone()
    return state


def _tokenize_without_special_tokens(tokenizer, text: str) -> List[int]:
    encoded = tokenizer(text, add_special_tokens=False)
    input_ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
    if input_ids and isinstance(input_ids[0], list):
        if len(input_ids) != 1:
            raise ValueError("仅支持单条文本的 token 编码。")
        input_ids = input_ids[0]
    return list(input_ids)


def _find_subsequence(sequence: List[int], pattern: List[int]) -> Optional[int]:
    if not pattern or len(pattern) > len(sequence):
        return None
    limit = len(sequence) - len(pattern) + 1
    for start in range(limit):
        if sequence[start:start + len(pattern)] == pattern:
            return start
    return None


def resolve_token_indices(tokenizer, prompt_text: str, target_text: str) -> List[int]:
    """
    找到 prompt 中某个目标词对应的编码位置。

    这里返回的是[编码后的 token 位置], 不是字符位置。
    如果 tokenizer 把目标词拆成多个 sub-token，
    则这些位置都会被视为可学习部分。
    """
    if target_text not in prompt_text:
        raise ValueError(f"prompt 中未找到目标词：{target_text}")

    full_ids_no_special = _tokenize_without_special_tokens(tokenizer, prompt_text)
    full_ids_with_special = tokenizer(
        prompt_text,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).input_ids[0].tolist()

    full_start = _find_subsequence(full_ids_with_special, full_ids_no_special)
    if full_start is None:
        raise ValueError("无法在带 special tokens 的序列中定位 prompt 主体。")

    candidate_texts = [f" {target_text}", target_text]
    for candidate_text in candidate_texts:
        candidate_ids = _tokenize_without_special_tokens(tokenizer, candidate_text)
        start = _find_subsequence(full_ids_no_special, candidate_ids)
        if start is not None:
            return list(
                range(
                    full_start + start,
                    full_start + start + len(candidate_ids),
                )
            )

    raise ValueError(
        f"无法在 prompt={prompt_text!r} 中定位目标词 {target_text!r} 对应的编码位置。"
    )




def compose_prompt_embeds(
    prompt_embeds_base: torch.Tensor,
    learnable_prompt_embeds: torch.Tensor,
    learnable_token_indices: torch.Tensor,
) -> torch.Tensor:
    prompt_embeds = prompt_embeds_base.clone()
    prompt_embeds[:, learnable_token_indices, :] = learnable_prompt_embeds
    return prompt_embeds


def init_prompt_state_from_text(
    pipe,
    prompt_spec: PromptSpec,
    is_sdxl: bool,
    device: torch.device,
    height: int,
    width: int,
) -> dict:
    """根据文本 prompt 初始化 prompt 状态。用于 e_0 原始类别 prompt。"""
    with torch.no_grad():
        if is_sdxl:
            prompt_2 = prompt_spec.prompt_2 if prompt_spec.prompt_2 is not None else prompt_spec.prompt
            prompt_embeds, _, pooled_prompt_embeds, _ = pipe.encode_prompt(
                prompt=prompt_spec.prompt, prompt_2=prompt_2,
                device=device, num_images_per_prompt=1,
                do_classifier_free_guidance=False,
            )
            add_time_ids = pipe._get_add_time_ids(
                original_size=(height, width), crops_coords_top_left=(0, 0),
                target_size=(height, width), dtype=prompt_embeds.dtype,
                text_encoder_projection_dim=pooled_prompt_embeds.shape[-1],
            ).to(device)
            return {
                "prompt_embeds": prompt_embeds.detach().float().clone(),
                "pooled_prompt_embeds": pooled_prompt_embeds.detach().float().clone(),
                "add_time_ids": add_time_ids.detach().float().clone(),
                "prompt_text": prompt_spec.prompt,
                "prompt_text_2": prompt_2,
            }

        prompt_embeds, _ = pipe.encode_prompt(
            prompt=prompt_spec.prompt, device=device,
            num_images_per_prompt=1, do_classifier_free_guidance=False,
        )
        return {
            "prompt_embeds": prompt_embeds.detach().float().clone(),
            "prompt_text": prompt_spec.prompt,
        }


def save_prompt_state(prompt_state: dict, save_path: Path) -> None:
    """
    保存优化后的 prompt 状态。

    注意：
    - 由于优化后的是连续 embeddings，不再是可读文本，
      所以这里将它保存为 .pt 张量文件。
    """
    to_save = {}
    for k, v in prompt_state.items():
        if torch.is_tensor(v):
            to_save[k] = v.detach().cpu()
        else:
            to_save[k] = v
    torch.save(to_save, save_path)


def collate_prompt_states_for_batch(prompt_states: List[dict], is_sdxl: bool) -> dict:
    """
    将多张图像对应的 prompt state 拼成 batch 形式。
    """
    if not prompt_states:
        raise ValueError("prompt_states 不能为空。")

    batch_state = {
        "prompt_embeds": torch.cat(
            [state["prompt_embeds"].detach().clone() for state in prompt_states],
            dim=0,
        ),
    }

    if is_sdxl:
        batch_state["pooled_prompt_embeds"] = torch.cat(
            [state["pooled_prompt_embeds"].detach().clone() for state in prompt_states],
            dim=0,
        )
        batch_state["add_time_ids"] = torch.cat(
            [state["add_time_ids"].detach().clone() for state in prompt_states],
            dim=0,
        )

    return batch_state


def snapshot_prompt_state(
    prompt_embeds: torch.Tensor,
    is_sdxl: bool,
    prompt_text: str,
    pooled_prompt_embeds: Optional[torch.Tensor] = None,
    add_time_ids: Optional[torch.Tensor] = None,
    prompt_text_2: Optional[str] = None,
    extra_state: Optional[dict] = None,
) -> dict:
    """
    把当前 prompt 参数快照成普通张量字典，便于评估与保存。
    """
    state = {
        "prompt_embeds": prompt_embeds.detach().clone(),
        "prompt_text": prompt_text,
    }
    if prompt_text_2 is not None:
        state["prompt_text_2"] = prompt_text_2
    if is_sdxl:
        if pooled_prompt_embeds is None or add_time_ids is None:
            raise ValueError("SDXL 模式下必须同时提供 pooled_prompt_embeds 和 add_time_ids。")
        state["pooled_prompt_embeds"] = pooled_prompt_embeds.detach().clone()
        state["add_time_ids"] = add_time_ids.detach().clone()
    if extra_state is not None:
        for k, v in extra_state.items():
            if torch.is_tensor(v):
                state[k] = v.detach().clone()
            else:
                state[k] = v
    return state


def compute_prompt_regularization(
    prompt_embeds: torch.Tensor,
    prompt_embeds_ref: torch.Tensor,
    is_sdxl: bool,
    pooled_prompt_embeds: Optional[torch.Tensor] = None,
    pooled_prompt_embeds_ref: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    计算 prompt 参数相对初始值的 L2 正则项。

    目的：
    - 让优化后的 prompt embeddings 不要偏离初始 embeddings 太远
    - 降低 prompt 阶段跑到过于异常的连续空间区域的风险
    """
    reg = F.mse_loss(prompt_embeds, prompt_embeds_ref, reduction="mean")

    if is_sdxl:
        if pooled_prompt_embeds is None or pooled_prompt_embeds_ref is None:
            raise ValueError("SDXL 模式下必须同时提供 pooled_prompt_embeds 及其参考值。")
        reg = reg + F.mse_loss(
            pooled_prompt_embeds,
            pooled_prompt_embeds_ref,
            reduction="mean",
        )

    return reg


def _split_batched_prompt_state(
    prompt_state_batch: dict,
    prompt_state_templates: List[dict],
    is_sdxl: bool,
) -> List[dict]:
    """
    将 batch 形式的 prompt state 拆回逐样本字典，便于保存。
    """
    batch_size = prompt_state_batch["prompt_embeds"].shape[0]
    if batch_size != len(prompt_state_templates):
        raise ValueError("prompt_state_batch 与 prompt_state_templates 的 batch 大小不一致。")

    split_states: List[dict] = []
    for idx in range(batch_size):
        template = prompt_state_templates[idx]
        state = {
            "prompt_embeds": prompt_state_batch["prompt_embeds"][idx:idx + 1].detach().clone(),
            "prompt_text": template.get("prompt_text", ""),
            "learnable_token_indices": template["learnable_token_indices"].detach().clone(),
            "pseudo_init_mode": template.get("pseudo_init_mode", "hybrid"),
            "num_pseudo_tokens": template.get("num_pseudo_tokens", 0),
        }
        if "prompt_text_2" in template:
            state["prompt_text_2"] = template["prompt_text_2"]
        if is_sdxl:
            state["pooled_prompt_embeds"] = (
                prompt_state_batch["pooled_prompt_embeds"][idx:idx + 1].detach().clone()
            )
            state["add_time_ids"] = (
                prompt_state_batch["add_time_ids"][idx:idx + 1].detach().clone()
            )
        split_states.append(state)

    return split_states


# ============================================================
# 9. 单次前向与损失计算
# ============================================================
def _match_condition_batch_size(tensor: torch.Tensor, batch_size: int, name: str) -> torch.Tensor:
    """
    让条件张量的 batch 维与图像 batch 对齐。
    """
    if tensor.shape[0] == batch_size:
        return tensor
    if tensor.shape[0] == 1:
        expand_shape = [batch_size] + [-1] * (tensor.dim() - 1)
        return tensor.expand(*expand_shape)
    raise ValueError(
        f"{name} 的 batch 大小为 {tensor.shape[0]}，但图像 batch 大小为 {batch_size}，无法对齐。"
    )


def compute_noise_prediction_loss(
    pipe,
    x: torch.Tensor,
    t: torch.Tensor,
    prompt_state: dict,
    is_sdxl: bool,
    noise: Optional[torch.Tensor] = None,
    deterministic_vae: bool = False,
) -> torch.Tensor:
    """
    计算一次[给定图像 x、给定 prompt_state、给定扩散时间步 t]下的噪声预测损失。

    新增参数：
    - noise:
        如果传入，则使用固定噪声；否则内部随机采样噪声。
    - deterministic_vae:
        是否使用确定性的 VAE latent（posterior.mean）。
        评估日志时建议设为 True，减少方差。
    """
    model_dtype = next(pipe.unet.parameters()).dtype
    model_device = x.device
    batch_size = x.shape[0]

    x_model = x.to(dtype=model_dtype)
    # Infer target size from VAE latent spatial dims
    posterior = pipe.vae.encode(x_model).latent_dist
    if deterministic_vae:
        latents = posterior.mean * pipe.vae.config.scaling_factor
    else:
        latents = posterior.sample() * pipe.vae.config.scaling_factor

    if noise is None:
        noise = torch.randn_like(latents)
    else:
        noise = noise.to(device=latents.device, dtype=latents.dtype)

    noisy_latents = pipe.scheduler.add_noise(latents, noise, t)

    prompt_embeds = prompt_state["prompt_embeds"].to(device=model_device, dtype=model_dtype)
    prompt_embeds = _match_condition_batch_size(prompt_embeds, batch_size, "prompt_embeds")

    if is_sdxl:
        pooled_prompt_embeds = prompt_state["pooled_prompt_embeds"].to(
            device=model_device, dtype=model_dtype
        )
        pooled_prompt_embeds = _match_condition_batch_size(
            pooled_prompt_embeds, batch_size, "pooled_prompt_embeds"
        )
        add_time_ids = prompt_state["add_time_ids"].to(device=model_device, dtype=model_dtype)
        add_time_ids = _match_condition_batch_size(add_time_ids, batch_size, "add_time_ids")

        added_cond_kwargs = {
            "text_embeds": pooled_prompt_embeds,
            "time_ids": add_time_ids,
        }

        noise_pred = pipe.unet(
            noisy_latents,
            t,
            encoder_hidden_states=prompt_embeds,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=False,
        )[0]
    else:
        noise_pred = pipe.unet(
            noisy_latents,
            t,
            encoder_hidden_states=prompt_embeds,
            return_dict=False,
        )[0]

    prediction_type = pipe.scheduler.config.prediction_type
    if prediction_type == "epsilon":
        target = noise
    elif prediction_type == "v_prediction":
        target = pipe.scheduler.get_velocity(latents, noise, t)
    elif prediction_type == "sample":
        target = latents
    else:
        raise ValueError(f"当前脚本暂不支持 prediction_type={prediction_type}")

    loss = F.mse_loss(noise_pred.float(), target.float(), reduction="mean")
    return loss


def compute_noise_prediction_loss_and_attention(
    pipe,
    x: torch.Tensor,
    t: torch.Tensor,
    prompt_state: dict,
    attention_token_indices: torch.Tensor,
    attention_recorder: CrossAttentionMapRecorder,
    is_sdxl: bool,
    noise: Optional[torch.Tensor] = None,
    deterministic_vae: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    在同一次 U-Net 前向中，同时返回噪声预测损失与目标 token 的 attention 聚合图。
    """
    model_dtype = next(pipe.unet.parameters()).dtype
    model_device = x.device
    batch_size = x.shape[0]

    x_model = x.to(dtype=model_dtype)

    posterior = pipe.vae.encode(x_model).latent_dist
    if deterministic_vae:
        latents = posterior.mean * pipe.vae.config.scaling_factor
    else:
        latents = posterior.sample() * pipe.vae.config.scaling_factor

    if noise is None:
        noise = torch.randn_like(latents)
    else:
        noise = noise.to(device=latents.device, dtype=latents.dtype)

    noisy_latents = pipe.scheduler.add_noise(latents, noise, t)

    prompt_embeds = prompt_state["prompt_embeds"].to(device=model_device, dtype=model_dtype)
    prompt_embeds = _match_condition_batch_size(prompt_embeds, batch_size, "prompt_embeds")

    heatmap_size = (DEFAULT_ATTN_SAVE_SIZE, DEFAULT_ATTN_SAVE_SIZE)
    attention_recorder.set_target_size(*heatmap_size)
    attention_recorder.clear()

    if is_sdxl:
        pooled_prompt_embeds = prompt_state["pooled_prompt_embeds"].to(
            device=model_device, dtype=model_dtype
        )
        pooled_prompt_embeds = _match_condition_batch_size(
            pooled_prompt_embeds, batch_size, "pooled_prompt_embeds"
        )
        add_time_ids = prompt_state["add_time_ids"].to(device=model_device, dtype=model_dtype)
        add_time_ids = _match_condition_batch_size(add_time_ids, batch_size, "add_time_ids")

        added_cond_kwargs = {
            "text_embeds": pooled_prompt_embeds,
            "time_ids": add_time_ids,
        }

        noise_pred = pipe.unet(
            noisy_latents,
            t,
            encoder_hidden_states=prompt_embeds,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=False,
        )[0]
    else:
        noise_pred = pipe.unet(
            noisy_latents,
            t,
            encoder_hidden_states=prompt_embeds,
            return_dict=False,
        )[0]

    prediction_type = pipe.scheduler.config.prediction_type
    if prediction_type == "epsilon":
        target = noise
    elif prediction_type == "v_prediction":
        target = pipe.scheduler.get_velocity(latents, noise, t)
    elif prediction_type == "sample":
        target = latents
    else:
        raise ValueError(f"当前脚本暂不支持 prediction_type={prediction_type}")

    loss = F.mse_loss(noise_pred.float(), target.float(), reduction="mean")
    attention_map = attention_recorder.aggregate_attention_maps(
        token_indices=attention_token_indices.to(device=model_device),
        output_size=heatmap_size,
    )
    return loss, attention_map




def evaluate_before_after_loss_on_same_sample(
    pipe,
    x_before: torch.Tensor,
    prompt_state_before: dict,
    x_after: torch.Tensor,
    prompt_state_after: dict,
    scheduler_train_steps: int,
    is_sdxl: bool,
) -> Tuple[float, float]:
    """
    在同一个评估样本下，同时计算更新前 loss 和更新后 loss。

    这里[同一个评估样本]指：
    - 同一个时间步 t
    - 同一个 noise
    - 确定性的 VAE latent（posterior.mean）

    这样 before / after 才具有可比性。
    """
    with torch.no_grad():
        # 先用更新前状态得到 latent shape，用于构造固定噪声
        model_dtype = next(pipe.unet.parameters()).dtype
        x_before_model = x_before.to(dtype=model_dtype)
        posterior = pipe.vae.encode(x_before_model).latent_dist
        latents_before = posterior.mean * pipe.vae.config.scaling_factor

        t = sample_stratified_timesteps(
            num_samples=1,
            scheduler_train_steps=scheduler_train_steps,
            device=x_before.device,
        )
        fixed_noise = torch.randn_like(latents_before)

        loss_before = compute_noise_prediction_loss(
            pipe=pipe,
            x=x_before.detach(),
            t=t,
            prompt_state=prompt_state_before,
            is_sdxl=is_sdxl,
            noise=fixed_noise,
            deterministic_vae=True,
        )

        loss_after = compute_noise_prediction_loss(
            pipe=pipe,
            x=x_after.detach(),
            t=t,
            prompt_state=prompt_state_after,
            is_sdxl=is_sdxl,
            noise=fixed_noise,
            deterministic_vae=True,
        )

    return float(loss_before.cpu().item()), float(loss_after.cpu().item())


def evaluate_attention_maps_on_same_sample(
    pipe,
    x_a: torch.Tensor,
    prompt_state_a: dict,
    attention_token_indices_a: torch.Tensor,
    x_b: torch.Tensor,
    prompt_state_b: dict,
    attention_token_indices_b: torch.Tensor,
    scheduler_train_steps: int,
    attention_recorder: CrossAttentionMapRecorder,
    is_sdxl: bool,
    fixed_t: Optional[torch.Tensor] = None,
    fixed_noise: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    在同一个评估样本下导出两组可比较的 attention 热图。
    """
    with torch.no_grad():
        model_dtype = next(pipe.unet.parameters()).dtype
        x_a_model = x_a.to(dtype=model_dtype)
        posterior = pipe.vae.encode(x_a_model).latent_dist
        latents_a = posterior.mean * pipe.vae.config.scaling_factor

        if fixed_t is None:
            t = sample_stratified_timesteps(
                num_samples=1,
                scheduler_train_steps=scheduler_train_steps,
                device=x_a.device,
            )
        else:
            t = fixed_t.to(device=x_a.device)

        if fixed_noise is None:
            eval_noise = torch.randn_like(latents_a)
        else:
            eval_noise = fixed_noise.to(device=latents_a.device, dtype=latents_a.dtype)

        _, attention_map_a = compute_noise_prediction_loss_and_attention(
            pipe=pipe,
            x=x_a.detach(),
            t=t,
            prompt_state=prompt_state_a,
            attention_token_indices=attention_token_indices_a,
            attention_recorder=attention_recorder,
            is_sdxl=is_sdxl,
            noise=eval_noise,
            deterministic_vae=True,
        )
        _, attention_map_b = compute_noise_prediction_loss_and_attention(
            pipe=pipe,
            x=x_b.detach(),
            t=t,
            prompt_state=prompt_state_b,
            attention_token_indices=attention_token_indices_b,
            attention_recorder=attention_recorder,
            is_sdxl=is_sdxl,
            noise=eval_noise,
            deterministic_vae=True,
        )

    return attention_map_a.detach().cpu(), attention_map_b.detach().cpu()


def build_shared_attention_eval_sample(
    pipe,
    x_reference: torch.Tensor,
    scheduler_train_steps: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    为 attention 可视化预生成同一组评估条件，便于跨阶段直接比较。
    """
    with torch.no_grad():
        vae_dtype = next(pipe.vae.parameters()).dtype
        posterior = pipe.vae.encode(x_reference.to(dtype=vae_dtype)).latent_dist
        latents = posterior.mean * pipe.vae.config.scaling_factor
        t = sample_stratified_timesteps(
            num_samples=1,
            scheduler_train_steps=scheduler_train_steps,
            device=x_reference.device,
        )
        noise = torch.randn_like(latents)
    return t.detach().clone(), noise.detach().clone()


def compute_attention_intensity(
    attention_recorder: CrossAttentionMapRecorder,
    token_indices: torch.Tensor,
) -> torch.Tensor:
    """
    A = 1/|L| * sum_l 1/|H_l| * sum_h 1/S_l * sum_s P_s^{(l,h)}(x,t,e)
    对指定 token_indices 取平均。返回标量。
    """
    total_intensity = None
    num_layers = 0

    for record in attention_recorder.records:
        layer_name = record["layer_name"]
        if not any(target in layer_name for target in CROSS_ATTN_INTENSITY_LAYERS):
            continue

        attention_probs = record["attention_probs"]  # [B, H, S, K]
        valid = token_indices[token_indices < attention_probs.shape[-1]]
        if valid.numel() == 0:
            continue

        token_attn = attention_probs[:, :, :, valid].mean(dim=-1)  # [B, H, S]
        spatial_mean = token_attn.mean(dim=-1)  # [B, H]
        head_mean = spatial_mean.mean(dim=-1)  # [B]
        layer_intensity = head_mean.mean()  # scalar tensor with grad

        total_intensity = layer_intensity if total_intensity is None else total_intensity + layer_intensity
        num_layers += 1

    if num_layers == 0:
        raise RuntimeError("未找到匹配的 cross-attention 记录。")

    return total_intensity / float(num_layers)


# ============================================================
# 10. 阶段一：更新 prompt embeddings，使损失最大化
# ============================================================
def optimize_prompt_embeddings_for_batch(
    pipe,
    x_batch: torch.Tensor,
    prompt_states_init: List[dict],
    lr_p: float,
    attn_intensity_weight: float,
    timesteps: int,
    grad_accum: int,
    scheduler_train_steps: int,
    attention_recorder: CrossAttentionMapRecorder,
    metrics_log_path: Path,
    is_sdxl: bool,
    attention_eval_t: Optional[torch.Tensor] = None,
    attention_eval_noise: Optional[torch.Tensor] = None,
) -> Tuple[List[dict], Dict[str, torch.Tensor]]:
    """
    对一批图像并行执行[prompt embeddings 更新阶段]。

    目标：
    - 固定图像 batch
    - 更新每张图像各自对应的 prompt embeddings
    - 通过 Adam 梯度上升最大化 batch 平均损失

    对应你的步骤：
    1. 输入图像 x 和 prompt，按时间步分层采样得到一组 t，计算损失
    2. 对 prompt 求梯度并做梯度累积
    3. 达到 grad_accum 后，用 Adam 做一次参数更新
    4. 重复直到采样次数达到 timesteps

    关键说明：
    - 这里[覆盖原 prompt]指的是覆盖当前使用的连续 prompt embeddings，
      并不是覆盖文本字符串。
    - 当前版本只允许 `<noise>` 对应的编码位置更新；
      其它 token 位置和 SDXL 的 pooled_prompt_embeds 都保持固定。
    """
    if not prompt_states_init:
        raise ValueError("prompt_states_init 不能为空。")

    x_fixed = x_batch.detach()
    batch_size = x_fixed.shape[0]
    if len(prompt_states_init) != batch_size:
        raise ValueError("prompt_states_init 数量必须与图像 batch 大小一致。")

    learnable_token_indices = (
        prompt_states_init[0]["learnable_token_indices"].detach().clone().long()
    )
    for prompt_state in prompt_states_init[1:]:
        if not torch.equal(
            prompt_state["learnable_token_indices"].detach().cpu(),
            learnable_token_indices.detach().cpu(),
        ):
            raise ValueError("当前 batch 内所有样本的 learnable_token_indices 必须一致。")


    prompt_embeds_base = torch.cat(
        [state["prompt_embeds"].detach().clone().float() for state in prompt_states_init],
        dim=0,
    )
    prompt_embeds_ref = prompt_embeds_base.detach().clone()
    prompt_embeds_param = torch.nn.Parameter(
        prompt_embeds_base[:, learnable_token_indices, :].detach().clone()
    )
    pooled_prompt_embeds_fixed = None
    pooled_prompt_embeds_ref = None
    add_time_ids = None

    optim_params = [prompt_embeds_param]
    if is_sdxl:
        pooled_prompt_embeds_fixed = torch.cat(
            [
                state["pooled_prompt_embeds"].detach().clone().float()
                for state in prompt_states_init
            ],
            dim=0,
        )
        pooled_prompt_embeds_ref = pooled_prompt_embeds_fixed.detach().clone()
        add_time_ids = torch.cat(
            [state["add_time_ids"].detach().clone().float() for state in prompt_states_init],
            dim=0,
        )
    optimizer = torch.optim.Adam(optim_params, lr=lr_p)

    total_sample_count = 0
    update_round = 0



    # ---- Stage 1 before 热力图 ----
    
    with torch.no_grad():
        _, stage1_before_map = compute_noise_prediction_loss_and_attention(
            pipe=pipe,
            x=x_fixed,
            t=attention_eval_t if attention_eval_t is not None else sample_stratified_timesteps(
                num_samples=1, scheduler_train_steps=scheduler_train_steps, device=x_fixed.device,
            ),
            prompt_state=collate_prompt_states_for_batch(prompt_states_init, is_sdxl=is_sdxl),
            attention_token_indices=learnable_token_indices,
            attention_recorder=attention_recorder,
            is_sdxl=is_sdxl,
            noise=attention_eval_noise if attention_eval_noise is not None else torch.randn_like(
                pipe.vae.encode(x_fixed.to(dtype=next(pipe.unet.parameters()).dtype)).latent_dist.mean
                * pipe.vae.config.scaling_factor
            ),
            deterministic_vae=True,
        )
    stage1_before_map = stage1_before_map.detach().cpu()

    pbar_stage1 = tqdm(total=timesteps, desc="Stage 1 (prompt)", unit="t")
    while total_sample_count < timesteps:
        update_round += 1
        diff_values = []
        att_values = []
        total_values = []

        timestep_group = sample_stratified_timesteps(
            num_samples=grad_accum,
            scheduler_train_steps=scheduler_train_steps,
            device=x_fixed.device,
        )

        optimizer.zero_grad(set_to_none=True)

        for t in timestep_group:
            model_dtype = next(pipe.unet.parameters()).dtype
            latent_ref = pipe.vae.encode(x_fixed.to(dtype=model_dtype)).latent_dist.mean
            latent_ref = latent_ref * pipe.vae.config.scaling_factor
            fixed_noise = torch.randn_like(latent_ref)

            work_prompt_embeds = compose_prompt_embeds(
                prompt_embeds_base=prompt_embeds_base,
                learnable_prompt_embeds=prompt_embeds_param,
                learnable_token_indices=learnable_token_indices,
            )
            work_state = {
                "prompt_embeds": work_prompt_embeds,
            }

            if is_sdxl:
                work_state["pooled_prompt_embeds"] = pooled_prompt_embeds_fixed
                work_state["add_time_ids"] = add_time_ids

            diff_loss, _ = compute_noise_prediction_loss_and_attention(
                pipe=pipe,
                x=x_fixed,
                t=t,
                prompt_state=work_state,
                attention_token_indices=learnable_token_indices,
                attention_recorder=attention_recorder,
                is_sdxl=is_sdxl,
                noise=fixed_noise,
                deterministic_vae=True,
            )
            intensity_val = compute_attention_intensity(
                attention_recorder=attention_recorder,
                token_indices=learnable_token_indices,
            )
            diff_values.append(float(diff_loss))
            att_values.append(float(intensity_val))
            objective = -diff_loss + attn_intensity_weight * intensity_val
            total_values.append(float(objective))
            (objective / float(grad_accum)).backward()
            total_sample_count += 1

        if prompt_embeds_param.grad is None:
            raise RuntimeError("未能得到 prompt_embeds 的梯度，请检查计算图是否被意外截断。")

        prompt_embeds_param.grad = torch.nan_to_num(prompt_embeds_param.grad)

        optimizer.step()

        mean_diff = sum(diff_values) / len(diff_values)
        mean_att = sum(att_values) / len(att_values) if att_values else 0.0
        pbar_stage1.set_postfix(loss=f"{mean_diff:.4f}", attn=f"{mean_att:.4f}")
        append_jsonl_record(metrics_log_path, {
            "stage": "stage1", "update": update_round, "sampled": total_sample_count,
            "L_cond": mean_diff, "A_intensity": mean_att,
        })
        pbar_stage1.update(len(timestep_group))

    pbar_stage1.close()
    optimized_prompt_state_batch = snapshot_prompt_state(
        prompt_embeds=compose_prompt_embeds(
            prompt_embeds_base=prompt_embeds_base,
            learnable_prompt_embeds=prompt_embeds_param,
            learnable_token_indices=learnable_token_indices,
        ),
        pooled_prompt_embeds=pooled_prompt_embeds_fixed,
        add_time_ids=add_time_ids,
        is_sdxl=is_sdxl,
        prompt_text="",
        prompt_text_2=None,
    )
    optimized_prompt_states = _split_batched_prompt_state(
        optimized_prompt_state_batch,
        prompt_state_templates=prompt_states_init,
        is_sdxl=is_sdxl,
    )
    with torch.no_grad():
        model_dtype = next(pipe.unet.parameters()).dtype
        posterior = pipe.vae.encode(x_fixed.to(dtype=model_dtype)).latent_dist
        latents = posterior.mean * pipe.vae.config.scaling_factor
        eval_t = (
            attention_eval_t.to(device=x_fixed.device)
            if attention_eval_t is not None
            else sample_stratified_timesteps(
                num_samples=1,
                scheduler_train_steps=scheduler_train_steps,
                device=x_fixed.device,
            )
        )
        eval_noise = (
            attention_eval_noise.to(device=latents.device, dtype=latents.dtype)
            if attention_eval_noise is not None
            else torch.randn_like(latents)
        )
        _, stage1_mis_map = compute_noise_prediction_loss_and_attention(
            pipe=pipe,
            x=x_fixed,
            t=eval_t,
            prompt_state=collate_prompt_states_for_batch(optimized_prompt_states, is_sdxl=is_sdxl),
            attention_token_indices=learnable_token_indices,
            attention_recorder=attention_recorder,
            is_sdxl=is_sdxl,
            noise=eval_noise,
            deterministic_vae=True,
        )
    return optimized_prompt_states, {
        "pseudo_before": stage1_before_map,
        "pseudo_after": stage1_mis_map.detach().cpu(),
    }


# ============================================================
# 11. Surrogate UNet 训练
# ============================================================
def train_surrogate_unet(
    pipe,
    instance_images: torch.Tensor,
    instance_prompt: str,
    class_images_dir: str,
    class_prompt: str,
    num_steps: int = 20,
    learning_rate: float = 5e-6,
    prior_loss_weight: float = 1.0,
    num_class_images: int = 200,
    resolution: int = 512,
) -> torch.nn.Module:
    """
    Deep-copy UNet 并在 set_A 上做 DreamBooth 风格 fine-tune。
    仅更新 UNet 参数，text_encoder / VAE 保持冻结。

    参照 ASPL 的 train_one_epoch。
    """
    surrogate_unet = copy.deepcopy(pipe.unet)
    surrogate_unet.train()
    surrogate_unet.requires_grad_(True)

    optimizer = torch.optim.AdamW(
        surrogate_unet.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-08,
    )

    model_dtype = torch.bfloat16  # 与 ASPL 一致，全模型统一 bfloat16
    device = instance_images.device

    # 加载 class images
    class_dir_path = Path(class_images_dir)
    if not class_dir_path.exists() or not class_dir_path.is_dir():
        raise ValueError(f"surrogate_class_dir 不存在: {class_dir_path}")
    class_image_paths = sorted(
        [p for p in class_dir_path.iterdir() if p.is_file() and p.suffix.lower() in ALLOWED_IMAGE_EXTS]
    )[:num_class_images]
    if not class_image_paths:
        raise ValueError(f"surrogate_class_dir 中没有找到图像: {class_dir_path}")

    # 统一转 bfloat16 (与 ASPL 一致)，原始 dtype 由外层保存
    pipe.vae.to(device, dtype=model_dtype)
    pipe.text_encoder.to(device, dtype=model_dtype)
    pipe.vae.eval()
    pipe.vae.requires_grad_(False)
    pipe.unet.eval()

    surrogate_unet.to(device, dtype=model_dtype)
    instance_images_batch = instance_images.to(device, dtype=model_dtype)

    pbar = tqdm(range(num_steps), desc="Surrogate", unit="step")
    for step in pbar:
        surrogate_unet.train()

        # 加载一张 class image
        class_image = Image.open(class_image_paths[step % len(class_image_paths)]).convert("RGB")
        class_image = class_image.resize((resolution, resolution))
        class_arr = np.asarray(class_image).astype(np.float32) / 255.0
        class_tensor = torch.from_numpy(class_arr).permute(2, 0, 1).unsqueeze(0)
        class_tensor = class_tensor * 2.0 - 1.0
        class_tensor = class_tensor.to(device, dtype=model_dtype)

        # 随机取一张 instance image
        inst_idx = step % len(instance_images_batch)
        inst_img = instance_images_batch[inst_idx:inst_idx + 1]

        # batch: [instance, class]
        pixel_values = torch.cat([inst_img, class_tensor], dim=0)
        latents = pipe.vae.encode(pixel_values).latent_dist.sample()
        latents = latents * pipe.vae.config.scaling_factor

        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        timesteps = torch.randint(
            0, pipe.scheduler.config.num_train_timesteps, (bsz,), device=latents.device
        ).long()
        noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)

        # encode prompts
        with torch.no_grad():
            inst_ids = pipe.tokenizer(
                instance_prompt,
                truncation=True,
                padding="max_length",
                max_length=pipe.tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids.to(device)
            class_ids = pipe.tokenizer(
                class_prompt,
                truncation=True,
                padding="max_length",
                max_length=pipe.tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids.to(device)
            input_ids = torch.cat([inst_ids, class_ids], dim=0)
            encoder_hidden_states = pipe.text_encoder(input_ids)[0]

        model_pred = surrogate_unet(noisy_latents, timesteps, encoder_hidden_states).sample

        if pipe.scheduler.config.prediction_type == "epsilon":
            target = noise
        elif pipe.scheduler.config.prediction_type == "v_prediction":
            target = pipe.scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"不支持的 prediction_type: {pipe.scheduler.config.prediction_type}")

        model_pred_inst, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
        target_inst, target_prior = torch.chunk(target, 2, dim=0)

        instance_loss = F.mse_loss(model_pred_inst.float(), target_inst.float(), reduction="mean")
        prior_loss = F.mse_loss(model_pred_prior.float(), target_prior.float(), reduction="mean")
        loss = instance_loss + prior_loss_weight * prior_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(surrogate_unet.parameters(), 1.0, error_if_nonfinite=False)
        optimizer.step()

        pbar.set_postfix(loss=f"{loss.item():.4f}", inst=f"{instance_loss.item():.3f}", prior=f"{prior_loss.item():.3f}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    surrogate_unet.eval()
    surrogate_unet.requires_grad_(False)
    # 不恢复 VAE/TE dtype — Stage 2 用 surrogate 时需要全 bf16 对齐
    return surrogate_unet


# ============================================================
# 12. 阶段二：更新图像，使损失最小化
# ============================================================
def optimize_image_with_fixed_prompt_for_batch(
    pipe,
    x_init_batch: torch.Tensor,
    prompt_state_star_batch: dict,
    attention_token_indices_star: torch.Tensor,
    prompt_state_original_batch: dict,
    attention_token_indices_original: torch.Tensor,
    lr: float,
    pgd_eps: float,
    stage2_gamma: float,
    attn_intensity_weight: float,
    timesteps: int,
    grad_accum: int,
    scheduler_train_steps: int,
    attention_recorder: CrossAttentionMapRecorder,
    metrics_log_path: Path,
    is_sdxl: bool,
    attention_eval_t: Optional[torch.Tensor] = None,
    attention_eval_noise: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    对一批图像执行 PGD 图像更新阶段 (ASPL-style)。

    目标 (梯度下降形式):
    loss = -L_0 + λ_att·A_0 + γ·L_star - γ·λ_att·A_star

    等价于最大化 D_0 - γ·D_star, 其中 D = L_cond - λ_att·A。

    - L_0 / A_0: 原始类别 e_0 下的噪声预测损失 / attention 强度
    - L_star / A_star: 伪词 e^⋆ 下的噪声预测损失 / attention 强度

    使用 PGD sign 更新: x = x + lr * sign(grad), 再 clamp 到 [-1, 1]。
    """
    x_param = torch.nn.Parameter(x_init_batch.detach().clone())
    x_original = x_init_batch.detach().clone()
    total_sample_count = 0
    update_round = 0

    pbar_stage2 = tqdm(total=timesteps, desc="Stage 2 (PGD)", unit="t")
    while total_sample_count < timesteps:
        update_round += 1

        d0_values = []
        d_star_values = []
        l0_values = []
        l_star_values = []
        a0_values = []
        a_star_values = []
        total_values = []

        timestep_group = sample_stratified_timesteps(
            num_samples=grad_accum,
            scheduler_train_steps=scheduler_train_steps,
            device=x_param.device,
        )

        x_param.grad = None

        for t in timestep_group:
            model_dtype = next(pipe.unet.parameters()).dtype
            latent_ref = pipe.vae.encode(x_param.detach().to(dtype=model_dtype)).latent_dist.sample()
            latent_ref = latent_ref * pipe.vae.config.scaling_factor
            fixed_noise = torch.randn_like(latent_ref)

            # ---- 前向 1: e_0 (原始类别) ----
            diff_loss_0, _ = compute_noise_prediction_loss_and_attention(
                pipe=pipe,
                x=x_param,
                t=t,
                prompt_state=prompt_state_original_batch,
                attention_token_indices=attention_token_indices_original,
                attention_recorder=attention_recorder,
                is_sdxl=is_sdxl,
                noise=fixed_noise,
                deterministic_vae=False,
            )
            intensity_0 = compute_attention_intensity(
                attention_recorder=attention_recorder,
                token_indices=attention_token_indices_original,
            )
            D_0 = diff_loss_0 - attn_intensity_weight * intensity_0

            # ---- 前向 2: e^⋆ (伪词) ----
            diff_loss_star, _ = compute_noise_prediction_loss_and_attention(
                pipe=pipe,
                x=x_param,
                t=t,
                prompt_state=prompt_state_star_batch,
                attention_token_indices=attention_token_indices_star,
                attention_recorder=attention_recorder,
                is_sdxl=is_sdxl,
                noise=fixed_noise,
                deterministic_vae=False,
            )
            intensity_star = compute_attention_intensity(
                attention_recorder=attention_recorder,
                token_indices=attention_token_indices_star,
            )
            D_star = diff_loss_star - attn_intensity_weight * intensity_star

            # ---- 组合目标 ----
            # max D_0 - γ·D_star ⇔ min -D_0 + γ·D_star
            total_loss = -D_0 + stage2_gamma * D_star

            d0_values.append(float(D_0))
            d_star_values.append(float(D_star))
            l0_values.append(float(diff_loss_0))
            l_star_values.append(float(diff_loss_star))
            a0_values.append(float(intensity_0))
            a_star_values.append(float(intensity_star))
            total_values.append(float(total_loss))
            (total_loss / float(grad_accum)).backward()
            total_sample_count += 1

        if x_param.grad is None:
            raise RuntimeError("未能得到图像梯度，请检查计算图是否被意外截断。")

        # PGD sign 更新 (ASPL-style) + epsilon ball 投影
        x_param.grad = torch.nan_to_num(x_param.grad)
        adv = x_param.data + lr * x_param.grad.sign()
        eta = torch.clamp(adv - x_original, min=-pgd_eps, max=pgd_eps)
        x_param.data = torch.clamp(x_original + eta, min=-1.0, max=1.0)

        mD0 = sum(d0_values)/len(d0_values); mDs = sum(d_star_values)/len(d_star_values)
        mL0 = sum(l0_values)/len(l0_values); mLs = sum(l_star_values)/len(l_star_values)
        mA0 = sum(a0_values)/len(a0_values); mAs = sum(a_star_values)/len(a_star_values)
        pbar_stage2.set_postfix(D0=f"{mD0:.4f}", Ds=f"{mDs:.4f}", L0=f"{mL0:.3f}")
        append_jsonl_record(metrics_log_path, {
            "stage": "stage2", "update": update_round, "sampled": total_sample_count,
            "D_0": mD0, "D_star": mDs, "L_0": mL0, "L_star": mLs,
            "A_0": mA0, "A_star": mAs,
        })
        pbar_stage2.update(len(timestep_group))

    pbar_stage2.close()
    # ---- Stage 2 after 热力图: (x_perturbed, e_0) 和 (x_perturbed, e^⋆) ----
    with torch.no_grad():
        _, stage2_e0_after = compute_noise_prediction_loss_and_attention(
            pipe=pipe, x=x_param.detach(),
            t=attention_eval_t if attention_eval_t is not None else sample_stratified_timesteps(
                num_samples=1, scheduler_train_steps=scheduler_train_steps, device=x_param.device,
            ),
            prompt_state=prompt_state_original_batch,
            attention_token_indices=attention_token_indices_original,
            attention_recorder=attention_recorder, is_sdxl=is_sdxl,
            noise=attention_eval_noise if attention_eval_noise is not None else torch.randn_like(
                pipe.vae.encode(x_param.detach().to(dtype=next(pipe.unet.parameters()).dtype)).latent_dist.mean
                * pipe.vae.config.scaling_factor
            ),
            deterministic_vae=True,
        )
        _, stage2_estar_after = compute_noise_prediction_loss_and_attention(
            pipe=pipe, x=x_param.detach(),
            t=attention_eval_t if attention_eval_t is not None else sample_stratified_timesteps(
                num_samples=1, scheduler_train_steps=scheduler_train_steps, device=x_param.device,
            ),
            prompt_state=prompt_state_star_batch,
            attention_token_indices=attention_token_indices_star,
            attention_recorder=attention_recorder, is_sdxl=is_sdxl,
            noise=attention_eval_noise if attention_eval_noise is not None else torch.randn_like(
                pipe.vae.encode(x_param.detach().to(dtype=next(pipe.unet.parameters()).dtype)).latent_dist.mean
                * pipe.vae.config.scaling_factor
            ),
            deterministic_vae=True,
        )
    return x_param.detach(), {
        "e0_after": stage2_e0_after.detach().cpu(),
        "estar_after": stage2_estar_after.detach().cpu(),
    }


# ============================================================
def _read_metrics(jsonl_path: Path, stage: str) -> list:
    import json
    records = []
    if not jsonl_path.exists():
        return records
    with open(jsonl_path) as f:
        for line in f:
            r = json.loads(line)
            if r.get("stage") == stage:
                records.append(r)
    return records


def plot_stage1_metrics(jsonl_path: Path, output_dir: Path, stem: str = "") -> None:
    s1 = _read_metrics(jsonl_path, "stage1")
    if not s1:
        return
    prefix = f"{stem}_" if stem else ""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    x = [r["update"] for r in s1]
    ax1.plot(x, [r["L_cond"] for r in s1], "b-", alpha=0.7)
    ax1.set_title("Stage 1: L_cond"); ax1.set_xlabel("Update"); ax1.set_ylabel("MSE")
    ax2.plot(x, [r["A_intensity"] for r in s1], "r-", alpha=0.7)
    ax2.set_title("Stage 1: A(x,t,e)"); ax2.set_xlabel("Update")
    fig.tight_layout()
    fig.savefig(output_dir / f"{prefix}metrics_stage1.png", dpi=150)
    plt.close(fig)


def plot_stage2_metrics(jsonl_path: Path, output_dir: Path, stem: str = "") -> None:
    s2 = _read_metrics(jsonl_path, "stage2")
    if not s2:
        return
    prefix = f"{stem}_" if stem else ""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    x = [r["update"] for r in s2]
    ax1.plot(x, [r["D_0"] for r in s2], "r-", label="D_0 (orig)", alpha=0.7)
    ax1.plot(x, [r["D_star"] for r in s2], "b-", label="D_star (pseudo)", alpha=0.7)
    ax1.set_title("Stage 2: D_0 & D_star"); ax1.legend(); ax1.set_xlabel("Update")
    ax2.plot(x, [r["L_0"] for r in s2], "r--", label="L_0", alpha=0.5)
    ax2.plot(x, [r["L_star"] for r in s2], "b--", label="L_star", alpha=0.5)
    ax2.plot(x, [r["A_0"] for r in s2], "r-", label="A_0", alpha=0.7)
    ax2.plot(x, [r["A_star"] for r in s2], "b-", label="A_star", alpha=0.7)
    ax2.set_title("Stage 2: L & A"); ax2.legend(); ax2.set_xlabel("Update")
    fig.tight_layout()
    fig.savefig(output_dir / f"{prefix}metrics_stage2.png", dpi=150)
    plt.close(fig)


# 13. 主方法：先更新 prompt，再更新图像
# ============================================================
def optimize_folder_images_with_prompt_first(
    image_dir: str,
    model_name: str,
    lr: float,
    lr_p: Optional[float],
    original_word: Optional[str],
    timesteps: int,
    image_batch_size: int,
    grad_accum: int,
    prompt_timesteps: Optional[int] = None,
    prompt_grad_accum: Optional[int] = None,
    pseudo_init_mode: str = "hybrid",
    num_pseudo_tokens: int = 4,
    image_only: bool = False,
    prompt_reg_weight: float = 1e-3,
    attn_intensity_weight: float = 0.0,
    stage2_gamma: float = 1.0,
    pgd_eps: float = 0.05,
    surrogate_train_dir: Optional[str] = None,
    surrogate_train_steps: int = 20,
    surrogate_learning_rate: float = 5e-6,
    surrogate_class_dir: Optional[str] = None,
    surrogate_class_prompt: Optional[str] = None,
    surrogate_prior_loss_weight: float = 1.0,
    surrogate_num_class_images: int = 200,
    outer_loop_steps: int = 1,
    outer_main_train_steps: int = 20,
    output_base_dir: Optional[str] = None,
    attn_layer_filters: str = DEFAULT_ATTN_LAYER_FILTER,
    attn_save_size: int = DEFAULT_ATTN_SAVE_SIZE,
    device: Optional[str] = None,
    seed: int = 42,
    height: Optional[int] = None,
    width: Optional[int] = None,
    run_dreambooth: bool = False,
    dreambooth_class_dir: Optional[str] = None,
    dreambooth_output_subdir: str = "dreambooth",
    dreambooth_instance_prompt: Optional[str] = None,
    dreambooth_class_prompt: Optional[str] = None,
    dreambooth_prior_loss_weight: Optional[float] = None,
    dreambooth_train_batch_size: Optional[int] = None,
    dreambooth_gradient_accumulation_steps: Optional[int] = None,
    dreambooth_learning_rate: Optional[float] = None,
    dreambooth_lr_scheduler: Optional[str] = None,
    dreambooth_lr_warmup_steps: Optional[int] = None,
    dreambooth_num_class_images: Optional[int] = None,
    dreambooth_max_train_steps: Optional[int] = None,
) -> Path:
    """
    这是整个任务的主方法。

    对文件夹中的每张图像，按以下顺序执行：

    阶段一：更新 prompt embeddings，使损失最大化
    ------------------------------------------------
    1. 输入图像 x 和当前 prompt embeddings
    2. 按固定时间步分布分层采样一组 t
    3. 加噪、预测噪声、计算损失
    4. 反向传播，得到 prompt embeddings 的梯度
    5. 做梯度累积
    6. 每累计 grad_accum 次后，使用 Adam 做一次梯度上升
    7. 重复直到 prompt 阶段采样次数达到 timesteps

    阶段二：更新图像，使损失最小化
    ------------------------------------------------
    8. 固定已更新的 prompt embeddings
    9. 输入图像 x 和该 prompt embeddings
    10. 按固定时间步分布分层采样一组 t
    11. 加噪、预测噪声、计算损失
    12. 反向传播，得到图像 x 的梯度
    13. 做梯度累积
    14. 每累计 grad_accum 次后，先做梯度预处理，再用带 momentum 的优化器做一次梯度下降
    15. 重复直到图像阶段采样次数达到 timesteps

    最终输出：
    - 图像保存到 outputs/文件夹名/
    - 优化后的 prompt 状态保存到 outputs/文件夹名/_prompt_states/
    - 若开启 run_dreambooth，则在全部图像处理完成后自动执行 DreamBooth
    """
    if timesteps <= 0:
        raise ValueError("timesteps 必须 > 0")
    if grad_accum <= 0:
        raise ValueError("grad_accum 必须 > 0")
    if timesteps % grad_accum != 0:
        raise ValueError("timesteps 必须是 grad_accum 的正整数倍")
    if prompt_reg_weight < 0:
        raise ValueError("prompt_reg_weight 必须 >= 0")
    if attn_intensity_weight < 0 or stage2_gamma < 0:
        raise ValueError("attn_intensity_weight / stage2_gamma 必须 >= 0")
    if attn_save_size <= 0:
        raise ValueError("attn_save_size 必须 > 0")
    if not image_only:
        if lr_p is None:
            raise ValueError("未开启 image_only 时，必须提供 --lr_p。")
        if prompt_timesteps is None:
            raise ValueError("未开启 image_only 时，必须提供 --prompt_timesteps。")
        if prompt_grad_accum is None:
            raise ValueError("未开启 image_only 时，必须提供 --prompt_grad_accum。")
        if prompt_timesteps <= 0:
            raise ValueError("prompt_timesteps 必须 > 0")
        if prompt_grad_accum <= 0:
            raise ValueError("prompt_grad_accum 必须 > 0")
        if original_word is None or not str(original_word).strip():
            raise ValueError("未开启 image_only 时，必须提供 --original_word。")
        if prompt_timesteps % prompt_grad_accum != 0:
            raise ValueError("prompt_timesteps 必须是 prompt_grad_accum 的正整数倍")
    image_dir = Path(image_dir)
    if surrogate_train_dir is not None:
        if surrogate_class_dir is None:
            raise ValueError("开启 surrogate 训练时，必须提供 --surrogate_class_dir。")
        if surrogate_class_prompt is None:
            raise ValueError("开启 surrogate 训练时，必须提供 --surrogate_class_prompt。")
    if not image_dir.exists() or not image_dir.is_dir():
        raise ValueError(f"image_dir 不存在或不是文件夹：{image_dir}")

    dreambooth_config = None
    if run_dreambooth:
        dreambooth_config = validate_dreambooth_config(
            model_name=model_name,
            class_dir=dreambooth_class_dir,
            output_subdir=dreambooth_output_subdir,
            instance_prompt=dreambooth_instance_prompt,
            class_prompt=dreambooth_class_prompt,
            prior_loss_weight=dreambooth_prior_loss_weight,
            train_batch_size=dreambooth_train_batch_size,
            gradient_accumulation_steps=dreambooth_gradient_accumulation_steps,
            learning_rate=dreambooth_learning_rate,
            lr_scheduler=dreambooth_lr_scheduler,
            lr_warmup_steps=dreambooth_lr_warmup_steps,
            num_class_images=dreambooth_num_class_images,
            max_train_steps=dreambooth_max_train_steps,
        )

    torch.manual_seed(seed)
    np.random.seed(seed)

    device_obj = resolve_device(device)
    print(f"[INFO] 加载模型 {model_name} ...")
    pipe = load_diffusion_pipeline(model_name, device_obj)
    print(f"[INFO] 模型加载完成")

    model_cfg = MODEL_REGISTRY[model_name]
    is_sdxl = bool(model_cfg["is_sdxl"])
    default_size = int(model_cfg["size"])
    resolved_prompt = BASE_PROMPT  # pseudo prompt 不再依赖 noise_token
    original_prompt_spec = None
    if not image_only and original_word is not None:
        original_prompt_text = BASE_PROMPT + " " + str(original_word)
        original_prompt_spec = PromptSpec(prompt=original_prompt_text, prompt_2=original_prompt_text)

    height = height or default_size
    width = width or default_size

    image_paths = list_images(image_dir)
    if image_batch_size is None:
        image_batch_size = len(image_paths)
    if image_batch_size <= 0:
        raise ValueError("image_batch_size 必须 > 0")

    if output_base_dir is not None:
        output_dir = Path(output_base_dir) / image_dir.parent.name
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_stem = build_output_stem(image_dir.name, model_name)
        output_dir = make_unique_output_dir(Path("outputs") / output_stem)
        output_dir.mkdir(parents=True, exist_ok=True)
    normalized_dreambooth_output_subdir = (
        dreambooth_config["output_subdir"] if dreambooth_config is not None else dreambooth_output_subdir
    )
    dreambooth_output_dir = output_dir / normalized_dreambooth_output_subdir
    dreambooth_instance_dir = output_dir / "dreambooth_instance_images"
    pseudo_word_state_dir = output_dir / "_pseudo_word_states"
    pseudo_word_state_dir.mkdir(parents=True, exist_ok=True)
    attention_dir = output_dir / "_attention_maps"
    attention_stage1_dir = attention_dir / "stage1"
    attention_stage2_dir = attention_dir / "stage2"
    attention_stage1_dir.mkdir(parents=True, exist_ok=True)
    attention_stage2_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = output_dir / "_metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    if run_dreambooth:
        dreambooth_instance_dir.mkdir(parents=True, exist_ok=True)

    attn_layer_filter_list = [item.strip() for item in attn_layer_filters.split(",")]
    attention_recorder = CrossAttentionMapRecorder(
        pipe.unet,
        layer_filters=attn_layer_filter_list,
    )

    run_config = {
        "image_dir": str(image_dir),
        "prompt": resolved_prompt,
        "pseudo_init_mode": pseudo_init_mode,
        "num_pseudo_tokens": num_pseudo_tokens,
        "model": model_name,
        "model_output_tag": str(model_cfg["output_tag"]),
        "lr": lr,
        "lr_p": lr_p,
        "image_only": image_only,
        "prompt_optimizer": "adam",
        "image_optimizer": "pgd_sign",
        "timesteps": timesteps,
        "image_batch_size": image_batch_size,
        "grad_accum": grad_accum,
        "prompt_timesteps": prompt_timesteps,
        "prompt_grad_accum": prompt_grad_accum,
        "prompt_reg_weight": prompt_reg_weight,
        "attn_intensity_weight": attn_intensity_weight,
        "attn_intensity_layers": CROSS_ATTN_INTENSITY_LAYERS,
        "original_word": original_word,
        "stage2_gamma": stage2_gamma,
        "pgd_eps": pgd_eps,
        "surrogate_train_dir": surrogate_train_dir,
        "surrogate_train_steps": surrogate_train_steps,
        "surrogate_learning_rate": surrogate_learning_rate,
        "surrogate_class_dir": surrogate_class_dir,
        "surrogate_class_prompt": surrogate_class_prompt,
        "surrogate_prior_loss_weight": surrogate_prior_loss_weight,
        "surrogate_num_class_images": surrogate_num_class_images,
        "attn_layer_filters": attn_layer_filter_list,
        "attn_save_size": attn_save_size,
        "device": str(device_obj),
        "seed": seed,
        "height": height,
        "width": width,
        "pseudo_word_state_dir": str(pseudo_word_state_dir),
        "attention_dir": str(attention_dir),
        "metrics_dir": str(metrics_dir),
        "run_dreambooth": run_dreambooth,
        "dreambooth_class_dir": (
            str(dreambooth_config["class_dir_path"]) if dreambooth_config is not None else None
        ),
        "dreambooth_output_subdir": normalized_dreambooth_output_subdir,
        "dreambooth_output_dir": str(dreambooth_output_dir),
        "dreambooth_instance_dir": (
            str(dreambooth_instance_dir) if run_dreambooth else None
        ),
        "dreambooth_cuda_visible_devices": infer_dreambooth_cuda_visible_devices(device),
        "dreambooth_instance_prompt": (
            dreambooth_config["instance_prompt"] if dreambooth_config is not None else None
        ),
        "dreambooth_class_prompt": (
            dreambooth_config["class_prompt"] if dreambooth_config is not None else None
        ),
        "dreambooth_prior_loss_weight": (
            dreambooth_config["prior_loss_weight"] if dreambooth_config is not None else None
        ),
        "dreambooth_train_batch_size": (
            dreambooth_config["train_batch_size"] if dreambooth_config is not None else None
        ),
        "dreambooth_gradient_accumulation_steps": (
            dreambooth_config["gradient_accumulation_steps"] if dreambooth_config is not None else None
        ),
        "dreambooth_learning_rate": (
            dreambooth_config["learning_rate"] if dreambooth_config is not None else None
        ),
        "dreambooth_lr_scheduler": (
            dreambooth_config["lr_scheduler"] if dreambooth_config is not None else None
        ),
        "dreambooth_lr_warmup_steps": (
            dreambooth_config["lr_warmup_steps"] if dreambooth_config is not None else None
        ),
        "dreambooth_num_class_images": (
            dreambooth_config["num_class_images"] if dreambooth_config is not None else None
        ),
        "dreambooth_max_train_steps": (
            dreambooth_config["max_train_steps"] if dreambooth_config is not None else None
        ),
        "resolved_output_dir": str(output_dir),
    }
    save_run_config(run_config, output_dir / "run_config.json")

    scheduler_train_steps = int(pipe.scheduler.config.num_train_timesteps)
    image_path_batches = chunk_list(image_paths, image_batch_size)

    try:
        for batch_idx, image_path_batch in enumerate(image_path_batches, start=1):


            x_init_list: List[torch.Tensor] = []
            prompt_states_init: List[dict] = []
            original_prompt_states: List[dict] = []

            for image_path in image_path_batch:
                

                x_init = load_image_as_tensor(image_path, height, width, device_obj)
                x_init_list.append(x_init)

                prompt_state_init = init_pseudo_state_hybrid(
                    pipe=pipe, num_pseudo_tokens=num_pseudo_tokens,
                    is_sdxl=is_sdxl, device=device_obj,
                    height=height, width=width,
                )
                prompt_states_init.append(prompt_state_init)

                if not image_only and original_prompt_spec is not None:
                    original_prompt_state = init_prompt_state_from_text(
                        pipe=pipe,
                        prompt_spec=original_prompt_spec,
                        is_sdxl=is_sdxl,
                        device=device_obj,
                        height=height,
                        width=width,
                    )
                    original_prompt_states.append(original_prompt_state)

            x_init_batch = torch.cat(x_init_list, dim=0)
            # 每张图像独立记录指标
            metrics_log_path = metrics_dir / f"{image_path_batch[0].stem}_metrics.jsonl"
            metrics_log_path.write_text("")

            attention_eval_t, attention_eval_noise = build_shared_attention_eval_sample(
                pipe=pipe,
                x_reference=x_init_batch,
                scheduler_train_steps=scheduler_train_steps,
            )

            if image_only:
                
                optimized_prompt_states = prompt_states_init
                stage1_attention_maps = None
            else:
                optimized_prompt_states, stage1_attention_maps = optimize_prompt_embeddings_for_batch(
                    pipe=pipe,
                    x_batch=x_init_batch,
                    prompt_states_init=prompt_states_init,
                    lr_p=lr_p,
                    attn_intensity_weight=attn_intensity_weight,
                    timesteps=prompt_timesteps,
                    grad_accum=prompt_grad_accum,
                    scheduler_train_steps=scheduler_train_steps,
                    attention_recorder=attention_recorder,
                    metrics_log_path=metrics_log_path,
                    is_sdxl=is_sdxl,
                    attention_eval_t=attention_eval_t,
                    attention_eval_noise=attention_eval_noise,
                )

            for image_path, optimized_prompt_state in zip(image_path_batch, optimized_prompt_states):
                pseudo_word_state_path = pseudo_word_state_dir / f"{image_path.stem}_pseudo_word_state.pt"
                save_prompt_state(optimized_prompt_state, pseudo_word_state_path)
                

            if stage1_attention_maps is not None:
                for sample_idx, image_path in enumerate(image_path_batch):
                    save_attention_map_as_image(
                        stage1_attention_maps["pseudo_before"][sample_idx],
                        attention_stage1_dir / f"{image_path.stem}_pseudo_before.png",
                    )
                    save_attention_map_as_image(
                        stage1_attention_maps["pseudo_after"][sample_idx],
                        attention_stage1_dir / f"{image_path.stem}_pseudo_after.png",
                    )
            release_cuda_memory()

            # ---- Stage 1 指标图 ----
            if not image_only and metrics_log_path.exists():
                stem = image_path_batch[0].stem
                plot_stage1_metrics(metrics_log_path, metrics_dir, stem)

            prompt_state_batch = collate_prompt_states_for_batch(
                optimized_prompt_states, is_sdxl=is_sdxl,
            )
            # 伪词 attention = learnable positions; 原始 = class word 位置
            attention_token_indices_star = optimized_prompt_states[0]["learnable_token_indices"].detach().clone().long()
            orig_txt = BASE_PROMPT + " " + str(original_word) if original_word else BASE_PROMPT
            attention_token_indices_original = torch.tensor(
                resolve_token_indices(pipe.tokenizer, orig_txt, str(original_word)),
                dtype=torch.long, device=device_obj,
            )
            prompt_state_original_batch = collate_prompt_states_for_batch(
                original_prompt_states, is_sdxl=is_sdxl,
            ) if original_prompt_states else prompt_state_batch

            # ---- 保存原始 UNet, 初始化当前图像 ----
            original_unet = pipe.unet
            original_vae_dtype = next(pipe.vae.parameters()).dtype
            original_te_dtype = next(pipe.text_encoder.parameters()).dtype
            x_curr = x_init_batch.detach().clone()
            surrogate_e0_prompt = BASE_PROMPT + " " + str(original_word) if original_word else resolved_prompt

            # ---- Stage 2 before 热力图 (原始图像, 外层循环前) ----
            with torch.no_grad():
                _, stage2_e0_before = compute_noise_prediction_loss_and_attention(
                    pipe=pipe, x=x_curr,
                    t=attention_eval_t if attention_eval_t is not None else sample_stratified_timesteps(
                        num_samples=1, scheduler_train_steps=scheduler_train_steps, device=x_curr.device),
                    prompt_state=prompt_state_original_batch,
                    attention_token_indices=attention_token_indices_original,
                    attention_recorder=attention_recorder, is_sdxl=is_sdxl,
                    noise=attention_eval_noise if attention_eval_noise is not None else torch.randn_like(
                        pipe.vae.encode(x_curr.to(dtype=next(pipe.unet.parameters()).dtype)).latent_dist.mean
                        * pipe.vae.config.scaling_factor),
                    deterministic_vae=True,
                )
                _, stage2_estar_before = compute_noise_prediction_loss_and_attention(
                    pipe=pipe, x=x_curr,
                    t=attention_eval_t if attention_eval_t is not None else sample_stratified_timesteps(
                        num_samples=1, scheduler_train_steps=scheduler_train_steps, device=x_curr.device),
                    prompt_state=prompt_state_batch,
                    attention_token_indices=attention_token_indices_star,
                    attention_recorder=attention_recorder, is_sdxl=is_sdxl,
                    noise=attention_eval_noise if attention_eval_noise is not None else torch.randn_like(
                        pipe.vae.encode(x_curr.to(dtype=next(pipe.unet.parameters()).dtype)).latent_dist.mean
                        * pipe.vae.config.scaling_factor),
                    deterministic_vae=True,
                )
            stage2_e0_before = stage2_e0_before.detach().cpu()
            stage2_estar_before = stage2_estar_before.detach().cpu()

            # ---- ASPL 外层循环 ----
            outer_pbar = tqdm(range(outer_loop_steps), desc="Outer loop", unit="iter")
            for outer_step in outer_pbar:
                outer_pbar.set_postfix(iter=f"{outer_step + 1}/{outer_loop_steps}")

                # ---- Step A: 训练 surrogate UNet (在 set_A 上) ----
                if surrogate_train_dir is not None:
                    surr_tensors = []
                    for surr_path in list_images(Path(surrogate_train_dir)):
                        surr_tensors.append(load_image_as_tensor(surr_path, height, width, device_obj))
                    surrogate_batch = torch.cat(surr_tensors, dim=0)

                    original_unet.to("cpu")
                    release_cuda_memory()
                    surrogate_unet = train_surrogate_unet(
                        pipe=pipe, instance_images=surrogate_batch,
                        instance_prompt=surrogate_e0_prompt,
                        class_images_dir=surrogate_class_dir,
                        class_prompt=surrogate_class_prompt,
                        num_steps=surrogate_train_steps,
                        learning_rate=surrogate_learning_rate,
                        prior_loss_weight=surrogate_prior_loss_weight,
                        num_class_images=surrogate_num_class_images,
                        resolution=height,
                    )
                    attention_recorder.close()
                    pipe.unet = surrogate_unet
                    attention_recorder = CrossAttentionMapRecorder(
                        pipe.unet, layer_filters=attn_layer_filter_list,
                    )
                    

                # ---- Step B: PGD 攻击 (Stage 2) ----
                x_curr, stage2_attention_maps = optimize_image_with_fixed_prompt_for_batch(
                    pipe=pipe, x_init_batch=x_curr,
                    prompt_state_star_batch=prompt_state_batch,
                    attention_token_indices_star=attention_token_indices_star,
                    prompt_state_original_batch=prompt_state_original_batch,
                    attention_token_indices_original=attention_token_indices_original,
                    lr=lr, pgd_eps=pgd_eps, stage2_gamma=stage2_gamma,
                    attn_intensity_weight=attn_intensity_weight,
                    timesteps=timesteps, grad_accum=grad_accum,
                    scheduler_train_steps=scheduler_train_steps,
                    attention_recorder=attention_recorder,
                    metrics_log_path=metrics_log_path, is_sdxl=is_sdxl,
                    attention_eval_t=attention_eval_t,
                    attention_eval_noise=attention_eval_noise,
                )

                # ---- Step C: 用攻击后图像训练原始 UNet (ASPL step 3) ----
                if surrogate_train_dir is not None and outer_step < outer_loop_steps - 1:
                    attention_recorder.close()
                    del pipe.unet
                    pipe.unet = original_unet
                    pipe.unet.to(device_obj)
                    unet_dtype = next(pipe.unet.parameters()).dtype
                    pipe.vae.to(device_obj, dtype=unet_dtype)
                    pipe.text_encoder.to(device_obj, dtype=unet_dtype)
                    release_cuda_memory()

                    pipe.unet = train_surrogate_unet(
                        pipe=pipe, instance_images=x_curr.detach(),
                        instance_prompt=surrogate_e0_prompt,
                        class_images_dir=surrogate_class_dir,
                        class_prompt=surrogate_class_prompt,
                        num_steps=outer_main_train_steps,
                        learning_rate=surrogate_learning_rate,
                        prior_loss_weight=surrogate_prior_loss_weight,
                        num_class_images=surrogate_num_class_images,
                        resolution=height,
                    )
                    original_unet = pipe.unet  # 更新 original 为训练后的模型
                    original_unet.to("cpu")
                    release_cuda_memory()
                    attention_recorder = CrossAttentionMapRecorder(
                        pipe.unet, layer_filters=attn_layer_filter_list,
                    )
                    

            # ---- 保存最终结果 ----
            x_optimized_batch = x_curr.detach()
            for sample_idx, image_path in enumerate(image_path_batch):
                x_optimized = x_optimized_batch[sample_idx:sample_idx + 1]
                image_save_path = output_dir / image_path.name
                save_tensor_as_image(x_optimized, image_save_path)
                

                save_attention_map_as_image(
                    stage2_e0_before[sample_idx],
                    attention_stage2_dir / f"{image_path.stem}_e0_before.png",
                )
                save_attention_map_as_image(
                    stage2_attention_maps["e0_after"][sample_idx],
                    attention_stage2_dir / f"{image_path.stem}_e0_after.png",
                )
                save_attention_map_as_image(
                    stage2_estar_before[sample_idx],
                    attention_stage2_dir / f"{image_path.stem}_estar_before.png",
                )
                save_attention_map_as_image(
                    stage2_attention_maps["estar_after"][sample_idx],
                    attention_stage2_dir / f"{image_path.stem}_estar_after.png",
                )

                if run_dreambooth:
                    save_tensor_as_image(x_optimized, dreambooth_instance_dir / image_path.name)

            # ---- Stage 2 指标图 ----
            if metrics_log_path.exists():
                stem = image_path_batch[0].stem
                plot_stage2_metrics(metrics_log_path, metrics_dir, stem)

            # ---- 恢复原始模型 ----
            attention_recorder.close()
            del pipe.unet
            pipe.unet = original_unet
            pipe.unet.to(device_obj)
            unet_dtype = next(pipe.unet.parameters()).dtype
            pipe.vae.to(device_obj, dtype=unet_dtype)
            pipe.text_encoder.to(device_obj, dtype=unet_dtype)
            release_cuda_memory()
            attention_recorder = CrossAttentionMapRecorder(
                pipe.unet, layer_filters=attn_layer_filter_list,
            )
            
    finally:
        attention_recorder.close()

    if run_dreambooth:
        dreambooth_resolution = height if height == width else default_size
        if height != width:
            print(
                f"[INFO] DreamBooth 使用单一 resolution 参数，当前 height={height}, width={width}，"
                f"将回退到模型默认分辨率 {default_size}。"
            )

        run_dreambooth_training(
            model_name=model_name,
            instance_dir=dreambooth_instance_dir,
            class_dir=dreambooth_config["class_dir_path"],
            checkpoint_dir=dreambooth_output_dir,
            resolution=dreambooth_resolution,
            device_arg=device,
            instance_prompt=dreambooth_config["instance_prompt"],
            class_prompt=dreambooth_config["class_prompt"],
            prior_loss_weight=dreambooth_config["prior_loss_weight"],
            train_batch_size=dreambooth_config["train_batch_size"],
            gradient_accumulation_steps=dreambooth_config["gradient_accumulation_steps"],
            learning_rate=dreambooth_config["learning_rate"],
            lr_scheduler=dreambooth_config["lr_scheduler"],
            lr_warmup_steps=dreambooth_config["lr_warmup_steps"],
            num_class_images=dreambooth_config["num_class_images"],
            max_train_steps=dreambooth_config["max_train_steps"],
        )
        if model_name == "sd1.5":
            run_sd15_inference_sampling(
                model_dir=dreambooth_output_dir,
                output_dir=output_dir / "dreambooth_samples",
                subject_prompt=dreambooth_config["instance_prompt"],
                device_arg=device,
                seed=seed,
            )
        else:
            print(f"[INFO] 当前模型 {model_name} 未接入 sd15_inference.py，跳过 DreamBooth 采样。")

    return output_dir


# ============================================================
# 13. main
# ============================================================
def main() -> None:
    args = parse_args()

    output_dir = optimize_folder_images_with_prompt_first(
        image_dir=args.image_dir,
        model_name=args.model,
        lr=args.lr,
        pseudo_init_mode=args.pseudo_init_mode,
        num_pseudo_tokens=args.num_pseudo_tokens,
        lr_p=args.lr_p,
        original_word=args.original_word,
        timesteps=args.timesteps,
        image_batch_size=args.image_batch_size,
        grad_accum=args.grad_accum,
        prompt_timesteps=args.prompt_timesteps,
        prompt_grad_accum=args.prompt_grad_accum,
        image_only=args.image_only,
        prompt_reg_weight=args.prompt_reg_weight,
        attn_intensity_weight=args.attn_intensity_weight,
        stage2_gamma=args.stage2_gamma,
        pgd_eps=args.pgd_eps,
        surrogate_train_dir=args.surrogate_train_dir,
        surrogate_train_steps=args.surrogate_train_steps,
        surrogate_learning_rate=args.surrogate_learning_rate,
        surrogate_class_dir=args.surrogate_class_dir,
        surrogate_class_prompt=args.surrogate_class_prompt,
        surrogate_prior_loss_weight=args.surrogate_prior_loss_weight,
        surrogate_num_class_images=args.surrogate_num_class_images,
        outer_loop_steps=args.outer_loop_steps,
        outer_main_train_steps=args.outer_main_train_steps,
        attn_layer_filters=args.attn_layer_filters,
        attn_save_size=args.attn_save_size,
        device=args.device,
        seed=args.seed,
        height=args.height,
        width=args.width,
        run_dreambooth=args.run_dreambooth,
        dreambooth_class_dir=args.dreambooth_class_dir,
        dreambooth_output_subdir=args.dreambooth_output_subdir,
        dreambooth_instance_prompt=args.dreambooth_instance_prompt,
        dreambooth_class_prompt=args.dreambooth_class_prompt,
        dreambooth_prior_loss_weight=args.dreambooth_prior_loss_weight,
        dreambooth_train_batch_size=args.dreambooth_train_batch_size,
        dreambooth_gradient_accumulation_steps=args.dreambooth_gradient_accumulation_steps,
        dreambooth_learning_rate=args.dreambooth_learning_rate,
        dreambooth_lr_scheduler=args.dreambooth_lr_scheduler,
        dreambooth_lr_warmup_steps=args.dreambooth_lr_warmup_steps,
        dreambooth_num_class_images=args.dreambooth_num_class_images,
        dreambooth_max_train_steps=args.dreambooth_max_train_steps,
        output_base_dir=args.output_dir,
    )

    print(f"\n[INFO] 全部处理完成，输出目录：{output_dir}")


if __name__ == "__main__":
    main()
