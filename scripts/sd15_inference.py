import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from diffusers import StableDiffusionPipeline
from diffusers.models.attention_processor import AttnProcessor


@dataclass
class TokenGroup:
    text: str
    token_indices: list[int]


class AttentionStore:
    """Collect conditional cross-attention maps for every generated image."""

    def __init__(self, base_batch_size: int, do_cfg: bool) -> None:
        self.base_batch_size = base_batch_size
        self.do_cfg = do_cfg
        self.records: list[dict[str, object]] = []

    def clear(self) -> None:
        self.records = []

    def add(self, *, name: str, attention_probs: torch.Tensor, heads: int) -> None:
        bh, q_len, k_len = attention_probs.shape
        if bh % heads != 0:
            return

        batch = bh // heads
        probs = attention_probs.reshape(batch, heads, q_len, k_len)

        if self.do_cfg and batch == self.base_batch_size * 2:
            probs = probs[self.base_batch_size :]
        elif batch >= self.base_batch_size:
            probs = probs[: self.base_batch_size]
        else:
            return

        self.records.append(
            {
                "name": name,
                "probs": probs.detach().float().cpu(),
                "q_len": q_len,
                "k_len": k_len,
            }
        )


class CrossAttnCaptureProcessor(AttnProcessor):
    """Attention processor copied from the reference script, with capture added."""

    def __init__(self, store: AttentionStore, name: str) -> None:
        super().__init__()
        self.store = store
        self.name = name

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        *args,
        **kwargs,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        else:
            batch_size = hidden_states.shape[0]

        sequence_length = hidden_states.shape[1] if encoder_hidden_states is None else encoder_hidden_states.shape[1]
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        is_cross = encoder_hidden_states is not None
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

        if is_cross:
            self.store.add(name=self.name, attention_probs=attention_probs, heads=attn.heads)

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SD1.5 DreamBooth inference.")

    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Checkpoint directory.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save samples.",
    )
    parser.add_argument(
        "--attention_dir",
        type=str,
        default=None,
        help="Directory to save cross-attention maps. Defaults to output_dir/attention_maps.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Torch device, e.g. cuda or cuda:0.",
    )

    parser.add_argument(
        "--token",
        type=str,
        default="sks",
        help="DreamBooth unique token, e.g. 'sks'.",
    )
    parser.add_argument(
        "--class_name",
        type=str,
        default="subject",
        help="Class name, e.g. 'cat', 'dog', 'person', 'wolf plushie'.",
    )

    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="Number of denoising steps.",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help="Classifier-free guidance scale.",
    )

    parser.add_argument(
        "--inference_seed",
        type=int,
        default=1234,
        help="Base seed for sampling.",
    )
    parser.add_argument(
        "--inference_seed_stride_per_prompt",
        type=int,
        default=100000,
        help="Seed increment from one prompt to the next.",
    )
    parser.add_argument(
        "--inference_seed_stride_per_image",
        type=int,
        default=1000,
        help="Seed increment from one image to the next within the same prompt.",
    )
    parser.add_argument(
        "--num_images_per_prompt",
        type=int,
        default=4,
        help="Number of images to generate per prompt.",
    )
    parser.add_argument(
        "--prompts_file",
        type=str,
        default=None,
        help="Path to a JSON file containing prompt templates. "
        "Templates use {token} and {class_name} placeholders.",
    )

    return parser.parse_args()


def set_cross_attention_processors(pipe: StableDiffusionPipeline, store: AttentionStore) -> None:
    processors = {}
    for name, _module in pipe.unet.attn_processors.items():
        is_cross = not name.endswith("attn1.processor")
        processors[name] = (
            CrossAttnCaptureProcessor(store=store, name=name)
            if is_cross
            else AttnProcessor()
        )
    pipe.unet.set_attn_processor(processors)


def attention_prompt_from_generation_prompt(prompt: str) -> str:
    """Use only the leading sentence fragment, e.g. text before the first comma."""
    return prompt.split(",", 1)[0].strip().rstrip(".")


def get_prompt_groups(tokenizer, prompt: str) -> tuple[list[TokenGroup], list[int], list[dict[str, object]]]:
    tokenized = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    ids = tokenized.input_ids[0].tolist()
    toks = tokenizer.convert_ids_to_tokens(ids)

    groups: list[TokenGroup] = []
    content_indices: list[int] = []
    current_text = ""
    current_indices: list[int] = []
    token_debug: list[dict[str, object]] = []

    special_tokens = {tokenizer.bos_token, tokenizer.eos_token, tokenizer.pad_token}

    for idx, tok in enumerate(toks):
        token_debug.append({"index": idx, "token": tok})
        if tok in special_tokens:
            continue

        content_indices.append(idx)
        piece = tok.replace("</w>", "")
        current_text += piece
        current_indices.append(idx)

        if tok.endswith("</w>"):
            text = current_text.strip()
            if text:
                groups.append(TokenGroup(text=text, token_indices=current_indices.copy()))
            current_text = ""
            current_indices = []

    if current_indices:
        text = current_text.strip()
        if text:
            groups.append(TokenGroup(text=text, token_indices=current_indices.copy()))

    return groups, content_indices, token_debug


def _reshape_attention_map(flat_map: torch.Tensor) -> torch.Tensor:
    n = flat_map.shape[-1]
    side = int(math.sqrt(n))
    if side * side != n:
        raise ValueError(f"Query length {n} is not a square number; cannot reshape to spatial map.")
    return flat_map.reshape(side, side)


def aggregate_heatmap(
    *,
    records: list[dict[str, object]],
    token_indices: list[int],
    image_index: int,
    latent_hw: tuple[int, int],
) -> np.ndarray:
    maps = []
    latent_h, latent_w = latent_hw

    for rec in records:
        probs: torch.Tensor = rec["probs"]
        if image_index >= probs.shape[0]:
            continue

        valid_indices = [idx for idx in token_indices if idx < probs.shape[-1]]
        if not valid_indices:
            continue

        token_map = probs[image_index, :, :, valid_indices].mean(dim=-1).mean(dim=0)
        spatial = _reshape_attention_map(token_map)
        spatial = spatial.unsqueeze(0).unsqueeze(0)
        spatial = F.interpolate(
            spatial,
            size=(latent_h, latent_w),
            mode="bilinear",
            align_corners=False,
        )
        maps.append(spatial[0, 0])

    if not maps:
        raise RuntimeError("No cross-attention maps were captured for the requested tokens.")

    heatmap = torch.stack(maps, dim=0).mean(dim=0)
    heatmap = heatmap - heatmap.min()
    heatmap = heatmap / (heatmap.max() + 1e-8)
    return heatmap.cpu().numpy()


def upsample_heatmap(heatmap: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    return np.array(
        Image.fromarray((heatmap * 255).astype(np.uint8)).resize(
            image_size,
            Image.Resampling.BILINEAR,
        )
    ).astype(np.float32) / 255.0


def save_heatmap_only(heatmap: np.ndarray, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6, 6))
    plt.imshow(heatmap, cmap="jet")
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0)
    plt.close()


def save_overlay(base_image: Image.Image, heatmap: np.ndarray, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    base_np = np.asarray(base_image.convert("RGB")).astype(np.float32) / 255.0
    plt.figure(figsize=(6, 6))
    plt.imshow(base_np)
    plt.imshow(heatmap, cmap="jet", alpha=0.45)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0)
    plt.close()


def safe_name(text: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in text.strip())
    return safe or "item"


def save_attention_maps_for_image(
    *,
    image: Image.Image,
    image_stem: str,
    image_index: int,
    prompt: str,
    attention_prompt: str,
    store: AttentionStore,
    tokenizer,
    output_dir: Path,
) -> None:
    sample_attention_dir = output_dir / image_stem
    sample_attention_dir.mkdir(parents=True, exist_ok=True)

    word_groups, sentence_token_indices, token_debug = get_prompt_groups(tokenizer, attention_prompt)
    latent_hw = (max(image.height // 8, 1), max(image.width // 8, 1))

    sentence_heatmap = aggregate_heatmap(
        records=store.records,
        token_indices=sentence_token_indices,
        image_index=image_index,
        latent_hw=latent_hw,
    )
    sentence_heatmap = upsample_heatmap(sentence_heatmap, image.size)
    save_heatmap_only(sentence_heatmap, sample_attention_dir / "sentence_heatmap.png")
    save_overlay(image, sentence_heatmap, sample_attention_dir / "sentence_overlay.png")

    words_dir = sample_attention_dir / "words"
    words_dir.mkdir(exist_ok=True)
    words_manifest = []

    for word_index, group in enumerate(word_groups):
        heatmap = aggregate_heatmap(
            records=store.records,
            token_indices=group.token_indices,
            image_index=image_index,
            latent_hw=latent_hw,
        )
        heatmap = upsample_heatmap(heatmap, image.size)

        word_name = safe_name(group.text)
        heatmap_path = words_dir / f"{word_index:02d}_{word_name}_heatmap.png"
        overlay_path = words_dir / f"{word_index:02d}_{word_name}_overlay.png"
        save_heatmap_only(heatmap, heatmap_path)
        save_overlay(image, heatmap, overlay_path)

        words_manifest.append(
            {
                "word": group.text,
                "token_indices": group.token_indices,
                "heatmap": str(heatmap_path.relative_to(sample_attention_dir)),
                "overlay": str(overlay_path.relative_to(sample_attention_dir)),
            }
        )

    manifest = {
        "prompt": prompt,
        "attention_prompt": attention_prompt,
        "image_stem": image_stem,
        "image_index_within_prompt_batch": image_index,
        "num_attention_records": len(store.records),
        "sentence_token_indices": sentence_token_indices,
        "sentence_heatmap": "sentence_heatmap.png",
        "sentence_overlay": "sentence_overlay.png",
        "words": words_manifest,
    }

    with open(sample_attention_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    with open(sample_attention_dir / "tokens.json", "w", encoding="utf-8") as f:
        json.dump(token_debug, f, ensure_ascii=False, indent=2)


def build_prompts(token: str, class_name: str, prompts_file: str | None = None) -> list[str]:
    if prompts_file is not None:
        with open(prompts_file, encoding="utf-8") as f:
            templates = json.load(f)
        return [
            t.format(token=token, class_name=class_name) for t in templates
        ]

    subject = f"{token} {class_name}"

    return [
        f"a photo of {subject}, centered composition, sharp focus, natural lighting, detailed texture.",
        f"a DSLR portrait photo of {subject}, clean background, realistic details, high resolution.",
        f"a full body photo of {subject}, standing alone, neutral background, studio lighting.",

        f"a photo of {subject} on a city street at night, neon lights, realistic photography.",
        f"a photo of {subject} in a forest, soft sunlight through trees, natural environment.",
        f"a photo of {subject} on a beach at sunset, warm lighting, cinematic composition.",
        f"a photo of {subject} inside a cozy room, soft window light, realistic details.",

        f"an oil painting of {subject}, classical portrait style, rich brush strokes, detailed lighting.",
        f"a 3D render of {subject}, soft lighting, highly detailed, realistic materials.",
        f"an anime illustration of {subject}, clean line art, vibrant colors, detailed background.",
        f"a watercolor painting of {subject}, soft colors, delicate texture, artistic composition.",
        f"an impressionistic depiction of {subject}.",
        f"an abstract representation of {subject}.",
        f"a cyberpunk style photo of {subject}, neon lighting, futuristic atmosphere.",
        f"a Van Gogh style painting of {subject}, swirling brush strokes, bold impasto texture, vibrant yellow and blue tones, expressive post-impressionist oil painting, detailed artistic background.",
        f"a Van Gogh style painting of {subject} under a starry night sky, swirling clouds, glowing stars, vivid blue and yellow colors, thick oil paint texture, post-impressionist artistic style.",
        f"a centered portrait of {subject} in the style of Vincent van Gogh, expressive brushwork, rich oil paint texture, swirling patterns, vibrant colors, artistic and detailed composition." ,

        f"a photo of {subject} running, dynamic pose, motion blur, realistic lighting.",
        f"a photo of {subject} sitting on a chair, relaxed pose, indoor lighting.",
        f"a photo of {subject} jumping in the air, dynamic composition, sharp focus.",
        f"a photo of {subject} wearing sunglasses, playful mood, studio photography.",

        f"a photo of {subject} placed on a wooden table, realistic lighting, detailed texture.",
        f"a photo of {subject} hanging on a wall, indoor scene, natural shadows.",
        f"a photo of {subject} floating in the air, surreal composition, sharp focus.",
        f"a photo of {subject} covered with snow, outdoor winter scene, realistic details.",
    ]


def main() -> None:
    args = parse_args()

    model_dir = Path(args.model_dir)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else model_dir.parent / "dreambooth_samples"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    attention_dir = (
        Path(args.attention_dir)
        if args.attention_dir is not None
        else output_dir / "attention_maps"
    )
    attention_dir.mkdir(parents=True, exist_ok=True)

    pipe = StableDiffusionPipeline.from_pretrained(
        str(model_dir),
        torch_dtype=torch.float16,
        safety_checker=None,
    )

    pipe = pipe.to(args.device)
    pipe.safety_checker = None
    attention_store = AttentionStore(
        base_batch_size=args.num_images_per_prompt,
        do_cfg=args.guidance_scale > 1.0,
    )
    set_cross_attention_processors(pipe, attention_store)

    prompts = build_prompts(args.token, args.class_name, args.prompts_file)

    prompts_txt_path = output_dir / "prompts.txt"
    if prompts_txt_path.exists():
        prompts_txt_path.unlink()

    global_image_index = 0

    for prompt_index, prompt in enumerate(prompts):
        attention_prompt = attention_prompt_from_generation_prompt(prompt)
        seeds = [
            args.inference_seed
            + prompt_index * args.inference_seed_stride_per_prompt
            + image_index * args.inference_seed_stride_per_image
            for image_index in range(args.num_images_per_prompt)
        ]

        generators = [
            torch.Generator(device=args.device).manual_seed(seed)
            for seed in seeds
        ]

        attention_store.clear()
        result = pipe(
            prompt=prompt,
            num_images_per_prompt=args.num_images_per_prompt,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            generator=generators,
        )

        for image_index_within_prompt, (image, seed) in enumerate(
            zip(result.images, seeds)
        ):
            image_filename = (
                f"{global_image_index:04d}"
                f"_prompt{prompt_index:02d}"
                f"_img{image_index_within_prompt:02d}"
                f"_seed{seed}.png"
            )
            image.save(os.path.join(output_dir, image_filename))
            image_stem = Path(image_filename).stem
            save_attention_maps_for_image(
                image=image,
                image_stem=image_stem,
                image_index=image_index_within_prompt,
                prompt=prompt,
                attention_prompt=attention_prompt,
                store=attention_store,
                tokenizer=pipe.tokenizer,
                output_dir=attention_dir,
            )

            with open(prompts_txt_path, "a", encoding="utf-8") as f:
                f.write(f"{image_filename}\t{prompt}\tattention_prompt={attention_prompt}\n")

            print(f"Saved {image_filename}: {prompt}")

            global_image_index += 1

    print(f"Done. Images saved to: {output_dir}")
    print(f"Attention maps saved to: {attention_dir}")
    print(f"Prompts saved to: {prompts_txt_path}")


if __name__ == "__main__":
    main()
