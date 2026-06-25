# MisBind: Misleading Subject–Condition Binding against Unauthorized Personalization


MisBind is a method for protecting personal images from being used to train personalized generative models (e.g., DreamBooth). It adds imperceptible adversarial perturbations to images, causing diffusion models trained on the perturbed images to produce degraded outputs.


## How It Works

MisBind employs a two-stage optimization:

1. **Stage I — Prompt Optimization (Pseudo-word Learning):** Learns pseudo-token embeddings appended to a base prompt ("a photo of a") by maximizing the diffusion noise-prediction loss via Adam gradient ascent.

2. **Stage II — Image Perturbation (PGD Attack):** Perturbs input images using Projected Gradient Descent within an L-infinity epsilon ball. The loss balances:
   - Maximizing loss under the original class prompt (disrupting the class)
   - Minimizing loss under the learned pseudo-token (preserving pseudo-token alignment)

An optional **ASPL outer loop** trains a surrogate UNet on clean reference images, making the attack transferable.

## Installation

```bash
git clone https://github.com/zhangxin/MisBind.git
cd MisBind
pip install -e .
```

### Requirements

- Python >= 3.9
- PyTorch >= 2.0
- diffusers >= 0.25
- accelerate >= 0.25

## Quick Start

### Single image directory

```bash
python -m misbind.core \
  --image_dir ./path/to/images \
  --model sd1.5 \
  --lr 0.01 \
  --lr_p 0.01 \
  --timesteps 200 \
  --grad_accum 10 \
  --prompt_timesteps 200 \
  --prompt_grad_accum 10 \
  --original_word "sks person" \
  --seed 42
```

### Batch processing (multiple subjects)

```bash
python -m misbind.batch \
  --data_dir ./data/cleandata \
  --output_dir ./outputs/MisBind \
  --model sd1.5 \
  --persons "person1,person2,person3"
```

See [examples/run.sh](examples/run.sh) for more usage examples.

## Supported Models

| Model Key | HuggingFace Repo | Resolution |
|-----------|-----------------|------------|
| `sd1.5` | `stable-diffusion-v1-5/stable-diffusion-v1-5` | 512×512 |
| `2.1base` | `Manojb/stable-diffusion-2-1-base` | 512×512 |
| `xlbase-1.0` | `stabilityai/stable-diffusion-xl-base-1.0` | 1024×1024 |

## Key Arguments

### Stage I (Prompt Optimization)
- `--lr_p`: Learning rate for prompt embedding optimization
- `--prompt_timesteps`: Total diffusion timestep samples for prompt stage
- `--prompt_grad_accum`: Gradient accumulation steps for prompt stage
- `--num_pseudo_tokens`: Number of pseudo-tokens to learn (default: 4)
- `--attn_intensity_weight`: Attention intensity penalty weight

### Stage II (Image Perturbation)
- `--lr`: PGD step size
- `--timesteps`: Total diffusion timestep samples for image stage
- `--grad_accum`: Gradient accumulation steps for image stage
- `--pgd_eps`: L-infinity perturbation budget (default: 0.05)
- `--stage2_gamma`: Balance coefficient between class disruption and pseudo-token alignment
- `--original_word`: Original class word for Stage II (e.g., "sks person")

### Surrogate UNet (ASPL)
- `--surrogate_train_dir`: Path to set_A images for surrogate training
- `--surrogate_train_steps`: Surrogate UNet training steps
- `--surrogate_learning_rate`: Surrogate UNet learning rate
- `--outer_loop_steps`: Number of ASPL outer loop iterations

### DreamBooth Evaluation
- `--run_dreambooth`: Enable automatic DreamBooth training after perturbation
- `--dreambooth_class_dir`: Class images for prior preservation
- `--dreambooth_instance_prompt`: Instance prompt for DreamBooth
- (plus other standard DreamBooth parameters)

## Directory Structure

```
MisBind/
├── misbind/
│   ├── __init__.py
│   ├── core.py          # Main algorithm
│   └── batch.py         # Batch processing script
├── scripts/
│   ├── train_dreambooth.py   # DreamBooth training (HuggingFace)
│   └── sd15_inference.py     # DreamBooth inference/sampling
├── examples/
│   └── run.sh           # Example commands
├── requirements.txt
├── setup.py
└── README.md
```

## License

This source code is made available for research purposes only.
