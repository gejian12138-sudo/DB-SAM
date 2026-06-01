# DB-SAM: Dual-stream Boundary-Awareness SAM

Official implementation of DB-SAM for medical image segmentation.

## Overview

DB-SAM extends SAM (ViT-B) with the following strategies:

| Strategy | Description |
|----------|-------------|
| **GSA** (Global Semantic Adaptation) | LoRA injection + Trainable Normalization on SAM ViT-B |
| **MDP** (Multi-Scale Detail Perception) | Lightweight CNN branch for fine-grained detail extraction |
| **DCA** (Dual-stream Collaborative Attention) | Adaptive fusion of ViT and CNN heterogeneous features |
| **PICA** (Prompt-Image Cross-Attention) | Intermediate prompt injection at each encoder fusion stage |
| **MBA** (Multi-level Boundary Awareness) | Cross-scale attention with Laplacian boundary refinement |

## Project Structure

```
├── config.py              # Configuration
├── dataset.py             # Dataset + augmentation
├── train.py               # Training script
├── test.py                # Evaluation script
├── requirements.txt       # Python dependencies
├── models/
│   ├── db_sam.py          # DB-SAM model + PICA
│   ├── encoder.py         # DualStreamEncoder (GSA, MDP, DCA)
│   ├── decoder.py         # MBADecoder (MBA, bridge adapters)
│   └── segment_anything/  # SAM core (ViT, prompt encoder, mask decoder)
└── utils/
    ├── loss.py            # Focal + Dice loss
    ├── metrics.py         # IoU / Dice metrics
    └── utils.py           # Helpers (preprocess, box extraction)
```

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download pretrained weights

**SAM ViT-B backbone** (required for training):

```bash
mkdir -p checkpoints
wget -O checkpoints/sam_vit_b_01ec64.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

**Trained DB-SAM checkpoint** (for evaluation):

| File | Link |
|------|------|
| `DB_SAM_best.pth` | [Google Drive](https://drive.google.com/file/d/1TRD5DvG7a53fijn7QnuNt8sjj0N2UbKM/view?usp=drive_link) |

Place the trained checkpoint at `workdir/model_train/DB_SAM/DB_SAM_best.pth`.

### 3. Prepare dataset

Download the polyp segmentation datasets:

| Dataset | Link | Train / Test |
|---------|------|--------------|
| Kvasir-SEG | [datasets.simula.no/kvasir-seg](https://datasets.simula.no/kvasir-seg/) | 900 / 100 |
| CVC-ClinicDB | [polyp.grand-challenge.org](https://polyp.grand-challenge.org/CVCClinicDB/) | 550 / 62 |
| CVC-ColonDB | [polyp.grand-challenge.org](https://polyp.grand-challenge.org/) | — / 380 |
| CVC-300 | [polyp.grand-challenge.org](https://polyp.grand-challenge.org/) | — / 60 |
| ETIS-LaribPolypDB | [polyp.grand-challenge.org](https://polyp.grand-challenge.org/) | — / 196 |

Organize data as:

```
dataset/data/
├── train/
│   ├── imgs/          # 900 Kvasir + 550 ClinicDB images (.png / .jpg)
│   └── gts/           # ground-truth masks (.png)
└── test/
    └── TestDataset/
        ├── Kvasir/
        │   ├── imgs/
        │   └── gts/
        ├── CVC-ClinicDB/
        │   ├── imgs/
        │   └── gts/
        ├── CVC-ColonDB/
        │   ├── imgs/
        │   └── gts/
        ├── CVC-300/
        │   ├── imgs/
        │   └── gts/
        └── ETIS-LaribPolypDB/
            ├── imgs/
            └── gts/
```

## Training

```bash
python train.py \
    --run_name DB_SAM \
    --epochs 200 \
    --batch_size 3 \
    --lr 1e-4 \
    --grad_accum_steps 1 \
    --device cuda:0
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--run_name` | `DB_SAM` | Experiment name for logs and checkpoints |
| `--epochs` | `200` | Number of training epochs |
| `--batch_size` | `3` | Batch size per GPU |
| `--lr` | `1e-4` | Learning rate |
| `--grad_accum_steps` | `1` | Gradient accumulation steps |
| `--resume` | `None` | Resume from checkpoint path |
| `--num_workers` | `16` | DataLoader workers |
| `--device` | `cuda:0` | Training device |

Outputs are saved to `workdir/`:
- `workdir/model_train/{run_name}/DB_SAM_best.pth` — best checkpoint
- `workdir/model_train/{run_name}/DB_SAM_latest.pth` — latest checkpoint
- `workdir/logs/` — training logs
- `workdir/{run_name}_total_loss.png` — total loss curve
- `workdir/{run_name}_focal_dice_loss.png` — focal/dice loss curves

## Evaluation

```bash
python test.py \
    --checkpoint workdir/model_train/DB_SAM/DB_SAM_best.pth \
    --data_root ./dataset/data/test/TestDataset \
    --device cuda
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--checkpoint` | `workdir/model_train/DB_SAM/DB_SAM_best.pth` | Model checkpoint path |
| `--data_root` | `./dataset/data/test/TestDataset` | Test dataset root |
| `--datasets` | All 5 | Specific datasets (e.g. `CVC-300 Kvasir`) |
| `--batch_size` | `10` | Test batch size |
| `--metrics` | `iou dice` | Evaluation metrics |

Outputs per dataset:
- `workdir/{run_name}/{dataset}/results/` — predicted masks
- `workdir/{run_name}/{dataset}/result_ranking.csv` — per-sample metrics
- `workdir/{run_name}/summary_*.csv` — aggregated results
