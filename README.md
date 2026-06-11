# CSCF — Cross-modal Semantic Consistency Fusion

This repository implements **Cross-modal Semantic Consistency Fusion (CSCF)** for **Multi-Modal Heterogeneous Category-set Learning (MMHCL)** on multi-modal data. The model projects each modality into a shared semantic space (derived from CLIP text features), then fuses the modalities via uncertainty-guided similarity-based logit combination.

## Supported Datasets

- **ActivityNet** — video+audio action recognition
- **Food101** — food image recognition
- **SUNRGBD** — RGB+depth scene recognition

## Data Preparation

Each dataset follows this directory layout under `data/`:

```
data/
└── <DatasetName>/
    ├── similarity_matrix.mat              # inter-class similarity matrix
    ├── <DatasetName>1/
    │   ├── train_data.mat                 # modality A train data
    │   └── test_data.mat                  # modality A test data
    └── <DatasetName>2/
        ├── train_data.mat                 # modality B train data
        └── test_data.mat                  # modality B test data
```

Each `.mat` file should contain:

| Field | Description |
|---|---|
| `audio_att` | Modality A features (e.g., audio/RGB) |
| `video_att` | Modality B features (e.g., video/depth) |
| `text_att` / `extend_text_att` | Semantic attributes (CLIP text features) |
| `label` | Class labels (1-indexed) |

The `similarity_matrix.mat` should contain:
- `similarity_text` — cosine similarity matrix between class text embeddings (for `--text_features clip`)
- `similarity_extend_text` — extended similarity matrix (for `--text_features clipExtend`)

## Environment Setup

```bash
conda create -n CSCF python=3.8
conda activate CSCF
```

Install PyTorch 1.12.0 (CUDA 11.3):

```bash
pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113
```

Then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

## Model Architecture

The CSCF model has three main components:

### 1. Single-Mode Module (`SingleModeModule`)
- Projects each modality to the **semantic (text) space** via `Visual2CommonProj`
- Uses **4 parallel sub-networks** (ensemble) with progressively larger hidden dimensions
- Each sub-network computes logits by cosine similarity between projected visual features and semantic prototypes
- Trained with cross-entropy loss on each sub-network + mean-fused logits

### 2. Fusion Module (`FusionModule`)
- Combines logits from multiple modalities using **uncertainty estimation**
- Uncertainty = cross-modal entropy inconsistency + intra-modal sub-network entropy variance
- **Similarity-guided fusion**: uses inter-class similarity matrix to augment logits of the less-uncertain modality with the more-uncertain one
- Fuses with top-k similarity sparsification

### 3. CSCF Wrapper
- Orchestrates `SingleModeModule`s (one per modality) and the `FusionModule`
- Training loss = sum of single-mode losses + fusion loss

## Training

```bash
python train.py \
  --device cuda \
  --data_root data \
  --dataset SUNRGBD \
  --epochs 50 \
  --test_type all \
  --topk 1 \
  --text_features clipExtend
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--seed` | `42` | Random seed |
| `--device` | `cuda` | Device (`cuda` or `cpu`) |
| `--data_root` | `data` | Path to data directory |
| `--save_dir` | `trained_models` | Directory to save trained models |
| `--dataset` | `UCF` | Dataset: `ActivityNet`, `Food101`, `SUNRGBD` |
| `--test_type` | `same` | Evaluation protocol (see below) |
| `--text_features` | `clipExtend` | Text feature type: `clip` or `clipExtend` |
| `--epochs` | `50` | Number of training epochs |
| `--weight_decay` | `0.0001` | Weight decay for Adam optimizer |
| `--topk` | `1` | Top-k for similarity matrix sparsification |

The learning rate is set per dataset: `ActivityNet=3e-3`, `Food101=1e-3`, `SUNRGBD=1e-4`.

## Testing

Test a pre-trained model:

```bash
python test.py \
  --device cuda \
  --data_root data \
  --dataset SUNRGBD \
  --test_type all \
  --text_features clipExtend
```

The script loads from `trained_models/<dataset>_<text_features>.pth`.

## Evaluation Protocols (`--test_type`)

| Type | Description |
|---|---|
| `same` | **Seen classes only**: A seen classes (A_s) + B seen classes (B_s) |
| `contra` | **Unseen classes only**: A unseen (A_u) + B unseen (B_u) |
| `data1` | A seen classes + B unseen classes (A_s + B_u) |
| `data2` | A unseen classes + B seen classes (A_u + B_s) |
| `total` | All classes from both modalities |
| `mixture` | Mixed seen+unseen from both modalities |
| `all` | Runs all of the above sequentially |

In the output, **A** and **B** denote the two modalities (e.g., audio/video or RGB/depth).

## Key Files

| File | Description |
|---|---|
| `CSCF.py` | Main model: `MLP`, `Visual2CommonProj`, `SingleModeModule`, `FusionModule`, `CSCF` |
| `FusionLogits.py` | Uncertainty computation and similarity-based logit fusion |
| `MultiViewDataset.py` | Multi-view data loading and dataset construction |
| `train.py` | Training entry point |
| `test.py` | Testing/evaluation entry point |
