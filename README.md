# CSCF

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

## Training

```bash
python train.py \
  --device cuda \
  --data_root data \
  --dataset SUNRGBD \
  --epochs 50 \
  --test_type all \
  --topk 1
```

## Testing

```bash
python test.py \
  --device cuda \
  --data_root data \
  --dataset SUNRGBD \
  --test_type all \
  --text_features clipExtend
```
