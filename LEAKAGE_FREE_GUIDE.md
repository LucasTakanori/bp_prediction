# 🔒 Leakage-Free Training Guide

## Problem Solved
Your scripts now use **subject-level train/val/test splits** to prevent data leakage with minimal code changes.

## 🧪 Three-Way Split Strategy
- **Train (80%)**: For model training
- **Validation (10%)**: For early stopping and hyperparameter tuning  
- **Test (10%)**: **ISOLATED** for final evaluation - never seen during training

## Quick Usage

### 1. Train VAE
```bash
python scripts/train_vae_multisubject_enhanced.py \
    --config configs/vae_multisubject_fixed.yaml \
    --split-seed 42 \
    --train-ratio 0.8 \
    --val-ratio 0.1 \
    --test-ratio 0.1
```

### 2. Train BiLSTM (SAME SEED!)
```bash
python scripts/train_bilstm_multisubject_enhanced.py \
    --config configs/bilstm_multisubject.yaml \
    --vae-checkpoint /path/to/vae/best_model.pt \
    --split-seed 42 \
    --train-ratio 0.8 \
    --val-ratio 0.1 \
    --test-ratio 0.1
```

### 3. Final Evaluation (Automatic!)
The BiLSTM training script **automatically evaluates on the test set** at the end of training - no separate script needed!

## Key Arguments
- `--split-seed 42`: **CRITICAL** - Same seed for both scripts
- `--train-ratio 0.8`: 80% subjects for training  
- `--val-ratio 0.1`: 10% subjects for validation
- `--test-ratio 0.1`: 10% subjects for test (**ISOLATED**)

## Guarantees
✅ No subject overlap between train/val/test
✅ Same subjects used in VAE and BiLSTM
✅ Test subjects completely isolated
✅ Final evaluation on truly unseen data
✅ Reproducible results with same seed
✅ Automatic validation of splits

## What Changed
- Added subject-level splitting functions
- Added new command line arguments
- Replaced `random_split()` with subject splits
- Enhanced logging and metadata

**Everything else stays the same!** 