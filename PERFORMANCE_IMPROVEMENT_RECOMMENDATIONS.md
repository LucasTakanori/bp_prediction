# Blood Pressure Prediction Performance Improvement Recommendations

## Current Performance Issues

Based on your evaluation results, there are several critical problems causing poor performance:

### Single Frame Model Results:
- **R² Systolic**: -5.06 (terrible - worse than predicting mean)
- **R² Diastolic**: -5.42 (terrible - worse than predicting mean)
- **MAE**: ~10-11 mmHg (clinically poor)
- **Systematic bias**: -10 to -11 mmHg (consistent underprediction)

### Seven Frame Model Results:
- **R² Systolic**: -3.76 (still terrible)
- **R² Diastolic**: -4.50 (still terrible)
- **MAE**: ~9-10 mmHg (still clinically poor)
- **Systematic bias**: -9 mmHg (consistent underprediction)

## Root Cause Analysis

### 1. **CRITICAL: Massive Data Leakage/Overfitting**
```
Single frame: 200,000 sequences from 400 samples (500 per sample)
Seven frame: Similar overfitting pattern
```
- **Problem**: Creating sequences from every frame creates extreme overlap
- **Impact**: Model memorizes training data but can't generalize
- **Solution**: Use proper step size (10+ frames) to reduce overlap

### 2. **Normalization Mismatch**
```python
# Current (problematic)
frame = torch.nan_to_num(frame, nan=0.0)  # Only NaN handling

# VAE training (correct)
frame = (frame - frame_min) / (frame_max - frame_min)  # [0,1] normalization
```
- **Problem**: VAE was trained on [0,1] normalized data, but you're feeding different ranges
- **Impact**: VAE features are corrupted, leading to poor predictions
- **Solution**: Match VAE training normalization exactly

### 3. **Poor Loss Function Weighting**
```yaml
# Current
waveform_weight: 0.6
systolic_weight: 0.2  
diastolic_weight: 0.2

# Recommended
waveform_weight: 0.4
systolic_weight: 0.3
diastolic_weight: 0.3
```
- **Problem**: Too much emphasis on waveform, not enough on clinical values
- **Solution**: Rebalance weights to emphasize systolic/diastolic prediction

### 4. **Insufficient Regularization**
- **Problem**: Model overfits despite high capacity
- **Solution**: Increase dropout, use better optimizer (AdamW), add regularization

## Recommended Solutions

### 1. **Fix Data Leakage (CRITICAL)**

```python
# OLD (problematic)
central_indices = list(range(valid_start, valid_end, 1))  # Every frame
# Creates 500 sequences per sample = massive overfitting

# NEW (fixed)
central_indices = list(range(valid_start, valid_end, 10))  # Every 10th frame
# Creates ~50 sequences per sample = proper sampling
```

### 2. **Fix Normalization (CRITICAL)**

```python
# NEW: VAE-compatible normalization
frame = torch.nan_to_num(frame, nan=0.0)
frame_min = frame.min()
frame_max = frame.max()
if frame_max > frame_min:
    frame = (frame - frame_min) / (frame_max - frame_min)  # [0,1] range
```

### 3. **Use Multi-Frame Temporal Modeling**

```yaml
# Recommended configuration
pattern_offsets: [-4, -3, -2, -1, 0, 1, 2]  # 7 frames for temporal context
sequence_length: 7
use_attention: true
```

### 4. **Improve Model Architecture**

```yaml
# Reduced complexity to prevent overfitting
hidden_dim: 128        # Reduced from 256
num_layers: 2          # Reduced from 3
num_attention_heads: 4 # Reduced from 8
dropout_rate: 0.4      # Increased from 0.3
```

### 5. **Better Loss Function**

```python
# Use Huber loss for robustness
waveform_loss = nn.functional.huber_loss(pred_waveform, targets, delta=1.0)
systolic_loss = nn.functional.huber_loss(pred_sys, target_sys, delta=1.0)
diastolic_loss = nn.functional.huber_loss(pred_dias, target_dias, delta=1.0)
```

### 6. **Better Training Configuration**

```yaml
# Improved training setup
optimizer: "adamw"              # Better than adam
learning_rate: 0.0005          # Higher initial LR
weight_decay: 1e-4             # Stronger regularization
scheduler_type: "reduce_on_plateau"  # Adaptive LR reduction
batch_size: 16                 # Larger batches
num_epochs: 50                 # More epochs for convergence
```

## Expected Performance Improvements

With these changes, you should expect:

### Target Performance (Clinically Acceptable):
- **R² Systolic**: > 0.7 (good correlation)
- **R² Diastolic**: > 0.6 (acceptable correlation)  
- **MAE**: < 5 mmHg (clinically excellent)
- **Clinical Accuracy**: 
  - 5mmHg tolerance: > 60%
  - 10mmHg tolerance: > 85%
  - 15mmHg tolerance: > 95%

### Systematic Bias Reduction:
- **Mean difference**: < ±2 mmHg (minimal bias)
- **Standard deviation**: < 5 mmHg (good precision)

## Implementation Steps

### Step 1: Use Improved Configuration
```bash
python scripts/train_improved_bp_predictor.py --config configs/sophisticated_bp_predictor_improved.yaml
```

### Step 2: Monitor Training Carefully
- Watch for overfitting (val_loss should decrease steadily)
- Monitor gradient norms (should be stable, not exploding)
- Check attention patterns (should focus on current frame)

### Step 3: Evaluate with Multiple Subjects
```bash
# Test on different subjects
python scripts/evaluate_bp_models.py \
  --model_path experiments/improved_bp_predictor/checkpoints/best_model.pt \
  --data_path /home/lucas_takanori/phd/data \
  --output_dir ./evaluation_results/improved_model \
  --model_name "Improved BP Predictor"
```

### Step 4: If Still Poor Performance, Try:

1. **Reduce Model Complexity Further**:
   ```yaml
   hidden_dim: 64
   num_layers: 1
   dropout_rate: 0.5
   ```

2. **Use Different Loss Functions**:
   ```yaml
   loss_type: "systolic_distance"  # Focus only on systolic
   # or
   loss_type: "diastolic_distance"  # Focus only on diastolic
   ```

3. **Try Different VAE Checkpoints**:
   - Check if your VAE was properly trained
   - Consider retraining VAE with better normalization

4. **Add More Regularization**:
   ```yaml
   weight_decay: 1e-3
   dropout_rate: 0.6
   use_augmentation: true
   noise_level: 0.01
   ```

## Key Files Created

1. **`configs/sophisticated_bp_predictor_improved.yaml`** - Fixed configuration
2. **`scripts/train_improved_bp_predictor.py`** - Improved training script (skeleton)
3. **`scripts/evaluate_bp_models.py`** - Already working evaluation script

## Next Steps

1. **Train with improved config**: Use the new configuration file
2. **Monitor training closely**: Watch for proper convergence
3. **Evaluate thoroughly**: Test on multiple subjects
4. **Iterate if needed**: Adjust hyperparameters based on results

The current negative R² values indicate fundamental issues that these changes should resolve. The improvements focus on the most critical problems: data leakage, normalization mismatch, and insufficient regularization. 