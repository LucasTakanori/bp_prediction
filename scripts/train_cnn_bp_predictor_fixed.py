#!/usr/bin/env python3
"""
CNN-Based Blood Pressure Prediction Training Script (Fixed)

This script implements a custom CNN feature extractor instead of VAE for BP prediction.
FIXED: Tensor dimension mismatch in input_projection layer.
"""

import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import yaml
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Import utilities
from utils.data_utils import PviDataset, DataPathManager


class CNNBPDataset(Dataset):
    """Dataset for CNN-based BP prediction with proper normalization"""
    
    def __init__(self, data_root: str, pattern_offsets: List[int], 
                 max_samples_per_subject: int = 1000, sequence_step_size: int = 15,
                 use_augmentation: bool = False, noise_level: float = 0.0):
        self.data_root = data_root
        self.pattern_offsets = pattern_offsets
        self.max_samples_per_subject = max_samples_per_subject
        self.sequence_step_size = sequence_step_size
        self.use_augmentation = use_augmentation
        self.noise_level = noise_level
        
        print(f"Loading data using PviDataset from: {data_root}")
        self.pvi_dataset = PviDataset(data_root)
        
        self.sequences = []
        self.targets = []
        
        print(f"Loaded {len(self.pvi_dataset)} samples from PviDataset")
        
        # Process samples with proper sampling strategy
        for sample_idx in range(min(len(self.pvi_dataset), self.max_samples_per_subject)):
            sample = self.pvi_dataset[sample_idx]
            
            # Extract PVI images and BP signal
            pvi_img = sample['pviHP']['img']  # [32, 32, num_frames]
            bp_signal = sample['bp']['signal']  # [num_frames]
            
            num_frames = pvi_img.shape[-1]
            
            # Determine valid central indices with proper step size
            min_offset = min(self.pattern_offsets)
            max_offset = max(self.pattern_offsets)
            valid_start = max(0, -min_offset)
            valid_end = min(num_frames, num_frames - max_offset)
            
            # Use step size to reduce overlap
            central_indices = list(range(valid_start, valid_end, self.sequence_step_size))
            
            print(f"Sample {sample_idx}: Creating {len(central_indices)} sequences (step={self.sequence_step_size})")
            
            for central_idx in central_indices:
                # Create sequence using pattern offsets
                seq_frames = []
                valid_sequence = True
                
                for offset in self.pattern_offsets:
                    frame_idx = central_idx + offset
                    if 0 <= frame_idx < num_frames:
                        # Extract frame and normalize for CNN
                        frame = pvi_img[:, :, frame_idx]  # [32, 32]
                        frame = torch.tensor(frame, dtype=torch.float32)
                        
                        # Clean NaN values
                        frame = torch.nan_to_num(frame, nan=0.0)
                        
                        # Standard 0-1 normalization for CNN
                        frame_min = frame.min()
                        frame_max = frame.max()
                        if frame_max > frame_min:
                            frame = (frame - frame_min) / (frame_max - frame_min)
                        
                        # Add small noise for regularization if enabled
                        if self.use_augmentation:
                            noise = torch.randn_like(frame) * self.noise_level
                            frame = torch.clamp(frame + noise, 0, 1)
                        
                        # Add channel dimension: [1, 32, 32]
                        frame = frame.unsqueeze(0)
                        seq_frames.append(frame)
                    else:
                        valid_sequence = False
                        break
                
                if valid_sequence:
                    sequence = torch.stack(seq_frames)  # [seq_len, 1, 32, 32]
                    
                    # Use BP signal for current frame as target
                    if bp_signal.dim() == 1:
                        target_bp = bp_signal
                    else:
                        target_bp = bp_signal[central_idx] if central_idx < bp_signal.shape[0] else bp_signal[0]
                    
                    # Ensure target is exactly 50 samples
                    if target_bp.dim() > 1:
                        target_bp = target_bp.flatten()
                    
                    current_length = target_bp.shape[0]
                    if current_length != 50:
                        if current_length > 50:
                            indices = torch.linspace(0, current_length - 1, 50).long()
                            target_bp = target_bp[indices]
                        else:
                            target_bp = torch.nn.functional.interpolate(
                                target_bp.unsqueeze(0).unsqueeze(0), 
                                size=50, 
                                mode='linear', 
                                align_corners=False
                            ).squeeze()
                    
                    # Normalize BP to 0-1 range for training stability
                    # Expected BP range: 40-200 mmHg
                    target_bp = target_bp.float()
                    target_bp = torch.clamp(target_bp, 40.0, 200.0)  # Clamp to valid range
                    target_bp = (target_bp - 40.0) / (200.0 - 40.0)  # Normalize to [0, 1]
                    
                    self.sequences.append(sequence)
                    self.targets.append(target_bp)
        
        print(f"Created {len(self.sequences)} sequences with step size {self.sequence_step_size}")
        if len(self.sequences) > 0:
            print(f"Sequence shape: {self.sequences[0].shape}")
            print(f"Target shape: {self.targets[0].shape}")
            print(f"BP range: {self.targets[0].min():.1f} - {self.targets[0].max():.1f} mmHg")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return {
            'sequences': self.sequences[idx],
            'targets': self.targets[idx]
        }


class PVIFeatureExtractor(nn.Module):
    """Custom lightweight CNN for PVI feature extraction"""
    
    def __init__(self, input_channels=1, output_dim=128, use_batch_norm=True, dropout_rate=0.2):
        super(PVIFeatureExtractor, self).__init__()
        
        self.features = nn.Sequential(
            # Conv Block 1: 32x32 -> 16x16
            nn.Conv2d(input_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32) if use_batch_norm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32) if use_batch_norm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 16x16
            
            # Conv Block 2: 16x16 -> 8x8
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64) if use_batch_norm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64) if use_batch_norm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 8x8
            
            # Conv Block 3: 8x8 -> 4x4
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128) if use_batch_norm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4))  # 4x4
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, output_dim)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class CNNBPPredictor(nn.Module):
    """CNN-based BP predictor with end-to-end training"""
    
    def __init__(self, cnn_config: dict, bilstm_config: dict, attention_config: dict,
                 pattern_offsets: List[int], auxiliary_heads: dict = None):
        super(CNNBPPredictor, self).__init__()
        
        self.pattern_offsets = pattern_offsets
        self.use_attention = attention_config.get('use_attention', True)
        self.use_auxiliary_heads = auxiliary_heads.get('use_auxiliary_heads', True) if auxiliary_heads else False
        self.use_physiological_features = attention_config.get('use_physiological_features', True)
        
        # CNN Feature Extractor
        self.feature_extractor = PVIFeatureExtractor(
            input_channels=cnn_config.get('input_channels', 1),
            output_dim=cnn_config.get('feature_dim', 128),
            use_batch_norm=cnn_config.get('use_batch_norm', True),
            dropout_rate=cnn_config.get('dropout_rate', 0.2)
        )
        
        # Input projection
        feature_dim = cnn_config.get('feature_dim', 128)
        hidden_dim = bilstm_config.get('hidden_dim', 128)
        
        # Add physiological features if enabled
        input_dim = feature_dim
        if self.use_physiological_features:
            input_dim += 16  # Additional physiological features
        
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(bilstm_config.get('dropout_rate', 0.3) * 0.5)
        )
        
        # BiLSTM
        self.bilstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=bilstm_config.get('num_layers', 2),
            batch_first=True,
            bidirectional=bilstm_config.get('bidirectional', True),
            dropout=bilstm_config.get('dropout_rate', 0.3) if bilstm_config.get('num_layers', 2) > 1 else 0
        )
        
        # Attention mechanism
        if self.use_attention:
            self.attention = MultiHeadAttention(
                hidden_dim=hidden_dim * 2 if bilstm_config.get('bidirectional', True) else hidden_dim,
                num_heads=attention_config.get('num_attention_heads', 4),
                dropout=attention_config.get('attention_dropout', 0.2),
                current_frame_bias=attention_config.get('current_frame_bias', 1.5)
            )
        
        # Output projection
        lstm_output_dim = hidden_dim * 2 if bilstm_config.get('bidirectional', True) else hidden_dim
        self.output_projection = nn.Sequential(
            nn.Linear(lstm_output_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(bilstm_config.get('dropout_rate', 0.3))
        )
        
        # Waveform prediction head
        self.waveform_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, bilstm_config.get('output_dim', 50)),
            nn.Sigmoid()  # Output [0,1] for normalized BP values
        )
        
        # Auxiliary heads
        if self.use_auxiliary_heads:
            self.systolic_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 4),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim // 4, 1),
                nn.Sigmoid()  # Output [0,1] for normalized BP values
            )
            
            self.diastolic_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 4),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim // 4, 1),
                nn.Sigmoid()  # Output [0,1] for normalized BP values
            )
    
    def forward(self, x_seq, return_attention=False):
        batch_size, seq_len, channels, height, width = x_seq.shape
        
        # Extract features for each frame
        features = []
        for i in range(seq_len):
            frame_features = self.feature_extractor(x_seq[:, i])  # [batch, feature_dim]
            features.append(frame_features)
        
        features = torch.stack(features, dim=1)  # [batch, seq_len, feature_dim]
        
        # Add physiological features if enabled
        if self.use_physiological_features:
            # Extract meaningful physiological features from the waveform sequence
            # Use statistical features: mean, std, min, max for each frame
            phys_features = []
            for i in range(seq_len):
                frame_stats = torch.stack([
                    features[:, i].mean(dim=-1, keepdim=True),  # mean
                    features[:, i].std(dim=-1, keepdim=True),   # std
                    features[:, i].min(dim=-1, keepdim=True)[0], # min
                    features[:, i].max(dim=-1, keepdim=True)[0], # max
                ], dim=-1)  # [batch, 4]
                
                # Add frame position encoding
                position_encoding = torch.ones(batch_size, 1, device=features.device) * (i / seq_len)
                
                # Combine stats and position
                frame_phys = torch.cat([frame_stats, position_encoding], dim=-1)  # [batch, 5]
                phys_features.append(frame_phys)
            
            phys_features = torch.stack(phys_features, dim=1)  # [batch, seq_len, 5]
            
            # Pad to 16 features with zeros for compatibility
            padding = torch.zeros(batch_size, seq_len, 11, device=features.device)
            phys_features = torch.cat([phys_features, padding], dim=-1)  # [batch, seq_len, 16]
            
            features = torch.cat([features, phys_features], dim=-1)
        
        # Input projection
        features = self.input_projection(features)
        
        # BiLSTM
        lstm_out, _ = self.bilstm(features)
        
        # Attention
        attention_weights = None
        if self.use_attention:
            attended_features, attention_weights = self.attention(lstm_out, self.pattern_offsets)
            features = attended_features
        else:
            # Use middle frame or average pooling
            middle_idx = len(self.pattern_offsets) // 2
            features = lstm_out[:, middle_idx]
        
        # Output projection
        features = self.output_projection(features)
        
        # Predictions
        waveform_pred = self.waveform_head(features)
        
        outputs = {'waveform': waveform_pred}
        
        if self.use_auxiliary_heads:
            outputs['systolic'] = self.systolic_head(features)
            outputs['diastolic'] = self.diastolic_head(features)
        
        if return_attention and attention_weights is not None:
            outputs['attention_weights'] = attention_weights
        
        return outputs


class MultiHeadAttention(nn.Module):
    """Multi-head attention for temporal sequence processing"""
    
    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.2, 
                 current_frame_bias: float = 1.5):
        super(MultiHeadAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.current_frame_bias = current_frame_bias
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        self.q_linear = nn.Linear(hidden_dim, hidden_dim)
        self.k_linear = nn.Linear(hidden_dim, hidden_dim)
        self.v_linear = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x, pattern_offsets=None):
        batch_size = x.size(0)
        seq_len = x.size(1)
        
        # Linear transformations
        Q = self.q_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = self.k_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = self.v_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Transpose for attention computation
        Q = Q.transpose(1, 2)  # [batch, heads, seq_len, head_dim]
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # Compute attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        # Apply current frame bias
        if pattern_offsets is not None:
            current_frame_idx = None
            for i, offset in enumerate(pattern_offsets):
                if offset == 0:
                    current_frame_idx = i
                    break
            
            if current_frame_idx is not None:
                scores[:, :, :, current_frame_idx] *= self.current_frame_bias
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention
        attended = torch.matmul(attention_weights, V)
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.hidden_dim
        )
        
        output = self.out(attended)
        
        # Return middle frame or weighted average
        if pattern_offsets is not None and len(pattern_offsets) > 1:
            middle_idx = len(pattern_offsets) // 2
            output = output[:, middle_idx]
        else:
            output = output.mean(dim=1)
        
        return output, attention_weights.mean(dim=1)  # Average over heads


class CNNBPLoss(nn.Module):
    """Composite loss for CNN-based BP prediction"""
    
    def __init__(self, waveform_weight=0.4, systolic_weight=0.3, diastolic_weight=0.3, 
                 huber_delta=1.0, physiological_constraint=True, pulse_pressure_weight=0.05):
        super(CNNBPLoss, self).__init__()
        self.waveform_weight = waveform_weight
        self.systolic_weight = systolic_weight
        self.diastolic_weight = diastolic_weight
        self.huber_delta = huber_delta
        self.physiological_constraint = physiological_constraint
        self.pulse_pressure_weight = pulse_pressure_weight
        
    def forward(self, predictions, targets):
        losses = {}
        
        # Waveform loss (Huber)
        waveform_pred = predictions['waveform']
        waveform_target = targets
        
        waveform_loss = F.huber_loss(waveform_pred, waveform_target, delta=self.huber_delta)
        losses['waveform'] = waveform_loss
        
        # Extract BP values
        pred_sys, pred_dias = self.extract_bp_values(waveform_pred)
        true_sys, true_dias = self.extract_bp_values(waveform_target)
        
        # Systolic and diastolic losses
        if 'systolic' in predictions:
            sys_loss = F.huber_loss(predictions['systolic'].squeeze(), pred_sys, delta=self.huber_delta)
        else:
            sys_loss = F.huber_loss(pred_sys, true_sys, delta=self.huber_delta)
        losses['systolic'] = sys_loss
        
        if 'diastolic' in predictions:
            dias_loss = F.huber_loss(predictions['diastolic'].squeeze(), pred_dias, delta=self.huber_delta)
        else:
            dias_loss = F.huber_loss(pred_dias, true_dias, delta=self.huber_delta)
        losses['diastolic'] = dias_loss
        
        # Physiological constraint
        if self.physiological_constraint:
            pred_pp = pred_sys - pred_dias
            true_pp = true_sys - true_dias
            pp_loss = F.huber_loss(pred_pp, true_pp, delta=self.huber_delta)
            losses['pulse_pressure'] = pp_loss
        
        # Composite loss
        total_loss = (self.waveform_weight * waveform_loss + 
                     self.systolic_weight * sys_loss + 
                     self.diastolic_weight * dias_loss)
        
        if self.physiological_constraint:
            total_loss += self.pulse_pressure_weight * pp_loss
        
        losses['total'] = total_loss
        return losses
    
    def extract_bp_values(self, waveform):
        """Extract systolic and diastolic values from normalized waveform [0,1]"""
        batch_size = waveform.shape[0]
        device = waveform.device
        
        systolic_values = torch.zeros(batch_size, device=device)
        diastolic_values = torch.zeros(batch_size, device=device)
        
        for i in range(batch_size):
            signal = waveform[i]
            
            # Systolic: maximum value
            sys_val, sys_idx = torch.max(signal, dim=0)
            systolic_values[i] = sys_val
            
            # Diastolic: minimum in second half
            search_start = max(sys_idx.item() + 1, len(signal) // 2)
            if search_start < len(signal):
                dias_val, _ = torch.min(signal[search_start:], dim=0)
                diastolic_values[i] = dias_val
            else:
                diastolic_values[i] = torch.min(signal)
        
        return systolic_values, diastolic_values


def load_config_with_timestamp(config_path):
    """Load YAML configuration file and replace dynamic placeholders"""
    import datetime
    
    with open(config_path, 'r') as file:
        config_text = file.read()
    
    # Replace timestamp placeholder with current datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    config_text = config_text.replace('{timestamp}', timestamp)
    
    # Parse the modified YAML
    config = yaml.safe_load(config_text)
    return config


def main():
    parser = argparse.ArgumentParser(description='CNN-Based BP Predictor Training (Fixed)')
    
    parser.add_argument('--config', type=str, required=True,
                       help='Path to YAML configuration file')
    
    args = parser.parse_args()
    
    # Load configuration
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")
    
    print(f"Loading configuration from: {args.config}")
    config = load_config_with_timestamp(args.config)
    
    # Extract configuration sections
    data_config = config.get('data_config', {})
    model_config = config.get('model_config', {})
    training_config = config.get('training_config', {})
    loss_config = config.get('loss_config', {})
    output_config = config.get('output_config', {})
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = output_config.get('output_dir', './experiments/cnn_bp_predictor')
    os.makedirs(output_dir, exist_ok=True)
    checkpoints_dir = os.path.join(output_dir, 'checkpoints')
    results_dir = os.path.join(output_dir, 'results')
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    print("="*60)
    print("CNN-BASED BP PREDICTOR TRAINING (FIXED)")
    print("="*60)
    print("Key features:")
    print("- Custom CNN feature extractor")
    print("- End-to-end training (no frozen components)")
    print("- Task-specific optimization")
    print("- Multi-frame temporal modeling")
    print("- FIXED: Tensor dimension mismatch")
    print("="*60)
    
    # Setup data
    data_root = data_config.get('root_path', '/home/lucas_takanori/phd/data')
    path_manager = DataPathManager(
        subject="subject001",
        session="baseline",
        root=data_root
    )
    h5_file_path = str(path_manager._h5_path)
    
    print(f"Data file: {h5_file_path}")
    print(f"Pattern offsets: {data_config.get('pattern_offsets')}")
    print(f"Output directory: {output_dir}")
    print("="*60)
    
    # Create dataset
    print("\nCreating CNN dataset...")
    dataset = CNNBPDataset(
        data_root=h5_file_path,
        pattern_offsets=data_config.get('pattern_offsets'),
        max_samples_per_subject=data_config.get('max_samples_per_subject', 1000),
        sequence_step_size=data_config.get('sequence_step_size', 15),
        use_augmentation=data_config.get('use_augmentation', False),
        noise_level=data_config.get('noise_level', 0.0)
    )
    
    # Split dataset
    train_split = data_config.get('train_split', 0.8)
    val_split = data_config.get('val_split', 0.1)
    
    train_size = int(train_split * len(dataset))
    val_size = int(val_split * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    # Create data loaders
    batch_size = training_config.get('batch_size', 8)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Create model
    print("\nCreating CNN-based BP predictor...")
    model = CNNBPPredictor(
        cnn_config=model_config.get('cnn_config', {}),
        bilstm_config=model_config.get('bilstm_config', {}),
        attention_config=model_config.get('attention_config', {}),
        pattern_offsets=data_config.get('pattern_offsets'),
        auxiliary_heads=model_config.get('auxiliary_heads', {})
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    loss_weights = loss_config.get('loss_weights', {})
    criterion = CNNBPLoss(
        waveform_weight=loss_weights.get('waveform_weight', 0.4),
        systolic_weight=loss_weights.get('systolic_weight', 0.3),
        diastolic_weight=loss_weights.get('diastolic_weight', 0.3),
        huber_delta=loss_config.get('loss_params', {}).get('huber_delta', 1.0),
        physiological_constraint=loss_config.get('physiological_constraint', True),
        pulse_pressure_weight=loss_config.get('pulse_pressure_weight', 0.05)
    )
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config.get('learning_rate', 0.001)),
        weight_decay=float(training_config.get('weight_decay', 1e-4)),
        betas=training_config.get('optimizer_config', {}).get('betas', [0.9, 0.999]),
        amsgrad=training_config.get('optimizer_config', {}).get('amsgrad', True)
    )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        patience=int(training_config.get('scheduler_config', {}).get('scheduler_patience', 6)),
        factor=float(training_config.get('scheduler_config', {}).get('factor', 0.7)),
        min_lr=float(training_config.get('scheduler_config', {}).get('min_lr', 1e-6)),
        verbose=True
    )
    
    # Training loop
    num_epochs = training_config.get('num_epochs', 80)
    print(f"\nStarting training for {num_epochs} epochs...")
    
    best_val_mae = float('inf')
    training_history = {
        'train_loss': [],
        'val_loss': [],
        'val_mae': [],
        'learning_rates': []
    }
    
    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        
        # Training
        model.train()
        train_losses = []
        
        for batch in tqdm(train_loader, desc="Training"):
            sequences = batch['sequences'].to(device)
            targets = batch['targets'].to(device)
            
            optimizer.zero_grad()
            
            outputs = model(sequences)
            losses = criterion(outputs, targets)
            
            loss = losses['total']
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 
                                         training_config.get('optimization', {}).get('grad_clip_norm', 0.5))
            
            optimizer.step()
            train_losses.append(loss.item())
        
        avg_train_loss = np.mean(train_losses)
        
        # Validation
        model.eval()
        val_losses = []
        val_maes = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                sequences = batch['sequences'].to(device)
                targets = batch['targets'].to(device)
                
                outputs = model(sequences)
                losses = criterion(outputs, targets)
                
                val_losses.append(losses['total'].item())
                
                # Calculate MAE
                mae = F.l1_loss(outputs['waveform'], targets)
                val_maes.append(mae.item())
        
        avg_val_loss = np.mean(val_losses)
        avg_val_mae = np.mean(val_maes)
        
        # Update scheduler
        scheduler.step(avg_val_mae)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Save history
        training_history['train_loss'].append(avg_train_loss)
        training_history['val_loss'].append(avg_val_loss)
        training_history['val_mae'].append(avg_val_mae)
        training_history['learning_rates'].append(current_lr)
        
        print(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val MAE: {avg_val_mae:.4f}, LR: {current_lr:.2e}")
        
        # Save best model
        if avg_val_mae < best_val_mae:
            best_val_mae = avg_val_mae
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_mae': avg_val_mae,
                'config': config
            }, os.path.join(checkpoints_dir, 'best_model.pt'))
            print(f"🎉 New best model saved! Val MAE: {best_val_mae:.4f}")
        
        # Save checkpoint
        if epoch % training_config.get('checkpointing', {}).get('save_every_n_epochs', 5) == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'training_history': training_history,
                'config': config
            }, os.path.join(checkpoints_dir, f'checkpoint_epoch_{epoch}.pt'))
    
    print("\n" + "="*60)
    print("🎉 TRAINING COMPLETED!")
    print("="*60)
    print(f"Best validation MAE: {best_val_mae:.4f}")
    print(f"Models saved to: {checkpoints_dir}")
    print("="*60)


if __name__ == '__main__':
    main() 