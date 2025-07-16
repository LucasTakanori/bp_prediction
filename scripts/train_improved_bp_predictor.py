#!/usr/bin/env python3
"""
Improved Blood Pressure Prediction Training Script

This script addresses the critical issues found in the previous version:
1. Massive data leakage/overfitting from too many overlapping sequences
2. Normalization mismatch between VAE training and BP prediction
3. Poor loss function weighting
4. Insufficient regularization

Key improvements:
- Proper sequence sampling with reduced overlap
- VAE-compatible normalization
- Better regularization and dropout
- Improved loss function with Huber loss
- Multi-frame temporal modeling
- Better hyperparameters
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


class ImprovedBPDataset(Dataset):
    """Improved dataset with proper sampling and normalization"""
    
    def __init__(self, data_root: str, pattern_offsets: List[int], 
                 max_samples_per_subject: int = 100, sequence_step_size: int = 10,
                 use_augmentation: bool = False, noise_level: float = 0.005):
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
            
            # CRITICAL: Use step size to reduce overlap and prevent overfitting
            central_indices = list(range(valid_start, valid_end, self.sequence_step_size))
            
            print(f"Sample {sample_idx}: Creating {len(central_indices)} sequences (step={self.sequence_step_size})")
            
            for central_idx in central_indices:
                # Create sequence using pattern offsets
                seq_frames = []
                valid_sequence = True
                
                for offset in self.pattern_offsets:
                    frame_idx = central_idx + offset
                    if 0 <= frame_idx < num_frames:
                        # Extract frame and apply VAE-compatible normalization
                        frame = pvi_img[:, :, frame_idx]  # [32, 32]
                        frame = torch.tensor(frame, dtype=torch.float32)
                        
                        # CRITICAL: Apply same normalization as VAE training
                        frame = torch.nan_to_num(frame, nan=0.0)
                        
                        # Normalize to [0, 1] like VAE training
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
                    
                    # Keep BP in raw mmHg values (no normalization)
                    target_bp = target_bp.float()
                    
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


# Enhanced model with better regularization
class ImprovedBPPredictor(nn.Module):
    """Improved BP predictor with better regularization and physiologically-informed features"""
    
    def __init__(self, vae_model, latent_dim: int = 64, 
                 hidden_dim: int = 128, num_layers: int = 2, num_heads: int = 4,
                 dropout: float = 0.4, use_attention: bool = True, 
                 pattern_offsets: List[int] = None, current_frame_bias: float = 1.5,
                 use_physiological_features: bool = True):
        super(ImprovedBPPredictor, self).__init__()
        
        self.vae = vae_model
        self.use_attention = use_attention
        self.use_physiological_features = use_physiological_features
        self.pattern_offsets = pattern_offsets or [-7,-6, -5,-4, -3, -2, -1, 0, 1, 2]
        
        # Freeze VAE parameters
        for param in self.vae.parameters():
            param.requires_grad = False
        
        # Enhanced input projection with physiological feature extraction
        input_dim = latent_dim
        if use_physiological_features:
            # Add features like pulse pressure, mean arterial pressure, etc.
            input_dim += 16  # Additional physiological features
            
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5)
        )
        
        # Physiological feature extractor (optional)
        if use_physiological_features:
            self.physio_extractor = nn.Sequential(
                nn.Linear(latent_dim, 32),
                nn.ReLU(inplace=True),
                nn.Linear(32, 16),
                nn.Tanh()  # Bounded activation for physiological features
            )
        
        # Bidirectional LSTM with reduced complexity
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        lstm_output_dim = hidden_dim * 2
        
        # Self-attention with reduced complexity
        if use_attention:
            self.attention = ImprovedMultiHeadAttention(
                hidden_dim=lstm_output_dim,
                num_heads=num_heads,
                dropout=dropout * 0.5,
                current_frame_bias=current_frame_bias
            )
        else:
            self.attention = None
        
        # Temporal processing
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(lstm_output_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        # Output layers with stronger regularization
        self.output_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5)
        )
        
        # Prediction heads with improved initialization
        self.waveform_head = nn.Linear(hidden_dim // 4, 50)
        
        # Enhanced SBP/DBP heads with residual connections
        self.bp_feature_extractor = nn.Sequential(
            nn.Linear(hidden_dim // 4, hidden_dim // 8),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.3)
        )
        
        self.systolic_head = nn.Sequential(
            nn.Linear(hidden_dim // 8, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1)
        )
        
        self.diastolic_head = nn.Sequential(
            nn.Linear(hidden_dim // 8, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1)
        )
        
        # Initialize weights for better SBP/DBP prediction
        self._initialize_bp_heads()
        
        # Initialize other key components for better gradient flow
        self._initialize_other_components()
    
    def _initialize_bp_heads(self):
        """Initialize BP prediction heads for full physiological range learning"""
        import math
        
        # Calculate fan_in for proper initialization
        fan_in = self.systolic_head[-1].in_features
        
        # Use Xavier/Glorot initialization for weights to allow full range learning
        std = math.sqrt(2.0 / fan_in)  # He initialization for ReLU networks
        
        with torch.no_grad():
            # Systolic head: Initialize for physiological range (90-180 mmHg)
            self.systolic_head[-1].weight.data.normal_(0, std)
            # Set bias to middle of physiological range, not fixed value
            self.systolic_head[-1].bias.data.uniform_(110, 130)  # Random around 120
            
            # Diastolic head: Initialize for physiological range (60-100 mmHg) 
            self.diastolic_head[-1].weight.data.normal_(0, std)
            # Set bias to middle of physiological range, not fixed value
            self.diastolic_head[-1].bias.data.uniform_(70, 90)   # Random around 80
            
        # Also initialize earlier layers with proper scaling
        for layer in self.systolic_head[:-1]:
            if hasattr(layer, 'weight') and layer.weight is not None:
                torch.nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
            if hasattr(layer, 'bias') and layer.bias is not None:
                layer.bias.data.zero_()
                
        for layer in self.diastolic_head[:-1]:
            if hasattr(layer, 'weight') and layer.weight is not None:
                torch.nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
            if hasattr(layer, 'bias') and layer.bias is not None:
                layer.bias.data.zero_()
    
    def _initialize_other_components(self):
        """Initialize other model components for better learning dynamics"""
        # Initialize waveform head with appropriate scaling for physiological range
        with torch.no_grad():
            # Waveform should predict BP range (40-200 mmHg), so scale appropriately
            if self.waveform_head.weight.dim() >= 2:
                torch.nn.init.kaiming_normal_(self.waveform_head.weight, mode='fan_out', nonlinearity='relu')
            # Initialize bias to predict reasonable baseline (around 70 mmHg for diastolic baseline)
            if self.waveform_head.bias is not None:
                self.waveform_head.bias.data.uniform_(60, 80)
        
        # Initialize BP feature extractor with proper scaling
        for layer in self.bp_feature_extractor:
            if hasattr(layer, 'weight') and layer.weight is not None and layer.weight.dim() >= 2:
                torch.nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
            if hasattr(layer, 'bias') and layer.bias is not None:
                layer.bias.data.zero_()
        
        # Initialize output layers with careful scaling
        for layer in self.output_layers:
            if hasattr(layer, 'weight') and layer.weight is not None and layer.weight.dim() >= 2:
                torch.nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
            if hasattr(layer, 'bias') and layer.bias is not None:
                layer.bias.data.zero_()
    
    def forward(self, x_seq, return_attention=False):
        batch_size, seq_len = x_seq.shape[:2]
        
        # Encode frames with frozen VAE
        latent_seq = []
        with torch.no_grad():
            for t in range(seq_len):
                mu_t, _ = self.vae.encode(x_seq[:, t])
                latent_seq.append(mu_t)
        
        latent_seq = torch.stack(latent_seq, dim=1)  # [batch_size, seq_len, latent_dim]
        
        # Extract physiological features if enabled
        if self.use_physiological_features:
            physio_features = []
            for t in range(seq_len):
                physio_feat = self.physio_extractor(latent_seq[:, t])
                physio_features.append(physio_feat)
            physio_seq = torch.stack(physio_features, dim=1)
            
            # Concatenate latent and physiological features
            enhanced_seq = torch.cat([latent_seq, physio_seq], dim=-1)
        else:
            enhanced_seq = latent_seq
        
        # Project and process
        projected_seq = self.input_projection(enhanced_seq)
        lstm_out, _ = self.lstm(projected_seq)
        
        # Apply attention
        attention_weights = None
        if self.attention is not None:
            lstm_out, attention_weights = self.attention(lstm_out, self.pattern_offsets)
        
        # Temporal convolution and aggregation
        conv_input = lstm_out.transpose(1, 2)
        conv_out = self.temporal_conv(conv_input)
        features = torch.mean(conv_out, dim=2)
        
        # Generate predictions
        final_features = self.output_layers(features)
        
        # Extract BP-specific features
        bp_features = self.bp_feature_extractor(final_features)
        
        outputs = {
            'waveform': self.waveform_head(final_features),
            'systolic': self.systolic_head(bp_features),
            'diastolic': self.diastolic_head(bp_features)
        }
        
        if return_attention and attention_weights is not None:
            outputs['attention_weights'] = attention_weights
        
        return outputs


class ImprovedMultiHeadAttention(nn.Module):
    """Improved attention with better regularization"""
    
    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.2, 
                 current_frame_bias: float = 1.5):
        super(ImprovedMultiHeadAttention, self).__init__()
        assert hidden_dim % num_heads == 0
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.current_frame_bias = current_frame_bias
        
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, x, pattern_offsets=None):
        batch_size, seq_len, hidden_dim = x.shape
        
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        
        # Add positional bias for current frame
        if pattern_offsets is not None and self.current_frame_bias > 0:
            try:
                current_frame_idx = pattern_offsets.index(0)
                bias_matrix = torch.zeros_like(scores[0, 0])
                bias_matrix[:, current_frame_idx] += self.current_frame_bias
                bias_matrix[current_frame_idx, :] += self.current_frame_bias * 0.5
                bias_matrix = bias_matrix.unsqueeze(0).unsqueeze(0)
                scores = scores + bias_matrix.to(scores.device)
            except ValueError:
                pass
        
        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        attended = torch.matmul(attention_weights, V)
        attended = attended.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_dim)
        output = self.output_proj(attended)
        
        return self.layer_norm(x + output), attention_weights.mean(dim=1)


class ImprovedBPLoss(nn.Module):
    """Improved loss function with Huber loss, better weighting, and physiological constraints"""
    
    def __init__(self, waveform_weight=0.4, systolic_weight=0.3, diastolic_weight=0.3, 
                 huber_delta=1.0, loss_type='composite', 
                 physiological_constraint=True, pulse_pressure_weight=0.1):
        super(ImprovedBPLoss, self).__init__()
        self.waveform_weight = waveform_weight
        self.systolic_weight = systolic_weight
        self.diastolic_weight = diastolic_weight
        self.huber_delta = huber_delta
        self.loss_type = loss_type
        self.physiological_constraint = physiological_constraint
        self.pulse_pressure_weight = pulse_pressure_weight
        
        # Normalize weights
        total = waveform_weight + systolic_weight + diastolic_weight
        self.waveform_weight /= total
        self.systolic_weight /= total
        self.diastolic_weight /= total
    
    def forward(self, predictions, targets):
        pred_waveform = predictions['waveform']
        
        # Extract ground truth values using improved method
        target_systolic, target_diastolic = self.extract_bp_values_improved(targets)
        
        # Waveform loss with Huber loss for robustness
        waveform_loss = nn.functional.huber_loss(pred_waveform, targets, delta=self.huber_delta)
        
        # Component losses
        if 'systolic' in predictions:
            pred_systolic = predictions['systolic'].squeeze()
            # Ensure physiological constraints (SBP > DBP)
            if self.physiological_constraint and 'diastolic' in predictions:
                pred_diastolic = predictions['diastolic'].squeeze()
                # Add penalty if SBP <= DBP
                invalid_bp = (pred_systolic <= pred_diastolic).float()
                physiological_penalty = torch.mean(invalid_bp * torch.abs(pred_systolic - pred_diastolic))
            else:
                physiological_penalty = 0.0
                
            systolic_loss = nn.functional.huber_loss(pred_systolic, target_systolic, delta=self.huber_delta)
            systolic_loss += physiological_penalty
        else:
            pred_sys, _ = self.extract_bp_values_improved(pred_waveform)
            systolic_loss = nn.functional.huber_loss(pred_sys, target_systolic, delta=self.huber_delta)
        
        if 'diastolic' in predictions:
            pred_diastolic = predictions['diastolic'].squeeze()
            diastolic_loss = nn.functional.huber_loss(pred_diastolic, target_diastolic, delta=self.huber_delta)
        else:
            _, pred_dias = self.extract_bp_values_improved(pred_waveform)
            diastolic_loss = nn.functional.huber_loss(pred_dias, target_diastolic, delta=self.huber_delta)
        
        # Add pulse pressure constraint (optional)
        pulse_pressure_loss = 0.0
        if self.physiological_constraint and 'systolic' in predictions and 'diastolic' in predictions:
            pred_pp = predictions['systolic'].squeeze() - predictions['diastolic'].squeeze()
            target_pp = target_systolic - target_diastolic
            pulse_pressure_loss = nn.functional.huber_loss(pred_pp, target_pp, delta=self.huber_delta)
        
        # Compute total loss
        if self.loss_type == 'composite':
            total_loss = (self.waveform_weight * waveform_loss + 
                         self.systolic_weight * systolic_loss + 
                         self.diastolic_weight * diastolic_loss +
                         self.pulse_pressure_weight * pulse_pressure_loss)
        else:
            total_loss = waveform_loss
        
        return {
            'total_loss': total_loss,
            'waveform_loss': waveform_loss,
            'systolic_loss': systolic_loss,
            'diastolic_loss': diastolic_loss,
            'pulse_pressure_loss': pulse_pressure_loss
        }
    
    def extract_bp_values_improved(self, waveform):
        """Extract systolic and diastolic values with smooth, continuous extraction to prevent block patterns"""
        batch_size, signal_length = waveform.shape
        device = waveform.device
        
        systolic_values = torch.zeros(batch_size, device=device)
        diastolic_values = torch.zeros(batch_size, device=device)
        
        for i in range(batch_size):
            signal = waveform[i]
            
            # Apply stronger smoothing for continuous extraction
            if signal_length > 7:
                # Use larger kernel for smoother extraction
                kernel = torch.tensor([0.1, 0.2, 0.4, 0.2, 0.1], device=device)
                signal_smooth = torch.nn.functional.conv1d(
                    signal.unsqueeze(0).unsqueeze(0), 
                    kernel.unsqueeze(0).unsqueeze(0), 
                    padding=2
                ).squeeze()
            else:
                signal_smooth = signal
            
            # Systolic: Use parabolic interpolation around maximum for continuity
            sys_val, sys_idx = torch.max(signal_smooth, dim=0)
            sys_idx = sys_idx.item()
            
            # Parabolic interpolation for sub-sample precision
            if 1 <= sys_idx <= signal_length - 2:
                y1, y2, y3 = signal_smooth[sys_idx-1], signal_smooth[sys_idx], signal_smooth[sys_idx+1]
                # Parabolic interpolation formula
                a = (y1 + y3 - 2*y2) / 2
                if abs(a) > 1e-6:  # Avoid division by zero
                    offset = (y1 - y3) / (4 * a)
                    sys_val = y2 + a * offset * offset
                    
            # Diastolic: Use smooth minimum finding with parabolic interpolation
            search_start = max(sys_idx + 1, int(signal_length * 0.6))
            search_end = min(signal_length, int(signal_length * 0.95))
            
            if search_start < search_end:
                diastolic_window = signal_smooth[search_start:search_end]
                dias_val_rel, local_min_idx = torch.min(diastolic_window, dim=0)
                global_min_idx = search_start + local_min_idx.item()
                
                # Parabolic interpolation for diastolic minimum
                if search_start + 1 <= global_min_idx <= search_end - 2:
                    y1 = signal_smooth[global_min_idx-1]
                    y2 = signal_smooth[global_min_idx] 
                    y3 = signal_smooth[global_min_idx+1]
                    a = (y1 + y3 - 2*y2) / 2
                    if abs(a) > 1e-6:
                        offset = (y1 - y3) / (4 * a)
                        dias_val = y2 + a * offset * offset
                    else:
                        dias_val = dias_val_rel
                else:
                    dias_val = dias_val_rel
            else:
                # Fallback: use global minimum with smoothing
                dias_val = torch.min(signal_smooth)
                
            # Ensure physiological constraint with smooth enforcement
            pulse_pressure = sys_val - dias_val
            min_pulse_pressure = 15.0  # Minimum realistic pulse pressure
            
            if pulse_pressure < min_pulse_pressure:
                # Smoothly adjust to maintain minimum pulse pressure
                center_pressure = (sys_val + dias_val) / 2
                sys_val = center_pressure + min_pulse_pressure / 2
                dias_val = center_pressure - min_pulse_pressure / 2
                
            systolic_values[i] = sys_val
            diastolic_values[i] = dias_val
        
        return systolic_values, diastolic_values
    
    def extract_bp_values(self, waveform):
        """Legacy method for backward compatibility"""
        return self.extract_bp_values_improved(waveform)


# Include VAE definition (same as before)
class VAE(nn.Module):
    """VAE model matching the trained checkpoint"""
    
    def __init__(self, latent_dim=64):
        super(VAE, self).__init__()
        
        # Encoder
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        
        self.fc_mu = nn.Linear(512 * 2 * 2, latent_dim)
        self.fc_logvar = nn.Linear(512 * 2 * 2, latent_dim)
        
        # Decoder
        self.fc_decoder = nn.Linear(latent_dim, 512 * 2 * 2)
        self.bn_dec = nn.BatchNorm1d(512 * 2 * 2)
        
        self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn_dec1 = nn.BatchNorm2d(256)
        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn_dec2 = nn.BatchNorm2d(128)
        self.deconv3 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn_dec3 = nn.BatchNorm2d(64)
        self.deconv4 = nn.ConvTranspose2d(64, 1, kernel_size=3, stride=2, padding=1, output_padding=1)
        
        self.dropout = nn.Dropout(0.2)
        
    def encode(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        
        x = x.view(x.size(0), -1)
        
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def decode(self, z):
        x = self.fc_decoder(z)
        x = F.relu(self.bn_dec(x))
        x = x.view(x.size(0), 512, 2, 2)
        
        x = F.relu(self.bn_dec1(self.deconv1(x)))
        x = F.relu(self.bn_dec2(self.deconv2(x)))
        x = self.dropout(x)
        x = F.relu(self.bn_dec3(self.deconv3(x)))
        x = torch.sigmoid(self.deconv4(x))
        
        return x
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar


def load_config_file(config_path):
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
    parser = argparse.ArgumentParser(description='Improved BP Predictor Training')
    
    parser.add_argument('--config', type=str, required=True,
                       help='Path to YAML configuration file')
    
    args = parser.parse_args()
    
    # Load configuration
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")
    
    print(f"Loading configuration from: {args.config}")
    config = load_config_file(args.config)
    
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
    output_dir = output_config.get('output_dir', './experiments/improved_bp_predictor')
    os.makedirs(output_dir, exist_ok=True)
    checkpoints_dir = os.path.join(output_dir, 'checkpoints')
    results_dir = os.path.join(output_dir, 'results')
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    print("="*60)
    print("IMPROVED BP PREDICTOR TRAINING")
    print("="*60)
    print("Key improvements:")
    print("- Reduced data overlap (step size 10 vs 1)")
    print("- VAE-compatible normalization")
    print("- Multi-frame temporal modeling (7 frames)")
    print("- Better regularization and dropout")
    print("- Huber loss for robustness")
    print("- AdamW optimizer with ReduceLROnPlateau")
    print("="*60)

    data_root = data_config.get('root_path', '/home/lucas_takanori/phd/data')
    path_manager = DataPathManager(
        subject="subject001",
        session="baseline",
        root=data_root
    )
    h5_file_path = str(path_manager._h5_path)
    
    print(f"Data file: {h5_file_path}")
    print(f"Pattern offsets: {data_config.get('pattern_offsets', [-7,-6,-5,-4, -3, -2, -1, 0, 1, 2])}")
    print(f"Output directory: {output_dir}")
    print("="*60)
    
    # Load pre-trained VAE
    print("\nLoading pre-trained VAE...")
    vae_config = model_config.get('vae_config', {})
    vae = VAE(latent_dim=vae_config.get('latent_dim', 64))
    
    vae_checkpoint_path = vae_config.get('vae_checkpoint_path')
    if vae_checkpoint_path and os.path.exists(vae_checkpoint_path):
        checkpoint = torch.load(vae_checkpoint_path, map_location=device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            vae.load_state_dict(checkpoint['model_state_dict'])
        else:
            vae.load_state_dict(checkpoint)
        print(f"Loaded VAE from {vae_checkpoint_path}")
    else:
        print(f"Warning: VAE checkpoint not found at {vae_checkpoint_path}")
    
    vae.to(device)
    vae.eval()
    
    # Create improved dataset
    print("\nCreating improved dataset...")
    dataset = ImprovedBPDataset(
        data_root=h5_file_path,
        pattern_offsets=data_config.get('pattern_offsets', [-7,-6,-5,-4, -3, -2, -1, 0, 1, 2]),
        max_samples_per_subject=data_config.get('max_samples_per_subject', 100),
        sequence_step_size=data_config.get('sequence_step_size', 10),
        use_augmentation=data_config.get('use_augmentation', False),
        noise_level=data_config.get('noise_level', 0.005)
    )
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    batch_size = training_config.get('batch_size', 16)
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
    
    # Create improved model
    print("\nCreating improved BP predictor...")
    bilstm_config = model_config.get('bilstm_config', {})
    attention_config = model_config.get('attention_config', {})
    
    model = ImprovedBPPredictor(
        vae_model=vae,
        latent_dim=vae_config.get('latent_dim', 128),
        hidden_dim=bilstm_config.get('hidden_dim', 128),
        num_layers=bilstm_config.get('num_layers', 2),
        num_heads=attention_config.get('num_attention_heads', 4),
        dropout=bilstm_config.get('dropout_rate', 0.4),
        use_attention=attention_config.get('use_attention', True),
        pattern_offsets=data_config.get('pattern_offsets', [-7,-6,-5,-4, -3, -2, -1, 0, 1, 2]),
        current_frame_bias=attention_config.get('current_frame_bias', 1.5),
        use_physiological_features=attention_config.get('use_physiological_features', True)
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Initialize improved loss and optimizer
    loss_weights = loss_config.get('loss_weights', {})
    criterion = ImprovedBPLoss(
        waveform_weight=loss_weights.get('waveform_weight', 0.4),
        systolic_weight=loss_weights.get('systolic_weight', 0.3),
        diastolic_weight=loss_weights.get('diastolic_weight', 0.3),
        huber_delta=loss_config.get('loss_params', {}).get('huber_delta', 1.0),
        loss_type=loss_config.get('loss_type', 'composite'),
        physiological_constraint=loss_config.get('physiological_constraint', True),
        pulse_pressure_weight=loss_config.get('pulse_pressure_weight', 0.1)
    )
    
    # Use AdamW optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config.get('learning_rate', 0.0005)),
        weight_decay=float(training_config.get('weight_decay', 1e-4)),
        betas=training_config.get('optimizer_config', {}).get('betas', [0.9, 0.999]),
        amsgrad=training_config.get('optimizer_config', {}).get('amsgrad', True)
    )
    
    # Use ReduceLROnPlateau scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        patience=int(training_config.get('scheduler_config', {}).get('scheduler_patience', 5)),
        factor=float(training_config.get('scheduler_config', {}).get('factor', 0.7)),
        min_lr=float(training_config.get('scheduler_config', {}).get('min_lr', 1e-6)),
        verbose=True
    )
    
    # Training loop
    num_epochs = training_config.get('num_epochs', 50)
    print(f"\nStarting training for {num_epochs} epochs...")
    
    best_val_mae = float('inf')
    training_history = {
        'train_loss': [],
        'val_loss': [],
        'val_mae': [],
        'learning_rates': [],
        'train_waveform_loss': [],
        'train_systolic_loss': [],
        'train_diastolic_loss': [],
        'train_pulse_pressure_loss': [],
        'val_waveform_loss': [],
        'val_systolic_loss': [],
        'val_diastolic_loss': [],
        'val_pulse_pressure_loss': [],
        'val_systolic_mae': [],
        'val_diastolic_mae': []
    }
    
    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        
        # Training
        model.train()
        train_loss = 0.0
        train_waveform_loss = 0.0
        train_systolic_loss = 0.0
        train_diastolic_loss = 0.0
        train_pulse_pressure_loss = 0.0
        num_train_batches = 0
        
        train_pbar = tqdm(train_loader, desc="Training")
        for batch in train_pbar:
            try:
                sequences = batch['sequences'].to(device)
                targets = batch['targets'].to(device)
                
                optimizer.zero_grad()
                
                outputs = model(sequences)
                loss_dict = criterion(outputs, targets)
                total_loss = loss_dict['total_loss']
                
                total_loss.backward()
                
                # Gradient clipping
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 
                    max_norm=training_config.get('optimization', {}).get('grad_clip_norm', 0.5)
                )
                
                optimizer.step()
                
                # Track individual loss components
                train_loss += total_loss.item()
                train_waveform_loss += loss_dict['waveform_loss'].item()
                train_systolic_loss += loss_dict['systolic_loss'].item()
                train_diastolic_loss += loss_dict['diastolic_loss'].item()
                train_pulse_pressure_loss += loss_dict['pulse_pressure_loss'].item()
                num_train_batches += 1
                
                train_pbar.set_postfix({
                    'Loss': f"{total_loss.item():.4f}",
                    'SBP': f"{loss_dict['systolic_loss'].item():.3f}",
                    'DBP': f"{loss_dict['diastolic_loss'].item():.3f}",
                    'LR': f"{optimizer.param_groups[0]['lr']:.2e}",
                    'GradNorm': f"{grad_norm:.2f}"
                })
                
            except Exception as e:
                print(f"Training error: {e}")
                continue
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_waveform_loss = 0.0
        val_systolic_loss = 0.0
        val_diastolic_loss = 0.0
        val_pulse_pressure_loss = 0.0
        val_predictions = []
        val_targets = []
        val_systolic_predictions = []
        val_diastolic_predictions = []
        val_systolic_targets = []
        val_diastolic_targets = []
        num_val_batches = 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc="Validation")
            for batch in val_pbar:
                try:
                    sequences = batch['sequences'].to(device)
                    targets = batch['targets'].to(device)
                    
                    outputs = model(sequences)
                    loss_dict = criterion(outputs, targets)
                    
                    # Track individual loss components
                    val_loss += loss_dict['total_loss'].item()
                    val_waveform_loss += loss_dict['waveform_loss'].item()
                    val_systolic_loss += loss_dict['systolic_loss'].item()
                    val_diastolic_loss += loss_dict['diastolic_loss'].item()
                    val_pulse_pressure_loss += loss_dict['pulse_pressure_loss'].item()
                    num_val_batches += 1
                    
                    # Store predictions for metrics
                    val_predictions.append(outputs['waveform'].cpu().numpy())
                    val_targets.append(targets.cpu().numpy())
                    
                    # Store SBP/DBP predictions and targets for detailed analysis
                    if 'systolic' in outputs and 'diastolic' in outputs:
                        val_systolic_predictions.append(outputs['systolic'].cpu().numpy())
                        val_diastolic_predictions.append(outputs['diastolic'].cpu().numpy())
                        
                        # Extract ground truth SBP/DBP
                        target_sys, target_dias = criterion.extract_bp_values_improved(targets)
                        val_systolic_targets.append(target_sys.cpu().numpy())
                        val_diastolic_targets.append(target_dias.cpu().numpy())
                    
                except Exception as e:
                    print(f"Validation error: {e}")
                    continue
        
        # Calculate epoch metrics
        if num_train_batches > 0:
            train_loss /= num_train_batches
            train_waveform_loss /= num_train_batches
            train_systolic_loss /= num_train_batches
            train_diastolic_loss /= num_train_batches
            train_pulse_pressure_loss /= num_train_batches
            
        if num_val_batches > 0:
            val_loss /= num_val_batches
            val_waveform_loss /= num_val_batches
            val_systolic_loss /= num_val_batches
            val_diastolic_loss /= num_val_batches
            val_pulse_pressure_loss /= num_val_batches
        
        # Calculate MAE
        val_mae = 0.0
        val_systolic_mae = 0.0
        val_diastolic_mae = 0.0
        
        if val_predictions and val_targets:
            all_val_pred = np.concatenate(val_predictions, axis=0)
            all_val_target = np.concatenate(val_targets, axis=0)
            val_mae = np.mean(np.abs(all_val_pred - all_val_target))
            
            # Calculate SBP/DBP specific MAEs
            if val_systolic_predictions and val_systolic_targets:
                all_sys_pred = np.concatenate(val_systolic_predictions, axis=0).flatten()
                all_sys_target = np.concatenate(val_systolic_targets, axis=0).flatten()
                val_systolic_mae = np.mean(np.abs(all_sys_pred - all_sys_target))
                
                all_dias_pred = np.concatenate(val_diastolic_predictions, axis=0).flatten()
                all_dias_target = np.concatenate(val_diastolic_targets, axis=0).flatten()
                val_diastolic_mae = np.mean(np.abs(all_dias_pred - all_dias_target))
        
        # Update scheduler
        scheduler.step(val_mae)
        
        # Store history
        training_history['train_loss'].append(train_loss)
        training_history['val_loss'].append(val_loss)
        training_history['val_mae'].append(val_mae)
        training_history['learning_rates'].append(optimizer.param_groups[0]['lr'])
        training_history['train_waveform_loss'].append(train_waveform_loss)
        training_history['train_systolic_loss'].append(train_systolic_loss)
        training_history['train_diastolic_loss'].append(train_diastolic_loss)
        training_history['train_pulse_pressure_loss'].append(train_pulse_pressure_loss)
        training_history['val_waveform_loss'].append(val_waveform_loss)
        training_history['val_systolic_loss'].append(val_systolic_loss)
        training_history['val_diastolic_loss'].append(val_diastolic_loss)
        training_history['val_pulse_pressure_loss'].append(val_pulse_pressure_loss)
        training_history['val_systolic_mae'].append(val_systolic_mae)
        training_history['val_diastolic_mae'].append(val_diastolic_mae)
        
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val MAE: {val_mae:.2f}")
        print(f"  └─ Waveform: T={train_waveform_loss:.4f}, V={val_waveform_loss:.4f}")
        print(f"  └─ Systolic:  T={train_systolic_loss:.4f}, V={val_systolic_loss:.4f}, MAE={val_systolic_mae:.2f}")
        print(f"  └─ Diastolic: T={train_diastolic_loss:.4f}, V={val_diastolic_loss:.4f}, MAE={val_diastolic_mae:.2f}")
        print(f"  └─ PulsePres: T={train_pulse_pressure_loss:.4f}, V={val_pulse_pressure_loss:.4f}")
        
        # Save best model based on MAE
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_mae': best_val_mae,
                'config': config,
                'training_history': training_history
            }
            
            torch.save(checkpoint, os.path.join(checkpoints_dir, 'best_model.pt'))
            print(f"New best model saved! Val MAE: {best_val_mae:.2f} (SBP: {val_systolic_mae:.2f}, DBP: {val_diastolic_mae:.2f})")
    
    # Enhanced plotting of training curves
    epochs = range(1, len(training_history['train_loss']) + 1)
    
    plt.figure(figsize=(20, 12))
    
    # Plot 1: Overall Loss
    plt.subplot(3, 4, 1)
    plt.plot(epochs, training_history['train_loss'], label='Train', color='blue')
    plt.plot(epochs, training_history['val_loss'], label='Validation', color='red')
    plt.title('Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Waveform Loss
    plt.subplot(3, 4, 2)
    plt.plot(epochs, training_history['train_waveform_loss'], label='Train', color='blue')
    plt.plot(epochs, training_history['val_waveform_loss'], label='Validation', color='red')
    plt.title('Waveform Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Systolic Loss
    plt.subplot(3, 4, 3)
    plt.plot(epochs, training_history['train_systolic_loss'], label='Train', color='blue')
    plt.plot(epochs, training_history['val_systolic_loss'], label='Validation', color='red')
    plt.title('Systolic Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Diastolic Loss
    plt.subplot(3, 4, 4)
    plt.plot(epochs, training_history['train_diastolic_loss'], label='Train', color='blue')
    plt.plot(epochs, training_history['val_diastolic_loss'], label='Validation', color='red')
    plt.title('Diastolic Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 5: Pulse Pressure Loss
    plt.subplot(3, 4, 5)
    plt.plot(epochs, training_history['train_pulse_pressure_loss'], label='Train', color='blue')
    plt.plot(epochs, training_history['val_pulse_pressure_loss'], label='Validation', color='red')
    plt.title('Pulse Pressure Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 6: Validation MAE
    plt.subplot(3, 4, 6)
    plt.plot(epochs, training_history['val_mae'], color='purple')
    plt.title('Validation MAE (Waveform)')
    plt.xlabel('Epoch')
    plt.ylabel('MAE (mmHg)')
    plt.grid(True, alpha=0.3)
    
    # Plot 7: Systolic MAE
    plt.subplot(3, 4, 7)
    plt.plot(epochs, training_history['val_systolic_mae'], color='green')
    plt.title('Validation Systolic MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE (mmHg)')
    plt.grid(True, alpha=0.3)
    
    # Plot 8: Diastolic MAE
    plt.subplot(3, 4, 8)
    plt.plot(epochs, training_history['val_diastolic_mae'], color='orange')
    plt.title('Validation Diastolic MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE (mmHg)')
    plt.grid(True, alpha=0.3)
    
    # Plot 9: Learning Rate
    plt.subplot(3, 4, 9)
    plt.plot(epochs, training_history['learning_rates'], color='brown')
    plt.title('Learning Rate')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    # Plot 10: Loss Components Comparison (Bar)
    plt.subplot(3, 4, 10)
    final_losses = [
        training_history['val_waveform_loss'][-1],
        training_history['val_systolic_loss'][-1],
        training_history['val_diastolic_loss'][-1],
        training_history['val_pulse_pressure_loss'][-1]
    ]
    labels = ['Waveform', 'Systolic', 'Diastolic', 'Pulse Press']
    colors = ['blue', 'green', 'orange', 'red']
    plt.bar(labels, final_losses, color=colors, alpha=0.7)
    plt.title('Final Validation Loss Components')
    plt.ylabel('Loss')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Plot 11: MAE Components Comparison (Bar)
    plt.subplot(3, 4, 11)
    final_maes = [
        training_history['val_mae'][-1],
        training_history['val_systolic_mae'][-1],
        training_history['val_diastolic_mae'][-1]
    ]
    mae_labels = ['Waveform', 'Systolic', 'Diastolic']
    mae_colors = ['purple', 'green', 'orange']
    plt.bar(mae_labels, final_maes, color=mae_colors, alpha=0.7)
    plt.title('Final Validation MAE Components')
    plt.ylabel('MAE (mmHg)')
    plt.grid(True, alpha=0.3)
    
    # Plot 12: Combined SBP/DBP MAE Evolution
    plt.subplot(3, 4, 12)
    plt.plot(epochs, training_history['val_systolic_mae'], label='Systolic MAE', color='green')
    plt.plot(epochs, training_history['val_diastolic_mae'], label='Diastolic MAE', color='orange')
    plt.title('SBP/DBP MAE Evolution')
    plt.xlabel('Epoch')
    plt.ylabel('MAE (mmHg)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'enhanced_training_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Generate prediction examples and attention visualizations
    print("\n" + "="*60)
    print("GENERATING PREDICTION EXAMPLES AND ATTENTION MAPS")
    print("="*60)
    
    # Load best model for visualization
    best_checkpoint = torch.load(os.path.join(checkpoints_dir, 'best_model.pt'), 
                                map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint['model_state_dict'])
    model.eval()
    
    # Generate prediction examples
    visualizations_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(visualizations_dir, exist_ok=True)
    
    generate_prediction_examples(
        model=model,
        dataset=dataset,
        device=device,
        num_examples=8,
        pattern_offsets=data_config.get('pattern_offsets', [-4, -3, -2, -1, 0, 1, 2]),
        save_dir=visualizations_dir
    )
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED")
    print("="*60)
    print(f"Best validation MAE: {best_val_mae:.2f} mmHg")
    print(f"Results saved to: {output_dir}")
    print(f"Visualizations saved to: {visualizations_dir}")
    
    # COMPREHENSIVE ANALYSIS OF SBP/DBP PREDICTION ISSUES
    print("\n" + "="*80)
    print("ANALYSIS: SBP/DBP PREDICTION CHALLENGES & SOLUTIONS")
    print("="*80)
    
    # Load training history for analysis
    final_epoch = len(training_history['val_systolic_mae']) - 1
    final_sys_mae = training_history['val_systolic_mae'][final_epoch] if training_history['val_systolic_mae'] else 0
    final_dias_mae = training_history['val_diastolic_mae'][final_epoch] if training_history['val_diastolic_mae'] else 0
    final_sys_loss = training_history['val_systolic_loss'][final_epoch] if training_history['val_systolic_loss'] else 0
    final_dias_loss = training_history['val_diastolic_loss'][final_epoch] if training_history['val_diastolic_loss'] else 0
    
    print(f"\nFINAL RESULTS:")
    print(f"- Systolic MAE: {final_sys_mae:.2f} mmHg (Loss: {final_sys_loss:.4f})")
    print(f"- Diastolic MAE: {final_dias_mae:.2f} mmHg (Loss: {final_dias_loss:.4f})")
    print(f"- Overall Waveform MAE: {best_val_mae:.2f} mmHg")
    
    print(f"\nISSUE ANALYSIS:")
    if final_sys_mae > 15 or final_dias_mae > 10:
        print("❌ HIGH SBP/DBP ERRORS DETECTED!")
        print("\nPOTENTIAL CAUSES:")
        print("1. VAE LATENT SPACE LIMITATIONS:")
        print("   - 64-dim latent space may not capture fine BP variations")
        print("   - VAE reconstruction loss doesn't prioritize physiological features")
        print("   - Information bottleneck at VAE encoding stage")
        
        print("\n2. TEMPORAL MODELING ISSUES:")
        print("   - BiLSTM may not capture cardiac cycle dynamics effectively")
        print("   - Multi-frame attention might dilute current frame information")
        print("   - Pattern offsets may not align with physiological events")
        
        print("\n3. LOSS FUNCTION WEIGHTING:")
        print("   - Waveform loss may dominate over SBP/DBP losses")
        print("   - Huber loss might be too robust for fine-grained BP prediction")
        print("   - Separate heads competing during training")
        
        print("\n4. DATASET & PREPROCESSING:")
        print("   - Limited training samples per subject")
        print("   - Sequence step size reducing temporal correlation")
        print("   - Normalization mismatch between VAE and BP prediction")
        
        print("\nRECOMMENDED SOLUTIONS:")
        print("\n🔧 IMMEDIATE FIXES:")
        print("1. Adjust loss weights: increase systolic_weight and diastolic_weight")
        print("2. Reduce sequence_step_size to capture more temporal dynamics")
        print("3. Use curriculum learning: start with waveform, add SBP/DBP gradually")
        print("4. Add physiological constraints in loss function")
        
        print("\n🚀 ARCHITECTURAL IMPROVEMENTS:")
        print("1. ENHANCED VAE:")
        print("   - β-VAE with controlled disentanglement")
        print("   - Conditional VAE with BP information")
        print("   - Larger latent dimension (128 or 256)")
        print("   - Multi-scale VAE with different resolutions")
        
        print("\n2. ALTERNATIVE ARCHITECTURES:")
        print("   - Transformer-based temporal modeling")
        print("   - CNN-LSTM hybrid with temporal convolutions")
        print("   - U-Net style encoder-decoder for waveform reconstruction")
        print("   - Graph Neural Networks for physiological relationships")
        
        print("\n3. MULTI-TASK LEARNING:")
        print("   - Joint training with multiple physiological signals")
        print("   - Auxiliary tasks: heart rate, pulse pressure prediction")
        print("   - Contrastive learning for physiological feature extraction")
        
        print("\n4. PHYSIOLOGICALLY-INFORMED FEATURES:")
        print("   - Hand-crafted features: pulse pressure, mean arterial pressure")
        print("   - Fourier features for cardiac cycle analysis")
        print("   - Wavelet decomposition for multi-scale analysis")
        print("   - Peak detection and morphology analysis")
        
        print("\n📊 DATA & TRAINING IMPROVEMENTS:")
        print("1. DATA AUGMENTATION:")
        print("   - Synthetic BP waveform generation")
        print("   - Temporal jittering and scaling")
        print("   - Cross-subject data mixing")
        
        print("\n2. TRAINING STRATEGIES:")
        print("   - Progressive growing: start simple, add complexity")
        print("   - Adversarial training for realistic waveforms")
        print("   - Self-supervised pre-training")
        print("   - Transfer learning from larger datasets")
        
        print("\n3. EVALUATION & VALIDATION:")
        print("   - Cross-subject validation")
        print("   - Clinical validation with gold standard")
        print("   - Temporal consistency analysis")
        print("   - Physiological plausibility checks")
    else:
        print("✅ SBP/DBP errors within acceptable range!")
        print("Current approach is working reasonably well.")
        
    print("\n📋 NEXT STEPS PRIORITY:")
    print("1. High Priority: Adjust loss weights and try β-VAE")
    print("2. Medium Priority: Implement Transformer architecture")
    print("3. Low Priority: Explore multi-task learning approaches")
    print("4. Research: Investigate physiologically-informed architectures")
    
    print("\n💡 EXPERIMENTAL CONFIGURATIONS TO TRY:")
    print("Loss weights: systolic_weight=0.4, diastolic_weight=0.4, waveform_weight=0.2")
    print("VAE: latent_dim=128, β=4.0 for disentanglement")
    print("Data: sequence_step_size=5, max_samples_per_subject=200")
    print("Architecture: hidden_dim=256, num_layers=3, num_heads=8")
    
    print("\n" + "="*80)


def extract_bp_values_numpy(waveform):
    """Extract systolic and diastolic values with smooth, continuous extraction (numpy version)"""
    if isinstance(waveform, torch.Tensor):
        waveform = waveform.detach().cpu().numpy()
    
    if len(waveform.shape) > 1:
        waveform = waveform.flatten()
    
    # Apply smoothing for continuous extraction
    if len(waveform) > 7:
        from scipy import ndimage
        # Gaussian smoothing for continuous values
        smoothed = ndimage.gaussian_filter1d(waveform, sigma=1.0)
    else:
        smoothed = waveform
    
    # Systolic with parabolic interpolation
    sys_idx = np.argmax(smoothed)
    systolic = smoothed[sys_idx]
    
    # Parabolic interpolation for sub-sample precision
    if 1 <= sys_idx <= len(smoothed) - 2:
        y1, y2, y3 = smoothed[sys_idx-1], smoothed[sys_idx], smoothed[sys_idx+1]
        a = (y1 + y3 - 2*y2) / 2
        if abs(a) > 1e-6:
            offset = (y1 - y3) / (4 * a)
            systolic = y2 + a * offset * offset
    
    # Diastolic with smooth minimum finding
    search_start = max(sys_idx + 1, int(len(smoothed) * 0.6))
    search_end = min(len(smoothed), int(len(smoothed) * 0.95))
    
    if search_start < search_end:
        post_systolic = smoothed[search_start:search_end]
        local_min_idx = np.argmin(post_systolic)
        global_min_idx = search_start + local_min_idx
        diastolic = smoothed[global_min_idx]
        
        # Parabolic interpolation for diastolic
        if search_start + 1 <= global_min_idx <= search_end - 2:
            y1 = smoothed[global_min_idx-1]
            y2 = smoothed[global_min_idx]
            y3 = smoothed[global_min_idx+1]
            a = (y1 + y3 - 2*y2) / 2
            if abs(a) > 1e-6:
                offset = (y1 - y3) / (4 * a)
                diastolic = y2 + a * offset * offset
    else:
        diastolic = np.min(smoothed)
    
    # Ensure physiological constraint with smooth enforcement
    pulse_pressure = systolic - diastolic
    min_pulse_pressure = 15.0
    
    if pulse_pressure < min_pulse_pressure:
        center_pressure = (systolic + diastolic) / 2
        systolic = center_pressure + min_pulse_pressure / 2
        diastolic = center_pressure - min_pulse_pressure / 2
    
    return systolic, diastolic


def generate_prediction_examples(model, dataset, device, num_examples=8, 
                               pattern_offsets=None, save_dir=None):
    """Generate prediction examples and attention visualizations"""
    import seaborn as sns
    
    if pattern_offsets is None:
        pattern_offsets = [-7,-6,-5,-4, -3, -2, -1, 0, 1, 2]
    
    model.eval()
    examples = []
    
    # Select random examples
    indices = np.random.choice(len(dataset), num_examples, replace=False)
    
    print(f"Generating {num_examples} prediction examples...")
    
    with torch.no_grad():
        for i, idx in enumerate(indices):
            sample = dataset[idx]
            sequences = sample['sequences'].unsqueeze(0).to(device)  # Add batch dimension
            targets = sample['targets'].unsqueeze(0).to(device)
            
            # Get prediction with attention
            outputs = model(sequences, return_attention=True)
            pred_waveform = outputs['waveform'].cpu().numpy()[0]
            target_waveform = targets.cpu().numpy()[0]
            
            # Extract BP values
            pred_sys, pred_dias = extract_bp_values_numpy(pred_waveform)
            target_sys, target_dias = extract_bp_values_numpy(target_waveform)
            
            # Get attention weights if available
            attention_weights = None
            if 'attention_weights' in outputs:
                attention_weights = outputs['attention_weights'].cpu().numpy()[0]
            
            examples.append({
                'idx': idx,
                'pred_waveform': pred_waveform,
                'target_waveform': target_waveform,
                'pred_sys': pred_sys,
                'pred_dias': pred_dias,
                'target_sys': target_sys,
                'target_dias': target_dias,
                'attention_weights': attention_weights,
                'sequences': sequences.cpu().numpy()[0]
            })
    
    # Create prediction examples visualization
    fig, axes = plt.subplots(num_examples, 2, figsize=(15, 4*num_examples))
    if num_examples == 1:
        axes = axes.reshape(1, -1)
    
    for i, example in enumerate(examples):
        # Plot waveform comparison
        ax1 = axes[i, 0]
        time_points = np.linspace(0, 1, len(example['target_waveform']))
        
        ax1.plot(time_points, example['target_waveform'], 'b-', label='Ground Truth', linewidth=2)
        ax1.plot(time_points, example['pred_waveform'], 'r--', label='Prediction', linewidth=2)
        
        # Add BP value annotations
        ax1.axhline(y=example['target_sys'], color='blue', linestyle=':', alpha=0.7, 
                   label=f'True SBP: {example["target_sys"]:.1f}')
        ax1.axhline(y=example['target_dias'], color='blue', linestyle=':', alpha=0.7,
                   label=f'True DBP: {example["target_dias"]:.1f}')
        ax1.axhline(y=example['pred_sys'], color='red', linestyle=':', alpha=0.7,
                   label=f'Pred SBP: {example["pred_sys"]:.1f}')
        ax1.axhline(y=example['pred_dias'], color='red', linestyle=':', alpha=0.7,
                   label=f'Pred DBP: {example["pred_dias"]:.1f}')
        
        ax1.set_xlabel('Normalized Time')
        ax1.set_ylabel('Blood Pressure (mmHg)')
        ax1.set_title(f'Example {i+1}: BP Waveform Prediction\n'
                     f'MAE: {np.mean(np.abs(example["pred_waveform"] - example["target_waveform"])):.2f} mmHg')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Plot attention heatmap
        ax2 = axes[i, 1]
        if example['attention_weights'] is not None:
            # Create heatmap
            sns.heatmap(example['attention_weights'], 
                       xticklabels=[f't{offset:+d}' for offset in pattern_offsets],
                       yticklabels=[f't{offset:+d}' for offset in pattern_offsets],
                       annot=True, fmt='.3f', cmap='Blues', ax=ax2)
            ax2.set_title(f'Attention Weights\n(Focus on current frame t+0)')
            ax2.set_xlabel('Key Frames')
            ax2.set_ylabel('Query Frames')
        else:
            ax2.text(0.5, 0.5, 'No Attention\nWeights Available', 
                    ha='center', va='center', transform=ax2.transAxes, fontsize=12)
            ax2.set_title('Attention Weights')
    
    plt.tight_layout()
    
    if save_dir:
        examples_path = os.path.join(save_dir, 'prediction_examples.png')
        plt.savefig(examples_path, dpi=300, bbox_inches='tight')
        print(f"Prediction examples saved to: {examples_path}")
    
    plt.close()
    
    # Create attention analysis if we have attention weights
    attention_examples = [ex for ex in examples if ex['attention_weights'] is not None]
    
    if attention_examples:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Average attention pattern
        ax1 = axes[0, 0]
        avg_attention = np.mean([ex['attention_weights'] for ex in attention_examples], axis=0)
        
        sns.heatmap(avg_attention, 
                   xticklabels=[f't{offset:+d}' for offset in pattern_offsets],
                   yticklabels=[f't{offset:+d}' for offset in pattern_offsets],
                   annot=True, fmt='.3f', cmap='Blues', ax=ax1)
        ax1.set_title('Average Attention Pattern')
        ax1.set_xlabel('Key Frames')
        ax1.set_ylabel('Query Frames')
        
        # 2. Attention to current frame (t+0)
        ax2 = axes[0, 1]
        try:
            current_frame_idx = pattern_offsets.index(0)  # Index of t+0
            current_frame_attention = [ex['attention_weights'][:, current_frame_idx] 
                                     for ex in attention_examples]
            
            attention_matrix = np.array(current_frame_attention).T
            sns.heatmap(attention_matrix, 
                       yticklabels=[f't{offset:+d}' for offset in pattern_offsets],
                       cmap='Blues', ax=ax2)
            ax2.set_title('Attention TO Current Frame (t+0)')
            ax2.set_xlabel('Example')
            ax2.set_ylabel('Query Frames')
        except ValueError:
            ax2.text(0.5, 0.5, 'Current frame\nt+0 not found', 
                    ha='center', va='center', transform=ax2.transAxes, fontsize=12)
        
        # 3. Attention FROM current frame
        ax3 = axes[1, 0]
        try:
            from_current_attention = [ex['attention_weights'][current_frame_idx, :] 
                                    for ex in attention_examples]
            
            attention_matrix = np.array(from_current_attention).T
            sns.heatmap(attention_matrix, 
                       yticklabels=[f't{offset:+d}' for offset in pattern_offsets],
                       cmap='Blues', ax=ax3)
            ax3.set_title('Attention FROM Current Frame (t+0)')
            ax3.set_xlabel('Example')
            ax3.set_ylabel('Key Frames')
        except (ValueError, NameError):
            ax3.text(0.5, 0.5, 'Current frame\nt+0 not found', 
                    ha='center', va='center', transform=ax3.transAxes, fontsize=12)
        
        # 4. Temporal attention distribution
        ax4 = axes[1, 1]
        temporal_attention = np.mean([ex['attention_weights'] for ex in attention_examples], axis=(0, 1))
        
        ax4.bar(range(len(pattern_offsets)), temporal_attention, 
               color='skyblue', alpha=0.7)
        ax4.set_xticks(range(len(pattern_offsets)))
        ax4.set_xticklabels([f't{offset:+d}' for offset in pattern_offsets])
        ax4.set_title('Average Temporal Attention Distribution')
        ax4.set_xlabel('Frame Offset')
        ax4.set_ylabel('Average Attention Weight')
        ax4.grid(True, alpha=0.3)
        
        # Highlight current frame if it exists
        try:
            ax4.bar(current_frame_idx, temporal_attention[current_frame_idx], 
                   color='red', alpha=0.7, label='Current Frame (t+0)')
            ax4.legend()
        except (ValueError, NameError):
            pass
        
        plt.tight_layout()
        
        if save_dir:
            attention_path = os.path.join(save_dir, 'attention_analysis.png')
            plt.savefig(attention_path, dpi=300, bbox_inches='tight')
            print(f"Attention analysis saved to: {attention_path}")
        
        plt.close()
    
    # Save summary statistics
    stats = {
        'num_examples': len(examples),
        'avg_mae': np.mean([np.mean(np.abs(ex['pred_waveform'] - ex['target_waveform'])) 
                           for ex in examples]),
        'avg_sys_error': np.mean([abs(ex['pred_sys'] - ex['target_sys']) for ex in examples]),
        'avg_dias_error': np.mean([abs(ex['pred_dias'] - ex['target_dias']) for ex in examples]),
        'sys_correlation': np.corrcoef([ex['pred_sys'] for ex in examples],
                                     [ex['target_sys'] for ex in examples])[0, 1],
        'dias_correlation': np.corrcoef([ex['pred_dias'] for ex in examples],
                                      [ex['target_dias'] for ex in examples])[0, 1]
    }
    
    if save_dir:
        stats_path = os.path.join(save_dir, 'prediction_stats.txt')
        with open(stats_path, 'w') as f:
            f.write("PREDICTION EXAMPLES STATISTICS\n")
            f.write("="*40 + "\n\n")
            f.write(f"Number of examples: {stats['num_examples']}\n")
            f.write(f"Average MAE: {stats['avg_mae']:.2f} mmHg\n")
            f.write(f"Average Systolic Error: {stats['avg_sys_error']:.2f} mmHg\n")
            f.write(f"Average Diastolic Error: {stats['avg_dias_error']:.2f} mmHg\n")
            f.write(f"Systolic Correlation: {stats['sys_correlation']:.3f}\n")
            f.write(f"Diastolic Correlation: {stats['dias_correlation']:.3f}\n")
        
        print(f"Statistics saved to: {stats_path}")
    
    print(f"\nPrediction Examples Summary:")
    print(f"- Average MAE: {stats['avg_mae']:.2f} mmHg")
    print(f"- Systolic Error: {stats['avg_sys_error']:.2f} mmHg")
    print(f"- Diastolic Error: {stats['avg_dias_error']:.2f} mmHg")
    print(f"- Systolic Correlation: {stats['sys_correlation']:.3f}")
    print(f"- Diastolic Correlation: {stats['dias_correlation']:.3f}")
    
    return examples, stats


if __name__ == "__main__":
    main() 