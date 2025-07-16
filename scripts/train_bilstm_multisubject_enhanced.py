#!/usr/bin/env python3
"""
Enhanced Multi-Subject BiLSTM Training Script
Uses 10-frame temporal window and consistent mask handling with VAE training
Includes comprehensive evaluation at the end of training
"""

import os
import sys
import argparse
import logging
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Tuple
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
import gc
from types import SimpleNamespace
import yaml
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import pandas as pd
from scipy import stats
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import psutil
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import working modules
from src.utils.paths import setup_paths, PathConfig
from src.utils.config import load_config, create_config_from_yaml
from src.utils.logging import setup_logging, get_logger
from utils.data_utils_enhanced import EnhancedPviDataset, load_dataset_with_best_mask, analyze_all_mask_formats

logger = get_logger(__name__)


def create_subject_level_splits(subjects: List[str], train_ratio: float = 0.6, 
                               val_ratio: float = 0.2, test_ratio: float = 0.2, 
                               seed: int = 42) -> Tuple[List[str], List[str], List[str]]:
    """
    Create subject-level train/val/test splits to prevent data leakage
    
    Args:
        subjects: List of all available subjects
        train_ratio: Ratio for training split (default: 0.6 for 60%)
        val_ratio: Ratio for validation split (default: 0.2 for 20%)
        test_ratio: Ratio for test split (default: 0.2 for 20%)
        seed: Random seed for reproducibility (MUST match VAE training!)
    
    Returns:
        Tuple of (train_subjects, val_subjects, test_subjects)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Split ratios must sum to 1.0"
    
    logger.info(f"🔒 Creating SUBJECT-LEVEL splits to prevent data leakage:")
    logger.info(f"   Train: {train_ratio:.1%}, Val: {val_ratio:.1%}, Test: {test_ratio:.1%}")
    logger.info(f"   Total subjects: {len(subjects)}")
    logger.info(f"   Random seed: {seed} (MUST match VAE training!)")
    
    # Set random seed for reproducibility (CRITICAL: same as VAE)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Shuffle subjects deterministically
    shuffled_subjects = subjects.copy()
    np.random.shuffle(shuffled_subjects)
    
    # Calculate split sizes
    n_total = len(shuffled_subjects)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    # Test gets the remaining (to ensure all subjects are used)
    
    # Create splits
    train_subjects = shuffled_subjects[:n_train]
    val_subjects = shuffled_subjects[n_train:n_train + n_val]
    test_subjects = shuffled_subjects[n_train + n_val:]
    
    logger.info(f"📊 Subject distribution:")
    logger.info(f"   Train: {len(train_subjects)} subjects: {train_subjects}")
    logger.info(f"   Val:   {len(val_subjects)} subjects: {val_subjects}")
    logger.info(f"   Test:  {len(test_subjects)} subjects: {test_subjects}")
    
    # Validate no overlap (CRITICAL)
    all_sets = [set(train_subjects), set(val_subjects), set(test_subjects)]
    for i, set1 in enumerate(all_sets):
        for j, set2 in enumerate(all_sets[i+1:], i+1):
            overlap = set1 & set2
            set_names = ['train', 'val', 'test']
            assert len(overlap) == 0, f"❌ LEAKAGE DETECTED: {set_names[i]}/{set_names[j]} overlap: {overlap}"
    
    logger.info("✅ Subject-level splits created successfully - NO DATA LEAKAGE")
    logger.info("🔒 Same subjects will be used as in VAE training (same seed)")
    logger.info("🧪 TEST SET ISOLATED: Test subjects will not be seen during training")
    
    return train_subjects, val_subjects, test_subjects


class ImprovedBiLSTMDataset(Dataset):
    """Multi-subject dataset with 10-frame temporal window and consistent mask handling"""
    
    def __init__(self, data_root: str, subjects: List[str], mask_type: str = "mask10",
                 pattern_offsets: List[int] = None, max_samples_per_subject: int = 200, 
                 sequence_step_size: int = 10, session: str = "baseline",
                 use_augmentation: bool = False, noise_level: float = 0.005):
        
        self.data_root = data_root
        self.subjects = subjects
        self.mask_type = mask_type
        self.pattern_offsets = pattern_offsets or [-7, -6, -5, -4, -3, -2, -1, 0, 1, 2]
        self.max_samples_per_subject = max_samples_per_subject
        self.sequence_step_size = sequence_step_size
        self.session = session
        self.use_augmentation = use_augmentation
        self.noise_level = noise_level
        
        self.sequences = []
        self.targets = []
        self.subject_labels = []
        
        logger.info(f"Creating BiLSTM dataset with {len(subjects)} subjects")
        logger.info(f"Pattern offsets: {self.pattern_offsets}")
        logger.info(f"Mask type: {self.mask_type}")
        logger.info(f"Sequence step size: {self.sequence_step_size}")
        
        self._load_all_subjects()
        
        logger.info(f"Dataset created with {len(self.sequences)} sequences")
        if len(self.sequences) > 0:
            logger.info(f"Sequence shape: {self.sequences[0].shape}")
            logger.info(f"Target shape: {self.targets[0].shape}")
    
    def _load_all_subjects(self):
        """Load data from all subjects with consistent mask handling"""
        successful_subjects = []
        failed_subjects = []
        
        for subject in self.subjects:
            try:
                # Create file path
                file_path = Path(self.data_root) / f"{subject}_{self.session}_masked.h5"
                
                if not file_path.exists():
                    logger.warning(f"⚠️  Data file not found for {subject}, skipping")
                    failed_subjects.append(subject)
                    continue
                
                # Load dataset using enhanced loader with mask handling
                dataset = load_dataset_with_best_mask(str(file_path), self.mask_type)
                
                # Process this subject's data
                subject_sequences, subject_targets = self._process_subject_data(dataset, subject)
                
                if len(subject_sequences) > 0:
                    self.sequences.extend(subject_sequences)
                    self.targets.extend(subject_targets)
                    self.subject_labels.extend([subject] * len(subject_sequences))
                    successful_subjects.append(subject)
                    
                    logger.info(f"✅ Loaded {subject}: {len(subject_sequences)} sequences")
                    
                    # Log mask info
                    mask_info = dataset.get_mask_info()
                    logger.info(f"   Used mask: {mask_info.get('recommended_mask', 'unknown')}")
                else:
                    logger.warning(f"⚠️  No valid sequences for {subject}")
                    failed_subjects.append(subject)
                    
            except Exception as e:
                logger.warning(f"❌ Failed to load {subject}: {e}")
                failed_subjects.append(subject)
                continue
        
        logger.info(f"Successfully loaded {len(successful_subjects)} subjects")
        logger.info(f"Failed to load {len(failed_subjects)} subjects")
        
        # Memory cleanup
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    def _process_subject_data(self, dataset, subject):
        """Process a single subject's data with 10-frame temporal window"""
        subject_sequences = []
        subject_targets = []
        
        # Limit samples per subject
        num_samples = min(len(dataset), self.max_samples_per_subject)
        
        for sample_idx in range(num_samples):
            try:
                sample = dataset[sample_idx]
                
                # Extract PVI images and BP signal
                pvi_img = sample['pviHP']['img']  # [32, 32, num_frames]
                bp_signal = sample['bp']['signal']  # [num_frames]
                
                num_frames = pvi_img.shape[-1]
                
                # Determine valid central indices with pattern offsets
                min_offset = min(self.pattern_offsets)
                max_offset = max(self.pattern_offsets)
                valid_start = max(0, -min_offset)
                valid_end = min(num_frames, num_frames - max_offset)
                
                # Use step size to reduce overlap and prevent overfitting
                central_indices = list(range(valid_start, valid_end, self.sequence_step_size))
                
                for central_idx in central_indices:
                    # Create 10-frame sequence using pattern offsets
                    seq_frames = []
                    valid_sequence = True
                    
                    for offset in self.pattern_offsets:
                        frame_idx = central_idx + offset
                        if 0 <= frame_idx < num_frames:
                            # Extract frame and apply VAE-compatible normalization
                            frame = pvi_img[:, :, frame_idx]  # [32, 32]
                            frame = torch.tensor(frame, dtype=torch.float32)
                            
                            # Apply same normalization as VAE training
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
                        sequence = torch.stack(seq_frames)  # [10, 1, 32, 32]
                        
                        # Extract BP signal for current frame as target
                        if bp_signal.dim() == 1:
                            target_bp = bp_signal
                        else:
                            target_bp = bp_signal[central_idx] if central_idx < bp_signal.shape[0] else bp_signal[0]
                        
                        # Ensure target is exactly 50 samples (like improved BP predictor)
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
                        
                        subject_sequences.append(sequence)
                        subject_targets.append(target_bp)
                        
            except Exception as e:
                logger.warning(f"Error processing sample {sample_idx} for {subject}: {e}")
                continue
        
        return subject_sequences, subject_targets
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return {
            'sequences': self.sequences[idx],
            'targets': self.targets[idx],
            'subject': self.subject_labels[idx]
        }


class ImprovedBiLSTMBPPredictor(nn.Module):
    """BiLSTM BP predictor with 10-frame temporal window and VAE encoder"""
    
    def __init__(self, vae_model, latent_dim: int = 64, 
                 hidden_dim: int = 128, num_layers: int = 2, num_heads: int = 4,
                 dropout: float = 0.4, use_attention: bool = True, 
                 pattern_offsets: List[int] = None, current_frame_bias: float = 3.0,  # Increased to focus on t+0
                 use_physiological_features: bool = True):
        super(ImprovedBiLSTMBPPredictor, self).__init__()
        
        self.vae = vae_model
        self.use_attention = use_attention
        self.use_physiological_features = use_physiological_features
        self.pattern_offsets = pattern_offsets or [-7, -6, -5, -4, -3, -2, -1, 0, 1, 2]
        
        # Freeze VAE parameters
        for param in self.vae.parameters():
            param.requires_grad = False
        
        # Enhanced input projection with physiological feature extraction
        input_dim = latent_dim
        if use_physiological_features:
            input_dim += 16  # Additional physiological features
            
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5)
        )
        
        # Physiological feature extractor
        if use_physiological_features:
            self.physio_extractor = nn.Sequential(
                nn.Linear(latent_dim, 32),
                nn.ReLU(inplace=True),
                nn.Linear(32, 16),
                nn.Tanh()
            )
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        lstm_output_dim = hidden_dim * 2
        
        # Self-attention mechanism
        if use_attention:
            self.attention = ImprovedMultiHeadAttention(
                hidden_dim=lstm_output_dim,
                num_heads=num_heads,
                dropout=dropout * 0.5,
                current_frame_bias=current_frame_bias
            )
        
        # Temporal processing
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(lstm_output_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        # Output layers
        self.output_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5)
        )
        
        # Prediction heads
        self.waveform_head = nn.Linear(hidden_dim // 4, 50)
        
        # Enhanced SBP/DBP heads
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
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights for better BP prediction"""
        import math
        
        # Initialize BP heads for physiological range
        with torch.no_grad():
            # Systolic head (90-180 mmHg)
            fan_in = self.systolic_head[-1].in_features
            std = math.sqrt(2.0 / fan_in)
            self.systolic_head[-1].weight.data.normal_(0, std)
            self.systolic_head[-1].bias.data.uniform_(110, 130)
            
            # Diastolic head (60-100 mmHg)
            self.diastolic_head[-1].weight.data.normal_(0, std)
            self.diastolic_head[-1].bias.data.uniform_(70, 90)
            
            # Waveform head
            if self.waveform_head.weight.dim() >= 2:
                torch.nn.init.kaiming_normal_(self.waveform_head.weight, mode='fan_out', nonlinearity='relu')
            if self.waveform_head.bias is not None:
                self.waveform_head.bias.data.uniform_(60, 80)
    
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
    """Multi-head attention with current frame bias"""
    
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
        
        # Add stronger positional bias for current frame (t+0)
        if pattern_offsets is not None and self.current_frame_bias > 0:
            try:
                current_frame_idx = pattern_offsets.index(0)
                bias_matrix = torch.zeros_like(scores[0, 0])
                
                # Strong bias toward current frame in all attention heads
                # When any position attends TO the current frame
                bias_matrix[:, current_frame_idx] += self.current_frame_bias * 2.0
                
                # Even stronger bias when current frame attends to anything
                bias_matrix[current_frame_idx, :] += self.current_frame_bias * 3.0
                
                # Maximum bias for current frame attending to itself
                bias_matrix[current_frame_idx, current_frame_idx] += self.current_frame_bias * 2.0
                
                # Apply bias to all heads
                bias_matrix = bias_matrix.unsqueeze(0).unsqueeze(0)
                scores = scores + bias_matrix.to(scores.device)
                
                # Add additional temporal decay bias (frames further from t+0 get less attention)
                for i, offset in enumerate(pattern_offsets):
                    if offset != 0:
                        temporal_distance = abs(offset)
                        decay_factor = 1.0 / (1.0 + temporal_distance * 0.2)  # Decay with distance
                        bias_matrix_decay = torch.zeros_like(scores[0, 0])
                        bias_matrix_decay[:, i] -= (1.0 - decay_factor) * self.current_frame_bias * 0.5
                        bias_matrix_decay = bias_matrix_decay.unsqueeze(0).unsqueeze(0)
                        scores = scores + bias_matrix_decay.to(scores.device)
                        
            except ValueError:
                pass
        
        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        attended = torch.matmul(attention_weights, V)
        attended = attended.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_dim)
        output = self.output_proj(attended)
        
        return self.layer_norm(x + output), attention_weights.mean(dim=1)


class ImprovedBPLoss(nn.Module):
    """Improved loss function with Huber loss and physiological constraints"""
    
    def __init__(self, waveform_weight=0.4, systolic_weight=0.3, diastolic_weight=0.3, 
                 huber_delta=1.0, physiological_constraint=True, pulse_pressure_weight=0.1):
        super(ImprovedBPLoss, self).__init__()
        self.waveform_weight = waveform_weight
        self.systolic_weight = systolic_weight
        self.diastolic_weight = diastolic_weight
        self.huber_delta = huber_delta
        self.physiological_constraint = physiological_constraint
        self.pulse_pressure_weight = pulse_pressure_weight
        
        # Normalize weights
        total = waveform_weight + systolic_weight + diastolic_weight
        self.waveform_weight /= total
        self.systolic_weight /= total
        self.diastolic_weight /= total
    
    def forward(self, predictions, targets):
        pred_waveform = predictions['waveform']
        
        # Extract ground truth values
        target_systolic, target_diastolic = self.extract_bp_values_improved(targets)
        
        # Waveform loss
        waveform_loss = nn.functional.huber_loss(pred_waveform, targets, delta=self.huber_delta)
        
        # SBP/DBP losses
        if 'systolic' in predictions:
            pred_systolic = predictions['systolic'].squeeze()
            systolic_loss = nn.functional.huber_loss(pred_systolic, target_systolic, delta=self.huber_delta)
            
            # Physiological constraint
            if self.physiological_constraint and 'diastolic' in predictions:
                pred_diastolic = predictions['diastolic'].squeeze()
                invalid_bp = (pred_systolic <= pred_diastolic).float()
                physiological_penalty = torch.mean(invalid_bp * torch.abs(pred_systolic - pred_diastolic))
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
        
        # Pulse pressure constraint
        pulse_pressure_loss = 0.0
        if self.physiological_constraint and 'systolic' in predictions and 'diastolic' in predictions:
            pred_pp = predictions['systolic'].squeeze() - predictions['diastolic'].squeeze()
            target_pp = target_systolic - target_diastolic
            pulse_pressure_loss = nn.functional.huber_loss(pred_pp, target_pp, delta=self.huber_delta)
        
        # Total loss
        total_loss = (self.waveform_weight * waveform_loss + 
                     self.systolic_weight * systolic_loss + 
                     self.diastolic_weight * diastolic_loss +
                     self.pulse_pressure_weight * pulse_pressure_loss)
        
        return {
            'total_loss': total_loss,
            'waveform_loss': waveform_loss,
            'systolic_loss': systolic_loss,
            'diastolic_loss': diastolic_loss,
            'pulse_pressure_loss': pulse_pressure_loss
        }
    
    def extract_bp_values_improved(self, waveform):
        """Extract systolic and diastolic values with smooth extraction"""
        batch_size, signal_length = waveform.shape
        device = waveform.device
        
        systolic_values = torch.zeros(batch_size, device=device)
        diastolic_values = torch.zeros(batch_size, device=device)
        
        for i in range(batch_size):
            signal = waveform[i]
            
            # Apply smoothing
            if signal_length > 7:
                kernel = torch.tensor([0.1, 0.2, 0.4, 0.2, 0.1], device=device)
                signal_smooth = torch.nn.functional.conv1d(
                    signal.unsqueeze(0).unsqueeze(0), 
                    kernel.unsqueeze(0).unsqueeze(0), 
                    padding=2
                ).squeeze()
            else:
                signal_smooth = signal
            
            # Systolic: maximum with parabolic interpolation
            sys_val, sys_idx = torch.max(signal_smooth, dim=0)
            sys_idx = sys_idx.item()
            
            if 1 <= sys_idx <= signal_length - 2:
                y1, y2, y3 = signal_smooth[sys_idx-1], signal_smooth[sys_idx], signal_smooth[sys_idx+1]
                a = (y1 + y3 - 2*y2) / 2
                if abs(a) > 1e-6:
                    offset = (y1 - y3) / (4 * a)
                    sys_val = y2 + a * offset * offset
            
            # Diastolic: minimum in post-systolic region
            search_start = max(sys_idx + 1, int(signal_length * 0.6))
            search_end = min(signal_length, int(signal_length * 0.95))
            
            if search_start < search_end:
                diastolic_window = signal_smooth[search_start:search_end]
                dias_val_rel, local_min_idx = torch.min(diastolic_window, dim=0)
                global_min_idx = search_start + local_min_idx.item()
                
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
                dias_val = torch.min(signal_smooth)
            
            # Ensure physiological constraint
            pulse_pressure = sys_val - dias_val
            min_pulse_pressure = 15.0
            
            if pulse_pressure < min_pulse_pressure:
                center_pressure = (sys_val + dias_val) / 2
                sys_val = center_pressure + min_pulse_pressure / 2
                dias_val = center_pressure - min_pulse_pressure / 2
            
            systolic_values[i] = sys_val
            diastolic_values[i] = dias_val
        
        return systolic_values, diastolic_values


# VAE definition (matching the trained checkpoint)
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


class BiLSTMEvaluator:
    """Evaluation class for the BiLSTM BP Predictor model"""
    
    def __init__(self, model, criterion, device):
        self.model = model
        self.criterion = criterion
        self.device = device
    
    def evaluate_model(self, val_loader, max_samples: int = 500):
        """Evaluate the model on validation data"""
        logger.info(f"🚀 Starting BiLSTM evaluation with max {max_samples} samples...")
        
        # Collect predictions
        all_predictions = []
        all_targets = []
        
        self.model.eval()
        sample_count = 0
        
        logger.info("🔍 Running model inference...")
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Evaluating"):
                if sample_count >= max_samples:
                    break
                    
                sequences = batch['sequences'].to(self.device)
                targets = batch['targets'].to(self.device)
                
                # Model prediction
                outputs = self.model(sequences)
                pred_waveforms = outputs['waveform']
                pred_systolic = outputs['systolic'].squeeze() if 'systolic' in outputs else None
                pred_diastolic = outputs['diastolic'].squeeze() if 'diastolic' in outputs else None
                
                # Extract ground truth values using same method as training
                target_systolic, target_diastolic = self.criterion.extract_bp_values_improved(targets)
                
                # Store results
                for i in range(len(targets)):
                    if sample_count >= max_samples:
                        break
                        
                    pred_dict = {
                        'waveform': pred_waveforms[i].cpu().numpy()
                    }
                    
                    target_dict = {
                        'waveform': targets[i].cpu().numpy(),
                        'systolic': target_systolic[i].cpu().numpy() if isinstance(target_systolic, torch.Tensor) else target_systolic,
                        'diastolic': target_diastolic[i].cpu().numpy() if isinstance(target_diastolic, torch.Tensor) else target_diastolic
                    }
                    
                    # Add direct predictions if available
                    if pred_systolic is not None:
                        pred_dict['systolic'] = pred_systolic[i].cpu().numpy()
                    else:
                        pred_sys, _ = extract_bp_values_numpy(pred_waveforms[i].cpu().numpy())
                        pred_dict['systolic'] = pred_sys
                    
                    if pred_diastolic is not None:
                        pred_dict['diastolic'] = pred_diastolic[i].cpu().numpy()
                    else:
                        _, pred_dias = extract_bp_values_numpy(pred_waveforms[i].cpu().numpy())
                        pred_dict['diastolic'] = pred_dias
                    
                    all_predictions.append(pred_dict)
                    all_targets.append(target_dict)
                    sample_count += 1
        
        logger.info(f"✅ Collected {len(all_predictions)} predictions")
        
        return {
            'predictions': all_predictions,
            'targets': all_targets
        }
    
    def calculate_metrics(self, predictions: List[Dict], targets: List[Dict]) -> Dict:
        """Calculate comprehensive evaluation metrics"""
        
        # Extract arrays
        pred_sys = np.array([float(p['systolic']) for p in predictions])
        pred_dias = np.array([float(p['diastolic']) for p in predictions])
        pred_waveforms = np.array([p['waveform'] for p in predictions])
        
        true_sys = np.array([float(t['systolic']) for t in targets])
        true_dias = np.array([float(t['diastolic']) for t in targets])
        true_waveforms = np.array([t['waveform'] for t in targets])
        
        metrics = {}
        
        # Systolic metrics
        metrics['systolic'] = {
            'r2': r2_score(true_sys, pred_sys),
            'mae': mean_absolute_error(true_sys, pred_sys),
            'rmse': np.sqrt(mean_squared_error(true_sys, pred_sys)),
            'mean_diff': np.mean(pred_sys - true_sys),
            'std_diff': np.std(pred_sys - true_sys),
            'pearson_r': stats.pearsonr(true_sys, pred_sys)[0],
            'pearson_p': stats.pearsonr(true_sys, pred_sys)[1]
        }
        
        # Diastolic metrics
        metrics['diastolic'] = {
            'r2': r2_score(true_dias, pred_dias),
            'mae': mean_absolute_error(true_dias, pred_dias),
            'rmse': np.sqrt(mean_squared_error(true_dias, pred_dias)),
            'mean_diff': np.mean(pred_dias - true_dias),
            'std_diff': np.std(pred_dias - true_dias),
            'pearson_r': stats.pearsonr(true_dias, pred_dias)[0],
            'pearson_p': stats.pearsonr(true_dias, pred_dias)[1]
        }
        
        # Waveform metrics
        waveform_r2_scores = []
        waveform_mae_scores = []
        for i in range(len(pred_waveforms)):
            waveform_r2_scores.append(r2_score(true_waveforms[i], pred_waveforms[i]))
            waveform_mae_scores.append(mean_absolute_error(true_waveforms[i], pred_waveforms[i]))
        
        metrics['waveform'] = {
            'r2': np.mean(waveform_r2_scores),
            'mae': np.mean(waveform_mae_scores),
            'rmse': np.sqrt(mean_squared_error(true_waveforms.flatten(), pred_waveforms.flatten())),
            'mean_diff': np.mean(pred_waveforms - true_waveforms),
            'std_diff': np.std(pred_waveforms - true_waveforms),
        }
        
        # Clinical accuracy metrics
        metrics['clinical'] = {
            'systolic_5mmhg': np.mean(np.abs(pred_sys - true_sys) <= 5) * 100,
            'systolic_10mmhg': np.mean(np.abs(pred_sys - true_sys) <= 10) * 100,
            'systolic_15mmhg': np.mean(np.abs(pred_sys - true_sys) <= 15) * 100,
            'diastolic_5mmhg': np.mean(np.abs(pred_dias - true_dias) <= 5) * 100,
            'diastolic_10mmhg': np.mean(np.abs(pred_dias - true_dias) <= 10) * 100,
            'diastolic_15mmhg': np.mean(np.abs(pred_dias - true_dias) <= 15) * 100,
        }
        
        return metrics
    
    def create_evaluation_plots(self, predictions: List[Dict], targets: List[Dict], 
                              output_dir: Path, model_name: str = "BiLSTM BP Predictor"):
        """Create comprehensive evaluation plots"""
        
        # Extract data
        pred_sys = np.array([float(p['systolic']) for p in predictions])
        pred_dias = np.array([float(p['diastolic']) for p in predictions])
        pred_waveforms = np.array([p['waveform'] for p in predictions])
        
        true_sys = np.array([float(t['systolic']) for t in targets])
        true_dias = np.array([float(t['diastolic']) for t in targets])
        true_waveforms = np.array([t['waveform'] for t in targets])
        
        # Calculate metrics
        metrics = self.calculate_metrics(predictions, targets)
        
        # Create figure
        fig = plt.figure(figsize=(15, 12))
        
        # Helper function for correlation plot
        def plot_correlation(ax, true_vals, pred_vals, title, metrics_dict):
            ax.scatter(true_vals, pred_vals, alpha=0.6, s=20)
            
            # Perfect correlation line
            min_val, max_val = min(true_vals.min(), pred_vals.min()), max(true_vals.max(), pred_vals.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.8, label='Perfect')
            
            # Regression line
            z = np.polyfit(true_vals, pred_vals, 1)
            p = np.poly1d(z)
            ax.plot(true_vals, p(true_vals), "r-", alpha=0.8, label='Fit')
            
            ax.set_xlabel(f'True {title} (mmHg)')
            ax.set_ylabel(f'Predicted {title} (mmHg)')
            ax.set_title(f'{title} Correlation')
            
            # Add metrics text
            r2 = metrics_dict.get('r2', 0)
            p_val = metrics_dict.get('pearson_p', 1)
            ax.text(0.05, 0.95, f'R²={r2:.4f}\np={p_val:.4f}', 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        # Helper function for Bland-Altman plot
        def plot_bland_altman(ax, true_vals, pred_vals, title, metrics_dict):
            diff = pred_vals - true_vals
            mean_vals = (pred_vals + true_vals) / 2
            
            ax.scatter(mean_vals, diff, alpha=0.6, s=20)
            
            # Mean difference line
            mean_diff = np.mean(diff)
            ax.axhline(mean_diff, color='blue', linestyle='-', alpha=0.8, label=f'Mean: {mean_diff:.2f}')
            
            # 95% limits of agreement
            std_diff = np.std(diff)
            upper_loa = mean_diff + 1.96 * std_diff
            lower_loa = mean_diff - 1.96 * std_diff
            
            ax.axhline(upper_loa, color='red', linestyle='--', alpha=0.8, label=f'+1.96SD: {upper_loa:.2f}')
            ax.axhline(lower_loa, color='red', linestyle='--', alpha=0.8, label=f'-1.96SD: {lower_loa:.2f}')
            
            ax.set_xlabel('Mean (mmHg)')
            ax.set_ylabel('Difference (mmHg)')
            ax.set_title(f'{title} Bland-Altman')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        # Helper function for error histogram
        def plot_error_histogram(ax, true_vals, pred_vals, title, metrics_dict):
            errors = np.abs(pred_vals - true_vals)
            
            # Create histogram with bins up to 40 mmHg
            bins = np.arange(0, 41, 1)
            counts, _, _ = ax.hist(errors, bins=bins, alpha=0.7, color='skyblue', edgecolor='black')
            
            # Add vertical lines for clinical thresholds
            ax.axvline(5, color='green', linestyle='--', alpha=0.8, linewidth=2)
            ax.axvline(10, color='orange', linestyle='--', alpha=0.8, linewidth=2)
            ax.axvline(15, color='red', linestyle='--', alpha=0.8, linewidth=2)
            
            ax.set_xlabel('Absolute error (mmHg)')
            ax.set_ylabel('Occurrences')
            ax.set_title(title)
            
            # Calculate statistics
            mae = np.mean(errors)
            std_error = np.std(errors)
            acc_5 = np.mean(errors <= 5) * 100
            acc_10 = np.mean(errors <= 10) * 100
            acc_15 = np.mean(errors <= 15) * 100
            
            # Create stats text box
            stats_text = f'mean: {mae:.2f}\nstd: {std_error:.2f}\n5-tol: {acc_5:.2f} %\n10-tol: {acc_10:.2f} %\n15-tol: {acc_15:.2f} %'
            ax.text(0.75, 0.9, stats_text, 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, 
                   verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, pad=0.5),
                   fontsize=10)
            
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 40)
        
        # Create subplots (3x3 grid)
        # Row 1: Full waveform
        ax1 = plt.subplot(3, 3, 1)
        plot_correlation(ax1, true_waveforms.flatten(), pred_waveforms.flatten(), 
                        'Waveform', metrics['waveform'])
        
        ax2 = plt.subplot(3, 3, 2)
        plot_bland_altman(ax2, true_waveforms.flatten(), pred_waveforms.flatten(), 
                         'Waveform', metrics['waveform'])
        
        ax3 = plt.subplot(3, 3, 3)
        plot_error_histogram(ax3, true_waveforms.flatten(), pred_waveforms.flatten(), 
                           'FULL', metrics['waveform'])
        
        # Row 2: Systolic BP
        ax4 = plt.subplot(3, 3, 4)
        plot_correlation(ax4, true_sys, pred_sys, 'Systolic', metrics['systolic'])
        
        ax5 = plt.subplot(3, 3, 5)
        plot_bland_altman(ax5, true_sys, pred_sys, 'Systolic', metrics['systolic'])
        
        ax6 = plt.subplot(3, 3, 6)
        plot_error_histogram(ax6, true_sys, pred_sys, 'SBP', metrics['systolic'])
        
        # Row 3: Diastolic BP
        ax7 = plt.subplot(3, 3, 7)
        plot_correlation(ax7, true_dias, pred_dias, 'Diastolic', metrics['diastolic'])
        
        ax8 = plt.subplot(3, 3, 8)
        plot_bland_altman(ax8, true_dias, pred_dias, 'Diastolic', metrics['diastolic'])
        
        ax9 = plt.subplot(3, 3, 9)
        plot_error_histogram(ax9, true_dias, pred_dias, 'DBP', metrics['diastolic'])
        
        # Add main title
        n_samples = len(predictions)
        fig.suptitle(f'{model_name} Evaluation | {n_samples} samples', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = output_dir / 'bilstm_evaluation.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"📊 Evaluation plot saved: {plot_path}")
        
        return metrics
    
    def save_results(self, metrics: Dict, output_dir: Path):
        """Save evaluation results to files"""
        
        # Save metrics to text file
        metrics_path = output_dir / 'bilstm_evaluation_metrics.txt'
        with open(metrics_path, 'w') as f:
            f.write("BILSTM BP PREDICTOR EVALUATION RESULTS\n")
            f.write("=" * 50 + "\n\n")
            
            for category in ['systolic', 'diastolic', 'waveform']:
                if category in metrics:
                    f.write(f"{category.upper()} METRICS:\n")
                    for metric, value in metrics[category].items():
                        f.write(f"  {metric}: {value:.4f}\n")
                    f.write("\n")
            
            if 'clinical' in metrics:
                f.write("CLINICAL ACCURACY:\n")
                for metric, value in metrics['clinical'].items():
                    f.write(f"  {metric}: {value:.2f}%\n")
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy_types(obj):
            """Recursively convert numpy types to native Python types"""
            if isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            elif isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj
        
        # Save metrics to JSON for programmatic access
        json_path = output_dir / 'bilstm_evaluation_metrics.json'
        json_compatible_metrics = convert_numpy_types(metrics)
        with open(json_path, 'w') as f:
            json.dump(json_compatible_metrics, f, indent=2)
        
        logger.info(f"📄 Metrics saved: {metrics_path}")
        logger.info(f"📄 JSON metrics saved: {json_path}")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Enhanced Multi-Subject BiLSTM Training with 10-frame window",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--config", 
        type=str, 
        required=True,
        help="Path to BiLSTM configuration YAML file"
    )
    parser.add_argument(
        "--vae-checkpoint",
        type=str,
        required=True,
        help="Path to trained VAE checkpoint"
    )
    parser.add_argument(
        "--max-subjects",
        type=int,
        default=None,
        help="Maximum number of subjects to use"
    )
    parser.add_argument(
        "--mask-type",
        type=str,
        default="mask10",
        choices=["auto", "metadata", "mask01", "mask05", "mask10", "mask15"],
        help="Mask type to use for all subjects"
    )
    parser.add_argument(
        "--data-root",
        type=str,
        help="Override data root directory"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to use for training"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )
    parser.add_argument(
        "--max-eval-samples",
        type=int,
        default=500,
        help="Maximum number of samples for evaluation"
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="Seed for reproducible subject-level splits (MUST match VAE training!)"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Training split ratio (default: 0.8 for 80%)"
    )
    parser.add_argument(
        "--val-ratio", 
        type=float,
        default=0.2,
        help="Validation split ratio (default: 0.2 for 20%)"
    )
    parser.add_argument(
        "--test-ratio", 
        type=float,
        default=0.2,
        help="Test split ratio (default: 0.2 for 20%)"
    )
    
    return parser.parse_args()


def setup_environment(args):
    """Setup environment variables and paths"""
    if args.data_root:
        os.environ['BP_DATA_ROOT'] = args.data_root
    
    path_config = PathConfig(
        data_root=args.data_root,
        experiments_root="./experiments"
    )
    path_manager = setup_paths(path_config)
    
    logger.info("Environment setup completed")
    logger.info(f"Data root: {path_manager.data_root}")
    logger.info(f"Experiments root: {path_manager.experiments_root}")
    
    return path_manager


def get_available_subjects(data_root: str) -> List[str]:
    """Get list of available subjects from data directory"""
    data_path = Path(data_root)
    subjects = []
    
    for file_path in data_path.glob("subject*_baseline_masked.h5"):
        subject = file_path.stem.split('_')[0]
        subjects.append(subject)
    
    return sorted(subjects)


def setup_device(device_arg: str):
    """Setup and validate device"""
    if device_arg == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"🚀 Using CUDA device: {torch.cuda.get_device_name()}")
        else:
            device = torch.device("cpu")
            logger.info("💻 Using CPU device")
    else:
        device = torch.device(device_arg)
        logger.info(f"📱 Using specified device: {device}")
    
    return device


def load_config_file(config_path):
    """Load YAML configuration file"""
    with open(config_path, 'r') as file:
        config_text = file.read()
    
    # Replace timestamp placeholder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_text = config_text.replace('{timestamp}', timestamp)
    
    config = yaml.safe_load(config_text)
    return config


def create_experiment_directory(path_manager, experiment_name=None):
    """Create experiment directory with timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if experiment_name is None:
        experiment_name = f"enhanced_bilstm_multisubject_{timestamp}"
    
    experiment_dir = path_manager.experiments_root / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (experiment_dir / "checkpoints").mkdir(exist_ok=True)
    (experiment_dir / "logs").mkdir(exist_ok=True)
    (experiment_dir / "results").mkdir(exist_ok=True)
    (experiment_dir / "evaluation").mkdir(exist_ok=True)
    (experiment_dir / "config").mkdir(exist_ok=True)
    
    logger.info(f"✅ Experiment directory created: {experiment_dir}")
    return experiment_dir


def save_system_info(experiment_dir):
    """Save system and environment information"""
    import platform
    import psutil
    
    system_info = {
        'system': {
            'platform': platform.platform(),
            'system': platform.system(),
            'release': platform.release(),
            'version': platform.version(),
            'machine': platform.machine(),
            'processor': platform.processor(),
            'python_version': platform.python_version(),
            'python_implementation': platform.python_implementation()
        },
        'hardware': {
            'cpu_count': psutil.cpu_count(),
            'cpu_count_logical': psutil.cpu_count(logical=True),
            'memory_total_gb': round(psutil.virtual_memory().total / (1024**3), 2),
            'memory_available_gb': round(psutil.virtual_memory().available / (1024**3), 2),
            'disk_usage_gb': round(psutil.disk_usage('/').total / (1024**3), 2)
        },
        'pytorch': {
            'version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
            'cudnn_version': torch.backends.cudnn.version() if torch.cuda.is_available() else None,
            'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'current_device': torch.cuda.current_device() if torch.cuda.is_available() else None
        },
        'environment': {
            'conda_env': os.environ.get('CONDA_DEFAULT_ENV', 'N/A'),
            'virtual_env': os.environ.get('VIRTUAL_ENV', 'N/A'),
            'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES', 'N/A'),
            'slurm_job_id': os.environ.get('SLURM_JOB_ID', 'N/A'),
            'slurm_nodelist': os.environ.get('SLURM_NODELIST', 'N/A'),
            'slurm_cpus_per_task': os.environ.get('SLURM_CPUS_PER_TASK', 'N/A'),
            'wandb_mode': os.environ.get('WANDB_MODE', 'N/A')
        }
    }
    
    # Add GPU information if available
    if torch.cuda.is_available():
        gpu_info = {}
        for i in range(torch.cuda.device_count()):
            gpu_info[f'gpu_{i}'] = {
                'name': torch.cuda.get_device_name(i),
                'memory_total_gb': round(torch.cuda.get_device_properties(i).total_memory / (1024**3), 2),
                'major': torch.cuda.get_device_properties(i).major,
                'minor': torch.cuda.get_device_properties(i).minor,
                'multi_processor_count': torch.cuda.get_device_properties(i).multi_processor_count
            }
        system_info['gpu'] = gpu_info
    
    # Try to get pip package list
    try:
        result = subprocess.run(['pip', 'list'], capture_output=True, text=True)
        if result.returncode == 0:
            system_info['pip_packages'] = result.stdout
    except:
        system_info['pip_packages'] = 'Unable to retrieve pip packages'
    
    # Save to file
    with open(experiment_dir / 'config' / 'system_info.json', 'w') as f:
        json.dump(system_info, f, indent=2, default=str)
    
    logger.info(f"✅ System information saved to: {experiment_dir / 'config' / 'system_info.json'}")
    return system_info


def main():
    """Main training function"""
    args = parse_arguments()
    
    # Setup logging
    level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(level=level)
    
    try:
        logger.info("🚀 Starting Enhanced Multi-Subject BiLSTM Training")
        logger.info("=" * 70)
        
        # Setup environment and paths
        path_manager = setup_environment(args)
        
        # Load configuration
        logger.info(f"Loading configuration from: {args.config}")
        config = load_config_file(args.config)
        
        # Extract configuration sections
        data_config = config.get('data_config', {})
        model_config = config.get('model_config', {})
        training_config = config.get('training_config', {})
        loss_config = config.get('loss_config', {})
        
        # Override data root if provided
        if args.data_root:
            data_config['root_path'] = args.data_root
        
        # Setup device
        device = setup_device(args.device)
        
        # Get available subjects
        subjects = get_available_subjects(data_config.get('root_path', args.data_root))
        if args.max_subjects:
            subjects = subjects[:args.max_subjects]
        
        logger.info(f"📊 Target subjects: {subjects}")
        logger.info(f"📈 Total subjects: {len(subjects)}")
        
        # 🔒 CRITICAL: Create subject-level splits (SAME as VAE training)
        logger.info("=" * 80)
        logger.info("🔒 PREVENTING DATA LEAKAGE WITH SUBJECT-LEVEL SPLITS")
        logger.info("🔗 MUST MATCH VAE TRAINING SPLITS!")
        logger.info("=" * 80)
        
        train_subjects, val_subjects, test_subjects = create_subject_level_splits(
            subjects, 
            train_ratio=args.train_ratio, 
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio, 
            seed=args.split_seed
        )
        
        # Load pre-trained VAE
        logger.info(f"Loading pre-trained VAE from: {args.vae_checkpoint}")
        vae_config = model_config.get('vae_config', {})
        vae = VAE(latent_dim=vae_config.get('latent_dim', 64))
        
        if os.path.exists(args.vae_checkpoint):
            checkpoint = torch.load(args.vae_checkpoint, map_location=device, weights_only=False)
            if 'model_state_dict' in checkpoint:
                vae.load_state_dict(checkpoint['model_state_dict'])
            else:
                vae.load_state_dict(checkpoint)
            logger.info(f"✅ VAE loaded successfully")
        else:
            raise FileNotFoundError(f"VAE checkpoint not found: {args.vae_checkpoint}")
        
        vae.to(device)
        vae.eval()
        
        # Create enhanced multi-subject datasets with 10-frame window (SUBJECT-LEVEL SPLITS)
        logger.info("Creating enhanced multi-subject datasets with 10-frame window...")
        logger.info("🔒 Using subject-level splits - NO DATA LEAKAGE")
        pattern_offsets = data_config.get('pattern_offsets', [-7, -6, -5, -4, -3, -2, -1, 0, 1, 2])
        
        # Create TRAINING dataset from training subjects only
        logger.info("📈 Creating TRAINING dataset from training subjects...")
        train_dataset = ImprovedBiLSTMDataset(
            data_root=data_config.get('root_path', args.data_root),
            subjects=train_subjects,  # Only training subjects
            mask_type=args.mask_type,
            pattern_offsets=pattern_offsets,
            max_samples_per_subject=data_config.get('max_samples_per_subject', 1000),
            sequence_step_size=data_config.get('sequence_step_size', 10),
            session=data_config.get('session', 'baseline'),
            use_augmentation=data_config.get('use_augmentation', False),
            noise_level=data_config.get('noise_level', 0.005)
        )
        
        # Create VALIDATION dataset from validation subjects only
        logger.info("📉 Creating VALIDATION dataset from validation subjects...")
        val_dataset = ImprovedBiLSTMDataset(
            data_root=data_config.get('root_path', args.data_root),
            subjects=val_subjects,  # Only validation subjects
            mask_type=args.mask_type,
            pattern_offsets=pattern_offsets,
            max_samples_per_subject=data_config.get('max_samples_per_subject', 1000),
            sequence_step_size=data_config.get('sequence_step_size', 10),
            session=data_config.get('session', 'baseline'),
            use_augmentation=data_config.get('use_augmentation', False),
            noise_level=data_config.get('noise_level', 0.005)
        )
        
        # Create TEST dataset from test subjects only (ISOLATED for final evaluation)
        logger.info("🧪 Creating TEST dataset from test subjects...")
        logger.info("🔒 CRITICAL: Test subjects NEVER seen during training")
        test_dataset = ImprovedBiLSTMDataset(
            data_root=data_config.get('root_path', args.data_root),
            subjects=test_subjects,  # Only test subjects (ISOLATED)
            mask_type=args.mask_type,
            pattern_offsets=pattern_offsets,
            max_samples_per_subject=data_config.get('max_samples_per_subject', 1000),
            sequence_step_size=data_config.get('sequence_step_size', 10),
            session=data_config.get('session', 'baseline'),
            use_augmentation=False,  # No augmentation for test evaluation
            noise_level=0.0  # No noise for test evaluation
        )
        
        logger.info(f"✅ Datasets created with NO LEAKAGE:")
        logger.info(f"   Train: {len(train_dataset)} samples from {len(train_subjects)} subjects")
        logger.info(f"   Val: {len(val_dataset)} samples from {len(val_subjects)} subjects")
        logger.info(f"   Test: {len(test_dataset)} samples from {len(test_subjects)} subjects")
        logger.info(f"🔒 GUARANTEE: Zero subject overlap between train/val/test")
        
        # Create data loaders
        batch_size = training_config.get('batch_size', 16)
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=0,
            pin_memory=False
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=0,
            pin_memory=False
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=0,
            pin_memory=False
        )
        
        logger.info(f"✅ Data loaders created:")
        logger.info(f"   Train batches: {len(train_loader)}")
        logger.info(f"   Validation batches: {len(val_loader)}")
        logger.info(f"   Test batches: {len(test_loader)}")
        logger.info(f"   Batch size: {batch_size}")
        
        # Create BiLSTM model
        logger.info("Creating BiLSTM BP predictor...")
        bilstm_config = model_config.get('bilstm_config', {})
        attention_config = model_config.get('attention_config', {})
        
        model = ImprovedBiLSTMBPPredictor(
            vae_model=vae,
            latent_dim=vae_config.get('latent_dim', 64),
            hidden_dim=bilstm_config.get('hidden_dim', 128),
            num_layers=bilstm_config.get('num_layers', 2),
            num_heads=attention_config.get('num_attention_heads', 4),
            dropout=bilstm_config.get('dropout_rate', 0.4),
            use_attention=attention_config.get('use_attention', True),
            pattern_offsets=pattern_offsets,
            current_frame_bias=attention_config.get('current_frame_bias', 3.0),  # Increased from 1.5 to 3.0
            use_physiological_features=attention_config.get('use_physiological_features', True)
        ).to(device)
        
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"✅ BiLSTM model created with {total_params:,} trainable parameters")
        
        # Log attention configuration
        current_frame_bias = attention_config.get('current_frame_bias', 3.0)
        logger.info(f"🎯 Attention bias configuration:")
        logger.info(f"   Current frame bias (t+0): {current_frame_bias}")
        logger.info(f"   Pattern offsets: {pattern_offsets}")
        logger.info(f"   Current frame position: {pattern_offsets.index(0) if 0 in pattern_offsets else 'Not found'}")
        logger.info(f"   Strong focus on t+0 for prediction target")
        
        # Create experiment directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"enhanced_bilstm_multisubject_{len(subjects)}subjects_{args.mask_type}_{timestamp}"
        experiment_dir = create_experiment_directory(path_manager, experiment_name)
        
        # Save input configuration and arguments for reproducibility
        config_save_path = experiment_dir / 'config' / 'input_config.yaml'
        with open(config_save_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        
        args_save_path = experiment_dir / 'config' / 'command_line_args.json'
        with open(args_save_path, 'w') as f:
            json.dump(vars(args), f, indent=2, default=str)
        
        # Save system and environment information
        system_info = save_system_info(experiment_dir)
        
        logger.info(f"✅ Configuration files saved to: {experiment_dir / 'config'}")
        logger.info(f"   ├── Input config: input_config.yaml")
        logger.info(f"   ├── Command args: command_line_args.json")
        logger.info(f"   └── System info: system_info.json")
        
        # Initialize loss and optimizer
        loss_weights = loss_config.get('loss_weights', {})
        criterion = ImprovedBPLoss(
            waveform_weight=loss_weights.get('waveform_weight', 0.4),
            systolic_weight=loss_weights.get('systolic_weight', 0.3),
            diastolic_weight=loss_weights.get('diastolic_weight', 0.3),
            huber_delta=loss_config.get('loss_params', {}).get('huber_delta', 1.0),
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
        logger.info(f"🎯 Starting training for {num_epochs} epochs...")
        
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
            logger.info(f"\nEpoch {epoch}/{num_epochs}")
            
            # Training phase
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
                    logger.warning(f"Training error: {e}")
                    continue
            
            # Validation phase
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
                        
                        # Store SBP/DBP predictions and targets
                        if 'systolic' in outputs and 'diastolic' in outputs:
                            val_systolic_predictions.append(outputs['systolic'].cpu().numpy())
                            val_diastolic_predictions.append(outputs['diastolic'].cpu().numpy())
                            
                            # Extract ground truth SBP/DBP
                            target_sys, target_dias = criterion.extract_bp_values_improved(targets)
                            val_systolic_targets.append(target_sys.cpu().numpy())
                            val_diastolic_targets.append(target_dias.cpu().numpy())
                        
                    except Exception as e:
                        logger.warning(f"Validation error: {e}")
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
            
            logger.info(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val MAE: {val_mae:.2f}")
            logger.info(f"  └─ Waveform: T={train_waveform_loss:.4f}, V={val_waveform_loss:.4f}")
            logger.info(f"  └─ Systolic:  T={train_systolic_loss:.4f}, V={val_systolic_loss:.4f}, MAE={val_systolic_mae:.2f}")
            logger.info(f"  └─ Diastolic: T={train_diastolic_loss:.4f}, V={val_diastolic_loss:.4f}, MAE={val_diastolic_mae:.2f}")
            logger.info(f"  └─ PulsePres: T={train_pulse_pressure_loss:.4f}, V={val_pulse_pressure_loss:.4f}")
            
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
                    'training_history': training_history,
                    'subjects': subjects,
                    'mask_type': args.mask_type,
                    'pattern_offsets': pattern_offsets,
                    # Include comprehensive metadata in checkpoint
                    'full_metadata': {
                        'command_line_args': vars(args),
                        'data_config': data_config,
                        'model_config': model_config,
                        'training_config': training_config,
                        'loss_config': loss_config,
                        'vae_checkpoint': args.vae_checkpoint,
                        'experiment_timestamp': datetime.now().isoformat(),
                        'total_trainable_params': sum(p.numel() for p in model.parameters() if p.requires_grad)
                    }
                }
                
                torch.save(checkpoint, experiment_dir / 'checkpoints' / 'best_model.pt')
                logger.info(f"✅ New best model saved! Val MAE: {best_val_mae:.2f} (SBP: {val_systolic_mae:.2f}, DBP: {val_diastolic_mae:.2f})")
        
        # Plot training curves
        logger.info("📊 Generating training curves...")
        plot_training_curves(training_history, experiment_dir / 'results')
        
        # Generate prediction examples
        logger.info("🔍 Generating prediction examples...")
        generate_prediction_examples(
            model=model,
            val_loader=val_loader,
            device=device,
            criterion=criterion,
            pattern_offsets=pattern_offsets,
            save_dir=experiment_dir / 'results',
            num_examples=8
        )
        
        # Save comprehensive experiment metadata with all configuration and parameters
        experiment_metadata = {
            # Basic experiment info
            'experiment_name': experiment_name,
            'timestamp': timestamp,
            'experiment_dir': str(experiment_dir),
            
            # Command line arguments
            'command_line_args': {
                'config': args.config,
                'vae_checkpoint': args.vae_checkpoint,
                'max_subjects': args.max_subjects,
                'mask_type': args.mask_type,
                'data_root': args.data_root,
                'device': args.device,
                'debug': args.debug,
                'max_eval_samples': args.max_eval_samples
            },
            
            # Full configuration from YAML
            'full_config': config,
            
            # Data configuration with NO LEAKAGE
            'data_config': {
                'all_subjects': subjects,
                'train_subjects': train_subjects,
                'val_subjects': val_subjects,
                'test_subjects': test_subjects,  # CRITICAL: Save test subjects for evaluation
                'num_total_subjects': len(subjects),
                'num_train_subjects': len(train_subjects),
                'num_val_subjects': len(val_subjects),
                'num_test_subjects': len(test_subjects),
                'mask_type': args.mask_type,
                'pattern_offsets': pattern_offsets,
                'sequence_length': len(pattern_offsets),
                'data_root': data_config.get('root_path', args.data_root),
                'session': data_config.get('session', 'baseline'),
                'max_samples_per_subject': data_config.get('max_samples_per_subject', 1000),
                'sequence_step_size': data_config.get('sequence_step_size', 10),
                'use_augmentation': data_config.get('use_augmentation', False),
                'noise_level': data_config.get('noise_level', 0.005),
                'train_samples': len(train_dataset),
                'val_samples': len(val_dataset),
                'total_samples': len(train_dataset) + len(val_dataset),
                'split_method': 'subject_level_splits'
            },
            
            # Leakage prevention configuration
            'leakage_prevention': {
                'method': 'subject_level_splits',
                'seed': args.split_seed,
                'train_ratio': args.train_ratio,
                'val_ratio': args.val_ratio,
                'test_ratio': args.test_ratio,
                'no_subject_overlap': len(set(train_subjects) & set(val_subjects) & set(test_subjects)) == 0,
                'vae_consistency': 'same_seed_as_vae_training',
                'validation_passed': True,
                'test_set_isolated': True
            },
            
            # Model configuration
            'model_config': {
                'model_type': 'ImprovedBiLSTMBPPredictor',
                'vae_latent_dim': vae_config.get('latent_dim', 64),
                'vae_checkpoint': args.vae_checkpoint,
                'vae_frozen': vae_config.get('freeze_vae', True),
                'bilstm_hidden_dim': bilstm_config.get('hidden_dim', 128),
                'bilstm_num_layers': bilstm_config.get('num_layers', 2),
                'bilstm_dropout': bilstm_config.get('dropout_rate', 0.4),
                'attention_enabled': attention_config.get('use_attention', True),
                'attention_heads': attention_config.get('num_attention_heads', 4),
                'current_frame_bias': attention_config.get('current_frame_bias', 3.0),
                'use_physiological_features': attention_config.get('use_physiological_features', True),
                'total_trainable_params': sum(p.numel() for p in model.parameters() if p.requires_grad),
                'total_params': sum(p.numel() for p in model.parameters())
            },
            
            # Training configuration
            'training_config': {
                'num_epochs': training_config.get('num_epochs', 50),
                'batch_size': training_config.get('batch_size', 16),
                'learning_rate': float(training_config.get('learning_rate', 0.0005)),
                'weight_decay': float(training_config.get('weight_decay', 1e-4)),
                'optimizer': 'AdamW',
                'optimizer_betas': training_config.get('optimizer_config', {}).get('betas', [0.9, 0.999]),
                'optimizer_amsgrad': training_config.get('optimizer_config', {}).get('amsgrad', True),
                'scheduler': 'ReduceLROnPlateau',
                'scheduler_patience': training_config.get('scheduler_config', {}).get('scheduler_patience', 5),
                'scheduler_factor': float(training_config.get('scheduler_config', {}).get('factor', 0.7)),
                'scheduler_min_lr': float(training_config.get('scheduler_config', {}).get('min_lr', 1e-6)),
                'grad_clip_norm': training_config.get('optimization', {}).get('grad_clip_norm', 0.5),
                'early_stopping_patience': training_config.get('early_stopping_patience', 15),
                'early_stopping_min_delta': training_config.get('early_stopping_min_delta', 1e-5)
            },
            
            # Loss configuration
            'loss_config': {
                'loss_type': 'ImprovedBPLoss',
                'waveform_weight': loss_config.get('loss_weights', {}).get('waveform_weight', 0.4),
                'systolic_weight': loss_config.get('loss_weights', {}).get('systolic_weight', 0.3),
                'diastolic_weight': loss_config.get('loss_weights', {}).get('diastolic_weight', 0.3),
                'pulse_pressure_weight': loss_config.get('pulse_pressure_weight', 0.1),
                'huber_delta': loss_config.get('loss_params', {}).get('huber_delta', 1.0),
                'physiological_constraint': loss_config.get('physiological_constraint', True),
                'min_pulse_pressure': 15.0
            },
            
            # Hardware and environment
            'hardware_config': {
                'device': str(device),
                'cuda_available': torch.cuda.is_available(),
                'cuda_device_name': torch.cuda.get_device_name() if torch.cuda.is_available() else None,
                'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
                'num_workers': 0,
                'pin_memory': False
            },
            
                         # Environment info
             'environment_info': {
                 'python_version': sys.version,
                 'torch_version': torch.__version__,
                 'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
                 'working_directory': str(Path.cwd()),
                 'script_path': str(Path(__file__).absolute()),
                 'config_path': str(Path(args.config).absolute()),
                 'data_root_path': str(Path(data_config.get('root_path', args.data_root)).absolute()),
                 'experiment_timestamp': datetime.now().isoformat()
             },
             
             # Complete system information
             'system_info': system_info,
            
            # Training results
            'training_results': {
                'epochs_completed': len(training_history['train_loss']),
                'best_val_mae': best_val_mae,
                'final_train_loss': training_history['train_loss'][-1] if training_history['train_loss'] else None,
                'final_val_loss': training_history['val_loss'][-1] if training_history['val_loss'] else None,
                'final_val_mae': training_history['val_mae'][-1] if training_history['val_mae'] else None,
                'final_sys_mae': training_history['val_systolic_mae'][-1] if training_history['val_systolic_mae'] else 0,
                'final_dias_mae': training_history['val_diastolic_mae'][-1] if training_history['val_diastolic_mae'] else 0,
                'final_learning_rate': training_history['learning_rates'][-1] if training_history['learning_rates'] else None,
                'best_epoch': training_history['val_mae'].index(min(training_history['val_mae'])) + 1 if training_history['val_mae'] else None
            },
            
            # Dataset statistics with leakage prevention
            'dataset_stats': {
                'available_subjects': get_available_subjects(data_config.get('root_path', args.data_root)),
                'total_available_subjects': len(get_available_subjects(data_config.get('root_path', args.data_root))),
                'used_subjects': subjects,
                'used_subjects_count': len(subjects),
                'train_subjects': train_subjects,
                'val_subjects': val_subjects,
                'excluded_subjects': [s for s in get_available_subjects(data_config.get('root_path', args.data_root)) if s not in subjects],
                'max_samples_per_subject': data_config.get('max_samples_per_subject', 1000),
                'train_batches': len(train_loader),
                'val_batches': len(val_loader),
                'leakage_free_validation': True
            }
        }
        
        with open(experiment_dir / "experiment_metadata.json", 'w') as f:
            json.dump(experiment_metadata, f, indent=2, default=str)
        
        logger.info("=" * 80)
        logger.info("✅ LEAKAGE-FREE multi-subject BiLSTM training completed successfully!")
        logger.info("🔒 GUARANTEED: No data leakage - subject-level splits enforced")
        logger.info("🔗 Same subject splits as VAE training (same seed)")
        logger.info("=" * 80)
        logger.info(f"📁 Results saved to: {experiment_dir}")
        logger.info(f"📊 Training subjects: {len(train_subjects)} subjects")
        logger.info(f"📊 Validation subjects: {len(val_subjects)} subjects")
        logger.info(f"🧪 TEST subjects: {len(test_subjects)} subjects (ISOLATED for evaluation)")
        logger.info(f"📊 Mask type used: {args.mask_type}")
        logger.info(f"🔢 Split seed used: {args.split_seed} (matched VAE training)")
        logger.info(f"🎯 Best validation MAE: {best_val_mae:.2f} mmHg")
        logger.info(f"📈 Final SBP MAE: {experiment_metadata['training_results']['final_sys_mae']:.2f} mmHg")
        logger.info(f"📈 Final DBP MAE: {experiment_metadata['training_results']['final_dias_mae']:.2f} mmHg")
        logger.info(f"⏱️  Pattern offsets: {pattern_offsets}")
        logger.info(f"📋 Train subjects: {train_subjects}")
        logger.info(f"📋 Val subjects: {val_subjects}")
        logger.info(f"🧪 TEST subjects: {test_subjects}")
        logger.info("⚠️  CRITICAL: Test subjects saved in metadata for final evaluation!")
        
        # Log saved configuration files
        logger.info("\n📋 CONFIGURATION FILES SAVED:")
        logger.info(f"  ├── Input config: {experiment_dir / 'config' / 'input_config.yaml'}")
        logger.info(f"  ├── Command args: {experiment_dir / 'config' / 'command_line_args.json'}")
        logger.info(f"  ├── System info: {experiment_dir / 'config' / 'system_info.json'}")
        logger.info(f"  └── Full metadata: {experiment_dir / 'experiment_metadata.json'}")
        
        # ========================================
        # COMPREHENSIVE MODEL EVALUATION
        # ========================================
        logger.info("\n" + "=" * 80)
        logger.info("🩺 STARTING COMPREHENSIVE MODEL EVALUATION")
        logger.info("=" * 80)
        
        # Load best model for evaluation
        logger.info("Loading best model for evaluation...")
        best_checkpoint = torch.load(experiment_dir / 'checkpoints' / 'best_model.pt', 
                                    map_location=device, weights_only=False)
        model.load_state_dict(best_checkpoint['model_state_dict'])
        model.eval()
        
        # Create evaluator
        evaluator = BiLSTMEvaluator(model, criterion, device)
        
        # Run evaluation on TEST set (NO DATA LEAKAGE!)
        logger.info("🧪 EVALUATING ON TEST SET - NEVER SEEN DURING TRAINING")
        evaluation_results = evaluator.evaluate_model(test_loader, args.max_eval_samples)
        
        # Calculate comprehensive metrics
        metrics = evaluator.calculate_metrics(
            evaluation_results['predictions'], 
            evaluation_results['targets']
        )
        
        # Create evaluation plots
        evaluator.create_evaluation_plots(
            evaluation_results['predictions'], 
            evaluation_results['targets'], 
            experiment_dir / 'evaluation', 
            "Enhanced BiLSTM BP Predictor"
        )
        
        # Save evaluation results
        evaluator.save_results(metrics, experiment_dir / 'evaluation')
        
        # Print comprehensive evaluation summary
        logger.info("\n" + "=" * 80)
        logger.info("🎉 TEST SET EVALUATION COMPLETED!")
        logger.info("🔒 FINAL RESULTS - NO DATA LEAKAGE GUARANTEED")
        logger.info("=" * 80)
        logger.info(f"📊 Samples evaluated: {len(evaluation_results['predictions'])}")
        
        logger.info("\n📈 KEY EVALUATION METRICS:")
        logger.info("-" * 40)
        logger.info(f"Systolic  MAE: {metrics['systolic']['mae']:.2f} mmHg (R²={metrics['systolic']['r2']:.3f})")
        logger.info(f"Diastolic MAE: {metrics['diastolic']['mae']:.2f} mmHg (R²={metrics['diastolic']['r2']:.3f})")
        logger.info(f"Waveform  MAE: {metrics['waveform']['mae']:.2f} mmHg (R²={metrics['waveform']['r2']:.3f})")
        
        logger.info("\n🎯 CLINICAL ACCURACY:")
        logger.info(f"Systolic  ≤5mmHg:  {metrics['clinical']['systolic_5mmhg']:.1f}%")
        logger.info(f"Systolic  ≤10mmHg: {metrics['clinical']['systolic_10mmhg']:.1f}%")
        logger.info(f"Systolic  ≤15mmHg: {metrics['clinical']['systolic_15mmhg']:.1f}%")
        logger.info(f"Diastolic ≤5mmHg:  {metrics['clinical']['diastolic_5mmhg']:.1f}%")
        logger.info(f"Diastolic ≤10mmHg: {metrics['clinical']['diastolic_10mmhg']:.1f}%")
        logger.info(f"Diastolic ≤15mmHg: {metrics['clinical']['diastolic_15mmhg']:.1f}%")
        
        logger.info(f"\n📁 Complete results saved to: {experiment_dir}")
        logger.info(f"📊 Training results: {experiment_dir / 'results'}")
        logger.info(f"📋 Evaluation results: {experiment_dir / 'evaluation'}")
        
        # Complete file structure summary
        logger.info("\n📂 COMPLETE EXPERIMENT STRUCTURE:")
        logger.info("=" * 50)
        logger.info(f"📁 {experiment_dir.name}/")
        logger.info(f"  ├── 📁 checkpoints/")
        logger.info(f"  │   └── 🔗 best_model.pt (model weights + full metadata)")
        logger.info(f"  ├── 📁 config/")
        logger.info(f"  │   ├── 📄 input_config.yaml (original YAML config)")
        logger.info(f"  │   ├── 📄 command_line_args.json (CLI arguments)")
        logger.info(f"  │   └── 📄 system_info.json (hardware + environment)")
        logger.info(f"  ├── 📁 evaluation/")
        logger.info(f"  │   ├── 📊 bilstm_evaluation.png (comprehensive plots)")
        logger.info(f"  │   ├── 📄 bilstm_evaluation_metrics.txt (readable metrics)")
        logger.info(f"  │   └── 📄 bilstm_evaluation_metrics.json (JSON metrics)")
        logger.info(f"  ├── 📁 results/")
        logger.info(f"  │   ├── 📈 enhanced_bilstm_training_curves.png")
        logger.info(f"  │   ├── 🔍 bilstm_prediction_examples.png")
        logger.info(f"  │   └── 📄 bilstm_prediction_stats.txt")
        logger.info(f"  └── 📄 experiment_metadata.json (comprehensive metadata)")
        logger.info("=" * 50)
        logger.info("🎯 All files contain complete configuration for reproducibility!")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        raise


def plot_training_curves(training_history, save_dir):
    """Plot enhanced training curves"""
    save_dir.mkdir(exist_ok=True)
    
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
    
    # Plot 10: Loss Components Comparison
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
    
    # Plot 11: MAE Components Comparison
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
    plt.savefig(save_dir / 'enhanced_bilstm_training_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✅ Training curves saved to: {save_dir / 'enhanced_bilstm_training_curves.png'}")


def generate_prediction_examples(model, val_loader, device, criterion, pattern_offsets, save_dir, num_examples=8):
    """Generate prediction examples and attention visualizations"""
    save_dir.mkdir(exist_ok=True)
    
    model.eval()
    examples = []
    
    logger.info(f"Generating {num_examples} prediction examples...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if len(examples) >= num_examples:
                break
                
            sequences = batch['sequences'].to(device)
            targets = batch['targets'].to(device)
            
            # Get predictions with attention
            outputs = model(sequences, return_attention=True)
            pred_waveform = outputs['waveform'].cpu().numpy()
            target_waveform = targets.cpu().numpy()
            
            # Get attention weights if available
            attention_weights = None
            if 'attention_weights' in outputs:
                attention_weights = outputs['attention_weights'].cpu().numpy()
            
            # Process batch examples
            batch_size = sequences.shape[0]
            for i in range(min(batch_size, num_examples - len(examples))):
                # Extract BP values
                pred_sys, pred_dias = extract_bp_values_numpy(pred_waveform[i])
                target_sys, target_dias = extract_bp_values_numpy(target_waveform[i])
                
                examples.append({
                    'pred_waveform': pred_waveform[i],
                    'target_waveform': target_waveform[i],
                    'pred_sys': pred_sys,
                    'pred_dias': pred_dias,
                    'target_sys': target_sys,
                    'target_dias': target_dias,
                    'attention_weights': attention_weights[i] if attention_weights is not None else None,
                    'sequences': sequences[i].cpu().numpy()
                })
    
    # Create visualization
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
        ax1.set_title(f'BiLSTM Example {i+1}: BP Waveform Prediction\n'
                     f'MAE: {np.mean(np.abs(example["pred_waveform"] - example["target_waveform"])):.2f} mmHg')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Plot attention heatmap
        ax2 = axes[i, 1]
        if example['attention_weights'] is not None:
            sns.heatmap(example['attention_weights'], 
                       xticklabels=[f't{offset:+d}' for offset in pattern_offsets],
                       yticklabels=[f't{offset:+d}' for offset in pattern_offsets],
                       annot=True, fmt='.3f', cmap='Blues', ax=ax2)
            ax2.set_title(f'Attention Weights\n(10-Frame Window)')
            ax2.set_xlabel('Key Frames')
            ax2.set_ylabel('Query Frames')
        else:
            ax2.text(0.5, 0.5, 'No Attention\nWeights Available', 
                    ha='center', va='center', transform=ax2.transAxes, fontsize=12)
            ax2.set_title('Attention Weights')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'bilstm_prediction_examples.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✅ Prediction examples saved to: {save_dir / 'bilstm_prediction_examples.png'}")
    
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
    
    with open(save_dir / 'bilstm_prediction_stats.txt', 'w') as f:
        f.write("BILSTM PREDICTION EXAMPLES STATISTICS\n")
        f.write("="*40 + "\n\n")
        f.write(f"Number of examples: {stats['num_examples']}\n")
        f.write(f"Average MAE: {stats['avg_mae']:.2f} mmHg\n")
        f.write(f"Average Systolic Error: {stats['avg_sys_error']:.2f} mmHg\n")
        f.write(f"Average Diastolic Error: {stats['avg_dias_error']:.2f} mmHg\n")
        f.write(f"Systolic Correlation: {stats['sys_correlation']:.3f}\n")
        f.write(f"Diastolic Correlation: {stats['dias_correlation']:.3f}\n")
    
    logger.info(f"📊 BiLSTM Prediction Summary:")
    logger.info(f"   Average MAE: {stats['avg_mae']:.2f} mmHg")
    logger.info(f"   Systolic Error: {stats['avg_sys_error']:.2f} mmHg")
    logger.info(f"   Diastolic Error: {stats['avg_dias_error']:.2f} mmHg")
    
    return examples, stats


def extract_bp_values_numpy(waveform):
    """Extract systolic and diastolic values (numpy version)"""
    if isinstance(waveform, torch.Tensor):
        waveform = waveform.detach().cpu().numpy()
    
    if len(waveform.shape) > 1:
        waveform = waveform.flatten()
    
    # Apply smoothing
    if len(waveform) > 7:
        from scipy import ndimage
        smoothed = ndimage.gaussian_filter1d(waveform, sigma=1.0)
    else:
        smoothed = waveform
    
    # Systolic: maximum
    sys_idx = np.argmax(smoothed)
    systolic = smoothed[sys_idx]
    
    # Diastolic: minimum in post-systolic region
    search_start = max(sys_idx + 1, int(len(smoothed) * 0.6))
    search_end = min(len(smoothed), int(len(smoothed) * 0.95))
    
    if search_start < search_end:
        post_systolic = smoothed[search_start:search_end]
        local_min_idx = np.argmin(post_systolic)
        diastolic = post_systolic[local_min_idx]
    else:
        diastolic = np.min(smoothed)
    
    # Ensure physiological constraint
    if systolic - diastolic < 15.0:
        center = (systolic + diastolic) / 2
        systolic = center + 7.5
        diastolic = center - 7.5
    
    return systolic, diastolic


if __name__ == "__main__":
    main()