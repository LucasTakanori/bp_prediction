#!/usr/bin/env python3
"""
Enhanced Multi-Subject Transformer Training Script
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
import math
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
            if overlap:
                raise ValueError(f"CRITICAL ERROR: Data leakage detected! {set_names[i]} and {set_names[j]} "
                               f"sets have overlapping subjects: {overlap}")
    
    logger.info("✅ Subject-level splits created successfully - NO DATA LEAKAGE")
    logger.info("🔒 Same subjects will be used as in VAE training (same seed)")
    logger.info("🧪 TEST SET ISOLATED: Test subjects will not be seen during training")
    
    return train_subjects, val_subjects, test_subjects


class ImprovedTransformerDataset(Dataset):
    """Enhanced dataset for Transformer-based blood pressure prediction with VAE latent representations"""
    
    def __init__(self, data_root: str, subjects: List[str], mask_type: str = "mask10",
                 pattern_offsets: List[int] = None, max_samples_per_subject: int = 200, 
                 sequence_step_size: int = 10, session: str = "baseline",
                 use_augmentation: bool = False, noise_level: float = 0.005):
        
        self.data_root = Path(data_root)
        self.subjects = subjects
        self.mask_type = mask_type
        self.pattern_offsets = pattern_offsets or [-7, -6, -5, -4, -3, -2, -1, 0, 1, 2]
        self.max_samples_per_subject = max_samples_per_subject
        self.sequence_step_size = sequence_step_size
        self.session = session
        self.use_augmentation = use_augmentation
        self.noise_level = noise_level
        
        # Storage for all data
        self.sequences = []
        self.targets = []
        self.subject_ids = []
        self.dataset_masks = []
        
        logger.info(f"Creating Transformer dataset with {len(subjects)} subjects")
        logger.info(f"Pattern offsets: {self.pattern_offsets}")
        logger.info(f"Mask type: {mask_type}")
        logger.info(f"Sequence step size: {sequence_step_size}")
        
        self._load_all_subjects()
        
    def _load_all_subjects(self):
        """Load data from all subjects"""
        total_sequences = 0
        successful_subjects = 0
        failed_subjects = 0
        
        for subject in self.subjects:
            try:
                # Construct the correct file path for the subject (files are directly in data directory)
                file_path = self.data_root / f"{subject}_{self.session}_masked.h5"
                # Load dataset using enhanced loader with mask handling (SAME AS BILSTM)
                dataset = load_dataset_with_best_mask(str(file_path), self.mask_type)
                
                sequences_added = self._process_subject_data(dataset, subject)
                total_sequences += sequences_added
                successful_subjects += 1
                
                logger.info(f"✅ Loaded {subject}: {sequences_added} sequences")
                    
                # Log mask info (SAME AS BILSTM)
                mask_info = dataset.get_mask_info()
                logger.info(f"   Used mask: {mask_info.get('recommended_mask', 'unknown')}")
                
            except Exception as e:
                logger.error(f"❌ Failed to load {subject}: {str(e)}")
                failed_subjects += 1
                continue
        
        logger.info(f"Successfully loaded {successful_subjects} subjects")
        logger.info(f"Failed to load {failed_subjects} subjects")
        
    def _process_subject_data(self, dataset, subject):
        """Process data from a single subject using individual sample access"""
        # Access individual samples like BiLSTM does
        num_samples = min(len(dataset), self.max_samples_per_subject)
        sequence_length = len(self.pattern_offsets)
        
        # Create sequences using sliding window with step size
        sequences_added = 0
        max_sequences = min(self.max_samples_per_subject * 33, 33000)  # Cap at 33k sequences
        
        # Process individual samples like BiLSTM does
        for sample_idx in range(num_samples):
            if sequences_added >= max_sequences:
                break
                
            try:
                sample = dataset[sample_idx]
                
                # Extract PVI images and BP signal (same as BiLSTM)
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
                    if sequences_added >= max_sequences:
                        break
                        
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
                        
                        # Store the sequence and target
                        self.sequences.append(sequence)
                        self.targets.append(target_bp)
                        self.subject_ids.append(subject)
                        
                        sequences_added += 1
                        
            except Exception as e:
                logger.warning(f"Error processing sample {sample_idx} for {subject}: {e}")
                continue
        
        return sequences_added
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        target = self.targets[idx]
        subject_id = self.subject_ids[idx]
        
        return {
            'sequence': sequence,
            'target': target,
            'subject_id': subject_id
        }


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer architecture"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class ImprovedTransformerBPPredictor(nn.Module):
    """
    Enhanced Transformer-based Blood Pressure Predictor using VAE latent representations
    """
    
    def __init__(self, vae_model, latent_dim: int = 64, 
                 d_model: int = 256, nhead: int = 8, num_layers: int = 4,
                 dim_feedforward: int = 512, dropout: float = 0.1, 
                 pattern_offsets: List[int] = None, current_frame_bias: float = 3.0,
                 use_physiological_features: bool = True):
        super().__init__()
        
        self.vae_model = vae_model
        self.latent_dim = latent_dim
        self.d_model = d_model
        self.pattern_offsets = pattern_offsets or [-7, -6, -5, -4, -3, -2, -1, 0, 1, 2]
        self.current_frame_bias = current_frame_bias
        self.use_physiological_features = use_physiological_features
        
        # Freeze VAE
        for param in self.vae_model.parameters():
            param.requires_grad = False
        
        # Input projection from VAE latent to transformer dimension
        self.input_projection = nn.Linear(latent_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Additional feature extraction
        if self.use_physiological_features:
            self.feature_extractor = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, d_model // 4),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            feature_dim = d_model + d_model // 4
        else:
            feature_dim = d_model
        
        # Output prediction heads
        self.bp_predictor = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 50)  # 50 target frames
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model)
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize model weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)
    
    def forward(self, x_seq, return_attention=False):
        """
        Forward pass
        
        Args:
            x_seq: Input sequence [batch_size, seq_len, channels, height, width]
            return_attention: Whether to return attention weights
        """
        batch_size, seq_len = x_seq.shape[0], x_seq.shape[1]
        
        # Process each frame through VAE encoder
        latent_features = []
        for i in range(seq_len):
            frame = x_seq[:, i]  # [batch_size, channels, height, width]
            
            # Get VAE latent representation
            with torch.no_grad():
                mu, logvar = self.vae_model.encode(frame)
                latent = self.vae_model.reparameterize(mu, logvar)
            
            latent_features.append(latent)
        
        # Stack latent features
        latent_sequence = torch.stack(latent_features, dim=1)  # [batch_size, seq_len, latent_dim]
        
        # Project to transformer dimension
        transformer_input = self.input_projection(latent_sequence)  # [batch_size, seq_len, d_model]
        
        # Add positional encoding
        transformer_input = transformer_input.transpose(0, 1)  # [seq_len, batch_size, d_model]
        transformer_input = self.pos_encoder(transformer_input)
        transformer_input = transformer_input.transpose(0, 1)  # [batch_size, seq_len, d_model]
        
        # Apply layer normalization
        transformer_input = self.layer_norm(transformer_input)
        
        # Transformer encoding
        encoded_sequence = self.transformer_encoder(transformer_input)  # [batch_size, seq_len, d_model]
        
        # Focus on current frame (offset 0)
        current_frame_pos = self.pattern_offsets.index(0) if 0 in self.pattern_offsets else seq_len // 2
        current_features = encoded_sequence[:, current_frame_pos, :]  # [batch_size, d_model]
        
        # Optional: Add physiological features
        if self.use_physiological_features:
            # Extract additional features from the full sequence
            sequence_features = torch.mean(encoded_sequence, dim=1)  # Global average pooling
            additional_features = self.feature_extractor(sequence_features)
            
            # Concatenate current frame features with additional features
            combined_features = torch.cat([current_features, additional_features], dim=1)
        else:
            combined_features = current_features
        
        # Predict blood pressure waveform
        bp_prediction = self.bp_predictor(combined_features)  # [batch_size, 50]
        
        if return_attention:
            # Return dummy attention weights for compatibility
            attention_weights = torch.ones(batch_size, seq_len) / seq_len
            return bp_prediction, attention_weights.to(x_seq.device)
        
        return bp_prediction


class ImprovedBPLoss(nn.Module):
    """
    Enhanced loss function for blood pressure prediction with physiological constraints
    """
    
    def __init__(self, waveform_weight=0.4, systolic_weight=0.3, diastolic_weight=0.3, 
                 huber_delta=1.0, physiological_constraint=True, pulse_pressure_weight=0.1):
        super().__init__()
        self.waveform_weight = waveform_weight
        self.systolic_weight = systolic_weight
        self.diastolic_weight = diastolic_weight
        self.huber_delta = huber_delta
        self.physiological_constraint = physiological_constraint
        self.pulse_pressure_weight = pulse_pressure_weight
        
        self.huber_loss = nn.HuberLoss(delta=huber_delta)
        self.mse_loss = nn.MSELoss()
        
    def forward(self, predictions, targets):
        """
        Compute comprehensive BP loss
        """
        batch_size = predictions.size(0)
        
        # 1. Waveform reconstruction loss
        waveform_loss = self.huber_loss(predictions, targets)
        
        # 2. Extract systolic and diastolic values for each sample
        pred_systolic, pred_diastolic = self.extract_bp_values_improved(predictions)
        true_systolic, true_diastolic = self.extract_bp_values_improved(targets)
        
        # 3. Systolic and diastolic losses
        systolic_loss = self.mse_loss(pred_systolic, true_systolic)
        diastolic_loss = self.mse_loss(pred_diastolic, true_diastolic)
        
        # 4. Physiological constraints
        constraint_loss = 0.0
        if self.physiological_constraint:
            # Pulse pressure should be positive (systolic > diastolic)
            pred_pulse_pressure = pred_systolic - pred_diastolic
            true_pulse_pressure = true_systolic - true_diastolic
            
            # Encourage positive pulse pressure
            negative_pp_penalty = torch.relu(-pred_pulse_pressure).mean()
            
            # Pulse pressure similarity
            pp_loss = self.mse_loss(pred_pulse_pressure, true_pulse_pressure)
            
            constraint_loss = negative_pp_penalty + self.pulse_pressure_weight * pp_loss
        
        # 5. Combine all losses
        total_loss = (self.waveform_weight * waveform_loss + 
                     self.systolic_weight * systolic_loss + 
                     self.diastolic_weight * diastolic_loss + 
                     constraint_loss)
        
        return {
            'total_loss': total_loss,
            'waveform_loss': waveform_loss,
            'systolic_loss': systolic_loss,
            'diastolic_loss': diastolic_loss,
            'constraint_loss': constraint_loss
        }
    
    def extract_bp_values_improved(self, waveform):
        """
        Extract systolic and diastolic values from waveform using improved method
        """
        # Apply smoothing
        kernel_size = 5
        kernel = torch.ones(kernel_size, device=waveform.device) / kernel_size
        
        smoothed_waveforms = []
        for i in range(waveform.size(0)):
            # Pad and apply 1D convolution for smoothing
            # Convert 1D to 2D for padding, then back to 1D
            waveform_2d = waveform[i].unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len]
            padded_2d = F.pad(waveform_2d, (kernel_size//2, kernel_size//2), mode='reflect')
            padded = padded_2d.squeeze(0).squeeze(0)  # Back to 1D
            smoothed = F.conv1d(padded.unsqueeze(0).unsqueeze(0), 
                              kernel.unsqueeze(0).unsqueeze(0), 
                              padding=0).squeeze()
            smoothed_waveforms.append(smoothed)
        
        smoothed_waveform = torch.stack(smoothed_waveforms)
        
        # Find systolic (max) and diastolic (min) values
        systolic_values = torch.max(smoothed_waveform, dim=1)[0]
        diastolic_values = torch.min(smoothed_waveform, dim=1)[0]
        
        return systolic_values, diastolic_values


class VAE(nn.Module):
    """VAE model matching the trained checkpoint"""
    
    def __init__(self, latent_dim=128):
        super(VAE, self).__init__()
        
        # Encoder - matching the trained checkpoint architecture
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
        
        # Decoder - matching the trained checkpoint architecture
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
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu
    
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


class TransformerEvaluator:
    """Evaluator for Transformer-based blood pressure prediction"""
    
    def __init__(self, model, criterion, device):
        self.model = model
        self.criterion = criterion
        self.device = device
    
    def evaluate_model(self, val_loader, max_samples: int = 500):
        """Evaluate the model on validation data - FIXED to match BiLSTM exactly"""
        logger.info(f"🚀 Starting Transformer evaluation with max {max_samples} samples...")
        
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
                    
                sequences = batch['sequence'].to(self.device)
                targets = batch['target'].to(self.device)
                
                # Model prediction
                predictions = self.model(sequences)
                
                # Extract ground truth values using same method as training
                target_systolic, target_diastolic = self.criterion.extract_bp_values_improved(targets)
                
                # Store results
                for i in range(len(targets)):
                    if sample_count >= max_samples:
                        break
                        
                    pred_dict = {
                        'waveform': predictions[i].cpu().numpy()
                    }
                    
                    target_dict = {
                        'waveform': targets[i].cpu().numpy(),
                        'systolic': target_systolic[i].cpu().numpy() if isinstance(target_systolic, torch.Tensor) else target_systolic,
                        'diastolic': target_diastolic[i].cpu().numpy() if isinstance(target_diastolic, torch.Tensor) else target_diastolic
                    }
                    
                    # Extract BP values from predicted waveform
                    pred_sys, pred_dias = self._extract_bp_values_numpy(predictions[i].cpu().numpy())
                    pred_dict['systolic'] = pred_sys
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
        
        # Extract values
        pred_systolic = np.array([p['systolic'] for p in predictions])
        pred_diastolic = np.array([p['diastolic'] for p in predictions])
        pred_waveforms = np.array([p['waveform'] for p in predictions])
        
        true_systolic = np.array([t['systolic'] for t in targets])
        true_diastolic = np.array([t['diastolic'] for t in targets])
        true_waveforms = np.array([t['waveform'] for t in targets])
        
        # Calculate pulse pressure
        pred_pulse_pressure = pred_systolic - pred_diastolic
        true_pulse_pressure = true_systolic - true_diastolic
        
        # Systolic metrics
        sys_mae = mean_absolute_error(true_systolic, pred_systolic)
        sys_rmse = np.sqrt(mean_squared_error(true_systolic, pred_systolic))
        sys_r2 = r2_score(true_systolic, pred_systolic)
        sys_corr = np.corrcoef(true_systolic, pred_systolic)[0, 1]
        
        # Diastolic metrics
        dia_mae = mean_absolute_error(true_diastolic, pred_diastolic)
        dia_rmse = np.sqrt(mean_squared_error(true_diastolic, pred_diastolic))
        dia_r2 = r2_score(true_diastolic, pred_diastolic)
        dia_corr = np.corrcoef(true_diastolic, pred_diastolic)[0, 1]
        
        # Pulse pressure metrics
        pp_mae = mean_absolute_error(true_pulse_pressure, pred_pulse_pressure)
        pp_rmse = np.sqrt(mean_squared_error(true_pulse_pressure, pred_pulse_pressure))
        pp_r2 = r2_score(true_pulse_pressure, pred_pulse_pressure)
        pp_corr = np.corrcoef(true_pulse_pressure, pred_pulse_pressure)[0, 1]
        
        # Waveform metrics
        waveform_mae = mean_absolute_error(true_waveforms.flatten(), pred_waveforms.flatten())
        waveform_rmse = np.sqrt(mean_squared_error(true_waveforms.flatten(), pred_waveforms.flatten()))
        waveform_r2 = r2_score(true_waveforms.flatten(), pred_waveforms.flatten())
        
        return {
            'systolic': {
                'mae': sys_mae,
                'rmse': sys_rmse,
                'r2': sys_r2,
                'correlation': sys_corr,
                'mean_true': np.mean(true_systolic),
                'mean_pred': np.mean(pred_systolic),
                'std_true': np.std(true_systolic),
                'std_pred': np.std(pred_systolic)
            },
            'diastolic': {
                'mae': dia_mae,
                'rmse': dia_rmse,
                'r2': dia_r2,
                'correlation': dia_corr,
                'mean_true': np.mean(true_diastolic),
                'mean_pred': np.mean(pred_diastolic),
                'std_true': np.std(true_diastolic),
                'std_pred': np.std(pred_diastolic)
            },
            'pulse_pressure': {
                'mae': pp_mae,
                'rmse': pp_rmse,
                'r2': pp_r2,
                'correlation': pp_corr,
                'mean_true': np.mean(true_pulse_pressure),
                'mean_pred': np.mean(pred_pulse_pressure),
                'std_true': np.std(true_pulse_pressure),
                'std_pred': np.std(pred_pulse_pressure)
            },
            'waveform': {
                'mae': waveform_mae,
                'rmse': waveform_rmse,
                'r2': waveform_r2
            }
        }
    
    def save_results(self, metrics: Dict, output_dir: Path):
        """Save evaluation results to files"""
        
        # Save metrics to JSON for programmatic access
        json_path = output_dir / 'transformer_evaluation_metrics.json'
        
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            return obj
        
        json_compatible_metrics = convert_numpy_types(metrics)
        
        with open(json_path, 'w') as f:
            json.dump(json_compatible_metrics, f, indent=2)
        
        # Save human-readable summary
        summary_path = output_dir / 'evaluation_summary.txt'
        with open(summary_path, 'w') as f:
            f.write("Transformer Blood Pressure Prediction - Evaluation Summary\n")
            f.write("=" * 60 + "\n\n")
            
            for bp_type in ['systolic', 'diastolic', 'pulse_pressure']:
                f.write(f"{bp_type.upper().replace('_', ' ')} METRICS:\n")
                f.write(f"  MAE: {metrics[bp_type]['mae']:.3f}\n")
                f.write(f"  RMSE: {metrics[bp_type]['rmse']:.3f}\n")
                f.write(f"  R²: {metrics[bp_type]['r2']:.3f}\n")
                f.write(f"  Correlation: {metrics[bp_type]['correlation']:.3f}\n")
                f.write(f"  Mean True: {metrics[bp_type]['mean_true']:.3f}\n")
                f.write(f"  Mean Pred: {metrics[bp_type]['mean_pred']:.3f}\n")
                f.write("\n")
            
            f.write(f"WAVEFORM METRICS:\n")
            f.write(f"  MAE: {metrics['waveform']['mae']:.3f}\n")
            f.write(f"  RMSE: {metrics['waveform']['rmse']:.3f}\n")
            f.write(f"  R²: {metrics['waveform']['r2']:.3f}\n")
    
    def _extract_bp_values_numpy(self, waveform):
        """Extract systolic and diastolic values from numpy waveform"""
        # Simple smoothing
        if len(waveform) > 5:
            smoothed = np.convolve(waveform, np.ones(5)/5, mode='same')
        else:
            smoothed = waveform
        
        systolic = np.max(smoothed)
        diastolic = np.min(smoothed)
        
        return systolic, diastolic


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Enhanced Multi-Subject Transformer Training with 10-frame window",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--config", 
        type=str, 
        required=True,
        help="Path to Transformer configuration YAML file"
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
        default=0.1,
        help="Validation split ratio (default: 0.1 for 10%)"
    )
    parser.add_argument(
        "--test-ratio", 
        type=float,
        default=0.1,
        help="Test split ratio (default: 0.1 for 10%)"
    )
    
    return parser.parse_args()


def setup_environment(args):
    """Setup environment variables and paths"""
    if args.data_root:
        path_config = PathConfig(data_root=args.data_root)
    else:
        path_config = PathConfig()
    
    setup_paths(path_config)
    return path_config


def get_available_subjects(data_root: str) -> List[str]:
    """Get list of available subjects from data directory"""
    data_path = Path(data_root)
    subjects = []
    
    for file_path in data_path.glob("subject*_baseline_masked.h5"):
        subject_id = file_path.stem.split('_')[0]
        subjects.append(subject_id)
    
    return sorted(subjects)


def setup_device(device_arg: str):
    """Setup compute device"""
    if device_arg == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            logger.info(f"🚀 Using GPU: {torch.cuda.get_device_name()}")
            logger.info(f"📱 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
        else:
            logger.info("💻 Using CPU")
    else:
        device = torch.device(device_arg)
        logger.info(f"📱 Using specified device: {device}")
    
    return device


def load_config_file(config_path):
    """Load configuration from YAML file"""
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def create_experiment_directory(path_manager, experiment_name=None):
    """Create experiment directory with timestamp"""
    if experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"enhanced_transformer_multisubject_32subjects_mask10_{timestamp}"
    
    experiment_dir = Path("experiments") / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (experiment_dir / "checkpoints").mkdir(exist_ok=True)
    (experiment_dir / "plots").mkdir(exist_ok=True)
    (experiment_dir / "logs").mkdir(exist_ok=True)
    (experiment_dir / "results").mkdir(exist_ok=True)
    
    return experiment_dir


def save_system_info(experiment_dir):
    """Save system and environment information"""
    import platform
    import psutil
    
    system_info = {
        "timestamp": datetime.now().isoformat(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
        },
        "hardware": {
            "cpu_count": psutil.cpu_count(),
            "memory_gb": psutil.virtual_memory().total / (1024**3),
        },
        "packages": {
            "torch": torch.__version__,
            "numpy": np.__version__,
        }
    }
    
    # Add GPU information if available
    if torch.cuda.is_available():
        system_info["gpu"] = {
            "device_count": torch.cuda.device_count(),
            "current_device": torch.cuda.current_device(),
            "device_name": torch.cuda.get_device_name(),
            "memory_gb": torch.cuda.get_device_properties(0).total_memory / (1024**3),
        }
    
    # Get git information if available
    try:
        git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
        git_branch = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD']).decode('ascii').strip()
        system_info["git"] = {
            "commit_hash": git_hash,
            "branch": git_branch,
        }
    except:
        system_info["git"] = {"error": "Git information not available"}
    
    # Save to file
    system_info_path = experiment_dir / "system_info.json"
    with open(system_info_path, 'w') as f:
        json.dump(system_info, f, indent=2, default=str)
    
    logger.info(f"💾 System information saved to: {system_info_path}")


def main():
    """Main training function"""
    args = parse_arguments()
    
    # Setup logging
    level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(level=level)
    
    try:
        logger.info("🚀 Starting Enhanced Multi-Subject Transformer Training")
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
        vae = VAE(latent_dim=vae_config.get('latent_dim', 128))  # Default 128 to match config
        
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
        train_dataset = ImprovedTransformerDataset(
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
        val_dataset = ImprovedTransformerDataset(
            data_root=data_config.get('root_path', args.data_root),
            subjects=val_subjects,  # Only validation subjects
            mask_type=args.mask_type,
            pattern_offsets=pattern_offsets,
            max_samples_per_subject=data_config.get('max_samples_per_subject', 1000),
            sequence_step_size=data_config.get('sequence_step_size', 10),
            session=data_config.get('session', 'baseline'),
            use_augmentation=False,  # No augmentation for validation
            noise_level=data_config.get('noise_level', 0.005)
        )
        
        # Create TEST dataset from test subjects only
        logger.info("🧪 Creating TEST dataset from test subjects...")
        logger.info("🔒 CRITICAL: Test subjects NEVER seen during training")
        test_dataset = ImprovedTransformerDataset(
            data_root=data_config.get('root_path', args.data_root),
            subjects=test_subjects,  # Only test subjects
            mask_type=args.mask_type,
            pattern_offsets=pattern_offsets,
            max_samples_per_subject=data_config.get('max_samples_per_subject', 1000),
            sequence_step_size=data_config.get('sequence_step_size', 10),
            session=data_config.get('session', 'baseline'),
            use_augmentation=False,  # No augmentation for test
            noise_level=data_config.get('noise_level', 0.005)
        )
        
        logger.info("✅ Datasets created with NO LEAKAGE:")
        logger.info(f"   Train: {len(train_dataset)} samples from {len(train_subjects)} subjects")
        logger.info(f"   Val: {len(val_dataset)} samples from {len(val_subjects)} subjects")
        logger.info(f"   Test: {len(test_dataset)} samples from {len(test_subjects)} subjects")
        logger.info("🔒 GUARANTEE: Zero subject overlap between train/val/test")
        
        # Create data loaders
        hardware_config = config.get('hardware_config', {})
        train_loader = DataLoader(
            train_dataset, 
            batch_size=training_config.get('batch_size', 512), 
            shuffle=True, 
            num_workers=hardware_config.get('num_workers', 4),
            pin_memory=hardware_config.get('pin_memory', True)
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=training_config.get('batch_size', 512), 
            shuffle=False, 
            num_workers=hardware_config.get('num_workers', 4),
            pin_memory=hardware_config.get('pin_memory', True)
        )
        
        test_loader = DataLoader(
            test_dataset, 
            batch_size=training_config.get('batch_size', 512), 
            shuffle=False, 
            num_workers=hardware_config.get('num_workers', 4),
            pin_memory=hardware_config.get('pin_memory', True)
        )
        
        logger.info("✅ Data loaders created:")
        logger.info(f"   Train batches: {len(train_loader)}")
        logger.info(f"   Validation batches: {len(val_loader)}")
        logger.info(f"   Test batches: {len(test_loader)}")
        logger.info(f"   Batch size: {training_config.get('batch_size', 512)}")
        
        # Create Transformer model
        logger.info("Creating Transformer BP predictor...")
        transformer_config = model_config.get('transformer_config', {})
        attention_config = model_config.get('attention_config', {})
        
        model = ImprovedTransformerBPPredictor(
            vae_model=vae,
            latent_dim=vae_config.get('latent_dim', 128),
            d_model=transformer_config.get('d_model', 256),
            nhead=transformer_config.get('nhead', 8),
            num_layers=transformer_config.get('num_layers', 4),
            dim_feedforward=transformer_config.get('dim_feedforward', 512),
            dropout=transformer_config.get('dropout_rate', 0.1),
            pattern_offsets=pattern_offsets,
            current_frame_bias=attention_config.get('current_frame_bias', 3.0),
            use_physiological_features=attention_config.get('use_physiological_features', True)
        ).to(device)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        logger.info(f"✅ Transformer model created with {trainable_params:,} trainable parameters")
        
        # Attention bias info
        current_frame_pos = pattern_offsets.index(0) if 0 in pattern_offsets else len(pattern_offsets) // 2
        logger.info("🎯 Attention bias configuration:")
        logger.info(f"   Current frame bias (t+0): {attention_config.get('current_frame_bias', 3.0)}")
        logger.info(f"   Pattern offsets: {pattern_offsets}")
        logger.info(f"   Current frame position: {current_frame_pos}")
        logger.info("   Strong focus on t+0 for prediction target")
        
        # Create experiment directory
        experiment_dir = create_experiment_directory(path_manager)
        logger.info(f"✅ Experiment directory created: {experiment_dir}")
        
        # Save system info and configuration
        save_system_info(experiment_dir)
        
        # Save configuration
        config_save_path = experiment_dir / "config.yaml"
        with open(config_save_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        # Save experiment metadata
        experiment_metadata = {
            "experiment_name": experiment_dir.name,
            "timestamp": datetime.now().isoformat(),
            "model_type": "Transformer",
            "total_subjects": len(subjects),
            "train_subjects": len(train_subjects),
            "val_subjects": len(val_subjects),
            "test_subjects": len(test_subjects),
            "split_seed": args.split_seed,
            "mask_type": args.mask_type,
            "pattern_offsets": pattern_offsets,
            "model_parameters": {
                "total": total_params,
                "trainable": trainable_params,
                "d_model": transformer_config.get('d_model', 256),
                "nhead": transformer_config.get('nhead', 8),
                "num_layers": transformer_config.get('num_layers', 4)
            },
            "subjects": {
                "train": train_subjects,
                "val": val_subjects,
                "test": test_subjects
            }
        }
        
        with open(experiment_dir / "experiment_metadata.json", 'w') as f:
            json.dump(experiment_metadata, f, indent=2)
        
        # Setup training components
        criterion = ImprovedBPLoss(
            waveform_weight=loss_config.get('waveform_weight', 0.4),
            systolic_weight=loss_config.get('systolic_weight', 0.3),
            diastolic_weight=loss_config.get('diastolic_weight', 0.3),
            huber_delta=loss_config.get('huber_delta', 1.0),
            physiological_constraint=loss_config.get('physiological_constraint', True),
            pulse_pressure_weight=loss_config.get('pulse_pressure_weight', 0.1)
        )
        
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(training_config.get('learning_rate', 1e-4)),
            weight_decay=float(training_config.get('weight_decay', 1e-2))
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=training_config.get('factor', 0.7),
            patience=training_config.get('scheduler_patience', 8),
            verbose=True
        )
        
        # Setup evaluator
        evaluator = TransformerEvaluator(model, criterion, device)
        
        # Training loop
        logger.info("🎯 Starting training...")
        logger.info("=" * 70)
        
        num_epochs = training_config.get('num_epochs', 20)
        best_val_loss = float('inf')
        patience_counter = 0
        early_stopping_patience = training_config.get('early_stopping_patience', 15)
        
        training_history = {
            'train_losses': [],
            'val_losses': [],
            'learning_rates': []
        }
        
        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch+1}/{num_epochs}")
            logger.info("-" * 50)
            
            # Training phase
            model.train()
            train_loss = 0.0
            train_batches = 0
            
            for batch_idx, batch in enumerate(tqdm(train_loader, desc="Training")):
                sequences = batch['sequence'].to(device)
                targets = batch['target'].to(device)
                
                optimizer.zero_grad()
                predictions = model(sequences)
                loss_dict = criterion(predictions, targets)
                total_loss = loss_dict['total_loss']
                
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                train_loss += total_loss.item()
                train_batches += 1
                
                if batch_idx % 100 == 0:
                    logger.info(f"  Batch {batch_idx}/{len(train_loader)}, Loss: {total_loss.item():.4f}")
            
            avg_train_loss = train_loss / train_batches
            training_history['train_losses'].append(avg_train_loss)
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_batches = 0
            
            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Validation"):
                    sequences = batch['sequence'].to(device)
                    targets = batch['target'].to(device)
                    
                    predictions = model(sequences)
                    loss_dict = criterion(predictions, targets)
                    total_loss = loss_dict['total_loss']
                    
                    val_loss += total_loss.item()
                    val_batches += 1
            
            avg_val_loss = val_loss / val_batches
            training_history['val_losses'].append(avg_val_loss)
            training_history['learning_rates'].append(optimizer.param_groups[0]['lr'])
            
            # Learning rate scheduling
            scheduler.step(avg_val_loss)
            
            logger.info(f"  Train Loss: {avg_train_loss:.4f}")
            logger.info(f"  Val Loss: {avg_val_loss:.4f}")
            logger.info(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")
            
            # Save checkpoint if best
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                
                # Save best model
                checkpoint_path = experiment_dir / "checkpoints" / "transformer_best.pt"
                checkpoint_path.parent.mkdir(exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_val_loss': best_val_loss,
                    'training_history': training_history,
                    'pattern_offsets': data_config.get('pattern_offsets', [-7,-6,-5,-4,-3,-2,-1,0,1,2])
                }, checkpoint_path)
                logger.info(f"✅ New best model saved (val_loss: {avg_val_loss:.4f})")
            else:
                patience_counter += 1
                logger.info(f"⏳ No improvement for {patience_counter} epochs")
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                logger.info(f"🛑 Early stopping triggered after {patience_counter} epochs without improvement")
                break
            
            logger.info("")
        
        # Load best model for final evaluation
        best_checkpoint = experiment_dir / "checkpoints" / "transformer_best.pt"
        if best_checkpoint.exists():
            checkpoint = torch.load(best_checkpoint, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info("✅ Loaded best model for final evaluation")
        
        # Final evaluation on test set
        logger.info("🔍 Running final test evaluation...")
        evaluation_results = evaluator.evaluate_model(test_loader, args.max_eval_samples)
        
        # Calculate metrics from evaluation results
        test_metrics = evaluator.calculate_metrics(evaluation_results['predictions'], evaluation_results['targets'])
        
        # Save test results
        evaluator.save_results(test_metrics, experiment_dir / "results")
        
        # Print results
        logger.info("🎯 Test Results (100 samples):")
        logger.info("=" * 50)
        for bp_type in ['systolic', 'diastolic', 'pulse_pressure']:
            metrics = test_metrics[bp_type]
            logger.info(f"{bp_type.upper().replace('_', ' ')}:")
            logger.info(f"  MAE: {metrics['mae']:.3f}")
            logger.info(f"  RMSE: {metrics['rmse']:.3f}")
            logger.info(f"  R²: {metrics['r2']:.3f}")
            logger.info(f"  Correlation: {metrics['correlation']:.3f}")
        
        logger.info("=" * 70)
        logger.info("✅ LEAKAGE-FREE multi-subject Transformer setup completed successfully!")
        logger.info("🔒 GUARANTEED: No data leakage - subject-level splits enforced")
        logger.info("=" * 70)
        logger.info(f"📁 Results saved to: {experiment_dir}")
        logger.info(f"📊 Training subjects: {len(train_subjects)} subjects")
        logger.info(f"📊 Validation subjects: {len(val_subjects)} subjects")
        logger.info(f"🧪 TEST subjects: {len(test_subjects)} subjects (ISOLATED for evaluation)")
        logger.info(f"📊 Mask type used: {args.mask_type}")
        logger.info(f"🔢 Split seed used: {args.split_seed} (use same seed for reproducibility!)")
        logger.info(f"📋 Train subjects: {train_subjects}")
        logger.info(f"📋 Val subjects: {val_subjects}")
        logger.info(f"🧪 TEST subjects: {test_subjects}")
        logger.info("⚠️  CRITICAL: Test subjects saved in metadata for final evaluation!")
        
    except Exception as e:
        logger.error(f"❌ Training failed: {str(e)}")
        logger.error(f"💻 Error type: {type(e).__name__}")
        import traceback
        logger.error(f"🔍 Traceback:\n{traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    main()