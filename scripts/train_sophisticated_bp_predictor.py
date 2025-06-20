#!/usr/bin/env python3
"""
Sophisticated Blood Pressure Prediction Training Script

This script provides a comprehensive command-line interface for training and evaluating
the sophisticated BP prediction system with all the advanced features specified.

Features:
- Pre-trained VAE for feature extraction (frozen during training)
- Enhanced BiLSTM with multi-head self-attention
- Physiologically-informed loss functions
- Proper systolic/diastolic extraction
- 10-frame sliding window temporal patterns
- Comprehensive evaluation with clinical metrics
- Attention visualization
- Weights & Biases integration
- Complete CLI interface

Usage:
    python scripts/train_sophisticated_bp_predictor.py --data_root /path/to/data --vae_checkpoint /path/to/vae.pt
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
from typing import Dict, List, Tuple, Optional
import yaml
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Import utilities
from utils.data_utils import PviDataset, PviBatchServer, DataPathManager


class SophisticatedBPDataset(Dataset):
    """Dataset for sophisticated BP prediction using sliding window sequences"""
    
    def __init__(self, data_root: str, pattern_offsets: List[int], 
                 max_samples_per_subject: int = 50):
        self.data_root = data_root
        self.pattern_offsets = pattern_offsets
        self.max_samples_per_subject = max_samples_per_subject
        
        print(f"Loading data using PviDataset from: {data_root}")
        
        # Load data using the standard PviDataset
        self.pvi_dataset = PviDataset(data_root)
        
        self.sequences = []
        self.targets = []
        
        print(f"Loaded {len(self.pvi_dataset)} samples from PviDataset")
        
        # Debug: Print first sample structure
        if len(self.pvi_dataset) > 0:
            first_sample = self.pvi_dataset[0]
            print("First sample structure:")
            for key in first_sample.keys():
                if isinstance(first_sample[key], dict):
                    print(f"  {key}:")
                    for subkey in first_sample[key].keys():
                        if hasattr(first_sample[key][subkey], 'shape'):
                            print(f"    {subkey}: {first_sample[key][subkey].shape}")
                        else:
                            print(f"    {subkey}: {type(first_sample[key][subkey])}")
                else:
                    print(f"  {key}: {type(first_sample[key])}")
        
        # Process each sample from the dataset
        for sample_idx in range(min(len(self.pvi_dataset), self.max_samples_per_subject)):
            sample = self.pvi_dataset[sample_idx]
            
            # Extract PVI images and BP signal
            pvi_img = sample['pviHP']['img']  # [32, 32, num_frames]
            bp_signal = sample['bp']['signal']  # [num_frames] - each frame is 50 samples (1 second)
            
            print(f"Sample {sample_idx}: PVI shape {pvi_img.shape}, BP shape {bp_signal.shape}")
            
            # Get number of frames
            num_frames = pvi_img.shape[-1]
            
            # Determine valid central indices for sliding window
            min_offset = min(self.pattern_offsets)
            max_offset = max(self.pattern_offsets)
            valid_start = max(0, -min_offset)
            valid_end = min(num_frames, num_frames - max_offset)
            
            # Sample central indices (every 5 frames to avoid too much overlap)
            step_size = max(1, len(self.pattern_offsets) // 2)
            central_indices = list(range(valid_start, valid_end, step_size))
            
            print(f"Sample {sample_idx}: Creating sequences for {len(central_indices)} central indices")
            
            for central_idx in central_indices:
                # Create sequence using pattern offsets
                seq_frames = []
                valid_sequence = True
                
                for offset in self.pattern_offsets:
                    frame_idx = central_idx + offset
                    if 0 <= frame_idx < num_frames:
                        # Extract and normalize frame - match VAE training approach
                        frame = pvi_img[:, :, frame_idx]  # [32, 32]
                        frame = torch.tensor(frame, dtype=torch.float32)
                        frame = torch.nan_to_num(frame, nan=0.0)  # Only handle NaN like VAE training
                        
                        # Add channel dimension for VAE input: [1, 32, 32]
                        frame = frame.unsqueeze(0)
                        seq_frames.append(frame)
                    else:
                        valid_sequence = False
                        break
                
                if valid_sequence:
                    sequence = torch.stack(seq_frames)  # [seq_len, 1, 32, 32]
                    
                    # Use the BP signal for the CURRENT FRAME (central_idx) as target
                    # This corresponds to the frame at offset=0 in our pattern
                    if bp_signal.dim() == 1:
                        # If BP signal is 1D, it's already the signal for this central frame
                        target_bp = bp_signal
                    else:
                        # If BP signal is 2D [num_frames, signal_length], select the current frame
                        target_bp = bp_signal[central_idx] if central_idx < bp_signal.shape[0] else bp_signal[0]
                    
                    # If BP signal has multiple dimensions, we might need to slice
                    if target_bp.dim() > 1:
                        print(f"Warning: BP signal has shape {target_bp.shape}, using first dimension")
                        target_bp = target_bp[0] if target_bp.shape[0] == 1 else target_bp.flatten()
                    
                    # Ensure target is exactly 50 samples
                    current_length = target_bp.shape[0]
                    if current_length != 50:
                        print(f"Resampling BP signal from {current_length} to 50 samples")
                        if current_length > 50:
                            # Downsample by taking evenly spaced samples
                            indices = torch.linspace(0, current_length - 1, 50).long()
                            target_bp = target_bp[indices]
                        else:
                            # Upsample using interpolation
                            target_bp = torch.nn.functional.interpolate(
                                target_bp.unsqueeze(0).unsqueeze(0), 
                                size=50, 
                                mode='linear', 
                                align_corners=False
                            ).squeeze()
                    
                    # Keep BP signal in raw mmHg values (no normalization like data_utils.py)
                    target_bp = target_bp.float()
                    
                    self.sequences.append(sequence)
                    self.targets.append(target_bp)  # BP signal for current frame (t)
        
        print(f"Created {len(self.sequences)} sequences from sliding windows")
        if len(self.sequences) > 0:
            print(f"Sequence shape: {self.sequences[0].shape}")
            print(f"Target shape: {self.targets[0].shape}")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return {
            'sequences': self.sequences[idx],
            'targets': self.targets[idx]
        }


# Enhanced model definitions (inline for completeness)
class VAE(nn.Module):
    """VAE model matching the trained checkpoint"""
    
    def __init__(self, latent_dim=64):
        super(VAE, self).__init__()
        
        # Encoder - increased filter counts and added batch normalization
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        
        # Fully connected layers for mean and variance
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
        
        # Add dropout for regularization
        self.dropout = nn.Dropout(0.2)
        
    def encode(self, x):
        # Forward pass through encoder
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)  # Apply dropout
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Get mean and log variance
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        # Reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def decode(self, z):
        # Forward pass through decoder
        x = self.fc_decoder(z)
        x = F.relu(self.bn_dec(x))
        x = x.view(x.size(0), 512, 2, 2)
        
        x = F.relu(self.bn_dec1(self.deconv1(x)))
        x = F.relu(self.bn_dec2(self.deconv2(x)))
        x = self.dropout(x)  # Apply dropout
        x = F.relu(self.bn_dec3(self.deconv3(x)))
        x = torch.sigmoid(self.deconv4(x))
        
        return x
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention for temporal modeling"""
    
    def __init__(self, hidden_dim: int, num_heads: int = 8, dropout: float = 0.1, 
                 current_frame_bias: float = 2.0):
        super(MultiHeadSelfAttention, self).__init__()
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
        
        # Add positional bias to encourage attention on current frame (t=0)
        if pattern_offsets is not None and self.current_frame_bias > 0:
            # Find the index of the current frame (offset=0)
            try:
                current_frame_idx = pattern_offsets.index(0)
                
                # Create bias matrix - boost attention TO the current frame
                bias_matrix = torch.zeros_like(scores[0, 0])  # [seq_len, seq_len]
                bias_matrix[:, current_frame_idx] += self.current_frame_bias  # Boost attention TO current frame
                bias_matrix[current_frame_idx, :] += self.current_frame_bias * 0.5  # Boost attention FROM current frame
                
                # Apply bias to all heads and batches
                bias_matrix = bias_matrix.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
                scores = scores + bias_matrix.to(scores.device)
                
            except ValueError:
                # Current frame (offset=0) not in pattern_offsets, skip bias
                pass
        
        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        attended = torch.matmul(attention_weights, V)
        attended = attended.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_dim)
        output = self.output_proj(attended)
        
        return self.layer_norm(x + output), attention_weights.mean(dim=1)


class SophisticatedBPPredictor(nn.Module):
    """Sophisticated BP prediction model"""
    
    def __init__(self, vae_model: VAE, latent_dim: int = 64, 
                 hidden_dim: int = 256, num_layers: int = 3, num_heads: int = 8,
                 dropout: float = 0.3, use_attention: bool = True, 
                 pattern_offsets: List[int] = None, current_frame_bias: float = 2.0):
        super(SophisticatedBPPredictor, self).__init__()
        
        self.vae = vae_model
        self.use_attention = use_attention
        self.pattern_offsets = pattern_offsets or [-7, -6, -5, -4, -3, -2, -1, 0, 1, 2]
        
        # Freeze VAE parameters
        for param in self.vae.parameters():
            param.requires_grad = False
        
        # Input projection
        self.input_projection = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5)
        )
        
        # Bidirectional LSTM (optional for single frame)
        self.use_temporal_processing = len(self.pattern_offsets) > 1
        
        if self.use_temporal_processing:
            self.lstm = nn.LSTM(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=True
            )
            lstm_output_dim = hidden_dim * 2
        else:
            self.lstm = None
            lstm_output_dim = hidden_dim
        
        # Self-attention (only for multi-frame)
        if use_attention and self.use_temporal_processing:
            self.attention = MultiHeadSelfAttention(
                hidden_dim=lstm_output_dim,
                num_heads=num_heads,
                dropout=dropout,
                current_frame_bias=current_frame_bias
            )
        else:
            self.attention = None
        
        # Temporal convolution (only for multi-frame)
        if self.use_temporal_processing:
            self.temporal_conv = nn.Sequential(
                nn.Conv1d(lstm_output_dim, 128, kernel_size=3, padding=1),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            )
            final_feature_dim = 128
        else:
            self.temporal_conv = None
            final_feature_dim = hidden_dim
        
        # Output layers
        self.output_layers = nn.Sequential(
            nn.Linear(final_feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
        )
        
        # Prediction heads
        self.waveform_head = nn.Linear(hidden_dim // 2, 50)
        self.systolic_head = nn.Linear(hidden_dim // 2, 1)
        self.diastolic_head = nn.Linear(hidden_dim // 2, 1)
    
    def forward(self, x_seq, return_attention=False):
        batch_size, seq_len = x_seq.shape[:2]
        
        # Encode frames with frozen VAE
        latent_seq = []
        with torch.no_grad():
            for t in range(seq_len):
                mu_t, _ = self.vae.encode(x_seq[:, t])
                latent_seq.append(mu_t)
        
        latent_seq = torch.stack(latent_seq, dim=1)  # [batch_size, seq_len, latent_dim]
        
        # Project and process with LSTM
        projected_seq = self.input_projection(latent_seq)
        
        # Handle single frame case differently
        if not self.use_temporal_processing:
            # For single frame, skip LSTM and attention, use direct processing
            features = projected_seq.squeeze(1)  # Remove sequence dimension [batch_size, hidden_dim]
        else:
            # Multi-frame processing with LSTM and attention
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
        
        outputs = {
            'waveform': self.waveform_head(final_features),
            'systolic': self.systolic_head(final_features),
            'diastolic': self.diastolic_head(final_features)
        }
        
        if return_attention and self.use_temporal_processing and self.attention is not None:
            outputs['attention_weights'] = attention_weights
        
        return outputs


def extract_physiological_bp_values(waveform):
    """Extract physiologically correct systolic and diastolic values"""
    batch_size, signal_length = waveform.shape
    device = waveform.device
    
    systolic_values = torch.zeros(batch_size, device=device)
    diastolic_values = torch.zeros(batch_size, device=device)
    
    for i in range(batch_size):
        signal = waveform[i]
        
        # Systolic (maximum)
        sys_val, sys_idx = torch.max(signal, dim=0)
        
        # Diastolic (minimum after systolic peak)
        search_start = max(sys_idx.item() + 1, signal_length // 2)
        
        if search_start < signal_length:
            post_systolic = signal[search_start:]
            dias_val = torch.min(post_systolic)
        else:
            dias_val = torch.min(signal)
        
        systolic_values[i] = sys_val
        diastolic_values[i] = dias_val
    
    return systolic_values, diastolic_values


class ComprehensiveBPLoss(nn.Module):
    """Comprehensive BP loss with multiple components"""
    
    def __init__(self, waveform_weight=0.6, systolic_weight=0.2, diastolic_weight=0.2, loss_type='composite'):
        super(ComprehensiveBPLoss, self).__init__()
        self.waveform_weight = waveform_weight
        self.systolic_weight = systolic_weight
        self.diastolic_weight = diastolic_weight
        self.loss_type = loss_type
        
        # Normalize weights
        total = waveform_weight + systolic_weight + diastolic_weight
        self.waveform_weight /= total
        self.systolic_weight /= total
        self.diastolic_weight /= total
    
    def forward(self, predictions, targets):
        pred_waveform = predictions['waveform']
        
        # Extract ground truth values
        target_systolic, target_diastolic = extract_physiological_bp_values(targets)
        
        # Waveform loss
        waveform_loss = nn.functional.mse_loss(pred_waveform, targets)
        
        # Component losses
        if 'systolic' in predictions:
            systolic_loss = nn.functional.mse_loss(predictions['systolic'].squeeze(), target_systolic)
        else:
            pred_sys, _ = extract_physiological_bp_values(pred_waveform)
            systolic_loss = nn.functional.mse_loss(pred_sys, target_systolic)
        
        if 'diastolic' in predictions:
            diastolic_loss = nn.functional.mse_loss(predictions['diastolic'].squeeze(), target_diastolic)
        else:
            _, pred_dias = extract_physiological_bp_values(pred_waveform)
            diastolic_loss = nn.functional.mse_loss(pred_dias, target_diastolic)
        
        # Compute total loss
        if self.loss_type == 'mse':
            total_loss = waveform_loss
        elif self.loss_type == 'systolic_distance':
            total_loss = systolic_loss
        elif self.loss_type == 'diastolic_distance':
            total_loss = diastolic_loss
        elif self.loss_type == 'composite':
            total_loss = (self.waveform_weight * waveform_loss + 
                         self.systolic_weight * systolic_loss + 
                         self.diastolic_weight * diastolic_loss)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        return {
            'total_loss': total_loss,
            'waveform_loss': waveform_loss,
            'systolic_loss': systolic_loss,
            'diastolic_loss': diastolic_loss
        }


def create_visualizations(predictions, targets, attention_weights, output_dir, pattern_offsets):
    """Create comprehensive visualizations"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Prediction examples
    plt.figure(figsize=(15, 10))
    for i in range(min(6, len(predictions))):
        plt.subplot(2, 3, i+1)
        plt.plot(targets[i], label='Ground Truth', linewidth=2)
        plt.plot(predictions[i], label='Prediction', linewidth=2)
        plt.title(f'Example {i+1}')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'prediction_examples.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Attention heatmap (only for multi-frame)
    if attention_weights is not None and len(attention_weights) > 0 and len(pattern_offsets) > 1:
        plt.figure(figsize=(10, 8))
        avg_attention = np.mean(attention_weights[:5], axis=0)
        frame_labels = [f't{offset:+d}' if offset != 0 else 't' for offset in pattern_offsets]
        
        import seaborn as sns
        sns.heatmap(avg_attention, xticklabels=frame_labels, yticklabels=frame_labels,
                   cmap='Blues', annot=True, fmt='.3f')
        plt.title('Attention Patterns')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'attention_heatmap.png'), dpi=300, bbox_inches='tight')
        plt.close()
    else:
        print("Skipping attention visualization (single frame or no attention weights)")


def load_config_file(config_path):
    """Load YAML configuration file"""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def apply_config_to_args(args, config):
    """Apply config values to args, command line args take priority"""
    # Data configuration
    data_config = config.get('data_config', {})
    if not hasattr(args, 'data_root') or args.data_root is None:
        if 'data_file' in data_config:
            args.data_root = os.path.join(data_config.get('root_path', ''), data_config['data_file'])
        else:
            args.data_root = data_config.get('root_path', '/home/lucas_takanori/phd/data/subject001_baseline_masked.h5')
    
    # Model configuration
    model_config = config.get('model_config', {})
    vae_config = model_config.get('vae_config', {})
    bilstm_config = model_config.get('bilstm_config', {})
    attention_config = model_config.get('attention_config', {})
    
    if not hasattr(args, 'vae_checkpoint') or args.vae_checkpoint is None:
        args.vae_checkpoint = vae_config.get('vae_checkpoint_path')
    if args.latent_dim == 256:  # Default value
        args.latent_dim = vae_config.get('latent_dim', args.latent_dim)
    if args.hidden_dim == 256:  # Default value
        args.hidden_dim = bilstm_config.get('hidden_dim', args.hidden_dim)
    if args.num_layers == 3:  # Default value
        args.num_layers = bilstm_config.get('num_layers', args.num_layers)
    if args.num_attention_heads == 8:  # Default value
        args.num_attention_heads = attention_config.get('num_attention_heads', args.num_attention_heads)
    if not hasattr(args, 'current_frame_bias') or args.current_frame_bias == 2.0:  # Default value
        args.current_frame_bias = attention_config.get('current_frame_bias', 2.0)
    
    # Training configuration
    training_config = config.get('training_config', {})
    if args.epochs == 25:  # Default value
        args.epochs = training_config.get('num_epochs', args.epochs)
    if args.batch_size == 8:  # Default value
        args.batch_size = training_config.get('batch_size', args.batch_size)
    if args.learning_rate == 1e-4:  # Default value
        args.learning_rate = training_config.get('learning_rate', args.learning_rate)
    
    # Loss configuration
    loss_config = config.get('loss_config', {})
    if args.loss_type == 'composite':  # Default value
        args.loss_type = loss_config.get('loss_type', args.loss_type)
    
    # Output configuration
    output_config = config.get('output_config', {})
    if args.output_dir == './sophisticated_bp_experiments':  # Default value
        args.output_dir = output_config.get('output_dir', args.output_dir)
    
    # Logging configuration
    logging_config = config.get('logging_config', {})
    visualization = logging_config.get('visualization', {})
    if not hasattr(args, 'visualize_attention_set'):
        args.visualize_attention = visualization.get('visualize_attention', args.visualize_attention)
    
    # Update pattern offsets if specified in config
    data_config = config.get('data_config', {})
    if 'pattern_offsets' in data_config:
        args.pattern_offsets = data_config['pattern_offsets']
    
    return args

def main():
    parser = argparse.ArgumentParser(description='Sophisticated BP Predictor Training')
    
    # Config file
    parser.add_argument('--config', type=str, default=None,
                       help='Path to YAML configuration file')
    
    # Data arguments
    parser.add_argument('--data_root', type=str, default=None,
                       help='Root directory of PVI dataset')
    parser.add_argument('--vae_checkpoint', type=str, default=None,
                       help='Path to pre-trained VAE checkpoint')
    
    # Model hyperparameters
    parser.add_argument('--latent_dim', type=int, default=256,
                       help='VAE latent dimension')
    parser.add_argument('--hidden_dim', type=int, default=256,
                       help='LSTM hidden dimension')
    parser.add_argument('--num_layers', type=int, default=3,
                       help='Number of LSTM layers')
    parser.add_argument('--num_attention_heads', type=int, default=8,
                       help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.3,
                       help='Dropout rate')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=25,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--loss_type', type=str, default='composite',
                       choices=['mse', 'systolic_distance', 'diastolic_distance', 'composite'],
                       help='Loss function type')
    
    # Feature toggles
    parser.add_argument('--use_attention', action='store_true', default=True,
                       help='Use self-attention mechanism')
    parser.add_argument('--visualize_attention', action='store_true', default=True,
                       help='Visualize attention patterns')
    parser.add_argument('--current_frame_bias', type=float, default=2.0,
                       help='Bias to encourage attention on current frame (t=0)')
    
    # Output and experiment tracking
    parser.add_argument('--output_dir', type=str, default='./sophisticated_bp_experiments',
                       help='Output directory for results')
    parser.add_argument('--use_wandb', action='store_true', default=False,
                       help='Use Weights & Biases logging')
    parser.add_argument('--wandb_project', type=str, default='sophisticated-bp-prediction',
                       help='W&B project name')
    parser.add_argument('--wandb_mode', type=str, default='offline',
                       choices=['online', 'offline', 'disabled'],
                       help='W&B logging mode')
    
    # Temporal pattern
    parser.add_argument('--pattern_offsets', type=int, nargs='+', 
                       default=[-7, -6, -5, -4, -3, -2, -1, 0, 1, 2],
                       help='Frame offsets for sliding window pattern')
    
    args = parser.parse_args()
    
    # Load configuration file if provided
    if args.config:
        if not os.path.exists(args.config):
            raise FileNotFoundError(f"Config file not found: {args.config}")
        print(f"Loading configuration from: {args.config}")
        config = load_config_file(args.config)
        args = apply_config_to_args(args, config)
        print("Configuration loaded successfully")
    
    # Validate required arguments
    if args.data_root is None:
        raise ValueError("--data_root is required (either via command line or config file)")
    if args.vae_checkpoint is None:
        raise ValueError("--vae_checkpoint is required (either via command line or config file)")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize W&B if requested
    if args.use_wandb:
        import wandb
        wandb.init(
            project=args.wandb_project,
            config=vars(args),
            mode=args.wandb_mode
        )
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    checkpoints_dir = os.path.join(args.output_dir, 'checkpoints')
    results_dir = os.path.join(args.output_dir, 'results')
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    print("="*60)
    print("SOPHISTICATED BP PREDICTOR TRAINING")
    print("="*60)
    print(f"Data root: {args.data_root}")
    print(f"VAE checkpoint: {args.vae_checkpoint}")
    print(f"Pattern offsets: {args.pattern_offsets}")
    print(f"Loss type: {args.loss_type}")
    print(f"Use attention: {args.use_attention}")
    print(f"Output directory: {args.output_dir}")
    print("="*60)
    
    # Load pre-trained VAE
    print("\nLoading pre-trained VAE...")
    vae = VAE(latent_dim=args.latent_dim)
    
    if os.path.exists(args.vae_checkpoint):
        checkpoint = torch.load(args.vae_checkpoint, map_location=device)
        if 'model_state_dict' in checkpoint:
            vae.load_state_dict(checkpoint['model_state_dict'])
        else:
            vae.load_state_dict(checkpoint)
        print(f"Loaded VAE from {args.vae_checkpoint}")
    else:
        print(f"Warning: VAE checkpoint not found at {args.vae_checkpoint}")
    
    vae.to(device)
    vae.eval()  # Freeze for feature extraction
    
    # Create datasets
    print("\nCreating datasets...")
    
    # For this example, we'll create train/val split from the same data
    # In practice, you'd want separate train/val datasets
    dataset = SophisticatedBPDataset(
        data_root=args.data_root,
        pattern_offsets=args.pattern_offsets,
        max_samples_per_subject=20
    )
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=2,
        pin_memory=True
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create model
    print("\nCreating sophisticated BP predictor...")
    model = SophisticatedBPPredictor(
        vae_model=vae,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_attention_heads,
        dropout=args.dropout,
        use_attention=args.use_attention,
        pattern_offsets=args.pattern_offsets,
        current_frame_bias=args.current_frame_bias
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Initialize loss and optimizer
    criterion = ComprehensiveBPLoss(loss_type=args.loss_type)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=1e-5,
        betas=(0.9, 0.999)
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6
    )
    
    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    
    best_val_loss = float('inf')
    training_history = {
        'train_loss': [],
        'val_loss': [],
        'val_mae': [],
        'learning_rates': []
    }
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        
        # Training
        model.train()
        train_loss = 0.0
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
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += total_loss.item()
                num_train_batches += 1
                
                train_pbar.set_postfix({
                    'Loss': f"{total_loss.item():.4f}",
                    'LR': f"{optimizer.param_groups[0]['lr']:.2e}"
                })
                
            except Exception as e:
                print(f"Training error: {e}")
                continue
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_predictions = []
        val_targets = []
        val_attention_weights = []
        
        num_val_batches = 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc="Validation")
            for batch in val_pbar:
                try:
                    sequences = batch['sequences'].to(device)
                    targets = batch['targets'].to(device)
                    
                    outputs = model(sequences, return_attention=args.visualize_attention)
                    loss_dict = criterion(outputs, targets)
                    
                    val_loss += loss_dict['total_loss'].item()
                    num_val_batches += 1
                    
                    # Store raw BP values for metrics (no denormalization needed)
                    pred_raw = outputs['waveform'].cpu()
                    target_raw = targets.cpu()
                    
                    val_predictions.append(pred_raw.numpy())
                    val_targets.append(target_raw.numpy())
                    
                    if args.visualize_attention and 'attention_weights' in outputs:
                        val_attention_weights.append(outputs['attention_weights'].cpu().numpy())
                    
                except Exception as e:
                    print(f"Validation error: {e}")
                    continue
        
        # Update scheduler
        scheduler.step()
        
        # Calculate epoch metrics
        if num_train_batches > 0:
            train_loss /= num_train_batches
        if num_val_batches > 0:
            val_loss /= num_val_batches
        
        # Calculate MAE
        val_mae = 0.0
        if val_predictions and val_targets:
            all_val_pred = np.concatenate(val_predictions, axis=0)
            all_val_target = np.concatenate(val_targets, axis=0)
            val_mae = np.mean(np.abs(all_val_pred - all_val_target))
        
        # Store history
        training_history['train_loss'].append(train_loss)
        training_history['val_loss'].append(val_loss)
        training_history['val_mae'].append(val_mae)
        training_history['learning_rates'].append(optimizer.param_groups[0]['lr'])
        
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val MAE: {val_mae:.2f}")
        
        # Log to W&B
        if args.use_wandb:
            wandb.log({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_mae': val_mae,
                'learning_rate': optimizer.param_groups[0]['lr']
            })
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'args': vars(args),
                'training_history': training_history
            }
            
            torch.save(checkpoint, os.path.join(checkpoints_dir, 'best_model.pt'))
            print(f"New best model saved! Val Loss: {best_val_loss:.4f}")
            
            # Create visualizations for best model
            if val_predictions and val_targets:
                create_visualizations(
                    all_val_pred[:20], all_val_target[:20],
                    np.concatenate(val_attention_weights, axis=0)[:20] if val_attention_weights else None,
                    results_dir, args.pattern_offsets
                )
    
    # Plot training curves
    epochs = range(1, len(training_history['train_loss']) + 1)
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(epochs, training_history['train_loss'], label='Train')
    plt.plot(epochs, training_history['val_loss'], label='Validation')
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.plot(epochs, training_history['val_mae'])
    plt.title('Validation MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE (mmHg)')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    plt.plot(epochs, training_history['learning_rates'])
    plt.title('Learning Rate')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Final evaluation
    print("\n" + "="*60)
    print("TRAINING COMPLETED")
    print("="*60)
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Final validation MAE: {training_history['val_mae'][-1]:.2f} mmHg")
    print(f"Results saved to: {args.output_dir}")
    
    if args.use_wandb:
        wandb.finish()


if __name__ == '__main__':
    main() 