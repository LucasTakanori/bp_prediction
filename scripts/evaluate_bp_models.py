#!/usr/bin/env python3
"""
Comprehensive Blood Pressure Model Evaluation Script for Improved BP Predictor

Generates medical-grade evaluation plots with:
- Correlation plots (predicted vs true)
- Bland-Altman plots (difference vs mean)
- Error distribution histograms
- Clinical accuracy metrics
- Statistical significance tests

For DBP, SBP, and full waveform predictions from the ImprovedBPPredictor.
"""

import argparse
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
from scipy import stats
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings
import yaml
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import utilities
from utils.data_utils import PviDataset, DataPathManager

# Import model classes from the training script
try:
    from scripts.train_improved_bp_predictor import (
        VAE, 
        ImprovedBPPredictor, 
        ImprovedBPDataset,
        ImprovedMultiHeadAttention,
        ImprovedBPLoss,
        extract_bp_values_numpy
    )
    print("Successfully imported model classes from train_improved_bp_predictor.py")
except ImportError as e:
    print(f"Warning: Could not import from train_improved_bp_predictor.py: {e}")
    print("Falling back to local definitions...")
    
    # Fallback definitions (keeping the original classes as backup)
    import torch.nn as nn
    import torch.nn.functional as F
    
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
            self.pattern_offsets = pattern_offsets or [-4, -3, -2, -1, 0, 1, 2]
            
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
        
        def _initialize_bp_heads(self):
            """Initialize BP prediction heads with physiologically reasonable values"""
            # Initialize systolic head to predict around 120 mmHg
            with torch.no_grad():
                self.systolic_head[-1].bias.fill_(120.0)
                self.systolic_head[-1].weight.data.normal_(0, 0.1)
                
            # Initialize diastolic head to predict around 80 mmHg
            with torch.no_grad():
                self.diastolic_head[-1].bias.fill_(80.0)
                self.diastolic_head[-1].weight.data.normal_(0, 0.1)
        
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

    class ImprovedBPDataset(torch.utils.data.Dataset):
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

    def extract_bp_values_numpy(waveform):
        """Extract systolic and diastolic values from BP waveform (numpy version)"""
        if isinstance(waveform, torch.Tensor):
            waveform = waveform.detach().cpu().numpy()
        
        if len(waveform.shape) > 1:
            waveform = waveform.flatten()
        
        # Systolic (maximum)
        systolic = np.max(waveform)
        sys_idx = np.argmax(waveform)
        
        # Diastolic (minimum after systolic peak)
        search_start = max(sys_idx + 1, len(waveform) // 2)
        
        if search_start < len(waveform):
            post_systolic = waveform[search_start:]
            diastolic = np.min(post_systolic)
        else:
            diastolic = np.min(waveform)
        
        return systolic, diastolic


def extract_bp_values_improved(waveform):
    """Extract systolic and diastolic values with improved physiological accuracy"""
    if isinstance(waveform, torch.Tensor):
        if waveform.dim() == 1:
            # Single waveform
            signal = waveform.detach().cpu().numpy()
            sys_val = np.max(signal)
            sys_idx = np.argmax(signal)
            
            # Diastolic: Look in last 1/3 of signal or after systolic peak
            search_start = max(sys_idx + 1, int(len(signal) * 0.6))
            search_end = min(len(signal), int(len(signal) * 0.95))
            
            if search_start < search_end:
                diastolic_window = signal[search_start:search_end]
                dias_val = np.min(diastolic_window)
            else:
                dias_val = np.min(signal)
            
            # Ensure physiological constraint
            if dias_val >= sys_val:
                dias_val = sys_val - 10.0
                
            return sys_val, dias_val
        else:
            # Batch of waveforms
            batch_size = waveform.shape[0]
            systolic_values = []
            diastolic_values = []
            
            for i in range(batch_size):
                sys_val, dias_val = extract_bp_values_improved(waveform[i])
                systolic_values.append(sys_val)
                diastolic_values.append(dias_val)
            
            return np.array(systolic_values), np.array(diastolic_values)
    else:
        # Numpy array
        if waveform.ndim == 1:
            signal = waveform
            sys_val = np.max(signal)
            sys_idx = np.argmax(signal)
            
            search_start = max(sys_idx + 1, int(len(signal) * 0.6))
            search_end = min(len(signal), int(len(signal) * 0.95))
            
            if search_start < search_end:
                diastolic_window = signal[search_start:search_end]
                dias_val = np.min(diastolic_window)
            else:
                dias_val = np.min(signal)
            
            if dias_val >= sys_val:
                dias_val = sys_val - 10.0
                
            return sys_val, dias_val
        else:
            batch_size = waveform.shape[0]
            systolic_values = []
            diastolic_values = []
            
            for i in range(batch_size):
                sys_val, dias_val = extract_bp_values_improved(waveform[i])
                systolic_values.append(sys_val)
                diastolic_values.append(dias_val)
            
            return np.array(systolic_values), np.array(diastolic_values)


def load_config_file(config_path):
    """Load YAML configuration file"""
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    return None


class BPModelEvaluator:
    """Comprehensive BP model evaluation with medical-grade metrics for ImprovedBPPredictor"""
    
    def __init__(self, model_path: str, config_path: Optional[str] = None):
        self.model_path = model_path
        self.config_path = config_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load external config if provided
        self.external_config = load_config_file(config_path) if config_path else None
        
        # Load model and configuration
        self.model, self.model_config = self._load_model()
        
    def _load_model(self):
        """Load trained ImprovedBPPredictor model and configuration"""
        print(f"Loading ImprovedBPPredictor model from: {self.model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        
        # Extract configuration from checkpoint or external config
        if self.external_config:
            # Use external configuration file
            config = self.external_config
            data_config = config.get('data_config', {})
            model_config = config.get('model_config', {})
            vae_config = model_config.get('vae_config', {})
            bilstm_config = model_config.get('bilstm_config', {})
            attention_config = model_config.get('attention_config', {})
            
            extracted_config = {
                'latent_dim': vae_config.get('latent_dim', 64),
                'hidden_dim': bilstm_config.get('hidden_dim', 128),
                'num_layers': bilstm_config.get('num_layers', 2),
                'num_attention_heads': attention_config.get('num_attention_heads', 4),
                'dropout': bilstm_config.get('dropout_rate', 0.4),
                'use_attention': attention_config.get('use_attention', True),
                'pattern_offsets': data_config.get('pattern_offsets', [-4, -3, -2, -1, 0, 1, 2]),
                'current_frame_bias': attention_config.get('current_frame_bias', 1.5),
                'use_physiological_features': attention_config.get('use_physiological_features', True),
                'vae_checkpoint': vae_config.get('vae_checkpoint_path', None)
            }
            print("Using external configuration file")
        elif 'config' in checkpoint:
            # Use configuration from checkpoint
            config = checkpoint['config']
            data_config = config.get('data_config', {})
            model_config = config.get('model_config', {})
            vae_config = model_config.get('vae_config', {})
            bilstm_config = model_config.get('bilstm_config', {})
            attention_config = model_config.get('attention_config', {})
            
            extracted_config = {
                'latent_dim': vae_config.get('latent_dim', 64),
                'hidden_dim': bilstm_config.get('hidden_dim', 128),
                'num_layers': bilstm_config.get('num_layers', 2),
                'num_attention_heads': attention_config.get('num_attention_heads', 4),
                'dropout': bilstm_config.get('dropout_rate', 0.4),
                'use_attention': attention_config.get('use_attention', True),
                'pattern_offsets': data_config.get('pattern_offsets', [-4, -3, -2, -1, 0, 1, 2]),
                'current_frame_bias': attention_config.get('current_frame_bias', 1.5),
                'use_physiological_features': attention_config.get('use_physiological_features', True),
                'vae_checkpoint': vae_config.get('vae_checkpoint_path', None)
            }
            print("Using configuration from checkpoint")
        else:
            # Default configuration for backward compatibility
            extracted_config = {
                'latent_dim': 64,
                'hidden_dim': 128,
                'num_layers': 2,
                'num_attention_heads': 4,
                'dropout': 0.4,
                'use_attention': True,
                'pattern_offsets': [-4, -3, -2, -1, 0, 1, 2],
                'current_frame_bias': 1.5,
                'use_physiological_features': True,
                'vae_checkpoint': None
            }
            print("Using default configuration")
        
        # Load VAE
        vae = VAE(latent_dim=extracted_config['latent_dim'])
        if extracted_config['vae_checkpoint'] and os.path.exists(extracted_config['vae_checkpoint']):
            vae_checkpoint = torch.load(extracted_config['vae_checkpoint'], map_location=self.device, weights_only=False)
            if 'model_state_dict' in vae_checkpoint:
                vae.load_state_dict(vae_checkpoint['model_state_dict'])
            else:
                vae.load_state_dict(vae_checkpoint)
            print(f"Loaded VAE from: {extracted_config['vae_checkpoint']}")
        else:
            print("Warning: VAE checkpoint not found, using random initialization")
        
        vae.to(self.device)
        vae.eval()
        
        # Create ImprovedBPPredictor model
        model = ImprovedBPPredictor(
            vae_model=vae,
            latent_dim=extracted_config['latent_dim'],
            hidden_dim=extracted_config['hidden_dim'],
            num_layers=extracted_config['num_layers'],
            num_heads=extracted_config['num_attention_heads'],
            dropout=extracted_config['dropout'],
            use_attention=extracted_config['use_attention'],
            pattern_offsets=extracted_config['pattern_offsets'],
            current_frame_bias=extracted_config['current_frame_bias'],
            use_physiological_features=extracted_config['use_physiological_features']
        )
        
        # Load model state
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.to(self.device)
        model.eval()
        
        print(f"ImprovedBPPredictor loaded successfully")
        print(f"Pattern offsets: {extracted_config['pattern_offsets']}")
        print(f"Use attention: {extracted_config['use_attention']}")
        print(f"Use physiological features: {extracted_config['use_physiological_features']}")
        
        return model, extracted_config
    
    def evaluate_on_dataset(self, data_path: str, max_samples: int = 500) -> Dict:
        """Evaluate ImprovedBPPredictor model on dataset and return predictions"""
        print(f"Evaluating ImprovedBPPredictor on dataset: {data_path}")
        
        # Get the correct HDF5 file path using DataPathManager
        if os.path.isfile(data_path) and data_path.endswith('.h5'):
            h5_file_path = data_path
        else:
            path_manager = DataPathManager(
                subject="subject001",
                session="baseline",
                root=data_path
            )
            h5_file_path = path_manager._h5_path
        
        # Create dataset using ImprovedBPDataset
        dataset = ImprovedBPDataset(
            data_root=str(h5_file_path),
            pattern_offsets=self.model_config['pattern_offsets'],
            max_samples_per_subject=max_samples,
            sequence_step_size=10  # Use step size to match training
        )
        
        # Create dataloader
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=16, shuffle=False, num_workers=2
        )
        
        # Collect predictions
        all_predictions = []
        all_targets = []
        
        print("Running inference with ImprovedBPPredictor...")
        with torch.no_grad():
            for batch in dataloader:
                sequences = batch['sequences'].to(self.device)
                targets = batch['targets'].to(self.device)
                
                # Model prediction
                outputs = self.model(sequences)
                pred_waveforms = outputs['waveform']
                pred_systolic = outputs['systolic'].squeeze() if 'systolic' in outputs else None
                pred_diastolic = outputs['diastolic'].squeeze() if 'diastolic' in outputs else None
                
                # Extract physiological values from targets using improved method
                true_sys, true_dias = extract_bp_values_improved(targets)
                
                # Store results
                for i in range(len(targets)):
                    pred_dict = {
                        'waveform': pred_waveforms[i].cpu().numpy()
                    }
                    
                    target_dict = {
                        'waveform': targets[i].cpu().numpy(),
                        'systolic': true_sys[i] if isinstance(true_sys, np.ndarray) else true_sys,
                        'diastolic': true_dias[i] if isinstance(true_dias, np.ndarray) else true_dias
                    }
                    
                    # Add direct predictions if available
                    if pred_systolic is not None:
                        pred_dict['systolic'] = pred_systolic[i].cpu().numpy()
                    else:
                        # Extract from waveform prediction
                        pred_sys, _ = extract_bp_values_improved(pred_waveforms[i])
                        pred_dict['systolic'] = pred_sys
                    
                    if pred_diastolic is not None:
                        pred_dict['diastolic'] = pred_diastolic[i].cpu().numpy()
                    else:
                        # Extract from waveform prediction
                        _, pred_dias = extract_bp_values_improved(pred_waveforms[i])
                        pred_dict['diastolic'] = pred_dias
                    
                    all_predictions.append(pred_dict)
                    all_targets.append(target_dict)
        
        print(f"Collected {len(all_predictions)} predictions from ImprovedBPPredictor")
        
        return {
            'predictions': all_predictions,
            'targets': all_targets,
            'model_config': self.model_config
        }
    
    def calculate_metrics(self, predictions: List[Dict], targets: List[Dict]) -> Dict:
        """Calculate comprehensive evaluation metrics"""
        
        # Extract arrays with error handling
        pred_sys = []
        pred_dias = []
        pred_waveforms = []
        true_sys = []
        true_dias = []
        true_waveforms = []
        
        for p, t in zip(predictions, targets):
            # Handle systolic predictions
            if isinstance(p['systolic'], (np.ndarray, np.floating)):
                pred_sys.append(float(p['systolic']))
            else:
                pred_sys.append(p['systolic'])
                
            # Handle diastolic predictions  
            if isinstance(p['diastolic'], (np.ndarray, np.floating)):
                pred_dias.append(float(p['diastolic']))
            else:
                pred_dias.append(p['diastolic'])
                
            # Handle waveform predictions
            pred_waveforms.append(p['waveform'])
            
            # Handle targets
            if isinstance(t['systolic'], (np.ndarray, np.floating)):
                true_sys.append(float(t['systolic']))
            else:
                true_sys.append(t['systolic'])
                
            if isinstance(t['diastolic'], (np.ndarray, np.floating)):
                true_dias.append(float(t['diastolic']))
            else:
                true_dias.append(t['diastolic'])
                
            true_waveforms.append(t['waveform'])
        
        # Convert to numpy arrays
        pred_sys = np.array(pred_sys)
        pred_dias = np.array(pred_dias)
        pred_waveforms = np.array(pred_waveforms)
        true_sys = np.array(true_sys)
        true_dias = np.array(true_dias)
        true_waveforms = np.array(true_waveforms)
        
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
        
        # Waveform metrics (full signal)
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
        
        # Clinical accuracy metrics (within tolerance)
        metrics['clinical'] = {
            'systolic_5mmhg': np.mean(np.abs(pred_sys - true_sys) <= 5) * 100,
            'systolic_10mmhg': np.mean(np.abs(pred_sys - true_sys) <= 10) * 100,
            'systolic_15mmhg': np.mean(np.abs(pred_sys - true_sys) <= 15) * 100,
            'diastolic_5mmhg': np.mean(np.abs(pred_dias - true_dias) <= 5) * 100,
            'diastolic_10mmhg': np.mean(np.abs(pred_dias - true_dias) <= 10) * 100,
            'diastolic_15mmhg': np.mean(np.abs(pred_dias - true_dias) <= 15) * 100,
        }
        
        return metrics
    
    def create_evaluation_plot(self, predictions: List[Dict], targets: List[Dict], 
                             output_path: str, model_name: str = "BP Model"):
        """Create comprehensive 9-panel evaluation plot"""
        
        # Extract data
        pred_sys = np.array([p['systolic'] for p in predictions])
        pred_dias = np.array([p['diastolic'] for p in predictions])
        pred_waveforms = np.array([p['waveform'] for p in predictions])
        
        true_sys = np.array([t['systolic'] for t in targets])
        true_dias = np.array([t['diastolic'] for t in targets])
        true_waveforms = np.array([t['waveform'] for t in targets])
        
        # Calculate metrics
        metrics = self.calculate_metrics(predictions, targets)
        
        # Create figure
        fig = plt.figure(figsize=(15, 12))
        
        # Define subplot layout (3x3)
        # Row 1: Full waveform
        # Row 2: Systolic BP (SBP)  
        # Row 3: Diastolic BP (DBP)
        
        # Helper function for correlation plot
        def plot_correlation(ax, true_vals, pred_vals, title, metrics_dict):
            ax.scatter(true_vals, pred_vals, alpha=0.6, s=20)
            
            # Perfect correlation line
            min_val, max_val = min(true_vals.min(), pred_vals.min()), max(true_vals.max(), pred_vals.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.8)
            
            # Regression line
            z = np.polyfit(true_vals, pred_vals, 1)
            p = np.poly1d(z)
            ax.plot(true_vals, p(true_vals), "r-", alpha=0.8)
            
            ax.set_xlabel(f'True {title} (mmHg)')
            ax.set_ylabel(f'Predicted {title} (mmHg)')
            ax.set_title(f'{title}')
            
            # Add metrics text
            r2 = metrics_dict.get('r2', 0)
            p_val = metrics_dict.get('pearson_p', 1)
            ax.text(0.05, 0.95, f'r²={r2:.4f}\np={p_val:.4f}', 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            ax.grid(True, alpha=0.3)
        
        # Helper function for Bland-Altman plot
        def plot_bland_altman(ax, true_vals, pred_vals, title, metrics_dict):
            diff = pred_vals - true_vals
            mean_vals = (pred_vals + true_vals) / 2
            
            ax.scatter(mean_vals, diff, alpha=0.6, s=20)
            
            # Mean difference line
            mean_diff = np.mean(diff)
            ax.axhline(mean_diff, color='blue', linestyle='-', alpha=0.8)
            
            # 95% limits of agreement
            std_diff = np.std(diff)
            upper_loa = mean_diff + 1.96 * std_diff
            lower_loa = mean_diff - 1.96 * std_diff
            
            ax.axhline(upper_loa, color='red', linestyle='--', alpha=0.8)
            ax.axhline(lower_loa, color='red', linestyle='--', alpha=0.8)
            
            ax.set_xlabel(f'Means (mmHg)')
            ax.set_ylabel(f'Difference (mmHg)')
            ax.set_title(f'{title}')
            
            # Add statistics
            ax.text(0.05, 0.95, f'+1.96 SD:\n{upper_loa:.2f}', 
                   transform=ax.transAxes, verticalalignment='top')
            ax.text(0.05, 0.5, f'MEAN DIFF:\n{mean_diff:.2f}', 
                   transform=ax.transAxes, verticalalignment='center')
            ax.text(0.05, 0.05, f'-1.96 SD:\n{lower_loa:.2f}', 
                   transform=ax.transAxes, verticalalignment='bottom')
            
            ax.grid(True, alpha=0.3)
        
        # Helper function for error histogram
        def plot_error_histogram(ax, true_vals, pred_vals, title, metrics_dict):
            errors = np.abs(pred_vals - true_vals)
            
            ax.hist(errors, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            
            # Add vertical lines for clinical thresholds
            ax.axvline(5, color='green', linestyle='--', alpha=0.8, label='5 mmHg')
            ax.axvline(10, color='orange', linestyle='--', alpha=0.8, label='10 mmHg')
            ax.axvline(15, color='red', linestyle='--', alpha=0.8, label='15 mmHg')
            
            ax.set_xlabel('Absolute error (mmHg)')
            ax.set_ylabel('Occurrences')
            ax.set_title(f'{title}')
            
            # Add accuracy statistics
            mae = metrics_dict.get('mae', 0)
            std_val = metrics_dict.get('std_diff', 0)
            
            # Calculate clinical accuracies
            acc_5 = np.mean(errors <= 5) * 100
            acc_10 = np.mean(errors <= 10) * 100
            acc_15 = np.mean(errors <= 15) * 100
            
            stats_text = f'mean: {mae:.2f}\nstd: {std_val:.2f}\n5-tol: {acc_5:.2f} %\n10-tol: {acc_10:.2f} %\n15-tol: {acc_15:.2f} %'
            ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, 
                   verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            ax.grid(True, alpha=0.3)
        
        # Create subplots
        # Row 1: Full waveform
        ax1 = plt.subplot(3, 3, 1)
        plot_correlation(ax1, true_waveforms.flatten(), pred_waveforms.flatten(), 
                        'FULL', metrics['waveform'])
        
        ax2 = plt.subplot(3, 3, 2)
        plot_bland_altman(ax2, true_waveforms.flatten(), pred_waveforms.flatten(), 
                         'FULL', metrics['waveform'])
        
        ax3 = plt.subplot(3, 3, 3)
        plot_error_histogram(ax3, true_waveforms.flatten(), pred_waveforms.flatten(), 
                           'FULL', metrics['waveform'])
        
        # Row 2: Systolic BP
        ax4 = plt.subplot(3, 3, 4)
        plot_correlation(ax4, true_sys, pred_sys, 'SBP', metrics['systolic'])
        
        ax5 = plt.subplot(3, 3, 5)
        plot_bland_altman(ax5, true_sys, pred_sys, 'SBP', metrics['systolic'])
        
        ax6 = plt.subplot(3, 3, 6)
        plot_error_histogram(ax6, true_sys, pred_sys, 'SBP', metrics['systolic'])
        
        # Row 3: Diastolic BP
        ax7 = plt.subplot(3, 3, 7)
        plot_correlation(ax7, true_dias, pred_dias, 'DBP', metrics['diastolic'])
        
        ax8 = plt.subplot(3, 3, 8)
        plot_bland_altman(ax8, true_dias, pred_dias, 'DBP', metrics['diastolic'])
        
        ax9 = plt.subplot(3, 3, 9)
        plot_error_histogram(ax9, true_dias, pred_dias, 'DBP', metrics['diastolic'])
        
        # Add main title
        n_samples = len(predictions)
        fig.suptitle(f'{model_name} | subject 001 | #test={n_samples}', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Evaluation plot saved to: {output_path}")
        
        return metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate ImprovedBPPredictor Models')
    
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to evaluation dataset')
    parser.add_argument('--config_path', type=str, default=None,
                       help='Path to YAML configuration file (optional)')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results',
                       help='Output directory for results')
    parser.add_argument('--model_name', type=str, default='ImprovedBPPredictor',
                       help='Model name for plot title')
    parser.add_argument('--max_samples', type=int, default=500,
                       help='Maximum number of samples to evaluate')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*60)
    print("IMPROVED BP PREDICTOR EVALUATION")
    print("="*60)
    print(f"Model checkpoint: {args.model_path}")
    print(f"Data path: {args.data_path}")
    print(f"Config path: {args.config_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Max samples: {args.max_samples}")
    print("="*60)
    
    # Initialize evaluator
    evaluator = BPModelEvaluator(args.model_path, args.config_path)
    
    # Run evaluation
    results = evaluator.evaluate_on_dataset(args.data_path, args.max_samples)
    
    # Calculate metrics
    metrics = evaluator.calculate_metrics(results['predictions'], results['targets'])
    
    # Create evaluation plot
    plot_path = os.path.join(args.output_dir, 'evaluation_plot.png')
    evaluator.create_evaluation_plot(
        results['predictions'], 
        results['targets'], 
        plot_path, 
        args.model_name
    )
    
    # Save metrics to file
    metrics_path = os.path.join(args.output_dir, 'evaluation_metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write(f"Evaluation Results for {args.model_name}\n")
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
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETED!")
    print("="*60)
    print(f"Results saved to: {args.output_dir}")
    print(f"Plot: {plot_path}")
    print(f"Metrics: {metrics_path}")
    
    # Print key metrics summary
    print("\nKEY METRICS SUMMARY:")
    print("-" * 30)
    print(f"Systolic MAE:  {metrics['systolic']['mae']:.2f} mmHg (R²={metrics['systolic']['r2']:.3f})")
    print(f"Diastolic MAE: {metrics['diastolic']['mae']:.2f} mmHg (R²={metrics['diastolic']['r2']:.3f})")
    print(f"Waveform MAE:  {metrics['waveform']['mae']:.2f} mmHg (R²={metrics['waveform']['r2']:.3f})")
    print("\nCLINICAL ACCURACY:")
    print(f"Systolic ≤5mmHg:  {metrics['clinical']['systolic_5mmhg']:.1f}%")
    print(f"Systolic ≤10mmHg: {metrics['clinical']['systolic_10mmhg']:.1f}%") 
    print(f"Diastolic ≤5mmHg:  {metrics['clinical']['diastolic_5mmhg']:.1f}%")
    print(f"Diastolic ≤10mmHg: {metrics['clinical']['diastolic_10mmhg']:.1f}%")
    print("="*60)


if __name__ == '__main__':
    main() 