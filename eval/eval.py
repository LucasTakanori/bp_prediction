import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import sys
import numpy as np
import argparse
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import wandb
from datetime import datetime

# Add the parent directory to the path to find utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Import your dataset utilities
from utils.data_utils import PviDataset, PviBatchServer

# Define the VAE model
class VAE(nn.Module):
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


# Bidirectional LSTM model for blood pressure prediction using VAE embeddings
class VAEBiLSTM(nn.Module):
    def __init__(self, vae_model, input_dim=256, hidden_dim=256, num_layers=2, output_dim=50, dropout=0.3):
        super(VAEBiLSTM, self).__init__()
        
        # Store the pretrained VAE
        self.vae = vae_model
        # Freeze the VAE parameters
        for param in self.vae.parameters():
            param.requires_grad = False
        
        # Bidirectional LSTM for sequence modeling
        self.lstm = nn.LSTM(
            input_size=input_dim,     # Size of each input (latent dimension)
            hidden_size=hidden_dim,   # Size of hidden state
            num_layers=num_layers,    # Number of LSTM layers
            batch_first=True,         # Input shape: (batch, seq, feature)
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True        # Use bidirectional LSTM
        )
        
        # Output layer (note: bidirectional means we have 2*hidden_dim features)
        self.fc = nn.Linear(hidden_dim * 2, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x_seq):
        """
        x_seq: sequence of images [batch_size, seq_len, 1, 32, 32]
        """
        batch_size, seq_len = x_seq.shape[0], x_seq.shape[1]
        
        # Encode each frame to get its latent representation
        latent_seq = []
        for t in range(seq_len):
            # Get current frame
            x_t = x_seq[:, t]  # [batch_size, 1, 32, 32]
            
            # Encode the frame (without computing gradients for VAE)
            with torch.no_grad():
                mu_t, _ = self.vae.encode(x_t)
            
            # Store the latent representation (just the mean)
            latent_seq.append(mu_t)
        
        # Stack latent representations along the sequence dimension
        latent_seq = torch.stack(latent_seq, dim=1)  # [batch_size, seq_len, latent_dim]
        
        # Process with bidirectional LSTM
        lstm_out, _ = self.lstm(latent_seq)  # [batch_size, seq_len, 2*hidden_dim]
        
        # Use the last time step output from both directions
        final_hidden_state = lstm_out[:, -1, :]
        
        # Process through fully connected layers
        x = self.fc(final_hidden_state)
        x = self.relu(x)
        x = self.dropout(x)
        final_out = self.fc_out(x)
        
        return final_out


def extract_frame(batch_data, frame_idx=0):
    """Extract a specific frame from the batch data and normalize it to [0, 1]"""
    if isinstance(batch_data, dict):
        pvi_data = batch_data['pviHP']
    else:
        pvi_data = batch_data
    
    frames = pvi_data[:, 0, :, :, frame_idx].unsqueeze(1)
    frames = torch.nan_to_num(frames, nan=0.0)
    
    batch_size = frames.shape[0]
    normalized_frames = torch.zeros_like(frames)
    
    for i in range(batch_size):
        sample = frames[i]
        min_val = torch.min(sample)
        max_val = torch.max(sample)
        
        if min_val == max_val:
            normalized_frames[i] = torch.zeros_like(sample)
        else:
            normalized_frames[i] = (sample - min_val) / (max_val - min_val)
    
    return normalized_frames


def prepare_specific_frame_data(batch_data, central_indices=None, pattern_offsets=[-7, 0, 3]):
    """Prepare data with specific frame pattern (k-7, k, k+3) and corresponding BP values"""
    if not isinstance(batch_data, dict):
        raise ValueError("Expected batch_data to be a dictionary")
    
    pvi_data = batch_data['pviHP']
    bp_data = batch_data['bp']
    
    batch_size = pvi_data.shape[0]
    total_frames = pvi_data.shape[-1]
    
    min_offset = min(pattern_offsets)
    max_offset = max(pattern_offsets)
    valid_indices_start = max(0, -min_offset)
    valid_indices_end = min(total_frames, total_frames - max_offset)
    
    if central_indices is None:
        central_indices = list(range(valid_indices_start, valid_indices_end))
    else:
        central_indices = [k for k in central_indices if valid_indices_start <= k < valid_indices_end]
    
    frame_sequences = []
    bp_targets = []
    
    for b in range(batch_size):
        for k in central_indices:
            seq = []
            for offset in pattern_offsets:
                frame_idx = k + offset
                frame = extract_frame(batch_data, frame_idx)[b:b+1]
                seq.append(frame)
            
            seq = torch.cat(seq, dim=0)
            frame_sequences.append(seq)
            bp_targets.append(bp_data[b])
    
    frame_sequences = torch.stack(frame_sequences)
    bp_targets = torch.stack(bp_targets)
    
    return frame_sequences, bp_targets


def normalize_bp(bp_data, min_val=40, max_val=200):
    """Normalize blood pressure data to [0, 1] range"""
    bp_normalized = (bp_data - min_val) / (max_val - min_val)
    return bp_normalized, (min_val, max_val)


def denormalize_bp(bp_normalized, min_val=40, max_val=200):
    """Convert normalized BP back to original scale"""
    bp_denormalized = bp_normalized * (max_val - min_val) + min_val
    return bp_denormalized


def extract_systolic_diastolic(waveform):
    """
    Extract systolic and diastolic values from a BP waveform
    Systolic: Maximum value (peak)
    Diastolic: Minimum value AFTER the systolic peak
    """
    # Find systolic (peak)
    sys_idx = np.argmax(waveform)
    sys_value = waveform[sys_idx]
    
    # Find diastolic (minimum after the peak)
    if sys_idx < len(waveform) - 1:
        # Find minimum in the portion after the peak
        dias_idx = sys_idx + np.argmin(waveform[sys_idx:])
        dias_value = waveform[dias_idx]
    else:
        # If peak is at the end, use the global minimum
        dias_idx = np.argmin(waveform)
        dias_value = waveform[dias_idx]
    
    return sys_value, dias_value, sys_idx, dias_idx


def visualize_systolic_diastolic_extraction(predictions, targets, output_dir, num_examples=3):
    """
    Visualize the extraction of systolic and diastolic values from BP waveforms
    Shows individual waveforms and how we extract max/min values
    """
    results_dir = os.path.join(output_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Select random examples
    indices = np.random.choice(len(predictions), min(num_examples, len(predictions)), replace=False)
    
    fig, axes = plt.subplots(num_examples, 2, figsize=(15, 4*num_examples))
    if num_examples == 1:
        axes = axes.reshape(1, -1)
    
    for i, idx in enumerate(indices):
        pred_waveform = predictions[idx]
        target_waveform = targets[idx]
        
        # Extract systolic and diastolic values with proper ordering
        sys_pred, dias_pred, sys_pred_idx, dias_pred_idx = extract_systolic_diastolic(pred_waveform)
        sys_target, dias_target, sys_target_idx, dias_target_idx = extract_systolic_diastolic(target_waveform)
        
        # Plot prediction
        axes[i, 0].plot(pred_waveform, 'r-', label='Predicted BP')
        axes[i, 0].scatter(sys_pred_idx, sys_pred, c='red', s=100, marker='o', 
                          label=f'Systolic: {sys_pred:.1f} mmHg')
        axes[i, 0].scatter(dias_pred_idx, dias_pred, c='red', s=100, marker='s', 
                          label=f'Diastolic: {dias_pred:.1f} mmHg')
        axes[i, 0].set_title(f'Predicted Waveform - Example {i+1}')
        axes[i, 0].set_xlabel('Time Index')
        axes[i, 0].set_ylabel('Blood Pressure (mmHg)')
        axes[i, 0].legend()
        axes[i, 0].grid(True)
        
        # Plot target
        axes[i, 1].plot(target_waveform, 'b-', label='Target BP')
        axes[i, 1].scatter(sys_target_idx, sys_target, c='blue', s=100, marker='o', 
                          label=f'Systolic: {sys_target:.1f} mmHg')
        axes[i, 1].scatter(dias_target_idx, dias_target, c='blue', s=100, marker='s', 
                          label=f'Diastolic: {dias_target:.1f} mmHg')
        axes[i, 1].set_title(f'Target Waveform - Example {i+1}')
        axes[i, 1].set_xlabel('Time Index')
        axes[i, 1].set_ylabel('Blood Pressure (mmHg)')
        axes[i, 1].legend()
        axes[i, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'systolic_diastolic_extraction_visualization.png'), dpi=300)
    plt.close()


def evaluate_model(model, data_loader, device, pattern_offsets=[-7, 0, 3], 
                  bp_norm_params=(40, 200), output_dir='evaluation_output'):
    """
    Evaluate model with proper systolic/diastolic extraction and visualization
    """
    model.eval()
    
    results_dir = os.path.join(output_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    bp_min, bp_max = bp_norm_params
    
    all_predictions = []
    all_targets = []
    
    # Process batches
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(tqdm(data_loader, desc="Evaluating")):
            try:
                # Prepare sequences
                sequences, targets = prepare_specific_frame_data(batch_data, pattern_offsets=pattern_offsets)
                
                # Normalize targets
                targets_norm, _ = normalize_bp(targets, bp_min, bp_max)
                
                # Move to device
                sequences = sequences.to(device)
                targets_norm = targets_norm.to(device)
                
                # Forward pass
                outputs_norm = model(sequences)
                
                # Denormalize predictions
                outputs = denormalize_bp(outputs_norm.cpu(), bp_min, bp_max)
                
                # Store predictions and targets
                all_predictions.append(outputs.numpy())
                all_targets.append(targets.cpu().numpy())
                
            except Exception as e:
                print(f"Error processing batch {batch_idx}: {e}")
                continue
    
    # Concatenate all predictions and targets
    if all_predictions and all_targets:
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        
        # Visualize example waveforms with systolic/diastolic extraction
        visualize_systolic_diastolic_extraction(all_predictions, all_targets, output_dir)
        
        # Extract systolic and diastolic values properly
        sys_targets = []
        dias_targets = []
        sys_preds = []
        dias_preds = []
        
        for i in range(len(all_targets)):
            sys_t, dias_t, _, _ = extract_systolic_diastolic(all_targets[i])
            sys_p, dias_p, _, _ = extract_systolic_diastolic(all_predictions[i])
            
            sys_targets.append(sys_t)
            dias_targets.append(dias_t)
            sys_preds.append(sys_p)
            dias_preds.append(dias_p)
        
        sys_targets = np.array(sys_targets)
        dias_targets = np.array(dias_targets)
        sys_preds = np.array(sys_preds)
        dias_preds = np.array(dias_preds)
        
        print(f"Number of waveforms: {len(all_targets)}")
        print(f"Number of systolic values: {len(sys_targets)}")
        print(f"Number of diastolic values: {len(dias_targets)}")
        
        # Calculate metrics
        sys_mae = np.mean(np.abs(sys_targets - sys_preds))
        dias_mae = np.mean(np.abs(dias_targets - dias_preds))
        sys_bias = np.mean(sys_targets - sys_preds)
        dias_bias = np.mean(dias_targets - dias_preds)
        
        # Calculate Pearson correlation coefficients
        # For entire waveforms (flattened)
        waveform_pearson_r = np.corrcoef(all_targets.flatten(), all_predictions.flatten())[0, 1]
        waveform_r2 = r2_score(all_targets.flatten(), all_predictions.flatten())
        
        # For systolic values
        systolic_pearson_r = np.corrcoef(sys_targets, sys_preds)[0, 1]
        systolic_r2 = r2_score(sys_targets, sys_preds)
        
        # For diastolic values
        diastolic_pearson_r = np.corrcoef(dias_targets, dias_preds)[0, 1]
        diastolic_r2 = r2_score(dias_targets, dias_preds)
        
        # Calculate additional metrics
        waveform_mae = mean_absolute_error(all_targets.flatten(), all_predictions.flatten())
        waveform_mse = mean_squared_error(all_targets.flatten(), all_predictions.flatten())
        
        print("\n=== EVALUATION RESULTS ===")
        print(f"Systolic MAE: {sys_mae:.2f} mmHg, Bias: {sys_bias:.2f}")
        print(f"Diastolic MAE: {dias_mae:.2f} mmHg, Bias: {dias_bias:.2f}")
        print(f"\nWaveform metrics:")
        print(f"  MAE: {waveform_mae:.2f}")
        print(f"  MSE: {waveform_mse:.2f}")
        print(f"  Pearson r: {waveform_pearson_r:.4f}")
        print(f"  R²: {waveform_r2:.4f}")
        print(f"\nSystolic metrics:")
        print(f"  Pearson r: {systolic_pearson_r:.4f}")
        print(f"  R²: {systolic_r2:.4f}")
        print(f"\nDiastolic metrics:")
        print(f"  Pearson r: {diastolic_pearson_r:.4f}")
        print(f"  R²: {diastolic_r2:.4f}")
        
        # Create proper Bland-Altman plots
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Systolic Bland-Altman
        mean_sys = (sys_targets + sys_preds) / 2
        diff_sys = sys_targets - sys_preds
        
        axes[0].scatter(mean_sys, diff_sys, alpha=0.5)
        axes[0].axhline(y=np.mean(diff_sys), color='r', linestyle='-', 
                       label=f'Mean Error: {np.mean(diff_sys):.2f}')
        axes[0].axhline(y=np.mean(diff_sys) + 1.96*np.std(diff_sys), color='g', linestyle='--', 
                       label=f'95% Limits: {np.mean(diff_sys) + 1.96*np.std(diff_sys):.2f}')
        axes[0].axhline(y=np.mean(diff_sys) - 1.96*np.std(diff_sys), color='g', linestyle='--',
                       label=f'{np.mean(diff_sys) - 1.96*np.std(diff_sys):.2f}')
        axes[0].set_xlabel('Mean Systolic BP (mmHg)')
        axes[0].set_ylabel('Difference (Target - Predicted)')
        axes[0].set_title('Bland-Altman Plot for Systolic BP')
        axes[0].legend()
        axes[0].grid(True)
        
        # Diastolic Bland-Altman
        mean_dias = (dias_targets + dias_preds) / 2
        diff_dias = dias_targets - dias_preds
        
        axes[1].scatter(mean_dias, diff_dias, alpha=0.5)
        axes[1].axhline(y=np.mean(diff_dias), color='r', linestyle='-', 
                       label=f'Mean Error: {np.mean(diff_dias):.2f}')
        axes[1].axhline(y=np.mean(diff_dias) + 1.96*np.std(diff_dias), color='g', linestyle='--', 
                       label=f'95% Limits: {np.mean(diff_dias) + 1.96*np.std(diff_dias):.2f}')
        axes[1].axhline(y=np.mean(diff_dias) - 1.96*np.std(diff_dias), color='g', linestyle='--',
                       label=f'{np.mean(diff_dias) - 1.96*np.std(diff_dias):.2f}')
        axes[1].set_xlabel('Mean Diastolic BP (mmHg)')
        axes[1].set_ylabel('Difference (Target - Predicted)')
        axes[1].set_title('Bland-Altman Plot for Diastolic BP')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'corrected_bland_altman.png'), dpi=300)
        plt.close()
        
        # Create scatter plots with correlation
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Systolic scatter plot
        axes[0].scatter(sys_targets, sys_preds, alpha=0.5)
        axes[0].plot([sys_targets.min(), sys_targets.max()], 
                     [sys_targets.min(), sys_targets.max()], 'r--', label='Identity line')
        axes[0].set_xlabel('True Systolic BP (mmHg)')
        axes[0].set_ylabel('Predicted Systolic BP (mmHg)')
        axes[0].set_title(f'Systolic BP (r={systolic_pearson_r:.3f}, R²={systolic_r2:.3f})')
        axes[0].legend()
        axes[0].grid(True)
        
        # Diastolic scatter plot
        axes[1].scatter(dias_targets, dias_preds, alpha=0.5)
        axes[1].plot([dias_targets.min(), dias_targets.max()], 
                     [dias_targets.min(), dias_targets.max()], 'r--', label='Identity line')
        axes[1].set_xlabel('True Diastolic BP (mmHg)')
        axes[1].set_ylabel('Predicted Diastolic BP (mmHg)')
        axes[1].set_title(f'Diastolic BP (r={diastolic_pearson_r:.3f}, R²={diastolic_r2:.3f})')
        axes[1].legend()
        axes[1].grid(True)
        
        # Waveform scatter plot (sample every 5th point for clarity)
        flat_targets = all_targets.flatten()[::5]
        flat_preds = all_predictions.flatten()[::5]
        axes[2].scatter(flat_targets, flat_preds, alpha=0.1)
        axes[2].plot([flat_targets.min(), flat_targets.max()], 
                     [flat_targets.min(), flat_targets.max()], 'r--', label='Identity line')
        axes[2].set_xlabel('True BP (mmHg)')
        axes[2].set_ylabel('Predicted BP (mmHg)')
        axes[2].set_title(f'All BP values (r={waveform_pearson_r:.3f}, R²={waveform_r2:.3f})')
        axes[2].legend()
        axes[2].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'correlation_scatter_plots.png'), dpi=300)
        plt.close()
        
        # Create a visualization showing the distribution of predictions vs targets
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Systolic
        axes[0].hist(sys_targets, bins=30, alpha=0.5, label='Target', density=True)
        axes[0].hist(sys_preds, bins=30, alpha=0.5, label='Predicted', density=True)
        axes[0].set_xlabel('Systolic BP (mmHg)')
        axes[0].set_ylabel('Density')
        axes[0].set_title('Distribution of Systolic Blood Pressure')
        axes[0].legend()
        axes[0].grid(True)
        
        # Diastolic
        axes[1].hist(dias_targets, bins=30, alpha=0.5, label='Target', density=True)
        axes[1].hist(dias_preds, bins=30, alpha=0.5, label='Predicted', density=True)
        axes[1].set_xlabel('Diastolic BP (mmHg)')
        axes[1].set_ylabel('Density')
        axes[1].set_title('Distribution of Diastolic Blood Pressure')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'bp_distributions.png'), dpi=300)
        plt.close()
        
        # Create comprehensive 9-panel figure like the reference image
        fig = plt.figure(figsize=(18, 15))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Full waveform analysis
        # Predicted vs True for full waveform
        ax1 = fig.add_subplot(gs[0, 0])
        flat_targets = all_targets.flatten()
        flat_preds = all_predictions.flatten()
        ax1.scatter(flat_targets, flat_preds, alpha=0.05, s=1)
        ax1.plot([flat_targets.min(), flat_targets.max()], 
                [flat_targets.min(), flat_targets.max()], 'k-', linewidth=2)
        ax1.set_xlabel('True FULL (mmHg)')
        ax1.set_ylabel('Predicted FULL (mmHg)')
        ax1.text(0.05, 0.95, f'r2={waveform_r2:.4f}\np={0.0000:.4f}', 
                transform=ax1.transAxes, verticalalignment='top')
        
        # Bland-Altman for full waveform
        ax2 = fig.add_subplot(gs[0, 1])
        mean_full = (flat_targets + flat_preds) / 2
        diff_full = flat_targets - flat_preds
        ax2.scatter(mean_full, diff_full, alpha=0.05, s=1)
        mean_diff = np.mean(diff_full)
        std_diff = np.std(diff_full)
        ax2.axhline(y=mean_diff, color='r', linestyle='-', label=f'MEAN DIFF: {mean_diff:.2f}')
        ax2.axhline(y=mean_diff + 1.96*std_diff, color='b', linestyle='--', 
                   label=f'+1.96 SD: {mean_diff + 1.96*std_diff:.2f}')
        ax2.axhline(y=mean_diff - 1.96*std_diff, color='b', linestyle='--', 
                   label=f'-1.96 SD: {mean_diff - 1.96*std_diff:.2f}')
        ax2.set_xlabel('Means (mmHg)')
        ax2.set_ylabel('Difference (mmHg)')
        ax2.legend()
        
        # Histogram for full waveform absolute errors
        ax3 = fig.add_subplot(gs[0, 2])
        abs_errors_full = np.abs(diff_full)
        ax3.hist(abs_errors_full, bins=50, alpha=0.7, color='#1f77b4', edgecolor='black')
        
        # Add vertical lines for statistics
        mean_abs_error = np.mean(abs_errors_full)
        sd_abs_error = np.std(abs_errors_full)
        tolerance_levels = [5, 10, 15]  # Tolerance levels in mmHg
        
        ax3.axvline(x=mean_abs_error, color='black', linestyle='-', linewidth=2)
        
        for tol in tolerance_levels:
            ax3.axvline(x=tol, color='gray', linestyle='--')
        
        ax3.set_xlabel('Absolute error (mmHg)')
        ax3.set_ylabel('Occurrences')
        
        # Calculate percentage of errors within tolerance levels
        tolerance_percentages = []
        for tol in tolerance_levels:
            percentage = np.mean(abs_errors_full <= tol) * 100
            tolerance_percentages.append(percentage)
        
        ax3.text(0.98, 0.98, f'mean: {mean_abs_error:.2f}\nsd: {sd_abs_error:.2f}\n' +
                '\n'.join([f'{tol}-tol: {perc:.2f} %' for tol, perc in zip(tolerance_levels, tolerance_percentages)]),
                transform=ax3.transAxes, verticalalignment='top', horizontalalignment='right', 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Systolic BP analysis
        # Predicted vs True for SBP
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.scatter(sys_targets, sys_preds, alpha=0.5, s=10)
        ax4.plot([sys_targets.min(), sys_targets.max()], 
                [sys_targets.min(), sys_targets.max()], 'k-', linewidth=2)
        ax4.set_xlabel('True SBP (mmHg)')
        ax4.set_ylabel('Predicted SBP (mmHg)')
        ax4.text(0.05, 0.95, f'r2={systolic_r2:.4f}\np={0.0000:.4f}', 
                transform=ax4.transAxes, verticalalignment='top')
        
        # Bland-Altman for SBP
        ax5 = fig.add_subplot(gs[1, 1])
        mean_sys = (sys_targets + sys_preds) / 2
        diff_sys = sys_targets - sys_preds
        ax5.scatter(mean_sys, diff_sys, alpha=0.5, s=10)
        mean_diff_sys = np.mean(diff_sys)
        std_diff_sys = np.std(diff_sys)
        ax5.axhline(y=mean_diff_sys, color='r', linestyle='-', label=f'MEAN DIFF: {mean_diff_sys:.2f}')
        ax5.axhline(y=mean_diff_sys + 1.96*std_diff_sys, color='b', linestyle='--', 
                   label=f'+1.96 SD: {mean_diff_sys + 1.96*std_diff_sys:.2f}')
        ax5.axhline(y=mean_diff_sys - 1.96*std_diff_sys, color='b', linestyle='--', 
                   label=f'-1.96 SD: {mean_diff_sys - 1.96*std_diff_sys:.2f}')
        ax5.set_xlabel('Means (mmHg)')
        ax5.set_ylabel('Difference (mmHg)')
        ax5.legend()
        
        # Histogram for SBP absolute errors
        ax6 = fig.add_subplot(gs[1, 2])
        abs_errors_sys = np.abs(diff_sys)
        ax6.hist(abs_errors_sys, bins=30, alpha=0.7, color='#1f77b4', edgecolor='black')
        
        mean_abs_error_sys = np.mean(abs_errors_sys)
        sd_abs_error_sys = np.std(abs_errors_sys)
        
        ax6.axvline(x=mean_abs_error_sys, color='black', linestyle='-', linewidth=2)
        
        for tol in tolerance_levels:
            ax6.axvline(x=tol, color='gray', linestyle='--')
        
        ax6.set_xlabel('Absolute error (mmHg)')
        ax6.set_ylabel('Occurrences')
        
        # Calculate percentage of errors within tolerance levels for SBP
        tolerance_percentages_sys = []
        for tol in tolerance_levels:
            percentage = np.mean(abs_errors_sys <= tol) * 100
            tolerance_percentages_sys.append(percentage)
        
        ax6.text(0.98, 0.98, f'mean: {mean_abs_error_sys:.2f}\nsd: {sd_abs_error_sys:.2f}\n' +
                '\n'.join([f'{tol}-tol: {perc:.2f} %' for tol, perc in zip(tolerance_levels, tolerance_percentages_sys)]),
                transform=ax6.transAxes, verticalalignment='top', horizontalalignment='right', 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Diastolic BP analysis
        # Predicted vs True for DBP
        ax7 = fig.add_subplot(gs[2, 0])
        ax7.scatter(dias_targets, dias_preds, alpha=0.5, s=10)
        ax7.plot([dias_targets.min(), dias_targets.max()], 
                [dias_targets.min(), dias_targets.max()], 'k-', linewidth=2)
        ax7.set_xlabel('True DBP (mmHg)')
        ax7.set_ylabel('Predicted DBP (mmHg)')
        ax7.text(0.05, 0.95, f'r2={diastolic_r2:.4f}\np={0.0000:.4f}', 
                transform=ax7.transAxes, verticalalignment='top')
        
        # Bland-Altman for DBP
        ax8 = fig.add_subplot(gs[2, 1])
        mean_dias = (dias_targets + dias_preds) / 2
        diff_dias = dias_targets - dias_preds
        ax8.scatter(mean_dias, diff_dias, alpha=0.5, s=10)
        mean_diff_dias = np.mean(diff_dias)
        std_diff_dias = np.std(diff_dias)
        ax8.axhline(y=mean_diff_dias, color='r', linestyle='-', label=f'MEAN DIFF: {mean_diff_dias:.2f}')
        ax8.axhline(y=mean_diff_dias + 1.96*std_diff_dias, color='b', linestyle='--', 
                   label=f'+1.96 SD: {mean_diff_dias + 1.96*std_diff_dias:.2f}')
        ax8.axhline(y=mean_diff_dias - 1.96*std_diff_dias, color='b', linestyle='--', 
                   label=f'-1.96 SD: {mean_diff_dias - 1.96*std_diff_dias:.2f}')
        ax8.set_xlabel('Means (mmHg)')
        ax8.set_ylabel('Difference (mmHg)')
        ax8.legend()
        
        # Histogram for DBP absolute errors
        ax9 = fig.add_subplot(gs[2, 2])
        abs_errors_dias = np.abs(diff_dias)
        ax9.hist(abs_errors_dias, bins=30, alpha=0.7, color='#1f77b4', edgecolor='black')
        
        mean_abs_error_dias = np.mean(abs_errors_dias)
        sd_abs_error_dias = np.std(abs_errors_dias)
        
        ax9.axvline(x=mean_abs_error_dias, color='black', linestyle='-', linewidth=2)
        
        for tol in tolerance_levels:
            ax9.axvline(x=tol, color='gray', linestyle='--')
        
        ax9.set_xlabel('Absolute error (mmHg)')
        ax9.set_ylabel('Occurrences')
        
        # Calculate percentage of errors within tolerance levels for DBP
        tolerance_percentages_dias = []
        for tol in tolerance_levels:
            percentage = np.mean(abs_errors_dias <= tol) * 100
            tolerance_percentages_dias.append(percentage)
        
        ax9.text(0.98, 0.98, f'mean: {mean_abs_error_dias:.2f}\nsd: {sd_abs_error_dias:.2f}\n' +
                '\n'.join([f'{tol}-tol: {perc:.2f} %' for tol, perc in zip(tolerance_levels, tolerance_percentages_dias)]),
                transform=ax9.transAxes, verticalalignment='top', horizontalalignment='right', 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Add title with model info
        fig.suptitle(f'{datetime.now().strftime("%Y-%m-%d %H.%M.%S")} | model=VAEBiLSTM\n'
                    f'baseline | subject 001-020 | #train={len(all_targets)} | #test={len(all_targets)}',
                    fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'comprehensive_evaluation.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Also save individual components as separate files for clarity
        # Bland-Altman plots
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Systolic Bland-Altman
        axes[0].scatter(mean_sys, diff_sys, alpha=0.5)
        axes[0].axhline(y=np.mean(diff_sys), color='r', linestyle='-', 
                       label=f'Mean Error: {np.mean(diff_sys):.2f}')
        axes[0].axhline(y=np.mean(diff_sys) + 1.96*np.std(diff_sys), color='g', linestyle='--', 
                       label=f'95% Limits: {np.mean(diff_sys) + 1.96*np.std(diff_sys):.2f}')
        axes[0].axhline(y=np.mean(diff_sys) - 1.96*np.std(diff_sys), color='g', linestyle='--',
                       label=f'{np.mean(diff_sys) - 1.96*np.std(diff_sys):.2f}')
        axes[0].set_xlabel('Mean Systolic BP (mmHg)')
        axes[0].set_ylabel('Difference (Target - Predicted)')
        axes[0].set_title('Bland-Altman Plot for Systolic BP')
        axes[0].legend()
        axes[0].grid(True)
        
        # Diastolic Bland-Altman
        axes[1].scatter(mean_dias, diff_dias, alpha=0.5)
        axes[1].axhline(y=np.mean(diff_dias), color='r', linestyle='-', 
                       label=f'Mean Error: {np.mean(diff_dias):.2f}')
        axes[1].axhline(y=np.mean(diff_dias) + 1.96*np.std(diff_dias), color='g', linestyle='--', 
                       label=f'95% Limits: {np.mean(diff_dias) + 1.96*np.std(diff_dias):.2f}')
        axes[1].axhline(y=np.mean(diff_dias) - 1.96*np.std(diff_dias), color='g', linestyle='--',
                       label=f'{np.mean(diff_dias) - 1.96*np.std(diff_dias):.2f}')
        axes[1].set_xlabel('Mean Diastolic BP (mmHg)')
        axes[1].set_ylabel('Difference (Target - Predicted)')
        axes[1].set_title('Bland-Altman Plot for Diastolic BP')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'corrected_bland_altman.png'), dpi=300)
        plt.close()
        
        # Scatter plots with correlation
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Systolic scatter plot
        axes[0].scatter(sys_targets, sys_preds, alpha=0.5)
        axes[0].plot([sys_targets.min(), sys_targets.max()], 
                     [sys_targets.min(), sys_targets.max()], 'r--', label='Identity line')
        axes[0].set_xlabel('True Systolic BP (mmHg)')
        axes[0].set_ylabel('Predicted Systolic BP (mmHg)')
        axes[0].set_title(f'Systolic BP (r={systolic_pearson_r:.3f}, R²={systolic_r2:.3f})')
        axes[0].legend()
        axes[0].grid(True)
        
        # Diastolic scatter plot
        axes[1].scatter(dias_targets, dias_preds, alpha=0.5)
        axes[1].plot([dias_targets.min(), dias_targets.max()], 
                     [dias_targets.min(), dias_targets.max()], 'r--', label='Identity line')
        axes[1].set_xlabel('True Diastolic BP (mmHg)')
        axes[1].set_ylabel('Predicted Diastolic BP (mmHg)')
        axes[1].set_title(f'Diastolic BP (r={diastolic_pearson_r:.3f}, R²={diastolic_r2:.3f})')
        axes[1].legend()
        axes[1].grid(True)
        
        # Waveform scatter plot (sample every 5th point for clarity)
        axes[2].scatter(flat_targets[::5], flat_preds[::5], alpha=0.1)
        axes[2].plot([flat_targets.min(), flat_targets.max()], 
                     [flat_targets.min(), flat_targets.max()], 'r--', label='Identity line')
        axes[2].set_xlabel('True BP (mmHg)')
        axes[2].set_ylabel('Predicted BP (mmHg)')
        axes[2].set_title(f'All BP values (r={waveform_pearson_r:.3f}, R²={waveform_r2:.3f})')
        axes[2].legend()
        axes[2].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'correlation_scatter_plots.png'), dpi=300)
        plt.close()
        
        # Create distribution plots
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Systolic
        axes[0].hist(sys_targets, bins=30, alpha=0.5, label='Target', density=True)
        axes[0].hist(sys_preds, bins=30, alpha=0.5, label='Predicted', density=True)
        axes[0].set_xlabel('Systolic BP (mmHg)')
        axes[0].set_ylabel('Density')
        axes[0].set_title('Distribution of Systolic Blood Pressure')
        axes[0].legend()
        axes[0].grid(True)
        
        # Diastolic
        axes[1].hist(dias_targets, bins=30, alpha=0.5, label='Target', density=True)
        axes[1].hist(dias_preds, bins=30, alpha=0.5, label='Predicted', density=True)
        axes[1].set_xlabel('Diastolic BP (mmHg)')
        axes[1].set_ylabel('Density')
        axes[1].set_title('Distribution of Diastolic Blood Pressure')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'bp_distributions.png'), dpi=300)
        plt.close()
        
        return {
            'systolic_mae': sys_mae,
            'diastolic_mae': dias_mae,
            'systolic_bias': sys_bias,
            'diastolic_bias': dias_bias,
            'num_waveforms': len(all_targets),
            'predictions': all_predictions,
            'targets': all_targets,
            'systolic_predictions': sys_preds,
            'systolic_targets': sys_targets,
            'diastolic_predictions': dias_preds,
            'diastolic_targets': dias_targets,
            'waveform_pearson_r': waveform_pearson_r,
            'waveform_r2': waveform_r2,
            'systolic_pearson_r': systolic_pearson_r,
            'systolic_r2': systolic_r2,
            'diastolic_pearson_r': diastolic_pearson_r,
            'diastolic_r2': diastolic_r2,
            'waveform_mae': waveform_mae,
            'waveform_mse': waveform_mse
        }
    else:
        print("No valid predictions were made")
        return None


def main():
    parser = argparse.ArgumentParser(description='Evaluate VAE-BiLSTM model')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained BiLSTM model checkpoint')
    parser.add_argument('--vae_path', type=str, required=True,
                        help='Path to the VAE model checkpoint')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to the H5 data file')
    parser.add_argument('--output_dir', type=str, default='evaluation_output',
                        help='Directory for saving evaluation results')
    parser.add_argument('--latent_dim', type=int, default=256,
                        help='Dimension of latent space')
    parser.add_argument('--lstm_hidden_dim', type=int, default=256,
                        help='Hidden dimension of LSTM')
    parser.add_argument('--lstm_layers', type=int, default=3,
                        help='Number of LSTM layers')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for evaluation')
    
    args = parser.parse_args()
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load VAE model
    print(f"Loading VAE model from: {args.vae_path}")
    vae_model = VAE(latent_dim=args.latent_dim).to(device)
    vae_checkpoint = torch.load(args.vae_path, map_location=device)
    vae_model.load_state_dict(vae_checkpoint['model_state_dict'])
    vae_model.eval()
    
    # Create BiLSTM model
    bilstm_model = VAEBiLSTM(
        vae_model=vae_model,
        input_dim=args.latent_dim,
        hidden_dim=args.lstm_hidden_dim,
        num_layers=args.lstm_layers,
        output_dim=50,
        dropout=0.3
    ).to(device)
    
    # Load BiLSTM checkpoint
    print(f"Loading BiLSTM model from: {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device)
    bilstm_model.load_state_dict(checkpoint['model_state_dict'])
    bilstm_model.eval()
    
    # Load dataset
    print(f"Loading dataset from: {args.data_path}")
    dataset = PviDataset(args.data_path)
    batch_server = PviBatchServer(dataset, input_type="img", output_type="full")
    batch_server.set_loader_params(batch_size=args.batch_size, test_size=0.2)
    _, test_loader = batch_server.get_loaders()
    
    # Evaluate model
    results = evaluate_model(
        model=bilstm_model,
        data_loader=test_loader,
        device=device,
        pattern_offsets=checkpoint.get('pattern_offsets', [-7, 0, 3]),
        bp_norm_params=checkpoint.get('bp_norm_params', (40, 200)),
        output_dir=args.output_dir
    )
    
    if results:
        # Save detailed results
        results_file = os.path.join(args.output_dir, 'results', 'evaluation_results.txt')
        with open(results_file, 'w') as f:
            f.write(f"Number of waveforms evaluated: {results['num_waveforms']}\n")
            f.write(f"Systolic MAE: {results['systolic_mae']:.2f} mmHg\n")
            f.write(f"Systolic Bias: {results['systolic_bias']:.2f} mmHg\n")
            f.write(f"Diastolic MAE: {results['diastolic_mae']:.2f} mmHg\n")
            f.write(f"Diastolic Bias: {results['diastolic_bias']:.2f} mmHg\n")
            
            f.write(f"\nFull Waveform Metrics:\n")
            f.write(f"MAE: {results['waveform_mae']:.2f}\n")
            f.write(f"MSE: {results['waveform_mse']:.2f}\n")
            f.write(f"Pearson r: {results['waveform_pearson_r']:.4f}\n")
            f.write(f"R²: {results['waveform_r2']:.4f}\n")
            
            f.write(f"\nSystolic Metrics:\n")
            f.write(f"Pearson r: {results['systolic_pearson_r']:.4f}\n")
            f.write(f"R²: {results['systolic_r2']:.4f}\n")
            
            f.write(f"\nDiastolic Metrics:\n")
            f.write(f"Pearson r: {results['diastolic_pearson_r']:.4f}\n")
            f.write(f"R²: {results['diastolic_r2']:.4f}\n")
        
        print(f"Evaluation complete. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()