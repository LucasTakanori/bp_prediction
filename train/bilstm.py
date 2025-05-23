import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import sys
import numpy as np
import argparse
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import wandb

# Add the parent directory to the path to find utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Import your dataset utilities
from utils.data_utils import PviDataset, PviBatchServer

# Define the VAE model with improved capacity
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


# Attention mechanism for temporal sequences
class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim, attention_dim=128):
        super(TemporalAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.attention_dim = attention_dim
        
        # Attention layers
        self.query = nn.Linear(hidden_dim, attention_dim)
        self.key = nn.Linear(hidden_dim, attention_dim)
        self.value = nn.Linear(hidden_dim, attention_dim)
        self.out = nn.Linear(attention_dim, hidden_dim)
        
        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x):
        """
        x: [batch_size, seq_len, hidden_dim]
        """
        batch_size, seq_len, hidden_dim = x.shape
        
        # Generate Q, K, V
        Q = self.query(x)  # [batch_size, seq_len, attention_dim]
        K = self.key(x)    # [batch_size, seq_len, attention_dim]
        V = self.value(x)  # [batch_size, seq_len, attention_dim]
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.attention_dim)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attended = torch.matmul(attention_weights, V)  # [batch_size, seq_len, attention_dim]
        
        # Project back to hidden dimension
        attended = self.out(attended)  # [batch_size, seq_len, hidden_dim]
        
        # Residual connection and layer norm
        output = self.layer_norm(x + attended)
        
        return output, attention_weights


# Enhanced Bidirectional LSTM model with Attention for blood pressure prediction
class VAEBiLSTMWithAttention(nn.Module):
    def __init__(self, vae_model, input_dim=256, hidden_dim=256, num_layers=2, 
                 output_dim=50, dropout=0.3, use_attention=True, attention_dim=128):
        super(VAEBiLSTMWithAttention, self).__init__()
        
        # Store the pretrained VAE
        self.vae = vae_model
        self.use_attention = use_attention
        
        # Freeze the VAE parameters
        for param in self.vae.parameters():
            param.requires_grad = False
        
        # Input projection layer to enhance features
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.input_dropout = nn.Dropout(dropout * 0.5)
        
        # Bidirectional LSTM for sequence modeling
        self.lstm = nn.LSTM(
            input_size=hidden_dim,    # Use projected features
            hidden_size=hidden_dim,   # Size of hidden state
            num_layers=num_layers,    # Number of LSTM layers
            batch_first=True,         # Input shape: (batch, seq, feature)
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True        # Use bidirectional LSTM
        )
        
        # Attention mechanism for temporal modeling
        if self.use_attention:
            self.attention = TemporalAttention(
                hidden_dim=hidden_dim * 2,  # bidirectional output
                attention_dim=attention_dim
            )
        
        # Enhanced output layers with time-domain processing
        self.temporal_conv = nn.Conv1d(
            in_channels=hidden_dim * 2, 
            out_channels=hidden_dim, 
            kernel_size=3, 
            padding=1
        )
        self.temporal_bn = nn.BatchNorm1d(hidden_dim)
        
        # Multi-head output processing
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc_out = nn.Linear(hidden_dim // 2, output_dim)
        
        # Additional prediction heads for systolic/diastolic
        self.systolic_head = nn.Linear(hidden_dim // 2, 1)
        self.diastolic_head = nn.Linear(hidden_dim // 2, 1)
        
    def forward(self, x_seq, return_attention=False):
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
        
        # Project input features
        projected_seq = self.input_projection(latent_seq)  # [batch_size, seq_len, hidden_dim]
        projected_seq = self.input_dropout(projected_seq)
        
        # Process with bidirectional LSTM
        lstm_out, _ = self.lstm(projected_seq)  # [batch_size, seq_len, 2*hidden_dim]
        
        # Apply attention mechanism if enabled
        attention_weights = None
        if self.use_attention:
            lstm_out, attention_weights = self.attention(lstm_out)
        
        # Temporal convolution for time-domain processing
        # Transpose for conv1d: [batch_size, channels, seq_len]
        conv_input = lstm_out.transpose(1, 2)  # [batch_size, 2*hidden_dim, seq_len]
        conv_out = self.temporal_conv(conv_input)  # [batch_size, hidden_dim, seq_len]
        conv_out = self.temporal_bn(conv_out)
        conv_out = F.relu(conv_out)
        
        # Global average pooling over time dimension
        pooled = torch.mean(conv_out, dim=2)  # [batch_size, hidden_dim]
        
        # Process through fully connected layers
        x = self.fc1(pooled)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.relu(x)
        
        # Main BP waveform output
        bp_waveform = self.fc_out(x)
        
        # Additional outputs for systolic and diastolic
        systolic_pred = self.systolic_head(x)
        diastolic_pred = self.diastolic_head(x)
        
        outputs = {
            'waveform': bp_waveform,
            'systolic': systolic_pred,
            'diastolic': diastolic_pred
        }
        
        if return_attention and attention_weights is not None:
            outputs['attention_weights'] = attention_weights
            
        return outputs


# Extract and normalize a specific frame from each sample
def extract_frame(batch_data, frame_idx=0):
    """Extract a specific frame from the batch data and normalize it to [0, 1]"""
    if isinstance(batch_data, dict):
        # If batch is a dictionary (like in your debug output)
        pvi_data = batch_data['pviHP']
    else:
        # If batch is just the tensor
        pvi_data = batch_data
    
    # Extract the specific frame from each sample
    # Shape: [batch_size, 1, 32, 32]
    frames = pvi_data[:, 0, :, :, frame_idx].unsqueeze(1)
    
    # Replace NaN values with zeros
    frames = torch.nan_to_num(frames, nan=0.0)
    
    # Normalize to [0, 1] range for each sample individually
    # First find min and max per sample
    batch_size = frames.shape[0]
    normalized_frames = torch.zeros_like(frames)
    
    for i in range(batch_size):
        sample = frames[i]
        min_val = torch.min(sample)
        max_val = torch.max(sample)
        
        # Handle case where min == max (constant value)
        if min_val == max_val:
            normalized_frames[i] = torch.zeros_like(sample)
        else:
            # Normalize to [0, 1]
            normalized_frames[i] = (sample - min_val) / (max_val - min_val)
    
    return normalized_frames


# Prepare 10-frame sequence data (k-7 to k+2)
def prepare_sequence_data(batch_data, central_indices=None, pattern_offsets=None):
    """
    Prepare data with 10-frame sequence pattern (k-7 to k+2) and corresponding BP values
    
    Args:
        batch_data: Dictionary with 'pviHP' and 'bp' keys
        central_indices: List of indices to use as the central frame k
                        If None, use all valid indices
        pattern_offsets: List of offsets relative to central frame k
                        Default: [-7, -6, -5, -4, -3, -2, -1, 0, 1, 2] for 10 frames
    
    Returns:
        frame_sequences: Tensor of shape [batch_size, len(pattern_offsets), 1, 32, 32]
        bp_targets: Tensor of shape [batch_size, 50] (BP signal)
    """
    if pattern_offsets is None:
        pattern_offsets = [-7, -6, -5, -4, -3, -2, -1, 0, 1, 2]  # 10 frames
    
    if not isinstance(batch_data, dict):
        raise ValueError("Expected batch_data to be a dictionary")
    
    pvi_data = batch_data['pviHP']  # [batch_size, 1, 32, 32, 500]
    bp_data = batch_data['bp']      # [batch_size, 50]
    
    batch_size = pvi_data.shape[0]
    total_frames = pvi_data.shape[-1]
    
    # Determine valid central indices
    min_offset = min(pattern_offsets)
    max_offset = max(pattern_offsets)
    valid_indices_start = max(0, -min_offset)  # Ensure we don't go below 0
    valid_indices_end = min(total_frames, total_frames - max_offset)  # Ensure we don't exceed total_frames
    
    # If central_indices not provided, use all valid indices
    if central_indices is None:
        central_indices = list(range(valid_indices_start, valid_indices_end))
    else:
        # Filter central_indices to keep only valid ones
        central_indices = [k for k in central_indices if valid_indices_start <= k < valid_indices_end]
    
    # Create sequences and targets
    frame_sequences = []
    bp_targets = []
    
    for b in range(batch_size):
        for k in central_indices:
            # Extract and normalize frames based on pattern
            seq = []
            for offset in pattern_offsets:
                frame_idx = k + offset
                frame = extract_frame(batch_data, frame_idx)[b:b+1]  # Keep batch dimension
                seq.append(frame)
            
            # Stack frames into a sequence
            seq = torch.cat(seq, dim=0)  # [len(pattern_offsets), 1, 32, 32]
            frame_sequences.append(seq)
            
            # Add corresponding BP target
            bp_targets.append(bp_data[b])
    
    # Stack all sequences and targets
    frame_sequences = torch.stack(frame_sequences)  # [num_sequences, len(pattern_offsets), 1, 32, 32]
    bp_targets = torch.stack(bp_targets)            # [num_sequences, 50]
    
    return frame_sequences, bp_targets


# Normalize blood pressure data
def normalize_bp(bp_data, min_val=40, max_val=200):
    """
    Normalize blood pressure data to [0, 1] range for better training
    """
    # Scale to [0, 1]
    bp_normalized = (bp_data - min_val) / (max_val - min_val)
    return bp_normalized, (min_val, max_val)


# Denormalize blood pressure data
def denormalize_bp(bp_normalized, min_val=40, max_val=200):
    """
    Convert normalized BP back to original scale
    """
    # Scale back to original range
    bp_denormalized = bp_normalized * (max_val - min_val) + min_val
    return bp_denormalized


# Enhanced loss functions with multiple options
class BPLossFunction:
    def __init__(self, loss_type='mse', alpha=0.6, beta=0.2, gamma=0.2):
        """
        Enhanced loss function for blood pressure prediction
        
        Args:
            loss_type: 'mse', 'systolic_distance', 'diastolic_distance', 'composite'
            alpha: Weight for waveform loss (in composite)
            beta: Weight for systolic loss (in composite)
            gamma: Weight for diastolic loss (in composite)
        """
        self.loss_type = loss_type
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
    
    def __call__(self, outputs, targets):
        """
        Calculate loss based on selected type
        
        Args:
            outputs: Dictionary with 'waveform', 'systolic', 'diastolic' keys
            targets: Ground truth BP waveforms [batch_size, 50]
        """
        # Extract predictions
        if isinstance(outputs, dict):
            y_pred = outputs['waveform']
            sys_pred = outputs.get('systolic', None)
            dias_pred = outputs.get('diastolic', None)
        else:
            # Backward compatibility
            y_pred = outputs
            sys_pred = None
            dias_pred = None
        
        y_true = targets
        
        # Calculate true systolic and diastolic values
        sys_true = torch.max(y_true, dim=1)[0].unsqueeze(1)  # [batch_size, 1]
        dias_true = torch.min(y_true, dim=1)[0].unsqueeze(1)  # [batch_size, 1]
        
        if self.loss_type == 'mse':
            # Standard MSE loss on waveform
            return F.mse_loss(y_pred, y_true)
        
        elif self.loss_type == 'systolic_distance':
            # Focus on systolic pressure prediction
            if sys_pred is not None:
                return F.mse_loss(sys_pred, sys_true)
            else:
                # Extract systolic from waveform prediction
                sys_pred_extracted = torch.max(y_pred, dim=1)[0].unsqueeze(1)
                return F.mse_loss(sys_pred_extracted, sys_true)
        
        elif self.loss_type == 'diastolic_distance':
            # Focus on diastolic pressure prediction
            if dias_pred is not None:
                return F.mse_loss(dias_pred, dias_true)
            else:
                # Extract diastolic from waveform prediction
                dias_pred_extracted = torch.min(y_pred, dim=1)[0].unsqueeze(1)
                return F.mse_loss(dias_pred_extracted, dias_true)
        
        elif self.loss_type == 'composite':
            # Composite loss combining all components
            # Base MSE for overall waveform
            mse_loss = F.mse_loss(y_pred, y_true)
            
            # Systolic loss
            if sys_pred is not None:
                sys_loss = F.mse_loss(sys_pred, sys_true)
            else:
                sys_pred_extracted = torch.max(y_pred, dim=1)[0].unsqueeze(1)
                sys_loss = F.mse_loss(sys_pred_extracted, sys_true)
            
            # Diastolic loss
            if dias_pred is not None:
                dias_loss = F.mse_loss(dias_pred, dias_true)
            else:
                dias_pred_extracted = torch.min(y_pred, dim=1)[0].unsqueeze(1)
                dias_loss = F.mse_loss(dias_pred_extracted, dias_true)
            
            # Combined loss
            total_loss = self.alpha * mse_loss + self.beta * sys_loss + self.gamma * dias_loss
            return total_loss
        
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")


# Train the enhanced BiLSTM model
def train_enhanced_bilstm(model, train_loader, val_batches, device, num_epochs=20, 
                         pattern_offsets=None, bp_norm_params=(40, 200), 
                         loss_type='composite', output_dir=None):
    """Train the enhanced BiLSTM model for blood pressure prediction"""
    
    if pattern_offsets is None:
        pattern_offsets = [-7, -6, -5, -4, -3, -2, -1, 0, 1, 2]  # 10 frames
    
    # Create directories for saving results
    checkpoints_dir = os.path.join(output_dir, 'checkpoints')
    results_dir = os.path.join(output_dir, 'results')
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # Unpack normalization parameters
    bp_min, bp_max = bp_norm_params
    
    # Set up optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = BPLossFunction(loss_type=loss_type)
    
    # Use CosineAnnealingLR for smooth learning rate decay
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=num_epochs,
        eta_min=1e-6
    )
    
    # Track best model
    best_val_loss = float('inf')
    best_epoch = 0
    
    # For tracking metrics
    train_losses = []
    val_losses = []
    learning_rates = []
    
    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = 0
        sample_count = 0
        
        # Current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)
        
        # Training loop
        for batch_idx, batch_data in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
            try:
                # Prepare sequences with 10-frame pattern
                sequences, targets = prepare_sequence_data(batch_data, pattern_offsets=pattern_offsets)
                
                # Normalize targets
                targets_norm, _ = normalize_bp(targets, bp_min, bp_max)
                
                # Move to device
                sequences = sequences.to(device)
                targets_norm = targets_norm.to(device)
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(sequences)
                
                # Calculate loss
                if isinstance(outputs, dict):
                    # Normalize additional outputs if they exist
                    outputs_norm = outputs.copy()
                    outputs_norm['waveform'] = outputs['waveform']
                    if 'systolic' in outputs:
                        outputs_norm['systolic'] = (outputs['systolic'] - bp_min) / (bp_max - bp_min)
                    if 'diastolic' in outputs:
                        outputs_norm['diastolic'] = (outputs['diastolic'] - bp_min) / (bp_max - bp_min)
                else:
                    outputs_norm = outputs
                
                loss = criterion(outputs_norm, targets_norm)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping to prevent explosion
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # Update weights
                optimizer.step()
                
                # Accumulate loss and count
                train_loss += loss.item() * sequences.size(0)
                sample_count += sequences.size(0)
                
                # Log batch-level metrics to wandb
                if batch_idx % 2 == 0:  # Log every 2 batches
                    wandb.log({
                        "batch_loss": loss.item(),
                        "batch": epoch * len(train_loader) + batch_idx
                    })
                
            except Exception as e:
                print(f"Error processing batch {batch_idx}: {e}")
                continue
        
        # Calculate average loss
        if sample_count > 0:
            train_loss /= sample_count
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        val_sample_count = 0
        
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(tqdm(val_batches, desc="Validation")):
                try:
                    # Prepare sequences with 10-frame pattern
                    sequences, targets = prepare_sequence_data(batch_data, pattern_offsets=pattern_offsets)
                    
                    # Normalize targets
                    targets_norm, _ = normalize_bp(targets, bp_min, bp_max)
                    
                    # Move to device
                    sequences = sequences.to(device)
                    targets_norm = targets_norm.to(device)
                    
                    # Forward pass
                    outputs = model(sequences)
                    
                    # Calculate loss
                    if isinstance(outputs, dict):
                        # Normalize additional outputs if they exist
                        outputs_norm = outputs.copy()
                        outputs_norm['waveform'] = outputs['waveform']
                        if 'systolic' in outputs:
                            outputs_norm['systolic'] = (outputs['systolic'] - bp_min) / (bp_max - bp_min)
                        if 'diastolic' in outputs:
                            outputs_norm['diastolic'] = (outputs['diastolic'] - bp_min) / (bp_max - bp_min)
                    else:
                        outputs_norm = outputs
                    
                    loss = criterion(outputs_norm, targets_norm)
                    
                    # Accumulate loss and count
                    val_loss += loss.item() * sequences.size(0)
                    val_sample_count += sequences.size(0)
                    
                except Exception as e:
                    print(f"Error processing validation batch {batch_idx}: {e}")
                    continue
        
        # Calculate average validation loss
        if val_sample_count > 0:
            val_loss /= val_sample_count
        val_losses.append(val_loss)
        
        # Update scheduler
        scheduler.step()
        
        # Print progress
        print(f"Epoch {epoch}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}, LR = {current_lr:.6f}")
        
        # Log epoch-level metrics to wandb
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "learning_rate": current_lr
        })
        
        # Check if this is the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'bp_norm_params': bp_norm_params,
                'pattern_offsets': pattern_offsets,
                'loss_type': loss_type
            }, os.path.join(checkpoints_dir, 'enhanced_bilstm_best.pt'))
            
            print(f"New best model at epoch {epoch} with validation loss {val_loss:.6f}")
            
            # Log best model info to wandb
            wandb.run.summary["best_epoch"] = best_epoch
            wandb.run.summary["best_val_loss"] = best_val_loss
        
        # Save checkpoint every 5 epochs
        if epoch % 5 == 0 or epoch == num_epochs:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'bp_norm_params': bp_norm_params,
                'pattern_offsets': pattern_offsets,
                'loss_type': loss_type
            }, os.path.join(checkpoints_dir, f'enhanced_bilstm_epoch_{epoch}.pt'))
    
    # Plot training curves
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Enhanced BiLSTM Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot(range(1, num_epochs + 1), learning_rates)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate vs. Epoch')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'enhanced_bilstm_training_curves.png'), dpi=150)
    
    # Log training curves to wandb
    wandb.log({"training_curves": wandb.Image(plt)})
    
    plt.close()
    
    print(f"Training completed. Best model at epoch {best_epoch} with validation loss {best_val_loss:.6f}")
    
    return best_epoch, best_val_loss


# Evaluate the enhanced BiLSTM model
def evaluate_enhanced_bilstm(model, test_loader, device, pattern_offsets=None, 
                           bp_norm_params=(40, 200), output_dir=None, visualize_attention=False):
    """Evaluate the enhanced BiLSTM model on test data"""
    
    if pattern_offsets is None:
        pattern_offsets = [-7, -6, -5, -4, -3, -2, -1, 0, 1, 2]  # 10 frames
    
    results_dir = os.path.join(output_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Unpack normalization parameters
    bp_min, bp_max = bp_norm_params
    
    model.eval()
    criterion = nn.MSELoss()
    
    all_targets = []
    all_predictions = []
    all_attention_weights = []
    test_loss = 0
    sample_count = 0
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(tqdm(test_loader, desc="Testing")):
            try:
                # Prepare sequences with 10-frame pattern
                sequences, targets = prepare_sequence_data(batch_data, pattern_offsets=pattern_offsets)
                
                # Normalize targets
                targets_norm, _ = normalize_bp(targets, bp_min, bp_max)
                
                # Move to device
                sequences = sequences.to(device)
                targets_norm = targets_norm.to(device)
                
                # Forward pass with attention visualization
                outputs = model(sequences, return_attention=visualize_attention)
                
                # Extract waveform predictions
                if isinstance(outputs, dict):
                    outputs_norm = outputs['waveform']
                    if visualize_attention and 'attention_weights' in outputs:
                        all_attention_weights.append(outputs['attention_weights'].cpu().numpy())
                else:
                    outputs_norm = outputs
                
                # Calculate loss
                loss = criterion(outputs_norm, targets_norm)
                test_loss += loss.item() * sequences.size(0)
                sample_count += sequences.size(0)
                
                # Denormalize predictions for metrics calculation
                outputs_denorm = denormalize_bp(outputs_norm.cpu(), bp_min, bp_max)
                
                # Store predictions and targets
                all_predictions.append(outputs_denorm.numpy())
                all_targets.append(targets.cpu().numpy())
                
            except Exception as e:
                print(f"Error processing test batch {batch_idx}: {e}")
                continue
    
    # Calculate average test loss
    if sample_count > 0:
        test_loss /= sample_count
    
    # Concatenate all predictions and targets
    if all_predictions and all_targets:
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        
        # Calculate metrics
        mse = mean_squared_error(all_targets, all_predictions)
        mae = mean_absolute_error(all_targets, all_predictions)
        r2 = r2_score(all_targets, all_predictions)
        
        print(f"Test Loss (normalized): {test_loss:.6f}")
        print(f"MSE: {mse:.6f}")
        print(f"MAE: {mae:.6f}")
        print(f"R²: {r2:.6f}")
        
        # Log test metrics to wandb
        wandb.log({
            "test_loss": test_loss,
            "test_mse": mse,
            "test_mae": mae,
            "test_r2": r2
        })
        
        # Plot example predictions
        plot_enhanced_predictions(all_predictions, all_targets, output_dir)
        
        # Calculate systolic and diastolic metrics separately
        sys_mae, sys_error, dias_mae, dias_error = calculate_bp_metrics_enhanced(
            all_predictions, all_targets, output_dir
        )
        
        print(f"Systolic MAE: {sys_mae:.2f} mmHg (mean error: {sys_error:.2f})")
        print(f"Diastolic MAE: {dias_mae:.2f} mmHg (mean error: {dias_error:.2f})")
        
        # Log BP metrics to wandb
        wandb.log({
            "systolic_mae": sys_mae,
            "systolic_error": sys_error,
            "diastolic_mae": dias_mae,
            "diastolic_error": dias_error
        })
        
        # Visualize attention weights if available
        if visualize_attention and all_attention_weights:
            visualize_attention_patterns(all_attention_weights, pattern_offsets, output_dir)
        
        return test_loss, mse, mae, r2, sys_mae, dias_mae, sys_error, dias_error
    else:
        print("No valid predictions were made during evaluation")
        return test_loss, None, None, None, None, None, None, None


# Enhanced plotting functions
def plot_enhanced_predictions(predictions, targets, output_dir, num_examples=5):
    """Plot enhanced BP predictions vs targets with more detailed analysis"""
    
    results_dir = os.path.join(output_dir, 'results')
    
    # Sample a few random examples
    indices = np.random.choice(len(predictions), min(num_examples, len(predictions)), replace=False)
    
    # Create a figure
    fig, axes = plt.subplots(len(indices), 1, figsize=(12, 4*len(indices)))
    if len(indices) == 1:
        axes = [axes]
    
    for i, idx in enumerate(indices):
        # Get prediction and target
        pred = predictions[idx]
        target = targets[idx]
        
        # Calculate metrics for this example
        mse_sample = np.mean((pred - target) ** 2)
        mae_sample = np.mean(np.abs(pred - target))
        
        # Extract systolic and diastolic
        sys_pred = np.max(pred)
        sys_true = np.max(target)
        dias_pred = np.min(pred)
        dias_true = np.min(target)
        
        # Plot
        time_points = np.arange(len(target))
        axes[i].plot(time_points, target, label='Target BP', color='blue', linewidth=2)
        axes[i].plot(time_points, pred, label='Predicted BP', color='red', linestyle='--', linewidth=2)
        
        # Mark systolic and diastolic points
        sys_idx_true = np.argmax(target)
        dias_idx_true = np.argmin(target)
        sys_idx_pred = np.argmax(pred)
        dias_idx_pred = np.argmin(pred)
        
        axes[i].scatter(sys_idx_true, sys_true, color='blue', s=100, marker='^', label=f'Sys True: {sys_true:.1f}')
        axes[i].scatter(dias_idx_true, dias_true, color='blue', s=100, marker='v', label=f'Dias True: {dias_true:.1f}')
        axes[i].scatter(sys_idx_pred, sys_pred, color='red', s=100, marker='^', label=f'Sys Pred: {sys_pred:.1f}')
        axes[i].scatter(dias_idx_pred, dias_pred, color='red', s=100, marker='v', label=f'Dias Pred: {dias_pred:.1f}')
        
        axes[i].set_title(f'Example {i+1} - MSE: {mse_sample:.2f}, MAE: {mae_sample:.2f}')
        axes[i].set_xlabel('Time Points')
        axes[i].set_ylabel('Blood Pressure (mmHg)')
        axes[i].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'enhanced_bp_predictions.png'), dpi=150, bbox_inches='tight')
    
    # Log prediction examples to wandb
    wandb.log({"enhanced_prediction_examples": wandb.Image(plt)})
    
    plt.close()


def calculate_bp_metrics_enhanced(predictions, targets, output_dir):
    """Enhanced calculation of systolic and diastolic metrics"""
    
    results_dir = os.path.join(output_dir, 'results')
    
    # For each waveform, get systolic (max) and diastolic (min)
    sys_targets = np.max(targets, axis=1)
    dias_targets = np.min(targets, axis=1)
    sys_preds = np.max(predictions, axis=1)
    dias_preds = np.min(predictions, axis=1)
    
    # Calculate MAE for systolic and diastolic
    sys_mae = np.mean(np.abs(sys_targets - sys_preds))
    dias_mae = np.mean(np.abs(dias_targets - dias_preds))
    
    # Calculate mean error (for bias assessment)
    sys_error = np.mean(sys_targets - sys_preds)
    dias_error = np.mean(dias_targets - dias_preds)
    
    # Log detailed metrics to wandb
    wandb.log({
        "systolic_mae_detailed": sys_mae,
        "systolic_mean_error": sys_error,
        "diastolic_mae_detailed": dias_mae,
        "diastolic_mean_error": dias_error,
        "num_waveforms": len(targets)
    })
    
    # Enhanced Bland-Altman plot with additional statistics
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Systolic Bland-Altman
    mean_sys = (sys_targets + sys_preds) / 2
    diff_sys = sys_targets - sys_preds
    
    axes[0, 0].scatter(mean_sys, diff_sys, alpha=0.6)
    axes[0, 0].axhline(y=np.mean(diff_sys), color='r', linestyle='-', 
                      label=f'Mean Error: {np.mean(diff_sys):.2f}')
    axes[0, 0].axhline(y=np.mean(diff_sys) + 1.96*np.std(diff_sys), color='g', linestyle='--', 
                      label=f'+1.96σ: {np.mean(diff_sys) + 1.96*np.std(diff_sys):.2f}')
    axes[0, 0].axhline(y=np.mean(diff_sys) - 1.96*np.std(diff_sys), color='g', linestyle='--',
                      label=f'-1.96σ: {np.mean(diff_sys) - 1.96*np.std(diff_sys):.2f}')
    axes[0, 0].set_xlabel('Mean Systolic BP (mmHg)')
    axes[0, 0].set_ylabel('Difference (Target - Predicted)')
    axes[0, 0].set_title('Bland-Altman Plot for Systolic BP')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Diastolic Bland-Altman
    mean_dias = (dias_targets + dias_preds) / 2
    diff_dias = dias_targets - dias_preds
    
    axes[0, 1].scatter(mean_dias, diff_dias, alpha=0.6)
    axes[0, 1].axhline(y=np.mean(diff_dias), color='r', linestyle='-', 
                      label=f'Mean Error: {np.mean(diff_dias):.2f}')
    axes[0, 1].axhline(y=np.mean(diff_dias) + 1.96*np.std(diff_dias), color='g', linestyle='--', 
                      label=f'+1.96σ: {np.mean(diff_dias) + 1.96*np.std(diff_dias):.2f}')
    axes[0, 1].axhline(y=np.mean(diff_dias) - 1.96*np.std(diff_dias), color='g', linestyle='--',
                      label=f'-1.96σ: {np.mean(diff_dias) - 1.96*np.std(diff_dias):.2f}')
    axes[0, 1].set_xlabel('Mean Diastolic BP (mmHg)')
    axes[0, 1].set_ylabel('Difference (Target - Predicted)')
    axes[0, 1].set_title('Bland-Altman Plot for Diastolic BP')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Systolic scatter plot
    axes[1, 0].scatter(sys_targets, sys_preds, alpha=0.6)
    min_sys = min(np.min(sys_targets), np.min(sys_preds))
    max_sys = max(np.max(sys_targets), np.max(sys_preds))
    axes[1, 0].plot([min_sys, max_sys], [min_sys, max_sys], 'r--', label='Perfect Agreement')
    
    # Calculate R² for systolic
    sys_r2 = r2_score(sys_targets, sys_preds)
    axes[1, 0].set_xlabel('True Systolic BP (mmHg)')
    axes[1, 0].set_ylabel('Predicted Systolic BP (mmHg)')
    axes[1, 0].set_title(f'Systolic BP Prediction (R² = {sys_r2:.3f})')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Diastolic scatter plot
    axes[1, 1].scatter(dias_targets, dias_preds, alpha=0.6)
    min_dias = min(np.min(dias_targets), np.min(dias_preds))
    max_dias = max(np.max(dias_targets), np.max(dias_preds))
    axes[1, 1].plot([min_dias, max_dias], [min_dias, max_dias], 'r--', label='Perfect Agreement')
    
    # Calculate R² for diastolic
    dias_r2 = r2_score(dias_targets, dias_preds)
    axes[1, 1].set_xlabel('True Diastolic BP (mmHg)')
    axes[1, 1].set_ylabel('Predicted Diastolic BP (mmHg)')
    axes[1, 1].set_title(f'Diastolic BP Prediction (R² = {dias_r2:.3f})')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'enhanced_bp_analysis.png'), dpi=150)
    
    # Log enhanced analysis plots to wandb
    wandb.log({"enhanced_bp_analysis": wandb.Image(plt)})
    
    plt.close()
    
    return sys_mae, sys_error, dias_mae, dias_error


def visualize_attention_patterns(attention_weights_list, pattern_offsets, output_dir):
    """Visualize attention patterns across temporal sequences"""
    
    results_dir = os.path.join(output_dir, 'results')
    
    # Concatenate all attention weights
    all_attention = np.concatenate(attention_weights_list, axis=0)  # [num_samples, seq_len, seq_len]
    
    # Average attention across all samples
    avg_attention = np.mean(all_attention, axis=0)  # [seq_len, seq_len]
    
    # Create frame labels based on offsets
    frame_labels = [f'k{offset:+d}' if offset != 0 else 'k' for offset in pattern_offsets]
    
    # Plot attention heatmap
    plt.figure(figsize=(12, 10))
    
    # Main attention heatmap
    plt.subplot(2, 2, 1)
    im = plt.imshow(avg_attention, cmap='viridis', aspect='auto')
    plt.colorbar(im, label='Attention Weight')
    plt.xticks(range(len(frame_labels)), frame_labels, rotation=45)
    plt.yticks(range(len(frame_labels)), frame_labels)
    plt.xlabel('Key Frames')
    plt.ylabel('Query Frames')
    plt.title('Average Attention Patterns')
    
    # Attention weights for each time step (row-wise average)
    plt.subplot(2, 2, 2)
    row_avg_attention = np.mean(avg_attention, axis=1)
    plt.bar(range(len(frame_labels)), row_avg_attention)
    plt.xticks(range(len(frame_labels)), frame_labels, rotation=45)
    plt.xlabel('Frame Position')
    plt.ylabel('Average Attention Weight')
    plt.title('Average Attention per Frame Position')
    plt.grid(True, alpha=0.3)
    
    # Attention weights for each time step (column-wise average)
    plt.subplot(2, 2, 3)
    col_avg_attention = np.mean(avg_attention, axis=0)
    plt.bar(range(len(frame_labels)), col_avg_attention)
    plt.xticks(range(len(frame_labels)), frame_labels, rotation=45)
    plt.xlabel('Frame Position')
    plt.ylabel('Average Attention Weight')
    plt.title('Average Attention Received per Frame')
    plt.grid(True, alpha=0.3)
    
    # Diagonal attention (self-attention)
    plt.subplot(2, 2, 4)
    diagonal_attention = np.diag(avg_attention)
    plt.bar(range(len(frame_labels)), diagonal_attention)
    plt.xticks(range(len(frame_labels)), frame_labels, rotation=45)
    plt.xlabel('Frame Position')
    plt.ylabel('Self-Attention Weight')
    plt.title('Self-Attention per Frame')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'attention_patterns.png'), dpi=150, bbox_inches='tight')
    
    # Log attention visualization to wandb
    wandb.log({"attention_patterns": wandb.Image(plt)})
    
    plt.close()
    
    # Log attention statistics
    wandb.log({
        "avg_self_attention": np.mean(diagonal_attention),
        "max_attention_weight": np.max(avg_attention),
        "min_attention_weight": np.min(avg_attention),
        "attention_entropy": -np.sum(avg_attention * np.log(avg_attention + 1e-8), axis=1).mean()
    })


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Train and evaluate Enhanced VAE-BiLSTM for BP prediction')
    parser.add_argument('--output_dir', type=str, default='enhanced_vae_bilstm_output',
                        help='Directory for saving output files')
    parser.add_argument('--data_path', type=str, 
                        default=os.path.expanduser("~/phd/data/subject001_baseline_masked.h5"),
                        help='Path to the H5 data file')
    parser.add_argument('--vae_checkpoint', type=str, default=None,
                        help='Path to VAE checkpoint (if not provided, will search in default location)')
    parser.add_argument('--latent_dim', type=int, default=256,
                        help='Dimension of latent space')
    parser.add_argument('--lstm_hidden_dim', type=int, default=256,
                        help='Hidden dimension of LSTM')
    parser.add_argument('--lstm_layers', type=int, default=3,
                        help='Number of LSTM layers')
    parser.add_argument('--num_epochs', type=int, default=25,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training')
    parser.add_argument('--use_attention', action='store_true', default=True,
                        help='Use attention mechanism')
    parser.add_argument('--attention_dim', type=int, default=128,
                        help='Attention mechanism dimension')
    parser.add_argument('--loss_type', type=str, default='composite',
                        choices=['mse', 'systolic_distance', 'diastolic_distance', 'composite'],
                        help='Type of loss function to use')
    parser.add_argument('--visualize_attention', action='store_true', default=False,
                        help='Visualize attention patterns during evaluation')
    parser.add_argument('--wandb_project', type=str, default='enhanced-vae-bilstm-bp',
                        help='Weights & Biases project name')
    parser.add_argument('--wandb_name', type=str, default=None,
                        help='Weights & Biases run name')
    parser.add_argument('--wandb_mode', type=str, default='offline',
                        choices=['online', 'offline', 'disabled'],
                        help='Weights & Biases mode')
    
    args = parser.parse_args()
    
    # 10-frame pattern from k-7 to k+2
    pattern_offsets = [-7, -6, -5, -4, -3, -2, -1, 0, 1, 2]
    
    # Initialize wandb
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_name,
        config={
            "latent_dim": args.latent_dim,
            "lstm_hidden_dim": args.lstm_hidden_dim,
            "lstm_layers": args.lstm_layers,
            "num_epochs": args.num_epochs,
            "batch_size": args.batch_size,
            "use_attention": args.use_attention,
            "attention_dim": args.attention_dim,
            "pattern_offsets": pattern_offsets,
            "bp_norm_params": (40, 200),
            "learning_rate": 1e-3,
            "weight_decay": 1e-5,
            "dropout": 0.3,
            "loss_type": args.loss_type,
            "data_path": args.data_path,
            "output_dir": args.output_dir
        },
        mode=args.wandb_mode
    )
    
    # Create output directory and subdirectories
    os.makedirs(args.output_dir, exist_ok=True)
    checkpoints_dir = os.path.join(args.output_dir, 'checkpoints')
    results_dir = os.path.join(args.output_dir, 'results')
    
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # If VAE checkpoint path not provided, look in default location
    if args.vae_checkpoint is None:
        default_vae_path = os.path.join(args.output_dir, '../vae_output/checkpoints/vae_best.pt')
        if os.path.exists(default_vae_path):
            vae_checkpoint_path = default_vae_path
        else:
            # Try another common location
            alt_vae_path = "checkpoints/vae_best.pt"
            if os.path.exists(alt_vae_path):
                vae_checkpoint_path = alt_vae_path
            else:
                raise ValueError("No VAE checkpoint found. Please specify using --vae_checkpoint")
    else:
        vae_checkpoint_path = args.vae_checkpoint
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    wandb.config.update({"device": str(device)})
    
    # Set hyperparameters
    latent_dim = args.latent_dim
    lstm_hidden_dim = args.lstm_hidden_dim
    lstm_num_layers = args.lstm_layers
    num_epochs = args.num_epochs
    
    # Blood pressure normalization range
    bp_norm_params = (40, 200)  # Expected range: 40-200 mmHg
    
    print(f"Using 10-frame pattern with offsets: {pattern_offsets}")
    print(f"Loss function type: {args.loss_type}")
    print(f"Using attention: {args.use_attention}")
    
    # Load dataset
    print(f"Loading dataset from: {args.data_path}")
    dataset = PviDataset(args.data_path)
    print(f"Dataset loaded with {len(dataset)} samples")
    wandb.config.update({"dataset_size": len(dataset)})
    
    # Create batch server
    batch_server = PviBatchServer(dataset, input_type="img", output_type="full")
    
    # Set batch size using the correct method
    batch_server.set_loader_params(batch_size=args.batch_size, test_size=0.3)  # Use 30% for test+val
    
    # Load the pretrained VAE
    print(f"Loading pretrained VAE from: {vae_checkpoint_path}")
    vae_model = VAE(latent_dim=latent_dim).to(device)
    
    try:
        checkpoint = torch.load(vae_checkpoint_path, map_location=device)
        vae_model.load_state_dict(checkpoint['model_state_dict'])
        print(f"VAE loaded successfully from epoch {checkpoint.get('epoch', 'unknown')}")
    except Exception as e:
        print(f"Error loading VAE: {e}")
        print("Initializing VAE from scratch")
    
    # Set VAE to evaluation mode
    vae_model.eval()
    
    # Get loaders
    train_loader, test_val_loader = batch_server.get_loaders()
    
    # Split test_val_loader into validation and test sets
    test_val_batches = list(test_val_loader)
    val_size = len(test_val_batches) // 3  # 1/3 of test_val for validation, 2/3 for testing
    val_batches = test_val_batches[:val_size]
    test_batches = test_val_batches[val_size:]
    
    print(f"Train batches: {len(train_loader)}, Validation batches: {len(val_batches)}, Test batches: {len(test_batches)}")
    
    # Create Enhanced BiLSTM model with attention
    enhanced_bilstm_model = VAEBiLSTMWithAttention(
        vae_model=vae_model,
        input_dim=latent_dim,
        hidden_dim=lstm_hidden_dim,
        num_layers=lstm_num_layers,
        output_dim=50,  # BP signal length
        dropout=0.3,
        use_attention=args.use_attention,
        attention_dim=args.attention_dim
    ).to(device)
    
    print(f"Enhanced BiLSTM model created with {sum(p.numel() for p in enhanced_bilstm_model.parameters() if p.requires_grad):,} trainable parameters")
    wandb.config.update({"trainable_parameters": sum(p.numel() for p in enhanced_bilstm_model.parameters() if p.requires_grad)})
    
    # Train the Enhanced BiLSTM model
    best_epoch, best_val_loss = train_enhanced_bilstm(
        model=enhanced_bilstm_model,
        train_loader=train_loader,
        val_batches=val_batches,
        device=device,
        num_epochs=num_epochs,
        pattern_offsets=pattern_offsets,
        bp_norm_params=bp_norm_params,
        loss_type=args.loss_type,
        output_dir=args.output_dir
    )
    
    # Create a simple test DataLoader from the test batches list
    class BatchListDataLoader:
        def __init__(self, batch_list):
            self.batch_list = batch_list
        
        def __iter__(self):
            return iter(self.batch_list)
        
        def __len__(self):
            return len(self.batch_list)
    
    test_loader = BatchListDataLoader(test_batches)
    
    # Load the best model for evaluation
    best_checkpoint = torch.load(os.path.join(args.output_dir, 'checkpoints/enhanced_bilstm_best.pt'), map_location=device)
    enhanced_bilstm_model.load_state_dict(best_checkpoint['model_state_dict'])
    
    # Get saved hyperparameters
    bp_norm_params = best_checkpoint.get('bp_norm_params', bp_norm_params)
    pattern_offsets = best_checkpoint.get('pattern_offsets', pattern_offsets)
    loss_type = best_checkpoint.get('loss_type', args.loss_type)
    
    print(f"Loaded best model trained with loss type: {loss_type}")
    
    # Evaluate on test data
    test_loss, mse, mae, r2, sys_mae, dias_mae, sys_error, dias_error = evaluate_enhanced_bilstm(
        model=enhanced_bilstm_model,
        test_loader=test_loader,
        device=device,
        pattern_offsets=pattern_offsets,
        bp_norm_params=bp_norm_params,
        output_dir=args.output_dir,
        visualize_attention=args.visualize_attention
    )
    
    # Create a table for wandb
    if mse is not None:
        metrics_table = wandb.Table(columns=["Metric", "Value", "Description"])
        
        # Add basic metrics
        metrics_table.add_data("Test Loss (normalized)", test_loss, "Loss computed on normalized data")
        metrics_table.add_data("MSE", mse, "Mean Squared Error on denormalized data")
        metrics_table.add_data("MAE", mae, "Mean Absolute Error on denormalized data")
        metrics_table.add_data("R² (denormalized)", r2, "R-squared on denormalized data")
        
        # Add systolic and diastolic metrics
        metrics_table.add_data("Systolic MAE", sys_mae, "Mean Absolute Error for systolic values (mmHg)")
        metrics_table.add_data("Systolic Mean Error", sys_error, "Mean Error (bias) for systolic values (mmHg)")
        metrics_table.add_data("Diastolic MAE", dias_mae, "Mean Absolute Error for diastolic values (mmHg)")
        metrics_table.add_data("Diastolic Mean Error", dias_error, "Mean Error (bias) for diastolic values (mmHg)")
        
        # Log the table to wandb
        wandb.log({"evaluation_metrics_table": metrics_table})
    
    # Save the evaluation metrics
    results_file = os.path.join(args.output_dir, 'results/evaluation_metrics.txt')
    with open(results_file, 'w') as f:
        f.write(f"Enhanced VAE-BiLSTM Evaluation Results\n")
        f.write(f"=====================================\n")
        f.write(f"Best Epoch: {best_epoch}\n")
        f.write(f"Best Validation Loss: {best_val_loss:.6f}\n")
        f.write(f"Test Loss (normalized): {test_loss:.6f}\n")
        f.write(f"Frame Pattern Offsets: {pattern_offsets}\n")
        f.write(f"Number of frames: {len(pattern_offsets)}\n")
        f.write(f"Loss Function Type: {loss_type}\n")
        f.write(f"Use Attention: {args.use_attention}\n")
        f.write(f"Attention Dimension: {args.attention_dim}\n")
        f.write(f"VAE Checkpoint: {vae_checkpoint_path}\n")
        f.write(f"Latent Dimension: {latent_dim}\n")
        f.write(f"LSTM Hidden Dimension: {lstm_hidden_dim}\n")
        f.write(f"LSTM Layers: {lstm_num_layers}\n")
        
        if mse is not None:
            f.write(f"\nPerformance Metrics:\n")
            f.write(f"MSE: {mse:.6f}\n")
            f.write(f"MAE: {mae:.6f}\n")
            f.write(f"R² (denormalized): {r2:.6f}\n")
            f.write(f"Systolic MAE: {sys_mae:.2f} mmHg (mean error: {sys_error:.2f})\n")
            f.write(f"Diastolic MAE: {dias_mae:.2f} mmHg (mean error: {dias_error:.2f})\n")
        else:
            f.write("No valid predictions were made during evaluation\n")
    
    print(f"Evaluation metrics saved to {results_file}")
    
    # Finish wandb run
    wandb.finish()


if __name__ == "__main__":
    main()