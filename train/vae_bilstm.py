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


# Prepare specific frame pattern data for the model
def prepare_specific_frame_data(batch_data, central_indices=None, pattern_offsets=[-7, 0, 3]):
    """
    Prepare data with specific frame pattern (k-7, k, k+3) and corresponding BP values
    
    Args:
        batch_data: Dictionary with 'pviHP' and 'bp' keys
        central_indices: List of indices to use as the central frame k
                        If None, use all valid indices
        pattern_offsets: List of offsets relative to central frame k
                        Default: [-7, 0, 3] for frames (k-7, k, k+3)
    
    Returns:
        frame_sequences: Tensor of shape [batch_size, len(pattern_offsets), 1, 32, 32]
        bp_targets: Tensor of shape [batch_size, 50] (BP signal)
    """
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
    
    Args:
        bp_data: Blood pressure tensor [batch_size, 50]
        min_val: Minimum physiological BP value (default 40 mmHg)
        max_val: Maximum physiological BP value (default 200 mmHg)
    
    Returns:
        Normalized BP data and the normalization parameters
    """
    # Scale to [0, 1]
    bp_normalized = (bp_data - min_val) / (max_val - min_val)
    
    return bp_normalized, (min_val, max_val)


# Denormalize blood pressure data
def denormalize_bp(bp_normalized, min_val=40, max_val=200):
    """
    Convert normalized BP back to original scale
    
    Args:
        bp_normalized: Normalized BP tensor [batch_size, 50]
        min_val: Minimum value used in normalization
        max_val: Maximum value used in normalization
    
    Returns:
        Denormalized BP data
    """
    # Scale back to original range
    bp_denormalized = bp_normalized * (max_val - min_val) + min_val
    
    return bp_denormalized


def composite_bp_loss(y_pred, y_true):
    # Base MSE for overall waveform
    mse_loss = F.mse_loss(y_pred, y_true)
    
    # Calculate systolic and diastolic values
    sys_true = torch.max(y_true, dim=1)[0]
    sys_pred = torch.max(y_pred, dim=1)[0]
    dias_true = torch.min(y_true, dim=1)[0]
    dias_pred = torch.min(y_pred, dim=1)[0]
    
    # Calculate MSE specifically for systolic and diastolic
    sys_loss = F.mse_loss(sys_pred, sys_true)
    dias_loss = F.mse_loss(dias_pred, dias_true)
    
    # Weight the components (you can adjust these weights)
    alpha = 0.6  # Weight for overall waveform
    beta = 0.2   # Weight for systolic
    gamma = 0.2  # Weight for diastolic
    
    total_loss = alpha * mse_loss + beta * sys_loss + gamma * dias_loss
    
    return total_loss

# Train the BiLSTM model
def train_bilstm(model, train_loader, val_batches, device, num_epochs=20, 
                pattern_offsets=[-7, 0, 3], bp_norm_params=(40, 200), output_dir=None):
    """Train the BiLSTM model for blood pressure prediction"""
    
    # Create directories for saving results
    checkpoints_dir = os.path.join(output_dir, 'checkpoints')
    results_dir = os.path.join(output_dir, 'results')
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # Unpack normalization parameters
    bp_min, bp_max = bp_norm_params
    
    # Set up optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)  # Added weight decay
    criterion = composite_bp_loss
    
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
                # Prepare sequences with specific frame pattern
                sequences, targets = prepare_specific_frame_data(batch_data, pattern_offsets=pattern_offsets)
                
                # Normalize targets
                targets_norm, _ = normalize_bp(targets, bp_min, bp_max)
                
                # Move to device
                sequences = sequences.to(device)
                targets_norm = targets_norm.to(device)
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs_norm = model(sequences)
                
                # Calculate loss
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
                    # Prepare sequences with specific frame pattern
                    sequences, targets = prepare_specific_frame_data(batch_data, pattern_offsets=pattern_offsets)
                    
                    # Normalize targets
                    targets_norm, _ = normalize_bp(targets, bp_min, bp_max)
                    
                    # Move to device
                    sequences = sequences.to(device)
                    targets_norm = targets_norm.to(device)
                    
                    # Forward pass
                    outputs_norm = model(sequences)
                    
                    # Calculate loss
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
                'pattern_offsets': pattern_offsets
            }, os.path.join(checkpoints_dir, 'bilstm_best.pt'))
            
            print(f"New best model at epoch {epoch} with validation loss {val_loss:.6f}")
        
        # Save checkpoint every 5 epochs
        if epoch % 5 == 0 or epoch == num_epochs:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'bp_norm_params': bp_norm_params,
                'pattern_offsets': pattern_offsets
            }, os.path.join(checkpoints_dir, f'bilstm_epoch_{epoch}.pt'))
    
    # Plot training curves
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('BiLSTM Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot(range(1, num_epochs + 1), learning_rates)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate vs. Epoch')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'bilstm_training_curves.png'), dpi=150)
    plt.close()
    
    print(f"Training completed. Best model at epoch {best_epoch} with validation loss {best_val_loss:.6f}")
    
    return best_epoch, best_val_loss


# Evaluate the BiLSTM model
def evaluate_bilstm(model, test_loader, device, pattern_offsets=[-7, 0, 3], 
                   bp_norm_params=(40, 200), output_dir=None):
    """Evaluate the BiLSTM model on test data"""
    
    results_dir = os.path.join(output_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Unpack normalization parameters
    bp_min, bp_max = bp_norm_params
    
    model.eval()
    criterion = nn.MSELoss()
    
    all_targets = []
    all_predictions = []
    test_loss = 0
    sample_count = 0
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(tqdm(test_loader, desc="Testing")):
            try:
                # Prepare sequences with specific frame pattern
                sequences, targets = prepare_specific_frame_data(batch_data, pattern_offsets=pattern_offsets)
                
                # Normalize targets
                targets_norm, _ = normalize_bp(targets, bp_min, bp_max)
                
                # Move to device
                sequences = sequences.to(device)
                targets_norm = targets_norm.to(device)
                
                # Forward pass
                outputs_norm = model(sequences)
                
                # Calculate loss
                loss = criterion(outputs_norm, targets_norm)
                test_loss += loss.item() * sequences.size(0)
                sample_count += sequences.size(0)
                
                # Denormalize predictions for metrics calculation
                outputs = denormalize_bp(outputs_norm.cpu(), bp_min, bp_max)
                
                # Store predictions and targets
                all_predictions.append(outputs.numpy())
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
        
        # Plot example predictions
        plot_predictions(all_predictions, all_targets, output_dir)
        
        # Calculate systolic and diastolic metrics separately
        sys_mae, sys_error, dias_mae, dias_error = calculate_bp_metrics(all_predictions, all_targets, output_dir)
        
        print(f"Systolic MAE: {sys_mae:.2f} mmHg (mean error: {sys_error:.2f})")
        print(f"Diastolic MAE: {dias_mae:.2f} mmHg (mean error: {dias_error:.2f})")
        
        return test_loss, mse, mae, r2, sys_mae, dias_mae
    else:
        print("No valid predictions were made during evaluation")
        return test_loss, None, None, None, None, None


# Plot example predictions vs targets
def plot_predictions(predictions, targets, output_dir, num_examples=5):
    """Plot example BP predictions vs targets"""
    
    results_dir = os.path.join(output_dir, 'results')
    
    # Sample a few random examples
    indices = np.random.choice(len(predictions), min(num_examples, len(predictions)), replace=False)
    
    # Create a figure
    fig, axes = plt.subplots(len(indices), 1, figsize=(10, 3*len(indices)))
    if len(indices) == 1:
        axes = [axes]
    
    for i, idx in enumerate(indices):
        # Get prediction and target
        pred = predictions[idx]
        target = targets[idx]
        
        # Plot
        axes[i].plot(target, label='Target BP', color='blue')
        axes[i].plot(pred, label='Predicted BP', color='red', linestyle='--')
        axes[i].set_title(f'Example {i+1}')
        axes[i].set_xlabel('Time')
        axes[i].set_ylabel('Blood Pressure')
        axes[i].legend()
        axes[i].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'bp_predictions.png'), dpi=150)
    plt.close()


# Calculate systolic and diastolic metrics
def calculate_bp_metrics(predictions, targets, output_dir):
    """Calculate metrics for systolic and diastolic pressure"""
    
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
    
    # Plot Bland-Altman plot for systolic pressure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Systolic Bland-Altman
    mean_sys = (sys_targets + sys_preds) / 2
    diff_sys = sys_targets - sys_preds
    
    axes[0].scatter(mean_sys, diff_sys, alpha=0.5)
    axes[0].axhline(y=np.mean(diff_sys), color='r', linestyle='-', label=f'Mean Error: {np.mean(diff_sys):.2f}')
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
    axes[1].axhline(y=np.mean(diff_dias), color='r', linestyle='-', label=f'Mean Error: {np.mean(diff_dias):.2f}')
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
    plt.savefig(os.path.join(results_dir, 'bp_bland_altman.png'), dpi=150)
    plt.close()
    
    return sys_mae, sys_error, dias_mae, dias_error
def improved_evaluation(model, test_loader, device, pattern_offsets=[-7, 0, 3], bp_norm_params=(40, 200), output_dir=None):
    """Enhanced evaluation with correlation coefficients and improved plotting"""
    
    results_dir = os.path.join(output_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Unpack normalization parameters
    bp_min, bp_max = bp_norm_params
    
    model.eval()
    
    all_targets = []
    all_predictions = []
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(tqdm(test_loader, desc="Testing")):
            try:
                # Prepare sequences with specific frame pattern
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
                print(f"Error processing test batch {batch_idx}: {e}")
                continue
    
    # Concatenate all predictions and targets
    if all_predictions and all_targets:
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        
        # Calculate standard metrics
        mse = mean_squared_error(all_targets.flatten(), all_predictions.flatten())
        mae = mean_absolute_error(all_targets.flatten(), all_predictions.flatten())
        r2 = r2_score(all_targets.flatten(), all_predictions.flatten())
        
        # Calculate Pearson correlation coefficient
        pearson_r = np.corrcoef(all_targets.flatten(), all_predictions.flatten())[0, 1]
        
        # Get systolic and diastolic values
        sys_targets = np.max(all_targets, axis=1)
        dias_targets = np.min(all_targets, axis=1)
        sys_preds = np.max(all_predictions, axis=1)
        dias_preds = np.min(all_predictions, axis=1)
        
        # Calculate systolic and diastolic metrics
        sys_mae = np.mean(np.abs(sys_targets - sys_preds))
        dias_mae = np.mean(np.abs(dias_targets - dias_preds))
        
        # Calculate Pearson correlation for systolic and diastolic
        sys_pearson = np.corrcoef(sys_targets, sys_preds)[0, 1]
        dias_pearson = np.corrcoef(dias_targets, dias_preds)[0, 1]
        
        # Plot prediction vs reference with error percentiles
        plot_prediction_vs_reference(sys_targets, sys_preds, 'Systolic', output_dir)
        plot_prediction_vs_reference(dias_targets, dias_preds, 'Diastolic', output_dir)
        
        # Print metrics
        print(f"MSE: {mse:.6f}")
        print(f"MAE: {mae:.6f}")
        print(f"R²: {r2:.6f}")
        print(f"Pearson Correlation: {pearson_r:.6f}")
        print(f"Systolic MAE: {sys_mae:.2f} mmHg, Pearson: {sys_pearson:.4f}")
        print(f"Diastolic MAE: {dias_mae:.2f} mmHg, Pearson: {dias_pearson:.4f}")
        
        return mse, mae, r2, pearson_r, sys_mae, dias_mae, sys_pearson, dias_pearson
    else:
        print("No valid predictions were made during evaluation")
        return None


def plot_prediction_vs_reference(targets, predictions, label, output_dir):
    """
    Create prediction vs. reference plot with error percentiles
    
    Args:
        targets: Ground truth values
        predictions: Predicted values
        label: Label for the plot ('Systolic' or 'Diastolic')
        output_dir: Directory to save output files
    """
    results_dir = os.path.join(output_dir, 'results')
    
    plt.figure(figsize=(10, 8))
    
    # Calculate error percentiles
    errors = targets - predictions
    p95 = np.percentile(errors, 95)
    p5 = np.percentile(errors, 5)
    
    # Plot scatter with alpha for density visualization
    plt.scatter(predictions, targets, alpha=0.3)
    
    # Add identity line
    min_val = min(np.min(predictions), np.min(targets))
    max_val = max(np.max(predictions), np.max(targets))
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', label='Identity Line')
    
    # Add error percentile lines
    x_range = np.linspace(min_val, max_val, 100)
    plt.plot(x_range, x_range + p95, 'r--', label=f'95th Percentile: {p95:.2f}')
    plt.plot(x_range, x_range + p5, 'g--', label=f'5th Percentile: {p5:.2f}')
    
    # Calculate and show correlation coefficient
    correlation = np.corrcoef(targets, predictions)[0, 1]
    plt.text(0.05, 0.95, f'Pearson r: {correlation:.4f}', 
             transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Add labels and title
    plt.xlabel(f'Predicted {label} BP (mmHg)')
    plt.ylabel(f'Reference {label} BP (mmHg)')
    plt.title(f'{label} Blood Pressure: Prediction vs. Reference')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Save figure
    plt.savefig(os.path.join(results_dir, f'{label.lower()}_prediction_vs_reference.png'), dpi=300)
    plt.close()


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Train and evaluate VAE-BiLSTM for BP prediction')
    parser.add_argument('--output_dir', type=str, default='vae_bilstm_output',
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
    parser.add_argument('--num_epochs', type=int, default=20,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training')
    
    args = parser.parse_args()
    
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
    
    # Set hyperparameters
    latent_dim = args.latent_dim
    lstm_hidden_dim = args.lstm_hidden_dim
    lstm_num_layers = args.lstm_layers
    num_epochs = args.num_epochs
    
    # Specific frame pattern as described (k-7, k, k+3)
    pattern_offsets = [-7, 0, 3]
    
    # Blood pressure normalization range
    bp_norm_params = (40, 200)  # Expected range: 40-200 mmHg
    
    print(f"Using frame pattern with offsets: {pattern_offsets}")
    
    # Load dataset
    print(f"Loading dataset from: {args.data_path}")
    dataset = PviDataset(args.data_path)
    print(f"Dataset loaded with {len(dataset)} samples")
    
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
    
    # Create BiLSTM model
    bilstm_model = VAEBiLSTM(
        vae_model=vae_model,
        input_dim=latent_dim,
        hidden_dim=lstm_hidden_dim,
        num_layers=lstm_num_layers,
        output_dim=50,  # BP signal length
        dropout=0.3     # Increased dropout
    ).to(device)
    
    print(f"BiLSTM model created with {sum(p.numel() for p in bilstm_model.parameters() if p.requires_grad):,} trainable parameters")
    
    # Train the BiLSTM model
    best_epoch, best_val_loss = train_bilstm(
        model=bilstm_model,
        train_loader=train_loader,
        val_batches=val_batches,
        device=device,
        num_epochs=num_epochs,
        pattern_offsets=pattern_offsets,
        bp_norm_params=bp_norm_params,
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
    best_checkpoint = torch.load(os.path.join(args.output_dir, 'checkpoints/bilstm_best.pt'), map_location=device)
    bilstm_model.load_state_dict(best_checkpoint['model_state_dict'])
    
    # Get saved hyperparameters
    bp_norm_params = best_checkpoint.get('bp_norm_params', bp_norm_params)
    pattern_offsets = best_checkpoint.get('pattern_offsets', pattern_offsets)
    
    # Evaluate on test data
    test_loss, mse, mae, r2, sys_mae, dias_mae = evaluate_bilstm(
        model=bilstm_model,
        test_loader=test_loader,
        device=device,
        pattern_offsets=pattern_offsets,
        bp_norm_params=bp_norm_params,
        output_dir=args.output_dir
    )
    
    # Run improved evaluation with advanced metrics
    print("\nRunning improved evaluation with correlation analysis and detailed plots...")
    adv_metrics = improved_evaluation(
        model=bilstm_model,
        test_loader=test_loader,
        device=device,
        pattern_offsets=pattern_offsets,
        bp_norm_params=bp_norm_params,
        output_dir=args.output_dir
    )
    
    # Save the evaluation metrics
    results_file = os.path.join(args.output_dir, 'results/evaluation_metrics.txt')
    with open(results_file, 'w') as f:
        f.write(f"Best Epoch: {best_epoch}\n")
        f.write(f"Best Validation Loss: {best_val_loss:.6f}\n")
        f.write(f"Test Loss: {test_loss:.6f}\n")
        f.write(f"Frame Pattern Offsets: {pattern_offsets}\n")
        f.write(f"VAE Checkpoint: {vae_checkpoint_path}\n")
        f.write(f"Latent Dimension: {latent_dim}\n")
        f.write(f"LSTM Hidden Dimension: {lstm_hidden_dim}\n")
        f.write(f"LSTM Layers: {lstm_num_layers}\n")
        
        if mse is not None:
            f.write(f"MSE: {mse:.6f}\n")
            f.write(f"MAE: {mae:.6f}\n")
            f.write(f"R²: {r2:.6f}\n")
            f.write(f"Systolic MAE: {sys_mae:.2f} mmHg\n")
            f.write(f"Diastolic MAE: {dias_mae:.2f} mmHg\n")
            
            # Add advanced metrics if available
            if adv_metrics is not None:
                adv_mse, adv_mae, adv_r2, pearson_r, adv_sys_mae, adv_dias_mae, sys_pearson, dias_pearson = adv_metrics
                f.write(f"\nAdvanced Metrics:\n")
                f.write(f"Pearson Correlation: {pearson_r:.6f}\n")
                f.write(f"Systolic Pearson r: {sys_pearson:.6f}\n")
                f.write(f"Diastolic Pearson r: {dias_pearson:.6f}\n")
        else:
            f.write("No valid predictions were made during evaluation\n")
    
    print(f"Evaluation metrics saved to {results_file}")


if __name__ == "__main__":
    main()