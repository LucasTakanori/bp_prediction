import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import json
import sys

# Add parent directory to path for importing custom modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.data_utils import DataPathManager, PviDataset, PviBatchServer


class CNNFeatureExtractor(nn.Module):
    """
    CNN for extracting features from individual impedance images
    Designed to handle 32x32 input images with gradual downsampling
    """
    def __init__(self, input_channels=1, feature_dim=128):
        super(CNNFeatureExtractor, self).__init__()
        
        # Initial layers with smaller kernel and stride to preserve spatial information
        self.conv_layers = nn.Sequential(
            # Layer 1: 32x32 -> 16x16
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.1),
            
            # Layer 2: 16x16 -> 16x16 (no downsampling, increase features)
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            
            # Layer 3: 16x16 -> 8x8
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.1),
            
            # Layer 4: 8x8 -> 8x8 (no downsampling, increase features)
            nn.Conv2d(128, 192, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(192),
            nn.LeakyReLU(0.2),
            
            # Layer 5: 8x8 -> 4x4
            nn.Conv2d(192, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            
            # Global average pooling instead of flattening to handle variable input sizes
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Final projection to feature dimension with layer normalization for stability
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, feature_dim),
            nn.LayerNorm(feature_dim),  # Added for numerical stability
            nn.ReLU()
        )
        
        # Initialize weights properly
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)
            
    def forward(self, x):
        """
        Forward pass through the CNN feature extractor
        
        Args:
            x: Input image tensor of shape [batch_size, channels, height, width]
            
        Returns:
            features: Extracted features of shape [batch_size, feature_dim]
        """
        # Numerical stability check
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("Warning: Input contains NaN or Inf values")
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
            
        x = self.conv_layers(x)
        features = self.projection(x)
        return features


class LSTMSequenceProcessor(nn.Module):
    """
    LSTM for processing sequence of image features to predict BP waveform with
    explicit modeling of period K and surrounding context periods
    """
    def __init__(self, input_dim, hidden_dim=256, output_size=50, num_layers=2, dropout=0.3):
        super(LSTMSequenceProcessor, self).__init__()
        
        # Constants for period structure
        self.frames_per_period = 50  # 50 frames per second
        self.past_periods = 7        # 7 past periods
        self.future_periods = 2      # 2 future periods
        
        # Calculate indices for period K
        self.period_k_start = self.past_periods * self.frames_per_period
        self.period_k_end = self.period_k_start + self.frames_per_period
        
        # Input normalization for stability
        self.input_norm = nn.LayerNorm(input_dim)
        
        # LSTM for processing the entire sequence
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Output normalization for LSTM (critical for stability)
        self.lstm_norm = nn.LayerNorm(hidden_dim * 2)
        
        # Attention for weighting different time steps (with careful initialization)
        self.global_attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),  # Bounded activation for stability
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=1)
        )
        
        # Period-specific attention to focus on each frame within period K
        self.period_k_attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),  # Bounded activation for stability
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=1)
        )
        
        # Context integration module - combines period K features with context
        self.context_integration = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),  # 2 * hidden_dim*2 (period K + context)
            nn.LayerNorm(hidden_dim * 2),  # Critical for numerical stability
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Regression head to predict BP waveform for period K
        self.regression_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),  # Added for stability
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_size)
        )
        
        # Initialize all parameters with proper values
        self._init_lstm_weights()
        self.apply(self._init_weights)
        
    def _init_lstm_weights(self):
        """Initialize LSTM weights with orthogonal initialization for stability"""
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)  # Critical for recurrent connections
            elif 'bias' in name:
                param.data.fill_(0)
                # Set forget gate bias to 1 (helps with vanishing gradients)
                n = param.size(0)
                param.data[n//4:n//2].fill_(1.0)
                
    def _init_weights(self, m):
        """Initialize linear layers with careful initialization"""
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight, gain=0.5)  # Reduced gain for stability
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        """
        Forward pass through the LSTM sequence processor with explicit period K focus
        
        Args:
            x: Sequence of image features [batch_size, sequence_length, input_dim]
            
        Returns:
            bp_waveform: Predicted BP waveform for period K [batch_size, output_size]
        """
        batch_size, seq_len, _ = x.shape
        
        # Input normalization for numerical stability
        x = self.input_norm(x)
        
        # Verify sequence length matches expected total (7 past + 1 current + 2 future periods)
        expected_length = (self.past_periods + 1 + self.future_periods) * self.frames_per_period
        assert seq_len == expected_length, f"Expected sequence length {expected_length}, got {seq_len}"
        
        # Process the entire sequence with LSTM
        lstm_out, _ = self.lstm(x)  # [batch_size, seq_len, hidden_dim*2]
        
        # Apply layer normalization to LSTM outputs for stability
        lstm_out = self.lstm_norm(lstm_out)
        
        # 1. Extract LSTM outputs for period K
        period_k_features = lstm_out[:, self.period_k_start:self.period_k_end, :]  # [batch_size, 50, hidden_dim*2]
        
        # 2. Apply period K-specific attention with numerical stability
        period_k_weights = self.period_k_attention(period_k_features)  # [batch_size, 50, 1]
        # Ensure weights sum to 1 with epsilon for stability
        period_k_weights = period_k_weights / (period_k_weights.sum(dim=1, keepdim=True) + 1e-10)
        period_k_context = torch.sum(period_k_weights * period_k_features, dim=1)  # [batch_size, hidden_dim*2]
        
        # 3. Global context from all periods (with focus on surrounding periods)
        # Create a mask that downweights period K to focus on context
        context_mask = torch.ones(batch_size, seq_len, 1, device=x.device)
        context_mask[:, self.period_k_start:self.period_k_end, :] = 0.5  # Reduce weight for period K
        
        # Apply global attention with context mask and numerical stability
        global_weights = self.global_attention(lstm_out) * context_mask  # [batch_size, seq_len, 1]
        global_weights = global_weights / (global_weights.sum(dim=1, keepdim=True) + 1e-10)  # Renormalize with epsilon
        global_context = torch.sum(global_weights * lstm_out, dim=1)  # [batch_size, hidden_dim*2]
        
        # 4. Integrate period K features with global context
        combined_features = torch.cat([period_k_context, global_context], dim=1)  # [batch_size, hidden_dim*4]
        integrated_features = self.context_integration(combined_features)  # [batch_size, hidden_dim*2]
        
        # 5. Generate the BP waveform for period K
        bp_waveform = self.regression_head(integrated_features)  # [batch_size, output_size=50]
        
        return bp_waveform


class CNNLSTMModel(nn.Module):
    """
    Combined model using CNN for feature extraction and LSTM for sequence processing
    """
    def __init__(self, cnn_feature_extractor, lstm_processor):
        super(CNNLSTMModel, self).__init__()
        
        self.cnn = cnn_feature_extractor
        self.lstm = lstm_processor
        
    def forward(self, x):
        """
        Forward pass through the combined CNN-LSTM model
        
        Args:
            x: Sequence of images [batch_size, channels, height, width, time_steps]
                where time_steps = 500 (7 past periods + current period K + 2 future periods)
            
        Returns:
            bp_waveform: Predicted BP waveform for period K [batch_size, 50]
        """
        batch_size, channels, height, width, time_steps = x.shape
        
        # Verify we have the expected number of frames (500 total)
        frames_per_period = 50
        expected_frames = (7 + 1 + 2) * frames_per_period  # 7 past + current K + 2 future periods
        assert time_steps == expected_frames, f"Expected 500 frames, got {time_steps}"
        
        # Check for NaN/Inf in input data
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("Warning: Input contains NaN or Inf values")
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Reshape to process each image independently through CNN
        x_reshaped = x.reshape(batch_size * time_steps, channels, height, width)
        
        # Extract features for all images
        features = self.cnn(x_reshaped)  # [batch_size*time_steps, feature_dim]
        
        # Reshape back to sequence form
        features = features.view(batch_size, time_steps, -1)  # [batch_size, 500, feature_dim]
        
        # Process sequence with LSTM to predict BP waveform for period K only
        bp_waveform = self.lstm(features)  # [batch_size, 50]
        
        return bp_waveform


def stable_smooth_l1_loss(pred, target, beta=1.0):
    """
    Numerically stable implementation of Smooth L1 Loss
    """
    diff = torch.abs(pred - target)
    cond = diff < beta
    loss = torch.where(cond, 0.5 * diff ** 2 / beta, diff - 0.5 * beta)
    return loss.mean()


def train_model(model, train_loader, test_loader, num_epochs=50, learning_rate=5e-5,
                device='cuda', output_dir='cnn_lstm_output'):
    """
    Train the CNN-LSTM model for BP waveform prediction with enhanced numerical stability
    
    Args:
        model: The CNN-LSTM model
        train_loader: DataLoader for training data
        test_loader: DataLoader for test/validation data
        num_epochs: Number of training epochs
        learning_rate: Initial learning rate
        device: Device to use for training ('cuda' or 'cpu')
        output_dir: Directory to save model checkpoints and visualizations
        
    Returns:
        model: Trained model
        metrics: Dictionary containing training metrics
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize optimizer with weight decay for regularization (reduced value)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-6,  # Reduced for stability
        eps=1e-8  # Increased epsilon for numerical stability
    )
    
    # Learning rate scheduler with patience
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,  # Increased patience
        verbose=True,
        min_lr=1e-7
    )
    
    # Learning rate warmup parameters
    warmup_epochs = 3
    
    # Move model to device
    model.to(device)
    
    # Initialize metrics tracking
    metrics = {
        'train_losses': [],
        'test_losses': [],
        'learning_rates': [],
        'epochs': [],
        'gradient_norms': []
    }
    
    best_test_loss = float('inf')
    nan_count = 0  # Counter for tracking NaN occurrences
    
    for epoch in range(num_epochs):
        # Apply learning rate warmup
        if epoch < warmup_epochs:
            # Linear warmup
            lr_scale = min(1.0, (epoch + 1) / warmup_epochs)
            for pg in optimizer.param_groups:
                pg['lr'] = learning_rate * lr_scale
            print(f"Warmup LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Training phase
        model.train()
        train_loss = 0.0
        epoch_grad_norm = 0.0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Train)")
        for batch_idx, batch in enumerate(progress_bar):
            # Get data
            images = batch['pviHP'].to(device)  # [batch_size, 1, 32, 32, 500]
            bp_target = batch['bp'].to(device)  # [batch_size, 50]
            
            # Data validation (check for NaN in target)
            if torch.isnan(bp_target).any() or torch.isinf(bp_target).any():
                print(f"Warning: Target contains NaN/Inf values at batch {batch_idx}")
                bp_target = torch.nan_to_num(bp_target, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # Forward pass
            bp_pred = model(images)
            
            # Check for NaN in output
            if torch.isnan(bp_pred).any() or torch.isinf(bp_pred).any():
                print(f"Warning: NaN/Inf in model output at batch {batch_idx}")
                bp_pred = torch.nan_to_num(bp_pred, nan=0.0, posinf=1.0, neginf=-1.0)
                nan_count += 1
                if nan_count > 5:
                    print("Too many NaN occurrences, reducing learning rate...")
                    for pg in optimizer.param_groups:
                        pg['lr'] *= 0.5
                    nan_count = 0
            
            # Compute loss with stable implementation
            loss = stable_smooth_l1_loss(bp_pred, bp_target, beta=1.0)
            
            # Backward pass and optimization
            optimizer.zero_grad(set_to_none=True)  # More efficient and prevents accumulating small values
            
            # Check if loss is valid
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: Loss is NaN/Inf at batch {batch_idx}. Skipping batch.")
                continue
                
            loss.backward()
            
            # Gradient value check and NaN fix
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            epoch_grad_norm += grad_norm.item()
            
            # Check for NaN in gradients
            nan_in_grad = False
            for name, param in model.named_parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        print(f"NaN/Inf gradient detected in {name}")
                        param.grad = torch.nan_to_num(param.grad, nan=0.0, posinf=0.0, neginf=0.0)
                        nan_in_grad = True
            
            # Skip update if NaN gradients were found and fixed
            if not nan_in_grad:
                optimizer.step()
            
            # Update metrics
            train_loss += loss.item()
            num_batches += 1
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}", 'lr': f"{optimizer.param_groups[0]['lr']:.6f}"})
        
        # Calculate average metrics
        avg_train_loss = train_loss / max(num_batches, 1)  # Avoid division by zero
        avg_grad_norm = epoch_grad_norm / max(num_batches, 1)
        metrics['train_losses'].append(avg_train_loss)
        metrics['gradient_norms'].append(avg_grad_norm)
        
        # Evaluation phase
        model.eval()
        test_loss = 0.0
        num_test_batches = 0
        
        with torch.no_grad():
            progress_bar = tqdm(test_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Test)")
            for batch in progress_bar:
                # Get data
                images = batch['pviHP'].to(device)
                bp_target = batch['bp'].to(device)
                
                # Data validation
                if torch.isnan(bp_target).any():
                    bp_target = torch.nan_to_num(bp_target, nan=0.0)
                
                # Forward pass
                bp_pred = model(images)
                
                # Handle NaN in predictions
                if torch.isnan(bp_pred).any():
                    bp_pred = torch.nan_to_num(bp_pred, nan=0.0)
                
                # Compute loss
                loss = stable_smooth_l1_loss(bp_pred, bp_target)
                
                # Skip invalid loss values
                if torch.isnan(loss) or torch.isinf(loss):
                    continue
                
                # Update metrics
                test_loss += loss.item()
                num_test_batches += 1
                progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        avg_test_loss = test_loss / max(num_test_batches, 1)
        metrics['test_losses'].append(avg_test_loss)
        
        # Update learning rate
        scheduler.step(avg_test_loss)
        metrics['learning_rates'].append(optimizer.param_groups[0]['lr'])
        metrics['epochs'].append(epoch + 1)
        
        # Print epoch summary
        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Train Loss: {avg_train_loss:.4f}, "
              f"Test Loss: {avg_test_loss:.4f}, "
              f"Grad Norm: {avg_grad_norm:.4f}, "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if avg_test_loss < best_test_loss and not np.isnan(avg_test_loss):
            best_test_loss = avg_test_loss
            torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pt'))
            print(f"New best model saved with test loss: {best_test_loss:.4f}")
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(output_dir, f"checkpoint_epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'test_loss': avg_test_loss
            }, checkpoint_path)
            
            # Save current metrics
            with open(os.path.join(output_dir, 'training_metrics.json'), 'w') as f:
                json.dump(metrics, f, indent=4)
    
    # Plot training curves
    plt.figure(figsize=(15, 12))
    
    plt.subplot(3, 1, 1)
    plt.plot(metrics['epochs'], metrics['train_losses'], 'b-', label='Train Loss')
    plt.plot(metrics['epochs'], metrics['test_losses'], 'r-', label='Test Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(3, 1, 2)
    plt.plot(metrics['epochs'], metrics['learning_rates'], 'g-')
    plt.title('Learning Rate')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.grid(True)
    
    plt.subplot(3, 1, 3)
    plt.plot(metrics['epochs'], metrics['gradient_norms'], 'm-')
    plt.title('Gradient Norm')
    plt.xlabel('Epoch')
    plt.ylabel('Norm')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'))
    plt.close()
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(output_dir, 'final_model.pt'))
    
    print(f"Training complete! Best test loss: {best_test_loss:.4f}")
    return model, metrics


def evaluate_model(model, test_loader, device='cuda', output_dir='evaluation_results'):
    """
    Evaluate the trained CNN-LSTM model and generate visualizations
    
    Args:
        model: Trained CNN-LSTM model
        test_loader: DataLoader for test data
        device: Device to use for evaluation
        output_dir: Directory to save evaluation results
        
    Returns:
        metrics: Dictionary containing evaluation metrics
    """
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    
    all_losses = []
    all_mae = []
    all_predictions = []
    all_targets = []
    
    print("Evaluating model...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader)):
            # Get data
            images = batch['pviHP'].to(device)
            bp_target = batch['bp'].to(device)
            
            try:
                # Forward pass
                bp_pred = model(images)
                
                # Skip any batches with NaN predictions
                if torch.isnan(bp_pred).any() or torch.isnan(bp_target).any():
                    continue
                
                # Calculate metrics
                loss = F.mse_loss(bp_pred, bp_target)
                mae = F.l1_loss(bp_pred, bp_target)
                
                all_losses.append(loss.item())
                all_mae.append(mae.item())
                
                # Store predictions and targets for visualization
                all_predictions.append(bp_pred.cpu().numpy())
                all_targets.append(bp_target.cpu().numpy())
                
                # Visualize some predictions
                if batch_idx < 5:
                    plt.figure(figsize=(12, 6))
                    
                    # Plot first sample in batch
                    time_points = np.arange(bp_target.shape[1])
                    plt.plot(time_points, bp_target[0].cpu().numpy(), 'b-', label='Ground Truth', linewidth=2)
                    plt.plot(time_points, bp_pred[0].cpu().numpy(), 'r-', label='Prediction', linewidth=2)
                    
                    plt.title(f'BP Waveform Prediction - Sample {batch_idx+1}')
                    plt.xlabel('Time Points (within period K)')
                    plt.ylabel('Blood Pressure')
                    plt.legend()
                    plt.grid(True)
                    
                    plt.savefig(os.path.join(output_dir, f'prediction_sample_{batch_idx+1}.png'))
                    plt.close()
            except Exception as e:
                print(f"Error in evaluation batch {batch_idx}: {e}")
                continue
    
    if not all_losses:
        print("Warning: No valid evaluation batches found!")
        return {"mse": float('nan'), "mae": float('nan'), "correlation": float('nan')}
    
    # Calculate overall metrics
    mean_loss = np.mean(all_losses)
    mean_mae = np.mean(all_mae)
    
    # Concatenate all predictions and targets
    all_predictions = np.concatenate(all_predictions)
    all_targets = np.concatenate(all_targets)
    
    # Calculate correlation
    correlations = []
    for i in range(all_predictions.shape[0]):
        try:
            corr = np.corrcoef(all_predictions[i], all_targets[i])[0, 1]
            if not np.isnan(corr):
                correlations.append(corr)
        except:
            continue
    
    mean_correlation = np.mean(correlations) if correlations else float('nan')
    
    # Compile metrics
    metrics = {
        'mse': float(mean_loss),
        'mae': float(mean_mae),
        'correlation': float(mean_correlation)
    }
    
    # Save metrics
    with open(os.path.join(output_dir, 'evaluation_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print(f"Evaluation complete!")
    print(f"MSE: {mean_loss:.4f}")
    print(f"MAE: {mean_mae:.4f}")
    print(f"Correlation: {mean_correlation:.4f}")
    
    # Visualize error distribution
    errors = all_predictions - all_targets
    plt.figure(figsize=(10, 6))
    plt.hist(errors.flatten(), bins=50, alpha=0.7)
    plt.title('Error Distribution')
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'error_distribution.png'))
    plt.close()
    
    return metrics


def visualize_predictions(model, test_loader, num_samples=5, device='cuda', output_dir='visualization_output'):
    """
    Visualize BP waveform predictions alongside the corresponding impedance images for period K
    
    Args:
        model: Trained CNN-LSTM model
        test_loader: DataLoader for test data
        num_samples: Number of samples to visualize
        device: Device to use for inference
        output_dir: Directory to save visualizations
    """
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    
    # Get batch of data
    data_iter = iter(test_loader)
    batch = next(data_iter)
    
    images = batch['pviHP'].to(device)
    bp_target = batch['bp'].to(device)
    
    # Compute predictions
    with torch.no_grad():
        bp_pred = model(images)
    
    # Number of samples to visualize (limited by batch size)
    num_samples = min(num_samples, images.size(0))
    
    # Segment indices for period K (from frame 350 to 400)
    period_k_start = 350
    period_k_end = 400
    
    for i in range(num_samples):
        plt.figure(figsize=(15, 10))
        
        # Plot BP waveform prediction vs ground truth
        plt.subplot(2, 1, 1)
        time_points = np.arange(50)
        plt.plot(time_points, bp_target[i].cpu().numpy(), 'b-', label='Ground Truth', linewidth=2)
        plt.plot(time_points, bp_pred[i].cpu().numpy(), 'r-', label='Prediction', linewidth=2)
        plt.title(f'BP Waveform Prediction - Sample {i+1}')
        plt.xlabel('Time Points (within period K)')
        plt.ylabel('Blood Pressure')
        plt.legend()
        plt.grid(True)
        
        # Plot a montage of period K images
        plt.subplot(2, 1, 2)
        # Extract selected frames from period K
        frames_to_show = [period_k_start, period_k_start + 12, period_k_start + 24, period_k_start + 36, period_k_end - 1]
        
        for j, frame_idx in enumerate(frames_to_show):
            plt.subplot(2, 5, j + 6)  # Position in bottom row
            img = images[i, 0, :, :, frame_idx].cpu().numpy()
            plt.imshow(img, cmap='viridis')
            plt.title(f'Frame {frame_idx - period_k_start}')
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'visualization_sample_{i+1}.png'))
        plt.close()
    
    print(f"{num_samples} prediction visualizations saved to {output_dir}")


def main():
    """
    Main function to run the CNN-LSTM model training and evaluation
    with explicit handling of the temporal structure (7 past + current K + 2 future periods)
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Define paths and directories
    output_dir = "cnn_lstm_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Set data paths - adjust these to your specific setup
    subject_id = 'subject001'
    session = 'baseline'
    root = "/home/lucas_takanori/phd/data"  # Modify this path as needed
    
    print("Loading dataset...")
    pm = DataPathManager(
        subject=subject_id,
        session=session,
        root=root
    )
    
    dataset = PviDataset(pm._h5_path)
    
    # Create batch server for data preparation
    print("Setting up data loaders...")
    batch_server = PviBatchServer(
        dataset=dataset,
        input_type="img",       # Using images as input
        output_type="full"      # Using full BP signal as output
    )
    
    # Get data loaders
    train_loader, test_loader = batch_server.get_loaders()
    
    # Print data shapes
    data_shapes = batch_server.get_data_shapes()
    print(f"Data shapes: {data_shapes}")
    
    # Validate that the input structure matches our expectations
    # We need 500 timesteps (7 past + 1 current + 2 future periods, 50 frames each)
    input_shape = data_shapes.get('input', None)
    if input_shape:
        time_steps = input_shape[-1]
        expected_steps = (7 + 1 + 2) * 50  # 500
        assert time_steps == expected_steps, f"Expected {expected_steps} time steps, got {time_steps}"
        print(f"✓ Confirmed input temporal structure: {time_steps} frames")
    
    # And output should be 50 values (BP for period K only)
    output_shape = data_shapes.get('output', None)
    if output_shape:
        output_size = output_shape[0]
        expected_output = 50  # BP values for period K
        assert output_size == expected_output, f"Expected {expected_output} output values, got {output_size}"
        print(f"✓ Confirmed output structure: {output_size} BP values for period K")
    
    # Initialize CNN feature extractor
    feature_dim = 128  # Dimension of image features
    cnn = CNNFeatureExtractor(input_channels=1, feature_dim=feature_dim)
    
    # Initialize LSTM sequence processor with explicit period K handling
    lstm = LSTMSequenceProcessor(
        input_dim=feature_dim,
        hidden_dim=256,
        output_size=50,  # 50 BP values for period K
        num_layers=2,
        dropout=0.3
    )
    
    # Combine CNN and LSTM into full model
    model = CNNLSTMModel(cnn, lstm)
    
    # Print model architecture summary
    print("\nModel architecture:")
    print(f"CNN Feature Extractor: {sum(p.numel() for p in cnn.parameters()):,} parameters")
    print(f"LSTM Sequence Processor: {sum(p.numel() for p in lstm.parameters()):,} parameters")
    print(f"Total: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create training configuration file
    training_config = {
        "model_structure": {
            "cnn_feature_dim": feature_dim,
            "lstm_hidden_dim": 256,
            "lstm_layers": 2,
            "dropout": 0.3
        },
        "temporal_structure": {
            "frames_per_period": 50,
            "past_periods": 7,
            "current_period": 1,
            "future_periods": 2,
            "total_frames": 500,
            "target_frames": 50
        },
        "training_params": {
            "batch_size": data_shapes.get('batch_size', 16),
            "learning_rate": 5e-5,  # Reduced for stability
            "epochs": 1,
            "loss_function": "stable_smooth_l1_loss",
            "optimizer": "AdamW",
            "weight_decay": 1e-6,
            "warmup_epochs": 3
        }
    }
    
    # Save configuration
    with open(os.path.join(output_dir, 'training_config.json'), 'w') as f:
        json.dump(training_config, f, indent=4)
    
    print("\nTraining configuration:")
    print(f"- Using {training_config['temporal_structure']['total_frames']} frames to predict")
    print(f"  {training_config['temporal_structure']['target_frames']} BP values for period K")
    print(f"- Network processes {training_config['temporal_structure']['past_periods']} past periods +")
    print(f"  current period + {training_config['temporal_structure']['future_periods']} future periods")
    print(f"- Initial learning rate: {training_config['training_params']['learning_rate']} with {training_config['training_params']['warmup_epochs']} warmup epochs")
    
    # Train model
    print("\nTraining model...")
    trained_model, metrics = train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        num_epochs=training_config['training_params']['epochs'],
        learning_rate=training_config['training_params']['learning_rate'],
        device=device,
        output_dir=output_dir
    )
    
    # Evaluate model
    print("\nEvaluating model...")
    evaluation_metrics = evaluate_model(
        model=trained_model,
        test_loader=test_loader,
        device=device,
        output_dir=os.path.join(output_dir, 'evaluation')
    )
    
    # Create visualizations
    print("\nGenerating visualizations...")
    visualize_predictions(
        model=trained_model,
        test_loader=test_loader,
        num_samples=5,
        device=device,
        output_dir=os.path.join(output_dir, 'visualizations')
    )
    
    print("\nTraining and evaluation complete!")


if __name__ == "__main__":
    main()