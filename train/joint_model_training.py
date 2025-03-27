import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import json

# Import the VAE model definition
from vae_training import VAE


class LSTMPredictor(nn.Module):
    """
    LSTM model for predicting BP waveform from sequence of image embeddings
    """
    def __init__(self, input_dim, hidden_dim=128, output_size=50, num_layers=2, dropout=0.3):
        super(LSTMPredictor, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Since we use bidirectional LSTM, the hidden dimension is doubled
        self.regression_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_size)
        )
        
    def forward(self, x):
        # x shape: [batch_size, sequence_length, input_dim]
        
        # Process the entire sequence
        lstm_out, _ = self.lstm(x)
        
        # Get the output corresponding to the final state
        final_hidden = lstm_out[:, -1, :]
        
        # Generate the BP waveform for period K
        bp_waveform = self.regression_head(final_hidden)
        
        return bp_waveform


class VAELSTMModel(nn.Module):
    """
    Combined model using VAE for image encoding and LSTM for BP prediction
    """
    def __init__(self, vae, lstm_predictor, freeze_vae=True):
        super(VAELSTMModel, self).__init__()
        
        # Extract the encoder part of the VAE
        self.encoder = vae.encoder
        self.fc_mu = vae.fc_mu
        self.lstm_predictor = lstm_predictor
        
        # Freeze VAE encoder weights if specified
        if freeze_vae:
            for param in self.encoder.parameters():
                param.requires_grad = False
            for param in self.fc_mu.parameters():
                param.requires_grad = False
            
            print("VAE encoder weights frozen (not trainable)")
        else:
            print("VAE encoder weights unfrozen (trainable)")
    
    def encode_image(self, x):
        """Encode a single image to latent space"""
        features = self.encoder(x)
        mu = self.fc_mu(features)
        return mu
    
    def encode_sequence(self, x):
        """
        Encode a sequence of images to create embeddings
        x shape: [batch_size, channels, height, width, time_steps]
        """
        batch_size, channels, height, width, time_steps = x.shape
        
        # Reshape to process each image independently
        x_reshaped = x.reshape(batch_size * time_steps, channels, height, width)
        
        # Get embeddings for all images (using only mu from VAE)
        embeddings = self.encode_image(x_reshaped)
        
        # Reshape back to sequence form
        embeddings = embeddings.view(batch_size, time_steps, -1)
        
        return embeddings
    
    def forward(self, x):
        # Create embeddings from the image sequence
        embeddings = self.encode_sequence(x)
        
        # Predict BP waveform using LSTM
        bp_prediction = self.lstm_predictor(embeddings)
        
        return bp_prediction


def train_joint_model(model, train_loader, test_loader, num_epochs=50, learning_rate=1e-4, 
                      device='cuda', checkpoint_dir='joint_model_checkpoints', save_freq=5):
    """Train the combined VAE-LSTM model"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    model.to(device)
    
    # Track metrics
    metrics = {
        'train_losses': [],
        'test_losses': [],
        'learning_rates': [],
        'epochs': []
    }
    
    best_test_loss = float('inf')
    
    print("Starting joint model training...")
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Train)")
        for batch in progress_bar:
            # Extract data
            images = batch['pviHP'].to(device)  # [batch_size, 1, 32, 32, 500]
            bp_target = batch['bp'].to(device)  # [batch_size, 50]
            
            # Forward pass
            bp_pred = model(images)
            
            # BP prediction loss (MSE)
            loss = F.mse_loss(bp_pred, bp_target)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Evaluation phase
        model.eval()
        test_loss = 0
        
        with torch.no_grad():
            progress_bar = tqdm(test_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Test)")
            for batch in progress_bar:
                images = batch['pviHP'].to(device)
                bp_target = batch['bp'].to(device)
                
                bp_pred = model(images)
                loss = F.mse_loss(bp_pred, bp_target)
                
                test_loss += loss.item()
                progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        avg_test_loss = test_loss / len(test_loader)
        
        # Update learning rate
        scheduler.step(avg_test_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Update metrics
        metrics['train_losses'].append(avg_train_loss)
        metrics['test_losses'].append(avg_test_loss)
        metrics['learning_rates'].append(current_lr)
        metrics['epochs'].append(epoch + 1)
        
        # Print epoch results
        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Train Loss: {avg_train_loss:.4f}, '
              f'Test Loss: {avg_test_loss:.4f}, '
              f'LR: {current_lr:.6f}')
        
        # Save the model if it's the best so far
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            best_model_path = os.path.join(checkpoint_dir, 'best_vae_lstm_model.pt')
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with test loss: {best_test_loss:.4f}")
        
        # Save checkpoint at specified frequency
        if (epoch + 1) % save_freq == 0 or epoch == num_epochs - 1:
            checkpoint_path = os.path.join(checkpoint_dir, f'joint_model_checkpoint_epoch_{epoch+1}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'test_loss': avg_test_loss,
                'best_test_loss': best_test_loss
            }, checkpoint_path)
            
            # Save metrics to JSON
            metrics_path = os.path.join(checkpoint_dir, 'training_metrics.json')
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=4)
    
    # Plot training and test losses
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    epochs = range(1, num_epochs+1)
    plt.plot(epochs, metrics['train_losses'], 'b-', label='Train Loss')
    plt.plot(epochs, metrics['test_losses'], 'r-', label='Test Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss (MSE)')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(epochs, metrics['learning_rates'], 'g-')
    plt.title('Learning Rate')
    plt.ylabel('Learning Rate')
    plt.xlabel('Epoch')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(checkpoint_dir, 'joint_model_training.png'))
    
    # Save final model
    final_model_path = os.path.join(checkpoint_dir, 'joint_model_final.pt')
    torch.save(model.state_dict(), final_model_path)
    
    print(f"Training complete! Best test loss: {best_test_loss:.4f}")
    print(f"Final model saved to: {final_model_path}")
    return model


def evaluate_model(model, test_loader, device='cuda', output_dir='evaluation_results'):
    """Evaluate the trained model and generate visualizations"""
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    
    total_loss = 0.0
    all_predictions = []
    all_targets = []
    
    print("Evaluating model on test set...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader)):
            images = batch['pviHP'].to(device)
            bp_target = batch['bp'].to(device)
            
            bp_pred = model(images)
            loss = F.mse_loss(bp_pred, bp_target)
            total_loss += loss.item()
            
            # Store predictions and targets for visualization
            all_predictions.append(bp_pred.cpu().numpy())
            all_targets.append(bp_target.cpu().numpy())
            
            # Visualize a few examples
            if batch_idx < 5:  # Visualize first 5 batches
                # Plot the first sample in each batch
                plt.figure(figsize=(12, 6))
                time_points = np.arange(50)
                
                plt.plot(time_points, bp_target[0].cpu().numpy(), 'b-', label='Ground Truth', linewidth=2)
                plt.plot(time_points, bp_pred[0].cpu().numpy(), 'r-', label='Prediction', linewidth=2)
                
                plt.title(f'BP Waveform Prediction - Batch {batch_idx+1}, Sample 1')
                plt.xlabel('Time Points (within period K)')
                plt.ylabel('Blood Pressure')
                plt.legend()
                plt.grid(True)
                
                plt.savefig(os.path.join(output_dir, f'pred_vs_target_batch_{batch_idx+1}.png'))
                plt.close()
    
    # Calculate average loss
    avg_loss = total_loss / len(test_loader)
    print(f"Average Test Loss: {avg_loss:.4f}")
    
    # Concatenate all predictions and targets
    all_predictions = np.concatenate(all_predictions)
    all_targets = np.concatenate(all_targets)
    
    # Calculate global statistics
    mse = np.mean((all_predictions - all_targets) ** 2)
    mae = np.mean(np.abs(all_predictions - all_targets))
    
    # Save evaluation metrics
    evaluation_metrics = {
        'mse': float(mse),
        'mae': float(mae),
        'avg_loss': float(avg_loss)
    }
    
    with open(os.path.join(output_dir, 'evaluation_metrics.json'), 'w') as f:
        json.dump(evaluation_metrics, f, indent=4)
    
    print(f"Evaluation complete. MSE: {mse:.4f}, MAE: {mae:.4f}")
    return evaluation_metrics


def main():
    """Main function to train and evaluate the joint BP prediction model"""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Define paths and directories
    vae_model_path = "vae_output/vae_final.pt"
    output_dir = "joint_model_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the dataset
    from data_utils import DataPathManager, PviDataset, PviBatchServer
    
    # Set data paths
    subject_id = 'subject001'
    session = 'baseline'
    root = "/home/lucas_takanori/phd/data"
    
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
    print("Data shapes:", data_shapes)
    
    # Step 1: Load pre-trained VAE
    print(f"Loading pre-trained VAE from: {vae_model_path}")
    latent_dim = 64  # Must match the pre-trained VAE's latent dimension
    
    vae = VAE(image_size=32, image_channels=1, latent_dim=latent_dim)
    vae.load_state_dict(torch.load(vae_model_path))
    print("Pre-trained VAE loaded successfully")
    
    # Step 2: Initialize LSTM predictor
    print("Initializing LSTM predictor...")
    lstm_predictor = LSTMPredictor(
        input_dim=latent_dim,   # Input dimension = VAE latent space dimension
        hidden_dim=128,         # Hidden dimension for LSTM
        output_size=50,         # Output size = 50 BP values (period K)
        num_layers=2,           # Number of LSTM layers
        dropout=0.3             # Dropout rate
    )
    
    # Step 3: Create combined model
    print("Creating combined VAE-LSTM model...")
    combined_model = VAELSTMModel(
        vae=vae,
        lstm_predictor=lstm_predictor,
        freeze_vae=True  # Freeze VAE weights by default
    )
    
    # Step 4: Train the combined model
    print("Training combined model...")
    num_epochs = 50
    learning_rate = 1e-4
    
    trained_model = train_joint_model(
        model=combined_model,
        train_loader=train_loader,
        test_loader=test_loader,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        device=device,
        checkpoint_dir=output_dir
    )
    
    # Step 5: Evaluate the model
    print("Evaluating trained model...")
    evaluation_metrics = evaluate_model(
        model=trained_model,
        test_loader=test_loader,
        device=device,
        output_dir=os.path.join(output_dir, 'evaluation')
    )
    
    print("Joint model training and evaluation complete!")
    print(f"Final evaluation metrics: MSE={evaluation_metrics['mse']:.4f}, MAE={evaluation_metrics['mae']:.4f}")


if __name__ == "__main__":
    main()