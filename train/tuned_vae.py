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
import wandb

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
        x = torch.tanh(self.deconv4(x))
        
        return x
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar


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
    # batch_size = frames.shape[0]
    # normalized_frames = torch.zeros_like(frames)
    
    # for i in range(batch_size):
    #     sample = frames[i]
    #     min_val = torch.min(sample)
    #     max_val = torch.max(sample)
        
    #     # Handle case where min == max (constant value)
    #     if min_val == max_val:
    #         normalized_frames[i] = torch.zeros_like(sample)
    #     else:
    #         # Normalize to [0, 1]
    #         normalized_frames[i] = (sample - min_val) / (max_val - min_val)
    
    # return normalized_frames

    return frames


# Loss function with KL weight (beta)
def vae_loss(recon_x, x, mu, logvar, beta):
    """VAE loss = reconstruction loss + beta * KL divergence"""
    # Ensure inputs are valid for tanh range (between -1 and 1)
    x = torch.clamp(x, -1.0, 1.0)
    recon_x = torch.clamp(recon_x, -1.0, 1.0)
    
    # MSE loss - works well with tanh's [-1, 1] range
    recon_loss = F.mse_loss(recon_x, x, reduction='sum') * 100  # Scale factor to make comparable to BCE
    
    # KL divergence
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Total loss
    total_loss = recon_loss + beta * kl_div
    
    return total_loss, recon_loss, kl_div


# Visualize reconstructions and save the figure
def visualize_reconstructions(model, test_data, device, frame_idx=0, num_images=5, epoch=0, output_dir=None):
    reconstructions_dir = os.path.join(output_dir, 'reconstructions')
    
    model.eval()
    
    # Get a batch from test data
    with torch.no_grad():
        # Extract and normalize the specific frame
        frames = extract_frame(test_data, frame_idx).to(device)
        
        # Limit to num_images
        frames = frames[:num_images]
        
        # Get reconstructions
        recon_batch, _, _ = model(frames)
    
    # Create a figure
    fig, axes = plt.subplots(2, num_images, figsize=(15, 6))
    
    # Convert tensors to numpy for plotting
    frames = frames.cpu().numpy()
    recon_batch = recon_batch.cpu().numpy()
    
    # Plot original images with axes values
    for i in range(num_images):
        # Turn on axis to show coordinates
        axes[0, i].axis('on')
        
        # Plot the image with origin at lower left
        img_plot = axes[0, i].imshow(frames[i, 0], cmap='viridis', origin='lower')
        
        # Set x and y ticks to show coordinates
        img_height, img_width = frames[i, 0].shape
        x_ticks = np.linspace(0, img_width-1, 5, dtype=int)  # 5 ticks on x-axis
        y_ticks = np.linspace(0, img_height-1, 5, dtype=int)  # 5 ticks on y-axis
        
        axes[0, i].set_xticks(x_ticks)
        axes[0, i].set_yticks(y_ticks)
        
        # Add colorbar for each plot to show intensity values
        cbar = plt.colorbar(img_plot, ax=axes[0, i], fraction=0.046, pad=0.04)
        cbar.set_label('Intensity')
        
        # Add sample number as title
        axes[0, i].set_title(f'Original Sample {i+1}')
        
    # Plot reconstructed images with axes values
    for i in range(num_images):
        # Turn on axis
        axes[1, i].axis('on')
        
        # Plot the image with origin at lower left
        img_plot = axes[1, i].imshow(recon_batch[i, 0], cmap='viridis', origin='lower')
        
        # Set x and y ticks to show coordinates
        img_height, img_width = recon_batch[i, 0].shape
        x_ticks = np.linspace(0, img_width-1, 5, dtype=int)  # 5 ticks on x-axis
        y_ticks = np.linspace(0, img_height-1, 5, dtype=int)  # 5 ticks on y-axis
        
        axes[1, i].set_xticks(x_ticks)
        axes[1, i].set_yticks(y_ticks)
        
        # Add colorbar for each plot to show intensity values
        cbar = plt.colorbar(img_plot, ax=axes[1, i], fraction=0.046, pad=0.04)
        cbar.set_label('Intensity')
        
        # Add sample number as title
        axes[1, i].set_title(f'Reconstructed Sample {i+1}')
    
    plt.suptitle(f'Epoch {epoch}', fontsize=16)
    plt.tight_layout()
    
    # Save the figure
    save_path = os.path.join(reconstructions_dir, f'recon_epoch_{epoch}.png')
    plt.savefig(save_path, dpi=150)
    print(f"Saved reconstructions for epoch {epoch} to {save_path}")
    
    # Log the reconstruction plot to wandb
    wandb.log({f"reconstructions_epoch_{epoch}": wandb.Image(plt)})
    
    plt.close(fig)
    return recon_batch  # Return reconstructions for further analysis if needed

# Analyze latent space utilization


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Train VAE on PVI data')
    parser.add_argument('--output_dir', type=str, default='vae_output',
                        help='Directory for saving output files')
    parser.add_argument('--data_path', type=str, 
                        default=os.path.expanduser("~/phd/data/subject001_baseline_masked.h5"),
                        help='Path to the H5 data file')
    parser.add_argument('--latent_dim', type=int, default=256,  # Increased from 128 to 256
                        help='Dimension of latent space')
    parser.add_argument('--num_epochs', type=int, default=30,   # Increased from 20 to 30
                        help='Number of training epochs')
    parser.add_argument('--beta_min', type=float, default=0.01,
                        help='Minimum beta value for KL divergence weight')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Start learning rate')
    parser.add_argument('--beta_max', type=float, default=1.0,
                        help='Maximum beta value for KL divergence weight')
    parser.add_argument('--beta_warmup_epochs', type=int, default=15,  # Increased warmup period
                        help='Number of epochs to warm up beta')
    parser.add_argument('--wandb_project', type=str, default='vae-pvi',
                        help='Weights & Biases project name')
    parser.add_argument('--wandb_name', type=str, default=None,
                        help='Weights & Biases run name')
    parser.add_argument('--wandb_mode', type=str, default='offline',
                        choices=['online', 'offline', 'disabled'],
                        help='Weights & Biases mode')
    
    args = parser.parse_args()
    
    # Initialize wandb
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_name,
        config={
            "latent_dim": args.latent_dim,
            "num_epochs": args.num_epochs,
            "initial_lr": args.lr,
            "beta_min": args.beta_min,
            "beta_max": args.beta_max,
            "beta_warmup_epochs": args.beta_warmup_epochs,
            "batch_size": 16,
            "weight_decay": 1e-5,
            "dropout": 0.2,
            "frame_indices": [0, 100, 200, 300, 400],
            "data_path": args.data_path,
            "output_dir": args.output_dir
        },
        mode=args.wandb_mode
    )
    
    # Create output directory and subdirectories
    os.makedirs(args.output_dir, exist_ok=True)
    checkpoints_dir = os.path.join(args.output_dir, 'checkpoints')
    reconstructions_dir = os.path.join(args.output_dir, 'reconstructions')
    results_dir = os.path.join(args.output_dir, 'results')
    
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(reconstructions_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    wandb.config.update({"device": str(device)})
    
    # Set hyperparameters
    latent_dim = args.latent_dim
    num_epochs = args.num_epochs
    initial_lr = args.lr
    initial_beta = args.beta_min
    final_beta = args.beta_max
    beta_warmup_epochs = args.beta_warmup_epochs
    
    # Initialize the best model tracking
    best_loss = float('inf')
    best_epoch = 0
    best_model_state = None
    
    # Load dataset
    print(f"Loading dataset from: {args.data_path}")
    dataset = PviDataset(args.data_path)
    print(f"Dataset loaded with {len(dataset)} samples")
    wandb.config.update({"dataset_size": len(dataset)})
    
    # Create batch server
    batch_server = PviBatchServer(dataset, input_type="img", output_type="full")
    
    # Set batch size using the correct method
    batch_server.set_loader_params(batch_size=16, test_size=0.2)
    
    # Get loaders
    train_loader, test_loader = batch_server.get_loaders()
    
    # Initialize model
    model = VAE(latent_dim=latent_dim).to(device)
    print(f"Model initialized with latent dimension {latent_dim}")
    
    # Log model architecture to wandb
    #wandb.watch(model, log="all", log_freq=100)
    
    # Setup optimizer
    optimizer = optim.Adam(model.parameters(), lr=initial_lr, weight_decay=1e-5)  # Added weight decay for regularization
    
    # Setup learning rate scheduler - Use CosineAnnealingLR for smooth decay without restarts
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,  # Full cycle length equals total training epochs
        eta_min=1e-6       # Minimum learning rate
    )
    
    # Choose frames to use for training (multiple frames for more data)
    frame_indices = [0, 100, 200, 300, 400]  # Sample frames from different parts of the sequence
    
    # For recording losses
    epoch_losses = []
    recon_losses = []
    kl_losses = []
    learning_rates = []
    
    # For validation loss tracking
    val_losses = []
    val_recon_losses = []
    val_kl_losses = []
    
    # Training loop
    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = 0
        recon_loss_total = 0
        kl_loss_total = 0
        
        # Calculate beta for this epoch (linear annealing)
        if epoch <= beta_warmup_epochs:
            beta = initial_beta + (final_beta - initial_beta) * (epoch - 1) / (beta_warmup_epochs - 1)
        else:
            beta = final_beta
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)
        
        print(f"Epoch {epoch}, beta = {beta:.4f}, LR = {current_lr:.6f}")
        
        # Training
        for batch_idx, batch_data in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
            for frame_idx in frame_indices:
                try:
                    # Extract and normalize frames from this batch
                    frames = extract_frame(batch_data, frame_idx).to(device)
                    
                    # Forward pass
                    optimizer.zero_grad()
                    recon_batch, mu, logvar = model(frames)
                    
                    # Calculate loss with current beta
                    loss, recon, kl = vae_loss(recon_batch, frames, mu, logvar, beta)
                    
                    # Backward pass
                    loss.backward()
                    optimizer.step()
                    
                    # Accumulate losses
                    train_loss += loss.item()
                    recon_loss_total += recon.item()
                    kl_loss_total += kl.item()
                    
                    # Log batch-level metrics to wandb
                    if batch_idx % 10 == 0:  # Log every 10 batches
                        wandb.log({
                            "batch_loss": loss.item() / frames.size(0),
                            "batch_recon_loss": recon.item() / frames.size(0),
                            "batch_kl_loss": kl.item() / frames.size(0),
                            "batch": epoch * len(train_loader) + batch_idx
                        })
                    
                except Exception as e:
                    print(f"Error processing batch {batch_idx}, frame {frame_idx}: {e}")
                    continue
        
        # Calculate average training losses
        n_samples = len(train_loader) * len(frame_indices) * train_loader.batch_size
        avg_loss = train_loss / n_samples
        avg_recon = recon_loss_total / n_samples
        avg_kl = kl_loss_total / n_samples
        
        # Record training losses
        epoch_losses.append(avg_loss)
        recon_losses.append(avg_recon)
        kl_losses.append(avg_kl)
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_recon_loss = 0
        val_kl_loss = 0
        
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(tqdm(test_loader, desc=f"Validation Epoch {epoch}")):
                for frame_idx in frame_indices:
                    try:
                        # Extract and normalize frames from this batch
                        frames = extract_frame(batch_data, frame_idx).to(device)
                        
                        # Forward pass
                        recon_batch, mu, logvar = model(frames)
                        
                        # Calculate loss with current beta
                        loss, recon, kl = vae_loss(recon_batch, frames, mu, logvar, beta)
                        
                        # Accumulate losses
                        val_loss += loss.item()
                        val_recon_loss += recon.item()
                        val_kl_loss += kl.item()
                        
                    except Exception as e:
                        print(f"Error processing validation batch {batch_idx}, frame {frame_idx}: {e}")
                        continue
        
        # Calculate average validation losses
        n_val_samples = len(test_loader) * len(frame_indices) * test_loader.batch_size
        avg_val_loss = val_loss / n_val_samples
        avg_val_recon = val_recon_loss / n_val_samples
        avg_val_kl = val_kl_loss / n_val_samples
        
        # Record validation losses
        val_losses.append(avg_val_loss)
        val_recon_losses.append(avg_val_recon)
        val_kl_losses.append(avg_val_kl)
        
        # Print progress
        print(f'Epoch {epoch}: Train Loss = {avg_loss:.4f}, Val Loss = {avg_val_loss:.4f}, Train Recon = {avg_recon:.4f}, Val Recon = {avg_val_recon:.4f}')
        
        # Log epoch-level metrics to wandb
        wandb.log({
            "epoch": epoch,
            "train/total_loss": avg_loss,
            "train/reconstruction_loss": avg_recon,
            "train/kl_loss": avg_kl,
            "val/total_loss": avg_val_loss,
            "val/reconstruction_loss": avg_val_recon,
            "val/kl_loss": avg_val_kl,
            "learning_rate": current_lr,
            "beta": beta
        })
        
        # Update learning rate scheduler (regular step for CosineAnnealingLR)
        scheduler.step()
        
        # Check if this is the best model so far (using validation loss instead of training loss)
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_epoch = epoch
            best_model_state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_loss,
                'beta': beta,
                'latent_dim': latent_dim
            }
            print(f"New best model at epoch {epoch} with loss {best_loss:.4f}")
            
            # Log best model info to wandb
            wandb.run.summary["best_epoch"] = best_epoch
            wandb.run.summary["best_loss"] = best_loss
        
        # Visualize reconstructions on first epoch
        if epoch == 1:
            test_batch = next(iter(test_loader))
            visualize_reconstructions(model, test_batch, device, frame_idx=200, epoch=epoch, output_dir=args.output_dir)
            
        # Save checkpoint only every 5 epochs or at the last epoch
        if epoch % 5 == 0 or epoch == num_epochs:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_loss,
                'beta': beta,
                'latent_dim': latent_dim
            }, os.path.join(checkpoints_dir, f'vae_epoch_{epoch}.pt'))
            
            # Visualize reconstructions every 5 epochs or last epoch
            if epoch != 1:  # Skip if already done on first epoch
                test_batch = next(iter(test_loader))
                visualize_reconstructions(model, test_batch, device, frame_idx=200, epoch=epoch, output_dir=args.output_dir)
    
    # Save the best model at the end of training
    if best_model_state is not None:
        torch.save(best_model_state, os.path.join(checkpoints_dir, 'vae_best.pt'))
        print(f"Saved best model from epoch {best_epoch} with loss {best_loss:.4f}")
        
        # Save model artifact to wandb
        model_artifact = wandb.Artifact('vae_model', type='model')
        model_artifact.add_file(os.path.join(checkpoints_dir, 'vae_best.pt'))
        wandb.log_artifact(model_artifact)
    
    # Plot loss curves
    plt.figure(figsize=(20, 10))
    
    # Total Loss
    plt.subplot(2, 3, 1)
    plt.plot(range(1, num_epochs + 1), epoch_losses, label='Train')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Total Loss')
    plt.title('Total Loss vs. Epoch')
    plt.legend()
    plt.grid(True)
    
    # Reconstruction Loss
    plt.subplot(2, 3, 2)
    plt.plot(range(1, num_epochs + 1), recon_losses, label='Train Reconstruction')
    plt.plot(range(1, num_epochs + 1), val_recon_losses, label='Val Reconstruction')
    plt.xlabel('Epoch')
    plt.ylabel('Reconstruction Loss')
    plt.title('Reconstruction Loss vs. Epoch')
    plt.legend()
    plt.grid(True)
    
    # KL Divergence
    plt.subplot(2, 3, 3)
    plt.plot(range(1, num_epochs + 1), kl_losses, label='Train KL')
    plt.plot(range(1, num_epochs + 1), val_kl_losses, label='Val KL')
    plt.xlabel('Epoch')
    plt.ylabel('KL Divergence')
    plt.title('KL Divergence vs. Epoch')
    plt.legend()
    plt.grid(True)
    
    # Learning Rate
    plt.subplot(2, 3, 4)
    plt.plot(range(1, num_epochs + 1), learning_rates)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate vs. Epoch')
    plt.grid(True)
    
    # Beta Value
    plt.subplot(2, 3, 5)
    beta_values = []
    for epoch in range(1, num_epochs + 1):
        if epoch <= beta_warmup_epochs:
            beta = initial_beta + (final_beta - initial_beta) * (epoch - 1) / (beta_warmup_epochs - 1)
        else:
            beta = final_beta
        beta_values.append(beta)
    plt.plot(range(1, num_epochs + 1), beta_values)
    plt.xlabel('Epoch')
    plt.ylabel('Beta')
    plt.title('Beta vs. Epoch')
    plt.grid(True)
    
    # Train vs Val Loss Scatter
    # plt.subplot(2, 3, 6)
    # plt.scatter(epoch_losses, val_losses, c=range(1, num_epochs + 1), cmap='viridis')
    # plt.plot([min(epoch_losses), max(epoch_losses)], [min(epoch_losses), max(epoch_losses)], 'r--', label='y=x')
    # plt.xlabel('Training Loss')
    # plt.ylabel('Validation Loss')
    # plt.title('Training vs Validation Loss')
    # plt.colorbar(label='Epoch')
    # plt.legend()
    # plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'training_curves.png'), dpi=150)
    
    # Log final training curves to wandb
    wandb.log({"training_curves": wandb.Image(plt)})
    
    plt.close()
    
    print("Training completed!")
    print(f"Best model was at epoch {best_epoch} with loss {best_loss:.4f}")
    
    # Print directories
    print(f"All checkpoints saved in: {os.path.abspath(checkpoints_dir)}")
    print(f"All reconstructions saved in: {os.path.abspath(reconstructions_dir)}")
    print(f"Training results saved in: {os.path.abspath(results_dir)}")
    
    # Finish wandb run
    wandb.finish()


if __name__ == "__main__":
    main()