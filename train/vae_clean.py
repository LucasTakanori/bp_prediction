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
from torch.utils.data import Dataset, DataLoader

# Add the parent directory to the path to find utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Import your dataset utilities
from utils.data_utils import PviDataset, PviBatchServer

# Define the VAE model (keeping your architecture)
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


# OPTIMIZED: Efficient frame dataset with subsampling
class EfficientFrameDataset(Dataset):
    """Efficient dataset that samples frames on-the-fly - much faster!"""
    
    def __init__(self, original_dataset, frames_per_sample=10, seed=42):
        self.original_dataset = original_dataset
        self.frames_per_sample = frames_per_sample
        self.sample_indices = []
        
        print(f"Creating efficient frame dataset (sampling {frames_per_sample} frames per sample)...")
        
        # Pre-compute valid frame indices for each sample
        torch.manual_seed(seed)  # For reproducible sampling
        for sample_idx in range(len(original_dataset)):
            sample = original_dataset[sample_idx]
            
            try:
                pvi_data = sample['pviHP']['img']
                num_frames = pvi_data.shape[-1]
                
                # Sample frames_per_sample random indices from this sample
                if num_frames >= frames_per_sample:
                    # Random sampling without replacement
                    frame_indices = torch.randperm(num_frames)[:frames_per_sample].tolist()
                else:
                    # If fewer frames available, use all + repeat some
                    frame_indices = list(range(num_frames))
                    while len(frame_indices) < frames_per_sample:
                        frame_indices.extend(range(min(num_frames, frames_per_sample - len(frame_indices))))
                
                # Store (sample_idx, frame_idx) pairs
                for frame_idx in frame_indices:
                    self.sample_indices.append((sample_idx, frame_idx))
                    
            except Exception as e:
                print(f"Warning: Skipping sample {sample_idx}: {e}")
                continue
        
        print(f"Total frame indices created: {len(self.sample_indices)}")
        print(f"Effective frames per epoch: {len(self.sample_indices)} (vs 200,000 previously)")
    
    def __len__(self):
        return len(self.sample_indices)
    
    def __getitem__(self, idx):
        sample_idx, frame_idx = self.sample_indices[idx]
        
        # Get sample and extract specific frame
        sample = self.original_dataset[sample_idx]
        pvi_data = sample['pviHP']['img']
        
        # Extract frame: [32, 32]
        frame = pvi_data[:, :, frame_idx]
        frame = torch.nan_to_num(frame, nan=0.0)
        
        # Add channel dimension: [1, 32, 32]
        return frame.unsqueeze(0)


# Simplified loss function - no weird scaling
def vae_loss(recon_x, x, mu, logvar, beta):
    """VAE loss = reconstruction loss + beta * KL divergence"""
    # Clamp to valid tanh range
    x = torch.clamp(x, -1.0, 1.0)
    recon_x = torch.clamp(recon_x, -1.0, 1.0)
    
    # Simple MSE loss
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    
    # KL divergence
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Total loss
    total_loss = recon_loss + beta * kl_div
    
    return total_loss, recon_loss, kl_div


# Simplified visualization function
def visualize_reconstructions(model, frames, device, num_images=5, epoch=0, output_dir=None):
    reconstructions_dir = os.path.join(output_dir, 'reconstructions')
    
    model.eval()
    
    with torch.no_grad():
        # Use first few frames from the batch
        frames = frames[:num_images].to(device)
        
        # Get reconstructions
        recon_batch, _, _ = model(frames)
    
    # Create a figure
    fig, axes = plt.subplots(2, num_images, figsize=(15, 6))
    
    # Convert tensors to numpy for plotting
    frames = frames.cpu().numpy()
    recon_batch = recon_batch.cpu().numpy()
    
    # Plot original and reconstructed images
    for i in range(num_images):
        # Original
        axes[0, i].imshow(frames[i, 0], cmap='viridis', origin='lower')
        axes[0, i].set_title(f'Original {i+1}')
        axes[0, i].axis('on')
        
        # Reconstructed
        axes[1, i].imshow(recon_batch[i, 0], cmap='viridis', origin='lower')
        axes[1, i].set_title(f'Reconstructed {i+1}')
        axes[1, i].axis('on')
    
    plt.suptitle(f'Epoch {epoch}', fontsize=16)
    plt.tight_layout()
    
    # Save the figure
    save_path = os.path.join(reconstructions_dir, f'recon_epoch_{epoch}.png')
    plt.savefig(save_path, dpi=150)
    print(f"Saved reconstructions for epoch {epoch} to {save_path}")
    
    # Log to wandb
    wandb.log({f"reconstructions_epoch_{epoch}": wandb.Image(plt)})
    
    plt.close(fig)
    return recon_batch


def main():
    # Set up argument parser (keeping your arguments)
    parser = argparse.ArgumentParser(description='Train VAE on PVI data - Optimized Version')
    parser.add_argument('--output_dir', type=str, default='vae_output',
                        help='Directory for saving output files')
    parser.add_argument('--data_path', type=str, 
                        default=os.path.expanduser("~/phd/data/subject001_baseline_masked.h5"),
                        help='Path to the H5 data file')
    parser.add_argument('--latent_dim', type=int, default=256,
                        help='Dimension of latent space')
    parser.add_argument('--num_epochs', type=int, default=30,
                        help='Number of training epochs')
    parser.add_argument('--beta_min', type=float, default=0.01,
                        help='Minimum beta value for KL divergence weight')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Start learning rate')
    parser.add_argument('--beta_max', type=float, default=1.0,
                        help='Maximum beta value for KL divergence weight')
    parser.add_argument('--beta_warmup_epochs', type=int, default=15,
                        help='Number of epochs to warm up beta')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--frames_per_sample', type=int, default=20,
                        help='Number of frames to sample per data sample (controls epoch size)')
    parser.add_argument('--wandb_project', type=str, default='vae-pvi',
                        help='Weights & Biases project name')
    parser.add_argument('--wandb_name', type=str, default=None,
                        help='Weights & Biases run name')
    parser.add_argument('--wandb_mode', type=str, default='offline',
                        choices=['online', 'offline', 'disabled'],
                        help='Weights & Biases mode')
    
    args = parser.parse_args()
    
    # Initialize wandb (simplified config)
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
            "batch_size": args.batch_size,
            "frames_per_sample": args.frames_per_sample,
            "weight_decay": 1e-5,
            "dropout": 0.2,
            "training_approach": "efficient_sampling",  # Updated!
            "data_path": args.data_path,
            "output_dir": args.output_dir
        },
        mode=args.wandb_mode
    )
    
    # Create output directories
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
    
    # Load dataset
    print(f"Loading dataset from: {args.data_path}")
    dataset = PviDataset(args.data_path)
    print(f"Dataset loaded with {len(dataset)} samples")
    
    # OPTIMIZED: Create efficient frame dataset
    frame_dataset = EfficientFrameDataset(dataset, frames_per_sample=args.frames_per_sample)
    wandb.config.update({"total_frames": len(frame_dataset)})
    
    # Split into train/val (80/20)
    train_size = int(0.8 * len(frame_dataset))
    val_size = len(frame_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        frame_dataset, [train_size, val_size]
    )
    
    # Create dataloaders with more workers for speed
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                             num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                           num_workers=4, pin_memory=True)
    
    print(f"Training frames: {len(train_dataset)}")
    print(f"Validation frames: {len(val_dataset)}")
    print(f"Batches per epoch: {len(train_loader)}")
    print(f"⚡ Optimization: ~{args.frames_per_sample}x fewer frames per epoch vs full sequential")
    
    # Initialize model
    model = VAE(latent_dim=args.latent_dim).to(device)
    print(f"Model initialized with latent dimension {args.latent_dim}")
    
    # Setup optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.num_epochs,
        eta_min=1e-6
    )
    
    # Initialize tracking
    best_loss = float('inf')
    best_epoch = 0
    best_model_state = None
    
    # Training history
    epoch_losses = []
    recon_losses = []
    kl_losses = []
    learning_rates = []
    val_losses = []
    val_recon_losses = []
    val_kl_losses = []
    
    # OPTIMIZED TRAINING LOOP
    for epoch in range(1, args.num_epochs + 1):
        # Beta annealing
        if epoch <= args.beta_warmup_epochs:
            beta = args.beta_min + (args.beta_max - args.beta_min) * (epoch - 1) / (args.beta_warmup_epochs - 1)
        else:
            beta = args.beta_max
        
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)
        
        print(f"Epoch {epoch}, beta = {beta:.4f}, LR = {current_lr:.6f}")
        
        # Training phase
        model.train()
        train_loss = 0
        train_recon_loss = 0
        train_kl_loss = 0
        
        for batch_idx, frames in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch}")):
            frames = frames.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(frames)
            
            # Calculate loss
            loss, recon, kl = vae_loss(recon_batch, frames, mu, logvar, beta)
            
            loss.backward()
            optimizer.step()
            
            # Accumulate losses
            train_loss += loss.item()
            train_recon_loss += recon.item()
            train_kl_loss += kl.item()
            
            # Log batch metrics occasionally
            if batch_idx % 50 == 0:  # More frequent logging for faster feedback
                wandb.log({
                    "batch_loss": loss.item() / frames.size(0),
                    "batch_recon_loss": recon.item() / frames.size(0),
                    "batch_kl_loss": kl.item() / frames.size(0),
                    "batch": epoch * len(train_loader) + batch_idx
                })
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_recon_loss = 0
        val_kl_loss = 0
        
        with torch.no_grad():
            for frames in tqdm(val_loader, desc=f"Validation Epoch {epoch}"):
                frames = frames.to(device, non_blocking=True)
                recon_batch, mu, logvar = model(frames)
                
                loss, recon, kl = vae_loss(recon_batch, frames, mu, logvar, beta)
                
                val_loss += loss.item()
                val_recon_loss += recon.item()
                val_kl_loss += kl.item()
        
        # Calculate average losses
        avg_train_loss = train_loss / len(train_dataset)
        avg_val_loss = val_loss / len(val_dataset)
        avg_train_recon = train_recon_loss / len(train_dataset)
        avg_val_recon = val_recon_loss / len(val_dataset)
        avg_train_kl = train_kl_loss / len(train_dataset)
        avg_val_kl = val_kl_loss / len(val_dataset)
        
        # Record losses
        epoch_losses.append(avg_train_loss)
        recon_losses.append(avg_train_recon)
        kl_losses.append(avg_train_kl)
        val_losses.append(avg_val_loss)
        val_recon_losses.append(avg_val_recon)
        val_kl_losses.append(avg_val_kl)
        
        # Print progress
        print(f'Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}')
        print(f'           Train Recon = {avg_train_recon:.4f}, Val Recon = {avg_val_recon:.4f}')
        
        # Log to wandb
        wandb.log({
            "epoch": epoch,
            "train/total_loss": avg_train_loss,
            "train/reconstruction_loss": avg_train_recon,
            "train/kl_loss": avg_train_kl,
            "val/total_loss": avg_val_loss,
            "val/reconstruction_loss": avg_val_recon,
            "val/kl_loss": avg_val_kl,
            "learning_rate": current_lr,
            "beta": beta
        })
        
        # Update scheduler
        scheduler.step()
        
        # Check for best model
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_epoch = epoch
            best_model_state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'beta': beta,
                'latent_dim': args.latent_dim,
                'frames_per_sample': args.frames_per_sample
            }
            print(f"New best model at epoch {epoch} with val loss {best_loss:.4f}")
            
            wandb.run.summary["best_epoch"] = best_epoch
            wandb.run.summary["best_loss"] = best_loss
        
        # VISUALIZE RECONSTRUCTIONS AFTER EVERY EPOCH (as requested!)
        sample_frames = next(iter(val_loader))
        visualize_reconstructions(model, sample_frames, device, epoch=epoch, output_dir=args.output_dir)
        
        # Save checkpoint every 10 epochs or at the last epoch
        if epoch % 10 == 0 or epoch == args.num_epochs:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'beta': beta,
                'latent_dim': args.latent_dim,
                'frames_per_sample': args.frames_per_sample
            }, os.path.join(checkpoints_dir, f'vae_epoch_{epoch}.pt'))
    
    # Save best model
    if best_model_state is not None:
        torch.save(best_model_state, os.path.join(checkpoints_dir, 'vae_best.pt'))
        print(f"Saved best model from epoch {best_epoch} with loss {best_loss:.4f}")
        
        # Save model artifact to wandb
        model_artifact = wandb.Artifact('vae_model', type='model')
        model_artifact.add_file(os.path.join(checkpoints_dir, 'vae_best.pt'))
        wandb.log_artifact(model_artifact)
    
    # Plot training curves
    plt.figure(figsize=(15, 10))
    
    # Total Loss
    plt.subplot(2, 3, 1)
    plt.plot(range(1, args.num_epochs + 1), epoch_losses, label='Train')
    plt.plot(range(1, args.num_epochs + 1), val_losses, label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Total Loss')
    plt.title('Total Loss vs. Epoch')
    plt.legend()
    plt.grid(True)
    
    # Reconstruction Loss
    plt.subplot(2, 3, 2)
    plt.plot(range(1, args.num_epochs + 1), recon_losses, label='Train Reconstruction')
    plt.plot(range(1, args.num_epochs + 1), val_recon_losses, label='Val Reconstruction')
    plt.xlabel('Epoch')
    plt.ylabel('Reconstruction Loss')
    plt.title('Reconstruction Loss vs. Epoch')
    plt.legend()
    plt.grid(True)
    
    # KL Divergence
    plt.subplot(2, 3, 3)
    plt.plot(range(1, args.num_epochs + 1), kl_losses, label='Train KL')
    plt.plot(range(1, args.num_epochs + 1), val_kl_losses, label='Val KL')
    plt.xlabel('Epoch')
    plt.ylabel('KL Divergence')
    plt.title('KL Divergence vs. Epoch')
    plt.legend()
    plt.grid(True)
    
    # Learning Rate
    plt.subplot(2, 3, 4)
    plt.plot(range(1, args.num_epochs + 1), learning_rates)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate vs. Epoch')
    plt.grid(True)
    
    # Beta Value
    plt.subplot(2, 3, 5)
    beta_values = []
    for epoch in range(1, args.num_epochs + 1):
        if epoch <= args.beta_warmup_epochs:
            beta = args.beta_min + (args.beta_max - args.beta_min) * (epoch - 1) / (args.beta_warmup_epochs - 1)
        else:
            beta = args.beta_max
        beta_values.append(beta)
    plt.plot(range(1, args.num_epochs + 1), beta_values)
    plt.xlabel('Epoch')
    plt.ylabel('Beta')
    plt.title('Beta vs. Epoch')
    plt.grid(True)
    
    # Summary info
    plt.subplot(2, 3, 6)
    plt.text(0.1, 0.8, f"Training Summary:", fontsize=12, fontweight='bold')
    plt.text(0.1, 0.7, f"Total frames: {len(frame_dataset)}", fontsize=10)
    plt.text(0.1, 0.6, f"Train frames: {len(train_dataset)}", fontsize=10)
    plt.text(0.1, 0.5, f"Val frames: {len(val_dataset)}", fontsize=10)
    plt.text(0.1, 0.4, f"Frames/sample: {args.frames_per_sample}", fontsize=10)
    plt.text(0.1, 0.3, f"Best epoch: {best_epoch}", fontsize=10)
    plt.text(0.1, 0.2, f"Best val loss: {best_loss:.4f}", fontsize=10)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.axis('off')
    plt.title('Training Configuration')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'training_curves.png'), dpi=150)
    wandb.log({"training_curves": wandb.Image(plt)})
    plt.close()
    
    print("Training completed!")
    print(f"⚡ OPTIMIZED: Much faster epochs!")
    print(f"📸 Reconstructions saved after every epoch!")
    print(f"✅ Best model was at epoch {best_epoch} with loss {best_loss:.4f}")
    print(f"✅ All checkpoints saved in: {os.path.abspath(checkpoints_dir)}")
    print(f"✅ All reconstructions saved in: {os.path.abspath(reconstructions_dir)}")
    print(f"✅ Training results saved in: {os.path.abspath(results_dir)}")
    
    # Finish wandb
    wandb.finish()


if __name__ == "__main__":
    main()