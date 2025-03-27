import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

class VAE(nn.Module):
    """
    Variational Autoencoder for encoding PVI images into a latent space
    """
    def __init__(self, image_size=32, image_channels=1, latent_dim=64):
        super(VAE, self).__init__()
        
        # Encoder network
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            
            nn.Flatten()
        )
        
        # Calculate flattened size after convolutions
        conv_output_size = 256 * (image_size // 16) * (image_size // 16)
        
        # Mean and log variance projections for latent space
        self.fc_mu = nn.Linear(conv_output_size, latent_dim)
        self.fc_logvar = nn.Linear(conv_output_size, latent_dim)
        
        # Decoder network
        self.decoder_input = nn.Linear(latent_dim, conv_output_size)
        
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (256, image_size // 16, image_size // 16)),
            
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            
            nn.ConvTranspose2d(32, image_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
        
    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        h = self.decoder_input(z)
        return self.decoder(h)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar


class ImageDataset(Dataset):
    """
    Dataset for training the VAE on individual images from the sequence
    """
    def __init__(self, dataloader):
        self.images = []
        
        for batch in tqdm(dataloader, desc="Building image dataset"):
            # Extract images
            img_batch = batch['pviHP']  # [batch_size, 1, 32, 32, 500]
            
            # Process each sample
            for i in range(img_batch.size(0)):
                # Get all 500 images for this sample
                sample_images = img_batch[i]  # [1, 32, 32, 500]
                
                # Transpose to get [500, 1, 32, 32]
                sample_images = sample_images.permute(3, 0, 1, 2)
                
                # Add to our dataset
                self.images.append(sample_images)
        
        # Concatenate all images
        self.images = torch.cat(self.images, dim=0)
        
        # Normalize images to [0, 1] range if not already normalized
        if self.images.max() > 1.0:
            print("Normalizing images to [0, 1] range...")
            self.images = self.images / self.images.max()
        
        # Check for NaN or infinite values
        if torch.isnan(self.images).any() or torch.isinf(self.images).any():
            print("WARNING: Dataset contains NaN or infinite values. Replacing with zeros...")
            self.images = torch.nan_to_num(self.images, nan=0.0, posinf=1.0, neginf=0.0)
        
        # Print dataset statistics
        print(f"Created dataset with {len(self.images)} images")
        print(f"Image tensor shape: {self.images.shape}")
        print(f"Image value range: [{self.images.min().item():.6f}, {self.images.max().item():.6f}]")
        print(f"Image mean: {self.images.mean().item():.6f}, std: {self.images.std().item():.6f}")
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        return self.images[idx]


def train_vae(vae, dataloader, num_epochs=5, learning_rate=5e-5, device='cuda', checkpoint_dir='checkpoints'):
    """Train the VAE model on image data"""
    print("Preparing VAE training data...")
    optimizer = torch.optim.Adam(vae.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    vae.to(device)
    
    # Create checkpoint directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Create a dataset of individual images
    image_dataset = ImageDataset(dataloader)
    image_loader = DataLoader(image_dataset, batch_size=64, shuffle=True, num_workers=4)
    
    # Track losses
    train_losses = []
    recon_losses = []
    kl_losses = []
    
    print("Starting VAE training...")
    for epoch in range(num_epochs):
        vae.train()
        total_loss = 0
        recon_loss_sum = 0
        kl_loss_sum = 0
        
        progress_bar = tqdm(image_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for images in progress_bar:
            images = images.to(device)
            
            # Forward pass
            recon_images, mu, logvar = vae(images)
            
            # Reconstruction loss with numerical stability
            recon_loss = F.mse_loss(recon_images, images)
            
            # KL divergence with numerical stability safeguards
            # Clamp logvar.exp() to prevent numerical instability
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - torch.clamp(logvar.exp(), min=1e-10, max=1e10))
            
            # Total loss (with KL weight annealing)
            kl_weight = min(1.0, (epoch + 1) / (num_epochs / 2))  # Gradually increase from 0 to 1
            loss = recon_loss + kl_weight * kl_loss
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            recon_loss_sum += recon_loss.item()
            kl_loss_sum += kl_loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}" if not torch.isnan(loss).any() else "NaN",
                'recon': f"{recon_loss.item():.4f}" if not torch.isnan(recon_loss).any() else "NaN",
                'kl': f"{kl_loss.item():.4f}" if not torch.isnan(kl_loss).any() else "NaN"
            })
        
        # Calculate average losses
        avg_loss = total_loss / len(image_loader)
        avg_recon_loss = recon_loss_sum / len(image_loader)
        avg_kl_loss = kl_loss_sum / len(image_loader)
        
        # Update learning rate
        scheduler.step(avg_loss)
        
        # Print epoch results
        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Loss: {avg_loss:.4f}, '
              f'Recon Loss: {avg_recon_loss:.4f}, '
              f'KL Loss: {avg_kl_loss:.4f}, '
              f'LR: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # Track losses
        train_losses.append(avg_loss)
        recon_losses.append(avg_recon_loss)
        kl_losses.append(avg_kl_loss)
        
        # Save model checkpoint
        if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
            checkpoint_path = os.path.join(checkpoint_dir, f'vae_checkpoint_epoch_{epoch+1}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': vae.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'recon_loss': avg_recon_loss,
                'kl_loss': avg_kl_loss
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")
    
    # Plot training losses
    plt.figure(figsize=(12, 8))
    
    plt.subplot(3, 1, 1)
    plt.plot(train_losses)
    plt.title('VAE Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(3, 1, 2)
    plt.plot(recon_losses)
    plt.title('Reconstruction Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(3, 1, 3)
    plt.plot(kl_losses)
    plt.title('KL Divergence Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.tight_layout()
    plt.savefig(os.path.join(checkpoint_dir, 'vae_training_losses.png'))
    
    # Save final model
    final_model_path = os.path.join(checkpoint_dir, 'vae_final.pt')
    torch.save(vae.state_dict(), final_model_path)
    print(f"Final model saved to {final_model_path}")
    
    print("VAE training complete!")
    return vae


def main():
    """Main function to train the VAE model"""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load the dataset - add path to import properly
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from utils.data_utils import DataPathManager, PviDataset, PviBatchServer
    
    # Set data paths
    subject_id = 'subject001'
    session = 'baseline'
    root = "/home/lucas_takanori/phd/data"
    output_dir = "vae_output"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
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
    train_loader, _ = batch_server.get_loaders()
    
    # Print data shapes
    data_shapes = batch_server.get_data_shapes()
    print("Data shapes:", data_shapes)
    
    # Define VAE hyperparameters
    latent_dim = 64  # Dimensionality of the latent space
    
    # Add initialization check function
    def check_model_init(model):
        """Verify model weights are properly initialized"""
        has_nan = False
        for name, param in model.named_parameters():
            if torch.isnan(param).any():
                print(f"WARNING: NaN values found in {name}")
                has_nan = True
        return not has_nan
    num_epochs = 5
    learning_rate = 5e-5  # Reduced learning rate for stability
    
    # Initialize and train VAE
    print("Initializing VAE...")
    vae = VAE(image_size=32, image_channels=1, latent_dim=latent_dim)
    
    # Verify model initialization
    if check_model_init(vae):
        print("Model initialization verified: No NaN values in parameters")
    else:
        print("ERROR: Model has NaN values in parameters. Fix initialization.")
    
    print("Training VAE...")
    trained_vae = train_vae(
        vae=vae,
        dataloader=train_loader,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        device=device,
        checkpoint_dir=output_dir
    )
    
    # Visualize reconstructions (optional)
    # Could add code here to visualize some example reconstructions
    
    print("VAE training process completed successfully!")


if __name__ == "__main__":
    main()