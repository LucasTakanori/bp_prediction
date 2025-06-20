#!/usr/bin/env python3
import torch
import os
import glob

def check_vae_dimensions():
    # Find all VAE checkpoints
    vae_files = glob.glob('./*/checkpoints/vae_best.pt', recursive=True)
    vae_files.extend(glob.glob('./train/*/checkpoints/vae_best.pt', recursive=True))
    
    print("Checking VAE checkpoint dimensions:")
    print("=" * 50)
    
    for vae_file in vae_files:
        try:
            state = torch.load(vae_file, map_location='cpu')
            if 'fc_mu.weight' in state:
                latent_dim = state['fc_mu.weight'].shape[0]
                print(f"{vae_file}: latent_dim = {latent_dim}")
            else:
                print(f"{vae_file}: No fc_mu.weight found")
        except Exception as e:
            print(f"{vae_file}: Error - {e}")

if __name__ == "__main__":
    check_vae_dimensions() 