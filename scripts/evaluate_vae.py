#!/usr/bin/env python3
"""
VAE Model Evaluation Script
Evaluates VAE reconstruction quality and latent space properties for feature extraction
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import sys
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import pandas as pd
from scipy import stats
import json

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from train.tuned_vae import VAE, extract_frame  # Use the working VAE implementation
from utils.data_utils import PviDataset, PviBatchServer


class VAEEvaluator:
    """Comprehensive VAE evaluation focused on feature extraction capabilities"""
    
    def __init__(self, model_path, data_path, latent_dim=512, device='auto'):
        self.device = torch.device('cuda' if device == 'auto' and torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        self.latent_dim = latent_dim
        
        # Load model
        self.model = VAE(latent_dim=latent_dim).to(self.device)
        self._load_model(model_path)
        self.model.eval()
        
        # Load data
        self.dataset = PviDataset(data_path)
        self.batch_server = PviBatchServer(self.dataset, input_type="image", output_type="minmax")
        self.batch_server.set_loader_params(batch_size=16, test_size=0.2)
        _, self.test_loader = self.batch_server.get_loaders()
        
        print(f"Loaded VAE model with latent dimension: {latent_dim}")
        print(f"Dataset size: {len(self.dataset)}")
    
    def _load_model(self, model_path):
        """Load trained VAE model"""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        print(f"Model loaded from: {model_path}")
    
    def extract_embeddings_and_reconstructions(self, num_samples=1000, frame_indices=None):
        """Extract embeddings and reconstructions for analysis"""
        if frame_indices is None:
            frame_indices = [0, 125, 250, 375, 499]  # Sample frames across sequence
        
        all_embeddings = []
        all_originals = []
        all_reconstructions = []
        all_frame_labels = []
        
        samples_collected = 0
        
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(tqdm(self.test_loader, desc="Extracting embeddings")):
                if samples_collected >= num_samples:
                    break
                
                for frame_idx in frame_indices:
                    try:
                        # Extract frame
                        frames = extract_frame(batch_data, frame_idx).to(self.device)
                        
                        # Get embeddings (mu from encoder)
                        mu, logvar = self.model.encode(frames)
                        
                        # Get reconstructions
                        reconstructions = self.model.decode(mu)
                        
                        # Store data
                        all_embeddings.append(mu.cpu().numpy())
                        all_originals.append(frames.cpu().numpy())
                        all_reconstructions.append(reconstructions.cpu().numpy())
                        all_frame_labels.extend([frame_idx] * frames.size(0))
                        
                        samples_collected += frames.size(0)
                        
                        if samples_collected >= num_samples:
                            break
                            
                    except Exception as e:
                        print(f"Error processing batch {batch_idx}, frame {frame_idx}: {e}")
                        continue
        
        # Concatenate all data
        embeddings = np.concatenate(all_embeddings, axis=0)[:num_samples]
        originals = np.concatenate(all_originals, axis=0)[:num_samples]
        reconstructions = np.concatenate(all_reconstructions, axis=0)[:num_samples]
        frame_labels = np.array(all_frame_labels)[:num_samples]
        
        print(f"Extracted {len(embeddings)} samples with {len(frame_indices)} frame types")
        return embeddings, originals, reconstructions, frame_labels
    
    def visualize_reconstructions(self, originals, reconstructions, output_dir, num_examples=10):
        """Visualize reconstructions with value ranges in color"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create comprehensive reconstruction visualization
        fig, axes = plt.subplots(4, num_examples, figsize=(2*num_examples, 8))
        
        for i in range(num_examples):
            orig = originals[i, 0]  # Remove channel dimension
            recon = reconstructions[i, 0]
            
            # Original image with colorbar
            im1 = axes[0, i].imshow(orig, cmap='viridis', aspect='auto')
            axes[0, i].set_title(f'Original\nRange: [{orig.min():.3f}, {orig.max():.3f}]', fontsize=8)
            axes[0, i].axis('off')
            plt.colorbar(im1, ax=axes[0, i], fraction=0.046, pad=0.04)
            
            # Reconstruction with colorbar
            im2 = axes[1, i].imshow(recon, cmap='viridis', aspect='auto')
            axes[1, i].set_title(f'Reconstruction\nRange: [{recon.min():.3f}, {recon.max():.3f}]', fontsize=8)
            axes[1, i].axis('off')
            plt.colorbar(im2, ax=axes[1, i], fraction=0.046, pad=0.04)
            
            # Difference map
            diff = np.abs(orig - recon)
            im3 = axes[2, i].imshow(diff, cmap='hot', aspect='auto')
            axes[2, i].set_title(f'Abs Difference\nMAE: {diff.mean():.4f}', fontsize=8)
            axes[2, i].axis('off')
            plt.colorbar(im3, ax=axes[2, i], fraction=0.046, pad=0.04)
            
            # Relative error
            rel_error = np.abs(orig - recon) / (np.abs(orig) + 1e-8)
            im4 = axes[3, i].imshow(rel_error, cmap='plasma', aspect='auto')
            axes[3, i].set_title(f'Relative Error\nMean: {rel_error.mean():.4f}', fontsize=8)
            axes[3, i].axis('off')
            plt.colorbar(im4, ax=axes[3, i], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'vae_reconstructions_detailed.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create summary statistics plot
        self._plot_reconstruction_statistics(originals, reconstructions, output_dir)
    
    def _plot_reconstruction_statistics(self, originals, reconstructions, output_dir):
        """Plot detailed reconstruction statistics"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Calculate metrics
        mse_per_sample = np.mean((originals - reconstructions)**2, axis=(1, 2, 3))
        mae_per_sample = np.mean(np.abs(originals - reconstructions), axis=(1, 2, 3))
        
        # Value range comparison
        orig_mins = np.min(originals, axis=(1, 2, 3))
        orig_maxs = np.max(originals, axis=(1, 2, 3))
        recon_mins = np.min(reconstructions, axis=(1, 2, 3))
        recon_maxs = np.max(reconstructions, axis=(1, 2, 3))
        
        # MSE distribution
        axes[0, 0].hist(mse_per_sample, bins=50, alpha=0.7, edgecolor='black')
        axes[0, 0].set_title(f'MSE Distribution\nMean: {mse_per_sample.mean():.4f}')
        axes[0, 0].set_xlabel('MSE')
        axes[0, 0].set_ylabel('Count')
        
        # MAE distribution
        axes[0, 1].hist(mae_per_sample, bins=50, alpha=0.7, edgecolor='black')
        axes[0, 1].set_title(f'MAE Distribution\nMean: {mae_per_sample.mean():.4f}')
        axes[0, 1].set_xlabel('MAE')
        axes[0, 1].set_ylabel('Count')
        
        # Value range preservation
        axes[0, 2].scatter(orig_mins, recon_mins, alpha=0.5, label='Min values')
        axes[0, 2].scatter(orig_maxs, recon_maxs, alpha=0.5, label='Max values')
        axes[0, 2].plot([orig_mins.min(), orig_maxs.max()], [orig_mins.min(), orig_maxs.max()], 'r--', label='Perfect')
        axes[0, 2].set_xlabel('Original Values')
        axes[0, 2].set_ylabel('Reconstructed Values')
        axes[0, 2].set_title('Value Range Preservation')
        axes[0, 2].legend()
        
        # Pixel-wise correlation
        orig_flat = originals.reshape(-1)
        recon_flat = reconstructions.reshape(-1)
        correlation = np.corrcoef(orig_flat, recon_flat)[0, 1]
        
        # Sample 10k points for scatter plot
        sample_indices = np.random.choice(len(orig_flat), 10000, replace=False)
        axes[1, 0].scatter(orig_flat[sample_indices], recon_flat[sample_indices], alpha=0.1, s=1)
        axes[1, 0].plot([orig_flat.min(), orig_flat.max()], [orig_flat.min(), orig_flat.max()], 'r--')
        axes[1, 0].set_xlabel('Original Pixel Values')
        axes[1, 0].set_ylabel('Reconstructed Pixel Values')
        axes[1, 0].set_title(f'Pixel Correlation: {correlation:.4f}')
        
        # Error vs original value
        errors = np.abs(orig_flat - recon_flat)
        axes[1, 1].scatter(orig_flat[sample_indices], errors[sample_indices], alpha=0.1, s=1)
        axes[1, 1].set_xlabel('Original Pixel Values')
        axes[1, 1].set_ylabel('Absolute Error')
        axes[1, 1].set_title('Error vs Original Value')
        
        # Distribution comparison
        axes[1, 2].hist(orig_flat[sample_indices], bins=50, alpha=0.5, label='Original', density=True)
        axes[1, 2].hist(recon_flat[sample_indices], bins=50, alpha=0.5, label='Reconstructed', density=True)
        axes[1, 2].set_xlabel('Pixel Values')
        axes[1, 2].set_ylabel('Density')
        axes[1, 2].set_title('Value Distribution Comparison')
        axes[1, 2].legend()
        
        plt.tight_layout()
        plt.savefig(output_dir / 'vae_reconstruction_statistics.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def analyze_latent_space(self, embeddings, frame_labels, output_dir):
        """Analyze latent space properties for feature extraction"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print("Analyzing latent space properties...")
        
        # Basic statistics
        stats_dict = {
            'latent_dim': embeddings.shape[1],
            'num_samples': embeddings.shape[0],
            'mean_magnitude': float(np.linalg.norm(embeddings, axis=1).mean()),
            'std_magnitude': float(np.linalg.norm(embeddings, axis=1).std()),
            'mean_activation': float(embeddings.mean()),
            'std_activation': float(embeddings.std()),
            'activation_range': [float(embeddings.min()), float(embeddings.max())],
            'sparsity': float((np.abs(embeddings) < 0.01).mean()),
        }
        
        # Analyze embedding distribution
        self._analyze_embedding_distribution(embeddings, output_dir, stats_dict)
        
        # Dimensionality analysis
        self._analyze_dimensionality(embeddings, frame_labels, output_dir, stats_dict)
        
        # Feature extraction quality metrics
        self._compute_feature_quality_metrics(embeddings, frame_labels, output_dir, stats_dict)
        
        # Save statistics
        with open(output_dir / 'latent_space_stats.json', 'w') as f:
            json.dump(stats_dict, f, indent=2)
        
        return stats_dict
    
    def _analyze_embedding_distribution(self, embeddings, output_dir, stats_dict):
        """Analyze the distribution of embeddings"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Activation distribution
        axes[0, 0].hist(embeddings.flatten(), bins=100, alpha=0.7, edgecolor='black')
        axes[0, 0].set_title('Activation Distribution')
        axes[0, 0].set_xlabel('Activation Value')
        axes[0, 0].set_ylabel('Count')
        
        # Magnitude distribution
        magnitudes = np.linalg.norm(embeddings, axis=1)
        axes[0, 1].hist(magnitudes, bins=50, alpha=0.7, edgecolor='black')
        axes[0, 1].set_title(f'Embedding Magnitude\nMean: {magnitudes.mean():.3f}')
        axes[0, 1].set_xlabel('L2 Norm')
        axes[0, 1].set_ylabel('Count')
        
        # Dimension-wise variance
        dim_vars = np.var(embeddings, axis=0)
        axes[0, 2].plot(dim_vars)
        axes[0, 2].set_title('Dimension-wise Variance')
        axes[0, 2].set_xlabel('Latent Dimension')
        axes[0, 2].set_ylabel('Variance')
        
        # Correlation matrix (sample of dimensions)
        sample_dims = min(50, embeddings.shape[1])
        corr_matrix = np.corrcoef(embeddings[:, :sample_dims].T)
        im = axes[1, 0].imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        axes[1, 0].set_title('Dimension Correlation (first 50 dims)')
        plt.colorbar(im, ax=axes[1, 0])
        
        # Cumulative variance explained
        pca = PCA()
        pca.fit(embeddings)
        cumvar = np.cumsum(pca.explained_variance_ratio_)
        axes[1, 1].plot(cumvar)
        axes[1, 1].set_title('Cumulative Variance Explained')
        axes[1, 1].set_xlabel('Number of Components')
        axes[1, 1].set_ylabel('Cumulative Variance Ratio')
        
        # Add horizontal lines for common thresholds
        for threshold in [0.8, 0.9, 0.95]:
            idx = np.argmax(cumvar >= threshold)
            axes[1, 1].axhline(y=threshold, color='red', linestyle='--', alpha=0.5)
            axes[1, 1].text(idx, threshold, f'{threshold:.0%} at dim {idx}', fontsize=8)
        
        # Sparsity analysis
        sparsity_levels = []
        thresholds = np.logspace(-3, 0, 20)
        for thresh in thresholds:
            sparsity_levels.append((np.abs(embeddings) < thresh).mean())
        
        axes[1, 2].semilogx(thresholds, sparsity_levels)
        axes[1, 2].set_title('Sparsity vs Threshold')
        axes[1, 2].set_xlabel('Threshold')
        axes[1, 2].set_ylabel('Fraction of Values Below Threshold')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'latent_space_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Update stats
        stats_dict.update({
            'variance_explained_80pct': int(np.argmax(cumvar >= 0.8)),
            'variance_explained_90pct': int(np.argmax(cumvar >= 0.9)),
            'effective_dimensions': int(np.sum(dim_vars > dim_vars.mean() * 0.1)),
            'max_correlation': float(np.max(np.abs(corr_matrix[np.triu_indices_from(corr_matrix, k=1)])))
        })
    
    def _analyze_dimensionality(self, embeddings, frame_labels, output_dir, stats_dict):
        """Analyze dimensionality and clustering properties"""
        print("Computing dimensionality reduction...")
        
        # PCA
        pca = PCA(n_components=2)
        embeddings_pca = pca.fit_transform(embeddings)
        
        # t-SNE (on subset for speed)
        subset_size = min(1000, len(embeddings))
        subset_idx = np.random.choice(len(embeddings), subset_size, replace=False)
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, subset_size//4))
        embeddings_tsne = tsne.fit_transform(embeddings[subset_idx])
        
        # Plotting
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # PCA colored by frame
        unique_frames = np.unique(frame_labels)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_frames)))
        
        for i, frame in enumerate(unique_frames):
            mask = frame_labels == frame
            axes[0, 0].scatter(embeddings_pca[mask, 0], embeddings_pca[mask, 1], 
                             c=[colors[i]], label=f'Frame {frame}', alpha=0.6, s=10)
        axes[0, 0].set_title('PCA - Colored by Frame Index')
        axes[0, 0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)')
        axes[0, 0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)')
        axes[0, 0].legend()
        
        # PCA density
        axes[0, 1].hexbin(embeddings_pca[:, 0], embeddings_pca[:, 1], gridsize=30, cmap='Blues')
        axes[0, 1].set_title('PCA - Density Plot')
        axes[0, 1].set_xlabel('PC1')
        axes[0, 1].set_ylabel('PC2')
        
        # t-SNE colored by frame
        frame_subset = frame_labels[subset_idx]
        for i, frame in enumerate(unique_frames):
            mask = frame_subset == frame
            if np.any(mask):
                axes[1, 0].scatter(embeddings_tsne[mask, 0], embeddings_tsne[mask, 1], 
                                 c=[colors[i]], label=f'Frame {frame}', alpha=0.6, s=10)
        axes[1, 0].set_title('t-SNE - Colored by Frame Index')
        axes[1, 0].set_xlabel('t-SNE 1')
        axes[1, 0].set_ylabel('t-SNE 2')
        axes[1, 0].legend()
        
        # t-SNE density
        axes[1, 1].hexbin(embeddings_tsne[:, 0], embeddings_tsne[:, 1], gridsize=30, cmap='Blues')
        axes[1, 1].set_title('t-SNE - Density Plot')
        axes[1, 1].set_xlabel('t-SNE 1')
        axes[1, 1].set_ylabel('t-SNE 2')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'latent_space_projections.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _compute_feature_quality_metrics(self, embeddings, frame_labels, output_dir, stats_dict):
        """Compute metrics specific to feature extraction quality"""
        
        # Information content
        # 1. Effective rank (numerical rank of covariance matrix)
        cov_matrix = np.cov(embeddings.T)
        eigenvals = np.linalg.eigvals(cov_matrix)
        eigenvals = eigenvals[eigenvals > 0]  # Remove zero eigenvalues
        effective_rank = np.exp(-np.sum((eigenvals / eigenvals.sum()) * np.log(eigenvals / eigenvals.sum())))
        
        # 2. Participation ratio
        participation_ratio = (eigenvals.sum() ** 2) / (eigenvals ** 2).sum()
        
        # 3. Redundancy (average correlation between dimensions)
        corr_matrix = np.corrcoef(embeddings.T)
        off_diag_corr = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]
        redundancy = np.mean(np.abs(off_diag_corr))
        
        # Create feature quality summary plot
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Eigenvalue spectrum
        axes[0, 0].semilogy(sorted(eigenvals, reverse=True))
        axes[0, 0].set_title(f'Eigenvalue Spectrum\nEffective Rank: {effective_rank:.1f}')
        axes[0, 0].set_xlabel('Component')
        axes[0, 0].set_ylabel('Eigenvalue')
        
        # Correlation heatmap (subset)
        subset_size = min(50, len(corr_matrix))
        im = axes[0, 1].imshow(corr_matrix[:subset_size, :subset_size], cmap='coolwarm', vmin=-1, vmax=1)
        axes[0, 1].set_title(f'Correlation Matrix\nMean |Corr|: {redundancy:.3f}')
        plt.colorbar(im, ax=axes[0, 1])
        
        # Feature importance (variance-based)
        feature_importance = np.var(embeddings, axis=0)
        axes[1, 0].plot(sorted(feature_importance, reverse=True))
        axes[1, 0].set_title('Feature Importance (Variance)')
        axes[1, 0].set_xlabel('Feature Rank')
        axes[1, 0].set_ylabel('Variance')
        axes[1, 0].set_yscale('log')
        
        # Summary metrics
        metrics_text = f"""Feature Quality Metrics:
        
Effective Rank: {effective_rank:.1f} / {embeddings.shape[1]}
Participation Ratio: {participation_ratio:.1f}
Redundancy: {redundancy:.3f}

        Information Compression: {float(effective_rank)/embeddings.shape[1]:.2%}
        Dimension Utilization: {float(participation_ratio)/embeddings.shape[1]:.2%}
        """
        
        axes[1, 1].text(0.05, 0.95, metrics_text, transform=axes[1, 1].transAxes, 
                        verticalalignment='top', fontfamily='monospace', fontsize=10)
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'feature_quality_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Update stats
        stats_dict.update({
            'effective_rank': float(effective_rank),
            'participation_ratio': float(participation_ratio),
            'redundancy': float(redundancy),
            'information_compression_ratio': float(effective_rank / embeddings.shape[1]),
            'dimension_utilization_ratio': float(participation_ratio / embeddings.shape[1])
        })
    
    def run_evaluation(self, output_dir, num_samples=1000):
        """Run complete VAE evaluation"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print("Starting VAE evaluation...")
        
        # Extract embeddings and reconstructions
        embeddings, originals, reconstructions, frame_labels = self.extract_embeddings_and_reconstructions(num_samples)
        
        # Visualize reconstructions
        print("Creating reconstruction visualizations...")
        self.visualize_reconstructions(originals, reconstructions, output_dir)
        
        # Analyze latent space
        print("Analyzing latent space...")
        stats = self.analyze_latent_space(embeddings, frame_labels, output_dir)
        
        # Create summary report
        self._create_summary_report(stats, output_dir)
        
        print(f"Evaluation complete! Results saved to: {output_dir}")
        return stats
    
    def _create_summary_report(self, stats, output_dir):
        """Create a comprehensive summary report"""
        report = f"""# VAE Model Evaluation Report

## Model Configuration
- Latent Dimension: {stats['latent_dim']}
- Samples Analyzed: {stats['num_samples']}

## Reconstruction Quality
- Mean Magnitude: {stats['mean_magnitude']:.4f}
- Std Magnitude: {stats['std_magnitude']:.4f}
- Activation Range: [{stats['activation_range'][0]:.4f}, {stats['activation_range'][1]:.4f}]
- Sparsity: {stats['sparsity']:.2%}

## Feature Extraction Quality
- **Information Compression**: {stats.get('information_compression_ratio', 0):.2%}
- **Dimension Utilization**: {stats.get('dimension_utilization_ratio', 0):.2%}
- **Effective Rank**: {stats.get('effective_rank', 0):.1f} / {stats['latent_dim']}
- **Redundancy**: {stats.get('redundancy', 0):.3f}

## Dimensionality Analysis
- 80% Variance: {stats.get('variance_explained_80pct', 0)} dimensions
- 90% Variance: {stats.get('variance_explained_90pct', 0)} dimensions
- Effective Dimensions: {stats.get('effective_dimensions', 0)}
- Max Correlation: {stats.get('max_correlation', 0):.3f}

## Recommendations for BiLSTM Integration
- Recommended latent dim: {min(stats.get('effective_rank', stats['latent_dim']), 256):.0f}
- Feature selection: Use top {stats.get('variance_explained_90pct', stats['latent_dim']):.0f} dimensions for 90% variance
- Information efficiency: {stats.get('information_compression_ratio', 0):.2%}
"""
        
        with open(output_dir / 'evaluation_report.md', 'w') as f:
            f.write(report)
        
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        print(f"Information Compression: {stats.get('information_compression_ratio', 0):.2%}")
        print(f"Dimension Utilization: {stats.get('dimension_utilization_ratio', 0):.2%}")
        print(f"Effective Dimensions: {stats.get('effective_dimensions', 0)} / {stats['latent_dim']}")
        print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Evaluate VAE model for feature extraction')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained VAE model checkpoint')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to data file')
    parser.add_argument('--output_dir', type=str, default='vae_evaluation',
                        help='Output directory for evaluation results')
    parser.add_argument('--num_samples', type=int, default=1000,
                        help='Number of samples to analyze')
    parser.add_argument('--latent_dim', type=int, default=512,
                        help='Latent dimension of the model')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (auto, cpu, cuda)')
    
    args = parser.parse_args()
    
    # Run evaluation
    evaluator = VAEEvaluator(
        model_path=args.model_path,
        data_path=args.data_path,
        latent_dim=args.latent_dim,
        device=args.device
    )
    
    stats = evaluator.run_evaluation(
        output_dir=args.output_dir,
        num_samples=args.num_samples
    )


if __name__ == "__main__":
    main() 