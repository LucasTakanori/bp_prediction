#!/usr/bin/env python3
"""
Complete Training Workflow: VAE + Enhanced BiLSTM for Blood Pressure Prediction

This script demonstrates the complete pipeline from VAE training to BiLSTM evaluation.
It can be run as a standalone script or imported for custom workflows.

Author: Your Name
Date: 2024
"""

import os
import subprocess
import sys
import argparse
from datetime import datetime
from pathlib import Path
import json


class BPPredictionPipeline:
    """Complete pipeline for blood pressure prediction using VAE + BiLSTM"""
    
    def __init__(self, data_path, base_output_dir=None, latent_dim=256):
        """
        Initialize the training pipeline
        
        Args:
            data_path: Path to the H5 data file
            base_output_dir: Base directory for all outputs
            latent_dim: Latent dimension for VAE
        """
        self.data_path = data_path
        self.latent_dim = latent_dim
        
        # Create timestamped output directory
        if base_output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_output_dir = f"./experiments/bp_prediction_{timestamp}"
        
        self.base_output_dir = Path(base_output_dir)
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Define subdirectories
        self.vae_output_dir = self.base_output_dir / "vae_training"
        self.results_dir = self.base_output_dir / "results"
        
        # Store training configurations
        self.configs = {
            "data_path": str(data_path),
            "latent_dim": latent_dim,
            "base_output_dir": str(self.base_output_dir),
            "timestamp": datetime.now().isoformat()
        }
        
        print(f"Pipeline initialized:")
        print(f"  Data: {self.data_path}")
        print(f"  Output: {self.base_output_dir}")
        print(f"  Latent dim: {self.latent_dim}")
    
    def train_vae(self, num_epochs=30, beta_min=0.01, beta_max=1.0, 
                  beta_warmup_epochs=15, wandb_project="bp-vae", wandb_mode="offline"):
        """
        Train the VAE model
        
        Args:
            num_epochs: Number of training epochs
            beta_min: Minimum beta value for KL weight
            beta_max: Maximum beta value for KL weight
            beta_warmup_epochs: Epochs for beta warmup
            wandb_project: Weights & Biases project name
            wandb_mode: W&B logging mode (online/offline/disabled)
        
        Returns:
            Path to the best VAE checkpoint
        """
        print("\n" + "="*60)
        print("TRAINING VAE")
        print("="*60)
        
        # Build VAE training command
        cmd = [
            sys.executable, "tuned_vae.py",
            "--output_dir", str(self.vae_output_dir),
            "--data_path", str(self.data_path),
            "--latent_dim", str(self.latent_dim),
            "--num_epochs", str(num_epochs),
            "--beta_min", str(beta_min),
            "--beta_max", str(beta_max),
            "--beta_warmup_epochs", str(beta_warmup_epochs),
            "--wandb_project", wandb_project,
            "--wandb_name", f"vae_{datetime.now().strftime('%H%M%S')}",
            "--wandb_mode", wandb_mode
        ]
        
        # Store VAE config
        self.configs["vae"] = {
            "num_epochs": num_epochs,
            "beta_min": beta_min,
            "beta_max": beta_max,
            "beta_warmup_epochs": beta_warmup_epochs,
            "command": " ".join(cmd)
        }
        
        print(f"Running command: {' '.join(cmd)}")
        
        # Run VAE training
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print("VAE training completed successfully!")
            
            # Check for checkpoint
            vae_checkpoint = self.vae_output_dir / "checkpoints" / "vae_best.pt"
            if vae_checkpoint.exists():
                print(f"VAE checkpoint saved at: {vae_checkpoint}")
                self.configs["vae"]["checkpoint"] = str(vae_checkpoint)
                return vae_checkpoint
            else:
                raise FileNotFoundError(f"VAE checkpoint not found at {vae_checkpoint}")
                
        except subprocess.CalledProcessError as e:
            print(f"VAE training failed: {e}")
            print(f"Error output: {e.stderr}")
            raise e
    
    def train_bilstm(self, vae_checkpoint, loss_type="composite", use_attention=True,
                     num_epochs=25, lstm_hidden_dim=256, lstm_layers=3,
                     attention_dim=128, batch_size=8, visualize_attention=False,
                     wandb_project="bp-bilstm", wandb_mode="offline"):
        """
        Train the enhanced BiLSTM model
        
        Args:
            vae_checkpoint: Path to trained VAE checkpoint
            loss_type: Loss function type (mse, systolic_distance, diastolic_distance, composite)
            use_attention: Whether to use attention mechanism
            num_epochs: Number of training epochs
            lstm_hidden_dim: LSTM hidden dimension
            lstm_layers: Number of LSTM layers
            attention_dim: Attention mechanism dimension
            batch_size: Training batch size
            visualize_attention: Whether to visualize attention patterns
            wandb_project: Weights & Biases project name
            wandb_mode: W&B logging mode
        
        Returns:
            Path to the best BiLSTM checkpoint
        """
        print("\n" + "="*60)
        print(f"TRAINING BILSTM (loss_type={loss_type}, attention={use_attention})")
        print("="*60)
        
        # Create output directory for this configuration
        bilstm_output_dir = self.base_output_dir / f"bilstm_{loss_type}_attention_{use_attention}"
        
        # Build BiLSTM training command
        cmd = [
            sys.executable, "enhanced_vae_bilstm.py",
            "--output_dir", str(bilstm_output_dir),
            "--data_path", str(self.data_path),
            "--vae_checkpoint", str(vae_checkpoint),
            "--latent_dim", str(self.latent_dim),
            "--lstm_hidden_dim", str(lstm_hidden_dim),
            "--lstm_layers", str(lstm_layers),
            "--num_epochs", str(num_epochs),
            "--batch_size", str(batch_size),
            "--attention_dim", str(attention_dim),
            "--loss_type", loss_type,
            "--wandb_project", wandb_project,
            "--wandb_name", f"bilstm_{loss_type}_{datetime.now().strftime('%H%M%S')}",
            "--wandb_mode", wandb_mode
        ]
        
        # Add optional flags
        if use_attention:
            cmd.append("--use_attention")
        if visualize_attention:
            cmd.append("--visualize_attention")
        
        # Store BiLSTM config
        config_key = f"bilstm_{loss_type}_attention_{use_attention}"
        self.configs[config_key] = {
            "loss_type": loss_type,
            "use_attention": use_attention,
            "num_epochs": num_epochs,
            "lstm_hidden_dim": lstm_hidden_dim,
            "lstm_layers": lstm_layers,
            "attention_dim": attention_dim,
            "batch_size": batch_size,
            "visualize_attention": visualize_attention,
            "output_dir": str(bilstm_output_dir),
            "command": " ".join(cmd)
        }
        
        print(f"Running command: {' '.join(cmd)}")
        
        # Run BiLSTM training
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"BiLSTM training with {loss_type} completed successfully!")
            
            # Check for checkpoint
            bilstm_checkpoint = bilstm_output_dir / "checkpoints" / "enhanced_bilstm_best.pt"
            if bilstm_checkpoint.exists():
                print(f"BiLSTM checkpoint saved at: {bilstm_checkpoint}")
                self.configs[config_key]["checkpoint"] = str(bilstm_checkpoint)
                return bilstm_checkpoint
            else:
                raise FileNotFoundError(f"BiLSTM checkpoint not found at {bilstm_checkpoint}")
                
        except subprocess.CalledProcessError as e:
            print(f"BiLSTM training failed: {e}")
            print(f"Error output: {e.stderr}")
            raise e
    
    def run_ablation_study(self, vae_checkpoint, wandb_project="bp-ablation"):
        """
        Run ablation study with different loss functions
        
        Args:
            vae_checkpoint: Path to trained VAE checkpoint
            wandb_project: Weights & Biases project name
        
        Returns:
            Dictionary with all trained model checkpoints
        """
        print("\n" + "="*60)
        print("RUNNING ABLATION STUDY")
        print("="*60)
        
        # Define configurations to test
        ablation_configs = [
            {"loss_type": "mse", "use_attention": False, "epochs": 20},
            {"loss_type": "systolic_distance", "use_attention": True, "epochs": 20},
            {"loss_type": "diastolic_distance", "use_attention": True, "epochs": 20},
            {"loss_type": "composite", "use_attention": True, "epochs": 25, "visualize": True}
        ]
        
        results = {}
        
        for i, config in enumerate(ablation_configs, 1):
            print(f"\nAblation {i}/{len(ablation_configs)}: {config['loss_type']}")
            
            try:
                checkpoint = self.train_bilstm(
                    vae_checkpoint=vae_checkpoint,
                    loss_type=config["loss_type"],
                    use_attention=config["use_attention"],
                    num_epochs=config["epochs"],
                    visualize_attention=config.get("visualize", False),
                    wandb_project=wandb_project
                )
                results[config["loss_type"]] = checkpoint
                
            except Exception as e:
                print(f"Failed to train {config['loss_type']}: {e}")
                results[config["loss_type"]] = None
        
        return results
    
    def save_configuration(self):
        """Save the complete pipeline configuration"""
        config_file = self.base_output_dir / "pipeline_config.json"
        
        with open(config_file, 'w') as f:
            json.dump(self.configs, f, indent=2)
        
        print(f"Pipeline configuration saved to: {config_file}")
        return config_file
    
    def generate_summary_report(self):
        """Generate a summary report of the training pipeline"""
        summary_file = self.base_output_dir / "training_summary.txt"
        
        with open(summary_file, 'w') as f:
            f.write("Blood Pressure Prediction Training Pipeline Summary\n")
            f.write("="*60 + "\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Data: {self.data_path}\n")
            f.write(f"Latent Dimension: {self.latent_dim}\n")
            f.write(f"Output Directory: {self.base_output_dir}\n\n")
            
            # VAE section
            if "vae" in self.configs:
                f.write("VAE Training:\n")
                f.write("-" * 20 + "\n")
                vae_config = self.configs["vae"]
                f.write(f"Epochs: {vae_config['num_epochs']}\n")
                f.write(f"Beta range: {vae_config['beta_min']} -> {vae_config['beta_max']}\n")
                f.write(f"Beta warmup: {vae_config['beta_warmup_epochs']} epochs\n")
                if "checkpoint" in vae_config:
                    f.write(f"Checkpoint: {vae_config['checkpoint']}\n")
                f.write("\n")
            
            # BiLSTM sections
            bilstm_configs = {k: v for k, v in self.configs.items() if k.startswith("bilstm_")}
            if bilstm_configs:
                f.write("BiLSTM Models:\n")
                f.write("-" * 20 + "\n")
                for i, (name, config) in enumerate(bilstm_configs.items(), 1):
                    f.write(f"{i}. {name}:\n")
                    f.write(f"   Loss Type: {config['loss_type']}\n")
                    f.write(f"   Attention: {config['use_attention']}\n")
                    f.write(f"   Epochs: {config['num_epochs']}\n")
                    f.write(f"   LSTM Hidden: {config['lstm_hidden_dim']}\n")
                    f.write(f"   LSTM Layers: {config['lstm_layers']}\n")
                    if "checkpoint" in config:
                        f.write(f"   Checkpoint: {config['checkpoint']}\n")
                    f.write(f"   Output: {config['output_dir']}\n\n")
            
            f.write("\nFrame Pattern: 10 frames (k-7 to k+2)\n")
            f.write("For detailed results, check evaluation_metrics.txt in each model's results folder.\n")
        
        print(f"Summary report saved to: {summary_file}")
        return summary_file


def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description="Blood Pressure Prediction Training Pipeline")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to the H5 data file")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Base output directory (default: timestamped)")
    parser.add_argument("--latent_dim", type=int, default=256,
                        help="VAE latent dimension")
    parser.add_argument("--vae_epochs", type=int, default=30,
                        help="VAE training epochs")
    parser.add_argument("--bilstm_epochs", type=int, default=25,
                        help="BiLSTM training epochs")
    parser.add_argument("--loss_type", type=str, default="composite",
                        choices=["mse", "systolic_distance", "diastolic_distance", "composite"],
                        help="BiLSTM loss function type")
    parser.add_argument("--no_attention", action="store_true",
                        help="Disable attention mechanism")
    parser.add_argument("--ablation", action="store_true",
                        help="Run ablation study with all loss types")
    parser.add_argument("--wandb_mode", type=str, default="offline",
                        choices=["online", "offline", "disabled"],
                        help="Weights & Biases mode")
    parser.add_argument("--wandb_project", type=str, default="bp-prediction",
                        help="Weights & Biases project name")
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = BPPredictionPipeline(
        data_path=args.data_path,
        base_output_dir=args.output_dir,
        latent_dim=args.latent_dim
    )
    
    try:
        # Train VAE
        vae_checkpoint = pipeline.train_vae(
            num_epochs=args.vae_epochs,
            wandb_project=f"{args.wandb_project}-vae",
            wandb_mode=args.wandb_mode
        )
        
        if args.ablation:
            # Run ablation study
            pipeline.run_ablation_study(
                vae_checkpoint=vae_checkpoint,
                wandb_project=f"{args.wandb_project}-ablation"
            )
        else:
            # Train single BiLSTM model
            pipeline.train_bilstm(
                vae_checkpoint=vae_checkpoint,
                loss_type=args.loss_type,
                use_attention=not args.no_attention,
                num_epochs=args.bilstm_epochs,
                visualize_attention=(args.loss_type == "composite"),
                wandb_project=f"{args.wandb_project}-bilstm",
                wandb_mode=args.wandb_mode
            )
        
        # Save configuration and generate report
        pipeline.save_configuration()
        pipeline.generate_summary_report()
        
        print("\n" + "="*60)
        print("PIPELINE COMPLETED SUCCESSFULLY! 🎉")
        print("="*60)
        print(f"All results saved in: {pipeline.base_output_dir}")
        
    except Exception as e:
        print(f"\nPipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()


# =============================================================================
# USAGE EXAMPLES
# =============================================================================

"""
Example 1: Basic pipeline
python training_workflow.py --data_path ~/phd/data/subject001_baseline_masked.h5

Example 2: High-performance configuration
python training_workflow.py \
    --data_path ~/phd/data/subject001_baseline_masked.h5 \
    --latent_dim 512 \
    --vae_epochs 40 \
    --bilstm_epochs 30 \
    --wandb_mode online

Example 3: Systolic focus
python training_workflow.py \
    --data_path ~/phd/data/subject001_baseline_masked.h5 \
    --loss_type systolic_distance \
    --wandb_project systolic-bp-focus

Example 4: Ablation study
python training_workflow.py \
    --data_path ~/phd/data/subject001_baseline_masked.h5 \
    --ablation

Example 5: No attention baseline
python training_workflow.py \
    --data_path ~/phd/data/subject001_baseline_masked.h5 \
    --loss_type mse \
    --no_attention

Example 6: Custom output directory
python training_workflow.py \
    --data_path ~/phd/data/subject001_baseline_masked.h5 \
    --output_dir ./my_experiment \
    --wandb_mode disabled
"""