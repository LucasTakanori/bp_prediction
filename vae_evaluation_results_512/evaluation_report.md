# VAE Model Evaluation Report

## Model Configuration
- Latent Dimension: 512
- Samples Analyzed: 400

## Reconstruction Quality
- Mean Magnitude: 4.0020
- Std Magnitude: 1.8875
- Activation Range: [-9.2449, 6.8153]
- Sparsity: 45.31%

## Feature Extraction Quality
- **Information Compression**: 1.93%
- **Dimension Utilization**: 1.56%
- **Effective Rank**: 9.9 / 512
- **Redundancy**: 0.389

## Dimensionality Analysis
- 80% Variance: 6 dimensions
- 90% Variance: 8 dimensions
- Effective Dimensions: 25
- Max Correlation: 0.927

## Recommendations for BiLSTM Integration
- Recommended latent dim: 10
- Feature selection: Use top 8 dimensions for 90% variance
- Information efficiency: 1.93%
