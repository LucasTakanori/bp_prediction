# VAE Model Evaluation Report

## Model Configuration
- Latent Dimension: 64
- Samples Analyzed: 400

## Reconstruction Quality
- Mean Magnitude: 4.2432
- Std Magnitude: 1.9455
- Activation Range: [-11.0790, 4.9517]
- Sparsity: 12.84%

## Feature Extraction Quality
- **Information Compression**: 20.38%
- **Dimension Utilization**: 16.43%
- **Effective Rank**: 13.0 / 64
- **Redundancy**: 0.268

## Dimensionality Analysis
- 80% Variance: 8 dimensions
- 90% Variance: 11 dimensions
- Effective Dimensions: 24
- Max Correlation: 0.865

## Recommendations for BiLSTM Integration
- Recommended latent dim: 13
- Feature selection: Use top 11 dimensions for 90% variance
- Information efficiency: 20.38%
