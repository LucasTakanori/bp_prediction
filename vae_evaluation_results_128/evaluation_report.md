# VAE Model Evaluation Report

## Model Configuration
- Latent Dimension: 128
- Samples Analyzed: 400

## Reconstruction Quality
- Mean Magnitude: 3.6154
- Std Magnitude: 2.0743
- Activation Range: [-7.4455, 8.5294]
- Sparsity: 20.98%

## Feature Extraction Quality
- **Information Compression**: 7.96%
- **Dimension Utilization**: 5.20%
- **Effective Rank**: 10.2 / 128
- **Redundancy**: 0.275

## Dimensionality Analysis
- 80% Variance: 7 dimensions
- 90% Variance: 9 dimensions
- Effective Dimensions: 37
- Max Correlation: 0.810

## Recommendations for BiLSTM Integration
- Recommended latent dim: 10
- Feature selection: Use top 9 dimensions for 90% variance
- Information efficiency: 7.96%
