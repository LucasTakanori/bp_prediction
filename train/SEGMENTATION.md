The Segmentation Process
867 Periods → 400 Samples
Looking at your data_utils.py, the key is in the mask metadata:
pythondef _stack_samples(self):
    num_samples = len(self._h5meta['mask'])  # ← 400 samples defined by mask
    print(f'Stacking samples ({num_samples} total):')
    
    for k in range(num_samples):
        # Each sample takes MULTIPLE periods:
        sl = slice(*self._h5meta['mask'][k])  # ← Slice of periods for this sample
        
        # Stack multiple periods into one sample
        tup = self._h5data[signal_key][signal_field][sl]
        sample[signal_key][signal_field] = torch.cat(tup, dim=-1)  # ← Concatenate periods
What the Mask Does:
The mask defines meaningful segments in your physiological recording:

867 periods = individual cardiac/breathing cycles
400 samples = grouped periods that belong together
Each sample contains ~2-3 periods on average (867/400 ≈ 2.17)

For example:
Periods:    [1] [2] [3] [4] [5] [6] [7] [8] [9] ...
           └─────────┘ └─────────┘ └─────┘
Sample 1:   periods 1-3   Sample 2:   periods 4-6   Sample 3: periods 7-8
Why Each Sample Has 500 Frames:
From your debug output: torch.Size([32, 32, 500])
This means each sample contains multiple periods concatenated together, resulting in ~500 time frames total per sample.
Why This Segmentation Makes Sense:

Physiological grouping: Periods that belong to the same breathing cycle or measurement session
Temporal context: Each sample contains enough time to capture meaningful dynamics
Training stability: Longer sequences provide more context for learning

For Your VAE Training:
This is actually perfect for your setup! Each sample provides:

500 frames of 32×32 images
Rich temporal information across multiple cardiac cycles
200,000 total frames for training (400 samples × 500 frames/sample)

The VAE will learn to represent individual frames from these longer physiological sequences, which will give you a robust feature extractor for your eventual BiLSTM pipeline.
Bottom line: The mask-based segmentation groups related physiological periods together, giving you more meaningful training samples with richer temporal context! 🎯