"""
FilterAugment for SpecNoise Project
Adapted from: https://github.com/DCASE-REPO/DESED_task
Paper: https://arxiv.org/abs/2107.03649 (ICASSP 2022)

FilterAugment applies frequency-band filtering to spectrograms by:
1. Dividing the frequency axis into random number of bands
2. Applying different dB gain to each band (step or linear interpolation)
3. Simulates realistic acoustic environment variations
"""

import torch
import numpy as np


class FilterAugment:
    """
    FilterAugment data augmentation for spectrograms.
    
    Args:
        db_range (list): Range of dB gain [min, max], default [-6, 6]
        n_band (list): Range of number of frequency bands [min, max], default [3, 6]
        min_bw (int): Minimum bandwidth for each band, default 6
        filter_type (str): Type of filter - "linear" or "step", default "linear"
    """
    def __init__(self, db_range=[-3, 3], n_band=[3, 6], min_bw=6, filter_type="linear"):
        self.db_range = db_range
        self.n_band = n_band
        self.min_bw = min_bw
        self.filter_type = filter_type
        self.training = True  # Add training mode flag
    
    def __call__(self, features):
        """
        Apply FilterAugment to spectrogram features.
        
        Args:
            features (torch.Tensor): Input spectrogram tensor of shape (C, F, T) or (B, C, F, T)
            
        Returns:
            torch.Tensor: Augmented spectrogram tensor
        """
        # Only apply augmentation during training
        if not self.training:
            return features
        
        # Handle different input shapes
        if features.dim() == 3:
            # Single sample: (C, F, T) -> add batch dimension
            features = features.unsqueeze(0)
            return_single = True
        elif features.dim() == 4:
            # Batch: (B, C, F, T)
            return_single = False
        else:
            raise ValueError(f"Expected 3D or 4D tensor, got {features.dim()}D")
        
        batch_size, n_channels, n_freq_bin, n_time = features.shape
        
        # Generate ONE filter for all channels (they contain the same spectrogram)
        # Filter is generated in dB scale to match the input data
        freq_filt_db = self._generate_filter(batch_size, n_freq_bin, features)
        
        # Apply the same filter to all channels using ADDITION (not multiplication)
        # Because input data is in dB/log scale, filtering should be additive
        # freq_filt_db shape: (B, F, 1) -> (B, 1, F, 1) for broadcasting
        augmented_features = features + freq_filt_db.unsqueeze(1)
        
        if return_single:
            return augmented_features.squeeze(0)
        return augmented_features
    
    def train(self, mode=True):
        """Set training mode"""
        self.training = mode
        return self
    
    def eval(self):
        """Set evaluation mode"""
        self.training = False
        return self
    
    def _generate_filter(self, batch_size, n_freq_bin, features):
        """
        Generate frequency filter (same for all channels).
        
        Args:
            features (torch.Tensor): (B, C, F, T)
            batch_size (int): Batch size
            n_freq_bin (int): Number of frequency bins
            
        Returns:
            torch.Tensor: Frequency filter in dB scale of shape (B, F, 1)
        """
        # Randomly select number of frequency bands
        # Note: torch.randint(low, high) is [low, high), so we use n_band[1] directly (not +1)
        n_freq_band = torch.randint(low=self.n_band[0], high=self.n_band[1], size=(1,)).item()
        
        # Adjust minimum bandwidth if necessary
        min_bw = self.min_bw
        if n_freq_band > 1:
            while n_freq_bin - n_freq_band * min_bw + 1 < 0:
                min_bw -= 1
            
            # Generate random band boundaries
            band_bndry_freqs = torch.sort(
                torch.randint(0, n_freq_bin - n_freq_band * min_bw + 1, (n_freq_band - 1,))
            )[0] + torch.arange(1, n_freq_band) * min_bw
            band_bndry_freqs = torch.cat((
                torch.tensor([0]), 
                band_bndry_freqs, 
                torch.tensor([n_freq_bin])
            ))
            
            if self.filter_type == "step":
                # Step filter: constant gain within each band
                band_factors = torch.rand((batch_size, n_freq_band)).to(features) * \
                               (self.db_range[1] - self.db_range[0]) + self.db_range[0]
                # band_factors is already in dB scale, no need to convert
                
                freq_filt = torch.zeros((batch_size, n_freq_bin, 1)).to(features)
                for i in range(n_freq_band):
                    freq_filt[:, band_bndry_freqs[i]:band_bndry_freqs[i + 1], :] = \
                        band_factors[:, i].unsqueeze(-1).unsqueeze(-1)
            
            elif self.filter_type == "linear":
                # Linear filter: linear interpolation between band boundaries
                band_factors = torch.rand((batch_size, n_freq_band + 1)).to(features) * \
                               (self.db_range[1] - self.db_range[0]) + self.db_range[0]
                freq_filt = torch.zeros((batch_size, n_freq_bin, 1)).to(features)
                
                for i in range(n_freq_band):
                    for j in range(batch_size):
                        freq_filt[j, band_bndry_freqs[i]:band_bndry_freqs[i+1], :] = \
                            torch.linspace(
                                band_factors[j, i], 
                                band_factors[j, i+1],
                                band_bndry_freqs[i+1] - band_bndry_freqs[i]
                            ).unsqueeze(-1)
                # freq_filt is already in dB scale
            
            else:
                raise ValueError(f"Unknown filter_type: {self.filter_type}")
            
            return freq_filt
        else:
            # Only 1 band, return zero filter (no change in dB domain)
            return torch.zeros((batch_size, n_freq_bin, 1)).to(features)


class FilterAugmentProbabilistic:
    """
    Probabilistic FilterAugment with random filter type selection.
    
    This version randomly chooses between "step" and "linear" filters,
    similar to the updated FilterAugment algorithm.
    
    Args:
        db_range (list): Range of dB gain [min, max], default [-6, 6]
        prob_step (float): Probability of using step filter (0-1), default 0.5
    """
    def __init__(self, db_range=[-6, 6], prob_step=0.5):
        self.db_range = db_range
        self.prob_step = prob_step
    
    def __call__(self, features):
        """
        Apply probabilistic FilterAugment.
        
        Args:
            features (torch.Tensor): Input spectrogram tensor
            
        Returns:
            torch.Tensor: Augmented spectrogram tensor
        """
        # Randomly choose filter type
        if torch.rand(1).item() < self.prob_step:
            # Use step filter with tighter parameters
            augmenter = FilterAugment(
                db_range=self.db_range,
                n_band=[2, 5],
                min_bw=4,
                filter_type="step"
            )
        else:
            # Use linear filter with default parameters
            augmenter = FilterAugment(
                db_range=self.db_range,
                n_band=[3, 6],
                min_bw=6,
                filter_type="linear"
            )
        
        return augmenter(features)
