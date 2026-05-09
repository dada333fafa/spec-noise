"""
Visualization script for comparing different data augmentation methods.
Generates Figure 2 style comparison similar to the paper.

Usage:
    python visualize_augmentations.py --input <path_to_npy_file> --output <output_dir>
"""

import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import librosa
import librosa.display
import random
import time
from data_aug.specAugmentation import SpecAugmentation
from data_aug.randomReplacementDropStripes import RandomReplacementSpecAugmentation
from data_aug.filterAugment import FilterAugment
from pytorch_utils import specmix


class FilterAugmentVisualization:
    def __init__(self, db_range=[0.3, 3.0], n_band=[6, 10], min_bw=10, filter_type="linear"):
        self.db_range = db_range
        self.n_band = n_band
        self.min_bw = min_bw
        self.filter_type = filter_type
        self.training = True
    
    def __call__(self, features):
        if not self.training: return features
        if features.dim() == 3:
            features = features.unsqueeze(0)
            return_single = True
        elif features.dim() == 4:
            return_single = False
        else:
            raise ValueError(f"Expected 3D or 4D tensor, got {features.dim()}D")
        
        batch_size, n_channels, n_freq_bin, n_time = features.shape
        freq_filt = self._generate_filter(batch_size, n_freq_bin, features)
        augmented_features = features * freq_filt.unsqueeze(1)
        augmented_features = torch.clamp(augmented_features, 0, 1)
        
        if return_single: return augmented_features.squeeze(0)
        return augmented_features
    
    def train(self, mode=True):
        self.training = mode
        return self
    
    def eval(self):
        self.training = False
        return self
    
    def _generate_filter(self, batch_size, n_freq_bin, features):
        n_freq_band = torch.randint(low=self.n_band[0], high=self.n_band[1], size=(1,)).item()
        min_bw = self.min_bw
        if n_freq_band > 1:
            while n_freq_bin - n_freq_band * min_bw + 1 < 0:
                min_bw -= 1
            band_bndry_freqs = torch.sort(
                torch.randint(0, n_freq_bin - n_freq_band * min_bw + 1, (n_freq_band - 1,))
            )[0] + torch.arange(1, n_freq_band) * min_bw
            band_bndry_freqs = torch.cat((torch.tensor([0]), band_bndry_freqs, torch.tensor([n_freq_bin])))
            
            if self.filter_type == "step":
                band_factors = torch.rand((batch_size, n_freq_band)).to(features) * \
                               (self.db_range[1] - self.db_range[0]) + self.db_range[0]
                freq_filt = torch.ones((batch_size, n_freq_bin, 1)).to(features)
                for i in range(n_freq_band):
                    freq_filt[:, band_bndry_freqs[i]:band_bndry_freqs[i + 1], :] = band_factors[:, i].unsqueeze(-1).unsqueeze(-1)
            elif self.filter_type == "linear":
                band_factors = torch.rand((batch_size, n_freq_band + 1)).to(features) * \
                               (self.db_range[1] - self.db_range[0]) + self.db_range[0]
                freq_filt = torch.ones((batch_size, n_freq_bin, 1)).to(features)
                for i in range(n_freq_band):
                    for j in range(batch_size):
                        freq_filt[j, band_bndry_freqs[i]:band_bndry_freqs[i+1], :] = \
                            torch.linspace(band_factors[j, i], band_factors[j, i+1],
                                           band_bndry_freqs[i+1] - band_bndry_freqs[i]).unsqueeze(-1)
            return freq_filt
        else:
            return torch.ones((batch_size, n_freq_bin, 1)).to(features)


def load_spectrogram(npy_path):
    spec = np.load(npy_path)
    if spec.ndim == 1:
        sr = 22050 
        mel_spec = librosa.feature.melspectrogram(y=spec, sr=sr, n_mels=128, fmax=8000)
        spec = (mel_spec - mel_spec.min()) / (mel_spec.max() - mel_spec.min() + 1e-8)
    elif spec.max() > 1:
        spec = spec / 255.0
    
    spec_tensor = torch.from_numpy(spec).float()
    if spec_tensor.dim() == 2: spec_tensor = spec_tensor.unsqueeze(0)  
    if spec_tensor.shape[0] == 1: spec_tensor = spec_tensor.repeat(3, 1, 1)  
    return spec_tensor.unsqueeze(0)  


def apply_no_aug(spec):
    return spec.clone()


def apply_filteraugment(spec):
    random.seed(42)
    torch.manual_seed(42)
    augmenter = FilterAugmentVisualization(db_range=[0.2, 5.0], n_band=[4, 8], min_bw=15, filter_type="step")
    return augmenter(spec)


def apply_specaugment(spec, seed):
    random.seed(seed)
    torch.manual_seed(seed)
    spec_db = 10 * torch.log10(spec + 1e-8)
    
    class DropStripesVis:
        def __init__(self, dim, stripes_num):
            self.dim = dim
            self.stripes_num = stripes_num
            self.training = True
        
        def __call__(self, input_tensor):
            if not self.training: return input_tensor
            for n in range(input_tensor.shape[0]):
                self.transform_slice(input_tensor[n], input_tensor.shape[self.dim])
            return input_tensor
        
        def transform_slice(self, e, total_width):
            for _ in range(self.stripes_num):
                distance = max(1, total_width // 15) 
                bgn = random.randint(0, total_width - distance)
                
                if self.dim == 2:
                    e[:, bgn:bgn + distance, :] = -20.0  
                elif self.dim == 3:
                    e[:, :, bgn:bgn + distance] = -20.0  
    
    class SpecAugmentVis:
        def __init__(self, time_stripes_num, freq_stripes_num):
            self.time_dropper = DropStripesVis(dim=3, stripes_num=time_stripes_num)
            self.freq_dropper = DropStripesVis(dim=2, stripes_num=freq_stripes_num)
            self.training = True
        
        def __call__(self, input_tensor):
            x = self.time_dropper(input_tensor)
            x = self.freq_dropper(x)
            return x
    
    augmenter = SpecAugmentVis(time_stripes_num=1, freq_stripes_num=1)
    augmented_db = augmenter(spec_db)
    augmented = 10 ** (augmented_db / 10)
    return torch.clamp(augmented, 0, 1)


def apply_specmix(spec, spec2, seed):
    spec_db = 10 * torch.log10(spec + 1e-8)
    spec2_db = 10 * torch.log10(spec2 + 1e-8)
    
    augmented_db = spec_db.clone()
    
    F = augmented_db.shape[2]
    T = augmented_db.shape[3]
    freq_w = max(1, F // 15)
    time_w = max(1, T // 15)
    
    random.seed(seed)
    
    bgn_t = random.randint(0, T - time_w)
    bgn_f = random.randint(0, F - freq_w)
    
    augmented_db[:, :, :, bgn_t:bgn_t + time_w] = spec2_db[:, :, :, bgn_t:bgn_t + time_w]
    augmented_db[:, :, bgn_f:bgn_f + freq_w, :] = spec2_db[:, :, bgn_f:bgn_f + freq_w, :]
    
    augmented = 10 ** (augmented_db / 10)
    return torch.clamp(augmented, 0, 1)


def apply_specnoise(spec, seed):
    random.seed(seed)
    torch.manual_seed(seed)
    spec_db = 10 * torch.log10(spec + 1e-8)
    
    class RandomReplacementDropStripesVis:
        def __init__(self, dim, stripes_num, replacement='noise'):
            self.dim = dim
            self.stripes_num = stripes_num
            self.replacement = replacement
            self.training = True
        
        def __call__(self, input_tensor):
            if not self.training: return input_tensor
            for n in range(input_tensor.shape[0]):
                self.transform_slice(input_tensor[n], input_tensor.shape[self.dim])
            return input_tensor
        
        def transform_slice(self, e, total_width):
            for _ in range(self.stripes_num):
                distance = max(1, total_width // 15)
                bgn = random.randint(0, total_width - distance)
                
                if self.dim == 2:
                    if self.replacement == 'noise':
                        e[:, bgn:bgn + distance, :] = torch.clamp(
                            torch.randn_like(e[:, bgn:bgn + distance, :]) * 5 - 5, min=-80, max=0)
                elif self.dim == 3:
                    if self.replacement == 'noise':
                        e[:, :, bgn:bgn + distance] = torch.clamp(
                            torch.randn_like(e[:, :, bgn:bgn + distance]) * 5 - 5, min=-80, max=0)
    
    class SpecNoiseVis:
        def __init__(self, time_stripes_num, freq_stripes_num, replacement='noise'):
            self.time_dropper = RandomReplacementDropStripesVis(dim=3, stripes_num=time_stripes_num, replacement=replacement)
            self.freq_dropper = RandomReplacementDropStripesVis(dim=2, stripes_num=freq_stripes_num, replacement=replacement)
            self.training = True
        
        def __call__(self, input_tensor):
            x = self.time_dropper(input_tensor)
            x = self.freq_dropper(x)
            return x
    
    augmenter = SpecNoiseVis(time_stripes_num=1, freq_stripes_num=1, replacement='noise')
    augmented_db = augmenter(spec_db)
    augmented = 10 ** (augmented_db / 10)
    return torch.clamp(augmented, 0, 1)


# =====================================================================
# 【新增功能：专门为你生成论文 Figure 1 流程图的拆解素材！】
# =====================================================================
def export_specnoise_process_images(spec, seed, output_dir):
    """提取与主图绝对一致的单根竖线和单根横线"""
    print("[INFO] Generating separate SpecNoise stripes for Figure 1 process diagram...")
    
    spec_db = 10 * torch.log10(spec + 1e-8)
    F = spec_db.shape[2]
    T = spec_db.shape[3]
    freq_w = max(1, F // 15)
    time_w = max(1, T // 15)
    
    # 【核心】：重置为主种子，重走一遍获取随机坐标的流程，保证位置严丝合缝！
    random.seed(seed)
    torch.manual_seed(seed)
    
    # 模拟 SpecNoiseVis 里的执行顺序：先 Time，后 Freq
    bgn_t = random.randint(0, T - time_w)
    bgn_f = random.randint(0, F - freq_w)
    
    # 制作图1：只有竖线 (Vertical Only)
    torch.manual_seed(seed) # 保证每次画的高斯噪声雪花点也一样
    spec_vertical_only = spec_db.clone()
    spec_vertical_only[:, :, :, bgn_t:bgn_t + time_w] = torch.clamp(
        torch.randn_like(spec_vertical_only[:, :, :, bgn_t:bgn_t + time_w]) * 5 - 5, min=-80, max=0)
    linear_vertical = torch.clamp(10 ** (spec_vertical_only / 10), 0, 1)
    
    # 制作图2：只有横线 (Horizontal Only)
    torch.manual_seed(seed)
    spec_horizontal_only = spec_db.clone()
    spec_horizontal_only[:, :, bgn_f:bgn_f + freq_w, :] = torch.clamp(
        torch.randn_like(spec_horizontal_only[:, :, bgn_f:bgn_f + freq_w, :]) * 5 - 5, min=-80, max=0)
    linear_horizontal = torch.clamp(10 ** (spec_horizontal_only / 10), 0, 1)
    
    # 分别保存这两张图 (224x224 干净无边框)
    components = {
        'SpecNoise_Process_Vertical_Only': linear_vertical,
        'SpecNoise_Process_Horizontal_Only': linear_horizontal
    }
    
    for name, tensor in components.items():
        fig_ind = plt.figure(figsize=(0.7467, 0.7467), dpi=300)
        visualize_spectrogram(tensor, fig_ind.add_axes([0, 0, 1, 1]), '')
        ind_path = os.path.join(output_dir, f'{name}_clean.png')
        plt.savefig(ind_path, dpi=300, bbox_inches='tight', pad_inches=0, facecolor='none')
        plt.close(fig_ind)
    
    print("[INFO] Successfully saved individual stripes for process diagram!")
# =====================================================================


def visualize_spectrogram(spec, ax, title, sr=22050):
    if spec.dim() == 4: spec_2d = spec[0, 0, :, :].numpy()  
    elif spec.dim() == 3: spec_2d = spec[0, :, :].numpy()
    else: spec_2d = spec.numpy()
    
    spec_db = 20 * np.log10(spec_2d + 1e-8)
    spec_db = np.clip(spec_db, -80, 0)
    
    ax.imshow(spec_db, aspect='auto', origin='lower', cmap='viridis', vmin=-80, vmax=0)
    ax.axis('off')
    
    if title:
        ax.text(0.5, -0.15, title, ha='center', va='center', transform=ax.transAxes, fontsize=16, fontfamily='serif')
    
    return plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=-80, vmax=0))


def main():
    parser = argparse.ArgumentParser(description='Visualize data augmentation effects')
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, default='./augmentation_visualization')
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    spec_original = load_spectrogram(args.input)
    
    second_sample_path = r'D:\spec-noise\SpecNoise-main\images_dataset\Tachybaptus ruficollis\XC26929_3.npy'
    
    try:
        spec_second = load_spectrogram(second_sample_path)
        if spec_second.shape[-1] != spec_original.shape[-1]:
            target_len = spec_original.shape[-1]
            if spec_second.shape[-1] > target_len: spec_second = spec_second[:, :, :, :target_len]
            else:
                import torch.nn.functional as F
                spec_second = F.pad(spec_second, (0, target_len - spec_second.shape[-1]))
    except Exception as e:
        spec_second = torch.roll(spec_original, shifts=50, dims=-1)
    
    master_seed = int(time.time() * 1000) % 100000 
    print(f"[INFO] Generated master seed for this run: {master_seed}")

    aug_methods = {
        '(a) No Aug': apply_no_aug,
        '(b) FilterAugment': apply_filteraugment, 
        '(c) SpecAug': lambda x: apply_specaugment(x, seed=master_seed),
        '(d) SpecMix': lambda x: apply_specmix(x, spec_second, seed=master_seed),
        '(e) SpecNoise': lambda x: apply_specnoise(x, seed=master_seed),
    }
    
    augmented_specs = {}
    for name, aug_func in aug_methods.items():
        try:
            augmented_specs[name] = aug_func(spec_original)
        except Exception as e:
            print(f"  ✗ {name} failed: {e}")
            augmented_specs[name] = None
    
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 6, hspace=0.4, wspace=0.1, left=0.05, right=0.90, top=0.95, bottom=0.1)
    axes = [fig.add_subplot(gs[0, 0:2]), fig.add_subplot(gs[0, 2:4]), fig.add_subplot(gs[0, 4:6]),  
            fig.add_subplot(gs[1, 1:3]), fig.add_subplot(gs[1, 3:5])]
    
    cbar_sm = None
    for idx, (name, spec) in enumerate(augmented_specs.items()):
        if spec is not None:
            sm = visualize_spectrogram(spec, axes[idx], name)
            if cbar_sm is None: cbar_sm = sm
        else:
            axes[idx].axis('off')
    
    if cbar_sm is not None:
        cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
        fig.colorbar(cbar_sm, cax=cbar_ax, format='%+2.0f dB').ax.tick_params(labelsize=9)
    
    output_path = os.path.join(args.output, 'augmentation_comparison_paper_style.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    
    # 正常保存 5 张对比的单图
    for name, spec in augmented_specs.items():
        if spec is not None:
            ind_path = os.path.join(args.output, f'{name.replace(" ", "_").replace("(", "").replace(")", "")}_clean.png')
            fig_ind = plt.figure(figsize=(0.7467, 0.7467), dpi=300)
            visualize_spectrogram(spec, fig_ind.add_axes([0, 0, 1, 1]), '')
            plt.savefig(ind_path, dpi=300, bbox_inches='tight', pad_inches=0, facecolor='none')
            plt.close(fig_ind)
            
    # 【调用新功能】：导出用于流程图的拆解部件！
    export_specnoise_process_images(spec_original, master_seed, args.output)

if __name__ == '__main__':
    main()