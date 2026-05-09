"""
Quick script to convert an audio file to spectrogram for visualization.

Usage:
    python prepare_sample_spectrogram.py --audio <path_to_mp3> --output <output_npy> --feature_type mel
"""

import argparse
import numpy as np
import librosa
import cv2


def extract_features(y, sr, feature_type='mel', n_mels=224):
    """Extract spectrogram features from audio"""
    
    # Calculate hop length
    time_split = 1.0
    hop_length = time_split * 22050 / n_mels
    hop_length = int(np.ceil(hop_length))
    
    if feature_type == 'stft':
        stft_spec = librosa.stft(y=y, n_fft=512, hop_length=hop_length)
        stft_spec = librosa.amplitude_to_db(np.abs(stft_spec), ref=np.max)
        stft_spec = cv2.resize(stft_spec, (224, 224))
        return stft_spec
    
    elif feature_type == 'mel':
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length)
        mel_spec = librosa.amplitude_to_db(mel_spec, ref=np.max)
        mel_spec = cv2.resize(mel_spec, (224, 224))
        return mel_spec
    
    elif feature_type == 'mfcc':
        mfcc_spec = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, hop_length=hop_length)
        mfcc_spec = librosa.util.normalize(mfcc_spec)
        mfcc_spec = cv2.resize(mfcc_spec, (224, 224))
        return mfcc_spec
    
    else:
        raise ValueError(f"Unknown feature type: {feature_type}")


def main():
    parser = argparse.ArgumentParser(description='Convert audio to spectrogram')
    parser.add_argument('--audio', type=str, required=True, help='Path to audio file (.mp3, .wav)')
    parser.add_argument('--output', type=str, required=True, help='Output .npy file path')
    parser.add_argument('--feature_type', type=str, default='mel', 
                       choices=['stft', 'mel', 'mfcc'],
                       help='Type of spectrogram feature')
    args = parser.parse_args()
    
    print(f"[INFO] Loading audio: {args.audio}")
    y, sr = librosa.load(args.audio, sr=16000)
    
    print(f"[INFO] Extracting {args.feature_type} features...")
    spec = extract_features(y, sr, feature_type=args.feature_type)
    
    print(f"[INFO] Spectrogram shape: {spec.shape}")
    print(f"[INFO] Value range: [{spec.min():.2f}, {spec.max():.2f}]")
    
    # Save to .npy
    np.save(args.output, spec)
    print(f"[INFO] Saved to: {args.output}")


if __name__ == '__main__':
    main()
