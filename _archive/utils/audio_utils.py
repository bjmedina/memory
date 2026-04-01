import numpy as np
import librosa
import torch
import torch.nn.functional as F
import torchaudio

from scipy.signal import hilbert

import soundfile as sf
from scipy.io import wavfile

def rms_normalize(signal, target_rms=0.1):
    """Normalize signal to a target RMS value."""
    rms = np.sqrt(np.mean(signal**2))
    return signal * (target_rms / rms) if rms > 0 else signal

def compute_rms(signal):
    """Compute the Root Mean Square (RMS) of a signal."""
    signal = np.array(signal, dtype=np.float32)
    return np.sqrt(np.mean(signal ** 2))

    
def compute_silence_duration(sound, sr, threshold=0.01):
    """
    Computes the duration (in seconds) of silence in a single audio clip.

    Args:
        sound (np.ndarray): 1D audio signal.
        sr (int): Sampling rate.
        threshold (float): Envelope threshold below which a region is considered silent.

    Returns:
        float: Duration of silence in seconds.
    """
    # Normalize audio to [-1, 1]
    sound = sound / np.max(np.abs(sound))

    # Compute amplitude envelope using Hilbert transform
    envelope = np.abs(hilbert(sound))

    # Detect silent regions
    silent = envelope < threshold

    # Convert silent samples to duration
    silence_duration = np.sum(silent) / sr

    return silence_duration

def should_filter_out_pyin(signal, sr, 
                           min_freq=75, max_freq=450, 
                           voicing_thresh=0.8,
                           periodicity_thresh=0.8,
                           stability_thresh=10.0):
    """
    Detect whether a signal is pure-tone-like using librosa.pyin.
    """
    f0, voiced_flag, voiced_prob = librosa.pyin(signal,
                                                sr=sr,
                                                fmin=min_freq,
                                                fmax=max_freq)

    if f0 is None:
        return False, 0.0, float('inf')  # pyin failed completely

    # Optional: apply soft voicing threshold to filter low-confidence frames
    voiced_mask = voiced_prob > voicing_thresh
    voiced_f0 = f0[voiced_mask]

    periodicity_score = np.sum(voiced_f0) / len(f0)
    f0_std = np.std(voiced_f0) if len(voiced_f0) > 1 else np.inf

    #is_pure = (periodicity_score > periodicity_thresh) and (f0_std < stability_thresh)

    return periodicity_score, f0_std

def apply_linear_ramp(signal, sample_rate, ramp_duration_ms=5):
    """Applies a linear fade-in and fade-out to reduce clicks."""
    ramp_samples = int((ramp_duration_ms / 1000) * sample_rate)
    if ramp_samples * 2 >= len(signal):
        raise ValueError("Ramp duration too long for signal!")
    fade = np.linspace(0, 1, ramp_samples)
    signal[:ramp_samples] *= fade
    signal[-ramp_samples:] *= fade[::-1]
    return signal


def compute_autocorrelation_torch(signal):
    """Compute normalized autocorrelation using PyTorch."""
    signal = signal - signal.mean()
    corr = F.conv1d(signal[None, None], signal.flip(0)[None, None], padding=signal.shape[0]-1)[0, 0]
    return corr[corr.numel() // 2:] / corr.abs().max()

def should_filter_out_yin(data_np, sr, 
                           min_freq=100, max_freq=3000, 
                           threshold=0.5, std_thresh=0.1, 
                           frame_length=2048, hop_length=512, 
                           device="cuda"):
    """
    Detect whether a signal is pure-tone-like based on YIN periodicity estimation.
    """
    # Convert to torch tensor and move to device
    data = torch.tensor(data_np, dtype=torch.float32, device=device)

    # YIN pitch detection (from torchaudio)
    f0 = torchaudio.functional.detect_pitch_frequency(
        waveform=data.unsqueeze(0),  # (batch, time)
        sample_rate=sr,
        frame_time=frame_length / sr,  # frame size in seconds
        win_length=frame_length,
        freq_low=min_freq,
        freq_high=max_freq
    )[0]  # shape: (frames,)
    
    # Filter out invalid f0 values (0 means unvoiced/undetected)
    valid_f0 = f0[f0 > 0]

    # If too few frames have detected f₀, consider it not pure
    periodicity_score = valid_f0.numel() / f0.numel()

    # If detected, measure how stable (low std)
    if valid_f0.numel() > 1:
        stability_score = 1 / torch.std(valid_f0)
    else:
        stability_score = 0

    # Decision rule
    is_pure = (periodicity_score > threshold) and (torch.std(valid_f0) < std_thresh)

    return is_pure.item(), periodicity_score.item(), (torch.std(valid_f0)).item()

def should_filter_out(data_np, sr, min_freq=100, max_freq=3000, threshold=0.5, std_thresh=0.1, device="cuda"):
    """Detect whether a signal is likely a pure tone based on autocorrelation peaks."""
    data = torch.tensor(data_np, dtype=torch.float32, device=device)
    corr = compute_autocorrelation_torch(data)
    min_lag, max_lag = int(sr / max_freq), int(sr / min_freq)
    if max_lag >= len(corr): return False
    segment = corr[min_lag:max_lag]
    peak_vals, _ = torch.topk(segment, k=min(5, segment.numel()))
    return (peak_vals[0] > threshold and peak_vals.std() < std_thresh), peak_vals[0].item(), peak_vals.std().item()



def should_filter_out_yin_cpu(data_np, sr, 
                               min_freq=100, max_freq=3000, 
                               threshold=0.5, std_thresh=0.1, 
                               frame_length=2048, hop_length=512):
    """
    Detect whether a signal is pure-tone-like based on YIN (librosa) periodicity estimation.
    CPU version using librosa.yin.
    """
    # librosa expects 1D numpy arrays
    data_np = np.asarray(data_np, dtype=np.float32)

    # Use librosa's YIN function
    f0 = librosa.yin(
        y=data_np,
        fmin=min_freq,
        fmax=max_freq,
        sr=sr,
        frame_length=frame_length,
        hop_length=hop_length
    )

    # librosa returns np.nan when no f0 is found
    valid_f0 = f0[~np.isnan(f0)]

    # How many frames had a valid f0
    periodicity_score = valid_f0.size / f0.size

    # How stable are the detected f0s
    if valid_f0.size > 1:
        stability_std = np.std(valid_f0)
    else:
        stability_std = np.inf  # unstable if only one detection

    # Decision rule
    is_pure = (periodicity_score > threshold) and (stability_std < std_thresh)

    return is_pure, periodicity_score, stability_std

def process_clip(file_path, output_path, start=0.0, end=2.0, factor=0.05):
    """
    Extract a normalized segment from a WAV file and save it.

    Parameters:
        file_path (str): Path to the original audio file.
        output_path (str): Where to save the processed clip.
        start (float): Start time in seconds.
        end (float): End time in seconds.
        factor (float): Target RMS level.
    """
    # Read the full WAV file
    sample_rate, data = wavfile.read(file_path)
    
    # Ensure it's float32
    data = np.array(data, dtype=np.float32)

    # Check if mono or stereo; convert stereo to mono if needed
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)

    # Convert times to sample indices
    start_sample = int(start * sample_rate)
    end_sample = int(end * sample_rate)

    if end_sample > len(data):
        raise ValueError(f"Clip range [{start}, {end}] exceeds audio length.")

    # Trim
    clip = data[start_sample:end_sample]

    # Normalize
    rms = compute_rms(clip)
    if rms > 0:
        clip = clip * (factor / rms)

    # Save to output path
    sf.write(output_path, clip, sample_rate)

