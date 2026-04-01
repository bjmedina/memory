import torch
import torch.nn.functional as F
import torchaudio.functional as F



# def downsample_per_channel_loop(cochleagram_tensor, orig_rate, target_rate):
#     """
#     Downsamples each cochleagram channel independently using linear resampling.

#     Args:
#         cochleagram_tensor (torch.Tensor): Tensor of shape (channels, time).
#         orig_rate (int): Original sampling rate.
#         target_rate (int): Target sampling rate.

#     Returns:
#         torch.Tensor: Downsampled cochleagram of shape (channels, new_time).
#     """
#     return torch.cat([
#         F.resample(cochleagram_tensor[c][None, None], orig_rate, target_rate).squeeze(0)
#         for c in range(cochleagram_tensor.shape[0])
#     ], dim=0)

def downsample_per_channel_loop(cochleagram_tensor, orig_rate, target_rate):
    """
    Iteratively downsamples each cochleagram channel.
    Args:
        cochleagram_tensor: (channels, time)
    Returns:
        (channels, new_time)
    """
    channels, time = cochleagram_tensor.shape
    resampled_channels = []
    
    for c in range(channels):
        channel_waveform = cochleagram_tensor[c].unsqueeze(0).unsqueeze(0)  # (1, 1, time)
        resampled = F.resample(channel_waveform, orig_freq=orig_rate, new_freq=target_rate)
        resampled_channels.append(resampled.squeeze(0))  # (1, new_time)

    return torch.cat(resampled_channels, dim=0)  # (channels, new_time)


def compute_low_freq_auc_ratio(cochleagram, num_low_freq_channels=5):
    """
    Computes low-frequency energy ratio using area under the curve (AUC).

    Args:
        cochleagram (torch.Tensor): Shape (freq_channels, time_frames).
        num_low_freq_channels (int): Number of low-frequency channels to consider.

    Returns:
        float: Ratio of low-frequency AUC to total AUC.
    """
    auc_per_channel = cochleagram.sum(dim=1)
    return (auc_per_channel[:num_low_freq_channels].sum() / (auc_per_channel.sum() + 1e-8)).item()
