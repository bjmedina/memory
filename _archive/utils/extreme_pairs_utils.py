import os
import torch
import soundfile as sf


def get_extreme_pairs(mse_matrix, segment_list):
    """
    Identifies the most similar and most dissimilar segment pairs
    based on a pairwise MSE matrix.

    Args:
        mse_matrix (torch.Tensor): Symmetric [N x N] MSE matrix.
        segment_list (list): Identifiers (e.g., filenames or timestamps).

    Returns:
        tuple: 
            - max_pair_indices: (i, j) of most dissimilar pair
            - max_pair_segments: segment identifiers for max MSE
            - min_pair_indices: (i, j) of most similar pair
            - min_pair_segments: segment identifiers for min MSE
    """
    # Only consider upper triangle (exclude diagonal and duplicates)
    mask = torch.triu(torch.ones_like(mse_matrix), diagonal=1)

    # Max pair
    masked_mse = mse_matrix.clone()
    masked_mse[mask == 0] = float('-inf')
    max_idx = torch.argmax(masked_mse)
    max_pair = torch.nonzero(masked_mse == masked_mse.view(-1)[max_idx])[0]

    # Min pair
    masked_mse[mask == 0] = float('inf')
    min_idx = torch.argmin(masked_mse)
    min_pair = torch.nonzero(masked_mse == masked_mse.view(-1)[min_idx])[0]

    max_i, max_j = max_pair[0].item(), max_pair[1].item()
    min_i, min_j = min_pair[0].item(), min_pair[1].item()

    max_pair_segments = (segment_list[max_i], segment_list[max_j])
    min_pair_segments = (segment_list[min_i], segment_list[min_j])

    return (max_i, max_j), max_pair_segments, (min_i, min_j), min_pair_segments


def save_extreme_audio_pairs(
    mse_matrix,
    segment_times,
    waveform,
    sample_rate,
    segment_duration,
    ID, 
    save_dir="extreme_pairs"
):
    """
    Saves audio clips for the most and least similar pairs of segments.

    Args:
        mse_matrix (torch.Tensor): Pairwise MSE matrix (NxN).
        segment_times (list): Onset times (in seconds) for each segment.
        waveform (np.ndarray): Full waveform array.
        sample_rate (int): Sampling rate.
        segment_duration (float): Duration of each segment in seconds.
        ID (str): Unique identifier for labeling saved files.
        save_dir (str): Directory to save the clips.

    Returns:
        tuple: max and min index pairs ((i, j), (i, j))
    """
    os.makedirs(save_dir, exist_ok=True)
    mask = torch.triu(torch.ones_like(mse_matrix), diagonal=1)

    # Max pair
    masked_mse = mse_matrix.clone()
    masked_mse[mask == 0] = float('-inf')
    max_idx = torch.argmax(masked_mse)
    max_pair = torch.nonzero(masked_mse == masked_mse.view(-1)[max_idx])[0]

    # Min pair
    masked_mse[mask == 0] = float('inf')
    min_idx = torch.argmin(masked_mse)
    min_pair = torch.nonzero(masked_mse == masked_mse.view(-1)[min_idx])[0]

    max_i, max_j = max_pair.tolist()
    min_i, min_j = min_pair.tolist()

    # Save clips for max and min pairs
    for label, (i, j) in zip(["max", "min"], [(max_i, max_j), (min_i, min_j)]):
        for idx, k in enumerate([i, j]):
            onset = segment_times[k]
            start_sample = int(onset * sample_rate)
            end_sample = int((onset + segment_duration) * sample_rate)
            clip = waveform[start_sample:end_sample]

            out_path = os.path.join(save_dir, f"{label}_mse_{ID}_{idx}.wav")
            sf.write(out_path, clip, sample_rate)

    return (max_i, max_j), (min_i, min_j)
