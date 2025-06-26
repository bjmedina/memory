import torch


def pairwise_mse_db(time_avg_coch, eps=1e-10):
    """
    Computes pairwise MSE in dB between time-averaged cochleagram tensors.

    Args:
        time_avg_coch (torch.Tensor): Tensor of shape [N, D].
        eps (float): Epsilon for numerical stability.

    Returns:
        torch.Tensor: Pairwise MSE in decibels, shape [N, N].
    """
    power = torch.mean(time_avg_coch ** 2, dim=1)
    mse = torch.mean((time_avg_coch[:, None] - time_avg_coch[None, :]) ** 2, dim=2)
    avg_power = (power[:, None] + power[None, :]) / 2
    return 10 * torch.log10(mse / (avg_power + eps) + eps)


def pairwise_mse(x):
    """
    Calculates the pairwise MSE between vectors in a tensor.

    Args:
        x (torch.Tensor): Tensor of shape (N, D), where N is the number of vectors
                          and D is the dimension of each vector.

    Returns:
        torch.Tensor: Pairwise MSE matrix of shape (N, N).
    """
    x_squared = torch.sum(x ** 2, dim=1, keepdim=True)
    
    # Calculate squared Euclidean distances
    distances = x_squared - 2 * torch.matmul(x, x.T) + x_squared.T
    
    # Ensure distances are non-negative and return
    return distances / x.shape[1]