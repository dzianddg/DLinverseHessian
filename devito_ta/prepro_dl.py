# Adapted from :
# https://github.com/DeepWave-KAUST/DeepFWIHessian/blob/main/deepinvhessian/train.py
# https://github.com/DeepWave-KAUST/DeepFWIHessian/blob/main/deepinvhessian_old/prepare_data.py

from typing import Callable, List, Dict
import numpy as np
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import TensorDataset, DataLoader
from tqdm.notebook import tqdm


def train(network, dataloader, optimizer, loss_fn, epochs, device):
    """Train a PyTorch network on a dataset for a fixed number of epochs.

    This function performs a standard supervised training loop:
    forward pass -> loss computation -> backward pass -> optimizer step.

    Notes:
        - The model is set to training mode via `network.train()`.
        - Targets are reshaped with `unsqueeze(1)` to add a channel dimension.
        - Loss values are averaged per epoch and returned as a Python list.

    Args:
        network (torch.nn.Module): Model to train. Must accept input tensors
            shaped like the batches produced by `dataloader` and return
            predictions compatible with `loss_fn`.
        dataloader (torch.utils.data.DataLoader): Iterable that yields batches.
            Each batch must be indexable such that `sample[0]` is the input
            tensor and `sample[1]` is the target tensor.
        optimizer (torch.optim.Optimizer): Optimizer used to update model weights.
        loss_fn (Callable): Loss function with signature
            `loss_fn(pred, target) -> torch.Tensor`.
        epochs (int): Number of training epochs.
        device (torch.device or str): Device on which to run training
            (e.g., "cuda", "cpu", torch.device("cuda")).

    Returns:
        list[float]: List of average epoch losses of length `epochs`.
    """
    loss = []
    network.train()
    for ep in tqdm(range(epochs)):
        running_loss = 0
        for sample in dataloader:
            optimizer.zero_grad()
            x1, y1 = sample[0].to(device), sample[1].unsqueeze(1).to(device)
            dm_pred = network(x1)
            loss_ = loss_fn(dm_pred, y1)
            running_loss += loss_.item()
            loss_.backward()
            optimizer.step()

        loss.append(running_loss / len(dataloader))

        print(f"Training Epoch {ep+1}, Loss = {loss[-1]}")

        # optimizer_unet.param_groups[-1]['lr'] = lr_init
    return loss


'''  prepare data for pytorch '''
def prepare_data(x, d, y, patch_size, slide, batch_size):
    """Prepare a patch-based PyTorch DataLoader for supervised learning.

    This function constructs training samples by extracting overlapping 2D patches
    from input tensors `x`, `d`, and target tensor `y`. The input to the model is a
    2-channel tensor created by concatenating patches from `x` and `d` along the
    channel dimension, while the label is the corresponding patch from `y`.

    The function applies replication padding on the left side (1 pixel) before
    patch extraction.

    Important:
        This function moves tensors to GPU using `.cuda()` unconditionally.
        If you want CPU support, you will need to modify that behavior.

    Args:
        x (torch.Tensor): Input tensor (2D) with shape (H, W).
        d (torch.Tensor): Additional input tensor (2D) with shape (H, W),
            typically representing a second feature map to concatenate with `x`.
        y (torch.Tensor): Target tensor (2D) with shape (H, W).
        patch_size (int): Patch size `k` (patches are k x k).
        slide (int): Sliding factor `kk` controlling patch stride via `k // kk`.
            Larger `slide` yields smaller stride (more overlap).
        batch_size (int): Batch size for the returned DataLoader.

    Returns:
        torch.utils.data.DataLoader: A DataLoader yielding `(X, Y)` batches where:
            - `X` has shape (B, 2, k, k) (two channels: x patch and d patch)
            - `Y` has shape (B, k, k) (target patch)
    """
    pd = (1, 0, 0, 0)
    pad_replicate = nn.ReplicationPad2d(pd)
    assert x.shape == y.shape, "shape should be equal, "

    x = pad_replicate(x.unsqueeze(0)).squeeze(0).float().cuda()
    d = pad_replicate(d.unsqueeze(0)).squeeze(0).float().cuda()
    y = pad_replicate(y.unsqueeze(0)).squeeze(0).float().cuda()

    k = patch_size
    kk = slide

    X, D, L, Y = [], [], [], []
    for xi in range(int(x.shape[0] // (k / kk))):
        for yi in range(int(x.shape[1] // (k / kk))):
            patch1 = x[xi * (k // kk): xi * (k // kk) + k, yi * (k // kk): yi * (k // kk) + k]
            patchd = d[xi * (k // kk): xi * (k // kk) + k, yi * (k // kk): yi * (k // kk) + k]
            patch2 = y[xi * (k // kk): xi * (k // kk) + k, yi * (k // kk): yi * (k // kk) + k]

            if patch1.shape == (k, k):
                X.append(patch1)
                D.append(patchd)
                Y.append(patch2)

    X = torch.stack(X)
    D = torch.stack(D)
    Y = torch.stack(Y)
    X = torch.cat([X.unsqueeze(1), D.unsqueeze(1)], dim=1)

    dm_dataset = TensorDataset(X, Y)
    train_dataloader = DataLoader(dm_dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    return train_dataloader
