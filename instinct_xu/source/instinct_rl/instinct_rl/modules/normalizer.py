# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

#  Copyright (c) 2020 Preferred Networks, Inc.

from __future__ import annotations

import numpy as np
import torch
import torch.distributed as dist
from torch import nn


class EmpiricalNormalization(nn.Module):
    """Normalize mean and variance of values based on empirical values."""

    def __init__(self, shape, eps=1e-2, until=None):
        """Initialize EmpiricalNormalization module.

        Args:
            shape (int or tuple of int): Shape of input values except batch axis.
            eps (float): Small value for stability.
            until (int or None): If this arg is specified, the link learns input values until the sum of batch sizes
            exceeds it.
        """
        super().__init__()
        self.eps = eps
        self.until = until
        self.register_buffer("_mean", torch.zeros(shape).unsqueeze(0))
        self.register_buffer("_var", torch.ones(shape).unsqueeze(0))
        self.register_buffer("_std", torch.ones(shape).unsqueeze(0))
        self.register_buffer("count", torch.tensor(0, dtype=torch.long))

    @property
    def mean(self):
        return self._mean.squeeze(0).clone()

    @property
    def std(self):
        return self._std.squeeze(0).clone()

    def forward(self, x):
        """Normalize mean and variance of values based on empirical values.

        Args:
            x (ndarray or Variable): Input values

        Returns:
            ndarray or Variable: Normalized output values
        """

        if self.training:
            with torch.no_grad():
                self.update(x)
        return (x - self._mean) / (self._std + self.eps)

    @torch.jit.unused
    def update(self, x):
        """Learn input values without computing the output values of them"""

        if self.until is not None and self.count >= self.until:
            return

        count_x = x.shape[0]
        self.count += count_x
        rate = count_x / self.count

        var_x = torch.var(x, dim=0, unbiased=False, keepdim=True)
        mean_x = torch.mean(x, dim=0, keepdim=True)
        delta_mean = mean_x - self._mean
        self._mean += rate * delta_mean
        self._var += rate * (var_x - self._var + delta_mean * (mean_x - self._mean))
        self._std = torch.sqrt(self._var)

    @torch.jit.unused
    def inverse(self, y):
        return y * (self._std + self.eps) + self._mean

    def init_broadcast(self):
        """Broadcast buffers from rank 0 so all processes start from the same statistics."""
        if not dist.is_initialized() or dist.get_world_size() == 1:
            return
        for buffer in self.buffers():
            dist.broadcast(buffer, src=0)

    def sync_across_processes(self):
        if not dist.is_initialized() or dist.get_world_size() == 1:
            return

        world_size = dist.get_world_size()
        device = self._mean.device

        local_mean = self._mean.squeeze(0)
        local_var = self._var.squeeze(0)
        local_count = self.count.float()

        all_means = [torch.zeros_like(local_mean) for _ in range(world_size)]
        dist.all_gather(all_means, local_mean)

        all_vars = [torch.zeros_like(local_var) for _ in range(world_size)]
        dist.all_gather(all_vars, local_var)

        all_counts = [torch.zeros(1, device=device) for _ in range(world_size)]
        dist.all_gather(all_counts, local_count.unsqueeze(0))
        all_counts = [c.squeeze() for c in all_counts]

        total_count = sum(all_counts)
        if total_count == 0:
            return

        global_mean = sum(m * c for m, c in zip(all_means, all_counts)) / total_count
        global_var = sum(
            c * (v + (m - global_mean) ** 2)
            for m, v, c in zip(all_means, all_vars, all_counts)
        ) / total_count

        self._mean = global_mean.unsqueeze(0)
        self._var = global_var.unsqueeze(0)
        self._std = torch.sqrt(self._var)
        self.count = total_count.long()

    def export(self, path):
        np.savez(
            path,
            mean=self._mean.cpu().numpy(),
            std=self._std.cpu().numpy(),
            eps=self.eps,
            until=self.until,
        )


class EmpiricalNormalizationWithPassthrough(nn.Module):
    """EmpiricalNormalization that leaves selected flat slices unchanged.

    Used for vision policies (e.g. PHP VisionMLP): proprio gets running mean/std
    normalization while depth pixels pass through after env-side fixed scaling.
    """

    def __init__(self, shape, passthrough_slices=None, eps=1e-2, until=None):
        super().__init__()
        if isinstance(shape, (tuple, list)):
            self.total_size = int(np.prod(shape))
        else:
            self.total_size = int(shape)

        mask = torch.ones(self.total_size, dtype=torch.bool)
        for start_end in passthrough_slices or []:
            start, end = int(start_end[0]), int(start_end[1])
            if start < 0 or end > self.total_size or start > end:
                raise ValueError(
                    f"Invalid passthrough slice [{start}, {end}) for total size {self.total_size}"
                )
            mask[start:end] = False
        self.register_buffer("_normalize_mask", mask)
        self.emp_norm = EmpiricalNormalization(int(mask.sum().item()), eps=eps, until=until)

    def forward(self, x):
        if self.training:
            with torch.no_grad():
                self.emp_norm.update(x[..., self._normalize_mask])
        out = x.clone()
        out[..., self._normalize_mask] = (x[..., self._normalize_mask] - self.emp_norm._mean) / (
            self.emp_norm._std + self.emp_norm.eps
        )
        return out

    def init_broadcast(self):
        self.emp_norm.init_broadcast()

    def sync_across_processes(self):
        self.emp_norm.sync_across_processes()

    def export(self, path):
        # Build on CPU: emp_norm buffers may live on CUDA after training/play.
        mean = torch.zeros(1, self.total_size)
        std = torch.ones(1, self.total_size)
        mask = self._normalize_mask.cpu()
        mean[:, mask] = self.emp_norm._mean.detach().cpu()
        std[:, mask] = self.emp_norm._std.detach().cpu()
        until = self.emp_norm.until
        np.savez(
            path,
            mean=mean.numpy(),
            std=std.numpy(),
            eps=np.asarray(self.emp_norm.eps, dtype=np.float64),
            # Avoid object arrays when until is None (breaks allow_pickle=False loads).
            until=np.asarray(-1 if until is None else until, dtype=np.int64),
            normalize_mask=mask.numpy(),
        )


class EmpiricalDiscountedVariationNormalization(nn.Module):
    """Reward normalization from Pathak's large scale study on PPO.

    Reward normalization. Since the reward function is non-stationary, it is useful to normalize
    the scale of the rewards so that the value function can learn quickly. We did this by dividing
    the rewards by a running estimate of the standard deviation of the sum of discounted rewards.
    """

    def __init__(self, shape, eps=1e-2, gamma=0.99, until=None):
        super().__init__()

        self.emp_norm = EmpiricalNormalization(shape, eps, until)
        self.disc_avg = DiscountedAverage(gamma)

    def forward(self, rew):
        if self.training:
            # update discounected rewards
            avg = self.disc_avg.update(rew)

            # update moments from discounted rewards
            self.emp_norm.update(avg)

        if self.emp_norm._std > 0:
            return rew / self.emp_norm._std
        else:
            return rew

    def init_broadcast(self):
        self.emp_norm.init_broadcast()

    def sync_across_processes(self):
        self.emp_norm.sync_across_processes()


class DiscountedAverage:
    r"""Discounted average of rewards.

    The discounted average is defined as:

    .. math::

        \bar{R}_t = \gamma \bar{R}_{t-1} + r_t

    Args:
        gamma (float): Discount factor.
    """

    def __init__(self, gamma):
        self.avg = None
        self.gamma = gamma

    def update(self, rew: torch.Tensor) -> torch.Tensor:
        if self.avg is None:
            self.avg = rew
        else:
            self.avg = self.avg * self.gamma + rew
        return self.avg
