#
# Copyright (c) 2020 VCLA, UCLA [vcla.stat.ucla.edu]
# Written by Feng Shi <shi.feng@cs.ucla.edu>,
#

import torch

from .causal_prefix_cpu import causal_prefix_sum as causal_prefix_sum_cpu, \
    causal_prefix_backward as causal_prefix_backward_cpu

try:
    from .causal_prefix_cuda import \
        causal_prefix_sum as causal_prefix_sum_cuda, \
        causal_prefix_backward as causal_prefix_backward_cuda
except ImportError:
    causal_prefix_sum_cuda = causal_prefix_backward_cuda = None


class CausalPrefixSum(torch.autograd.Function):
    """Compute the weighted sum of values but attending only to previous
    values."""
    prefix = {
        "cpu": causal_prefix_sum_cpu,
        "cuda": causal_prefix_sum_cuda
    }
    prefix_backward = {
        "cpu": causal_prefix_backward_cpu,
        "cuda": causal_prefix_backward_cuda
    }

    @staticmethod
    def forward(ctx, Q, K, V):
        # Save the inputs for the gradient computation
        ctx.save_for_backward(Q, K, V)

        # Create the output tensor
        device = Q.device
        N, H, L, _ = Q.shape
        _, _, _, M = V.shape
        product = torch.zeros((N, H, L, M), device=device)

        # Actually perform the prefix sum
        CausalPrefixSum.prefix[device.type](
            Q.data,
            K.data,
            V.data,
            product
        )

        return product

    @staticmethod
    def backward(ctx, grad_out):
        # Extract the saved tensors
        Q, K, V = ctx.saved_tensors

        # Allocate memory for the gradients
        grad_Q = torch.zeros_like(Q)
        grad_K = torch.zeros_like(K)
        grad_V = torch.zeros_like(V)

        # Actually compute the gradients
        CausalPrefixSum.prefix_backward[Q.device.type](
            Q.data,
            K.data,
            V.data,
            grad_out,
            grad_Q,
            grad_K,
            grad_V
        )

        return grad_Q, grad_K, grad_V


# Alias the autograd functions to python style snake case naming
causal_prefix_sum = CausalPrefixSum.apply
