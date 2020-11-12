#
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>,
# Apoorv Vyas <avyas@idiap.ch>
#

import torch

from .segmented_product_cpu import segmented_dot_product as segmented_dot_product_cpu, \
    segmented_dot_backward as segmented_dot_backward_cpu

try:
    from .segmented_product_cuda import \
        segmented_dot_product as segmented_dot_product_cuda, \
        segmented_dot_backward as segmented_dot_backward_cuda
except ImportError:
    segmented_dot_product_cuda = segmented_dot_backward_cuda = None


class SegmentedDotProduct(torch.autograd.Function):
    """Compute the weighted sum of values but attending only to previous
    values."""
    dot = {
        "cpu": segmented_dot_product_cpu,
        "cuda": segmented_dot_product_cuda
    }
    dot_backward = {
        "cpu": segmented_dot_backward_cpu,
        "cuda": segmented_dot_backward_cuda
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

        # Actually perform the dot product
        SegmentedDotProduct.dot[device.type](
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
        SegmentedDotProduct.dot_backward[Q.device.type](
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
segmented_dot_product = SegmentedDotProduct.apply
