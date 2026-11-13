# Verbatim author source at generic_deltaModel commit
# 4b85efa7549a9d98b1b048331ad4347a913f290c.
import torch
import torch.nn


def createMask(x, dr):
    """Create a dropout mask."""
    mask = x.new().resize_as_(x).bernoulli_(1 - dr).div_(1 - dr).detach_()
    return mask


class DropMask(torch.autograd.function.InplaceFunction):
    """Dropout mask for CudnnLstm weights."""

    @classmethod
    def forward(cls, ctx, input, mask, train=False, inplace=False):
        """Forward method."""
        ctx.train = train
        ctx.inplace = inplace
        ctx.mask = mask

        if not ctx.train:
            return input
        else:
            if ctx.inplace:
                ctx.mark_dirty(input)
                output = input
            else:
                output = input.clone()

            output = output * ctx.mask  # Avoid using inplace mul_

        return output

    @staticmethod
    def backward(ctx, grad_output):
        """Backward method."""
        if ctx.train:
            return grad_output * ctx.mask, None, None, None
        else:
            return grad_output, None, None, None
