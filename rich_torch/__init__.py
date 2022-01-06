import rich
import torch

def rich_repr(t):
        yield "shape", list(t.shape)
        yield "dtype", str(t.dtype).replace("torch.", "")
        yield "device", str(t.device)
        yield "requires_grad", t.requires_grad

def print(data):
    # TODO context manager?
    torch.Tensor.__rich_repr__ = rich_repr
    torch.Tensor.__rich_repr__.angular = True
    rich.print(data)