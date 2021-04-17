import numpy as np


class Tensor(np.ndarray):
    """
    """
    def __init__(self, *args, **kwargs):
        self.grad = None


def from_array(arr):
    """Convert the input array-like to a tensor."""
    t = arr.view(Tensor)
    t.grad = None
    return t


def zeros(shape):
    """Return a new tensor of given shape, filled with zeros."""
    t = Tensor(shape)
    t.fill(0)
    return t


def ones(shape):
    """Return a new tensor of given shape, filled with ones."""
    t = Tensor(shape)
    t.fill(1)
    return t


def random(shape, loc=0.0, scale=0.01):
    """Return a new tensor of given shape, from normal distribution."""
    return from_array(np.random.normal(loc=loc, scale=scale, size=shape))


if __name__ == '__main__':
    import pdb; pdb.set_trace()
