import numbers

import numpy as np


def _num_samples(x):
    """Return number of samples in array-like x.
    """

    message = "Expected sequence or array-like, got %s" % type(x)
    if hasattr(x, "fit") and callable(x.fit):
        # Don"t get num_samples from an ensembles length!
        raise TypeError(message)

    if not hasattr(x, "__len__") and not hasattr(x, "shape"):
        if hasattr(x, "__array__"):
            x = np.asarray(x)
        else:
            raise TypeError(message)

    if hasattr(x, "shape") and x.shape is not None:
        if len(x.shape) == 0:
            raise TypeError("Singleton array %r cannot be considered"
                            " a valid collection." % x)
        # Check that shape is returning an integer or default to len
        # Dask dataframes may not return numeric shape[0] value
        if isinstance(x.shape[0], numbers.Integral):
            return x.shape[0]

    try:
        return len(x)
    except TypeError:
        raise TypeError(message)


def check_consistent_length(*arrays):
    """Check that all arrays have consistent first dimensions.
    Checks whether all objects in arrays have the same shape or length.

    Args:
        *arrays : list or tuple of input objects.
            Objects that will be checked for consistent length.
    """

    lengths = [_num_samples(X) for X in arrays if X is not None]
    uniques = np.unique(lengths)
    if len(uniques) > 1:
        raise ValueError("Found input variables with inconsistent numbers of"
                         " samples: %r" % [int(l) for l in lengths])
