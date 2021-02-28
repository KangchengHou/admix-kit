import numpy as np
from scipy.special import logsumexp

from collections import deque
import sys
import time


class ConvergenceMonitor:
    """Monitors and reports convergence to :data:`sys.stderr`.
    Parameters
    ----------
    tol : double
        Convergence threshold. EM has converged either if the maximum
        number of iterations is reached or the log probability
        improvement between the two consecutive iterations is less
        than threshold.
    n_iter : int
        Maximum number of iterations to perform.
    verbose : bool
        If ``True`` then per-iteration convergence reports are printed,
        otherwise the monitor is mute.
    Attributes
    ----------
    history : deque
        The log probability of the data for the last two training
        iterations. If the values are not strictly increasing, the
        model did not converge.
    iter : int
        Number of iterations performed while training the model.
    Examples
    --------
    Use custom convergence criteria by subclassing ``ConvergenceMonitor``
    and redefining the ``converged`` method. The resulting subclass can
    be used by creating an instance and pointing a model's ``monitor_``
    attribute to it prior to fitting.
    >>> from hmmlearn.base import ConvergenceMonitor
    >>> from hmmlearn import hmm
    >>>
    >>> class ThresholdMonitor(ConvergenceMonitor):
    ...     @property
    ...     def converged(self):
    ...         return (self.iter == self.n_iter or
    ...                 self.history[-1] >= self.tol)
    >>>
    >>> model = hmm.GaussianHMM(n_components=2, tol=5, verbose=True)
    >>> model.monitor_ = ThresholdMonitor(model.monitor_.tol,
    ...                                   model.monitor_.n_iter,
    ...                                   model.monitor_.verbose)
    """

    _template = "{iter:>10d} {logprob:>16.4f} {delta:>+16.4f} {iter_speed:>12.3f}s"

    def __init__(self, tol, n_iter, verbose):
        self.tol = tol
        self.n_iter = n_iter
        self.verbose = verbose
        self.history = deque(maxlen=2)
        self.iter = 0
        self.time_start = time.time()

    def __repr__(self):
        class_name = self.__class__.__name__
        params = sorted(dict(vars(self), history=list(self.history)).items())
        return (
            "{}(\n".format(class_name)
            + "".join(map("    {}={},\n".format, *zip(*params)))
            + ")"
        )

    def _reset(self):
        """Reset the monitor's state."""
        self.iter = 0
        self.history.clear()
        self.time_start = time.time()

    def report(self, logprob):
        """Reports convergence to :data:`sys.stderr`.
        The output consists of three columns: iteration number, log
        probability of the data at the current iteration and convergence
        rate.  At the first iteration convergence rate is unknown and
        is thus denoted by NaN.
        Parameters
        ----------
        logprob : float
            The log probability of the data as computed by EM algorithm
            in the current iteration.
        """
        if self.verbose:
            delta = logprob - self.history[-1] if self.history else np.nan
            message = self._template.format(
                iter=self.iter + 1,
                logprob=logprob,
                delta=delta,
                iter_speed=(time.time() - self.time_start) / (self.iter + 1),
            )
            print(message, file=sys.stderr)

        self.history.append(logprob)
        self.iter += 1

    @property
    def converged(self):
        """``True`` if the EM algorithm converged and ``False`` otherwise."""
        # XXX we might want to check that ``logprob`` is non-decreasing.
        return self.iter == self.n_iter or (
            len(self.history) == 2 and self.history[1] - self.history[0] < self.tol
        )


def log_normalize(a, axis=None):
    """
    Normalizes the input array so that ``sum(exp(a)) == 1``.
    Parameters
    ----------
    a : array
        Non-normalized input data.
    axis : int
        Dimension along which normalization is performed.
    Notes
    -----
    Modifies the input **inplace**.
    """
    with np.errstate(under="ignore"):
        a_lse = logsumexp(a, axis, keepdims=True)
    return a - a_lse


def iter_from_X_lengths(X, lengths):
    if lengths is None:
        yield 0, len(X)
    else:
        n_samples = X.shape[0]
        end = np.cumsum(lengths).astype(np.int32)
        start = end - lengths
        if end[-1] > n_samples:
            raise ValueError(
                "more than {:d} samples in lengths array {!s}".format(
                    n_samples, lengths
                )
            )

        for i in range(len(lengths)):
            yield start[i], end[i]


def log_mask_zero(a):
    """Computes the log of input probabilities masking divide by zero in log.
    Notes
    -----
    During the M-step of EM-algorithm, very small intermediate start
    or transition probabilities could be normalized to zero, causing a
    *RuntimeWarning: divide by zero encountered in log*.
    This function masks this unharmful warning.
    """
    a = np.asarray(a)
    with np.errstate(divide="ignore"):
        return np.log(a)


def normalize(a, axis=None):
    """
    Normalizes the input array so that it sums to 1.
    Parameters
    ----------
    a : array
        Non-normalized input data.
    axis : int
        Dimension along which normalization is performed.
    """
    a_sum = a.sum(axis)
    if axis and a.ndim > 1:
        # Make sure we don't divide by zero.
        a_sum[a_sum == 0] = 1
        shape = list(a.shape)
        shape[axis] = 1
        a_sum.shape = shape

    return a / a_sum
