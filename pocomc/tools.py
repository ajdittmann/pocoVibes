import numpy as np
import math
import torch
from tqdm import tqdm
import warnings

SQRTEPS = math.sqrt(float(np.finfo(np.float64).eps))


def trim_weights(samples, weights, ess=0.99, bins=1000):
    """
        Trim samples and weights to a given effective sample size.

    Parameters
    ----------
    samples : ``np.ndarray``
        Samples.
    weights : ``np.ndarray``
        Weights.
    ess : ``float``
        Effective sample size threshold.
    bins : ``int``
        Number of bins to use for trimming.

    Returns
    -------
    samples_trimmed : ``np.ndarray``
        Trimmed samples.
    weights_trimmed : ``np.ndarray``
        Trimmed weights.
    """

    # normalize weights (without modifying the input array)
    weights = weights / np.sum(weights)
    # compute untrimmed ess
    ess_total = 1.0 / np.sum(weights**2.0)
    # define percentile grid
    percentiles = np.linspace(0, 99, bins)

    # All weight thresholds at once (a single sort instead of one per percentile)
    thresholds = np.percentile(weights, percentiles)

    # For every threshold, the retained set {w >= threshold} is a suffix of the
    # sorted weights, so its trimmed ESS follows from suffix sums of w and w^2.
    w_sorted = np.sort(weights)
    s1 = np.cumsum(w_sorted[::-1])[::-1]
    s2 = np.cumsum((w_sorted**2.0)[::-1])[::-1]
    start = np.searchsorted(w_sorted, thresholds, side='left')
    start = np.minimum(start, len(w_sorted) - 1)
    ess_ratio = (s1[start]**2.0 / s2[start]) / ess_total

    # Scan from the highest percentile downwards (as in the sequential search),
    # confirming the candidate with the exact original computation.
    passing = np.nonzero(ess_ratio >= ess)[0]
    i = passing[-1] if len(passing) else bins - 1
    while True:
        # compute weight threshold
        threshold = thresholds[i]
        mask = weights >= threshold
        weights_trimmed = weights[mask]
        weights_trimmed /= np.sum(weights_trimmed)
        ess_trimmed = 1.0 / np.sum(weights_trimmed**2.0)
        if ess_trimmed / ess_total >= ess:
            break 
        i -= 1
    
    return samples[mask], weights_trimmed


def logsumexp(a, axis=None):
    """
        Numerically stable ``log(sum(exp(a)))``.

    Parameters
    ----------
    a : ``np.ndarray``
        Input array.
    axis : ``int`` or ``None``
        Axis over which the sum is taken (default is ``None``, i.e. all elements).

    Returns
    -------
    out : ``float`` or ``np.ndarray``
        Log of the sum of exponentials.
    """
    a = np.asarray(a)
    a_max = np.max(a, axis=axis, keepdims=True)
    a_max = np.where(np.isfinite(a_max), a_max, 0.0)
    with np.errstate(divide='ignore'):
        out = np.log(np.sum(np.exp(a - a_max), axis=axis, keepdims=True)) + a_max
    if axis is None:
        return out.reshape(())[()]
    return np.squeeze(out, axis=axis)


def effective_sample_size(weights):
    """
        Compute effective sample size (ESS).

    Parameters
    ----------
    weights : ``np.ndarray``
        Weights.

    Returns
    -------
    ess : ``float``
        Effective sample size.
    """
    weights = weights / np.sum(weights)
    return 1.0 / np.sum(weights**2.0)


def unique_sample_size(weights, k=None):
    """
        Compute unique sample size (ESS).

    Parameters
    ----------
    weights : ``np.ndarray``
        Weights.
    k : ``int``
        Number of resampled samples.

    Returns
    -------
    uss : ``float``
        Unique sample size.
    """
    if k is None:
        k = len(weights)
    weights = weights / np.sum(weights)
    return np.sum(1.0 - (1.0 - weights)**k)


def compute_ess(logw: np.ndarray):
    r"""
        Compute effective sample size (per centage).

    Parameters
    ----------
    logw : ``np.ndarray``
        Log-weights.
    Returns
    -------
    ess : float
        Effective sample size divided by actual number
        of particles (between 0 and 1)
    """
    logw_max = np.max(logw)
    logw_normed = logw - logw_max

    weights = np.exp(logw_normed) / np.sum(np.exp(logw_normed))
    return 1.0 / np.sum(weights * weights) / len(weights)


def increment_logz(logw: np.ndarray):
    r"""
        Compute log evidence increment factor.

    Parameters
    ----------
    logw : ``np.ndarray``
        Log-weights.
    Returns
    -------
    ess : float
        logZ increment.
    """
    logw_max = np.max(logw)
    logw_normed = logw - logw_max

    return logw_max + np.logaddexp.reduce(logw_normed)


def systematic_resample(size: np.ndarray, 
                        weights: np.ndarray, 
                        random_state: int = None):
    """
        Resample a new set of points from the weighted set of inputs
        such that they all have equal weight.

    Parameters
    ----------
    size : `int`
        Number of samples to draw.
    weights : `~numpy.ndarray` with shape (nsamples,)
        Corresponding weight of each sample.
    random_state : `int`, optional
        Random seed.    

    Returns
    -------
    indeces : `~numpy.ndarray` with shape (nsamples,)
        Indices of the resampled array.
    
    Examples
    --------
    >>> x = np.array([[1., 1.], [2., 2.], [3., 3.], [4., 4.]])
    >>> w = np.array([0.6, 0.2, 0.15, 0.05])
    >>> systematic_resample(4, w)
    array([0, 0, 0, 2])

    Notes
    -----
    Implements the systematic resampling method.
    """
    
    if random_state is not None:
        np.random.seed(random_state)

    if abs(np.sum(weights) - 1.) > SQRTEPS:
        weights = np.array(weights) / np.sum(weights)

    positions = (np.random.random() + np.arange(size)) / size
    cumulative_sum = np.cumsum(weights)
    # All positions are < 1; ending the cumulative sum at exactly 1 keeps every index in
    # range even when the weights sum to slightly less than 1 (rounding or tolerance above)
    cumulative_sum[-1] = 1.0
    indeces = np.searchsorted(cumulative_sum, positions)
    
    return indeces


class ProgressBar:
    """
        Progress bar class.

    Parameters
    ----------
    show : `bool`
        Whether or not to show a progress bar. Default is ``True``.
    """
    def __init__(self, show: bool = True, initial=0):
        self.progress_bar = tqdm(desc='Iter', disable=not show, initial=initial)
        self.info = dict()

    def update_stats(self, info):
        """
            Update shown stats.

        Parameters
        ----------
        info : dict
            Dictionary with stats to show.
        """
        self.info = {**self.info, **info}
        self.progress_bar.set_postfix(ordered_dict=self.info)

    def update_iter(self):
        """
            Update iteration counter.
        """
        self.progress_bar.update(1)

    def close(self):
        """
            Close progress bar.
        """
        self.progress_bar.close()


class FunctionWrapper(object):
    r"""
        Make the log-likelihood or log-prior function pickleable
        when ``args`` or ``kwargs`` are also included.

    Parameters
    ----------
    f : callable
        Log probability function.
    args : list
        Extra positional arguments to be passed to f.
    kwargs : dict
        Extra keyword arguments to be passed to f.
    """
    def __init__(self, f, args, kwargs):
        self.f = f
        self.args = [] if args is None else args
        self.kwargs = {} if kwargs is None else kwargs

    def __call__(self, x):
        """
            Evaluate log-likelihood or log-prior function.

        Parameters
        ----------
        x : ``np.ndarray``
            Input position array.

        Returns
        -------
        f : float or ``np.ndarray``
            f(x)
        """
        return self.f(x, *self.args, **self.kwargs)


def torch_to_numpy(x: torch.Tensor) -> np.ndarray:
    """
    Cast torch tensor to numpy.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.

    Returns
    -------
        Numpy array corresponding to the input tensor.
    """
    return x.detach().numpy()


def numpy_to_torch(x: np.ndarray) -> torch.Tensor:
    """
    Cast numpy array to torch tensor.

    Parameters
    ----------
    x : np.ndarray
        Input array.

    Returns
    -------
        Torch tensor corresponding to the input array.
    """
    return torch.tensor(x, dtype=torch.float32)


def torch_double_to_float(x: torch.Tensor, warn: bool = True):
    """
    Cast double precision (Float64) torch tensor to single precision (Float32).

    Parameters
    ----------
    x: torch.Tensor
        Input tensor.
    warn: bool
        If True, warn the user about the typecast.

    Returns
    -------
        Single precision (Float32) torch tensor.
    """
    if x.dtype == torch.float64 and warn:
        warnings.warn(f"Float64 data is currently unsupported, casting to Float32. Output will also have type Float32.")
        return x.float()
    elif x.dtype == torch.float32:
        return x
    else:
        raise ValueError(f"Unsupported datatype for input data: {x.dtype}")

class flow_numpy_wrapper:
    """
    Wrapper class for numpy flows.

    Parameters
    ----------
    flow : Flow object
        Flow object that implements forward and inverse
        transformations.
    
    Returns
    -------
    Flow object
    """
    def __init__(self, flow):
        self.flow = flow

    @torch.no_grad()
    def forward(self, v):
        v = numpy_to_torch(v)
        theta, logdetj = self.flow.forward(v)
        theta = torch_to_numpy(theta)
        logdetj = - torch_to_numpy(logdetj)
        return theta, logdetj

    @torch.no_grad()
    def inverse(self, theta):
        theta = numpy_to_torch(theta)
        v, logdetj = self.flow.inverse(theta)
        v = torch_to_numpy(v)
        logdetj = torch_to_numpy(logdetj)
        return v, logdetj
