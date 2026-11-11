from typing import Union, List

import numpy as np
from scipy.special import ndtr, ndtri, expit

from .input_validation import assert_array_float, assert_array_within_interval

class Reparameterize:
    """
    Class that reparameterises the model using change-of-variables parameter transformations.

    Parameters
    ----------
    n_dim : ``int``
        Dimensionality of sampling problem
    bounds : ``np.ndarray`` or ``list`` or ``None``
        Parameter bounds
    periodic : ``list``
        List of indices corresponding to periodic parameters (e.g. phases). A periodic
        parameter must have a finite lower and upper bound, which are treated as the same
        point. Before the logit/probit transformation is applied, a periodic parameter is
        shifted (and wrapped around) by a fraction ``periodic_shift`` of its period. This
        moves the point at which the period is cut open, which is the only point that the
        samples cannot cross. The shift is set by ``update_periodic_shift``, which places
        the cut where there are few samples. It is zero until then.
    transform : ``str``
        Type of transform to use for bounded parameters. Options are ``"probit"``
        (default) and ``"logit"``.
    scale : ``bool``
        Rescale parameters to zero mean and unit variance (default is true)
    diagonal : ``bool``
        Use diagonal transformation (i.e. ignore covariance) (default is true)

    Examples
    --------
    >>> import numpy as np
    >>> from pocomc.reparameterize import Reparameterize
    >>> bounds = np.array([[0, 1], [0, 1]])
    >>> reparam = Reparameterize(2, bounds)
    >>> x = np.array([[0.5, 0.5], [0.5, 0.5]])
    >>> reparam.forward(x)
    array([[0., 0.],
           [0., 0.]])
    >>> u = np.array([[0, 0], [0, 0]])
    >>> reparam.inverse(u)
    (array([[0.5, 0.5],
           [0.5, 0.5]]), array([0., 0.]))
    """
    def __init__(self,
                 n_dim: int,
                 bounds: Union[np.ndarray, list] = None,
                 periodic: List[int] = None,
                 transform: str = "probit",
                 scale: bool = True,
                 diagonal: bool = True):

        self.ndim = n_dim

        if bounds is None:
            bounds = np.tile(np.array([-np.inf, np.inf]), (self.ndim, 1))
        elif len(bounds) == 2 and not np.shape(bounds) == (2, 2):
            bounds = np.tile(np.array(bounds, dtype=np.float32).reshape(2, 1), self.ndim).T
        assert_array_float(bounds)

        self.low = bounds.T[0]
        self.high = bounds.T[1]

        if periodic is None or len(periodic) == 0:
            self.periodic = None
        else:
            self.periodic = [int(i) for i in periodic]

        # Shift of each parameter as a fraction of its period (non-zero only for periodic parameters)
        self.periodic_shift = np.zeros(self.ndim)

        if transform not in ["logit", "probit"]:
            raise ValueError("Please provide a valid transformation function (e.g. logit or probit)")
        else:
            self.transform = transform

        self.mu = None
        self.sigma = None
        self.cov = None
        self.L = None
        self.L_inv = None
        self.log_det_L = None
        self.scale = scale
        self.diagonal = diagonal

        self._create_masks()

        if self.periodic is not None:
            for i in self.periodic:
                if i < 0 or i >= self.ndim:
                    raise ValueError(f"Invalid periodic parameter index {i} for {self.ndim} parameters.")
                if not self.mask_both[i]:
                    raise ValueError(f"Periodic parameter {i} must have a finite lower and upper bound.")

    def update_periodic_shift(self, x: np.ndarray, weights: np.ndarray = None, threshold: float = 0.1):
        """
        Update the shift of the periodic parameters so that the samples ``x`` are centred
        in the interval, i.e. so that the point at which the period is cut open lies
        opposite to the (weighted) circular mean of the samples.

        The shift of a parameter is only changed if this improves the centring of the
        samples by more than ``threshold``. The centring is measured by the mean of
        ``cos(angle)``, where ``angle`` is the angular distance of a sample from the centre
        of the interval. It is at most equal to the mean resultant length ``R`` of the
        samples (``R`` is close to 0 for samples spread over the whole period and close
        to 1 for concentrated samples), which is attained when the circular mean is at
        the centre. Samples that are spread over the whole period therefore do not move
        the cut.

        Changing the shift changes the transformation between ``x`` and ``u``. Values of
        ``u`` computed before the change have to be recomputed from ``x``.

        Parameters
        ----------
        x : np.ndarray
            Samples of shape ``(n_samples, n_dim)``.
        weights : np.ndarray, optional
            Weights of the samples. The default is ``None`` (equal weights).
        threshold : float
            Minimum improvement of the centring that changes the shift (default is 0.1).

        Returns
        -------
        changed : bool
            ``True`` if the shift of any parameter changed.
        """
        if self.periodic is None:
            return False

        shift = np.array(self.periodic_shift, dtype=float)
        changed = False
        for i in self.periodic:
            # Angle of each sample from the centre of the interval, with the current shift
            q = np.mod((x[:, i] - self.low[i]) / (self.high[i] - self.low[i]) - shift[i], 1.0)
            angle = 2.0 * np.pi * (q - 0.5)
            c = float(np.average(np.cos(angle), weights=weights))
            s = float(np.average(np.sin(angle), weights=weights))
            if np.hypot(c, s) - c > threshold:
                # Move the circular mean to the centre of the interval
                shift[i] = np.mod(shift[i] + np.arctan2(s, c) / (2.0 * np.pi), 1.0)
                changed = True

        self.periodic_shift = shift
        return changed

    def _shifted_both(self):
        """
        Mask (over the parameters with both bounds) of the parameters with a non-zero
        periodic shift, or ``None`` if there are none.
        """
        shifted = self.periodic_shift[self.mask_both] != 0.0
        if np.any(shifted):
            return shifted
        return None

    def fit(self, x: np.ndarray):
        """
        Learn mean and standard deviation using for rescaling.
        
        Parameters
        ----------
        x : np.ndarray
            Input data used for training.
        """
        assert_array_within_interval(x, self.low, self.high)

        u = self._forward(x)
        self.mu = np.mean(u, axis=0)
        if self.diagonal:
            self.sigma = np.std(u, axis=0)
        else:
            self.cov = np.cov(u.T)
            self.L = np.linalg.cholesky(self.cov)
            self.L_inv = np.linalg.inv(self.L)
            self.log_det_L = np.linalg.slogdet(self.L)[1]

    def forward(self, x: np.ndarray, check_input=True):
        """
        Forward transformation (both logit/probit for bounds and affine for all parameters).

        Parameters
        ----------
        x : np.ndarray
            Input data
        check_input : bool
            Check if input is within bounds (default: True)
        Returns
        -------
        u : np.ndarray
            Transformed input data
        """
        if check_input:
            assert_array_within_interval(x, self.low, self.high)

        u = self._forward(x)
        if self.scale:
            u = self._forward_affine(u)

        return u

    def inverse(self, u: np.ndarray):
        """
        Inverse transformation (both logit^-1/probit^-1 for bounds and affine for all parameters).

        Parameters
        ----------
        u : np.ndarray
            Input data
        Returns
        -------
        x : np.ndarray
            Transformed input data
        log_det_J : np.array
            Logarithm of determinant of Jacobian matrix transformation.
        """
        if self.scale:
            x, log_det_J = self._inverse_affine(u)
            x, log_det_J_prime = self._inverse(x)
            log_det_J += log_det_J_prime
        else:
            x, log_det_J = self._inverse(u)

        return x, log_det_J

    def _forward(self, x: np.ndarray):
        """
        Forward transformation (only logit/probit for bounds).

        Parameters
        ----------
        x : np.ndarray
            Input data
        Returns
        -------
        u : np.ndarray
            Transformed input data
        """
        u = np.empty(x.shape)
        u[:, self.mask_none] = self._forward_none(x)
        u[:, self.mask_left] = self._forward_left(x)
        u[:, self.mask_right] = self._forward_right(x)
        u[:, self.mask_both] = self._forward_both(x)

        return u

    def _inverse(self, u: np.ndarray):
        """
        Inverse transformation (only logit^-1/probit^-1 for bounds).

        Parameters
        ----------
        u : np.ndarray
            Input data
        Returns
        -------
        x : np.ndarray
            Transformed input data
        log_det_J : np.array
            Logarithm of determinant of Jacobian matrix transformation.
        """
        x = np.empty(u.shape)
        J = np.empty(u.shape)

        x[:, self.mask_none], J[:, self.mask_none] = self._inverse_none(u)
        x[:, self.mask_left], J[:, self.mask_left] = self._inverse_left(u)
        x[:, self.mask_right], J[:, self.mask_right] = self._inverse_right(u)
        x[:, self.mask_both], J[:, self.mask_both] = self._inverse_both(u)

        log_det_J = np.sum(J, axis=1)

        return x, log_det_J

    def _forward_affine(self, x: np.ndarray):
        """
        Forward affine transformation.

        Parameters
        ----------
        x : np.ndarray
            Input data
        Returns
        -------
        Transformed input data
        """
        if self.diagonal:
            return (x - self.mu) / self.sigma
        else:
            return np.dot(x - self.mu, self.L_inv.T)

    def _inverse_affine(self, u: np.ndarray):
        """
        Inverse affine transformation.

        Parameters
        ----------
        u : np.ndarray
            Input data
        Returns
        -------
        x : np.ndarray
            Transformed input data
        J : np.ndarray
            Diagonal of Jacobian matrix.
        """
        if self.diagonal:
            log_det_J = np.sum(np.log(self.sigma))
            return self.mu + self.sigma * u, log_det_J * np.ones(len(u))
        else:
            x = self.mu + np.dot(u, self.L.T)
            return x, self.log_det_L * np.ones(len(u))

    def _forward_left(self, x: np.ndarray):
        """
        Forward transformation for bounded parameters (only low).

        Parameters
        ----------
        x : np.ndarray
            Input data
        Returns
        -------
        Transformed input data
        """
        return np.log(x[:, self.mask_left] - self.low[self.mask_left])

    def _inverse_left(self, u: np.ndarray):
        """
        Inverse transformation for bounded parameters (only low).

        Parameters
        ----------
        u : np.ndarray
            Input data
        Returns
        -------
        x : np.ndarray
            Transformed input data
        J : np.array
            Diagonal of Jacobian matrix.
        """
        return np.exp(u[:, self.mask_left]) + self.low[self.mask_left], u[:, self.mask_left]

    def _forward_right(self, x: np.ndarray):
        """
        Forward transformation for bounded parameters (only high).

        Parameters
        ----------
        x : np.ndarray
            Input data
        Returns
        -------
        Transformed input data
        """
        return np.log(self.high[self.mask_right] - x[:, self.mask_right])

    def _inverse_right(self, u: np.ndarray):
        """
        Inverse transformation for bounded parameters (only high).

        Parameters
        ----------
        u : np.ndarray
            Input data
        Returns
        -------
        x : np.ndarray
            Transformed input data
        J : np.array
            Diagonal of Jacobian matrix.
        """

        return self.high[self.mask_right] - np.exp(u[:, self.mask_right]), u[:, self.mask_right]

    def _forward_both(self, x: np.ndarray):
        """
        Forward transformation for bounded parameters (both low and high).

        Parameters
        ----------
        x : np.ndarray
            Input data
        Returns
        -------
        Transformed input data
        """
        low = self.low[self.mask_both]
        high = self.high[self.mask_both]
        width = high - low
        xb = x[:, self.mask_both]

        # Fractional distances from the lower and upper bound, each computed directly so
        # that both tails keep full precision (p_low + p_high = 1)
        p_low = (xb - low) / width
        p_high = (high - xb) / width

        # Shift the periodic parameters (wrapping them around), which moves the cut of the period
        shifted = self._shifted_both()
        if shifted is not None:
            q = np.mod(p_low[:, shifted] - self.periodic_shift[self.mask_both][shifted], 1.0)
            p_low[:, shifted] = q
            p_high[:, shifted] = 1.0 - q

        p_low = np.clip(p_low, 1e-13, 1.0 - 1e-13)
        p_high = np.clip(p_high, 1e-13, 1.0 - 1e-13)

        if self.transform == "logit":
            u = np.log(p_low) - np.log(p_high)
        elif self.transform == "probit":
            u = np.where(p_low < 0.5, ndtri(p_low), -ndtri(p_high))

        return u

    def _inverse_both(self, u: np.ndarray):
        """
        Inverse transformation for bounded parameters (both low and high).

        Parameters
        ----------
        u : np.ndarray
            Input data
        Returns
        -------
        x : np.ndarray
            Transformed input data
        J : np.array
            Diagonal of Jacobian matrix.
        """
        low = self.low[self.mask_both]
        high = self.high[self.mask_both]
        width = high - low
        ub = u[:, self.mask_both]

        # Fractional distances from the lower (p_low) and upper (p_high) bound, each
        # accurate in its own tail
        if self.transform == "logit":
            p_low = expit(ub)
            p_high = expit(-ub)
            # log(width * sigmoid(u) * sigmoid(-u)), finite for all finite u
            J = np.log(width) - np.logaddexp(0.0, -ub) - np.logaddexp(0.0, ub)
        elif self.transform == "probit":
            p_low = ndtr(ub)
            p_high = ndtr(-ub)
            J = np.log(width) + (-ub**2.0 / 2.0) - np.log(np.sqrt(2.0 * np.pi))

        # Measure from the nearer bound so that points close to either bound keep full precision
        x = np.where(ub <= 0.0, low + p_low * width, high - p_high * width)

        # Undo the shift of the periodic parameters (the Jacobian is not affected)
        shifted = self._shifted_both()
        if shifted is not None:
            q = np.where(ub[:, shifted] <= 0.0, p_low[:, shifted], 1.0 - p_high[:, shifted])
            q = np.mod(q + self.periodic_shift[self.mask_both][shifted], 1.0)
            x[:, shifted] = np.where(q <= 0.5, low[shifted] + q * width[shifted], high[shifted] - (1.0 - q) * width[shifted])

        return x, J

    def _forward_none(self, x:np.ndarray):
        """
        Forward transformation for unbounded parameters (this does nothing).

        Parameters
        ----------
        x : np.ndarray
            Input data
        Returns
        -------
        u : np.ndarray
            Transformed input data
        """
        return x[:, self.mask_none]

    def _inverse_none(self, u:np.ndarray):
        """
        Inverse transformation for unbounded parameters (this does nothing).

        Parameters
        ----------
        u : np.ndarray
            Input data
        Returns
        -------
        x : np.ndarray
            Transformed input data
        log_det_J : np.array
            Logarithm of determinant of Jacobian matrix transformation.
        """
        return u[:, self.mask_none], np.zeros(u.shape)[:, self.mask_none]

    def _create_masks(self):
        """
        Create parameter masks for bounded parameters
        """

        self.mask_left = np.zeros(self.ndim, dtype=bool)
        self.mask_right = np.zeros(self.ndim, dtype=bool)
        self.mask_both = np.zeros(self.ndim, dtype=bool)
        self.mask_none = np.zeros(self.ndim, dtype=bool)

        # TODO: Do this more elegantly, it's a shame
        for i in range(self.ndim):
            if not np.isfinite(self.low[i]) and not np.isfinite(self.high[i]):
                self.mask_none[i] = True
                self.mask_left[i] = False
                self.mask_right[i] = False
                self.mask_both[i] = False
            elif not np.isfinite(self.low[i]) and np.isfinite(self.high[i]):
                self.mask_none[i] = False
                self.mask_left[i] = False
                self.mask_right[i] = True
                self.mask_both[i] = False
            elif np.isfinite(self.low[i]) and not np.isfinite(self.high[i]):
                self.mask_none[i] = False
                self.mask_left[i] = True
                self.mask_right[i] = False
                self.mask_both[i] = False
            else:
                self.mask_none[i] = False
                self.mask_left[i] = False
                self.mask_right[i] = False
                self.mask_both[i] = True
