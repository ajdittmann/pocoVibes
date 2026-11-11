from typing import Any
import numpy as np

from .tools import logsumexp

class Particles:
    """
    Class to store the particles and their associated weights.

    Parameters
    ----------
    n_particles : int
        Number of particles.
    n_dim : int
        Dimension of the parameter space.
    ess_threshold : float, optional
        Threshold for the effective sample size. If the effective sample size
        is below this threshold, the weights are set to zero. This is useful
        for the case where the effective sample size is very small, but not
        exactly zero, due to numerical errors.
    
    Attributes
    ----------
    n_particles : int
        Number of particles.
    n_dim : int
        Dimension of the parameter space.
    ess_threshold : float, optional
        Threshold for the effective sample size. If the effective sample size
        is below this threshold, the weights are set to zero. This is useful
        for the case where the effective sample size is very small, but not
        exactly zero, due to numerical errors.
    u : numpy.ndarray
        Array of shape (n_particles, n_dim) containing the particles.
    logdetj : numpy.ndarray
        Array of shape (n_particles,) containing the log-determinant of the
        Jacobian of the transformation from the unit hypercube to the
        parameter space.
    logl : numpy.ndarray
        Array of shape (n_particles,) containing the log-likelihoods.
    logp : numpy.ndarray
        Array of shape (n_particles,) containing the log-priors.
    logw : numpy.ndarray
        Array of shape (n_particles,) containing the log-weights.
    iter : numpy.ndarray
        Array of shape (n_particles,) containing the iteration number of each
        particle.
    logz : numpy.ndarray
        Array of shape (n_particles,) containing the log-evidence of each
        particle.
    calls : numpy.ndarray
        Array of shape (n_particles,) containing the number of likelihood
        evaluations of each particle.
    steps : numpy.ndarray
        Array of shape (n_particles,) containing the number of steps of each
        particle.
    efficiency : numpy.ndarray
        Array of shape (n_particles,) containing the efficiency of each
        particle.
    ess : numpy.ndarray
        Array of shape (n_particles,) containing the effective sample size of
        each particle.
    accept : numpy.ndarray
        Array of shape (n_particles,) containing the acceptance rate of each
        particle.
    beta : numpy.ndarray
        Array of shape (n_particles,) containing the inverse temperature of
        each particle.
    """

    def __init__(self, n_particles, n_dim):
        self.n_particles = n_particles
        self.n_dim = n_dim

        self.past = dict(
            u = [],
            x = [],
            logdetj = [],
            logl = [],
            logp = [],
            logw = [],
            blobs = [],
            iter = [],
            logz = [],
            calls = [],
            steps = [],
            efficiency = [],
            ess = [],
            accept = [],
            beta = [],
        )

        self.results_dict = None

        # Cache for the (beta_final-independent) part of the importance weights
        self._logw_cache = None

    def update(self, data):
        """
        Update the particles with the given data.

        Parameters
        ----------
        data : dict
            Dictionary containing the data to be added to the particles.
        
        Notes
        -----
        The dictionary must contain the following keys:
            u : numpy.ndarray
                Array of shape (n_particles, n_dim) containing the particles.
            logdetj : numpy.ndarray
                Array of shape (n_particles,) containing the log-determinant
                of the Jacobian of the transformation from the unit hypercube
                to the parameter space.
            logl : numpy.ndarray
                Array of shape (n_particles,) containing the log-likelihoods.
            logp : numpy.ndarray
                Array of shape (n_particles,) containing the log-priors.
            logw : numpy.ndarray
                Array of shape (n_particles,) containing the log-weights.
            blobs : numpy.ndarray
                Array of shape (n_particles,) containing the blobs (derived parameters).
            iter : numpy.ndarray
                Array of shape (n_particles,) containing the iteration number
                of each particle.
            logz : numpy.ndarray
                Array of shape (n_particles,) containing the log-evidence of
                each particle.
            calls : numpy.ndarray
                Array of shape (n_particles,) containing the number of
                likelihood evaluations of each particle.
            steps : numpy.ndarray
                Array of shape (n_particles,) containing the number of steps
                of each particle.
            efficiency : numpy.ndarray
                Array of shape (n_particles,) containing the efficiency of
                each particle.
            ess : numpy.ndarray
                Array of shape (n_particles,) containing the effective sample
                size of each particle.
            accept : numpy.ndarray
                Array of shape (n_particles,) containing the acceptance rate
                of each particle.
            beta : numpy.ndarray
                Array of shape (n_particles,) containing the inverse
                temperature of each particle.
        """
        for key in data.keys():
            if key in self.past.keys():
                value = data.get(key)
                # Save to past states
                self.past.get(key).append(value)
        # Stored particles changed: cached results are stale
        self.results_dict = None

    def pop(self, key):
        """
        Remove the last element of the given key.

        Parameters
        ----------
        key : str
            Key of the element to be removed.
        
        Notes
        -----
        This method is useful to remove the last element of the particles
        after the resampling step.
        """
        _ = self.past.get(key).pop()
        self.results_dict = None
        self._logw_cache = None

    def get(self, key, index=None, flat=False):
        """
        Get the element of the given key.

        Parameters
        ----------
        key : str
            Key of the element to be returned.
        index : int, optional
            Index of the element to be returned. If None, all elements are
            returned.
        flat : bool, optional
            If True, the elements are returned as a flattened array. Otherwise,
            the elements are returned as a numpy.ndarray.
        
        Returns
        -------
        element : numpy.ndarray
            Array of shape (n_particles,) or (n_particles, n_dim) containing
            the elements of the given key.
        
        Notes
        -----
        If index is None, the elements are returned as a numpy.ndarray. If
        index is not None, the elements are returned as a numpy.ndarray with
        shape (n_dim,). If flat is True, the elements are returned as a
        flattened array.

        Examples
        --------
        >>> particles = Particles(n_particles=10, n_dim=2)
        >>> particles.update(dict(u=np.random.randn(10,2)))
        >>> particles.get("u").shape
        (10, 2)
        >>> particles.get("u", index=0).shape
        (2,)
        >>> particles.get("u", index=0, flat=True).shape
        (2,)
        >>> particles.get("u", index=None, flat=True).shape
        (20,)
        """
        if index is None:
            if flat:
                return np.concatenate(self.past.get(key))
            else:
                return np.asarray(self.past.get(key))
        else:
            return self.past.get(key)[index]

    def take(self, key, flat_idx):
        """
        Return the elements of the given key at the given indices of the flattened history,
        i.e. ``get(key, flat=True)[flat_idx]``, without concatenating the whole history.

        Parameters
        ----------
        key : str
            Key of the elements to be returned.
        flat_idx : numpy.ndarray
            Indices into the flattened history (concatenation of all iterations).

        Returns
        -------
        elements : numpy.ndarray
            The selected elements, in the order of ``flat_idx``.
        """
        arrays = self.past.get(key)
        flat_idx = np.asarray(flat_idx, dtype=np.int64)
        lengths = np.fromiter((len(a) for a in arrays), dtype=np.int64, count=len(arrays))
        starts = np.concatenate(([0], np.cumsum(lengths)))
        n_total = starts[-1]
        if np.any(flat_idx < -n_total) or np.any(flat_idx >= n_total):
            raise IndexError(f"Index out of range for a history of {n_total} elements.")
        flat_idx = np.where(flat_idx < 0, flat_idx + n_total, flat_idx)

        # Iteration of each index; gather iteration by iteration
        iteration = np.searchsorted(starts, flat_idx, side="right") - 1
        # Indices from trimming are already sorted, which makes reordering unnecessary
        is_sorted = len(iteration) < 2 or bool(np.all(iteration[1:] >= iteration[:-1]))
        order = None if is_sorted else np.argsort(iteration, kind="stable")
        sorted_iteration = iteration if is_sorted else iteration[order]
        sorted_idx = flat_idx if is_sorted else flat_idx[order]
        bounds = np.searchsorted(sorted_iteration, np.arange(len(arrays) + 1), side="left")
        pieces = []
        for i in range(len(arrays)):
            if bounds[i] == bounds[i + 1]:
                continue
            local = sorted_idx[bounds[i]:bounds[i + 1]] - starts[i]
            pieces.append(np.asarray(arrays[i])[local])
        gathered = np.concatenate(pieces) if pieces else np.concatenate([np.asarray(a)[:0] for a in arrays])
        if is_sorted:
            return gathered
        # Restore the requested order
        result = np.empty_like(gathered)
        result[order] = gathered
        return result

    def _logw_denominator(self):
        """
        Return the stacked log-likelihoods and the ``beta_final``-independent
        mixture denominator of the importance weights.

        The denominator of particle ``n`` from iteration ``i`` is
        ``log(sum_j exp(logl[i, n] * beta[j] - logz[j]))`` over all stored
        iterations ``j``, evaluated as a sequential ``logaddexp`` over ``j``.
        Since past temperatures and evidences never change, the result is cached
        and only extended with the terms of newly added iterations: existing
        rows receive the new terms, and new rows are built from scratch.
        """
        logl_list = self.past.get("logl")
        beta_list = self.past.get("beta")
        logz_list = self.past.get("logz")
        n_iter = len(logl_list)
        if len(beta_list) != n_iter or len(logz_list) != n_iter:
            raise ValueError("Inconsistent number of stored iterations for logl, beta and logz.")

        cache = getattr(self, "_logw_cache", None)
        if cache is not None and cache[0] == n_iter:
            logl, B = cache[1], cache[2]
        else:
            if cache is not None and 0 < cache[0] < n_iter:
                # Extend the cached denominator with the new iterations
                n_old = cache[0]
                logl_old, B_old = cache[1], cache[2]
                logl_new = np.asarray(logl_list[n_old:])
                for j in range(n_old, n_iter):
                    B_old = np.logaddexp(B_old, logl_old * beta_list[j] - logz_list[j])
                B_new = logl_new * beta_list[0] - logz_list[0]
                for j in range(1, n_iter):
                    B_new = np.logaddexp(B_new, logl_new * beta_list[j] - logz_list[j])
                logl = np.concatenate([logl_old, logl_new], axis=0)
                B = np.concatenate([B_old, B_new], axis=0)
            else:
                logl = np.asarray(logl_list)
                B = logl * beta_list[0] - logz_list[0]
                for j in range(1, n_iter):
                    B = np.logaddexp(B, logl * beta_list[j] - logz_list[j])
            self._logw_cache = (n_iter, logl, B)

        return logl, B - np.log(n_iter)

    def compute_logw_and_logz(self, beta_final=1.0, normalize=True):

        logl, B = self._logw_denominator()

        A = logl * beta_final
        logw = A - B
        logw = np.concatenate(logw)
        logw_sum = logsumexp(logw)
        logz_new = logw_sum - np.log(len(logw))

        if normalize:
            logw -= logw_sum

        return logw, logz_new
    
    def compute_results(self):
        """
        Compute the results of the particles.

        Returns
        -------
        results_dict : dict
            Dictionary containing the results of the particles.

        Notes
        -----
        The dictionary contains the following keys:
            u : numpy.ndarray
                Array of shape (n_particles, n_dim) containing the particles.
            logdetj : numpy.ndarray
                Array of shape (n_particles,) containing the log-determinant
                of the Jacobian of the transformation from the unit hypercube
                to the parameter space.
            logl : numpy.ndarray
                Array of shape (n_particles,) containing the log-likelihoods.
            logp : numpy.ndarray
                Array of shape (n_particles,) containing the log-priors.
            logw : numpy.ndarray
                Array of shape (n_particles,) containing the log-weights.
            blobs : numpy.ndarray
                Array of shape (n_particles,) containing the blobs (derived parameters).
            iter : numpy.ndarray
                Array of shape (n_particles,) containing the iteration number
                of each particle.
            logz : numpy.ndarray
                Array of shape (n_particles,) containing the log-evidence of
                each particle.
            calls : numpy.ndarray
                Array of shape (n_particles,) containing the number of
                likelihood evaluations of each particle.
            steps : numpy.ndarray
                Array of shape (n_particles,) containing the number of steps
                of each particle.
            efficiency : numpy.ndarray
                Array of shape (n_particles,) containing the efficiency of
                each particle.
            ess : numpy.ndarray
                Array of shape (n_particles,) containing the effective sample
                size of each particle.
            accept : numpy.ndarray
                Array of shape (n_particles,) containing the acceptance rate
                of each particle.
            beta : numpy.ndarray
                Array of shape (n_particles,) containing the inverse
                temperature of each particle.

        Examples
        --------
        >>> particles = Particles(n_particles=10, n_dim=2)
        >>> particles.update(dict(u=np.random.randn(10,2)))
        >>> particles.compute_results().keys()
        dict_keys(['u', 'logdetj', 'logl', 'logp', 'logw', 'blobs', 'iter', 'logz', 'calls', 'steps', 'efficiency', 'ess', 'accept', 'beta'])
        """
        if self.results_dict is None:
            self.results_dict = dict()
            for key in self.past.keys():
                self.results_dict[key] = self.get(key)

            logw, _ = self.compute_logw_and_logz(1.0)

            self.results_dict["logw"] = logw
            #self.results_dict["ess"] = np.exp(log_ess)

        return self.results_dict

