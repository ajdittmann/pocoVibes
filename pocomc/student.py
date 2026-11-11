import math
import warnings
import numpy as np


def _digamma(x):
    """
    Digamma function for a positive real scalar ``x``.

    Uses the recurrence ``psi(x) = psi(x + 1) - 1/x`` to shift the argument to
    ``x >= 10`` and then the asymptotic expansion, which is accurate to double
    precision there.
    """
    x = float(x)
    result = 0.0
    while x < 10.0:
        result -= 1.0 / x
        x += 1.0
    f = 1.0 / (x * x)
    series = f * (1.0 / 12 - f * (1.0 / 120 - f * (1.0 / 252 - f * (1.0 / 240 - f * (1.0 / 132 - f * (691.0 / 32760 - f / 12.0))))))
    return result + math.log(x) - 0.5 / x - series


def _log_minus_digamma(x):
    """
    Compute ``log(x) - psi(x)`` for a positive real scalar ``x``.

    For ``x >= 10`` the asymptotic expansion is summed directly, which avoids the
    catastrophic cancellation of evaluating ``log(x)`` and ``psi(x)`` separately
    (both grow like ``log(x)`` while their difference decays like ``1/(2x)``).
    """
    x = float(x)
    if x < 10.0:
        return math.log(x) - _digamma(x)
    f = 1.0 / (x * x)
    series = f * (1.0 / 12 - f * (1.0 / 120 - f * (1.0 / 252 - f * (1.0 / 240 - f * (1.0 / 132 - f * (691.0 / 32760 - f / 12.0))))))
    return 0.5 / x + series


def _brent_log(func, a, b, x0=None, xtol=1e-14, maxiter=200):
    """
    Find a root of ``func`` in ``[a, b]`` (with ``0 < a < b``) using Brent's
    method in ``t = log(x)``. Requires ``func(a) > 0 > func(b)`` or the reverse.

    If ``x0`` is given, a bracket around ``log(x0)`` is searched first (expanding
    geometrically), which keeps the function evaluations at moderate ``x``.
    """
    lo, hi = math.log(a), math.log(b)
    g = lambda t: func(math.exp(t))

    if x0 is not None and a < x0 < b:
        t0 = math.log(x0)
        width = 1.0
        ta, tb = max(t0 - width, lo), min(t0 + width, hi)
        fa, fb = g(ta), g(tb)
        # The equation is positive below its root and negative above it, so a
        # common sign tells which way to extend the bracket.
        while (fa > 0) == (fb > 0) and fa != 0.0 and fb != 0.0:
            width *= 2.0
            if fa > 0:
                if tb == hi:
                    break
                ta, fa = tb, fb
                tb = min(t0 + width, hi)
                fb = g(tb)
            else:
                if ta == lo:
                    break
                tb, fb = ta, fa
                ta = max(t0 - width, lo)
                fa = g(ta)
    else:
        ta, tb = lo, hi
        fa, fb = g(ta), g(tb)

    if (fa > 0) == (fb > 0) and fa != 0.0 and fb != 0.0:
        ta, tb = lo, hi
        fa, fb = g(ta), g(tb)
    if fa == 0.0:
        return math.exp(ta)
    if fb == 0.0:
        return math.exp(tb)

    # Brent's method (as in Numerical Recipes' zbrent)
    eps = np.finfo(float).eps
    c, fc = tb, fb
    d = e = tb - ta
    for _ in range(maxiter):
        if (fb > 0) == (fc > 0):
            c, fc = ta, fa
            d = e = tb - ta
        if abs(fc) < abs(fb):
            ta, tb, c = tb, c, tb
            fa, fb, fc = fb, fc, fb
        tol1 = 2.0 * eps * abs(tb) + 0.5 * xtol
        xm = 0.5 * (c - tb)
        if abs(xm) <= tol1 or fb == 0.0:
            return math.exp(tb)
        if abs(e) >= tol1 and abs(fa) > abs(fb):
            s = fb / fa
            if ta == c:
                p = 2.0 * xm * s
                q = 1.0 - s
            else:
                q = fa / fc
                r = fb / fc
                p = s * (2.0 * xm * q * (q - r) - (tb - ta) * (r - 1.0))
                q = (q - 1.0) * (r - 1.0) * (s - 1.0)
            if p > 0.0:
                q = -q
            p = abs(p)
            if 2.0 * p < min(3.0 * xm * q - abs(tol1 * q), abs(e * q)):
                e = d
                d = p / q
            else:
                d = xm
                e = d
        else:
            d = xm
            e = d
        ta, fa = tb, fb
        if abs(d) > tol1:
            tb += d
        else:
            tb += math.copysign(tol1, xm)
        fb = g(tb)
    return math.exp(tb)


def fit_mvstud(data, tolerance=1e-6, max_iter=100):
    """
    Fit a multivariate Student's t distribution to data using the EM algorithm.

    Parameters
    ----------
    data : ndarray
        An array of shape (n, dim) containing n samples of dimension dim.
    tolerance : float, optional
        The relative tolerance for convergence of the degrees of freedom ``nu``
        (the EM iteration stops when ``|nu_new - nu_old| <= tolerance * max(nu_new, 1)``).
        The default is 1e-6.
    max_iter : int, optional
        The maximum number of iterations. The default is 100.
    
    Returns
    -------
    mu : ndarray
        The mean of the distribution.
    Sigma : ndarray
        The covariance matrix of the distribution.
    nu : float
        The degrees of freedom of the distribution.
    
    Examples
    --------
    >>> import numpy as np
    >>> from pocomc.student import fit_mvstud
    >>> data = np.random.randn(100, 2)
    >>> mu, Sigma, nu = fit_mvstud(data)
    >>> mu
    array([ 0.00323705, -0.05405479])
    >>> Sigma
    array([[ 1.00524016, -0.02020086],
           [-0.02020086,  0.99111344]])
    >>> nu
    20.000000000000004
    """
    # Largest finite degrees of freedom considered. Above this the t distribution
    # is indistinguishable from a Gaussian (``Geometry`` maps ``nu = inf`` to 1e6).
    nu_max = 1e6

    def opt_nu(delta_iobs, nu):
        def func0(nu):
            # Same equation as
            #   -psi(nu/2) + log(nu/2) + mean(log(w)) - mean(w) + 1 + psi((nu+dim)/2) - log((nu+dim)/2),
            # with w = (nu + dim) / (nu + delta), written without cancellation:
            # log(x) - psi(x) is evaluated directly and, with e = w - 1,
            # log(w) - w + 1 = log1p(e) - e.
            e_iobs = (dim - delta_iobs) / (nu + delta_iobs)
            f = _log_minus_digamma(nu/2) - _log_minus_digamma((nu+dim)/2) + np.mean(np.log1p(e_iobs) - e_iobs)
            return f

        # If the equation has no root below nu_max, the data are consistent with a Gaussian
        if func0(nu_max) >= 0:
            nu = np.inf
        else:
            # The root is located in log(nu) to a fraction of the EM tolerance
            nu = _brent_log(func0, 1e-300, nu_max, x0=nu, xtol=0.01 * tolerance)
        return nu

    data = data.T
    (dim,n) = data.shape
    mu = np.array([np.median(data,1)]).T
    Sigma = np.cov(data)*(n-1)/n + (1/n)*np.diag(np.var(data, axis=1))
    nu = 20

    last_nu = 0
    i = 0
    while np.abs(last_nu - nu) > tolerance * max(nu, 1.0) and i < max_iter:
        i += 1
        diffs = data - mu
        delta_iobs = np.sum(diffs * np.linalg.solve(Sigma,diffs), 0)
        
        # update nu
        last_nu = nu
        nu = opt_nu(delta_iobs, nu)
        if nu == np.inf:
            return mu.T[0], Sigma, nu

        # update Sigma
        w_iobs = (nu + dim) / (nu + delta_iobs)
        Sigma = np.dot(w_iobs*diffs, diffs.T) / n

        # update mu
        mu = np.sum(w_iobs * data, 1) / np.sum(w_iobs)
        mu = np.array([mu]).T

    # EM approaches nu slowly, so reaching max_iter with a tiny last change is normal; only
    # warn when nu was still changing appreciably (by more than 1% in the last iteration)
    if i == max_iter and np.abs(last_nu - nu) > 0.01 * max(nu, 1.0):
        warnings.warn(f"EM algorithm for the Student's t distribution did not converge in {max_iter} "
                      f"iterations (last nu: {last_nu}, current nu: {nu}).", RuntimeWarning, stacklevel=2)

    return mu.T[0], Sigma, nu
