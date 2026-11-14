"""Participant submission: batched model-based and model-free search methods.

Teams replace the search strategy in ``calibrate`` while preserving the three
public names below. The notebook independently reruns and scores the returned
parameters, so submissions must return only the best parameter dictionary.

``ALGORITHM_NAME`` selects the search strategy (see ``_ALGORITHMS``):

- ``random_search``: uniform sampling.
- ``bayesian_optimization`` (and ``_cl``, ``_mes``): Gaussian-process surrogate.
- ``simple_bayesian_optimization``: baseline with a fixed, unfitted GP.
- ``bayesian_optimization_screened``: screening design first, then Bayesian
  optimization focused on the parameters that matter for the building.
- ``ard_bayesian_optimisation``: Bayesian optimization that, mid-run, varies
  only the parameters with the shortest fitted GP length scales.
- ``caliblab``: the CalibLab pipeline (https://github.com/TUB-DVG/CalibLab):
  sensitivity analysis, GP metamodel, Bayesian calibration by MCMC.
- ``tpe``: Tree-structured Parzen Estimator.
- ``cma_es``: Covariance Matrix Adaptation Evolution Strategy.
- ``hyperband``: Hyperband over local-search arms (see its docstring: the
  objective has no cheap fidelity, so the resource is evaluations per arm).

Everything is implemented with NumPy, SciPy, and scikit-learn only, because
the official environment has no dedicated optimization packages.
"""

from __future__ import annotations

import warnings
from typing import Callable, Sequence

import numpy as np
from scipy.linalg import hadamard
from scipy.special import logsumexp
from scipy.stats import norm, qmc
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.gaussian_process.kernels import RBF, ExpSineSquared, RationalQuadratic  # caliblab


TEAM_NAME = "Calibra"
ALGORITHM_NAME = "ard_bayesian_optimisation"

# Stop if this many consecutive batches add no new (budget-charged) candidates,
# e.g. when all proposals collapse onto already-evaluated whole-hour schedules.
_MAX_STALLED_BATCHES = 5


class _History:
    """Evaluates candidates in the unit cube and remembers every result.

    The search methods work on [0, 1]^d; this class maps to physical bounds,
    sizes batches to the remaining budget, and tracks the best point so far.
    """

    def __init__(self, objective, lower: np.ndarray, upper: np.ndarray, budget: int):
        self.objective = objective
        self.lower = lower
        self.upper = upper
        self.budget = budget
        self.submitted = 0
        self.stalled = 0
        self.batches = 0
        self.X: list[np.ndarray] = []  # unit-cube points
        self.y: list[float] = []  # WAPE losses (%)

    def to_physical(self, unit: np.ndarray) -> np.ndarray:
        return self.lower + np.clip(unit, 0.0, 1.0) * (self.upper - self.lower)

    def to_unit(self, physical: np.ndarray) -> np.ndarray:
        span = np.where(self.upper > self.lower, self.upper - self.lower, 1.0)
        return np.clip((physical - self.lower) / span, 0.0, 1.0)

    def remaining(self) -> int:
        remaining = self.objective.remaining_budget
        if remaining is None:  # Unlimited objective: use our own upper-bound count.
            remaining = self.budget - self.submitted
        return max(0, int(remaining))

    def can_continue(self) -> bool:
        return self.remaining() > 0 and self.stalled < _MAX_STALLED_BATCHES

    def next_batch_size(self) -> int:
        return max(1, min(self.objective.max_batch_size, self.remaining()))

    def evaluate(self, units: np.ndarray) -> None:
        units = np.atleast_2d(units)[: self.remaining()]
        if len(units) == 0:
            return
        before = self.objective.remaining_budget
        vectors = [tuple(map(float, self.to_physical(u))) for u in units]
        losses = self.objective.evaluate_many(vectors)
        self.batches += 1
        self.submitted += len(units)
        if before is not None:
            self.stalled = self.stalled + 1 if self.objective.remaining_budget == before else 0
        self.X.extend(np.clip(units, 0.0, 1.0))
        self.y.extend(float(v) for v in losses)

    def best(self) -> tuple[np.ndarray, float]:
        index = int(np.argmin(self.y))
        return self.to_physical(self.X[index]), self.y[index]


def random_search(history: _History, start: np.ndarray, rng: np.random.Generator) -> None:
    """Uniform random sampling within the bounds, in batches.

    The start vector (the notebook's initial values) is evaluated first, so
    the result is never worse than the starting point.
    """

    history.evaluate(history.to_unit(start))
    while history.can_continue():
        history.evaluate(rng.uniform(size=(history.next_batch_size(), len(start))))


def bayesian_optimization(
    history: _History, start: np.ndarray, rng: np.random.Generator
) -> None:
    """Expected Improvement, batch filled with the kriging believer."""

    _bo_loop(history, start, rng, _expected_improvement, _kriging_believer)


def bayesian_optimization_cl(
    history: _History, start: np.ndarray, rng: np.random.Generator
) -> None:
    """Expected Improvement, batch filled with a pessimistic constant liar."""

    _bo_loop(history, start, rng, _expected_improvement, _constant_liar_max)


def bayesian_optimization_mes(
    history: _History, start: np.ndarray, rng: np.random.Generator
) -> None:
    """Max-value Entropy Search, batch filled with the kriging believer."""

    _bo_loop(history, start, rng, _max_value_entropy_search, _kriging_believer)

def bayesian_optimization_ucb(
    history: _History, start: np.ndarray, rng: np.random.Generator
) -> None:
    """GP confidence bound (UCB on -loss, i.e. LCB on loss), kriging believer."""

    _bo_loop(history, start, rng, _lower_confidence_bound, _kriging_believer)

def _bo_loop(
    history: _History,
    start: np.ndarray,
    rng: np.random.Generator,
    acquisition: Callable,
    fantasy: Callable,
) -> None:
    """Batched Bayesian optimization with a Gaussian-process surrogate.

    1. Initial design: the start vector plus a Latin hypercube sample.
    2. Each round: fit a GP (Matern 5/2, one length scale per parameter) to
       log(1 + WAPE), then fill a batch greedily: pick the candidate that
       maximizes ``acquisition``, add it to the data with the fake loss
       returned by ``fantasy``, recondition the GP, and pick the next one, so
       the batch spreads out instead of piling onto the same optimum.

    ``acquisition(gp, candidates, best, rng)`` returns one score per candidate
    (higher is better). ``fantasy(gp, point, y_observed)`` returns the fake
    log-loss assumed for a point that is chosen but not yet evaluated.
    """

    dim = len(start)
    # Enough points to fit a first GP, but at most half the budget, so the
    # model-guided rounds always get the other half.
    n_init = max(dim + 1, history.objective.max_batch_size)
    n_init = max(1, min(n_init, history.remaining() // 2))
    design = qmc.LatinHypercube(d=dim, seed=rng)
    initial = np.vstack([history.to_unit(start), design.random(max(0, n_init - 1))])
    history.evaluate(initial)

    while history.can_continue():
        batch = _propose_batch(
            np.array(history.X),
            np.array(history.y),
            history.next_batch_size(),
            rng,
            acquisition,
            fantasy,
        )
        history.evaluate(batch)


def _fit_gp(X: np.ndarray, y: np.ndarray, rng: np.random.Generator, kernel=None):
    dim = X.shape[1]
    if kernel is None:
        kernel = ConstantKernel(1.0, (1e-2, 1e2)) * Matern(
            length_scale=np.full(dim, 0.5), length_scale_bounds=(1e-2, 1e2), nu=2.5
        ) + WhiteKernel(1e-4, (1e-8, 1e-1))
        optimizer, restarts = "fmin_l_bfgs_b", 3
    else:  # Reuse fitted hyperparameters (fast refit on fantasy data).
        optimizer, restarts = None, 0
    gp = GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        optimizer=optimizer,
        n_restarts_optimizer=restarts,
        random_state=int(rng.integers(2**31 - 1)),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        gp.fit(X, y)
    return gp


def _expected_improvement(gp, candidates: np.ndarray, best: float, rng=None) -> np.ndarray:
    # ``rng`` is unused; it keeps the signature shared with the other acquisitions.
    mean, std = gp.predict(candidates, return_std=True)
    std = np.maximum(std, 1e-9)
    z = (best - mean) / std
    return (best - mean) * norm.cdf(z) + std * norm.pdf(z)


def _max_value_entropy_search(
    gp,
    candidates: np.ndarray,
    best: float,
    rng: np.random.Generator,
    n_samples: int = 10,
) -> np.ndarray:
    """Max-value Entropy Search (Wang & Jegelka, 2017), written for minimization.

    Scores how much evaluating x would reduce the uncertainty about the value
    of the global minimum m*. Samples of m* come from a Gumbel fit to
    P(m* > m) ~ prod_x P(f(x) > m) over the candidates and observed points.
    """

    mean, std = gp.predict(candidates, return_std=True)
    std = np.maximum(std, 1e-9)

    # Observed points pin m* to (at most) the best value seen so far.
    seen_mean, seen_std = gp.predict(gp.X_train_, return_std=True)
    all_mean = np.concatenate([mean, seen_mean])
    all_std = np.maximum(np.concatenate([std, seen_std]), 1e-9)

    def log_prob_min_above(m: np.ndarray) -> np.ndarray:
        # log P(m* > m) for each m, assuming independent points.
        return norm.logcdf((all_mean[None, :] - m[:, None]) / all_std[None, :]).sum(axis=1)

    # Bisection for the m with P(m* > m) = 0.75, 0.5, 0.25.
    quantiles = np.array([0.75, 0.5, 0.25])
    low = np.full(3, float(np.min(all_mean - 6.0 * all_std)))
    high = np.full(3, float(np.min(all_mean + 6.0 * all_std)))
    for _ in range(60):
        middle = 0.5 * (low + high)
        above = log_prob_min_above(middle) > np.log(quantiles)
        low = np.where(above, middle, low)
        high = np.where(above, high, middle)
    q75, q50, q25 = 0.5 * (low + high)

    # m* = -g* with g* ~ Gumbel(a, b) for the maximum of -f.
    b = max((q25 - q75) / (np.log(-np.log(0.25)) - np.log(-np.log(0.75))), 1e-12)
    a = -q50 + b * np.log(-np.log(0.5))
    u = rng.uniform(1e-12, 1.0 - 1e-12, size=n_samples)
    min_values = np.minimum(-(a - b * np.log(-np.log(u))), best)

    gamma = (mean[:, None] - min_values[None, :]) / std[:, None]
    log_cdf = norm.logcdf(gamma)
    pdf_over_cdf = np.exp(norm.logpdf(gamma) - log_cdf)  # stable for very negative gamma
    score = 0.5 * gamma * pdf_over_cdf - log_cdf
    return score.mean(axis=1)

UCB_KAPPA = 2  # exploration weight of the confidence-bound acquisitions


def _upper_confidence_bound(gp, candidates: np.ndarray, kappa: float | None = None) -> np.ndarray:
    """Return UCB values for a reward-maximizing Gaussian process."""
    kappa = UCB_KAPPA if kappa is None else kappa
    mean, std = gp.predict(candidates, return_std=True)
    return mean + kappa * np.maximum(std, 1e-9)


def _lower_confidence_bound(gp, candidates: np.ndarray, best: float, rng=None) -> np.ndarray:
    """Confidence bound for a loss-minimizing GP, negated so higher is better.

    ``best`` and ``rng`` are unused; they keep the shared acquisition signature.
    """
    mean, std = gp.predict(candidates, return_std=True)
    return -(mean - UCB_KAPPA * np.maximum(std, 1e-9))


def _kriging_believer(gp, point: np.ndarray, y_observed: np.ndarray) -> float:
    """Fake loss = the GP's own prediction at the chosen point."""

    return gp.predict(point[None, :])[0]


def _constant_liar_max(gp, point: np.ndarray, y_observed: np.ndarray) -> float:
    """Fake loss = the worst real loss so far (pessimistic: pushes the batch apart)."""

    return float(np.max(y_observed))


def _candidate_pool(X: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Global random points plus local perturbations around the best points."""

    dim = X.shape[1]
    global_points = rng.uniform(size=(2000, dim))
    elite = X[np.argsort(y)[: min(5, len(y))]]
    local = [
        e + rng.normal(scale=scale, size=(400, dim))
        for e in elite
        for scale in (0.03, 0.1, 0.25)
    ]
    return np.clip(np.vstack([global_points, *local]), 0.0, 1.0)


def _propose_batch(
    X: np.ndarray,
    losses: np.ndarray,
    size: int,
    rng: np.random.Generator,
    acquisition: Callable,
    fantasy: Callable,
    active: np.ndarray | None = None,
) -> np.ndarray:
    """Greedy batch; if ``active`` is given, only those parameters are varied
    and the others are pinned to the best observed point's values."""

    y = np.log1p(np.maximum(losses, 0.0))  # Compress large errors; smoother surface.
    gp = _fit_gp(X, y, rng)
    kernel = gp.kernel_
    pool = _candidate_pool(X, y, rng)
    if active is not None:
        pinned = np.tile(X[int(np.argmin(y))], (len(pool), 1))
        pinned[:, active] = pool[:, active]
        pool = pinned
    X_fantasy, y_fantasy = X.copy(), y.copy()
    batch: list[np.ndarray] = []
    for _ in range(size):
        score = acquisition(gp, pool, float(y_fantasy.min()), rng)
        choice = pool[int(np.argmax(score))]
        batch.append(choice)
        # Add a fake observation (kriging believer / constant liar) and
        # recondition the GP with the hyperparameters kept fixed.
        X_fantasy = np.vstack([X_fantasy, choice])
        y_fantasy = np.append(y_fantasy, fantasy(gp, choice, y))
        gp = _fit_gp(X_fantasy, y_fantasy, rng, kernel=kernel)
    return np.array(batch)


def _evaluate(history: _History, units: np.ndarray) -> np.ndarray:
    """Evaluate unit-cube points and return their losses.

    Fewer losses than points come back when the remaining budget cut the
    batch short; they belong to the first points, in order.
    """

    before = len(history.y)
    history.evaluate(units)
    return np.array(history.y[before:])


# --------------------------------------------------------------------------
# Simple Bayesian optimization (baseline for the model-based methods)
# --------------------------------------------------------------------------

SIMPLE_BO_INITIAL_POINTS = 5  # random points before the model is used (at least one batch)
SIMPLE_BO_CANDIDATES = 500  # random candidates scored per round
SIMPLE_BO_LENGTH_SCALE = 1.0  # fixed kernel length scale, in unit-cube distance
SIMPLE_BO_NOISE = 1e-4  # noise variance added to the kernel matrix


def simple_bayesian_optimization(
    history: _History, start: np.ndarray, rng: np.random.Generator
) -> None:
    """Bayesian optimization with a fixed Gaussian process: nothing is fitted.

    A baseline that shows what the more elaborate methods add. Compared with
    ``bayesian_optimization`` it has

    - a squared-exponential kernel with one fixed length scale for all
      parameters and unit variance, instead of fitted hyperparameters;
    - uniform random initial points and candidates, instead of a Latin
      hypercube and local perturbations around the best points.

    Shared with it: the start vector is evaluated first, the target is
    log(1 + WAPE) (here standardized, because the kernel variance is fixed at
    one), the acquisition is Expected Improvement, and the batch is filled
    with the kriging believer.
    """

    dim = len(start)
    n_init = max(SIMPLE_BO_INITIAL_POINTS, history.objective.max_batch_size)
    n_init = max(1, min(n_init, history.remaining() // 2))
    history.evaluate(np.vstack([history.to_unit(start), rng.uniform(size=(n_init - 1, dim))]))

    while history.can_continue():
        X = np.array(history.X)
        y = np.log1p(np.maximum(np.array(history.y), 0.0))
        y = (y - y.mean()) / max(float(y.std()), 1e-9)
        pool = rng.uniform(size=(SIMPLE_BO_CANDIDATES, dim))
        batch = []
        for _ in range(history.next_batch_size()):
            mean, std = _simple_gp_predict(X, y, pool)
            z = (y.min() - mean) / std
            choice = int(np.argmax(std * (z * norm.cdf(z) + norm.pdf(z))))  # Expected Improvement
            batch.append(pool[choice])
            # Kriging believer: the prediction stands in for the missing result.
            X = np.vstack([X, pool[choice]])
            y = np.append(y, mean[choice])
        history.evaluate(np.array(batch))


def _simple_gp_predict(
    X: np.ndarray, y: np.ndarray, candidates: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Posterior mean and standard deviation of a zero-mean, unit-variance GP."""

    def kernel(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        squared_distance = np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=2)
        return np.exp(-0.5 * squared_distance / SIMPLE_BO_LENGTH_SCALE**2)

    L = np.linalg.cholesky(kernel(X, X) + SIMPLE_BO_NOISE * np.eye(len(X)))
    K_star = kernel(candidates, X)
    mean = K_star @ np.linalg.solve(L.T, np.linalg.solve(L, y))
    v = np.linalg.solve(L, K_star.T)
    variance = np.maximum(1.0 - np.sum(v**2, axis=0), 1e-6)
    return mean, np.sqrt(variance)


# --------------------------------------------------------------------------
# Tree-structured Parzen Estimator
# --------------------------------------------------------------------------

_TPE_GOOD_FRACTION = 0.25  # share of the observations modelled as "good"
_TPE_CANDIDATES = 100  # samples drawn from the good density per batch slot


def tpe(history: _History, start: np.ndarray, rng: np.random.Generator) -> None:
    """Tree-structured Parzen Estimator (Bergstra et al., 2011), batched.

    1. Initial design: the start vector plus a Latin hypercube sample (the
       same design, for the same seed, as the Bayesian optimization methods).
    2. Each round: split the observations into the best 25 % ("good") and the
       rest ("bad"), fit one density to each (l and g), draw candidates from l
       and pick the one that maximizes l(x) / g(x), which is equivalent to
       maximizing Expected Improvement under the TPE model.

    The batch is filled with a constant liar: a chosen but not yet evaluated
    point is added to the bad set before the next one is picked, so the batch
    spreads out.
    """

    dim = len(start)
    n_init = max(dim + 1, history.objective.max_batch_size)
    n_init = max(1, min(n_init, history.remaining() // 2))
    design = qmc.LatinHypercube(d=dim, seed=rng)
    history.evaluate(np.vstack([history.to_unit(start), design.random(max(0, n_init - 1))]))

    while history.can_continue():
        X, losses = np.array(history.X), np.array(history.y)
        order = np.argsort(losses, kind="stable")
        n_good = min(max(1, int(np.ceil(_TPE_GOOD_FRACTION * len(order)))), 25)
        good, bad = X[order[:n_good]], X[order[n_good:]]
        good_density = _parzen_fit(good)
        batch = []
        for _ in range(history.next_batch_size()):
            candidates = _parzen_sample(good_density, _TPE_CANDIDATES, rng)
            score = _parzen_log_pdf(good_density, candidates)
            if len(bad):
                score = score - _parzen_log_pdf(_parzen_fit(bad), candidates)
            choice = candidates[int(np.argmax(score))]
            batch.append(choice)
            bad = np.vstack([bad, choice])  # constant liar: assume it turns out bad
        history.evaluate(np.array(batch))


def _parzen_fit(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """One independent 1-D Parzen estimator per parameter on [0, 1].

    Each estimator is a mixture of a uniform prior and one truncated Gaussian
    per observation, all with equal weight. A Gaussian's width is the larger
    distance to its neighbours (the bounds count as neighbours), clipped to
    [1 / min(100, n + 1), 1] as in Hyperopt: wide while data is scarce.
    Returns (means, widths), both of shape (n, dim).
    """

    count, dim = points.shape
    order = np.argsort(points, axis=0)
    ordered = np.take_along_axis(points, order, axis=0)
    padded = np.vstack([np.zeros((1, dim)), ordered, np.ones((1, dim))])
    gaps = np.maximum(padded[1:-1] - padded[:-2], padded[2:] - padded[1:-1])
    sigma = np.empty_like(points)
    np.put_along_axis(sigma, order, np.clip(gaps, 1.0 / min(100, count + 1), 1.0), axis=0)
    return points, sigma


def _parzen_log_pdf(density: tuple[np.ndarray, np.ndarray], x: np.ndarray) -> np.ndarray:
    """Log density of each row of ``x`` (parameters are independent, so logs add)."""

    mu, sigma = density
    inside = norm.cdf((1.0 - mu) / sigma) - norm.cdf(-mu / sigma)  # mass within [0, 1]
    z = (x[:, None, :] - mu[None]) / sigma[None]
    kernels = norm.logpdf(z) - np.log(sigma * inside)[None]
    uniform = np.zeros((len(x), 1, x.shape[1]))
    mixture = logsumexp(np.concatenate([kernels, uniform], axis=1), axis=1) - np.log(len(mu) + 1)
    return mixture.sum(axis=1)


def _parzen_sample(
    density: tuple[np.ndarray, np.ndarray], count: int, rng: np.random.Generator
) -> np.ndarray:
    mu, sigma = density
    dim = mu.shape[1]
    component = rng.integers(0, len(mu) + 1, size=(count, dim))  # last index = uniform prior
    uniform = rng.uniform(size=(count, dim))
    columns = np.arange(dim)[None, :]
    kernel = np.minimum(component, len(mu) - 1)
    m, s = mu[kernel, columns], sigma[kernel, columns]
    # Inverse-CDF sample of a Gaussian truncated to [0, 1].
    low, high = norm.cdf(-m / s), norm.cdf((1.0 - m) / s)
    truncated = m + s * norm.ppf(low + rng.uniform(size=(count, dim)) * (high - low))
    return np.clip(np.where(component == len(mu), uniform, truncated), 0.0, 1.0)


# --------------------------------------------------------------------------
# CMA-ES
# --------------------------------------------------------------------------

_CMA_INITIAL_STEP = 0.3  # initial standard deviation, as a share of each parameter range


def cma_es(history: _History, start: np.ndarray, rng: np.random.Generator) -> None:
    """(mu/mu_w, lambda)-CMA-ES (Hansen, 2016 tutorial), one generation per batch.

    The search distribution starts at the start vector, which is itself
    evaluated in the first generation. The population size is the batch size
    (at least the CMA-ES default 4 + 3 ln d), reduced if needed so that the
    budget allows about four generations. Points outside the bounds are
    clipped onto them before evaluation and the update uses the clipped points.
    """

    dim = len(start)
    lam = max(4 + int(3 * np.log(dim)), history.objective.max_batch_size)
    lam = max(4, min(lam, history.remaining() // 4))
    mu = lam // 2
    weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    weights /= weights.sum()
    mu_eff = 1.0 / np.sum(weights**2)

    # Standard learning rates.
    c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
    d_sigma = 1 + 2 * max(0.0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
    c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
    c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
    c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
    chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))  # E||N(0, I)||

    mean = history.to_unit(start)
    sigma = _CMA_INITIAL_STEP
    C = np.eye(dim)
    p_sigma, p_c = np.zeros(dim), np.zeros(dim)
    generation = 0

    while history.can_continue():
        eigenvalues, B = np.linalg.eigh(C)
        D = np.sqrt(np.maximum(eigenvalues, 1e-20))
        z = rng.standard_normal((lam, dim))
        if generation == 0:
            z[0] = 0.0  # the start vector itself
        x = np.clip(mean + sigma * (z * D) @ B.T, 0.0, 1.0)
        losses = _evaluate(history, x)
        if len(losses) < lam:
            continue  # The budget ended inside this generation.

        elite = np.argsort(losses, kind="stable")[:mu]
        y = (x[elite] - mean) / sigma  # steps actually taken (after clipping)
        y_mean = weights @ y
        mean = mean + sigma * y_mean

        generation += 1
        inv_sqrt_C = (B / D) @ B.T
        p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (
            inv_sqrt_C @ y_mean
        )
        norm_p_sigma = np.linalg.norm(p_sigma)
        h_sigma = float(
            norm_p_sigma / np.sqrt(1 - (1 - c_sigma) ** (2 * generation))
            < (1.4 + 2 / (dim + 1)) * chi_n
        )
        p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_mean
        C = (
            (1 - c_1 - c_mu) * C
            + c_1 * (np.outer(p_c, p_c) + (1 - h_sigma) * c_c * (2 - c_c) * C)
            + c_mu * (y.T * weights) @ y
        )
        sigma *= np.exp(c_sigma / d_sigma * (norm_p_sigma / chi_n - 1))
        sigma = float(np.clip(sigma, 1e-3, 1.0))


# --------------------------------------------------------------------------
# Hyperband
# --------------------------------------------------------------------------

_HYPERBAND_ETA = 3  # keep the best 1/eta of the arms at every rung
_HYPERBAND_INITIAL_STEP = 0.15  # local-search step, as a share of each parameter range


def hyperband(history: _History, start: np.ndarray, rng: np.random.Generator) -> None:
    """Hyperband (Li et al., 2018) with evaluations per arm as the resource.

    Hyperband normally saves budget by judging configurations at a cheap, low
    fidelity (few epochs, less data). The organizer objective has no such
    knob: every evaluation is a full simulation of all years. The resource is
    therefore the number of evaluations spent on an arm. An arm is a random
    starting configuration that is improved by local search (Gaussian steps
    around its best point so far), and its loss is the best loss it has found.

    As in Hyperband, bracket s starts ceil((s_max + 1) / (s + 1) * eta^s) arms
    with eta^(s_max - s) evaluations each, then repeatedly keeps the best
    1/eta of them and gives the survivors eta times the resource (successive
    halving; survivors continue their search, so resources are cumulative).
    Brackets run from the most exploratory (many arms, one evaluation each)
    to the least, and the budget simply ends the last one early.

    s_max is the largest value for which the most exploratory bracket costs
    at most half the budget. Budget 48 gives s_max = 2: brackets of 9, 5, and
    3 arms with at most 9 evaluations per arm. The very first arm is the
    start vector.
    """

    dim = len(start)
    eta = _HYPERBAND_ETA
    budget = history.remaining()
    s_max = 0
    while _hyperband_cost(_hyperband_rungs(s_max + 1, s_max + 1, eta)) <= budget / 2:
        s_max += 1

    first_arm = history.to_unit(start)
    while history.can_continue():  # Repeat the brackets if budget is left over.
        for s in range(s_max, -1, -1):
            rungs = _hyperband_rungs(s, s_max, eta)
            arms = [
                {"x": rng.uniform(size=dim), "loss": np.inf, "step": _HYPERBAND_INITIAL_STEP, "used": 0}
                for _ in range(rungs[0][0])
            ]
            if first_arm is not None:
                arms[0]["x"], first_arm = first_arm, None
            for keep, resource in rungs:
                arms = sorted(arms, key=lambda arm: arm["loss"])[:keep]
                _hyperband_advance(history, arms, resource, rng)
                if not history.can_continue():
                    return


def _hyperband_rungs(s: int, s_max: int, eta: int) -> list[tuple[int, int]]:
    """(number of arms, cumulative evaluations per arm) for each rung of bracket s."""

    arms = int(np.ceil((s_max + 1) / (s + 1) * eta**s))
    return [(max(1, arms // eta**i), eta ** (s_max - s + i)) for i in range(s + 1)]


def _hyperband_cost(rungs: list[tuple[int, int]]) -> int:
    cost, previous = 0, 0
    for arms, resource in rungs:
        cost += arms * (resource - previous)
        previous = resource
    return cost


def _hyperband_advance(
    history: _History, arms: list[dict], resource: int, rng: np.random.Generator
) -> None:
    """Run every arm until it has used ``resource`` evaluations.

    A new arm first evaluates its own configuration. After that it proposes
    Gaussian steps around its best point. The arms share each batch equally,
    so all workers stay busy; with one worker this is a (1+1) search per arm.
    The step grows after a batch that improved the arm and shrinks otherwise.
    """

    dim = len(arms[0]["x"])
    while history.can_continue():
        active = [arm for arm in arms if arm["used"] < resource]
        if not active:
            return
        share = max(1, history.next_batch_size() // len(active))
        proposals = []
        for arm in active:
            if arm["used"] == 0:
                proposals.append([arm["x"]])
            else:
                count = min(share, resource - arm["used"])
                steps = arm["step"] * rng.standard_normal((count, dim))
                proposals.append(list(np.clip(arm["x"] + steps, 0.0, 1.0)))
        searching = [arm for arm in active if arm["used"] > 0]
        # Interleave the arms so a budget cut is shared between them.
        queue = [
            (arm, points[k])
            for k in range(max(map(len, proposals)))
            for arm, points in zip(active, proposals)
            if k < len(points)
        ]
        losses = _evaluate(history, np.array([point for _, point in queue]))
        improved = set()
        for (arm, point), loss in zip(queue, losses):
            if loss < arm["loss"]:
                arm["x"], arm["loss"] = point, float(loss)
                improved.add(id(arm))
            arm["used"] += 1
        for arm in searching:
            arm["step"] *= 1.2 if id(arm) in improved else 0.6

LASSO_MAX_ACTIVE_FRACTION = 0.6
LASSO_EXPLAINED_FRACTION = 0.9
LASSO_ALPHA = 0.03

def lasso_bayesian_optimisation(
    history: _History, start: np.ndarray, rng: np.random.Generator
) -> None:
    """LassoBO-style optimization using this file's GP and acquisition code.

    The Lasso ranks variables from the observed reward. The GP then searches
    the UCB acquisition function while varying only the selected active
    dimensions; inactive dimensions are kept at promising background values.
    """
    dim = len(start)
    n_init = max(dim + 1, history.objective.max_batch_size)
    n_init = max(1, min(n_init, int(history.remaining() * 0.35)))
    design = qmc.LatinHypercube(d=dim, seed=rng)
    initial = np.vstack([history.to_unit(start), design.random(max(0, n_init - 1))])
    history.evaluate(initial)

    while history.can_continue():
        X = np.asarray(history.X)
        losses = np.asarray(history.y)
        reward = -np.log1p(np.maximum(losses, 0.0))
        coefficients = _lasso_coefficients(X, reward)
        active = _select_active_dimensions(
            coefficients,
            explained_fraction=LASSO_EXPLAINED_FRACTION,
            max_active_fraction=LASSO_MAX_ACTIVE_FRACTION,
        )
        batch = _propose_lasso_batch(
            X,
            losses,
            history.next_batch_size(),
            active,
            rng,
        )
        history.evaluate(batch)


def _lasso_coefficients(
    X: np.ndarray,
    target: np.ndarray,
    alpha: float = LASSO_ALPHA,
    iterations: int = 250,
) -> np.ndarray:
    """Fit a standardized linear Lasso using coordinate descent."""
    means = X.mean(axis=0)
    scales = X.std(axis=0)
    scales = np.where(scales > 1e-12, scales, 1.0)
    Z = (X - means) / scales
    residual = target - target.mean()
    coefficients = np.zeros(X.shape[1])
    column_norms = np.maximum(np.sum(Z * Z, axis=0), 1e-12)
    for _ in range(iterations):
        maximum_change = 0.0
        for column in range(X.shape[1]):
            old = coefficients[column]
            partial = np.dot(Z[:, column], residual + Z[:, column] * old)
            new = np.sign(partial) * max(abs(partial) - alpha * len(X), 0.0)
            new /= column_norms[column]
            residual -= Z[:, column] * (new - old)
            coefficients[column] = new
            maximum_change = max(maximum_change, abs(new - old))
        if maximum_change < 1e-7:
            break
    return coefficients


def _select_active_dimensions(
    coefficients: np.ndarray,
    explained_fraction: float,
    max_active_fraction: float,
) -> np.ndarray:
    """Select variables explaining most of the absolute Lasso signal."""
    scores = np.abs(coefficients)
    if not np.any(scores > 1e-12):
        return np.array([int(np.argmax(scores))])
    order = np.argsort(-scores)
    target = explained_fraction * float(scores.sum())
    max_active = max(1, int(np.ceil(max_active_fraction * len(scores))))
    selected, explained = [], 0.0
    for index in order:
        selected.append(int(index))
        explained += scores[index]
        if explained >= target or len(selected) >= max_active:
            break
    return np.array(sorted(selected), dtype=int)


def _lasso_candidate_pool(
    X: np.ndarray,
    losses: np.ndarray,
    active: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate global and local candidates with inactive values fixed."""
    dim = X.shape[1]
    inactive = np.setdiff1d(np.arange(dim), active)
    elite = X[np.argsort(losses)[: min(3, len(losses))]]
    candidates = []
    for point in elite:
        local = np.tile(point, (400, 1))
        local[:, active] = np.clip(
            local[:, active] + rng.normal(0.0, 0.15, (400, len(active))), 0.0, 1.0
        )
        candidates.append(local)
    global_points = rng.uniform(size=(2000, dim))
    global_points[:, inactive] = elite[0, inactive]
    candidates.append(global_points)
    lhs = qmc.LatinHypercube(d=dim, seed=rng).random(1000)
    lhs[:, inactive] = elite[0, inactive]
    candidates.append(lhs)
    return np.clip(np.vstack(candidates), 0.0, 1.0)


def _propose_lasso_batch(
    X: np.ndarray,
    losses: np.ndarray,
    size: int,
    active: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Fit the existing GP and select a batch using UCB in active dimensions."""
    reward = -np.log1p(np.maximum(losses, 0.0))
    gp = _fit_gp(X, reward, rng)
    kernel = gp.kernel_
    X_fantasy, reward_fantasy = X.copy(), reward.copy()
    batch = []
    for _ in range(size):
        pool = _lasso_candidate_pool(X_fantasy, -reward_fantasy, active, rng)
        scores = _upper_confidence_bound(gp, pool)
        choice = pool[int(np.argmax(scores))]
        batch.append(choice)
        X_fantasy = np.vstack([X_fantasy, choice])
        reward_fantasy = np.append(reward_fantasy, gp.predict(choice[None, :])[0])
        gp = _fit_gp(X_fantasy, reward_fantasy, rng, kernel=kernel)
    return np.asarray(batch)


# --------------------------------------------------------------------------
# Bayesian optimization focused by ARD length scales
# --------------------------------------------------------------------------

ARD_INITIAL_FRACTION = 0.75  # set by experiments.py
ARD_FOCUS_END_FRACTION = 0.8  # after this share of the budget every parameter is searched again
ARD_TOP_FRACTION = 0.5  # share of the parameters searched during the focus stage
ARD_RKN = 1  # set by experiments.py


def ard_bayesian_optimisation(
    history: _History, start: np.ndarray, rng: np.random.Generator
) -> None:
    """Bayesian optimization in three stages, focused by ARD length scales.

    Model, acquisition, candidates, and batch filling are those of
    ``bayesian_optimization`` (Expected Improvement, kriging believer).

    1. Initial stage: the start vector plus a Latin hypercube sample.
    2. Focus stage (until ARD_FOCUS_END_FRACTION of the budget is used): only
       the ARD_TOP_FRACTION of the parameters with the shortest fitted GP
       length scales are varied; the others are pinned to the best point's
       values. The ranking is redone every ARD_RKN of the focus stage.
    3. Final stage: every parameter is searched again.
    """

    if not 0 <= ARD_RKN <= 1:
        raise ValueError("ARD_RKN must be between 0 and 1")

    dim = len(start)
    n_init = max(dim + 1, history.objective.max_batch_size)
    n_init = max(1, min(n_init, int(history.remaining() * ARD_INITIAL_FRACTION)))
    design = qmc.LatinHypercube(d=dim, seed=rng)
    history.evaluate(np.vstack([history.to_unit(start), design.random(max(0, n_init - 1))]))

    active = None
    next_ranking = 0.0  # focus-stage progress at which the ranking is redone

    while history.can_continue():
        progress = history.submitted / max(1, history.budget)
        if progress < ARD_FOCUS_END_FRACTION:
            focus_span = max(ARD_FOCUS_END_FRACTION - ARD_INITIAL_FRACTION, 1e-12)
            focus_progress = (progress - ARD_INITIAL_FRACTION) / focus_span
            if active is None or ARD_RKN == 0 or focus_progress >= next_ranking:
                y = np.log1p(np.maximum(np.array(history.y), 0.0))
                active = _ard_active_dimensions(_fit_gp(np.array(history.X), y, rng))
                next_ranking = focus_progress + ARD_RKN
                print(f"ard: searching parameters {active.tolist()} of {dim}")
        else:
            active = None  # final stage: all parameters

        batch = _propose_batch(
            np.array(history.X),
            np.array(history.y),
            history.next_batch_size(),
            rng,
            _expected_improvement,
            _kriging_believer,
            active=active,
        )
        history.evaluate(batch)


def _ard_active_dimensions(gp) -> np.ndarray:
    """The ARD_TOP_FRACTION of the parameters with the shortest length scales."""

    # ConstantKernel * Matern + WhiteKernel: the Matern is kernel_.k1.k2.
    length_scales = np.asarray(gp.kernel_.k1.k2.length_scale, dtype=float).reshape(-1)
    count = max(1, int(np.ceil(len(length_scales) * ARD_TOP_FRACTION)))
    return np.sort(np.argsort(length_scales, kind="stable")[:count])


# --------------------------------------------------------------------------
# CalibLab: sensitivity analysis, GP metamodel, Bayesian calibration (MCMC)
# --------------------------------------------------------------------------
#
# Re-implementation of the approach of CalibLab (TU Berlin, MIT license):
#   https://github.com/TUB-DVG/CalibLab
#   Fülep, K., Kalathiparambil Kennedy, M. L., Brandt, S., & Streblow, R. (2024).
#   An Open-Source Framework: Bayesian Building Energy Model Calibration with
#   Uncertainty Quantification. uSIM2024, Edinburgh.
#   Fülep, K., Chen, S., Brandt, S., & Streblow, R. (2024). Advancing Building
#   Energy Modeling: An Open-Source Bayesian Calibration Framework for
#   Non-Residential Buildings. SimBuild 2024, Denver.
# No CalibLab code is copied; SALib and PyMC3, which it uses, are replaced by
# the NumPy/SciPy code below.

CALIBLAB_NUM_PARAMETERS = 5  # parameters kept after the sensitivity analysis (CalibLab: num_bc_param)
CALIBLAB_SENSITIVITY_FRACTION = 0.35  # at most this share of the budget goes to the sensitivity runs
CALIBLAB_GP_FRACTION = 0.25  # share of the budget for the metamodel's Latin hypercube runs
CALIBLAB_GP_TEST_SIZE = 0.25  # share of the metamodel runs held out to pick the kernel (gp_test_size)
CALIBLAB_SOBOL_LOG2_SAMPLES = 8  # the Sobol indices use 2**this base samples
CALIBLAB_ERROR_SCALE = 0.1  # prior mean of the observation error, as a share of the mean output
CALIBLAB_CHAINS = 4  # MCMC chains
CALIBLAB_DRAWS = 250  # kept MCMC steps per chain
CALIBLAB_TUNE = 500  # discarded MCMC steps per chain, during which the step size adapts
CALIBLAB_RHAT_THRESHOLD = 1.1  # sample again (twice the draws) while any R-hat is above this
CALIBLAB_MAX_RESAMPLES = 2  # ... but at most this many times


def caliblab(history: _History, start: np.ndarray, rng: np.random.Generator) -> None:
    """The CalibLab pipeline (see the citation above), adapted to this challenge.

    CalibLab calibrates in three steps, each kept here:

    1. Sensitivity analysis: Sobol total-order indices rank the parameters and
       the CALIBLAB_NUM_PARAMETERS most influential ones are calibrated.
    2. Metamodel: the simulator is run on a Latin hypercube over those
       parameters (the others stay fixed) and a Gaussian process is trained on
       the runs. Three kernels are tried and the one with the lowest RMSE on
       held-out runs is used.
    3. Bayesian calibration: MCMC samples the posterior of the parameters,
       with the metamodel in place of the simulator, a model-discrepancy term
       N(0, sigma_gp), an Exponential prior on the observation error, and
       sampling repeated with twice the draws until R-hat is small enough.

    What had to change, and why:

    - CalibLab runs thousands of simulations for the Sobol indices. With a
      budget of a few dozen, the indices are computed on a GP fitted to the
      start vector plus a Latin hypercube over all parameters.
    - The objective returns only the WAPE, not the simulated series, so the
      metamodel predicts log(1 + WAPE) and the "observation" is a perfect
      fit, i.e. 0. The priors are uniform over the organizer bounds.
    - The fixed parameters are pinned to the best point of step 1.
    - CalibLab ends with the posterior; the challenge needs one evaluated
      parameter set. The remaining budget is therefore spent in rounds:
      simulate a batch of posterior draws, add them to the metamodel's
      training runs, and repeat steps 2 (training) and 3.
    - The sampler is random-walk Metropolis instead of PyMC3's NUTS.
    """

    dim = len(start)
    n_init = max(dim + 1, history.objective.max_batch_size)
    n_init = max(1, min(n_init, int(history.remaining() * CALIBLAB_SENSITIVITY_FRACTION)))
    design = qmc.LatinHypercube(d=dim, seed=rng)
    history.evaluate(np.vstack([history.to_unit(start), design.random(max(0, n_init - 1))]))
    if not history.can_continue():
        return

    # 1. Sensitivity analysis on a metamodel of all parameters.
    X = np.array(history.X)
    y = np.log1p(np.maximum(np.array(history.y), 0.0))
    total_effects = _sobol_total_indices(_fit_gp(X, y, rng), dim, rng)
    selected = np.sort(np.argsort(-total_effects, kind="stable")[:CALIBLAB_NUM_PARAMETERS])
    print(f"caliblab: calibrating parameters {selected.tolist()} of {dim}")

    base = X[int(np.argmin(y))]  # the fixed parameters keep these values

    def embed(points: np.ndarray) -> np.ndarray:
        full = np.tile(base, (len(points), 1))
        full[:, selected] = points
        return full

    # 2. Metamodel runs: a Latin hypercube over the selected parameters.
    count = max(1, min(int(history.budget * CALIBLAB_GP_FRACTION), history.next_batch_size()))
    samples = qmc.LatinHypercube(d=len(selected), seed=rng).random(count)
    losses = _evaluate(history, embed(samples))
    Z = np.vstack([base[selected], samples[: len(losses)]])
    z_losses = np.concatenate([[float(np.min(history.y[: len(X)]))], losses])

    # 2. (training) and 3., repeated while budget is left.
    while history.can_continue():
        target = np.log1p(np.maximum(z_losses, 0.0))
        gp, sigma_gp = _caliblab_metamodel(Z, target, rng)
        draws = _caliblab_posterior(gp, sigma_gp, Z, target, rng)
        batch = _caliblab_candidates(gp, draws, Z, history.next_batch_size(), rng)
        losses = _evaluate(history, embed(batch))
        Z = np.vstack([Z, batch[: len(losses)]])
        z_losses = np.concatenate([z_losses, losses])


def _sobol_total_indices(gp, dim: int, rng: np.random.Generator) -> np.ndarray:
    """Sobol total-order index of every parameter for the GP mean (Jansen's estimator)."""

    points = qmc.Sobol(d=2 * dim, seed=rng).random_base2(CALIBLAB_SOBOL_LOG2_SAMPLES)
    A, B = points[:, :dim], points[:, dim:]
    f_A = gp.predict(A)
    variance = max(float(np.var(np.concatenate([f_A, gp.predict(B)]))), 1e-12)
    indices = np.empty(dim)
    for i in range(dim):
        AB = A.copy()
        AB[:, i] = B[:, i]  # only parameter i differs from A
        indices[i] = 0.5 * np.mean((f_A - gp.predict(AB)) ** 2) / variance
    return indices


def _caliblab_metamodel(Z: np.ndarray, target: np.ndarray, rng: np.random.Generator):
    """Train the GP metamodel; return it with sigma_gp, its mean held-out std.

    The kernels are CalibLab's three candidates. Each is trained on the
    training split and scored by its RMSE on the held-out split; the winner is
    then trained again on all runs.
    """

    def kernels():
        scale = (1e-3, 1e3)
        return [
            ConstantKernel(1.0, scale) * RationalQuadratic(1.0, 1.0, (1e-5, 1e5), (1e-5, 1e5)),
            ConstantKernel(1.0, scale) * RBF(1.0, (1e-5, 1e5))
            + ConstantKernel(1.0, scale) * ExpSineSquared(1.0, 1.0, (1e-5, 1e5), (1e-5, 1e5)),
            ConstantKernel(1.0, scale) * RBF(1.0, (1e-5, 1e5)),
        ]

    def fit(kernel, rows):
        gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=max(1, int(0.1 * len(rows))),
            random_state=int(rng.integers(2**31 - 1)),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gp.fit(Z[rows], target[rows])
        return gp

    order = rng.permutation(len(Z))
    n_test = max(1, int(round(CALIBLAB_GP_TEST_SIZE * len(Z))))
    test, train = order[:n_test], order[n_test:]
    if len(train) < 2:  # too few runs to hold any out
        test = train = order

    best = None  # (rmse, kernel index, sigma_gp)
    for index, kernel in enumerate(kernels()):
        try:
            mean, std = fit(kernel, train).predict(Z[test], return_std=True)
        except (np.linalg.LinAlgError, ValueError):
            continue
        rmse = float(np.sqrt(np.mean((mean - target[test]) ** 2)))
        if np.isfinite(rmse) and (best is None or rmse < best[0]):
            best = (rmse, index, float(np.mean(std)))
    if best is None:
        best = (np.inf, 2, float(np.std(target)))  # plain RBF
    return fit(kernels()[best[1]], order), max(best[2], 1e-6)


def _caliblab_posterior(
    gp, sigma_gp: float, Z: np.ndarray, target: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Posterior draws of the calibrated parameters, shape (draws, parameters).

    Model, as in CalibLab: uniform priors on the parameters, observation
    error ~ Exponential with mean CALIBLAB_ERROR_SCALE * mean output, and the
    observation (here a perfect fit, 0) ~ Normal(metamodel + discrepancy,
    error) with discrepancy ~ Normal(0, sigma_gp); the discrepancy is
    integrated out. Random-walk Metropolis over the parameters and log(error).
    """

    chains, parameters = CALIBLAB_CHAINS, Z.shape[1]
    rate = 1.0 / (CALIBLAB_ERROR_SCALE * max(float(np.mean(target)), 1e-9))

    def log_posterior(theta: np.ndarray, log_error: np.ndarray) -> np.ndarray:
        inside = np.all((theta >= 0.0) & (theta <= 1.0), axis=1)
        mean = gp.predict(np.clip(theta, 0.0, 1.0))
        variance = sigma_gp**2 + np.exp(2.0 * log_error)
        value = -0.5 * mean**2 / variance - 0.5 * np.log(variance)  # likelihood of 0
        value += log_error - rate * np.exp(log_error)  # Exponential prior, in log(error)
        return np.where(inside, value, -np.inf)

    # The chains start at the best runs so far.
    theta = Z[np.resize(np.argsort(target, kind="stable"), chains)].copy()
    log_error = np.full(chains, -np.log(rate))
    current = log_posterior(theta, log_error)
    step = np.full(chains, 0.1)

    def sample(steps: int, adapt: bool) -> np.ndarray:
        nonlocal theta, log_error, current, step
        trace = np.empty((steps, chains, parameters + 1))
        for i in range(steps):
            new_theta = theta + step[:, None] * rng.standard_normal((chains, parameters))
            new_log_error = log_error + 2.0 * step * rng.standard_normal(chains)
            proposed = log_posterior(new_theta, new_log_error)
            accept = np.log(rng.uniform(size=chains)) < proposed - current
            theta = np.where(accept[:, None], new_theta, theta)
            log_error = np.where(accept, new_log_error, log_error)
            current = np.where(accept, proposed, current)
            if adapt:  # settles at about 24 % accepted steps
                step = np.clip(step * np.where(accept, 1.1, 0.97), 1e-4, 1.0)
            trace[i, :, :parameters], trace[i, :, parameters] = theta, log_error
        return trace

    draws, tune = CALIBLAB_DRAWS, CALIBLAB_TUNE
    for attempt in range(CALIBLAB_MAX_RESAMPLES + 1):
        sample(tune, adapt=True)
        trace = sample(draws, adapt=False)
        r_hat = _r_hat(trace)
        if np.all(r_hat <= CALIBLAB_RHAT_THRESHOLD):
            break
        draws, tune = 2 * draws, 4 * draws  # CalibLab: twice the draws, tune = twice the new draws
    print(f"caliblab: {len(trace)} draws x {chains} chains, largest R-hat {float(np.max(r_hat)):.2f}")
    return trace[:, :, :parameters].reshape(-1, parameters)


def _r_hat(trace: np.ndarray) -> np.ndarray:
    """Gelman-Rubin R-hat per variable for a trace of shape (steps, chains, variables)."""

    steps = len(trace)
    within = trace.var(axis=0, ddof=1).mean(axis=0)
    between = trace.mean(axis=0).var(axis=0, ddof=1)
    ratio = between / np.maximum(within, 1e-12)
    return np.minimum(np.sqrt((steps - 1) / steps + ratio), 99.0)  # capped: stuck chains


def _caliblab_candidates(
    gp, draws: np.ndarray, Z: np.ndarray, size: int, rng: np.random.Generator
) -> np.ndarray:
    """Posterior draws to simulate next: the best predicted one, then random ones.

    Draws closer than 0.001 (in any unit range) to a run or to an earlier
    choice are skipped; uniform points fill the batch if too few are left.
    """

    draws = np.unique(draws, axis=0)
    rest = rng.permutation(len(draws))
    first = int(np.argmin(gp.predict(draws)))
    taken = Z
    batch = []
    for index in [first, *rest[rest != first]]:
        if len(batch) == size:
            break
        if np.min(np.max(np.abs(taken - draws[index]), axis=1)) > 1e-3:
            batch.append(draws[index])
            taken = np.vstack([taken, draws[index]])
    fill = rng.uniform(size=(size - len(batch), Z.shape[1]))
    return np.vstack([*batch, fill]) if batch else fill


# --------------------------------------------------------------------------
# Bayesian optimization with a runtime parameter screening
# --------------------------------------------------------------------------

SCREENING_LOW_LEVEL = 0.25  # low level of the screening design (high = 1 - low), unit range
SCREENING_THRESHOLD = 0.1  # a parameter is important if its effect >= this share of the largest
SCREENING_FULL_SPACE_SHARE = 0.2  # share of candidates that still vary every parameter


def bayesian_optimization_screened(
    history: _History, start: np.ndarray, rng: np.random.Generator
) -> None:
    """Bayesian optimization that screens the parameters first, then focuses.

    Meant to run with every parameter active: it finds out per building which
    parameters matter instead of relying on a hand-picked subset.

    1. Initial design: the start vector plus a two-level screening design
       (``_screening_design``; 16 runs for 9 to 15 parameters). It replaces
       the Latin hypercube of the other Bayesian optimization methods.
    2. First round: a parameter's effect is its linear main effect on
       log(1 + WAPE), which the screening design estimates from all runs.
    3. Later rounds: the effect is re-estimated from the GP as the range of
       the predicted log-loss when only that parameter is moved across its
       range through the best point. A parameter can therefore come back.
    4. Each round, parameters whose effect is at least SCREENING_THRESHOLD of
       the largest effect are searched. In most candidates the others are
       pinned to the best point's values (not to the defaults); a
       SCREENING_FULL_SPACE_SHARE of the candidates still varies everything,
       so the selection is soft.

    Model, acquisition, and batch filling are those of
    ``bayesian_optimization`` (Expected Improvement, kriging believer).
    If the screening design would cost more than half the budget, the Latin
    hypercube design is used and every round uses the GP-based effects.
    """

    dim = len(start)
    design = _screening_design(dim)
    screened = 1 + len(design) <= history.remaining() // 2
    if not screened:
        n_init = max(dim + 1, history.objective.max_batch_size)
        n_init = max(1, min(n_init, history.remaining() // 2))
        design = qmc.LatinHypercube(d=dim, seed=rng).random(max(0, n_init - 1))
    history.evaluate(np.vstack([history.to_unit(start), design]))

    while history.can_continue():
        X = np.array(history.X)
        y = np.log1p(np.maximum(np.array(history.y), 0.0))
        gp = _fit_gp(X, y, rng)
        best = X[int(np.argmin(y))]
        effects = _linear_effects(X, y) if screened else _profile_effects(gp, best)
        screened = False  # Only the first round uses the design's main effects.
        active = effects >= SCREENING_THRESHOLD * effects.max()
        print(f"screening: searching parameters {np.flatnonzero(active).tolist()} of {dim}")

        # Candidates of plain Bayesian optimization, most of them moved into
        # the subspace of the important parameters through the best point.
        pool = _candidate_pool(X, y, rng)
        pinned = np.where(active[None, :], pool, best[None, :])
        full_space = rng.uniform(size=len(pool)) < SCREENING_FULL_SPACE_SHARE
        pool = np.where(full_space[:, None], pool, pinned)

        kernel = gp.kernel_
        X_fantasy, y_fantasy = X.copy(), y.copy()
        batch = []
        for _ in range(history.next_batch_size()):
            score = _expected_improvement(gp, pool, float(y_fantasy.min()))
            choice = pool[int(np.argmax(score))]
            batch.append(choice)
            X_fantasy = np.vstack([X_fantasy, choice])
            y_fantasy = np.append(y_fantasy, _kriging_believer(gp, choice, y))
            gp = _fit_gp(X_fantasy, y_fantasy, rng, kernel=kernel)
        history.evaluate(np.array(batch))


def _screening_design(dim: int) -> np.ndarray:
    """Two-level orthogonal design in the unit cube, one column per parameter.

    The columns come from a Hadamard matrix (without its constant column), so
    every parameter is at its low level in half of the runs and at its high
    level in the other half, and all columns are uncorrelated. The number of
    runs is the smallest power of two above ``dim``. Main effects are mixed
    with two-parameter interactions (resolution III), which is the price for
    so few runs.
    """

    runs = 2 ** int(np.ceil(np.log2(dim + 1)))
    signs = hadamard(runs)[:, 1 : dim + 1]
    return np.where(signs > 0, 1.0 - SCREENING_LOW_LEVEL, SCREENING_LOW_LEVEL)


def _linear_effects(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """|change of y| per parameter across its full range, from a linear fit."""

    design = np.column_stack([np.ones(len(X)), X])
    coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
    return np.abs(coefficients[1:])


def _profile_effects(gp, best: np.ndarray, steps: int = 11) -> np.ndarray:
    """Range of the GP mean when one parameter at a time crosses its range.

    All other parameters stay at ``best``, so this measures what each
    parameter can still change around the best point found so far.
    """

    dim = len(best)
    points = np.tile(best, (dim, steps, 1))
    points[np.arange(dim), :, np.arange(dim)] = np.linspace(0.0, 1.0, steps)
    mean = gp.predict(points.reshape(dim * steps, dim)).reshape(dim, steps)
    return mean.max(axis=1) - mean.min(axis=1)


_ALGORITHMS = {
    "random_search": random_search,
    "bayesian_optimization": bayesian_optimization,
    "bayesian_optimization_cl": bayesian_optimization_cl,
    "bayesian_optimization_mes": bayesian_optimization_mes,
    "bayesian_optimization_ucb": bayesian_optimization_ucb,
    "simple_bayesian_optimization": simple_bayesian_optimization,
    "tpe": tpe,
    "cma_es": cma_es,
    "hyperband": hyperband,
    "lasso_bayesian_optimisation": lasso_bayesian_optimisation,
    "ard_bayesian_optimisation": ard_bayesian_optimisation,
    "caliblab": caliblab,
    "bayesian_optimization_screened": bayesian_optimization_screened,
}


def calibrate(
    objective: Callable[[Sequence[float]], float],
    parameter_space,
    budget: int,
    seed: int,
) -> dict[str, float]:
    """Return the best parameters found by the method named in ALGORITHM_NAME.

    Dimensions, bounds, defaults, budget, and seed are read at run time from
    the organizer-supplied arguments; nothing is hard-coded.
    """

    if budget < 1:
        raise ValueError("budget must be at least one")
    if ALGORITHM_NAME not in _ALGORITHMS:
        raise ValueError(
            f"Unknown ALGORITHM_NAME {ALGORITHM_NAME!r}; choose from {sorted(_ALGORITHMS)}"
        )

    lower = np.array([float(p.lower) for p in parameter_space])
    upper = np.array([float(p.upper) for p in parameter_space])
    start = np.array([float(p.default) for p in parameter_space])

    if lower.size == 0:
        # No active parameters: one evaluation of the organizer defaults.
        objective(())
        return {}

    history = _History(objective, lower, upper, budget)
    _ALGORITHMS[ALGORITHM_NAME](history, start, np.random.default_rng(seed))
    best_vector, best_loss = history.best()

    print(
        f"{ALGORITHM_NAME}: best WAPE {best_loss:.2f}% after "
        f"{history.batches} batches / {len(history.y)} candidates"
    )

    # Clip defensively so the returned values are always in bounds.
    best_vector = np.clip(best_vector, lower, upper)
    return {
        p.name: float(value)
        for p, value in zip(parameter_space, best_vector, strict=True)
    }
