# st_grf.py
"""
Spatio-temporal GRF components for MLQuantile4SpaceTime.
"""

from __future__ import annotations

import functools
import logging
import math
from typing import Optional, Sequence, Tuple

import numpy as np
import jax
import jax.numpy as jnp
from scipy.linalg import cholesky, solve_triangular
from scipy.optimize import minimize
from scipy.special import kv, gamma

logger = logging.getLogger(__name__)


# ============================================================
# Covariance / correlation helpers
# ============================================================
def matern_correlation(h: np.ndarray, kappa: float, nu: float) -> np.ndarray:
    """Unit-variance Matérn correlation ρ(h)."""
    h = np.asarray(h, float)
    scaled = kappa * np.abs(h)
    out = np.empty_like(scaled)
    zero = scaled == 0.0
    out[zero] = 1.0
    nz = ~zero
    if np.any(nz):
        out[nz] = (2.0 ** (1.0 - nu)) / gamma(nu) * (scaled[nz] ** nu) * kv(nu, scaled[nz])
    return out


def gneiting_correlation(
    h: np.ndarray,
    u: np.ndarray,
    a: float,
    kappa: float,
    nu: float,
    alpha: float,
    b: float,
    tau: float,
) -> np.ndarray:
    """Spatio-temporal Gneiting correlation ρ(h,u)."""
    tmp = 1.0 + a * np.abs(u) ** (2.0 * alpha)
    spatial_arg = h / (tmp ** (b / 2.0))
    return matern_correlation(spatial_arg, kappa, nu) / (tmp ** tau)


# ============================================================
# Simulation utilities (JAX)
# ============================================================
def random_invgamma(key, alpha, scale=1.0):
    gamma_sample = jax.random.gamma(key, alpha) + jnp.finfo(jnp.float32).tiny
    return scale / gamma_sample


def gamma_func(h, a, alpha):
    return a * (np.abs(h) ** (2 * alpha))


def simulate_gneiting_jax(
    spatial_coordinates,     # shape (n_s, d)
    temporal_coordinates,    # shape (n_t,)
    params,                  # [a, kappa, nu, alpha, tau, q]
    L,                       # total number of draws
    chunk_size=1000,
    nugget=1e-6
):
    from scipy.linalg import cholesky as _chol

    seed = np.random.SeedSequence().entropy % (2**32 - 1)
    key = jax.random.PRNGKey(seed)

    d = spatial_coordinates.shape[1]

    a, kappa, nu, alpha, tau, q = params
    b = 2 * tau / d * q
    delta = tau - (b * d / 2)

    spatial_coordinates = jnp.asarray(spatial_coordinates, dtype=jnp.float32)
    temporal_coordinates = jnp.asarray(temporal_coordinates, dtype=jnp.float32)

    n_s = spatial_coordinates.shape[0]
    n_t = temporal_coordinates.shape[0]

    # 1) Temporal covariance matrices
    delta_t = jnp.abs(temporal_coordinates[:, None] - temporal_coordinates[None, :])
    C_T = (1.0 + gamma_func(delta_t, a, alpha)) ** (-delta)
    gamma_b_mat = (1.0 + gamma_func(delta_t, a, alpha)) ** b - 1.0
    g0 = (1.0 + gamma_func(temporal_coordinates[:, None], a, alpha)) ** b - 1.0
    g1 = (1.0 + gamma_func(temporal_coordinates[None, :], a, alpha)) ** b - 1.0
    C_W = g0 + g1 - gamma_b_mat

    try:
        L_Z_T = _chol(C_T, lower=False)
    except np.linalg.LinAlgError:
        L_Z_T = _chol(C_T + nugget * np.eye(n_t), lower=False)

    try:
        L_W = _chol(C_W, lower=False)
    except np.linalg.LinAlgError:
        L_W = _chol(C_W + nugget * np.eye(n_t), lower=False)

    # 2) One-draw generator
    @functools.partial(jax.jit, static_argnums=(4,))
    def single_draw_fn(subkey, spatial_coords, LZ, LW, dim):
        xi_key, v_key, phi_key, z_key, w_key = jax.random.split(subkey, 5)

        # Xi ~ InvGamma(nu, scale=kappa^2 / 4)
        xi = random_invgamma(xi_key, alpha=nu, scale=(kappa ** 2) / 4)

        # V ~ N(0, I_d)
        V = jax.random.normal(v_key, shape=(dim,), dtype=jnp.float32)
        Omega = jnp.sqrt(2.0 * xi) * V
        V_norm_scaled = jnp.linalg.norm(V) / jnp.sqrt(2.0)

        # Phi ~ Uniform(0, 2π)
        Phi = jax.random.uniform(phi_key, minval=0.0, maxval=2.0 * jnp.pi, dtype=jnp.float32)

        # Z_T ~ N(0, C_T)
        Z_T = jax.random.normal(z_key, shape=(n_t,), dtype=jnp.float32) @ LZ

        # W ~ N(0, C_W)
        W = jax.random.normal(w_key, shape=(n_t,), dtype=jnp.float32) @ LW

        # Field
        arg = (spatial_coords @ Omega)[None, :] + (V_norm_scaled * W)[:, None] + Phi
        return Z_T[:, None] * jnp.cos(arg)

    # 3) Chunked averaging
    num_chunks = math.ceil(L / chunk_size)
    chunk_master_keys = jax.random.split(key, num_chunks)
    total_sum = jnp.zeros((n_t, n_s), dtype=jnp.float32)

    for chunk_idx in range(num_chunks):
        start = chunk_idx * chunk_size
        end = min(start + chunk_size, L)
        n_draws = end - start

        chunk_key = chunk_master_keys[chunk_idx]
        chunk_keys = jax.random.split(chunk_key, n_draws)

        chunk_draws = jax.vmap(
            lambda sk: single_draw_fn(sk, spatial_coordinates, L_Z_T, L_W, d)
        )(chunk_keys)

        total_sum += jnp.sum(chunk_draws, axis=0)

    return jnp.sqrt(2.0 / L) * total_sum


# ============================================================
# Composite-likelihood estimator (GneitingModel)
# ============================================================
class GneitingModel:
    r"""
    Composite-likelihood estimation of the Matérn–Gneiting spatio-temporal covariance.

    Parameters
    ----------
    coords : array-like, shape (n_sites, d)
        Spatial coordinates.
    t_max : int
        Look-back window (in discrete time steps) when building blocks.
    block_size : int
        Number of observations in every block.
    n_blocks : int
        Total number of blocks used in the composite likelihood.
    strategy : {"random","anchor","balanced"}
        Block sampling strategy.
    strata_bins : tuple (n_h, n_u)
        Number of strata along spatial and temporal lags (only for "balanced").
    initial_params : sequence [a, kappa, nu, alpha, tau, q]
        Starting values.
    epsilon : float
        Nugget added to the diagonal for numerical stability.
    random_state : int or None
        RNG seed.
    estimate_nu : bool
        If False (default), fix nu at `nu_fixed` and do NOT estimate it.
        If True, estimate nu alongside other parameters.
    nu_fixed : float
        Value of nu when `estimate_nu=False` (default 1.5).
    """

    def __init__(
        self,
        coords: np.ndarray,
        *,
        n1: int = 10,
        n2: int = 10,
        t_max: int = 4,
        block_size: int = 150,
        n_blocks: int = 500,
        strategy: str = "random",
        strata_bins: Tuple[int, int] = (8, 5),
        initial_params: Optional[Sequence[float]] = None,
        epsilon: float = 1e-8,
        random_state: Optional[int] = 10,
        estimate_nu: bool = False,
        nu_fixed: float = 1.5,
    ) -> None:
        self.coords = np.asarray(coords, float)
        self.t_max = int(t_max)
        self.block_size = int(block_size)
        self.n_blocks = int(n_blocks)

        valid_strategies = {"random", "anchor", "balanced"}
        if strategy not in valid_strategies:
            raise ValueError(f"strategy must be one of {valid_strategies}")
        self.strategy = strategy

        self.strata_bins = strata_bins
        self.epsilon = float(epsilon)
        self.rng = np.random.default_rng(random_state)
        self._logger = logging.getLogger(__name__)

        if initial_params is None:
            #      a,  kappa,  nu,  alpha, tau,   q
            initial_params = [0.1, 1.0,   1.5,  0.5,  0.1,  0.1]
        self.x0 = np.array(initial_params, dtype=float)

        self.estimate_nu = bool(estimate_nu)
        self.nu_fixed = float(nu_fixed)

        self.estimated_params: Optional[Tuple[float, ...]] = None
        self.optim_result = None

        self._blocks: Optional[np.ndarray] = None
        self._h_edges: Optional[np.ndarray] = None
        self._u_edges: Optional[np.ndarray] = None
        self._dist_mat: Optional[np.ndarray] = None
        self._spatial_inited = False

        self._precompute_spatial_distances()

    def _precompute_spatial_distances(self) -> None:
        diff = self.coords[:, None, :] - self.coords[None, :, :]
        self._dist_mat = np.linalg.norm(diff, axis=-1)

        if self.strategy == "balanced":
            n_h, n_u = self.strata_bins
            flat_h = self._dist_mat[np.triu_indices_from(self._dist_mat, k=1)]
            self._h_edges = np.quantile(flat_h, np.linspace(0.0, 1.0, n_h + 1))
            self._u_edges = np.linspace(0.0, self.t_max, n_u + 1)

    # --- correlation wrappers
    @staticmethod
    def matern_correlation(h: np.ndarray, kappa: float, nu: float) -> np.ndarray:
        return matern_correlation(h, kappa, nu)

    def gneiting_correlation(
        self,
        h: np.ndarray,
        u: np.ndarray,
        a: float,
        kappa: float,
        nu: float,
        alpha: float,
        b: float,
        tau: float,
    ) -> np.ndarray:
        return gneiting_correlation(h, u, a, kappa, nu, alpha, b, tau)

    # --- blocks
    def _draw_block(self, Z: np.ndarray, valid_idx: np.ndarray, times: np.ndarray) -> np.ndarray:
        K = self.block_size
        if self.strategy == "random":
            chosen = self.rng.choice(valid_idx, size=K, replace=False)
            return np.column_stack(np.unravel_index(chosen, Z.shape))

        anchor_flat = int(self.rng.choice(valid_idx))
        t0, s0 = np.unravel_index(anchor_flat, Z.shape)

        spatial_pool = np.delete(np.arange(self.coords.shape[0]), s0)
        t_pool = times[(times >= max(0, t0 - self.t_max)) & (times <= t0)]

        cand_ts, cand_ss = np.meshgrid(t_pool, spatial_pool, indexing="ij")
        cand_pairs = np.column_stack([cand_ts.ravel(), cand_ss.ravel()])

        mask = ~np.isnan(Z[cand_pairs[:, 0], cand_pairs[:, 1]])
        cand_pairs = cand_pairs[mask]

        if self.strategy == "balanced":
            return self._balanced_sample(Z, cand_pairs, anchor=(t0, s0), K=K)

        if len(cand_pairs) >= K - 1:
            picks = self.rng.choice(len(cand_pairs), size=K - 1, replace=False)
            block = np.vstack([[t0, s0], cand_pairs[picks]])
        else:
            block = self._draw_block(Z, valid_idx, times)
        return block

    def _balanced_sample(
        self,
        Z: np.ndarray,
        cand_pairs: np.ndarray,
        *,
        anchor: Tuple[int, int],
        K: int,
    ) -> np.ndarray:
        t0, s0 = anchor
        h_edges = self._h_edges
        u_edges = self._u_edges
        n_h, n_u = self.strata_bins

        h_vals = self._dist_mat[s0, cand_pairs[:, 1]]
        u_vals = np.abs(cand_pairs[:, 0] - t0)

        h_id = np.digitize(h_vals, h_edges, right=False) - 1
        u_id = np.digitize(u_vals, u_edges, right=False) - 1
        h_id = np.clip(h_id, 0, n_h - 1)
        u_id = np.clip(u_id, 0, n_u - 1)
        stratum_id = h_id * n_u + u_id

        buckets = {k: [] for k in range(n_h * n_u)}
        for idx, sid in enumerate(stratum_id):
            buckets[sid].append(idx)
        for lst in buckets.values():
            self.rng.shuffle(lst)

        chosen_idx: list[int] = []
        sid_order = list(buckets.keys())
        self.rng.shuffle(sid_order)

        while len(chosen_idx) < K - 1 and sid_order:
            for sid in list(sid_order):
                if not buckets[sid]:
                    sid_order.remove(sid)
                    continue
                chosen_idx.append(buckets[sid].pop())
                if len(chosen_idx) == K - 1:
                    break

        if len(chosen_idx) < K - 1 and len(cand_pairs) >= K - 1:
            remaining = np.setdiff1d(np.arange(len(cand_pairs)), chosen_idx, assume_unique=True)
            extra = self.rng.choice(remaining, size=K - 1 - len(chosen_idx), replace=False)
            chosen_idx.extend(extra)

        if len(chosen_idx) < K - 1:
            return self._draw_block(Z, np.flatnonzero(~np.isnan(Z)), np.arange(Z.shape[0]))

        return np.vstack([[t0, s0], cand_pairs[chosen_idx]])

    def _generate_blocks(self, Z: np.ndarray) -> None:
        n_t, _ = Z.shape
        times = np.arange(n_t)
        valid_idx = np.flatnonzero(~np.isnan(Z))
        blocks = [self._draw_block(Z, valid_idx, times) for _ in range(self.n_blocks)]
        self._blocks = np.array(blocks, dtype=int)

    # --- params
    def _unpack_params(self, x: np.ndarray) -> Tuple[float, float, float, float, float, float]:
        if self.estimate_nu:
            a, kappa, nu, alpha, tau, q = x
        else:
            a, kappa, alpha, tau, q = x
            nu = self.nu_fixed
        return float(a), float(kappa), float(nu), float(alpha), float(tau), float(q)

    # --- likelihood
    def _block_loglik(self, z: np.ndarray, Sigma: np.ndarray) -> float:
        L = cholesky(Sigma + self.epsilon * np.eye(len(z)), lower=True)
        alpha = solve_triangular(L, z, lower=True)
        return -0.5 * (2.0 * np.sum(np.log(np.diag(L))) + alpha @ alpha)

    def _composite_nll(self, x: np.ndarray, Z: np.ndarray) -> float:
        a, kappa, nu, alpha, tau, q = self._unpack_params(x)
        d = self.coords.shape[1]
        b = 2.0 * tau * q / d

        ll_total = 0.0
        for block in self._blocks:
            t_idx, s_idx = block.T
            z = Z[t_idx, s_idx]

            h = np.linalg.norm(
                self.coords[s_idx][:, None, :] - self.coords[s_idx][None, :, :],
                axis=2,
            )
            u = np.abs(t_idx[:, None] - t_idx[None, :])

            Sigma = self.gneiting_correlation(h, u, a, kappa, nu, alpha, b, tau)
            ll_total += self._block_loglik(z, Sigma)
        return -ll_total

    # --- public
    def fit(self, Z: np.ndarray, *, maxiter: int = 50_000, verbose: bool = True):
        self._generate_blocks(Z)

        if self.estimate_nu:
            x0 = self.x0.copy()
            bounds = [
                (1e-5, None),     # a
                (1e-5, None),     # kappa
                (1e-5, None),     # nu
                (1e-5, 1 - 1e-5), # alpha
                (1e-5, 1 - 1e-5), # tau
                (1e-5, 1 - 1e-5), # q
            ]
        else:
            x0 = np.array([self.x0[0], self.x0[1], self.x0[3], self.x0[4], self.x0[5]], dtype=float)
            bounds = [
                (1e-5, None),     # a
                (1e-5, None),     # kappa
                (1e-5, 1 - 1e-5), # alpha
                (1e-5, 1 - 1e-5), # tau
                (1e-5, 1 - 1e-5), # q
            ]

        res = minimize(
            fun=self._composite_nll,
            x0=x0,
            args=(Z,),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": maxiter, "disp": verbose},
        )
        if not res.success:
            self._logger.warning("Optimiser did not converge: %s", res.message)

        if self.estimate_nu:
            a, kappa, nu, alpha, tau, q = self._unpack_params(res.x)
        else:
            a, kappa, alpha, tau, q = res.x
            nu = self.nu_fixed

        self.estimated_params = (a, kappa, nu, alpha, tau, q)
        self.optim_result = res
        return self.estimated_params, res

    def fit_for_weather_type(self, Z: np.ndarray, wt_indices: Sequence[int], *, maxiter: int = 500_000, verbose: bool = True):
        wt_indices = np.asarray(wt_indices, dtype=int)
        if wt_indices.ndim != 1:
            raise ValueError("wt_indices must be 1-D sequence of integers")

        Z_subset = Z.copy()
        mask_rows = np.ones(Z.shape[0], dtype=bool)
        mask_rows[wt_indices] = False
        Z_subset[mask_rows, :] = np.nan

        return self.fit(Z_subset, maxiter=maxiter, verbose=verbose)
