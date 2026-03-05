# pipeline.py
"""
Quantile2SpaceTime pipeline

End-to-end framework:
  1) Fit conditional marginals Y|X via quantile regression (KNN/QRF/QRNN)
  2) Transform Y -> latent Gaussian Z using U = F_{Y|X}(Y), Z = Phi^{-1}(U)
  3) Fit spatio-temporal GRF parameters in latent space (e.g., Matérn–Gneiting)
  4) Simulate Z(s,t) on target times and invert back to Y via learned marginals

This file is intentionally thin: it orchestrates the components implemented in
  - marginal.py:   SitewiseMarginal
  - fit_gneiting.py: GneitingModel
  - simulate.py:   simulate_gneiting_jax
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .marginal import SitewiseMarginal
from .fit_gneiting import GneitingModel
from .simulate import simulate_gneiting_jax


@dataclass
class Quantile2SpaceTimeConfig:
    # --- Marginal stage
    marginal_method: str = "knn"  # "knn" | "qrf" | "qrnn"
    marginal_kwargs: Optional[Dict[str, Any]] = None
    marginal_taus: Optional[np.ndarray] = None
    var_select: bool = False
    var_select_kwargs: Optional[Dict[str, Any]] = None

    # --- Latent GRF stage (GneitingModel)
    gneiting_strategy: str = "balanced"
    gneiting_strata_bins: Tuple[int, int] = (8, 5)
    gneiting_initial_params: Optional[Sequence[float]] = None  # [a,kappa,nu,alpha,tau,q] or None
    gneiting_estimate_nu: bool = False
    gneiting_nu_fixed: float = 1.5

    # Composite-likelihood controls
    block_size: int = 150
    n_blocks: int = 500
    t_max: int = 3
    epsilon: float = 1e-8
    random_state: int = 42

    # --- Simulation controls (defaults; can override per call)
    L_draws_default: int = 50_000
    chunk_size_default: int = 500
    nugget_default: float = 1e-8


class Quantile2SpaceTimeModel:
    """
    Main pipeline entry-point.

    Parameters
    ----------
    coords : array-like, shape (n_sites, d)
        Spatial coordinates.
    config : Quantile2SpaceTimeConfig or None
        Configuration for marginal fitting, GRF fitting, and defaults for simulation.
    """

    def __init__(
        self,
        coords: np.ndarray,
        *,
        config: Optional[Quantile2SpaceTimeConfig] = None,
        # convenience overrides (optional)
        marginal_method: Optional[str] = None,
        marginal_kwargs: Optional[Dict[str, Any]] = None,
        marginal_taus: Optional[np.ndarray] = None,
        var_select: Optional[bool] = None,
        var_select_kwargs: Optional[Dict[str, Any]] = None,
        gneiting_strategy: Optional[str] = None,
        gneiting_strata_bins: Optional[Tuple[int, int]] = None,
        gneiting_initial_params: Optional[Sequence[float]] = None,
        gneiting_estimate_nu: Optional[bool] = None,
        gneiting_nu_fixed: Optional[float] = None,
        block_size: Optional[int] = None,
        n_blocks: Optional[int] = None,
        t_max: Optional[int] = None,
        epsilon: Optional[float] = None,
        random_state: Optional[int] = None,
    ) -> None:
        self.coords = np.asarray(coords, float)
        self.n_sites = self.coords.shape[0]

        cfg = Quantile2SpaceTimeConfig() if config is None else config

        # apply overrides (keeping it explicit and predictable)
        if marginal_method is not None:
            cfg.marginal_method = str(marginal_method)
        if marginal_kwargs is not None:
            cfg.marginal_kwargs = dict(marginal_kwargs)
        if marginal_taus is not None:
            cfg.marginal_taus = np.asarray(marginal_taus, dtype=np.float32)
        if var_select is not None:
            cfg.var_select = bool(var_select)
        if var_select_kwargs is not None:
            cfg.var_select_kwargs = dict(var_select_kwargs)

        if gneiting_strategy is not None:
            cfg.gneiting_strategy = str(gneiting_strategy)
        if gneiting_strata_bins is not None:
            cfg.gneiting_strata_bins = tuple(gneiting_strata_bins)
        if gneiting_initial_params is not None:
            cfg.gneiting_initial_params = list(gneiting_initial_params)
        if gneiting_estimate_nu is not None:
            cfg.gneiting_estimate_nu = bool(gneiting_estimate_nu)
        if gneiting_nu_fixed is not None:
            cfg.gneiting_nu_fixed = float(gneiting_nu_fixed)

        if block_size is not None:
            cfg.block_size = int(block_size)
        if n_blocks is not None:
            cfg.n_blocks = int(n_blocks)
        if t_max is not None:
            cfg.t_max = int(t_max)
        if epsilon is not None:
            cfg.epsilon = float(epsilon)
        if random_state is not None:
            cfg.random_state = int(random_state)

        self.config = cfg

        # fitted artifacts
        self._is_fitted = False
        self.dates_: Optional[pd.DatetimeIndex] = None
        self.X_cov_: Optional[np.ndarray] = None
        self.Y_obs_: Optional[np.ndarray] = None

        self.marginal_: Optional[SitewiseMarginal] = None
        self.Z_train_: Optional[np.ndarray] = None

        self.gneiting_: Optional[GneitingModel] = None
        self.gneiting_params_: Optional[Tuple[float, ...]] = None

        # surfaced info
        self.marginal_selected_hyperparams_: Optional[Dict[str, Any]] = None
        self.selected_cols_: Optional[np.ndarray] = None
        self.feature_importances_: Optional[np.ndarray] = None

    # ---------------------------------------------------------------------
    # Public helpers
    # ---------------------------------------------------------------------
    @property
    def is_fitted(self) -> bool:
        return bool(self._is_fitted)

    def get_selected_hyperparams(self) -> Optional[Dict[str, Any]]:
        """Return a copy of the marginal method's selected hyperparameters (if available)."""
        return None if self.marginal_selected_hyperparams_ is None else dict(self.marginal_selected_hyperparams_)

    # ---------------------------------------------------------------------
    # Fit
    # ---------------------------------------------------------------------
    def fit(self, *, X_cov: np.ndarray, Y_obs: np.ndarray, dates: Sequence[Any]) -> "Quantile2SpaceTimeModel":
        """
        Fit the full pipeline on training data.

        Parameters
        ----------
        X_cov : array-like, shape (n_time, n_features)
        Y_obs : array-like, shape (n_time, n_sites)
        dates : array-like, length n_time
            Anything parseable by pandas.to_datetime.

        Returns
        -------
        self
        """
        X_cov = np.asarray(X_cov, dtype=np.float32)
        Y_obs = np.asarray(Y_obs, dtype=np.float32)
        dates = pd.to_datetime(dates)

        if X_cov.ndim != 2:
            raise ValueError("X_cov must be 2D: (n_time, n_features).")
        if Y_obs.ndim != 2:
            raise ValueError("Y_obs must be 2D: (n_time, n_sites).")
        if X_cov.shape[0] != Y_obs.shape[0]:
            raise ValueError("X_cov and Y_obs must have the same n_time.")
        if Y_obs.shape[1] != self.n_sites:
            raise ValueError(f"Y_obs must have n_sites={self.n_sites} columns to match coords.")

        self.X_cov_ = X_cov
        self.Y_obs_ = Y_obs
        self.dates_ = pd.DatetimeIndex(dates)

        # 1) Fit marginals
        self.marginal_ = SitewiseMarginal(
            X_train=self.X_cov_,
            Y_train=self.Y_obs_,
            method=self.config.marginal_method,
            model_kwargs={} if self.config.marginal_kwargs is None else dict(self.config.marginal_kwargs),
            taus=self.config.marginal_taus,
            var_select=self.config.var_select,
            var_select_kwargs={} if self.config.var_select_kwargs is None else dict(self.config.var_select_kwargs),
        )

        self.marginal_selected_hyperparams_ = dict(getattr(self.marginal_, "selected_hyperparams_", {}))

        if self.config.var_select:
            self.selected_cols_ = getattr(self.marginal_, "selected_cols_", None)
            self.feature_importances_ = getattr(self.marginal_, "feature_importances_", None)

        # 2) Transform training Y -> Z
        self.Z_train_ = self.marginal_.y_to_z(self.X_cov_, self.Y_obs_)

        # 3) Fit latent GRF parameters (Gneiting composite-likelihood)
        self.gneiting_ = GneitingModel(
            coords=self.coords,
            t_max=self.config.t_max,
            block_size=self.config.block_size,
            n_blocks=self.config.n_blocks,
            strategy=self.config.gneiting_strategy,
            strata_bins=self.config.gneiting_strata_bins,
            initial_params=self.config.gneiting_initial_params,
            epsilon=self.config.epsilon,
            random_state=self.config.random_state,
            estimate_nu=self.config.gneiting_estimate_nu,
            nu_fixed=self.config.gneiting_nu_fixed,
        )

        self.gneiting_params_, _ = self.gneiting_.fit(self.Z_train_)
        self._is_fitted = True
        return self

    # ---------------------------------------------------------------------
    # Simulate
    # ---------------------------------------------------------------------
    def simulate(
        self,
        *,
        X_test: np.ndarray,
        test_dates: Sequence[Any],
        n_simulations: int = 1,
        L_draws: Optional[int] = None,
        chunk_size: Optional[int] = None,
        nugget: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Simulate spatio-temporal trajectories over a test window.

        Parameters
        ----------
        X_test : array-like, shape (n_test, n_features)
            Covariates at which to condition the marginal inversion Z -> Y.
        test_dates : array-like, length n_test
            Dates associated with the test window (stored for provenance; not used in simulation).
        n_simulations : int
            Number of independent simulations to generate.
        L_draws : int or None
            Spectral draw budget used by simulate_gneiting_jax.
        chunk_size : int or None
            Chunk size for spectral draws.
        nugget : float or None
            Numerical nugget used in Cholesky factorizations.

        Returns
        -------
        dummy : np.ndarray
            Placeholder array with shape (n_simulations, n_test). Kept for backward compatibility
            with earlier code that returned a regime sequence.
        Z_out : np.ndarray
            Latent simulations, shape (n_simulations, n_test, n_sites)
        Y_out : np.ndarray
            Data-space simulations, shape (n_simulations, n_test, n_sites)
        """
        if not self._is_fitted or self.marginal_ is None or self.gneiting_params_ is None:
            raise RuntimeError("Model is not fitted. Call .fit(...) first.")

        X_test = np.asarray(X_test, dtype=np.float32)
        _ = pd.to_datetime(test_dates)  # validate; stored by the caller if needed

        if X_test.ndim != 2:
            raise ValueError("X_test must be 2D: (n_test, n_features).")

        n_test = X_test.shape[0]
        n_simulations = int(n_simulations)
        if n_simulations <= 0:
            raise ValueError("n_simulations must be >= 1.")

        L = int(self.config.L_draws_default if L_draws is None else L_draws)
        cs = int(self.config.chunk_size_default if chunk_size is None else chunk_size)
        ng = float(self.config.nugget_default if nugget is None else nugget)

        # Simple time index for the latent simulator
        temporal_coordinates = np.arange(n_test, dtype=float)

        Z_out = np.zeros((n_simulations, n_test, self.n_sites), dtype=np.float32)
        Y_out = np.zeros((n_simulations, n_test, self.n_sites), dtype=np.float32)

        for i in range(n_simulations):
            Z_sim = simulate_gneiting_jax(
                spatial_coordinates=self.coords,
                temporal_coordinates=temporal_coordinates,
                params=self.gneiting_params_,
                L=L,
                chunk_size=cs,
                nugget=ng,
            )
            Y_sim = self.marginal_.z_to_y(X_test, Z_sim)

            Z_out[i] = np.asarray(Z_sim, dtype=np.float32)
            Y_out[i] = np.asarray(Y_sim, dtype=np.float32)

        # kept as placeholder for backward compatibility (no regimes in this repo by default)
        dummy = np.zeros((n_simulations, n_test), dtype=int)
        return dummy, Z_out, Y_out
