# marginal.py
"""
Marginal models for Quantile2SpaceTime.

Implements SitewiseMarginal:
  - method="knn": KNN-based conditional CDF / quantiles (kernel-weighted)
  - method="qrf": Quantile Regression Forests (quantile_forest preferred; sklearn-quantile fallback)
  - method="qrnn": Quantile Regression Neural Network (quantnn, PyTorch backend)

Core API:
  - predict_quantiles(X_new) -> Q(N, S, m)
  - predict_cdf(X_new, Y_eval) -> U(N, S)
  - y_to_z(X_new, Y) -> Z(N, S)
  - z_to_y(X_new, Z) -> Y_hat(N, S)

Notes:
  - Optional RF-based feature selection (var_select=True)
  - TimeSeriesSplit CV for hyperparam selection
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np
from scipy.stats import norm
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neighbors import KDTree

logger = logging.getLogger(__name__)


# ============================================================
# Helpers: kernels
# ============================================================
def gaussian_kernel(u):
    return np.exp(-0.5 * (u**2))


def epanechnikov_kernel(u):
    out = 1.0 - np.minimum(u, 1.0) ** 2
    out[u > 1.0] = 0.0
    return np.maximum(out, 0.0)


def inverse_distance_simple(u, eps=1e-12):
    return 1.0 / np.maximum(u, eps)


# ============================================================
# Prefer quantile_forest's QRF; fallback to sklearn_quantile if needed
# ============================================================
try:
    from quantile_forest import RandomForestQuantileRegressor as _QRF  # type: ignore

    QRF_IMPL = "quantile_forest"
except Exception:
    try:
        from sklearn_quantile import RandomForestQuantileRegressor as _QRF  # type: ignore

        QRF_IMPL = "sklearn_quantile"
    except Exception:
        _QRF = None
        QRF_IMPL = None


# ============================================================
# quantnn imports (optional)
# ============================================================
try:
    import quantnn

    quantnn.set_default_backend("pytorch")
    from quantnn.qrnn import QRNN
    from quantnn.models.pytorch import FullyConnected
except Exception:
    quantnn = None
    QRNN = None
    FullyConnected = None


def to_np32(x):
    if isinstance(x, np.ndarray):
        return x.astype(np.float32, copy=False)
    try:
        import torch

        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy().astype(np.float32, copy=False)
    except Exception:
        pass
    return np.asarray(x, dtype=np.float32)


class SitewiseMarginal:
    """
    Methods: 'knn' (pure CDF), 'qrf', 'qrnn'.

    Cross-validation (for all three):
      - Mean pinball loss
      - TimeSeriesSplit
      - Random subset of sites (cv_n_sites=10 by default)

    Parallelization:
      - QRF: parallel across sites only if rf.n_jobs == 1 (avoid nested parallel).
      - QRNN: parallel across sites with threads (no pickling), or sequential if n_jobs_sites==1.

    Exposes:
      - selected_hyperparams_: dict describing chosen hyperparameters for the active method.
    """

    def __init__(
        self,
        X_train,
        Y_train,
        method="knn",
        model_kwargs=None,
        taus=None,
        eps=1e-3,
        *,
        var_select=False,
        var_select_kwargs=None,
    ):
        self.eps = float(eps)
        self.method = str(method).lower()
        self.taus = (
            np.asarray(taus, np.float32)
            if taus is not None
            else np.linspace(0.0, 1.0, 200, dtype=np.float32)
        )

        self.selected_hyperparams_ = {"method": self.method}

        self.X_train_full = np.asarray(X_train, np.float32)
        self.Y_train = np.asarray(Y_train, np.float32)

        # Drop rows with non-finite X to keep KDTree happy
        ok_rows = np.isfinite(self.X_train_full).all(axis=1)
        if (~ok_rows).any():
            n_drop = int((~ok_rows).sum())
            self.X_train_full = self.X_train_full[ok_rows]
            self.Y_train = self.Y_train[ok_rows]
            print(f"[SitewiseMarginal] Dropped {n_drop} rows with non-finite X.")

        self.N, self.d_full = self.X_train_full.shape
        _, self.S = self.Y_train.shape

        # Optional: feature selection
        self.var_select = bool(var_select)
        self.var_select_kwargs = {} if var_select_kwargs is None else dict(var_select_kwargs)
        if self.var_select:
            self.selected_cols_, self.feature_importances_ = self._rf_select_columns(
                self.X_train_full, self.Y_train, **self._vs_defaults(self.var_select_kwargs)
            )
            X_sel = self.X_train_full[:, self.selected_cols_]
        else:
            self.selected_cols_ = np.arange(self.d_full, dtype=int)
            self.feature_importances_ = None
            X_sel = self.X_train_full

        # Fallback fillers for non-finite query features
        self._x_fill = np.nanmean(X_sel, axis=0, dtype=np.float32)
        self._x_fill[~np.isfinite(self._x_fill)] = 0.0
        self.N, self.d = X_sel.shape

        kw = {} if model_kwargs is None else dict(model_kwargs)
        self.n_jobs_sites = kw.get("n_jobs_sites", 1)
        self.standardize_X = bool(kw.get("standardize_X", True))
        if self.standardize_X:
            self._x_mean = X_sel.mean(0, keepdims=True)
            self._x_std = np.clip(X_sel.std(0, keepdims=True), self.eps, None)
            X_proc = (X_sel - self._x_mean) / self._x_std
        else:
            self._x_mean = self._x_std = None
            X_proc = X_sel

        if not np.isfinite(X_proc).all():
            X_proc = np.where(~np.isfinite(X_proc), 0.0, X_proc)
        self.X_train = X_proc

        # Route
        if self.method == "knn":
            self._init_knn(kw)
        elif self.method == "qrf":
            self._init_qrf(kw)
        elif self.method == "qrnn":
            self._init_qrnn(kw)
        else:
            raise ValueError("method must be 'knn', 'qrf', or 'qrnn'")

    # ─────────────────────────────────────────────────────────────
    # Feature selection helpers
    # ─────────────────────────────────────────────────────────────
    def _vs_defaults(self, vs_kw):
        import copy

        d = dict(cum_thr=0.95, n_sites=10, random_state=42, rf_kwargs=None)
        d.update(copy.deepcopy(vs_kw))
        return d

    def _rf_select_columns(self, X, Y, *, cum_thr=0.95, n_sites=10, random_state=42, rf_kwargs=None):
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.multioutput import MultiOutputRegressor

        N, P = X.shape
        rng = np.random.default_rng(random_state)
        sites = rng.choice(np.arange(Y.shape[1]), size=min(n_sites, Y.shape[1]), replace=False)

        rf_params = dict(
            n_estimators=300, max_depth=None, min_samples_leaf=1, n_jobs=-1, random_state=random_state
        )
        if rf_kwargs:
            rf_params.update(rf_kwargs)

        rf = MultiOutputRegressor(RandomForestRegressor(**rf_params))
        rf.fit(X, Y[:, sites])

        importances = np.zeros(P, dtype=float)
        for est in rf.estimators_:
            importances += est.feature_importances_
        importances /= max(1, len(rf.estimators_))
        total = importances.sum()

        if total <= 1e-12:
            selected = np.arange(P, dtype=int)
        else:
            cols_sorted = np.argsort(importances)[::-1]
            cum = np.cumsum(importances[cols_sorted]) / (total + 1e-12)
            k_keep = int(np.searchsorted(cum, float(cum_thr))) + 1
            selected = cols_sorted[:k_keep]

        print(f"[RF select] cumulative {cum_thr:.0%} → keep {selected.size}/{P} columns")
        return selected.astype(int), importances

    def _prepare_X(self, X):
        X = np.asarray(X, np.float32)[:, self.selected_cols_]
        bad = ~np.isfinite(X)
        if bad.any():
            X = np.where(bad, self._x_fill[None, :], X)
        if self.standardize_X:
            X = (X - self._x_mean) / self._x_std
            X = np.where(~np.isfinite(X), 0.0, X)
        return X

    def _row_normalize_after_mask(self, ww):
        denom = ww.sum(axis=1, keepdims=True)
        safe_rows = (denom > self.eps).squeeze(1)
        if np.any(safe_rows):
            ww[safe_rows, :] = ww[safe_rows, :] / denom[safe_rows, 0][:, None]
        return ww, safe_rows

    # ─────────────────────────────────────────────────────────────
    # KNN
    # ─────────────────────────────────────────────────────────────
    def _resolve_bandwidth(self, d, k_used):
        if not self._kernel_uses_bandwidth:
            return None
        if isinstance(self.h_mode, tuple) and self.h_mode[0] == "adaptive":
            c = max(float(self.h_mode[1]), self.eps)
            return np.maximum(d[:, k_used - 1], self.eps)[:, None] * c
        elif self.h_mode == "adaptive":
            return np.maximum(d[:, k_used - 1], self.eps)[:, None]
        else:
            hval = max(float(self.h_mode), self.eps)
            return np.full((d.shape[0], 1), hval, dtype=d.dtype)

    def _init_knn(self, kw):
        ker = kw.get("kernel", "gaussian")
        self._kernel_uses_bandwidth = True
        if callable(ker):
            self.kernel = ker
            kernel_name = getattr(ker, "__name__", "callable")
            self._kernel_uses_bandwidth = getattr(self, "_kernel_uses_bandwidth", True)
        elif ker == "gaussian":
            self.kernel = gaussian_kernel
            kernel_name = "gaussian"
        elif ker == "epanechnikov":
            self.kernel = epanechnikov_kernel
            kernel_name = "epanechnikov"
        elif ker in ("inverse_simple", "inverse", "inv"):
            self.kernel = inverse_distance_simple
            kernel_name = "inverse_simple"
            self._kernel_uses_bandwidth = False
        else:
            raise ValueError(f"unknown kernel: {ker}")

        self.h_mode = kw.get("h", "adaptive")  # float | "adaptive" | ("adaptive", c) | "auto"

        want_auto_k = kw.get("select_k", False) or kw.get("k", "auto") == "auto"
        want_auto_h = (kw.get("h", "adaptive") == "auto")
        if want_auto_k or want_auto_h:
            self.k, self.h_mode = self._select_kh_via_cv(
                k_grid=kw.get("k_grid", np.arange(20, 181, 20)),
                n_splits=kw.get("cv_splits", 5),
                cv_n_sites=kw.get("cv_n_sites", 10),
                cv_random_state=kw.get("cv_random_state", 42),
                c_grid=kw.get("c_grid", None),
                h_grid=kw.get("h_grid", None),
            )
        else:
            self.k = int(kw.get("k", 100))

        if not np.isfinite(self.X_train).all():
            raise ValueError("[SitewiseMarginal] X_train passed to KDTree contains NaN/Inf after preprocessing.")
        self.tree = KDTree(self.X_train)
        self.taus = np.linspace(0.001, 0.999, 200, dtype=np.float32)

        hp = {"method": "knn", "kernel": kernel_name, "k": int(self.k)}
        if not self._kernel_uses_bandwidth:
            hp.update({"bandwidth_mode": "none"})
        elif isinstance(self.h_mode, tuple) and self.h_mode[0] == "adaptive":
            hp.update({"bandwidth_mode": "adaptive", "c": float(self.h_mode[1])})
        elif self.h_mode == "adaptive":
            hp.update({"bandwidth_mode": "adaptive", "c": 1.0})
        else:
            hp.update({"bandwidth_mode": "fixed", "h": float(self.h_mode)})
        hp["standardize_X"] = bool(self.standardize_X)
        self.selected_hyperparams_ = hp

    def _knn_query(self, X):
        Xq = self._prepare_X(X)
        d, idx = self.tree.query(Xq, k=self.k)
        if self._kernel_uses_bandwidth:
            h = self._resolve_bandwidth(d, self.k)
            u = d / h
        else:
            u = d
        w = self.kernel(u)
        w = np.maximum(w, 1e-12)
        w, _ = self._row_normalize_after_mask(w)
        return idx, w

    def _knn_neighbors(self, X, s0=0, s1=None):
        idx, w = self._knn_query(X)
        if s1 is None:
            s1 = self.S
        Y_block = self.Y_train[:, s0:s1]
        Y_nb = Y_block[idx]
        valid = np.isfinite(Y_nb)
        return idx, w, Y_nb, valid

    def _knn_predict_cdf(self, X_new, Y_eval):
        X_new = np.asarray(X_new, np.float32)
        Y_eval = np.asarray(Y_eval, np.float32)
        Nq = X_new.shape[0]
        U = np.full((Nq, self.S), np.nan, dtype=np.float32)

        s_batch = min(self.S, 256)
        for s0 in range(0, self.S, s_batch):
            s1 = min(self.S, s0 + s_batch)
            _, w, Y_nb, valid = self._knn_neighbors(X_new, s0, s1)
            for sb in range(s1 - s0):
                y_th = Y_eval[:, s0 + sb]
                y_nb = Y_nb[..., sb]
                msk = valid[..., sb]
                ok_thr = np.isfinite(y_th)
                if not np.any(ok_thr):
                    continue
                ww = w[ok_thr] * msk[ok_thr]
                ww, safe_rows = self._row_normalize_after_mask(ww)
                y_nb_ok = y_nb[ok_thr]
                thr_ok = y_th[ok_thr]
                ind = (y_nb_ok <= thr_ok[:, None]).astype(np.float32)
                u = (ww * ind).sum(axis=1)
                u = np.where(safe_rows, np.clip(u, self.eps, 1.0 - self.eps), np.nan).astype(np.float32)
                U[ok_thr, s0 + sb] = u
        return U

    # ─────────────────────────────────────────────────────────────
    # QRF
    # ─────────────────────────────────────────────────────────────
    def _init_qrf(self, kw):
        if _QRF is None:
            raise ImportError("No QRF implementation available. Install 'quantile_forest' or 'sklearn-quantile'.")

        do_cv = bool(kw.get("qrf_select_hyperparams", True))
        if do_cv:
            best_leaf, best_mf = self._select_qrf_hyperparams_via_cv(
                n_splits=kw.get("cv_splits", 5),
                cv_n_sites=kw.get("cv_n_sites", 10),
                cv_random_state=kw.get("cv_random_state", 42),
                leaf_grid=kw.get("qrf_leaf_grid", [3, 5, 10]),
                maxfeat_grid=kw.get("qrf_maxfeat_grid", ["sqrt", 0.5]),
                n_estimators=kw.get("n_estimators", 500),
                bootstrap=kw.get("bootstrap", False),
                min_samples_split=kw.get("min_samples_split", 2),
                max_depth=kw.get("max_depth", None),
            )
            kw = dict(kw)
            kw["min_samples_leaf"] = best_leaf
            kw["max_features"] = best_mf

        params = dict(
            n_estimators=kw.get("n_estimators", 500),
            min_samples_leaf=kw.get("min_samples_leaf", 5),
            max_features=kw.get("max_features", "sqrt"),
            max_depth=kw.get("max_depth", None),
            bootstrap=kw.get("bootstrap", False),
            min_samples_split=kw.get("min_samples_split", 2),
            random_state=kw.get("random_state", 42),
            n_jobs=kw.get("n_jobs", -1 if self.n_jobs_sites == 1 else 1),
        )
        strip = {
            "k",
            "k_grid",
            "cv_splits",
            "select_k",
            "kernel",
            "h",
            "standardize_X",
            "c_grid",
            "h_grid",
            "cv_n_sites",
            "cv_random_state",
            "qrf_select_hyperparams",
            "qrf_leaf_grid",
            "qrf_maxfeat_grid",
        }
        params.update({k: v for k, v in kw.items() if k not in strip})

        self.taus = np.linspace(0.001, 0.999, 100, dtype=np.float32)
        Xp = self._prepare_X(self.X_train_full)

        def _fit_qrf_site(s):
            rf = _QRF(**params).fit(Xp, self.Y_train[:, s])
            return rf

        if int(params["n_jobs"]) == 1 and self.n_jobs_sites > 1:
            from joblib import Parallel, delayed

            self.models = Parallel(n_jobs=self.n_jobs_sites, prefer="processes")(
                delayed(_fit_qrf_site)(s) for s in range(self.S)
            )
            site_fit_parallel = True
        else:
            self.models = [_fit_qrf_site(s) for s in range(self.S)]
            site_fit_parallel = False

        rf0 = self.models[0]
        self.selected_hyperparams_ = {
            "method": "qrf",
            "impl": QRF_IMPL,
            "n_estimators": int(getattr(rf0, "n_estimators", params["n_estimators"])),
            "min_samples_leaf": int(getattr(rf0, "min_samples_leaf", params["min_samples_leaf"])),
            "max_features": getattr(rf0, "max_features", params["max_features"]),
            "max_depth": getattr(rf0, "max_depth", params["max_depth"]),
            "bootstrap": bool(getattr(rf0, "bootstrap", params["bootstrap"])),
            "min_samples_split": int(getattr(rf0, "min_samples_split", params["min_samples_split"])),
            "standardize_X": bool(self.standardize_X),
            "site_fit_parallel": site_fit_parallel,
            "n_jobs_sites": int(self.n_jobs_sites),
            "rf_n_jobs": int(params["n_jobs"]),
        }

    def _select_qrf_hyperparams_via_cv(
        self,
        n_splits,
        cv_n_sites,
        cv_random_state,
        leaf_grid,
        maxfeat_grid,
        n_estimators=500,
        bootstrap=False,
        min_samples_split=2,
        max_depth=None,
    ):
        rng = np.random.default_rng(cv_random_state)
        site_idx_all = np.arange(self.S)
        leaf_grid = np.asarray(list(dict.fromkeys(leaf_grid)), dtype=int)
        maxfeat_grid = list(dict.fromkeys(maxfeat_grid))
        scores = np.zeros((len(leaf_grid), len(maxfeat_grid)), dtype=float)

        for tr, val in TimeSeriesSplit(n_splits).split(self.X_train_full):
            n_sites = int(min(cv_n_sites, self.S))
            sites = rng.choice(site_idx_all, size=n_sites, replace=False)

            X_tr_full, Y_tr = self.X_train_full[tr], self.Y_train[tr][:, sites]
            X_val_full, Y_val = self.X_train_full[val], self.Y_train[val][:, sites]

            X_tr = X_tr_full[:, self.selected_cols_]
            X_val = X_val_full[:, self.selected_cols_]
            if self.standardize_X:
                x_mean = X_tr.mean(0, keepdims=True)
                x_std = np.clip(X_tr.std(0, keepdims=True), self.eps, None)
                X_tr_s = (X_tr - x_mean) / x_std
                X_val_s = (X_val - x_mean) / x_std
            else:
                X_tr_s, X_val_s = X_tr, X_val

            for i_leaf, leaf in enumerate(leaf_grid):
                for j_mf, mf in enumerate(maxfeat_grid):
                    Q_pred = np.full((X_val_s.shape[0], n_sites, len(self.taus)), np.nan, dtype=np.float32)
                    for j in range(n_sites):
                        y_tr_site = Y_tr[:, j]
                        try:
                            rf = _QRF(
                                n_estimators=n_estimators,
                                min_samples_leaf=int(leaf),
                                max_features=mf,
                                max_depth=max_depth,
                                bootstrap=bootstrap,
                                min_samples_split=min_samples_split,
                                random_state=12345,
                                n_jobs=-1,
                            )
                            rf.fit(X_tr_s, y_tr_site)
                            Q_pred[:, j, :] = rf.predict(X_val_s, quantiles=self.taus.tolist()).astype(np.float32)
                        except Exception:
                            pass
                    scores[i_leaf, j_mf] += self._cv_score(Y_val, Q_pred)

        mean_scores = scores / n_splits
        i_best, j_best = np.unravel_index(np.argmin(mean_scores), mean_scores.shape)
        return int(leaf_grid[i_best]), maxfeat_grid[j_best]

    # ─────────────────────────────────────────────────────────────
    # QRNN
    # ─────────────────────────────────────────────────────────────
    def _init_qrnn(self, kw):
        if QRNN is None or FullyConnected is None:
            raise ImportError("Provide QRNN and FullyConnected (PyTorch) for method='qrnn'.")

        do_cv = bool(kw.get("qrnn_select_hyperparams", True))
        device = kw.get("device", "cpu")
        batch_sz = int(kw.get("batch_size", 128))
        lr = float(kw.get("lr", 1e-3))

        if do_cv:
            best_layers, best_width, best_epochs = self._select_qrnn_hyperparams_via_cv(
                n_splits=kw.get("cv_splits", 5),
                cv_n_sites=kw.get("cv_n_sites", 10),
                cv_random_state=kw.get("cv_random_state", 42),
                layers_grid=kw.get("qrnn_layers_grid", [1, 2, 3]),
                width_grid=kw.get("qrnn_width_grid", [32, 64, 128]),
                epochs_grid=kw.get("qrnn_epochs_grid", [20, 40, 60]),
                batch_size=batch_sz,
                lr=lr,
                device=device,
            )
            n_layers = best_layers
            width = best_width
            n_epochs = best_epochs
        else:
            n_layers = int(kw.get("n_layers", 2))
            width = int(kw.get("width", 64))
            n_epochs = int(kw.get("n_epochs", 60))

        self.taus = np.linspace(0.01, 0.99, 100, dtype=np.float32)
        Xp = self._prepare_X(self.X_train_full)

        # Warm-up instantiate in main thread to finalize backend imports
        try:
            _net_warm = FullyConnected(
                self.d, len(self.taus), n_layers=n_layers, width=width, activation="sigmoid", batch_norm=False
            )
            _ = QRNN(quantiles=self.taus, model=_net_warm)
            del _net_warm
        except Exception:
            pass

        def _fit_qrnn_site(s):
            quantnn.set_default_backend("pytorch")
            net = FullyConnected(
                self.d,
                len(self.taus),
                n_layers=int(n_layers),
                width=int(width),
                activation="sigmoid",
                batch_norm=False,
            )
            qnn = QRNN(quantiles=self.taus, model=net)
            qnn.train((Xp, self.Y_train[:, s]), n_epochs=int(n_epochs), batch_size=int(batch_sz), device=str(device))
            if getattr(qnn, "model", None) is None:
                raise RuntimeError("QRNN training produced model=None (backend issue).")
            return qnn

        if self.n_jobs_sites > 1:
            from joblib import Parallel, delayed

            self.models = Parallel(
                n_jobs=self.n_jobs_sites, backend="threading", prefer="threads", require="sharedmem"
            )(delayed(_fit_qrnn_site)(s) for s in range(self.S))
            site_fit_parallel = True
        else:
            self.models = [_fit_qrnn_site(s) for s in range(self.S)]
            site_fit_parallel = False

        self.selected_hyperparams_ = {
            "method": "qrnn",
            "n_layers": int(n_layers),
            "width": int(width),
            "n_epochs": int(n_epochs),
            "batch_size": int(batch_sz),
            "lr": float(lr),
            "device": str(device),
            "standardize_X": bool(self.standardize_X),
            "site_fit_parallel": site_fit_parallel,
            "n_jobs_sites": int(self.n_jobs_sites),
        }

    def _select_qrnn_hyperparams_via_cv(
        self,
        n_splits,
        cv_n_sites,
        cv_random_state,
        layers_grid,
        width_grid,
        epochs_grid,
        batch_size=128,
        lr=1e-3,
        device="cpu",
    ):
        rng = np.random.default_rng(cv_random_state)
        site_idx_all = np.arange(self.S)
        layers_grid = np.asarray(list(dict.fromkeys(layers_grid)), dtype=int)
        width_grid = np.asarray(list(dict.fromkeys(width_grid)), dtype=int)
        epochs_grid = np.asarray(list(dict.fromkeys(epochs_grid)), dtype=int)
        scores = np.zeros((len(layers_grid), len(width_grid), len(epochs_grid)), dtype=float)

        for tr, val in TimeSeriesSplit(n_splits).split(self.X_train_full):
            n_sites = int(min(cv_n_sites, self.S))
            sites = rng.choice(site_idx_all, size=n_sites, replace=False)

            X_tr_full, Y_tr_sel = self.X_train_full[tr], self.Y_train[tr][:, sites]
            X_val_full, Y_val_sel = self.X_train_full[val], self.Y_train[val][:, sites]

            X_tr = X_tr_full[:, self.selected_cols_]
            X_val = X_val_full[:, self.selected_cols_]
            if self.standardize_X:
                x_mean = X_tr.mean(0, keepdims=True)
                x_std = np.clip(X_tr.std(0, keepdims=True), self.eps, None)
                X_tr_s = (X_tr - x_mean) / x_std
                X_val_s = (X_val - x_mean) / x_std
            else:
                X_tr_s, X_val_s = X_tr, X_val

            for iL, L in enumerate(layers_grid):
                for jW, W in enumerate(width_grid):
                    for kE, E in enumerate(epochs_grid):
                        Q_pred = np.full((X_val_s.shape[0], n_sites, len(self.taus)), np.nan, dtype=np.float32)
                        for j in range(n_sites):
                            y_tr_site = Y_tr_sel[:, j]
                            try:
                                net = FullyConnected(
                                    X_tr_s.shape[1],
                                    len(self.taus),
                                    n_layers=int(L),
                                    width=int(W),
                                    activation="sigmoid",
                                    batch_norm=False,
                                )
                                qnn = QRNN(quantiles=self.taus, model=net)
                                qnn.train(
                                    (X_tr_s, y_tr_site),
                                    n_epochs=int(E),
                                    batch_size=int(batch_size),
                                    device=str(device),
                                )
                                Q_pred[:, j, :] = qnn.predict(X_val_s).astype(np.float32)
                            except Exception:
                                pass
                        scores[iL, jW, kE] += self._cv_score(Y_val_sel, Q_pred)

        mean_scores = scores / n_splits
        iL, jW, kE = np.unravel_index(np.argmin(mean_scores), mean_scores.shape)
        return int(layers_grid[iL]), int(width_grid[jW]), int(epochs_grid[kE])

    # ─────────────────────────────────────────────────────────────
    # Common quantiles/CDF + KNN CV
    # ─────────────────────────────────────────────────────────────
    def _cv_quantiles_from_weights(self, nb, w):
        Nv, k, S = nb.shape
        m = len(self.taus)
        Q = np.full((Nv, S, m), np.nan, dtype=np.float32)
        for s in range(S):
            y = nb[..., s]
            msk = np.isfinite(y)
            ww = w * msk
            ww, safe_rows = self._row_normalize_after_mask(ww)
            order = np.argsort(y, axis=1)
            ys = np.take_along_axis(y, order, axis=1)
            ws = np.take_along_axis(ww, order, axis=1)
            cdf = np.cumsum(ws, axis=1)
            for n in range(Nv):
                if not safe_rows[n]:
                    continue
                Q[n, s] = np.interp(
                    self.taus, cdf[n], ys[n], left=ys[n, 0] - self.eps, right=ys[n, -1] + self.eps
                )
        return Q

    def _cv_score(self, Y_true, Q):
        Y_true = np.asarray(Y_true, np.float32)  # (Nv,S_subset)
        mask = np.isfinite(Y_true)[..., None]  # (Nv,S_subset,1)
        diff = (Y_true[..., None] - Q)  # (Nv,S_subset,m)
        pin = np.where(diff >= 0, self.taus * diff, (self.taus - 1) * diff)
        pin = np.where(mask, pin, 0.0)
        denom = max(int(mask.sum()) * len(self.taus), 1)
        return float(pin.sum() / denom)

    def _select_kh_via_cv(self, k_grid, n_splits, cv_n_sites=10, cv_random_state=42, c_grid=None, h_grid=None):
        k_grid = np.asarray(sorted(set(k_grid)), int)
        if k_grid.size == 0:
            raise ValueError("k_grid is empty.")
        if not self._kernel_uses_bandwidth:
            return self._select_k_via_cv(k_grid, n_splits, cv_n_sites, cv_random_state), self.h_mode

        if c_grid is None and h_grid is None:
            if (isinstance(self.h_mode, tuple) and self.h_mode[0] == "adaptive") or (self.h_mode == "adaptive"):
                c_grid = np.array([1.0], float)
            else:
                h_grid = np.array([float(self.h_mode)], float)

        use_adaptive = c_grid is not None
        rng = np.random.default_rng(cv_random_state)
        site_idx_all = np.arange(self.S)
        scores = np.zeros((k_grid.size, len(c_grid)), float) if use_adaptive else np.zeros((k_grid.size, len(h_grid)), float)

        for tr, val in TimeSeriesSplit(n_splits).split(self.X_train_full):
            n_sites = int(min(cv_n_sites, self.S))
            sites = rng.choice(site_idx_all, size=n_sites, replace=False)

            X_tr_full, Y_tr = self.X_train_full[tr], self.Y_train[tr][:, sites]
            X_val_full, Y_val = self.X_train_full[val], self.Y_train[val][:, sites]

            X_tr = X_tr_full[:, self.selected_cols_]
            X_val = X_val_full[:, self.selected_cols_]
            if self.standardize_X:
                x_mean = X_tr.mean(0, keepdims=True)
                x_std = np.clip(X_tr.std(0, keepdims=True), self.eps, None)
                X_tr_s = (X_tr - x_mean) / x_std
                X_val_s = (X_val - x_mean) / x_std
            else:
                X_tr_s, X_val_s = X_tr, X_val

            tree = KDTree(X_tr_s)
            k_max = int(k_grid.max())
            dist_all, idx_all = tree.query(X_val_s, k=k_max)
            d_k_cols = [dist_all[:, k - 1][:, None] for k in k_grid]

            for ik, k in enumerate(k_grid):
                d_k = dist_all[:, :k]
                nb = Y_tr[idx_all[:, :k]]
                if use_adaptive:
                    base_h = np.maximum(d_k_cols[ik], self.eps)
                    for jc, c in enumerate(np.asarray(c_grid, float)):
                        h = np.maximum(c, self.eps) * base_h
                        w = np.maximum(self.kernel(d_k / h), 1e-12)
                        w, _ = self._row_normalize_after_mask(w)
                        Q = self._cv_quantiles_from_weights(nb, w)
                        scores[ik, jc] += self._cv_score(Y_val, Q)
                else:
                    for jh, hval in enumerate(np.asarray(h_grid, float)):
                        h = np.full((d_k.shape[0], 1), max(float(hval), self.eps), dtype=d_k.dtype)
                        w = np.maximum(self.kernel(d_k / h), 1e-12)
                        w, _ = self._row_normalize_after_mask(w)
                        Q = self._cv_quantiles_from_weights(nb, w)
                        scores[ik, jh] += self._cv_score(Y_val, Q)

        mean_scores = scores / n_splits
        if use_adaptive:
            ik, jc = np.unravel_index(np.argmin(mean_scores), mean_scores.shape)
            return int(k_grid[ik]), ("adaptive", float(np.asarray(c_grid, float)[jc]))
        ik, jh = np.unravel_index(np.argmin(mean_scores), mean_scores.shape)
        return int(k_grid[ik]), float(np.asarray(h_grid, float)[jh])

    def _select_k_via_cv(self, k_grid, n_splits, cv_n_sites=10, cv_random_state=42):
        scores = np.zeros(len(k_grid), float)
        rng = np.random.default_rng(cv_random_state)
        site_idx_all = np.arange(self.S)

        for tr, val in TimeSeriesSplit(n_splits).split(self.X_train_full):
            n_sites = int(min(cv_n_sites, self.S))
            sites = rng.choice(site_idx_all, size=n_sites, replace=False)

            X_tr_full, Y_tr = self.X_train_full[tr], self.Y_train[tr][:, sites]
            X_val_full, Y_val = self.X_train_full[val], self.Y_train[val][:, sites]

            X_tr = X_tr_full[:, self.selected_cols_]
            X_val = X_val_full[:, self.selected_cols_]
            if self.standardize_X:
                x_mean = X_tr.mean(0, keepdims=True)
                x_std = np.clip(X_tr.std(0, keepdims=True), self.eps, None)
                X_tr_s = (X_tr - x_mean) / x_std
                X_val_s = (X_val - x_mean) / x_std
            else:
                X_tr_s, X_val_s = X_tr, X_val

            tree = KDTree(X_tr_s)
            k_max = int(np.max(k_grid))
            dist, idx = tree.query(X_val_s, k=k_max)

            for i, k in enumerate(k_grid):
                d = dist[:, :k]
                nb = Y_tr[idx[:, :k]]
                w = np.maximum(self.kernel(d), 1e-12)
                w, _ = self._row_normalize_after_mask(w)
                Q = self._cv_quantiles_from_weights(nb, w)
                scores[i] += self._cv_score(Y_val, Q)

        mean_scores = scores / n_splits
        return int(k_grid[np.argmin(mean_scores)])

    # ─────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────
    def _quant_all_sites(self, X):
        X = np.asarray(X, np.float32)
        if self.method == "knn":
            taus = self.taus
            Nq, S, m = X.shape[0], self.S, len(taus)
            Q = np.full((Nq, S, m), np.nan, dtype=np.float32)
            for s0 in range(0, S, min(S, 256)):
                s1 = min(self.S, s0 + min(self.S, 256))
                _, w, Y_nb, valid = self._knn_neighbors(X, s0, s1)
                for sb in range(s1 - s0):
                    y = Y_nb[..., sb]
                    msk = valid[..., sb]
                    ww = w * msk
                    ww, safe_rows = self._row_normalize_after_mask(ww)
                    order = np.argsort(y, axis=1)
                    y_sorted = np.take_along_axis(y, order, axis=1)
                    w_sorted = np.take_along_axis(ww, order, axis=1)
                    cdf = np.cumsum(w_sorted, axis=1)
                    for n in range(Nq):
                        if not safe_rows[n]:
                            continue
                        Q[n, s0 + sb] = np.interp(
                            taus,
                            cdf[n],
                            y_sorted[n],
                            left=y_sorted[n, 0] - self.eps,
                            right=y_sorted[n, -1] + self.eps,
                        )
            return Q

        Xp = self._prepare_X(X)
        Nq, m = Xp.shape[0], len(self.taus)
        Q = np.empty((Nq, self.S, m), np.float32)
        if self.method == "qrf":
            tau_list = self.taus.tolist()
            for s, rf in enumerate(self.models):
                Q[:, s] = rf.predict(Xp, quantiles=tau_list).astype(np.float32)
        else:  # qrnn
            for s, qnn in enumerate(self.models):
                pred = qnn.predict(Xp.astype(np.float32))
                Q[:, s] = to_np32(pred)
        return Q

    def predict_quantiles(self, X_new):
        return self._quant_all_sites(X_new)

    def predict_cdf(self, X_new, Y_eval):
        if self.method == "knn":
            return self._knn_predict_cdf(X_new, Y_eval)

        X_new = np.asarray(X_new, np.float32)
        Y_eval = np.asarray(Y_eval, np.float32)
        Q = self._quant_all_sites(X_new)  # (Nq,S,m)
        ycdf = np.concatenate(([0.0], self.taus, [1.0]))
        Nq, S, _ = Q.shape
        U = np.full((Nq, S), np.nan, dtype=np.float32)
        for s in range(S):
            xcdf = np.concatenate([Q[:, s, 0:1] - self.eps, Q[:, s], Q[:, s, -1:] + self.eps], 1)
            for n in range(Nq):
                y_th = Y_eval[n, s]
                if not np.isfinite(y_th):
                    continue
                U[n, s] = np.interp(y_th, xcdf[n], ycdf).astype(np.float32)
        finite = np.isfinite(U)
        if np.any(finite):
            U[finite] = np.clip(U[finite], self.eps, 1.0 - self.eps)
        return U

    def y_to_z(self, X_new, Y):
        X_new, Y = np.asarray(X_new, np.float32), np.asarray(Y, np.float32)
        U = self.predict_cdf(X_new, Y)
        Z = np.full_like(U, np.nan, dtype=np.float32)
        finite = np.isfinite(U)
        if np.any(finite):
            Z[finite] = norm.ppf(np.clip(U[finite], self.eps, 1.0 - self.eps)).astype(np.float32)
        return Z

    def z_to_y(self, X_new, Z):
        X_new, Z = np.asarray(X_new, np.float32), np.asarray(Z, np.float32)
        U_all = np.full_like(Z, np.nan, dtype=np.float32)
        finite_z = np.isfinite(Z)
        if np.any(finite_z):
            U_all[finite_z] = norm.cdf(Z[finite_z]).astype(np.float32)
            U_all[finite_z] = np.clip(U_all[finite_z], self.eps, 1.0 - self.eps)

        if self.method != "knn":
            Q = self._quant_all_sites(X_new)
            ycdf = np.concatenate(([0.0], self.taus, [1.0]))
            Y_hat = np.full_like(Z, np.nan, dtype=np.float32)
            for s in range(self.S):
                xcdf = np.concatenate([Q[:, s, 0:1] - self.eps, Q[:, s], Q[:, s, -1:] + self.eps], 1)
                for n in range(Z.shape[0]):
                    u = U_all[n, s]
                    if not np.isfinite(u):
                        continue
                    Y_hat[n, s] = np.interp(u, ycdf, xcdf[n]).astype(np.float32)
            return Y_hat

        # KNN inverse ECDF
        Nq = X_new.shape[0]
        Y_hat = np.full_like(Z, np.nan, dtype=np.float32)
        s_batch = min(self.S, 256)
        for s0 in range(0, self.S, s_batch):
            s1 = min(self.S, s0 + s_batch)
            _, w, Y_nb, valid = self._knn_neighbors(X_new, s0, s1)
            for sb in range(s1 - s0):
                y = Y_nb[..., sb]
                msk = valid[..., sb]
                ww = w * msk
                ww, safe_rows = self._row_normalize_after_mask(ww)
                order = np.argsort(y, axis=1)
                y_sorted = np.take_along_axis(y, order, axis=1)
                w_sorted = np.take_along_axis(ww, order, axis=1)
                cdf = np.cumsum(w_sorted, axis=1)
                for n in range(Nq):
                    u = U_all[n, s0 + sb]
                    if not np.isfinite(u) or not safe_rows[n]:
                        continue
                    if u <= cdf[n, 0]:
                        Y_hat[n, s0 + sb] = y_sorted[n, 0]
                        continue
                    if u >= cdf[n, -1]:
                        Y_hat[n, s0 + sb] = y_sorted[n, -1]
                        continue
                    i = np.searchsorted(cdf[n], u, side="right")
                    lo = i - 1
                    hi = i
                    c_lo, c_hi = cdf[n, lo], cdf[n, hi]
                    y_lo, y_hi = y_sorted[n, lo], y_sorted[n, hi]
                    frac = (u - c_lo) / max(c_hi - c_lo, self.eps)
                    Y_hat[n, s0 + sb] = y_lo + frac * (y_hi - y_lo)
        return Y_hat
