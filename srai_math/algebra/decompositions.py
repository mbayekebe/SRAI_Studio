"""Singular value decomposition, PCA, and low-rank utilities."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .matrices import as_matrix


def svd_decomposition(
    a: ArrayLike,
    full_matrices: bool = False,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return U, singular values, and V^T."""
    A = as_matrix(a)
    return np.linalg.svd(A, full_matrices=full_matrices)


def reconstruct_from_svd(
    u: ArrayLike,
    singular_values: ArrayLike,
    vt: ArrayLike,
) -> NDArray[np.float64]:
    """Reconstruct a matrix from its SVD factors."""
    U = as_matrix(u)
    s = np.asarray(singular_values, dtype=float)
    Vt = as_matrix(vt)
    if s.ndim != 1:
        raise ValueError("singular_values must be one-dimensional.")
    if U.shape[1] != s.shape[0] or Vt.shape[0] != s.shape[0]:
        raise ValueError("SVD factor dimensions do not agree.")
    return U @ np.diag(s) @ Vt


def rank_k_approximation(a: ArrayLike, k: int) -> NDArray[np.float64]:
    """Return the best rank-k approximation in Frobenius norm."""
    A = as_matrix(a)
    if not isinstance(k, int) or k < 0 or k > min(A.shape):
        raise ValueError("k must be an integer between 0 and min(A.shape).")
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    return U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]


def explained_variance_ratio_from_singular_values(
    singular_values: ArrayLike,
) -> NDArray[np.float64]:
    """Return variance proportions associated with singular values."""
    s = np.asarray(singular_values, dtype=float)
    if s.ndim != 1:
        raise ValueError("singular_values must be one-dimensional.")
    energy = s**2
    total = float(np.sum(energy))
    if np.isclose(total, 0.0):
        return np.zeros_like(energy)
    return energy / total


def pca_fit_transform(
    x: ArrayLike,
    n_components: int | None = None,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Center X and return scores, components, explained ratios, and mean."""
    X = as_matrix(x)
    n_samples, n_features = X.shape
    max_components = min(n_samples, n_features)
    if n_components is None:
        n_components = max_components
    if not isinstance(n_components, int) or not 1 <= n_components <= max_components:
        raise ValueError("n_components is out of range.")

    mean = X.mean(axis=0)
    Xc = X - mean
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    scores = U[:, :n_components] * s[:n_components]
    components = Vt[:n_components, :]
    explained = explained_variance_ratio_from_singular_values(s)[:n_components]
    return scores, components, explained, mean


def pca_inverse_transform(
    scores: ArrayLike,
    components: ArrayLike,
    mean: ArrayLike,
) -> NDArray[np.float64]:
    """Reconstruct observations from PCA scores."""
    Z = as_matrix(scores)
    C = as_matrix(components)
    mu = np.asarray(mean, dtype=float)
    if mu.ndim != 1 or C.shape[1] != mu.shape[0]:
        raise ValueError("Mean and component dimensions do not agree.")
    if Z.shape[1] != C.shape[0]:
        raise ValueError("Score and component dimensions do not agree.")
    return Z @ C + mu


def frobenius_error(a: ArrayLike, approximation: ArrayLike) -> float:
    """Return Frobenius reconstruction error."""
    A = as_matrix(a)
    B = as_matrix(approximation)
    if A.shape != B.shape:
        raise ValueError("Matrices must have identical shapes.")
    return float(np.linalg.norm(A - B, ord="fro"))
