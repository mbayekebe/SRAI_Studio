import numpy as np


def as_matrix(a):
    m=np.asarray(a,dtype=float)
    if m.ndim!=2: raise ValueError("Expected a two-dimensional matrix.")
    return m
def rank(a,tol=None): return int(np.linalg.matrix_rank(as_matrix(a),tol=tol))


def matrix_add(a, b):
    a, b = as_matrix(a), as_matrix(b)
    if a.shape != b.shape:
        raise ValueError("Matrices must have identical shapes.")
    return a + b


def matrix_multiply(a, b):
    a, b = as_matrix(a), as_matrix(b)
    if a.shape[1] != b.shape[0]:
        raise ValueError("Inner matrix dimensions must agree.")
    return a @ b


def transpose(a):
    return as_matrix(a).T


def trace(a):
    matrix = as_matrix(a)
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Trace requires a square matrix.")
    return float(np.trace(matrix))


def determinant_2x2(a):
    matrix = as_matrix(a)
    if matrix.shape != (2, 2):
        raise ValueError("A 2x2 matrix is required.")
    return float(matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0])


def gaussian_elimination(a, b):
    matrix = as_matrix(a)
    vector = np.asarray(b, dtype=float)
    if matrix.shape[0] != matrix.shape[1] or vector.shape != (matrix.shape[0],):
        raise ValueError("Require a square matrix and a conformable vector.")
    try:
        return np.linalg.solve(matrix, vector)
    except np.linalg.LinAlgError as error:
        raise ValueError("Matrix is singular.") from error


def residual(a, x, b):
    return np.asarray(b, dtype=float) - as_matrix(a) @ np.asarray(x, dtype=float)


def condition_number(a):
    return float(np.linalg.cond(as_matrix(a)))
