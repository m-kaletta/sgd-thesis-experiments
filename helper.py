import numpy as np


def base_vec(n_dim: int, basis_idx: int):
    basis_vec = np.zeros(n_dim)
    basis_vec[basis_idx] = 1.0
    return basis_vec


# diagonal of val and upper off-diagonal of 1: hence a single jordan-block
def mat_jordan_block(n_dim: int, val: float=1):
    diag = np.identity(n_dim)
    superdiagonal = np.roll(diag, 1)
    superdiagonal[0, 0] = 0
    return val*diag + superdiagonal


# diagonal of 2 and off-diagonals of -1
def mat_tridiag(n_dim: int):
    diag = np.identity(n_dim)
    superdiagonal = np.roll(diag, 1)
    superdiagonal[0, 0] = 0
    subdiagonal = np.roll(diag, -1)
    subdiagonal[-1, -1] = 0
    return 2 * diag - superdiagonal - subdiagonal


def mat_extreme_spectrum(n_dim: int, base: float=2):
    # base^(-i) for i even, base^i if uneven
    eigenvals = [base**(((i % 2) * 2 - 1) * i) for i in range(1, n_dim + 1)]
    return np.diag(eigenvals)
