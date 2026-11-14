import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
import pandas as pd


def fit_ellipse_subset(points):
    """Fit the normalized quadratic ellipse through five points."""
    points = np.asarray(points, dtype=float)
    if points.shape != (5, 2):
        raise ValueError("Exactly five points with shape (5, 2) are required.")

    x = points[:, 0]
    y = points[:, 1]
    # The five rows provide the five equations needed for the ellipse coefficients.
    design = np.column_stack((x ** 2, 2 * x * y, y ** 2, -2 * x, -2 * y))

    if np.linalg.matrix_rank(design) < 5:
        raise ValueError("The five points do not uniquely determine an ellipse.")

    A, B, C, D, E = np.linalg.solve(design, -np.ones(5))
    quadratic = np.array([[A, B], [B, C]])
    # The centre satisfies Qb = [D, E], before the quadratic is normalized.
    center = np.linalg.solve(quadratic, np.array([D, E]))
    # Scale the quadratic so that the ellipse has the form (x-b)^T Q (x-b)=1.
    alpha = 1 / (np.dot(np.array([D, E]), center) - 1)
    Q = alpha * quadratic
    if not np.all(np.linalg.eigvalsh(Q) > 0):
        raise ValueError("The fitted quadratic is not a positive definite ellipse.")

    b = center
    return alpha, Q, b, (A, B, C, D, E)


points = np.array([
    (2.92, -6.01),
    (3.40, -7.20),
    (4.99, -7.84),
    (5.48, -7.04),
    (4.20, -5.91),
])

alpha, Q, b, coefficients = fit_ellipse_subset(points)
print("alpha =", alpha)
print("Q =\n", Q)
print("b =", b)
print("(A, B, C, D, E) =", coefficients)
