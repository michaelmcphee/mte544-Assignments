"""
Student Name & Last Name: Michael McPhee
Origianl Author : Pi Thanacha Choopojcharoen
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
import random
import pandas as pd
from pathlib import Path

def plot_ellipse(S, b, ax):
    eigvals, eigvecs = np.linalg.eigh(S)
    if np.any(eigvals <= 0):
        raise ValueError("The ellipse matrix must be positive definite.")
    radii = 1 / np.sqrt(eigvals)

    theta = np.linspace(0, 2 * np.pi, 100)
    unit_circle = np.array([np.cos(theta), np.sin(theta)])
    ellipse_param = radii[:, np.newaxis] * unit_circle
    ellipse_points = eigvecs @ ellipse_param + b[:, np.newaxis]

    ax.plot(ellipse_points[0, :], ellipse_points[1, :], 'b-', label='Fitted Ellipse')
    ax.plot(b[0], b[1], 'go', label='Center')
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend(loc='best')
    ax.grid(True)

def visualize_data(p, ax, inliers, threshold):
    ax.scatter(p[:, 0], p[:, 1], color='red', alpha=0.5, label='Raw Measurements (Ellipse)')
    ax.scatter(inliers[:, 0], inliers[:, 1], color='purple', alpha=0.7, label='Inliers')

    for point in inliers:
        circle = plt.Circle(point, threshold, color='orange', fill=False, linestyle='--', alpha=0.7)
        ax.add_patch(circle)

def is_positive_definite(matrix):
    # A Cholesky factor exists only for a symmetric positive-definite matrix.
    try:
        np.linalg.cholesky(matrix)
        return True
    except np.linalg.LinAlgError:
        return False

def fit_ellipse_subset(points):
    # Five points are the minimum needed to solve for A, B, C, D, and E.
    points = np.asarray(points, dtype=float)

    x = points[:, 0]
    y = points[:, 1]
    # Each sampled point contributes one equation to the normalized ellipse.
    design = np.column_stack((x ** 2, 2 * x * y, y ** 2, -2 * x, -2 * y))

    A, B, C, D, E = np.linalg.solve(design, -np.ones(5))
    quadratic = np.array([[A, B], [B, C]])
    # Solve for the centre, then scale the quadratic to obtain
    # (x - b)^T Q (x - b) = 1.
    b = np.linalg.solve(quadratic, np.array([D, E]))
    alpha = 1 / (np.dot(np.array([D, E]), b) - 1)
    Q = alpha * quadratic
    if not is_positive_definite(Q):
        raise ValueError("The fitted quadratic is not a positive definite ellipse.")

    return Q, b

def ransac_ellipse(data, num_iterations=1000, threshold=0.2):
    # Given the data sets, perform RANSAC to find the best Q and b as well as the inliers
    # Hint: You should use fit_ellipse_subset 
    # Hint: in some case, the Q matrix might not be positive defintie, use is_positive_definite to check.
    data = np.asarray(data, dtype=float)

    best_inliers = np.empty((0, 2))
    best_Q = None
    best_b = None

    for _ in range(num_iterations):
        # A new five-point sample gives one candidate ellipse per iteration.
        subset = data[np.random.choice(data.shape[0], 5, replace=False)]
        try:
            candidate_Q, candidate_b = fit_ellipse_subset(subset)
        except (ValueError, np.linalg.LinAlgError):
            continue

        offsets = data - candidate_b
        quadratic_values = np.einsum("ni,ij,nj->n", offsets, candidate_Q, offsets)

        # Reject degenerate fits whose center is clearly outside the dataset.
        if np.any(candidate_b < data.min(axis=0)) or np.any(candidate_b > data.max(axis=0)):
            continue
        # Points close to the ellipse have sqrt((p-b)^T Q (p-b)) close to one.
        residuals = np.abs(np.sqrt(quadratic_values) - 1)
        candidate_inliers = data[residuals <= threshold]

        # Keep the model that explains the largest number of measurements.
        if candidate_inliers.shape[0] > best_inliers.shape[0]:
            best_Q = candidate_Q
            best_b = candidate_b
            best_inliers = candidate_inliers

    if best_Q is None:
        raise RuntimeError("RANSAC could not find a positive definite ellipse.")
    return best_Q, best_b, best_inliers

def run_parameter_experiments(all_data):
    """Compare the effects of dataset size, iterations, and threshold."""
    experiments = [
        ("dataset size", 100, 1000, 0.2),
        ("dataset size", 500, 1000, 0.2),
        ("dataset size", 1000, 1000, 0.2),
        ("iterations", 500, 100, 0.2),
        ("iterations", 500, 1000, 0.2),
        ("iterations", 500, 5000, 0.2),
        ("threshold", 500, 1000, 0.05),
        ("threshold", 500, 1000, 0.2),
        ("threshold", 500, 1000, 0.5),
    ]

    print("\nRANSAC parameter experiments")
    print("Parameter       Dataset  Iterations  Threshold  Inliers")
    for parameter, dataset_size, iterations, threshold in experiments:
        # Reset the seeds so that rows can be compared across experiments.
        rng = np.random.default_rng(7)
        dataset = all_data[rng.choice(all_data.shape[0], dataset_size, replace=False)]
        np.random.seed(7)
        _, _, inliers = ransac_ellipse(dataset, iterations, threshold)
        print(
            f"{parameter:<15} {dataset_size:>7}  {iterations:>10}  "
            f"{threshold:>9.2f}  {inliers.shape[0]:>7}"
        )

if __name__ == "__main__":
    # Load the measurements and select the requested dataset size.
    N = 500
    all_data = pd.read_csv('data_x_y.csv').to_numpy()
    dataset = all_data[np.random.choice(all_data.shape[0], N, replace=False), :]
    
    Q, b_est, inliers = ransac_ellipse(dataset)
    run_parameter_experiments(all_data)
    
    # Plot the raw measurements and fitted ellipse
    fig, ax1 = plt.subplots(1, 1, figsize=(16, 8))
    visualize_data(dataset, ax1, inliers, threshold=0.2)
    plot_ellipse(Q, b_est, ax1)
    ax1.set_title("RANSAC Ellipse Fitting with Threshold Visualization")

    images_dir = Path(__file__).resolve().parent / "images"
    images_dir.mkdir(exist_ok=True)
    fig.savefig(images_dir / "ransac_result.png", dpi=300, bbox_inches="tight")
    plt.show()