import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb


def bernstein_basis_all(n: int, x: float) -> np.ndarray:
    """Return all Bernstein basis values of order n evaluated at x.

    B_{n,k}(x) = C(n,k) x^k (1-x)^{n-k} for k=0..n
    """
    k_values = np.arange(n + 1)
    binomial_coefficients = comb(n, k_values)
    powers_of_x = x ** k_values
    powers_of_one_minus_x = (1 - x) ** (n - k_values)
    return binomial_coefficients * powers_of_x * powers_of_one_minus_x


def bernstein_polynomial_fast(function_values: np.ndarray, basis_at_x: np.ndarray) -> float:
    """Evaluate the Bernstein polynomial at x given precomputed function values on the grid and basis at x."""
    return float(np.dot(function_values, basis_at_x))


def Bn_1_fast(f, n: int, x: float) -> float:
    grid = np.linspace(0.0, 1.0, n + 1)
    function_values = f(grid)
    basis_at_x = bernstein_basis_all(n, x)
    return bernstein_polynomial_fast(function_values, basis_at_x)


def Bn_2_fast(f, n: int, x: float) -> float:
    grid = np.linspace(0.0, 1.0, n + 1)
    function_values = f(grid)
    basis_at_x = bernstein_basis_all(n, x)
    Bn1_over_grid = np.array([
        bernstein_polynomial_fast(function_values, bernstein_basis_all(n, t))
        for t in grid
    ])
    return bernstein_polynomial_fast(Bn1_over_grid, basis_at_x)


def Bn_3_fast(f, n: int, x: float) -> float:
    grid = np.linspace(0.0, 1.0, n + 1)
    function_values = f(grid)
    basis_at_x = bernstein_basis_all(n, x)
    Bn1_over_grid = np.array([
        bernstein_polynomial_fast(function_values, bernstein_basis_all(n, t))
        for t in grid
    ])
    Bn2_over_grid = np.array([
        bernstein_polynomial_fast(Bn1_over_grid, bernstein_basis_all(n, t))
        for t in grid
    ])
    return bernstein_polynomial_fast(Bn2_over_grid, basis_at_x)


def Bn_4_fast(f, n: int, x: float) -> float:
    grid = np.linspace(0.0, 1.0, n + 1)
    function_values = f(grid)
    basis_at_x = bernstein_basis_all(n, x)
    Bn1_over_grid = np.array([
        bernstein_polynomial_fast(function_values, bernstein_basis_all(n, t))
        for t in grid
    ])
    Bn2_over_grid = np.array([
        bernstein_polynomial_fast(Bn1_over_grid, bernstein_basis_all(n, t))
        for t in grid
    ])
    Bn3_over_grid = np.array([
        bernstein_polynomial_fast(Bn2_over_grid, bernstein_basis_all(n, t))
        for t in grid
    ])
    return bernstein_polynomial_fast(Bn3_over_grid, basis_at_x)


def l_function(alpha_star: float):
    """Return the function l(x) = (alpha * x) / (alpha * x + 1 - x)."""
    def l(x: np.ndarray | float) -> np.ndarray | float:
        numerator = alpha_star * x
        denominator = (alpha_star * x) + (1.0 - x)
        return numerator / denominator
    return l


def run_experiment(q: float, alpha_star: float, output_path: str) -> None:
    """Run the Bernstein bias experiments for given q and alpha, saving a log-log plot."""
    l = l_function(alpha_star)

    n_values = np.arange(10, 1001, 100)
    results_f1: list[float] = []
    results_f2: list[float] = []
    results_f3: list[float] = []
    results_f4: list[float] = []

    for n in n_values:
        Bn_1_value = Bn_1_fast(l, n, q)
        Bn_2_value = Bn_2_fast(l, n, q)
        Bn_3_value = Bn_3_fast(l, n, q)
        Bn_4_value = Bn_4_fast(l, n, q)

        baseline = l(q)
        f1 = Bn_1_value - baseline
        f2 = (2 * Bn_1_value) - Bn_2_value - baseline
        f3 = (3 * Bn_1_value) - (3 * Bn_2_value) + Bn_3_value - baseline
        f4 = (4 * Bn_1_value) - (6 * Bn_2_value) + (4 * Bn_3_value) - Bn_4_value - baseline

        results_f1.append(f1)
        results_f2.append(f2)
        results_f3.append(f3)
        results_f4.append(f4)

    plt.figure(figsize=(8, 6))
    plt.loglog(n_values, np.abs(np.array(results_f1)), marker="o", markersize=4, label=r"$\mathbb{E}_{X^n}\{D_{n,1}g(T/n)\}-g(q)$")
    plt.loglog(n_values, np.abs(np.array(results_f2)), marker="o", markersize=4, label=r"$\mathbb{E}_{X^n}\{D_{n,2}g(T/n)\}-g(q)$")
    plt.loglog(n_values, np.abs(np.array(results_f3)), marker="o", markersize=4, label=r"$\mathbb{E}_{X^n}\{D_{n,3}g(T/n)\}-g(q)$")
    plt.loglog(n_values, np.abs(np.array(results_f4)), marker="o", markersize=4, label=r"$\mathbb{E}_{X^n}\{D_{n,4}g(T/n)\}-g(q)$")

    # Reference slopes
    plt.loglog(n_values, n_values ** (-1.0), linestyle="--", label=r"$\mathcal{O}(n^{-1})$ Reference")
    plt.loglog(n_values, n_values ** (-2.0), linestyle="--", label=r"$\mathcal{O}(n^{-2})$ Reference")
    plt.loglog(n_values, n_values ** (-3.0), linestyle="--", label=r"$\mathcal{O}(n^{-3})$ Reference")
    plt.loglog(n_values, n_values ** (-4.0), linestyle="--", label=r"$\mathcal{O}(n^{-4})$ Reference")

    plt.xlabel("n (log scale)")
    plt.ylabel("Bias (log scale)")
    plt.legend(loc="best", fontsize=9, markerscale=1, frameon=True, framealpha=0.7)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)

    os.makedirs(os.path.dirname(output_path), exist_ok=True) if os.path.dirname(output_path) else None
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    # Experiment 1: q = 2/5, y* = 2, alpha* = exp(y* - 1/2)
    q1 = 2.0 / 5.0
    y_star_1 = 2.0
    alpha_star_1 = float(np.exp(y_star_1 - 0.5))
    output_1 = os.path.join("results", "convergence_q_2_5_y_2_k.pdf")
    run_experiment(q1, alpha_star_1, output_1)

    # Experiment 2: q = 3/11, y* = 1, alpha* = exp(4*y* - 2)
    q2 = 3.0 / 11.0
    y_star_2 = 1.0
    alpha_star_2 = float(np.exp(4.0 * y_star_2 - 2.0))
    output_2 = os.path.join("results", "convergence_q_3_11_y_1_k.pdf")
    run_experiment(q2, alpha_star_2, output_2)

    print(f"Saved plots to: {output_1} and {output_2}")


