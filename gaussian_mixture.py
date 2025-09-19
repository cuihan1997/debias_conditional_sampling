import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.mixture import GaussianMixture
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp


def compute_true_value(y_star: float, sigma2: float) -> float:
    total_var = 1.0 + sigma2
    phi = lambda x: norm.pdf(x, loc=0.0, scale=np.sqrt(total_var))

    tau2 = sigma2 / (sigma2 + 1.0)
    tau = np.sqrt(tau2)

    mu1 = y_star / (sigma2 + 1.0)
    mu2 = (y_star + sigma2) / (sigma2 + 1.0)

    tail1 = 1.0 - norm.cdf(0.5, loc=mu1, scale=tau)
    tail2 = 1.0 - norm.cdf(0.5, loc=mu2, scale=tau)

    denominator = 0.5 * phi(y_star) + 0.5 * phi(y_star - 1.0)
    numerator = 0.5 * phi(y_star) * tail1 + 0.5 * phi(y_star - 1.0) * tail2
    return float(numerator / denominator)


def run_mc_estimator(n: int, random_seed: int, y_star: float, sigma2: float) -> float:
    gmm = GaussianMixture(n_components=2, random_state=random_seed)
    gmm.weights_ = np.array([0.5, 0.5])
    gmm.means_ = np.array([[0.0], [1.0]])
    gmm.covariances_ = np.array([[[1.0]], [[1.0]]])
    gmm.precisions_cholesky_ = np.array([[[1.0]], [[1.0]]])

    samples, _ = gmm.sample(n)
    x = samples[:, 0]

    # Compute likelihood directly
    std = np.sqrt(sigma2)
    weights = norm.pdf(y_star - x, loc=0.0, scale=std)
    
    numerator = np.sum(weights[x >= 0.5])
    denominator = np.sum(weights)
    return float(numerator / denominator)


def run_bias_corrected_estimator(n: int, random_seed: int, y_star: float, sigma2: float) -> float:
    gmm = GaussianMixture(n_components=2, random_state=random_seed)
    gmm.weights_ = np.array([0.5, 0.5])
    gmm.means_ = np.array([[0.0], [1.0]])
    gmm.covariances_ = np.array([[[1.0]], [[1.0]]])
    gmm.precisions_cholesky_ = np.array([[[1.0]], [[1.0]]])

    samples, _ = gmm.sample(n)
    x = samples[:, 0]

    # Bootstrap resample
    np.random.seed(random_seed)
    x_resample = np.random.choice(x, size=n, replace=True)

    # Compute likelihoods directly
    std = np.sqrt(sigma2)
    weights = norm.pdf(y_star - x, loc=0.0, scale=std)
    weights_resample = norm.pdf(y_star - x_resample, loc=0.0, scale=std)

    num = np.sum(weights[x >= 0.5])
    den = np.sum(weights)
    f_hat = float(num / den)

    num_res = np.sum(weights_resample[x_resample >= 0.5])
    den_res = np.sum(weights_resample)
    f_hat_res = float(num_res / den_res)

    return float(2.0 * f_hat - f_hat_res)


def run_mc_batch(args):
    """Run a batch of Monte Carlo replicates for multiprocessing."""
    n, start_seed, batch_size, y_star, sigma2, bias_corrected = args
    
    if bias_corrected:
        estimates = [run_bias_corrected_estimator(n, start_seed + i, y_star, sigma2) 
                    for i in range(batch_size)]
    else:
        estimates = [run_mc_estimator(n, start_seed + i, y_star, sigma2) 
                    for i in range(batch_size)]
    
    return np.array(estimates)


def run_parallel_mc(n: int, B: int, y_star: float, sigma2: float, bias_corrected: bool, 
                   n_cores: int = None) -> tuple[float, float]:
    """Run Monte Carlo estimation in parallel across multiple cores."""
    if n_cores is None:
        n_cores = mp.cpu_count()
    
    # Split work into batches
    batch_size = max(1, B // (n_cores * 4))  # 4 batches per core for good load balancing
    n_batches = (B + batch_size - 1) // batch_size
    
    # Prepare batch arguments
    batch_args = []
    for i in range(n_batches):
        start_seed = i * batch_size
        actual_batch_size = min(batch_size, B - start_seed)
        batch_args.append((n, start_seed, actual_batch_size, y_star, sigma2, bias_corrected))
    
    # Run in parallel
    all_estimates = []
    with ProcessPoolExecutor(max_workers=n_cores) as executor:
        futures = [executor.submit(run_mc_batch, args) for args in batch_args]
        
        for future in as_completed(futures):
            batch_estimates = future.result()
            all_estimates.extend(batch_estimates)
    
    estimates = np.array(all_estimates[:B])  # Ensure we have exactly B estimates
    return float(np.mean(estimates)), float(np.var(estimates))


def main() -> None:
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    # Problem setup
    y_star = 0.8
    sigma2 = 1.0 / 16.0
    true_value = compute_true_value(y_star, sigma2)

    # Get number of CPU cores (use all available)
    n_cores = mp.cpu_count()
    print(f"Using {n_cores} CPU cores for parallel computation")

    # Experiment 1: D_{n,1}
    n_values = np.arange(10, 110, 10)
    mean_mc_list: list[float] = []
    var_mc_list: list[float] = []

    for n in n_values:
        B = int(round(n ** 3))
        print(f"n: {n}, B: {B}, running D_{n,1} estimator...")
        
        mean_mc, var_mc = run_parallel_mc(n, B, y_star, sigma2, bias_corrected=False, n_cores=n_cores)
        print(f"n: {n}, Mean MC Estimate: {mean_mc:.4f}, Variance: {var_mc:.4f}")
        mean_mc_list.append(mean_mc)
        var_mc_list.append(var_mc)

    # Save arrays
    np.save(os.path.join(results_dir, "mean_mc_list.npy"), np.array(mean_mc_list))
    np.save(os.path.join(results_dir, "var_mc_list.npy"), np.array(var_mc_list))

    # Experiment 2: D_{n,2}
    n_values = np.arange(10, 110, 10)
    mean_mc_list2: list[float] = []
    var_mc_list2: list[float] = []

    for n in n_values:
        B = int(round(n ** 4))
        print(f"n: {n}, B: {B}, running D_{n,2} estimator...")
        
        mean_mc, var_mc = run_parallel_mc(n, B, y_star, sigma2, bias_corrected=True, n_cores=n_cores)
        print(f"n: {n}, Mean MC Estimate: {mean_mc:.4f}, Variance: {var_mc:.4f}")
        mean_mc_list2.append(mean_mc)
        var_mc_list2.append(var_mc)

    # Save arrays
    np.save(os.path.join(results_dir, "mean_mc_list2.npy"), np.array(mean_mc_list2))
    np.save(os.path.join(results_dir, "var_mc_list2.npy"), np.array(var_mc_list2))

    # Combined bias plot
    plt.figure(figsize=(8, 6))
    plt.loglog(n_values, np.abs(np.array(mean_mc_list) - true_value), marker="o", label=r"$\mathbb{E}_{X^n}\{D_{n,1}f(\hat{\pi})(A)\}-f(\pi)(A)$")
    plt.loglog(n_values, np.abs(np.array(mean_mc_list2) - true_value), marker="o", label=r"$\mathbb{E}_{X^n}\{D_{n,2}f(\hat{\pi})(A)\}-f(\pi)(A)$")
    plt.loglog(n_values, (n_values) ** (-1.0), linestyle="--", label=r"$\mathcal{O}(n^{-1})$ Reference")
    plt.loglog(n_values, (n_values) ** (-2.0), linestyle="--", label=r"$\mathcal{O}(n^{-2})$ Reference")
    plt.xlabel("n (log scale)")
    plt.ylabel("Bias (log scale)")
    plt.legend(loc="best", fontsize=9, markerscale=1, frameon=True, framealpha=0.7)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.savefig(os.path.join(results_dir, "convergence_mixture.pdf"), dpi=300, bbox_inches="tight")
    plt.close()

    # Variance plot
    plt.figure(figsize=(8, 6))
    plt.loglog(n_values, np.array(var_mc_list), marker="o", label=r"$\text{Var}_{X^n}\{D_{n,1}f(\hat{\pi})(A)\}$")
    plt.loglog(n_values, np.array(var_mc_list2), marker="o", label=r"$\text{Var}_{X^n}\{D_{n,2}f(\hat{\pi})(A)\}$")
    plt.loglog(n_values, (n_values) ** (-1.0), linestyle="--", label=r"$\mathcal{O}(n^{-1})$ Reference")
    plt.xlabel("n (log scale)")
    plt.ylabel("Variance (log scale)")
    plt.legend(loc="best", fontsize=9, markerscale=1, frameon=True, framealpha=0.7)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.savefig(os.path.join(results_dir, "convergence_mixture_variance.pdf"), dpi=300, bbox_inches="tight")
    plt.close()

    print("Saved arrays and plots under:", os.path.abspath(results_dir))


if __name__ == "__main__":
    main()


