import os

os.environ["OMP_NUM_THREADS"] = str(os.cpu_count())

import numpy as np
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.special import binom
from scipy.special import logsumexp
import matplotlib.pyplot as plt

# Labels
plt.rcParams.update(
    {
        "axes.labelsize": 20,  # x and y labels
        "axes.titlesize": 20,  # title size
        "xtick.labelsize": 14,  # x tick labels
        "ytick.labelsize": 14,  # y tick labels
        "legend.fontsize": 14,  # legend text
    }
)

sys.path.append(str(Path(__file__).resolve().parents[1]))
from inference import poisson_logpdf, hmm_normalizer

# Fixed Parameters for both models
T = 100
dt = 1 / T  # 10 ms bins for T=100
Rh = 50
K = 100
M = 30
# M = 5


# 1) Relevant Functions for Ramp and Step Models
# 1.1) Ramp Model
# Initial Distribution
def initial_distribution_ramp(x0, sigma, K, dt):
    dx = 1 / (K - 1)
    pi = np.zeros(K)
    sigma_ramp = sigma * np.sqrt(dt)
    for s in range(K):
        if s == 0:
            z = dx / 2 - x0
            pi[0] = norm.cdf(z, loc=0, scale=sigma_ramp)
        elif s == K - 1:
            z = 1 - dx / 2 - x0
            pi[K - 1] = 1 - norm.cdf(z, loc=0, scale=sigma_ramp)
        else:
            z_1 = (s + 0.5) * dx - x0
            z_2 = (s - 0.5) * dx - x0
            pi[s] = norm.cdf(z_1, loc=0, scale=sigma_ramp) - norm.cdf(
                z_2, loc=0, scale=sigma_ramp
            )
    sum_pi = np.sum(pi)
    pi_normalised = pi / sum_pi  # Normalise to ensure it sums to 1
    return pi_normalised


# Calculate Transition Matrix
def transition_matrix_ramp(sigma, K, dt, beta):
    dx = 1 / (K - 1)
    T_matrix = np.zeros((K, K))
    sigma_ramp = sigma * np.sqrt(dt)
    for s in range(K):
        if s == K - 1:
            for s_next in range(K):
                if s_next == K - 1:
                    T_matrix[s, s_next] = 1
                else:
                    T_matrix[s, s_next] = 0
        else:
            for s_next in range(K):
                if s_next == 0:
                    z = dx / 2 - s * dx - beta * dt
                    T_matrix[s, 0] = norm.cdf(z, loc=0, scale=sigma_ramp)
                elif s_next == K - 1:
                    z = 1 - dx / 2 - s * dx - beta * dt
                    T_matrix[s, K - 1] = 1 - norm.cdf(z, loc=0, scale=sigma_ramp)
                else:
                    z_1 = (s_next + 0.5) * dx - s * dx - beta * dt
                    z_2 = (s_next - 0.5) * dx - s * dx - beta * dt
                    T_matrix[s, s_next] = norm.cdf(
                        z_1, loc=0, scale=sigma_ramp
                    ) - norm.cdf(z_2, loc=0, scale=sigma_ramp)
    sum_T = np.sum(T_matrix, axis=1)
    T_normalised = T_matrix / sum_T[:, None]  # Normalise rows to ensure they sum to 1
    return T_normalised


# Simulate s_t trajectories and get r_t trajectories
def get_r_t_ramp(s_0_samples_ramp, T_matrix_ramp, K, Rh, Ntrials):
    s_t_ramp = np.zeros((Ntrials, T), dtype=int)
    s_t_ramp[:, 0] = s_0_samples_ramp
    for t in range(1, T):
        for i in range(Ntrials):
            s_t_ramp[i, t] = np.random.choice(K, p=T_matrix_ramp[s_t_ramp[i, t - 1]])
    # x_t and r_t trajectories
    x_t_ramp = s_t_ramp / (K - 1)
    r_t_ramp = np.zeros((Ntrials, T), dtype=float)
    for i in range(Ntrials):
        for t in range(T):
            r_t_ramp[i, t] = Rh if x_t_ramp[i, t] >= 1 else x_t_ramp[i, t] * Rh
    return r_t_ramp


# Generate spike counts from firing rates
def get_n_ramp(r_t_ramp, Ntrials):
    n_t_ramp = np.zeros((Ntrials, T), dtype=int)
    for i in range(Ntrials):
        for t in range(T):
            # Sample from Poisson distribution with rate r_t * dt
            n_t_ramp[i, t] = np.random.poisson(r_t_ramp[i, t] * dt)
    return n_t_ramp


# Lambdas for Poisson
def get_lambdas_ramp(K, Rh):
    lambdas_ramp = np.zeros(K)
    for s in range(K):
        x_s = s / (K - 1)
        rate_s = min(x_s, 1.0) * Rh
        lambdas_ramp[s] = rate_s * dt
    return lambdas_ramp


# 1.2) Step Model
# Initial Distribution
def get_initial_distribution_2ih(m, r):
    p = r / (m + r)
    P = p**r
    pi_2ih = np.array([1 - P, P])
    return pi_2ih


# Calculate Transition Matrix
def get_transition_2ih(m, r):
    p = r / (m + r)
    T_matrix_2ih = np.zeros((T - 1, 2, 2))
    for t in range(1, T):
        prob_sum = 0
        for s in range(t):
            prob_sum += binom(s + r - 1, s) * (p**r) * ((1 - p) ** (s))
        if prob_sum >= 1:
            Pt = 0.0
        else:
            Pt = binom(t + r - 1, t) * (p**r) * ((1 - p) ** t) / (1 - prob_sum)
            Pt = min(Pt, 1.0)  # safety clip
        T_matrix_2ih_t = np.array([[1 - Pt, Pt], [0, 1]])
        T_matrix_2ih[t - 1] = T_matrix_2ih_t
    return T_matrix_2ih


# Simulate the Markov Chain
def get_s_t_2ih(m, r, s_0_samples_2ih, Ntrials):
    s_t_2ih = np.zeros((Ntrials, T), dtype=int)
    s_t_2ih[:, 0] = s_0_samples_2ih
    T_matrix_2ih = get_transition_2ih(m, r)
    for t in range(1, T):
        for i in range(Ntrials):
            T_matrix_2ih_t = T_matrix_2ih[t - 1]
            s_t_2ih[i, t] = np.random.choice(2, p=T_matrix_2ih_t[s_t_2ih[i, t - 1]])
    return s_t_2ih


# Get fire rates
def get_r_t_2ih(x_t_values_2ih, Rh, Ntrials):
    r_t_2ih = np.zeros((Ntrials, T), dtype=float)
    for i in range(Ntrials):
        for t in range(T):
            r_t_2ih[i, t] = (
                Rh if x_t_values_2ih[i, t] >= 1 else x_t_values_2ih[i, t] * Rh
            )
    return r_t_2ih


# Generate spike counts from firing rates
def get_n_2ih(r_t_2ih, Ntrials):
    n_t_2ih = np.zeros((Ntrials, T), dtype=int)
    for i in range(Ntrials):
        for t in range(T):
            # Sample from Poisson distribution with rate r_t * dt
            n_t_2ih[i, t] = np.random.poisson(r_t_2ih[i, t] * dt)
    return n_t_2ih


# Lambdas for Poisson
def get_lambdas_2ih(x0, Rh):
    lambdas_2ih = np.zeros(2)
    for state in range(2):
        if state == 1:
            lambdas_2ih[state] = 1.0 * Rh * dt
        else:
            lambdas_2ih[state] = x0 * Rh * dt
    return lambdas_2ih


# 2) Simulation and Inference
# Get Spike Counts for both Models
def simulate_data(model, x0, sigma, beta, m, r, Ntrials, K, Rh, dt):
    if model == "ramp":
        pi_ramp = initial_distribution_ramp(x0, sigma, K, dt)
        s_0_samples_ramp = np.random.choice(K, size=Ntrials, p=pi_ramp)
        T_matrix_ramp = transition_matrix_ramp(sigma, K, dt, beta)
        r_t_ramp = get_r_t_ramp(s_0_samples_ramp, T_matrix_ramp, K, Rh, Ntrials)
        n_t_ramp = get_n_ramp(r_t_ramp, Ntrials)
        return n_t_ramp
    elif model == "step":
        pi_2ih = get_initial_distribution_2ih(m, r)
        s_0_samples_2ih = np.random.choice(2, size=Ntrials, p=pi_2ih)
        s_t_2ih = get_s_t_2ih(m, r, s_0_samples_2ih, Ntrials)
        x_t_values_2ih = np.where(s_t_2ih == 1, 1.0, x0)
        r_t_2ih = get_r_t_2ih(x_t_values_2ih, Rh, Ntrials)
        n_t_2ih = get_n_2ih(r_t_2ih, Ntrials)
        return n_t_2ih
    else:
        raise ValueError("Invalid model type. Choose 'ramp' or 'step'.")


# Log Prior Distributions
def get_log_prior(model, beta_values, ln_sigma_values, m_values, r_values):
    if model == "ramp":
        M = len(beta_values)
        prior_ramp = np.zeros((M, M))
        log_beta = -np.log(4)
        log_ln_sigma = -np.log(np.log(4) - np.log(0.04))
        for i in range(M):
            for j in range(M):
                prior_ramp[i, j] = log_beta + log_ln_sigma
        return prior_ramp
    elif model == "step":
        M = len(m_values)
        prior_step = np.zeros((M, len(r_values)))
        log_p_m = -np.log(3 * T / 4)
        log_p_r = -np.log(len(r_values))
        for i in range(M):
            for j in range(len(r_values)):
                prior_step[i, j] = log_p_m + log_p_r
        return prior_step
    else:
        raise ValueError("Invalid model type. Choose 'ramp' or 'step'.")


# Get Posterior Probability Grid
def get_posterior(
    model,
    spike_counts,
    lambdas,
    beta_values,
    ln_sigma_values,
    m_values,
    r_values,
    Ntrials,
):
    if model == "ramp":
        M = len(beta_values)
        sigma_values = np.exp(ln_sigma_values)
        ll_ramp = np.zeros((len(beta_values), len(ln_sigma_values)))
        for i in range(M):
            for j in range(M):
                # Simulate data for Ramp
                initial_prob_ramp = pi_ramp_cache[(i, j)]
                T_matrix_ramp = T_ramp_cache[(i, j)]
                total_ll = 0
                for w in range(Ntrials):
                    ll = poisson_logpdf(counts=spike_counts[w], lambdas=lambdas.copy())
                    total_ll += hmm_normalizer(
                        pi0=initial_prob_ramp, Ps=T_matrix_ramp, ll=ll
                    )
                ll_ramp[i, j] = total_ll
        prior_ramp = get_log_prior("ramp", beta_values, ln_sigma_values, None, None)
        log_posterior_ramp = prior_ramp + ll_ramp
        # Normalise the posterior
        log_posterior_ramp_normalised = log_posterior_ramp - logsumexp(
            log_posterior_ramp
        )
        # Get posterior from log(posterior)
        posterior_ramp = np.exp(log_posterior_ramp_normalised)
        return posterior_ramp
    elif model == "step":
        M = len(m_values)
        ll_step = np.zeros((len(m_values), len(r_values)))
        for i in range(M):
            for j in range(len(r_values)):
                # Simulate data for Step
                initial_prob_step = pi_step_cache[(i, j)]
                T_matrix_step = T_step_cache[(i, j)]
                total_ll = 0
                for w in range(Ntrials):
                    ll = poisson_logpdf(counts=spike_counts[w], lambdas=lambdas.copy())
                    total_ll += hmm_normalizer(
                        pi0=initial_prob_step, Ps=T_matrix_step, ll=ll
                    )
                ll_step[i, j] = total_ll
        prior_step = get_log_prior("step", None, None, m_values, r_values)
        log_posterior_step = prior_step + ll_step
        # Normalise the posterior
        log_posterior_step_normalised = log_posterior_step - logsumexp(
            log_posterior_step
        )
        # Get posterior from log(posterior)
        posterior_step = np.exp(log_posterior_step_normalised)
        return posterior_step
    else:
        raise ValueError("Invalid model type. Choose 'ramp' or 'step'.")


# 3) Functions for 3.1.1)
# Ramp axes
def label_ramp_axes(beta_values, ln_sigma_values):
    xtick_idx = np.arange(0, len(ln_sigma_values), max(1, len(ln_sigma_values) // 6))
    plt.xticks(
        ticks=xtick_idx,
        labels=[f"{ln_sigma_values[k]:.2f}" for k in xtick_idx],
    )
    ytick_idx = np.arange(0, len(beta_values), max(1, len(beta_values) // 6))
    plt.yticks(
        ticks=ytick_idx,
        labels=[f"{beta_values[k]:.2f}" for k in ytick_idx],
    )
    plt.xlabel("ln(sigma)")
    plt.ylabel("beta")


# Step axes
def label_step_axes(m_values, r_values):
    xtick_idx = np.arange(len(r_values))
    plt.xticks(
        ticks=xtick_idx,
        labels=[f"{r_values[k]}" for k in xtick_idx],
    )
    ytick_idx = np.arange(0, len(m_values), max(1, len(m_values) // 6))
    plt.yticks(
        ticks=ytick_idx,
        labels=[f"{m_values[k]:.1f}" for k in ytick_idx],
    )
    plt.xlabel("r")
    plt.ylabel("m")


# Calculate the True Coordinates of the Parameters in the Grid
def get_true_coordinates(array_1, array_2, standard_value1, standard_value2):
    true_coord_1 = np.interp(standard_value1, array_1, np.arange(len(array_1)))
    true_coord_2 = np.interp(standard_value2, array_2, np.arange(len(array_2)))
    return true_coord_1, true_coord_2


# 4) Functions for 3.1.2)
# Expected Value
def expected_value(log_posterior_grid, values_array):
    posterior_probabilities = np.exp(log_posterior_grid - np.max(log_posterior_grid))
    posterior_probabilities /= np.sum(posterior_probabilities)
    expected_value = np.sum(values_array * posterior_probabilities)
    return expected_value


# Variances
def variance(log_posterior_grid, values_array):
    posterior_probabilities = np.exp(log_posterior_grid - np.max(log_posterior_grid))
    posterior_probabilities /= np.sum(posterior_probabilities)
    expected_value = np.sum(values_array * posterior_probabilities)
    variance = np.sum(((values_array - expected_value) ** 2) * posterior_probabilities)
    return variance


# Expected Value and Standard Deviation for both models
# To marginalise over axis 1 (keeping axis 0) we logsumexp over axis=1.
def get_expected_and_std(posterior_grid, values_axis0, values_axis1):
    log_posterior_grid = np.log(posterior_grid)
    log_post_axis0 = logsumexp(log_posterior_grid, axis=1)  # marginal over axis 1
    log_post_axis1 = logsumexp(log_posterior_grid, axis=0)  # marginal over axis 0
    E0 = expected_value(log_post_axis0, values_axis0)
    E1 = expected_value(log_post_axis1, values_axis1)
    std0 = np.sqrt(variance(log_post_axis0, values_axis0))
    std1 = np.sqrt(variance(log_post_axis1, values_axis1))
    return E0, std0, E1, std1


# Estimation Error
def estimation_error(true_value, expected_value):
    return np.abs(true_value - expected_value)


# Function for plotTING the Expected Value +/- Posterior Std (Error Bars) with True Value
def plot_estimate_with_errorbars(Ns, estimates, stds, true_val, name):
    plt.figure(figsize=(7, 4))
    plt.errorbar(
        Ns,
        estimates,
        yerr=stds,
        fmt="o-",
        capsize=4,
        label=f"estimate $\\pm$ posterior std",
    )
    plt.axhline(true_val, color="red", ls="--", label=f"true {name}")
    plt.xscale("log")
    plt.xlabel("N_trials")
    plt.ylabel(name)
    # plt.title(f"{name}: posterior estimate with uncertainty vs N_trials")
    plt.legend()
    plt.tight_layout()
    plt.show()


# Function for comparing the estimation error vs posterior std, both as functions of N
def plot_error_vs_uncertainty(Ns, errors, stds, name):
    plt.figure(figsize=(7, 4))
    # Use np.maximum to avoid plotting zero error on a log scale, which would cause issues
    plt.plot(Ns, np.maximum(errors, 1e-12), "o-", label="estimation error (accuracy)")
    plt.plot(Ns, stds, "s--", label="posterior std (uncertainty)")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("N_trials")
    plt.ylabel(name)
    # plt.title(f"{name}: estimation error vs posterior uncertainty")
    plt.legend()
    plt.tight_layout()
    plt.show()


# 5) Parameters
# List of values to vary for each parameter
# Ntrials_list = [2, 3]
# beta_list = [0.1, 0.5]
# sigma_list = [0.04, 0.2]
# m_list = [T / 4, T / 2]
# r_list = [1, 3]
# x0_list = [0, 0.2]


Ntrials_list = [1, 10, 100, 400]
beta_list = [0.1, 0.5, 3.5]
sigma_list = [0.04, 0.2, 3.9]
m_list = [T / 4, T / 2, 3 * T / 4]
r_list = [1, 3, 5]
x0_list = [0, 0.2, 0.5]
# Also change Ntrials_standard and M


# Prior Distributions
# Beta Values - beta = np.random.uniform(0, 4)
beta_edges = np.linspace(0, 4, M + 1)
beta_values = (beta_edges[:-1] + beta_edges[1:]) / 2
# Ln(Sigma) Values - ln_sigma = np.random.uniform(np.log(0.04), np.log(4))
ln_sigma_edges = np.linspace(np.log(0.04), np.log(4), M + 1)
ln_sigma_values = (ln_sigma_edges[:-1] + ln_sigma_edges[1:]) / 2
# m values - m = np.random.uniform(0, 3 * T / 4)
m_edges = np.linspace(0, 3 * T / 4, M + 1)
m_values = (m_edges[:-1] + m_edges[1:]) / 2
# r values - r = np.random.choice([1, 2, 3, 4, 5, 6])
r_values = np.array([1, 2, 3, 4, 5, 6])
# X0 value - x0 = np.random.uniform(0, 0.5)
x0_edges = np.linspace(0, 0.5, M + 1)
x0_values = (x0_edges[:-1] + x0_edges[1:]) / 2


# Standard Values
Ntrials_standard = 10
# Ntrials_standard = 2
x0_standard = 0.2
beta_standard = 0.5
sigma_standard = 0.2
ln_sigma_standard = np.log(sigma_standard)
m_standard = 50.0
r_standard = 3


# Get lambdas for both models
lambdas_ramp = get_lambdas_ramp(K, Rh)
lambdas_step = get_lambdas_2ih(x0_standard, Rh)


# Number of Datasets - for 3.1.2)
n_datasets = 3


# Precompute all transition matrices once before any get_error_rates calls
# Ramp matrices
pi_ramp_cache = {}
T_ramp_cache = {}
for i, beta in enumerate(beta_values):
    for j, ln_sigma in enumerate(ln_sigma_values):
        sigma = np.exp(ln_sigma)
        pi_ramp_cache[(i, j)] = initial_distribution_ramp(x0_standard, sigma, K, dt)
        T_ramp_cache[(i, j)] = transition_matrix_ramp(sigma, K, dt, beta)

# Step matrices
pi_step_cache = {}
T_step_cache = {}
for i, m in enumerate(m_values):
    for j, r in enumerate(r_values):
        pi_step_cache[(i, j)] = get_initial_distribution_2ih(m, r)
        T_step_cache[(i, j)] = get_transition_2ih(m, r)


# 6) Solution for 3.1.1)
# Ramp Model Posterior Grids and Plots
# Vary Ntrials
for Ntrials in Ntrials_list:
    # Simulate Spike Counts for Ramp Model
    spike_counts_ramp = simulate_data(
        "ramp",
        x0_standard,
        sigma_standard,
        beta_standard,
        None,
        None,
        Ntrials,
        K,
        Rh,
        dt,
    )
    # Calculate posterior grid for Ramp Model
    posterior_grid_ramp = get_posterior(
        "ramp",
        spike_counts_ramp,
        lambdas_ramp,
        beta_values,
        ln_sigma_values,
        None,
        None,
        Ntrials,
    )
    # Plot posterior grid - Ramp Model
    plt.imshow(posterior_grid_ramp, cmap="viridis")
    plt.colorbar()
    coord_1, coord_2 = get_true_coordinates(
        beta_values, ln_sigma_values, beta_standard, ln_sigma_standard
    )
    plt.scatter(coord_2, coord_1, color="red", marker="x", s=100)
    label_ramp_axes(beta_values, ln_sigma_values)
    # plt.title(f"Ramp Model Grid: (Ntrials={Ntrials}) with beta={beta_standard} and sigma={sigma_standard}")
    plt.show()


# Vary beta
for beta in beta_list:
    # Simulate Spike Counts for Ramp Model
    spike_counts_ramp = simulate_data(
        "ramp",
        x0_standard,
        sigma_standard,
        beta,
        None,
        None,
        Ntrials_standard,
        K,
        Rh,
        dt,
    )
    # Calculate log posterior grid for Ramp Model
    posterior_grid_ramp = get_posterior(
        "ramp",
        spike_counts_ramp,
        lambdas_ramp,
        beta_values,
        ln_sigma_values,
        None,
        None,
        Ntrials_standard,
    )
    # Plot log posterior grid - Ramp Model
    plt.imshow(posterior_grid_ramp, cmap="viridis")
    plt.colorbar()
    coord_1, coord_2 = get_true_coordinates(
        beta_values, ln_sigma_values, beta, ln_sigma_standard
    )
    plt.scatter(coord_2, coord_1, color="red", marker="x", s=100)
    label_ramp_axes(beta_values, ln_sigma_values)
    # plt.title(f"Ramp Model Grid: (beta={beta}) with sigma={sigma_standard} and Ntrials={Ntrials_standard}")
    plt.show()

# Vary sigma
for sigma in sigma_list:
    # Simulate Spike Counts for Ramp Model
    spike_counts_ramp = simulate_data(
        "ramp",
        x0_standard,
        sigma,
        beta_standard,
        None,
        None,
        Ntrials_standard,
        K,
        Rh,
        dt,
    )
    # Calculate log posterior grid for Ramp Model
    posterior_grid_ramp = get_posterior(
        "ramp",
        spike_counts_ramp,
        lambdas_ramp,
        beta_values,
        ln_sigma_values,
        None,
        None,
        Ntrials_standard,
    )
    # Plot log posterior grid - Ramp Model
    plt.imshow(posterior_grid_ramp, cmap="viridis")
    plt.colorbar()
    coord_1, coord_2 = get_true_coordinates(
        beta_values, ln_sigma_values, beta_standard, np.log(sigma)
    )
    plt.scatter(coord_2, coord_1, color="red", marker="x", s=100)
    label_ramp_axes(beta_values, ln_sigma_values)
    # plt.title(f"Ramp Model Grid: (sigma={sigma}) with beta={beta_standard} and Ntrials={Ntrials_standard}")
    plt.show()


# Step Model Posterior Grids and Plots
# Vary Ntrials
for Ntrials in Ntrials_list:
    # Simulate Spike Counts for Step Model
    spike_counts_step = simulate_data(
        "step",
        x0_standard,
        sigma_standard,
        beta_standard,
        m_standard,
        r_standard,
        Ntrials,
        K,
        Rh,
        dt,
    )
    # Calculate log posterior grid for Step Model
    posterior_grid_step = get_posterior(
        "step",
        spike_counts_step,
        lambdas_step,
        None,
        None,
        m_values,
        r_values,
        Ntrials,
    )
    # Plot log posterior grid - Step Model
    plt.imshow(
        posterior_grid_step, cmap="viridis", aspect=len(r_values) / len(m_values)
    )
    plt.colorbar()
    coord_1, coord_2 = get_true_coordinates(m_values, r_values, m_standard, r_standard)
    plt.scatter(coord_2, coord_1, color="red", marker="x", s=100)
    label_step_axes(m_values, r_values)
    # plt.title(f"Step Model Grid: (Ntrials={Ntrials}) with m={m_standard} and r={r_standard}")
    plt.show()

# Vary m
for m in m_list:
    # Simulate Spike Counts for Step Model
    spike_counts_step = simulate_data(
        "step",
        x0_standard,
        sigma_standard,
        beta_standard,
        m,
        r_standard,
        Ntrials_standard,
        K,
        Rh,
        dt,
    )
    # Calculate log posterior grid for Step Model
    posterior_grid_step = get_posterior(
        "step",
        spike_counts_step,
        lambdas_step,
        None,
        None,
        m_values,
        r_values,
        Ntrials_standard,
    )
    # Plot log posterior grid - Step Model
    plt.imshow(
        posterior_grid_step, cmap="viridis", aspect=len(r_values) / len(m_values)
    )
    plt.colorbar()
    coord_1, coord_2 = get_true_coordinates(m_values, r_values, m, r_standard)
    plt.scatter(coord_2, coord_1, color="red", marker="x", s=100)
    label_step_axes(m_values, r_values)
    # plt.title(f"Step Model Grid: (m={m}) with r={r_standard} and Ntrials={Ntrials_standard}")
    plt.show()

# Vary r
for r in r_list:
    # Simulate Spike Counts for Step Model
    spike_counts_step = simulate_data(
        "step",
        x0_standard,
        sigma_standard,
        beta_standard,
        m_standard,
        r,
        Ntrials_standard,
        K,
        Rh,
        dt,
    )
    # Calculate log posterior grid for Step Model
    posterior_grid_step = get_posterior(
        "step",
        spike_counts_step,
        lambdas_step,
        None,
        None,
        m_values,
        r_values,
        Ntrials_standard,
    )
    # Plot log posterior grid - Step Model
    plt.imshow(
        posterior_grid_step, cmap="viridis", aspect=len(r_values) / len(m_values)
    )
    plt.colorbar()
    coord_1, coord_2 = get_true_coordinates(m_values, r_values, m_standard, r)
    plt.scatter(coord_2, coord_1, color="red", marker="x", s=100)
    label_step_axes(m_values, r_values)
    # plt.title(f"Step Model Grid: (r={r}) with m={m_standard} and Ntrials={Ntrials_standard}")
    plt.show()


# Solution for 3.1.2)
# Ramp Model
est_beta, std_beta, error_beta = [], [], []
est_sigma, std_sigma, error_sigma = [], [], []
for Ntrials in Ntrials_list:
    est_beta_Ntrials, std_beta_Ntrials, error_beta_Ntrials = [], [], []
    est_sigma_Ntrials, std_sigma_Ntrials, error_sigma_Ntrials = [], [], []
    for q in range(n_datasets):
        spikes_counts_ramp = simulate_data(
            "ramp",
            x0_standard,
            sigma_standard,
            beta_standard,
            None,
            None,
            Ntrials,
            K,
            Rh,
            dt,
        )
        posterior_prob = get_posterior(
            "ramp",
            spikes_counts_ramp,
            lambdas_ramp,
            beta_values,
            ln_sigma_values,
            None,
            None,
            Ntrials,
        )
        E_b, S_b, E_ls, S_ls = get_expected_and_std(
            posterior_prob, beta_values, ln_sigma_values
        )
        est_beta_Ntrials.append(E_b)
        std_beta_Ntrials.append(S_b)
        error_beta_Ntrials.append(estimation_error(beta_standard, E_b))
        est_sigma_Ntrials.append(np.exp(E_ls))
        std_sigma_Ntrials.append(np.exp(E_ls) * S_ls)
        error_sigma_Ntrials.append(estimation_error(sigma_standard, np.exp(E_ls)))
    est_beta.append(np.mean(est_beta_Ntrials))
    std_beta.append(np.mean(std_beta_Ntrials))
    error_beta.append(np.mean(error_beta_Ntrials))
    est_sigma.append(np.mean(est_sigma_Ntrials))
    std_sigma.append(np.mean(std_sigma_Ntrials))
    error_sigma.append(np.mean(error_sigma_Ntrials))


# Step Model
est_m, std_m, error_m = [], [], []
est_r, std_r, error_r = [], [], []
for Ntrials in Ntrials_list:
    est_m_Ntrials, std_m_Ntrials, error_m_Ntrials = [], [], []
    est_r_Ntrials, std_r_Ntrials, error_r_Ntrials = [], [], []
    for q in range(n_datasets):
        spikes_counts_step = simulate_data(
            "step",
            x0_standard,
            None,
            None,
            m_standard,
            r_standard,
            Ntrials,
            K,
            Rh,
            dt,
        )
        posterior_prob = get_posterior(
            "step",
            spikes_counts_step,
            lambdas_step,
            None,
            None,
            m_values,
            r_values,
            Ntrials,
        )
        E_m, S_m, E_r, S_r = get_expected_and_std(posterior_prob, m_values, r_values)
        est_m_Ntrials.append(E_m)
        std_m_Ntrials.append(S_m)
        error_m_Ntrials.append(estimation_error(m_standard, E_m))
        est_r_Ntrials.append(E_r)
        std_r_Ntrials.append(S_r)
        error_r_Ntrials.append(estimation_error(r_standard, E_r))
    est_m.append(np.mean(est_m_Ntrials))
    std_m.append(np.mean(std_m_Ntrials))
    error_m.append(np.mean(error_m_Ntrials))
    est_r.append(np.mean(est_r_Ntrials))
    std_r.append(np.mean(std_r_Ntrials))
    error_r.append(np.mean(error_r_Ntrials))


# Plotting the Estimated Values with Error Bars
plot_estimate_with_errorbars(Ntrials_list, est_beta, std_beta, beta_standard, "beta")
plot_estimate_with_errorbars(
    Ntrials_list, est_sigma, std_sigma, sigma_standard, "sigma"
)
plot_estimate_with_errorbars(Ntrials_list, est_m, std_m, m_standard, "m")
plot_estimate_with_errorbars(Ntrials_list, est_r, std_r, r_standard, "r")


# Plotting the Estimation Error vs Posterior Uncertainty
plot_error_vs_uncertainty(Ntrials_list, error_beta, std_beta, "beta")
plot_error_vs_uncertainty(Ntrials_list, error_sigma, std_sigma, "sigma")
plot_error_vs_uncertainty(Ntrials_list, error_m, std_m, "m")
plot_error_vs_uncertainty(Ntrials_list, error_r, std_r, "r")
