import os

os.environ["OMP_NUM_THREADS"] = str(os.cpu_count())

import numpy as np
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import norm
import seaborn as sns
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
M = 10


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


# Truncated Gaussian Distributions
def get_truncated_gaussian_distribution(
    parameter_values, low, high, fraction, center="middle"
):
    # Mean
    if center == "middle":
        mean = 0.5 * (low + high)  # for beta, ln_sigma, m
    elif center == "low":
        mean = low  # for r
    else:
        raise ValueError("center must be 'middle' or 'low'")
    # Standard deviation
    std = fraction * (high - low)
    # Gaussian density on grid
    distribution = np.exp(-0.5 * ((parameter_values - mean) / std) ** 2)
    # Normalize over grid
    distribution /= np.sum(distribution)
    return distribution


# Log Prior for Truncated Gaussian Distribution
def log_prior_truncated_gaussian(
    model, beta_values, ln_sigma_values, m_values, r_values, fraction, T
):
    if model == "ramp":
        M = len(beta_values)
        prior_ramp = np.zeros((M, M))
        prior_beta = get_truncated_gaussian_distribution(
            beta_values, 0, 4, fraction, center="middle"
        )
        prior_ln_sigma = get_truncated_gaussian_distribution(
            ln_sigma_values, np.log(0.04), np.log(4), fraction, center="middle"
        )
        for i in range(M):
            for j in range(M):
                prior_ramp[i, j] = np.log(prior_beta[i]) + np.log(prior_ln_sigma[j])
        return prior_ramp
    elif model == "step":
        M = len(m_values)
        prior_step = np.zeros((M, len(r_values)))
        prior_m = get_truncated_gaussian_distribution(
            m_values, 0, 3 * T / 4, fraction, center="middle"
        )
        prior_r = get_truncated_gaussian_distribution(
            r_values, 1, 6, fraction, center="low"
        )
        for i in range(M):
            for j in range(len(r_values)):
                prior_step[i, j] = np.log(prior_m[i]) + np.log(prior_r[j])
        return prior_step
    else:
        raise ValueError("Invalid model type. Choose 'ramp' or 'step'.")


# Log Prior for Uniform Distribution
def log_prior_uniform(model, beta_values, ln_sigma_values, m_values, r_values):
    if model == "ramp":
        no_grid_cells_ramp = len(beta_values) * len(ln_sigma_values)
        return np.full(
            (len(beta_values), len(ln_sigma_values)), -np.log(no_grid_cells_ramp)
        )
    elif model == "step":
        no_grid_cells_step = len(m_values) * len(r_values)
        return np.full((len(m_values), len(r_values)), -np.log(no_grid_cells_step))
    else:
        raise ValueError("Invalid model type. Choose 'ramp' or 'step'.")


# Get Log Prior
def get_log_prior(
    model, beta_values, ln_sigma_values, m_values, r_values, method, fraction, T
):
    if method == "uniform":
        return log_prior_uniform(
            model, beta_values, ln_sigma_values, m_values, r_values
        )

    elif method == "gaussian":
        return log_prior_truncated_gaussian(
            model,
            beta_values,
            ln_sigma_values,
            m_values,
            r_values,
            fraction,
            T,
        )


# Sample Parameters
def sample_parameters(
    model, beta_values, ln_sigma_values, m_values, r_values, method, fraction, T
):
    if method == "uniform":
        if model == "ramp":
            beta = np.random.choice(beta_values)
            ln_sigma = np.random.choice(ln_sigma_values)
            return beta, ln_sigma
        else:
            m = np.random.choice(m_values)
            r = np.random.choice(r_values)
            return m, r

    elif method == "gaussian":
        if model == "ramp":
            prior_beta = get_truncated_gaussian_distribution(
                beta_values, 0, 4, fraction
            )
            prior_ln_sigma = get_truncated_gaussian_distribution(
                ln_sigma_values, np.log(0.04), np.log(4), fraction
            )
            beta = np.random.choice(beta_values, p=prior_beta)
            ln_sigma = np.random.choice(ln_sigma_values, p=prior_ln_sigma)
            return beta, ln_sigma

        else:
            prior_m = get_truncated_gaussian_distribution(
                m_values, 0, 3 * T / 4, fraction
            )
            prior_r = get_truncated_gaussian_distribution(
                r_values, 1, 6, fraction, center="low"
            )
            m = np.random.choice(m_values, p=prior_m)
            r = np.random.choice(r_values, p=prior_r)
            return m, r


# Get Bayes Factor for both Models
def get_Bayes_Factor(model, method, fraction, x0, spike_counts, lambdas, Ntrials):
    if model == "ramp":
        ll_ramp = np.zeros((len(beta_values), len(ln_sigma_values)))
        for i in range(len(beta_values)):
            for j in range(len(ln_sigma_values)):
                pi0 = pi_ramp_cache[(i, j)]
                Ps = T_ramp_cache[(i, j)]
                total_ll = 0
                for w in range(Ntrials):
                    ll = poisson_logpdf(counts=spike_counts[w], lambdas=lambdas.copy())
                    total_ll += hmm_normalizer(pi0=pi0, Ps=Ps, ll=ll)
                ll_ramp[i, j] = total_ll
        prior_ramp = get_log_prior(
            "ramp",
            beta_values,
            ln_sigma_values,
            m_values,
            r_values,
            method,
            fraction,
            T,
        )
        return logsumexp(prior_ramp + ll_ramp)

    elif model == "step":
        ll_step = np.zeros((len(m_values), len(r_values)))
        for i in range(len(m_values)):
            for j in range(len(r_values)):
                pi0 = pi_step_cache[(i, j)]
                Ps = T_step_cache[(i, j)]
                total_ll = 0
                for w in range(Ntrials):
                    ll = poisson_logpdf(counts=spike_counts[w], lambdas=lambdas.copy())
                    total_ll += hmm_normalizer(pi0=pi0, Ps=Ps, ll=ll)
                ll_step[i, j] = total_ll
        prior_step = get_log_prior(
            "step",
            beta_values,
            ln_sigma_values,
            m_values,
            r_values,
            method,
            fraction,
            T,
        )
        return logsumexp(prior_step + ll_step)


# Get log MLR
def ln_R(spike_counts, infer_method, fraction, Ntrials_data, x0):
    lr = get_Bayes_Factor(
        "ramp", infer_method, fraction, x0, spike_counts, lambdas_ramp, Ntrials_data
    )
    ls = get_Bayes_Factor(
        "step", infer_method, fraction, x0, spike_counts, lambdas_step, Ntrials_data
    )
    return lr - ls


# Calculate the Error Rates
def get_error_rates(
    sample_method, infer_method, fraction, Ntrials_data, n_datasets, x0
):
    wrong_ramp = wrong_step = 0
    for _ in range(n_datasets):
        # true ramp
        beta, ln_sigma = sample_parameters(
            "ramp",
            beta_values,
            ln_sigma_values,
            m_values,
            r_values,
            sample_method,
            fraction,
            T,
        )
        data_ramp = simulate_data(
            "ramp", x0, np.exp(ln_sigma), beta, None, None, Ntrials_data, K, Rh, dt
        )
        if ln_R(data_ramp, infer_method, fraction, Ntrials_data, x0) <= 0:
            wrong_ramp += 1
        # true step
        m, r = sample_parameters(
            "step",
            beta_values,
            ln_sigma_values,
            m_values,
            r_values,
            sample_method,
            fraction,
            T,
        )
        data_step = simulate_data("step", x0, None, None, m, r, Ntrials_data, K, Rh, dt)
        if ln_R(data_step, infer_method, fraction, Ntrials_data, x0) > 0:
            wrong_step += 1
    return wrong_ramp / n_datasets, wrong_step / n_datasets


# 3) Functions from 1.4)
# PSTH Function
def compute_psth(spikes, dt):
    return np.mean(spikes, axis=0) / dt


# Bin down to 100ms bins
def bin_spikes(spikes, bin_size):
    Ntrials, T = spikes.shape
    n_bins = T // bin_size
    return spikes[:, : n_bins * bin_size].reshape(Ntrials, n_bins, bin_size).sum(axis=2)


# Fano factor = variance / mean
def get_fano(spikes):
    mean = spikes.mean(axis=0)
    variance = spikes.var(axis=0)
    fano = np.full_like(mean, np.nan, dtype=float)
    np.divide(variance, mean, out=fano, where=mean > 0)
    return fano


# Function to evaluate criteria for a spike counter
def get_criteria(spikes, ratio, dt_new):
    # Bin down to 100ms bins and compute Fano factor and PSTH
    spikes_binned = bin_spikes(spikes, ratio)
    fano = get_fano(spikes_binned)
    psth = compute_psth(spikes_binned, dt_new)

    # Criteria counter for StepModel
    count = 0

    # Criterior 1 - Maximum Fano Factor
    max_fano = np.nanmax(fano)
    if max_fano > 1.4:
        count += 1

    # Criterion 2 - PSTH Jump Size
    psth_changes = np.abs(np.diff(psth))
    max_psth_change = np.max(psth_changes) if len(psth_changes) > 0 else 0
    if max_psth_change > 4.1:
        count += 1

    # Criterion 3 -  PSTH Jump Size Compared to Total Change
    if np.sum(psth_changes) > 0:
        comparison = np.max(psth_changes) / np.sum(psth_changes)
        if comparison < 0.3:
            count += 1

    # Criterion 4 - Fano Factor Peakiness (max/mean)
    mean_fano = np.nanmean(fano)
    if mean_fano > 0:
        peakiness = np.nanmax(fano) / mean_fano
        if peakiness > 1.5:
            count += 1

    # Combine criteria
    return "step" if count >= 3 else "ramp"


# Function to simulate for multiple datasets
def simulate(sample_method, fraction, Ntrials_data, n_datasets, ratio, dt_new, x0):
    correct_step = 0
    correct_ramp = 0
    total_step = 0
    total_ramp = 0
    for _ in range(n_datasets):
        # Randomly choose which model to use
        true_model = np.random.choice(["step", "ramp"])
        if true_model == "step":
            # Simulate data from the StepModel
            m, r = sample_parameters(
                "step",
                beta_values,
                ln_sigma_values,
                m_values,
                r_values,
                sample_method,
                fraction,
                T,
            )
            spikes = simulate_data(
                true_model, x0, None, None, m, r, Ntrials_data, K, Rh, dt
            )
        else:
            # Simulate data from the RampModel
            beta, ln_sigma = sample_parameters(
                "ramp",
                beta_values,
                ln_sigma_values,
                m_values,
                r_values,
                sample_method,
                fraction,
                T,
            )
            spikes = simulate_data(
                true_model,
                x0,
                np.exp(ln_sigma),
                beta,
                None,
                None,
                Ntrials_data,
                K,
                Rh,
                dt,
            )
        result = get_criteria(spikes, ratio, dt_new)
        if true_model == "step":
            total_step += 1
        else:
            total_ramp += 1
        if result == true_model and true_model == "step":
            correct_step += 1
        elif result == true_model and true_model == "ramp":
            correct_ramp += 1
        else:
            continue
    return correct_step, correct_ramp, total_step, total_ramp


# 4) Parameters
# List of values to vary for each parameter
# Ntrials_list = [1]
# beta_list = [0.1, 0.5]
# sigma_list = [0.04, 0.2]
# m_list = [T / 4, T / 2]
# r_list = [1, 3]
# x0_list = [0, 0.2]


Ntrials_list = [25, 100]
beta_list = [0.1, 0.5, 3.5]
sigma_list = [0.04, 0.2, 3.9]
m_list = [T / 4, T / 2, 3 * T / 4]
r_list = [1, 3, 5]
x0_list = [0, 0.2, 0.5]


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
x0 = 0.2  # Fix it at 0.2


# Get lambdas for both models
lambdas_ramp = get_lambdas_ramp(K, Rh)
lambdas_step = get_lambdas_2ih(x0, Rh)


# Number of Datasets
n_datasets = 20
fraction_list = [0.25, 0.5, 0.75]


# Precompute all transition matrices once before any get_error_rates calls
# Ramp matrices
pi_ramp_cache = {}
T_ramp_cache = {}
for i, beta in enumerate(beta_values):
    for j, ln_sigma in enumerate(ln_sigma_values):
        sigma = np.exp(ln_sigma)
        pi_ramp_cache[(i, j)] = initial_distribution_ramp(x0, sigma, K, dt)
        T_ramp_cache[(i, j)] = transition_matrix_ramp(sigma, K, dt, beta)

# Step matrices
pi_step_cache = {}
T_step_cache = {}
for i, m in enumerate(m_values):
    for j, r in enumerate(r_values):
        pi_step_cache[(i, j)] = get_initial_distribution_2ih(m, r)
        T_step_cache[(i, j)] = get_transition_2ih(m, r)


# 5) Solution for 3.2.1) Matched Uniform
print("Start 3.2.1")
err_sr_321, err_rs_321 = [], []
for Ntrial in Ntrials_list:
    er, es = get_error_rates("uniform", "uniform", None, Ntrial, n_datasets, x0)
    print(f"  Ntrial={Ntrial}: P(step|ramp)={er:.3f}, P(ramp|step)={es:.3f}")
    err_sr_321.append(er)
    err_rs_321.append(es)

plt.figure(figsize=(7, 4))
plt.plot(Ntrials_list, err_sr_321, "o-", label="P(step|ramp)")
plt.plot(Ntrials_list, err_rs_321, "s--", label="P(ramp|step)")
plt.xlabel("N_trials")
plt.ylabel("error rate")
# plt.title("3.2.1: error rates vs Ntrials, uniform prior")
plt.legend()
plt.show()


# 6) Solution for 3.2.2) Sample Uniform + Infer Gaussian
print("Start 3.2.2")
err_sr = {fraction: [] for fraction in fraction_list}  # P(step|ramp)
err_rs = {fraction: [] for fraction in fraction_list}  # P(ramp|step)
for fraction in fraction_list:
    for Ntrial in Ntrials_list:
        er, es = get_error_rates(
            "uniform", "gaussian", fraction, Ntrial, n_datasets, x0
        )
        err_sr[fraction].append(er)
        err_rs[fraction].append(es)
        print(
            f"  frac={fraction}, Ntrial={Ntrial}: P(step|ramp)={er:.3f}, P(ramp|step)={es:.3f}"
        )

plt.figure(figsize=(7, 4))
for fraction in fraction_list:
    plt.plot(
        Ntrials_list, err_sr[fraction], "o-", label=f"P(step|ramp), frac={fraction}"
    )
    plt.plot(
        Ntrials_list, err_rs[fraction], "s--", label=f"P(ramp|step), frac={fraction}"
    )
plt.xlabel("N_trials")
plt.ylabel("error rate")
# plt.title("3.2.2: error rates vs dataset size, by prior SD fraction")
plt.legend()
plt.tight_layout()
plt.show()


# 7) Solution for 3.2.3) Matched Gaussian
print("Start 3.2.3")
err_sr_323, err_rs_323 = [], []
for Ntrial in Ntrials_list:
    er, es = get_error_rates("gaussian", "gaussian", 0.25, Ntrial, n_datasets, x0)
    print(f"  N={Ntrial}: P(step|ramp)={er:.3f}, P(ramp|step)={es:.3f}")
    err_sr_323.append(er)
    err_rs_323.append(es)

plt.figure(figsize=(7, 4))
plt.plot(Ntrials_list, err_sr_323, "o-", label="P(step|ramp)")
plt.plot(Ntrials_list, err_rs_323, "s--", label="P(ramp|step)")
plt.xlabel("N_trials")
plt.ylabel("error rate")
# plt.title("3.2.3: error rates vs Ntrials, matched Gaussian prior")
plt.legend()
plt.show()


# Comparison with 1.4)

# We change the number of bins
ratio = 10  # dt_new = 100 ms bins and dt = 10ms
T_new = T // ratio
dt_new = 1 / T_new  # 100 ms bins for T_new=10


# Run over multiple datasets and count correct identifications
for Ntrial in Ntrials_list:
    correct_step, correct_ramp, total_step, total_ramp = simulate(
        "gaussian", 0.25, Ntrial, n_datasets, ratio, dt_new, x0
    )
    accuracy_step = correct_step / total_step if total_step > 0 else float("nan")
    accuracy_ramp = correct_ramp / total_ramp if total_ramp > 0 else float("nan")
    accuracy_overall = (correct_step + correct_ramp) / (total_step + total_ramp)
    print(f"Ntrials: {Ntrial} - Accuracy Step: {accuracy_step:.2%}")
    print(f"Ntrials: {Ntrial} - Accuracy Ramp: {accuracy_ramp:.2%}")
    print(f"Ntrials: {Ntrial} - Accuracy Overall: {accuracy_overall:.2%}")
    if accuracy_step > 0.7 and accuracy_ramp > 0.7:
        print(f"Ntrials: {Ntrial} - The criterion distinguishes both models well.")
    elif accuracy_step > 0.7:
        print(
            f"Ntrials: {Ntrial} - The criterion identifies steps well but misclassifies ramps."
        )
    elif accuracy_ramp > 0.7:
        print(
            f"Ntrials: {Ntrial} - The criterion identifies ramps well but misses steps."
        )
    else:
        print(f"Ntrials: {Ntrial} - The criterion is not effective for either model.")
