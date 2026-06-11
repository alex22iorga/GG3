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
from models import gamma_isi_point_process
from inference import poisson_logpdf, hmm_normalizer

# Fixed Parameters
T = 100
dt = 1 / T  # 10 ms bins for T=100
K = 100
M = 10
Ntrials_standard = 25
n_datasets = 20  # Between 10 and 100
Rh = 25
x0 = 0.5


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
def get_n_ramp(r_t_ramp, Ntrials, isi_gamma_shape):
    n_t_ramp = np.zeros((Ntrials, T), dtype=int)
    for i in range(Ntrials):
        n_t_ramp[i, :] = gamma_isi_point_process(r_t_ramp[i, :] * dt, isi_gamma_shape)
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
def get_n_2ih(r_t_2ih, Ntrials, isi_gamma_shape):
    n_t_2ih = np.zeros((Ntrials, T), dtype=int)
    for i in range(Ntrials):
        n_t_2ih[i, :] = gamma_isi_point_process(r_t_2ih[i, :] * dt, isi_gamma_shape)
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
def simulate_data(model, x0, sigma, beta, m, r, Ntrials, K, Rh, dt, isi_gamma_shape):
    if model == "ramp":
        pi_ramp = initial_distribution_ramp(x0, sigma, K, dt)
        s_0_samples_ramp = np.random.choice(K, size=Ntrials, p=pi_ramp)
        T_matrix_ramp = transition_matrix_ramp(sigma, K, dt, beta)
        r_t_ramp = get_r_t_ramp(s_0_samples_ramp, T_matrix_ramp, K, Rh, Ntrials)
        n_t_ramp = get_n_ramp(r_t_ramp, Ntrials, isi_gamma_shape)
        return n_t_ramp
    elif model == "step":
        pi_2ih = get_initial_distribution_2ih(m, r)
        s_0_samples_2ih = np.random.choice(2, size=Ntrials, p=pi_2ih)
        s_t_2ih = get_s_t_2ih(m, r, s_0_samples_2ih, Ntrials)
        x_t_values_2ih = np.where(s_t_2ih == 1, 1.0, x0)
        r_t_2ih = get_r_t_2ih(x_t_values_2ih, Rh, Ntrials)
        n_t_2ih = get_n_2ih(r_t_2ih, Ntrials, isi_gamma_shape)
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
    sample_method, infer_method, fraction, Ntrials_data, n_datasets, x0, isi_gamma_shape
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
            "ramp",
            x0,
            np.exp(ln_sigma),
            beta,
            None,
            None,
            Ntrials_data,
            K,
            Rh,
            dt,
            isi_gamma_shape,
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
        data_step = simulate_data(
            "step", x0, None, None, m, r, Ntrials_data, K, Rh, dt, isi_gamma_shape
        )
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
def simulate(
    sample_method,
    fraction,
    Ntrials_data,
    n_datasets,
    ratio,
    dt_new,
    x0,
    isi_gamma_shape,
):
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
                true_model,
                x0,
                None,
                None,
                m,
                r,
                Ntrials_data,
                K,
                Rh,
                dt,
                isi_gamma_shape,
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
                isi_gamma_shape,
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
# beta_list = [0.1, 0.5]
# sigma_list = [0.04, 0.2]
# m_list = [T / 4, T / 2]
# r_list = [1, 3]


beta_list = [0.1, 0.5, 3.5]
sigma_list = [0.04, 0.2, 3.9]
m_list = [T / 4, T / 2, 3 * T / 4]
r_list = [1, 3, 5]


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


# Get lambdas for both models
lambdas_ramp = get_lambdas_ramp(K, Rh)
lambdas_step = get_lambdas_2ih(x0, Rh)


# Number of Datasets
fraction_list = [0.25, 0.5, 0.75]
isi_gamma_shape_list = [1, 2, 3, 4, 5]


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


# 5) Solution for 4.1.1)
# Check that the Ramp Grid gives Ramp-like PSTHs
ratio = 10
dt_new = dt * ratio
T_new = T // ratio
time_ms_new = (np.arange(T_new) + 0.5) * dt_new * 1000

np.random.seed(42)
indices = np.random.choice(
    len(beta_values) * len(ln_sigma_values), size=10, replace=False
)
beta_idx, sigma_idx = np.unravel_index(
    indices, (len(beta_values), len(ln_sigma_values))
)

plt.figure(figsize=(8, 4))
for beta, ln_sigma in zip(beta_values[beta_idx], ln_sigma_values[sigma_idx]):
    data = simulate_data(
        "ramp",
        x0,
        np.exp(ln_sigma),
        beta,
        None,
        None,
        500,
        K,
        Rh,
        dt,
        isi_gamma_shape=1,
    )
    spikes_binned = bin_spikes(data, ratio)
    psth = compute_psth(spikes_binned, dt_new)
    plt.plot(time_ms_new, psth, label=f"β={beta:.2f}, σ={np.exp(ln_sigma):.2f}")
plt.xlabel("Time (ms)")
plt.ylabel("PSTH")
# plt.title("Ramp PSTHs at 10 random grid points — checking ramp-like shape")
plt.legend(fontsize=7, ncol=2, loc="upper left")
plt.show()


er_list, es_list = [], []
for shape in isi_gamma_shape_list:
    er, es = get_error_rates(
        "uniform", "uniform", None, Ntrials_standard, n_datasets, x0, shape
    )
    print(f"  shape factor={shape}: P(step|ramp)={er:.3f}, P(ramp|step)={es:.3f}")
    er_list.append(er)
    es_list.append(es)

plt.figure(figsize=(7, 4))
plt.plot(isi_gamma_shape_list, er_list, "o-", label="P(step|ramp)")
plt.plot(isi_gamma_shape_list, es_list, "s--", label="P(ramp|step)")
plt.xlabel("ISI gamma shape")
plt.ylabel("Error Rate")
# plt.title("4.1.1: error rates vs shape parameter")
plt.legend()
plt.show()


# 6) Solution for 4.1.2)
# compare prior mismatch vs non-Poisson mismatch vs N_trials
Ntrials_list = [10, 20, 50, 200]
# Ntrials_list = [2, 3]
fraction_fixed = 0.25
shape_fixed = 3

# storage: two error directions × two mismatch types
prior_sr, prior_rs = [], []  # prior mismatch:    P(step|ramp), P(ramp|step)
gamma_sr, gamma_rs = [], []  # non-Poisson mismatch: P(step|ramp), P(ramp|step)

for Nt in Ntrials_list:
    # PRIOR mismatch: sample uniform, infer gaussian, Poisson spikes (shape=1)
    er_p, es_p = get_error_rates(
        "uniform", "gaussian", fraction_fixed, Nt, n_datasets, x0, 1
    )
    prior_sr.append(er_p)
    prior_rs.append(es_p)

    # NON-POISSON mismatch: sample uniform, infer uniform, gamma spikes (shape_fixed)
    er_g, es_g = get_error_rates(
        "uniform", "uniform", None, Nt, n_datasets, x0, shape_fixed
    )
    gamma_sr.append(er_g)
    gamma_rs.append(es_g)

    print(
        f"  N={Nt}: prior[P(step|ramp)={er_p:.3f}, P(ramp|step)={es_p:.3f}]  "
        f"gamma[P(step|ramp)={er_g:.3f}, P(ramp|step)={es_g:.3f}]"
    )

plt.figure(figsize=(8, 5))
plt.plot(Ntrials_list, prior_sr, "o-", color="C0", label="prior mismatch  P(step|ramp)")
plt.plot(
    Ntrials_list, prior_rs, "s--", color="C0", label="prior mismatch  P(ramp|step)"
)
plt.plot(Ntrials_list, gamma_sr, "o-", color="C1", label="non-Poisson  P(step|ramp)")
plt.plot(Ntrials_list, gamma_rs, "s--", color="C1", label="non-Poisson  P(ramp|step)")
plt.xscale("log")
plt.xlabel("N_trials")
plt.ylabel("Error Rate")
# plt.title("4.1.2: misclassification bias vs N_trials — prior vs non-Poisson mismatch")
plt.legend()
plt.tight_layout()
plt.show()
