import os

os.environ["OMP_NUM_THREADS"] = str(os.cpu_count())

import numpy as np
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.special import binom
from scipy.special import logsumexp

sys.path.append(str(Path(__file__).resolve().parents[1]))
from inference import poisson_logpdf, hmm_normalizer

# Fixed Parameters for both models
T = 100
dt = 1 / T  # 10 ms bins for T=100
Rh = 50
K = 100
M = 5


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
def get_log_prior(model, beta_values, ln_sigma_values, m_values, r_values, x0_values):
    log_x0 = -np.log(0.5)
    if model == "ramp":
        M = len(beta_values)
        prior_ramp = np.zeros((M, M, M))
        log_beta = -np.log(4)
        log_ln_sigma = -np.log(np.log(4) - np.log(0.04))
        for i in range(M):
            for j in range(M):
                prior_ramp[i, j] = log_beta + log_ln_sigma + log_x0
        return prior_ramp
    elif model == "step":
        M = len(m_values)
        prior_step = np.zeros((M, len(r_values), M))
        log_p_m = -np.log(3 * T / 4)
        log_p_r = -np.log(len(r_values))
        for i in range(M):
            for j in range(len(r_values)):
                prior_step[i, j] = log_p_m + log_p_r + log_x0
        return prior_step
    else:
        raise ValueError("Invalid model type. Choose 'ramp' or 'step'.")


# Get Posterior Probability Grid
def get_posterior(
    model,
    spike_counts,
    beta_values,
    ln_sigma_values,
    m_values,
    r_values,
    Ntrials,
    K,
    dt,
    x0_values,
):
    if model == "ramp":
        M = len(beta_values)
        sigma_values = np.exp(ln_sigma_values)
        ll_ramp = np.zeros((len(beta_values), len(ln_sigma_values), len(x0_values)))
        for i in range(M):
            for j in range(M):
                for k in range(M):
                    # Simulate data for Ramp
                    initial_prob_ramp = initial_distribution_ramp(
                        x0_values[k], sigma_values[j], K, dt
                    )
                    T_matrix_ramp = transition_matrix_ramp(
                        sigma_values[j], K, dt, beta_values[i]
                    )
                    lambdas = get_lambdas_ramp(K, Rh)
                    total_ll = 0
                    for w in range(Ntrials):
                        ll = poisson_logpdf(counts=spike_counts[w], lambdas=lambdas)
                        total_ll += hmm_normalizer(
                            pi0=initial_prob_ramp, Ps=T_matrix_ramp, ll=ll
                        )
                    ll_ramp[i, j, k] = total_ll
        prior_ramp = get_log_prior(
            "ramp", beta_values, ln_sigma_values, None, None, x0_values
        )
        log_posterior_ramp = prior_ramp + ll_ramp
        # Normalise the posterior
        log_posterior_ramp_normalised = log_posterior_ramp - logsumexp(
            log_posterior_ramp.ravel()
        )
        # Get posterior from log(posterior)
        posterior_ramp = np.exp(log_posterior_ramp_normalised)
        return posterior_ramp
    elif model == "step":
        M = len(m_values)
        ll_step = np.zeros((len(m_values), len(r_values), len(x0_values)))
        for i in range(M):
            for j in range(len(r_values)):
                for k in range(M):
                    # Simulate data for Step
                    initial_prob_step = get_initial_distribution_2ih(
                        m_values[i], r_values[j]
                    )
                    T_matrix_step = get_transition_2ih(m_values[i], r_values[j])
                    lambdas = get_lambdas_2ih(x0_values[k], Rh)
                    total_ll = 0
                    for w in range(Ntrials):
                        ll = poisson_logpdf(counts=spike_counts[w], lambdas=lambdas)
                        total_ll += hmm_normalizer(
                            pi0=initial_prob_step, Ps=T_matrix_step, ll=ll
                        )
                    ll_step[i, j, k] = total_ll
        prior_step = get_log_prior("step", None, None, m_values, r_values, x0_values)
        log_posterior_step = prior_step + ll_step
        # Normalise the posterior
        log_posterior_step_normalised = log_posterior_step - logsumexp(
            log_posterior_step.ravel()
        )
        # Get posterior from log(posterior)
        posterior_step = np.exp(log_posterior_step_normalised)
        return posterior_step
    else:
        raise ValueError("Invalid model type. Choose 'ramp' or 'step'.")


# 5) Parameters
# List of values to vary for each parameter
Ntrials_list = [2, 3]
beta_list = [0.1, 0.5]
sigma_list = [0.04, 0.2]
m_list = [T / 4, T / 2]
r_list = [1, 3]
x0_list = [0, 0.2]


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
Ntrials_standard = 2
x0_standard = 0.2
beta_standard = 0.5
sigma_standard = 0.2
ln_sigma_standard = np.log(sigma_standard)
m_standard = 50.0
r_standard = 3


# Get lambdas for both models
lambdas_ramp = get_lambdas_ramp(K, Rh)
lambdas_step = get_lambdas_2ih(x0_standard, Rh)


# 3D Heatmap


# Simulate Spike Counts
spike_counts_ramp = simulate_data(
    "ramp",
    x0_standard,
    sigma_standard,
    beta_standard,
    None,
    None,
    Ntrials_standard,
    K,
    Rh,
    dt,
)
# Simulate Spike Counts
spike_counts_step = simulate_data(
    "step",
    x0_standard,
    None,
    None,
    m_standard,
    r_standard,
    Ntrials_standard,
    K,
    Rh,
    dt,
)
# Calculate posterior grid
posterior_grid_ramp = get_posterior(
    "ramp",
    spike_counts_ramp,
    beta_values,
    ln_sigma_values,
    None,
    None,
    Ntrials_standard,
    K,
    dt,
    x0_values,
)
# Calculate posterior grid
posterior_grid_step = get_posterior(
    "step",
    spike_counts_step,
    None,
    None,
    m_values,
    r_values,
    Ntrials_standard,
    K,
    dt,
    x0_values,
)

# 3D Scatter Heatmaps
# Ramp
fig = plt.figure(figsize=(14, 6))
ax1 = fig.add_subplot(121, projection="3d")
B, S, X = np.meshgrid(beta_values, ln_sigma_values, x0_values, indexing="ij")
sc1 = ax1.scatter(
    B.ravel(),
    S.ravel(),
    X.ravel(),
    c=posterior_grid_ramp.ravel(),
    cmap="hot",
    alpha=0.6,
    s=20,
)
fig.colorbar(sc1, ax=ax1, shrink=0.55, label="Posterior probability")
ax1.set_xlabel("beta")
ax1.set_ylabel("ln sigma")
ax1.set_zlabel("x0")
sizes = posterior_grid_ramp.ravel()
sizes = 500 * sizes / sizes.max()
ax1.scatter(
    [beta_standard],
    [ln_sigma_standard],
    [x0_standard],
    color="red",
    marker="*",
    s=200,
    zorder=5,
    label="True parameters",
)
ax1.legend()
plt.tight_layout()
plt.show()


# Step
fig = plt.figure(figsize=(14, 6))
ax2 = fig.add_subplot(122, projection="3d")
MG, RG, XG = np.meshgrid(m_values, r_values, x0_values, indexing="ij")
sc2 = ax2.scatter(
    MG.ravel(),
    RG.ravel(),
    XG.ravel(),
    c=posterior_grid_step.ravel(),
    cmap="hot",
    alpha=0.6,
    s=20,
)
fig.colorbar(sc2, ax=ax2, shrink=0.55, label="Posterior probability")
ax2.set_xlabel("m")
ax2.set_ylabel("r")
ax2.set_zlabel("x0")
sizes = posterior_grid_ramp.ravel()
sizes = 500 * sizes / sizes.max()
ax2.scatter(
    [m_standard],
    [r_standard],
    [x0_standard],
    color="red",
    marker="*",
    s=200,
    zorder=5,
    label="True parameters",
)
ax2.legend()
plt.tight_layout()
plt.show()
