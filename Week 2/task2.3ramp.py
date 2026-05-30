import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import sys
from pathlib import Path
import seaborn as sns

sys.path.append(str(Path(__file__).resolve().parents[1]))
from inference import hmm_expected_states, poisson_logpdf

# Fixed Parameters
Ntrials = 30
T = 100
x0 = 0.2
Rh = 50
dt = 1 / T  # 10 ms bins for T=100
m = 50
r = 10
K = 100
beta = 0.5
sigma = 0.2
p = r / (m + r)


# Calculate Initial Distribution
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


# Simulate the Markov process
# Sample s_0 from initial distribution
pi_ramp = initial_distribution_ramp(x0, sigma, K, dt)
s_0_samples_ramp = np.random.choice(K, size=Ntrials, p=pi_ramp)
# Simulate s_t trajectories
T_matrix_ramp = transition_matrix_ramp(sigma, K, dt, beta)


def get_s_x_r(s_0_samples_ramp, T_matrix_ramp, K, Rh):
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
    return s_t_ramp, x_t_ramp, r_t_ramp


s_t_ramp, x_t_ramp, r_t_ramp = get_s_x_r(s_0_samples_ramp, T_matrix_ramp, K, Rh)


# Generate spike counts from firing rates
def get_n(r_t_ramp):
    n_t_ramp = np.zeros((Ntrials, T), dtype=int)
    for i in range(Ntrials):
        for t in range(T):
            # Sample from Poisson distribution with rate r_t * dt
            n_t_ramp[i, t] = np.random.poisson(r_t_ramp[i, t] * dt)
    return n_t_ramp


n_t_ramp = get_n(r_t_ramp)


# Lambdas for Poisson
def get_lambdas_ramp(K, Rh):
    lambdas_ramp = np.zeros(K)
    for s in range(K):
        x_s = s / (K - 1)
        rate_s = min(x_s, 1.0) * Rh
        lambdas_ramp[s] = rate_s * dt
    return lambdas_ramp


lambdas_ramp = get_lambdas_ramp(K, Rh)


# x values for each state
x_values = np.arange(K) / (K - 1)


# Find Posterior Probability and Expected Value of x_t
# Include both filtering and smoothing results for comparison
plt.figure(figsize=(10, 5))
for i in range(2):
    counts = n_t_ramp[i]
    ll = poisson_logpdf(counts=counts, lambdas=lambdas_ramp, mask=None)
    posterior_smooth, normaliser_smooth = hmm_expected_states(
        pi0=pi_ramp, Ps=T_matrix_ramp, ll=ll, filter=False
    )
    posterior_filter, normaliser_filter = hmm_expected_states(
        pi0=pi_ramp, Ps=T_matrix_ramp, ll=ll, filter=True
    )
    E_x_smooth = np.dot(posterior_smooth, x_values)
    E_x_filter = np.dot(posterior_filter, x_values)
    plt.plot(
        np.arange(T) / T * 1000,
        x_t_ramp[i],
        color="red",
        label="True x_t" if i == 0 else "",
    )
    plt.plot(
        np.arange(T) / T * 1000,
        E_x_smooth,
        color="blue",
        label="Expected x_t (Smoothed)" if i == 0 else "",
    )
    plt.plot(
        np.arange(T) / T * 1000,
        E_x_filter,
        color="green",
        label="Expected x_t (Filtered)" if i == 0 else "",
    )
plt.xlabel("Time (ms)")
plt.ylabel("x_t")
plt.title(f"beta={beta}, sigma={sigma}")
plt.legend()
plt.show()


# Heatmap of error
# beta and sigma
def heat_map_beta_sigma(beta_values, sigma_values):
    error1 = np.zeros((len(beta_values), (len(sigma_values))))
    error2 = np.zeros((len(beta_values), (len(sigma_values))))
    for i in range(len(beta_values)):
        for j in range(len(sigma_values)):
            # Sample s_0 from initial distribution
            pi_ramp = initial_distribution_ramp(x0, sigma_values[j], K, dt)
            s_0_samples_ramp = np.random.choice(K, size=Ntrials, p=pi_ramp)
            # Simulate s_t trajectories
            T_matrix_ramp = transition_matrix_ramp(
                sigma_values[j], K, dt, beta_values[i]
            )
            s_t_ramp, x_t_ramp, r_t_ramp = get_s_x_r(
                s_0_samples_ramp, T_matrix_ramp, K, Rh
            )
            # Generate spike counts from firing rates
            n_t_ramp = get_n(r_t_ramp)
            # Lambdas for Poisson
            lambdas_ramp = get_lambdas_ramp(K, Rh)
            # x values for each state
            x_values = np.arange(K) / (K - 1)
            # Compute errors for multiple trials
            errors_smooth = []
            errors_filter = []
            for w in range(Ntrials):
                counts = n_t_ramp[w]
                ll = poisson_logpdf(counts=counts, lambdas=lambdas_ramp, mask=None)
                posterior_smooth, normaliser_smooth = hmm_expected_states(
                    pi0=pi_ramp, Ps=T_matrix_ramp, ll=ll, filter=False
                )
                posterior_filter, normaliser_filter = hmm_expected_states(
                    pi0=pi_ramp, Ps=T_matrix_ramp, ll=ll, filter=True
                )
                E_x_smooth = np.dot(posterior_smooth, x_values)
                E_x_filter = np.dot(posterior_filter, x_values)
                errors_smooth.append(np.mean(np.abs(E_x_smooth - x_t_ramp[w])))
                errors_filter.append(np.mean(np.abs(E_x_filter - x_t_ramp[w])))
            error1[i, j] = np.mean(errors_smooth)
            error2[i, j] = np.mean(errors_filter)
    return error1, error2


# beta and x0
def heat_map_beta_x0(beta_values, x0_values):
    error1 = np.zeros((len(beta_values), (len(x0_values))))
    error2 = np.zeros((len(beta_values), (len(x0_values))))
    for i in range(len(beta_values)):
        for j in range(len(x0_values)):
            # Sample s_0 from initial distribution
            pi_ramp = initial_distribution_ramp(x0_values[j], sigma, K, dt)
            s_0_samples_ramp = np.random.choice(K, size=Ntrials, p=pi_ramp)
            # Simulate s_t trajectories
            T_matrix_ramp = transition_matrix_ramp(sigma, K, dt, beta_values[i])
            s_t_ramp, x_t_ramp, r_t_ramp = get_s_x_r(
                s_0_samples_ramp, T_matrix_ramp, K, Rh
            )
            # Generate spike counts from firing rates
            n_t_ramp = get_n(r_t_ramp)
            # Lambdas for Poisson
            lambdas_ramp = get_lambdas_ramp(K, Rh)
            # x values for each state
            x_values = np.arange(K) / (K - 1)
            # Compute errors for multiple trials
            errors_smooth = []
            errors_filter = []
            for w in range(Ntrials):
                counts = n_t_ramp[w]
                ll = poisson_logpdf(counts=counts, lambdas=lambdas_ramp, mask=None)
                posterior_smooth, normaliser_smooth = hmm_expected_states(
                    pi0=pi_ramp, Ps=T_matrix_ramp, ll=ll, filter=False
                )
                posterior_filter, normaliser_filter = hmm_expected_states(
                    pi0=pi_ramp, Ps=T_matrix_ramp, ll=ll, filter=True
                )
                E_x_smooth = np.dot(posterior_smooth, x_values)
                E_x_filter = np.dot(posterior_filter, x_values)
                errors_smooth.append(np.mean(np.abs(E_x_smooth - x_t_ramp[w])))
                errors_filter.append(np.mean(np.abs(E_x_filter - x_t_ramp[w])))
            error1[i, j] = np.mean(errors_smooth)
            error2[i, j] = np.mean(errors_filter)
    return error1, error2


# beta and K
def heat_map_beta_K(beta_values, K_values):
    error1 = np.zeros((len(beta_values), (len(K_values))))
    error2 = np.zeros((len(beta_values), (len(K_values))))
    for i in range(len(beta_values)):
        for j in range(len(K_values)):
            # Sample s_0 from initial distribution
            pi_ramp = initial_distribution_ramp(x0, sigma, K_values[j], dt)
            s_0_samples_ramp = np.random.choice(K_values[j], size=Ntrials, p=pi_ramp)
            # Simulate s_t trajectories
            T_matrix_ramp = transition_matrix_ramp(
                sigma, K_values[j], dt, beta_values[i]
            )
            s_t_ramp, x_t_ramp, r_t_ramp = get_s_x_r(
                s_0_samples_ramp, T_matrix_ramp, K_values[j], Rh
            )
            # Generate spike counts from firing rates
            n_t_ramp = get_n(r_t_ramp)
            # Lambdas for Poisson
            lambdas_ramp = get_lambdas_ramp(K_values[j], Rh)
            # x values for each state
            x_values = np.arange(K_values[j]) / (K_values[j] - 1)
            # Compute errors for multiple trials
            errors_smooth = []
            errors_filter = []
            for w in range(Ntrials):
                counts = n_t_ramp[w]
                ll = poisson_logpdf(counts=counts, lambdas=lambdas_ramp, mask=None)
                posterior_smooth, normaliser_smooth = hmm_expected_states(
                    pi0=pi_ramp, Ps=T_matrix_ramp, ll=ll, filter=False
                )
                posterior_filter, normaliser_filter = hmm_expected_states(
                    pi0=pi_ramp, Ps=T_matrix_ramp, ll=ll, filter=True
                )
                E_x_smooth = np.dot(posterior_smooth, x_values)
                E_x_filter = np.dot(posterior_filter, x_values)
                errors_smooth.append(np.mean(np.abs(E_x_smooth - x_t_ramp[w])))
                errors_filter.append(np.mean(np.abs(E_x_filter - x_t_ramp[w])))
            error1[i, j] = np.mean(errors_smooth)
            error2[i, j] = np.mean(errors_filter)
    return error1, error2


# x0 and sigma
def heat_map_x0_sigma(x0_values, sigma_values):
    error1 = np.zeros((len(x0_values), (len(sigma_values))))
    error2 = np.zeros((len(x0_values), (len(sigma_values))))
    for i in range(len(x0_values)):
        for j in range(len(sigma_values)):
            # Sample s_0 from initial distribution
            pi_ramp = initial_distribution_ramp(x0_values[i], sigma_values[j], K, dt)
            s_0_samples_ramp = np.random.choice(K, size=Ntrials, p=pi_ramp)
            # Simulate s_t trajectories
            T_matrix_ramp = transition_matrix_ramp(sigma_values[j], K, dt, beta)
            s_t_ramp, x_t_ramp, r_t_ramp = get_s_x_r(
                s_0_samples_ramp, T_matrix_ramp, K, Rh
            )
            # Generate spike counts from firing rates
            n_t_ramp = get_n(r_t_ramp)
            # Lambdas for Poisson
            lambdas_ramp = get_lambdas_ramp(K, Rh)
            # x values for each state
            x_values = np.arange(K) / (K - 1)
            # Compute errors for multiple trials
            errors_smooth = []
            errors_filter = []
            for w in range(Ntrials):
                counts = n_t_ramp[w]
                ll = poisson_logpdf(counts=counts, lambdas=lambdas_ramp, mask=None)
                posterior_smooth, normaliser_smooth = hmm_expected_states(
                    pi0=pi_ramp, Ps=T_matrix_ramp, ll=ll, filter=False
                )
                posterior_filter, normaliser_filter = hmm_expected_states(
                    pi0=pi_ramp, Ps=T_matrix_ramp, ll=ll, filter=True
                )
                E_x_smooth = np.dot(posterior_smooth, x_values)
                E_x_filter = np.dot(posterior_filter, x_values)
                errors_smooth.append(np.mean(np.abs(E_x_smooth - x_t_ramp[w])))
                errors_filter.append(np.mean(np.abs(E_x_filter - x_t_ramp[w])))
            error1[i, j] = np.mean(errors_smooth)
            error2[i, j] = np.mean(errors_filter)
    return error1, error2


# K and sigma
def heat_map_K_sigma(K_values, sigma_values):
    error1 = np.zeros((len(K_values), (len(sigma_values))))
    error2 = np.zeros((len(K_values), (len(sigma_values))))
    for i in range(len(K_values)):
        for j in range(len(sigma_values)):
            # Sample s_0 from initial distribution
            pi_ramp = initial_distribution_ramp(x0, sigma_values[j], K_values[i], dt)
            s_0_samples_ramp = np.random.choice(K_values[i], size=Ntrials, p=pi_ramp)
            # Simulate s_t trajectories
            T_matrix_ramp = transition_matrix_ramp(
                sigma_values[j], K_values[i], dt, beta
            )
            s_t_ramp, x_t_ramp, r_t_ramp = get_s_x_r(
                s_0_samples_ramp, T_matrix_ramp, K_values[i], Rh
            )
            # Generate spike counts from firing rates
            n_t_ramp = get_n(r_t_ramp)
            # Lambdas for Poisson
            lambdas_ramp = get_lambdas_ramp(K_values[i], Rh)
            # x values for each state
            x_values = np.arange(K_values[i]) / (K_values[i] - 1)
            # Compute errors for multiple trials
            errors_smooth = []
            errors_filter = []
            for w in range(Ntrials):
                counts = n_t_ramp[w]
                ll = poisson_logpdf(counts=counts, lambdas=lambdas_ramp, mask=None)
                posterior_smooth, normaliser_smooth = hmm_expected_states(
                    pi0=pi_ramp, Ps=T_matrix_ramp, ll=ll, filter=False
                )
                posterior_filter, normaliser_filter = hmm_expected_states(
                    pi0=pi_ramp, Ps=T_matrix_ramp, ll=ll, filter=True
                )
                E_x_smooth = np.dot(posterior_smooth, x_values)
                E_x_filter = np.dot(posterior_filter, x_values)
                errors_smooth.append(np.mean(np.abs(E_x_smooth - x_t_ramp[w])))
                errors_filter.append(np.mean(np.abs(E_x_filter - x_t_ramp[w])))
            error1[i, j] = np.mean(errors_smooth)
            error2[i, j] = np.mean(errors_filter)
    return error1, error2


# K and x0
def heat_map_K_x0(K_values, x0_values):
    error1 = np.zeros((len(K_values), (len(x0_values))))
    error2 = np.zeros((len(K_values), (len(x0_values))))
    for i in range(len(K_values)):
        for j in range(len(x0_values)):
            # Sample s_0 from initial distribution
            pi_ramp = initial_distribution_ramp(x0_values[j], sigma, K_values[i], dt)
            s_0_samples_ramp = np.random.choice(K_values[i], size=Ntrials, p=pi_ramp)
            # Simulate s_t trajectories
            T_matrix_ramp = transition_matrix_ramp(sigma, K_values[i], dt, beta)
            s_t_ramp, x_t_ramp, r_t_ramp = get_s_x_r(
                s_0_samples_ramp, T_matrix_ramp, K_values[i], Rh
            )
            # Generate spike counts from firing rates
            n_t_ramp = get_n(r_t_ramp)
            # Lambdas for Poisson
            lambdas_ramp = get_lambdas_ramp(K_values[i], Rh)
            # x values for each state
            x_values = np.arange(K_values[i]) / (K_values[i] - 1)
            # Compute errors for multiple trials
            errors_smooth = []
            errors_filter = []
            for w in range(Ntrials):
                counts = n_t_ramp[w]
                ll = poisson_logpdf(counts=counts, lambdas=lambdas_ramp, mask=None)
                posterior_smooth, normaliser_smooth = hmm_expected_states(
                    pi0=pi_ramp, Ps=T_matrix_ramp, ll=ll, filter=False
                )
                posterior_filter, normaliser_filter = hmm_expected_states(
                    pi0=pi_ramp, Ps=T_matrix_ramp, ll=ll, filter=True
                )
                E_x_smooth = np.dot(posterior_smooth, x_values)
                E_x_filter = np.dot(posterior_filter, x_values)
                errors_smooth.append(np.mean(np.abs(E_x_smooth - x_t_ramp[w])))
                errors_filter.append(np.mean(np.abs(E_x_filter - x_t_ramp[w])))
            error1[i, j] = np.mean(errors_smooth)
            error2[i, j] = np.mean(errors_filter)
    return error1, error2


# K and Rh
def heat_map_K_Rh(K_values, Rh_values):
    error1 = np.zeros((len(K_values), (len(Rh_values))))
    error2 = np.zeros((len(K_values), (len(Rh_values))))
    for i in range(len(K_values)):
        for j in range(len(Rh_values)):
            # Sample s_0 from initial distribution
            pi_ramp = initial_distribution_ramp(x0, sigma, K_values[i], dt)
            s_0_samples_ramp = np.random.choice(K_values[i], size=Ntrials, p=pi_ramp)
            # Simulate s_t trajectories
            T_matrix_ramp = transition_matrix_ramp(sigma, K_values[i], dt, beta)
            s_t_ramp, x_t_ramp, r_t_ramp = get_s_x_r(
                s_0_samples_ramp, T_matrix_ramp, K_values[i], Rh_values[j]
            )
            # Generate spike counts from firing rates
            n_t_ramp = get_n(r_t_ramp)
            # Lambdas for Poisson
            lambdas_ramp = get_lambdas_ramp(K_values[i], Rh_values[j])
            # x values for each state
            x_values = np.arange(K_values[i]) / (K_values[i] - 1)
            # Compute errors for multiple trials
            errors_smooth = []
            errors_filter = []
            for w in range(Ntrials):
                counts = n_t_ramp[w]
                ll = poisson_logpdf(counts=counts, lambdas=lambdas_ramp, mask=None)
                posterior_smooth, normaliser_smooth = hmm_expected_states(
                    pi0=pi_ramp, Ps=T_matrix_ramp, ll=ll, filter=False
                )
                posterior_filter, normaliser_filter = hmm_expected_states(
                    pi0=pi_ramp, Ps=T_matrix_ramp, ll=ll, filter=True
                )
                E_x_smooth = np.dot(posterior_smooth, x_values)
                E_x_filter = np.dot(posterior_filter, x_values)
                errors_smooth.append(np.mean(np.abs(E_x_smooth - x_t_ramp[w])))
                errors_filter.append(np.mean(np.abs(E_x_filter - x_t_ramp[w])))
            error1[i, j] = np.mean(errors_smooth)
            error2[i, j] = np.mean(errors_filter)
    return error1, error2


# x0 and Rh
def heat_map_x0_Rh(x0_values, Rh_values):
    error1 = np.zeros((len(x0_values), (len(Rh_values))))
    error2 = np.zeros((len(x0_values), (len(Rh_values))))
    for i in range(len(x0_values)):
        for j in range(len(Rh_values)):
            # Sample s_0 from initial distribution
            pi_ramp = initial_distribution_ramp(x0_values[i], sigma, K, dt)
            s_0_samples_ramp = np.random.choice(K, size=Ntrials, p=pi_ramp)
            # Simulate s_t trajectories
            T_matrix_ramp = transition_matrix_ramp(sigma, K, dt, beta)
            s_t_ramp, x_t_ramp, r_t_ramp = get_s_x_r(
                s_0_samples_ramp, T_matrix_ramp, K, Rh_values[j]
            )
            # Generate spike counts from firing rates
            n_t_ramp = get_n(r_t_ramp)
            # Lambdas for Poisson
            lambdas_ramp = get_lambdas_ramp(K, Rh_values[j])
            # x values for each state
            x_values = np.arange(K) / (K - 1)
            # Compute errors for multiple trials
            errors_smooth = []
            errors_filter = []
            for w in range(Ntrials):
                counts = n_t_ramp[w]
                ll = poisson_logpdf(counts=counts, lambdas=lambdas_ramp, mask=None)
                posterior_smooth, normaliser_smooth = hmm_expected_states(
                    pi0=pi_ramp, Ps=T_matrix_ramp, ll=ll, filter=False
                )
                posterior_filter, normaliser_filter = hmm_expected_states(
                    pi0=pi_ramp, Ps=T_matrix_ramp, ll=ll, filter=True
                )
                E_x_smooth = np.dot(posterior_smooth, x_values)
                E_x_filter = np.dot(posterior_filter, x_values)
                errors_smooth.append(np.mean(np.abs(E_x_smooth - x_t_ramp[w])))
                errors_filter.append(np.mean(np.abs(E_x_filter - x_t_ramp[w])))
            error1[i, j] = np.mean(errors_smooth)
            error2[i, j] = np.mean(errors_filter)
    return error1, error2


# sigma and Rh
def heat_map_sigma_Rh(sigma_values, Rh_values):
    error1 = np.zeros((len(sigma_values), (len(Rh_values))))
    error2 = np.zeros((len(sigma_values), (len(Rh_values))))
    for i in range(len(sigma_values)):
        for j in range(len(Rh_values)):
            # Sample s_0 from initial distribution
            pi_ramp = initial_distribution_ramp(x0, sigma_values[i], K, dt)
            s_0_samples_ramp = np.random.choice(K, size=Ntrials, p=pi_ramp)
            # Simulate s_t trajectories
            T_matrix_ramp = transition_matrix_ramp(sigma_values[i], K, dt, beta)
            s_t_ramp, x_t_ramp, r_t_ramp = get_s_x_r(
                s_0_samples_ramp, T_matrix_ramp, K, Rh_values[j]
            )
            # Generate spike counts from firing rates
            n_t_ramp = get_n(r_t_ramp)
            # Lambdas for Poisson
            lambdas_ramp = get_lambdas_ramp(K, Rh_values[j])
            # x values for each state
            x_values = np.arange(K) / (K - 1)
            # Compute errors for multiple trials
            errors_smooth = []
            errors_filter = []
            for w in range(Ntrials):
                counts = n_t_ramp[w]
                ll = poisson_logpdf(counts=counts, lambdas=lambdas_ramp, mask=None)
                posterior_smooth, normaliser_smooth = hmm_expected_states(
                    pi0=pi_ramp, Ps=T_matrix_ramp, ll=ll, filter=False
                )
                posterior_filter, normaliser_filter = hmm_expected_states(
                    pi0=pi_ramp, Ps=T_matrix_ramp, ll=ll, filter=True
                )
                E_x_smooth = np.dot(posterior_smooth, x_values)
                E_x_filter = np.dot(posterior_filter, x_values)
                errors_smooth.append(np.mean(np.abs(E_x_smooth - x_t_ramp[w])))
                errors_filter.append(np.mean(np.abs(E_x_filter - x_t_ramp[w])))
            error1[i, j] = np.mean(errors_smooth)
            error2[i, j] = np.mean(errors_filter)
    return error1, error2


# beta and Rh
def heat_map_beta_Rh(beta_values, Rh_values):
    error1 = np.zeros((len(beta_values), (len(Rh_values))))
    error2 = np.zeros((len(beta_values), (len(Rh_values))))
    for i in range(len(beta_values)):
        for j in range(len(Rh_values)):
            # Sample s_0 from initial distribution
            pi_ramp = initial_distribution_ramp(x0, sigma, K, dt)
            s_0_samples_ramp = np.random.choice(K, size=Ntrials, p=pi_ramp)
            # Simulate s_t trajectories
            T_matrix_ramp = transition_matrix_ramp(sigma, K, dt, beta_values[i])
            s_t_ramp, x_t_ramp, r_t_ramp = get_s_x_r(
                s_0_samples_ramp, T_matrix_ramp, K, Rh_values[j]
            )
            # Generate spike counts from firing rates
            n_t_ramp = get_n(r_t_ramp)
            # Lambdas for Poisson
            lambdas_ramp = get_lambdas_ramp(K, Rh_values[j])
            # x values for each state
            x_values = np.arange(K) / (K - 1)
            # Compute errors for multiple trials
            errors_smooth = []
            errors_filter = []
            for w in range(Ntrials):
                counts = n_t_ramp[w]
                ll = poisson_logpdf(counts=counts, lambdas=lambdas_ramp, mask=None)
                posterior_smooth, normaliser_smooth = hmm_expected_states(
                    pi0=pi_ramp, Ps=T_matrix_ramp, ll=ll, filter=False
                )
                posterior_filter, normaliser_filter = hmm_expected_states(
                    pi0=pi_ramp, Ps=T_matrix_ramp, ll=ll, filter=True
                )
                E_x_smooth = np.dot(posterior_smooth, x_values)
                E_x_filter = np.dot(posterior_filter, x_values)
                errors_smooth.append(np.mean(np.abs(E_x_smooth - x_t_ramp[w])))
                errors_filter.append(np.mean(np.abs(E_x_filter - x_t_ramp[w])))
            error1[i, j] = np.mean(errors_smooth)
            error2[i, j] = np.mean(errors_filter)
    return error1, error2


# Arrays of values
beta_values = [0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
sigma_values = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
K_values = [10, 20, 30, 50, 75, 100, 120, 150]
x0_values = [0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
Rh_values = [1, 5, 10, 20, 50, 75, 100, 250, 500, 1000]


# Find errors
error1_beta_sigma, error2_beta_sigma = heat_map_beta_sigma(beta_values, sigma_values)
error1_beta_K, error2_beta_K = heat_map_beta_K(beta_values, K_values)
error1_beta_x0, error2_beta_x0 = heat_map_beta_x0(beta_values, x0_values)
error1_K_sigma, error2_K_sigma = heat_map_K_sigma(K_values, sigma_values)
error1_x0_sigma, error2_x0_sigma = heat_map_x0_sigma(x0_values, sigma_values)
error1_K_x0, error2_K_x0 = heat_map_K_x0(K_values, x0_values)
error1_K_Rh, error2_K_Rh = heat_map_K_Rh(K_values, Rh_values)
error1_sigma_Rh, error2_sigma_Rh = heat_map_sigma_Rh(sigma_values, Rh_values)
error1_beta_Rh, error2_beta_Rh = heat_map_beta_Rh(beta_values, Rh_values)
error1_x0_Rh, error2_x0_Rh = heat_map_x0_Rh(x0_values, Rh_values)


# Plot Heatmaps for beta vs sigma
plt.figure(figsize=(10, 5))
sns.heatmap(
    error1_beta_sigma,
    annot=True,
    fmt=".5f",
    cmap="viridis",
    xticklabels=[f"{sigma:.2f}" for sigma in sigma_values],
    yticklabels=[f"{beta:.2f}" for beta in beta_values],
)
plt.xlabel("sigma")
plt.ylabel("beta")
plt.title("Error Smoother: beta vs sigma")
plt.show()

plt.figure(figsize=(10, 5))
sns.heatmap(
    error2_beta_sigma,
    annot=True,
    fmt=".5f",
    cmap="viridis",
    xticklabels=[f"{sigma:.2f}" for sigma in sigma_values],
    yticklabels=[f"{beta:.2f}" for beta in beta_values],
)
plt.xlabel("sigma")
plt.ylabel("beta")
plt.title("Error Filter: beta vs sigma")
plt.show()

# Plot Heatmaps for beta vs K
plt.figure(figsize=(10, 5))
sns.heatmap(
    error1_beta_K,
    annot=True,
    fmt=".5f",
    cmap="viridis",
    xticklabels=[f"{K:.2f}" for K in K_values],
    yticklabels=[f"{beta:.2f}" for beta in beta_values],
)
plt.xlabel("K")
plt.ylabel("beta")
plt.title("Error Smoother: beta vs K")
plt.show()

plt.figure(figsize=(10, 5))
sns.heatmap(
    error2_beta_K,
    annot=True,
    fmt=".5f",
    cmap="viridis",
    xticklabels=[f"{K:.2f}" for K in K_values],
    yticklabels=[f"{beta:.2f}" for beta in beta_values],
)
plt.xlabel("K")
plt.ylabel("beta")
plt.title("Error Filter: beta vs K")
plt.show()

# Plot Heatmaps for beta vs x0
plt.figure(figsize=(10, 5))
sns.heatmap(
    error1_beta_x0,
    annot=True,
    fmt=".5f",
    cmap="viridis",
    xticklabels=[f"{x0:.2f}" for x0 in x0_values],
    yticklabels=[f"{beta:.2f}" for beta in beta_values],
)
plt.xlabel("x0")
plt.ylabel("beta")
plt.title("Error Smoother: beta vs x0")
plt.show()

plt.figure(figsize=(10, 5))
sns.heatmap(
    error2_beta_x0,
    annot=True,
    fmt=".5f",
    cmap="viridis",
    xticklabels=[f"{x0:.2f}" for x0 in x0_values],
    yticklabels=[f"{beta:.2f}" for beta in beta_values],
)
plt.xlabel("x0")
plt.ylabel("beta")
plt.title("Error Filter: beta vs x0")
plt.show()

# Plot Heatmaps for K vs sigma
plt.figure(figsize=(10, 5))
sns.heatmap(
    error1_K_sigma,
    annot=True,
    fmt=".5f",
    cmap="viridis",
    xticklabels=[f"{sigma:.2f}" for sigma in sigma_values],
    yticklabels=[f"{K:.2f}" for K in K_values],
)
plt.xlabel("sigma")
plt.ylabel("K")
plt.title("Error Smoother: K vs sigma")
plt.show()

plt.figure(figsize=(10, 5))
sns.heatmap(
    error2_K_sigma,
    annot=True,
    fmt=".5f",
    cmap="viridis",
    xticklabels=[f"{sigma:.2f}" for sigma in sigma_values],
    yticklabels=[f"{K:.2f}" for K in K_values],
)
plt.xlabel("sigma")
plt.ylabel("K")
plt.title("Error Filter: K vs sigma")
plt.show()

# Plot Heatmaps for x0 vs sigma
plt.figure(figsize=(10, 5))
sns.heatmap(
    error1_x0_sigma,
    annot=True,
    fmt=".5f",
    cmap="viridis",
    xticklabels=[f"{sigma:.2f}" for sigma in sigma_values],
    yticklabels=[f"{x0:.2f}" for x0 in x0_values],
)
plt.xlabel("sigma")
plt.ylabel("x0")
plt.title("Error Smoother: x0 vs sigma")
plt.show()

plt.figure(figsize=(10, 5))
sns.heatmap(
    error2_x0_sigma,
    annot=True,
    fmt=".5f",
    cmap="viridis",
    xticklabels=[f"{sigma:.2f}" for sigma in sigma_values],
    yticklabels=[f"{x0:.2f}" for x0 in x0_values],
)
plt.xlabel("sigma")
plt.ylabel("x0")
plt.title("Error Filter: x0 vs sigma")
plt.show()

# Plot Heatmaps for K vs x0
plt.figure(figsize=(10, 5))
sns.heatmap(
    error1_K_x0,
    annot=True,
    fmt=".5f",
    cmap="viridis",
    xticklabels=[f"{x0:.2f}" for x0 in x0_values],
    yticklabels=[f"{K:.2f}" for K in K_values],
)
plt.xlabel("x0")
plt.ylabel("K")
plt.title("Error Smoother: K vs x0")
plt.show()

plt.figure(figsize=(10, 5))
sns.heatmap(
    error2_K_x0,
    annot=True,
    fmt=".5f",
    cmap="viridis",
    xticklabels=[f"{x0:.2f}" for x0 in x0_values],
    yticklabels=[f"{K:.2f}" for K in K_values],
)
plt.xlabel("x0")
plt.ylabel("K")
plt.title("Error Filter: K vs x0")
plt.show()


# Plot Heatmaps for K vs Rh
plt.figure(figsize=(10, 5))
sns.heatmap(
    error1_K_Rh,
    annot=True,
    fmt=".5f",
    cmap="viridis",
    xticklabels=[f"{Rh:.2f}" for Rh in Rh_values],
    yticklabels=[f"{K:.2f}" for K in K_values],
)
plt.xlabel("Rh")
plt.ylabel("K")
plt.title("Error Smoother: K vs Rh")
plt.show()

plt.figure(figsize=(10, 5))
sns.heatmap(
    error2_K_Rh,
    annot=True,
    fmt=".5f",
    cmap="viridis",
    xticklabels=[f"{Rh:.2f}" for Rh in Rh_values],
    yticklabels=[f"{K:.2f}" for K in K_values],
)
plt.xlabel("x0")
plt.ylabel("K")
plt.title("Error Filter: K vs Rh")
plt.show()


# Plot Heatmaps for sigma vs Rh
plt.figure(figsize=(10, 5))
sns.heatmap(
    error1_sigma_Rh,
    annot=True,
    fmt=".5f",
    cmap="viridis",
    xticklabels=[f"{Rh:.2f}" for Rh in Rh_values],
    yticklabels=[f"{sigma:.2f}" for sigma in sigma_values],
)
plt.xlabel("Rh")
plt.ylabel("sigma")
plt.title("Error Smoother: sigma vs Rh")
plt.show()

plt.figure(figsize=(10, 5))
sns.heatmap(
    error2_sigma_Rh,
    annot=True,
    fmt=".5f",
    cmap="viridis",
    xticklabels=[f"{Rh:.2f}" for Rh in Rh_values],
    yticklabels=[f"{sigma:.2f}" for sigma in sigma_values],
)
plt.xlabel("x0")
plt.ylabel("sigma")
plt.title("Error Filter: sigma vs Rh")
plt.show()


# Plot Heatmaps for beta vs Rh
plt.figure(figsize=(10, 5))
sns.heatmap(
    error1_beta_Rh,
    annot=True,
    fmt=".5f",
    cmap="viridis",
    xticklabels=[f"{Rh:.2f}" for Rh in Rh_values],
    yticklabels=[f"{beta:.2f}" for beta in beta_values],
)
plt.xlabel("Rh")
plt.ylabel("beta")
plt.title("Error Smoother: beta vs Rh")
plt.show()

plt.figure(figsize=(10, 5))
sns.heatmap(
    error2_beta_Rh,
    annot=True,
    fmt=".5f",
    cmap="viridis",
    xticklabels=[f"{Rh:.2f}" for Rh in Rh_values],
    yticklabels=[f"{beta:.2f}" for beta in beta_values],
)
plt.xlabel("x0")
plt.ylabel("beta")
plt.title("Error Filter: beta vs Rh")
plt.show()


# Plot Heatmaps for x0 vs Rh
plt.figure(figsize=(10, 5))
sns.heatmap(
    error1_x0_Rh,
    annot=True,
    fmt=".5f",
    cmap="viridis",
    xticklabels=[f"{Rh:.2f}" for Rh in Rh_values],
    yticklabels=[f"{x0:.2f}" for x0 in x0_values],
)
plt.xlabel("Rh")
plt.ylabel("x0")
plt.title("Error Smoother: x0 vs Rh")
plt.show()

plt.figure(figsize=(10, 5))
sns.heatmap(
    error2_x0_Rh,
    annot=True,
    fmt=".5f",
    cmap="viridis",
    xticklabels=[f"{Rh:.2f}" for Rh in Rh_values],
    yticklabels=[f"{x0:.2f}" for x0 in x0_values],
)
plt.xlabel("x0")
plt.ylabel("x0")
plt.title("Error Filter: x0 vs Rh")
plt.show()
