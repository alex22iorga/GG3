import numpy as np
import sys
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import norm

sys.path.append(str(Path(__file__).resolve().parents[1]))
from models import RampModel

# Fixed Parameters
Ntrials = 30
T = 100
x0 = 0.2
Rh = 50
dt = 1 / T  # 10 ms bins for T=100
K = 100
beta = 0.5
sigma = 0.2


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


# Plot x_t trajectories
plt.figure(figsize=(10, 5))
for i in range(10):  # plot first 10 trials
    plt.plot(np.arange(T) / T * 1000, x_t_ramp[i], label=f"Trial {i + 1}")
plt.xlabel("Time (ms)")
plt.ylabel("Discrete Latent Variable x_t")
plt.title(f"beta={beta}, sigma={sigma}, x0={x0}, K={K}")
plt.legend()
plt.show()


# Plot r_t trajectories
plt.figure(figsize=(10, 5))
for i in range(10):  # plot first 10 trials
    plt.plot(np.arange(T) / T * 1000, r_t_ramp[i], label=f"Trial {i + 1}")
plt.xlabel("Time (ms)")
plt.ylabel("Discrete Calculation Rate (Hz)")
plt.title(f"beta={beta}, sigma={sigma}, x0={x0}, K={K}")
plt.legend()
plt.show()


# Continous Time RampModel Simulation
ramp = RampModel(beta=beta, sigma=sigma, x0=x0, Rh=Rh)
ramp_spikes, ramp_xs, ramp_rates = ramp.simulate(Ntrials=Ntrials, T=T, get_rate=True)


# Plot of x_t for several trials for RampModel
plt.figure(figsize=(10, 5))
for j in range(10):  # plot first 10 trials
    plt.plot(np.arange(T) / T * 1000, ramp_xs[j], label=f"Trial {j + 1}")
plt.xlabel("Time (ms)")
plt.ylabel("Continuous Latent Variable x_t")
plt.title(f"beta={beta}, sigma={sigma}, x0={x0}, K={K}")
plt.legend()
plt.show()


# Plot of r_t for several trials for RampModel
plt.figure(figsize=(10, 5))
for j in range(10):  # plot first 10 trials
    plt.plot(np.arange(T) / T * 1000, ramp_rates[j], label=f"Trial {j + 1}")
plt.xlabel("Time (ms)")
plt.ylabel("Continuous Calculation Rate (Hz)")
plt.title(f"beta={beta}, sigma={sigma}, x0={x0}, K={K}")
plt.legend()
plt.show()


#  Compare the Continous and Discrete Models for RampModel
def compare_models(beta, sigma, x0, Rh):

    # Simulate the Markov process
    # Sample s_0 from initial distribution
    pi_ramp = initial_distribution_ramp(x0, sigma, K, dt)
    s_0_samples_ramp = np.random.choice(K, size=Ntrials, p=pi_ramp)
    # Simulate s_t trajectories
    T_matrix_ramp = transition_matrix_ramp(sigma, K, dt, beta)
    s_t_ramp, x_t_ramp, r_t_ramp = get_s_x_r(s_0_samples_ramp, T_matrix_ramp, K, Rh)
    # Continous Time RampModel Simulation
    ramp = RampModel(beta=beta, sigma=sigma, x0=x0, Rh=Rh)
    ramp_spikes, ramp_xs, ramp_rates = ramp.simulate(
        Ntrials=Ntrials, T=T, get_rate=True
    )

    # 1) Compare mean and variance of x_t across trials for both models
    # Calculate mean and variance of x_t across trials for Discrete Ramp Model
    mean_x_t = np.mean(x_t_ramp, axis=0)
    var_x_t = np.var(x_t_ramp, axis=0)

    # Calculate mean and variance of x_t across trials for Continuous Ramp Model
    mean_ramp_xs = np.mean(ramp_xs, axis=0)
    var_ramp_xs = np.var(ramp_xs, axis=0)

    # Calculate variance/mean ratio for both models
    var_mean_ratio_x_t = var_x_t / (mean_x_t + 1e-8)
    var_mean_ratio_ramp_xs = var_ramp_xs / (mean_ramp_xs + 1e-8)

    # 2) Calculate a Correlation Coefficient between mean x_t trajectories of both models
    correlation_coefficient = np.corrcoef(mean_x_t, mean_ramp_xs)[0, 1]

    # Return all the calculated metrics
    return (
        mean_x_t,
        var_x_t,
        var_mean_ratio_x_t,
        mean_ramp_xs,
        var_ramp_xs,
        var_mean_ratio_ramp_xs,
        correlation_coefficient,
    )


# Compare models across multiple parameters
beta_values = [0.005, 0.3, 1.0]
sigma_values = [0.05, 0.2, 0.4]
x0_values = [0.2, 0.5]
colors = [
    "#2196F3",
    "#4CAF50",
    "#F44336",
    "#9C27B0",
    "#FF9800",
    "#00BCD4",
    "#1A237E",
    "#1B5E20",
    "#FFC107",
    "#CDDC39",
    "#212121",
    "#9E9E9E",
    "#E91E63",
    "#795548",
    "#E65100",
    "#006064",
    "#FF6F61",
    "#6A0572",
]

# Find the results
results = {}
for beta in beta_values:
    for sigma in sigma_values:
        for x0 in x0_values:
            results[(beta, sigma, x0)] = compare_models(beta, sigma, x0, Rh)


# Correlation Coefficients for different parameter combinations
for beta in beta_values:
    for sigma in sigma_values:
        for x0 in x0_values:
            (
                mean_x_t,
                var_x_t,
                var_mean_ratio_x_t,
                mean_ramp_xs,
                var_ramp_xs,
                var_mean_ratio_ramp_xs,
                correlation_coefficient,
            ) = results[(beta, sigma, x0)]
            print(
                f"Beta: {beta}, Sigma: {sigma}, X0: {x0}, Correlation Coefficient: {correlation_coefficient:.4f}"
            )


# Plot mean of x_t for both models
plt.figure(figsize=(10, 5))
color_idx11 = 0
for beta in beta_values:
    for sigma in sigma_values:
        for x0 in x0_values:
            (
                mean_x_t,
                var_x_t,
                var_mean_ratio_x_t,
                mean_ramp_xs,
                var_ramp_xs,
                var_mean_ratio_ramp_xs,
                correlation_coefficient,
            ) = results[(beta, sigma, x0)]
            plt.plot(
                np.arange(T) / T * 1000,
                mean_x_t,
                color=colors[color_idx11],
                label=f"beta ={beta}, sigma={sigma}, x0={x0}",
            )
            color_idx11 += 1
plt.xlabel("Time (ms)")
plt.ylabel("Discrete Mean of x_t")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0)
plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 5))
color_idx12 = 0
for beta in beta_values:
    for sigma in sigma_values:
        for x0 in x0_values:
            (
                mean_x_t,
                var_x_t,
                var_mean_ratio_x_t,
                mean_ramp_xs,
                var_ramp_xs,
                var_mean_ratio_ramp_xs,
                correlation_coefficient,
            ) = results[(beta, sigma, x0)]
            plt.plot(
                np.arange(T) / T * 1000,
                mean_ramp_xs,
                color=colors[color_idx12],
                label=f"beta ={beta}, sigma={sigma}, x0={x0}",
            )
            color_idx12 += 1
plt.xlabel("Time (ms)")
plt.ylabel("Continuous Mean of x_t")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0)
plt.tight_layout()
plt.show()


# Plot variance of x_t for both models
plt.figure(figsize=(10, 5))
color_idx21 = 0
for beta in beta_values:
    for sigma in sigma_values:
        for x0 in x0_values:
            (
                mean_x_t,
                var_x_t,
                var_mean_ratio_x_t,
                mean_ramp_xs,
                var_ramp_xs,
                var_mean_ratio_ramp_xs,
                correlation_coefficient,
            ) = results[(beta, sigma, x0)]
            plt.plot(
                np.arange(T) / T * 1000,
                var_x_t,
                color=colors[color_idx21],
                label=f"beta ={beta}, sigma={sigma}, x0={x0}",
            )
            color_idx21 += 1
plt.xlabel("Time (ms)")
plt.ylabel("Discrete Variance of x_t")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0)
plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 5))
color_idx22 = 0
for beta in beta_values:
    for sigma in sigma_values:
        for x0 in x0_values:
            (
                mean_x_t,
                var_x_t,
                var_mean_ratio_x_t,
                mean_ramp_xs,
                var_ramp_xs,
                var_mean_ratio_ramp_xs,
                correlation_coefficient,
            ) = results[(beta, sigma, x0)]
            plt.plot(
                np.arange(T) / T * 1000,
                var_ramp_xs,
                color=colors[color_idx22],
                label=f"beta={beta}, sigma={sigma}, x0={x0}",
            )
            color_idx22 += 1
plt.xlabel("Time (ms)")
plt.ylabel("Continuous Variance of x_t")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0)
plt.tight_layout()
plt.show()


# Plot variance/mean ratio of x_t for both models
plt.figure(figsize=(10, 5))
color_idx31 = 0
for beta in beta_values:
    for sigma in sigma_values:
        for x0 in x0_values:
            (
                mean_x_t,
                var_x_t,
                var_mean_ratio_x_t,
                mean_ramp_xs,
                var_ramp_xs,
                var_mean_ratio_ramp_xs,
                correlation_coefficient,
            ) = results[(beta, sigma, x0)]
            plt.plot(
                np.arange(T) / T * 1000,
                var_mean_ratio_x_t,
                color=colors[color_idx31],
                label=f"beta={beta}, sigma={sigma}, x0={x0}",
            )
            color_idx31 += 1
plt.xlabel("Time (ms)")
plt.ylabel("Discrete Variance/Mean Ratio of x_t")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0)
plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 5))
color_idx32 = 0
for beta in beta_values:
    for sigma in sigma_values:
        for x0 in x0_values:
            (
                mean_x_t,
                var_x_t,
                var_mean_ratio_x_t,
                mean_ramp_xs,
                var_ramp_xs,
                var_mean_ratio_ramp_xs,
                correlation_coefficient,
            ) = results[(beta, sigma, x0)]
            plt.plot(
                np.arange(T) / T * 1000,
                var_mean_ratio_ramp_xs,
                color=colors[color_idx32],
                label=f"beta={beta}, sigma={sigma}, x0={x0}",
            )
            color_idx32 += 1
plt.xlabel("Time (ms)")
plt.ylabel("Continuous Variance/Mean Ratio of x_t")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0)
plt.tight_layout()
plt.show()
