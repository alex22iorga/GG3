import os

os.environ["OMP_NUM_THREADS"] = str(os.cpu_count())

import numpy as np
import numpy.random as npr
import sys
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import signal
import seaborn as sns
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
from models import StepModel, RampModel

# Fixed Parameters for both models
Ntrials = 5000
T = 100
x0 = 0.2
Rh = 50
dt = 1 / T
m = 50
r = 10
beta = 0.5
sigma = 0.2


# Time axis: bin centres in ms
time_ms = (np.arange(T) + 0.5) * dt * 1000  # e.g. 5, 15, ..., 995 ms

# Definition of Larger Bins
ratio = 10
T_new = T // ratio
dt_new = dt * ratio  # 100 ms bins for T_new=10
time_ms_new = (np.arange(T_new) + 0.5) * dt_new * 1000  # e.g. 50, 150, ..., 950 ms


# Bin down to 100ms bins
def bin_spikes(spikes, bin_size):
    Ntrials, T = spikes.shape
    n_bins = T // bin_size
    return spikes[:, : n_bins * bin_size].reshape(Ntrials, n_bins, bin_size).sum(axis=2)


# StepModel label with parameters
def get_step_label(m, r):
    return f"StepModel\nm={m}, r={r}"


# RampModel label with parameters
def get_ramp_label(beta, sigma):
    return f"RampModel\nβ={beta}, σ={sigma}"


# Fano factor = variance / mean
def get_fano(spikes):
    fano = np.where(
        spikes.mean(axis=0) > 0, spikes.var(axis=0) / spikes.mean(axis=0), np.nan
    )
    return fano


def plot_fano(m_r_beta_sigma_values, x0, Rh, Ntrials, T, dt, ratio):
    # Larger Bins
    T_new = T // ratio
    dt_new = dt * ratio
    # Time axis: bin centres in ms
    time_ms_new = (np.arange(T_new) + 0.5) * dt_new * 1000
    # Plot
    plt.figure(figsize=(10, 6))
    for m_loop, r_loop, beta_loop, sigma_loop in m_r_beta_sigma_values:
        step_loop = StepModel(m=m_loop, r=r_loop, x0=x0, Rh=Rh)
        ramp_loop = RampModel(beta=beta_loop, sigma=sigma_loop, x0=x0, Rh=Rh)
        step_spikes_loop, step_jumps_loop, step_rates_loop = step_loop.simulate(
            Ntrials=Ntrials, T=T, get_rate=True
        )
        ramp_spikes_loop, ramp_xs_loop, ramp_rates_loop = ramp_loop.simulate(
            Ntrials=Ntrials, T=T, get_rate=True
        )
        fano_step_loop = get_fano(bin_spikes(step_spikes_loop, ratio))
        fano_ramp_loop = get_fano(bin_spikes(ramp_spikes_loop, ratio))
        # StepModel
        plt.plot(
            time_ms_new,
            fano_step_loop,
            label=get_step_label(m_loop, r_loop),
        )
        # RampModel
        plt.plot(
            time_ms_new,
            fano_ramp_loop,
            label=get_ramp_label(beta_loop, sigma_loop),
        )
    plt.axhline(1, color="red", linestyle=":", label="Poisson (Fano=1)")
    plt.xlabel("Time (ms)")
    plt.ylabel("Fano Factor")
    plt.legend()
    plt.show()


# Parameter Space
m_r_beta_sigma_values = [[50, 10, 0.5, 0.2], [80, 10, 0.1, 0.2], [50, 1000, 0.5, 100]]

# Plot Fano
plot_fano(m_r_beta_sigma_values, x0, Rh, Ntrials, T, dt, ratio)


# Plot over multiple parameters for StepModel
m_values_plot = [50, 80, 100, 200]
r_values_plot = [1, 10, 100, 1000]


# Create 16 distinct colors
colors = [
    "red",
    "blue",
    "green",
    "orange",
    "purple",
    "brown",
    "pink",
    "gray",
    "olive",
    "cyan",
    "magenta",
    "gold",
    "lime",
    "navy",
    "teal",
    "maroon",
]

# For in For over the parameters
plt.figure(figsize=(10, 6))
color_idx = 0
for i in range(len(m_values_plot)):
    for j in range(len(r_values_plot)):
        step_i = StepModel(m=m_values_plot[i], r=r_values_plot[j], x0=x0, Rh=Rh)
        step_spikes_i, step_jumps_i, step_rates_i = step_i.simulate(
            Ntrials=Ntrials, T=T, get_rate=True
        )
        step_spikes_i_binned = bin_spikes(step_spikes_i, ratio)
        fano_step_i = get_fano(step_spikes_i_binned)
        plt.plot(
            time_ms_new,
            fano_step_i,
            color=colors[color_idx],
            label=f"m={m_values_plot[i]}, r={r_values_plot[j]}",
        )
        color_idx += 1
plt.xlabel("Time (ms)")
plt.ylabel("Fano Factor")
plt.legend()
plt.show()


# Plot over multiple parameters for RampModel
beta_values_plot = [0.1, 0.5, 0.8, 0.99]
sigma_values_plot = [0.2, 1, 10, 100]


# For in For over the parameters
plt.figure(figsize=(10, 6))
color_idx = 0
for i in range(len(beta_values_plot)):
    for j in range(len(sigma_values_plot)):
        ramp_i = RampModel(
            beta=beta_values_plot[i], sigma=sigma_values_plot[j], x0=x0, Rh=Rh
        )
        ramp_spikes_i, ramp_xs_i, ramp_rates_i = ramp_i.simulate(
            Ntrials=Ntrials, T=T, get_rate=True
        )
        ramp_spikes_i_binned = bin_spikes(ramp_spikes_i, ratio)
        fano_ramp_i = get_fano(ramp_spikes_i_binned)
        plt.plot(
            time_ms_new,
            fano_ramp_i,
            color=colors[color_idx],
            label=f"β={beta_values_plot[i]}, σ={sigma_values_plot[j]}",
        )
        color_idx += 1
plt.xlabel("Time (ms)")
plt.ylabel("Fano Factor")
plt.legend()
plt.show()


# Parameter sweep to find the most similar parameters between the two models
# We fix the values for the RampModel and vary the parameters of the StepModel
m_values = [1, 5, 10, 30, 50, 70, 100, 150]
r_values = [0.01, 0.1, 0.5, 1, 1.5, 2, 5]
beta_sigma_values = [[0.5, 0.2], [0.1, 0.2], [0.5, 100]]


best_params_fano = []
for beta_loop, sigma_loop in beta_sigma_values:
    mse = np.zeros((len(m_values), (len(r_values))))
    ramp_loop = RampModel(beta=beta_loop, sigma=sigma_loop, x0=x0, Rh=Rh)
    ramp_spikes_loop, ramp_xs_loop, ramp_rates_loop = ramp_loop.simulate(
        Ntrials=Ntrials, T=T, get_rate=True
    )
    fano_ramp_loop = get_fano(bin_spikes(ramp_spikes_loop, ratio))

    # For in For over the parameters
    for i in range(len(m_values)):
        for j in range(len(r_values)):
            step_i = StepModel(m=m_values[i], r=r_values[j], x0=x0, Rh=Rh)
            step_spikes_i, step_jumps_i, step_rates_i = step_i.simulate(
                Ntrials=Ntrials, T=T, get_rate=True
            )
            step_spikes_i_binned = bin_spikes(step_spikes_i, ratio)
            fano_step_i = get_fano(step_spikes_i_binned)
            mse[i, j] = np.nanmean((fano_step_i - fano_ramp_loop) ** 2)

    # Heatmap of MSE values for different parameter combinations
    mse = np.array(mse)
    plt.figure(figsize=(10, 6))
    sns.heatmap(
        mse,
        annot=True,
        fmt=".5f",
        cmap="viridis",
        xticklabels=[f"{r}" for r in r_values],
        yticklabels=[f"{m}" for m in m_values],
    )
    plt.title(f"beta={beta_loop}, sigma={sigma_loop}")
    plt.xlabel("r")
    plt.ylabel("m")
    plt.show()

    # Best Parameter
    min_idx = np.unravel_index(mse.argmin(), mse.shape)
    print(
        f"Best parameters(beta={beta_loop}, sigma={sigma_loop}): m={m_values[min_idx[0]]}, r={r_values[min_idx[1]]}, MSE={mse[min_idx]:.4f}"
    )
    best_params_fano.append((beta_loop, sigma_loop, min_idx, fano_ramp_loop))

# Single Fano figure with all beta/sigma pairs
plt.figure(figsize=(10, 6))
for beta_loop, sigma_loop, min_idx, fano_ramp_loop in best_params_fano:
    # Plot for the most similar parameters
    # create models
    step_new = StepModel(m=m_values[min_idx[0]], r=r_values[min_idx[1]], x0=x0, Rh=Rh)

    # simulate spikes
    step_spikes_new, step_jumps_new, step_rates_new = step_new.simulate(
        Ntrials, T, get_rate=True
    )

    # Fano factor for new parameters
    step_spikes_new_binned = bin_spikes(step_spikes_new, ratio)
    fano_step_new = get_fano(step_spikes_new_binned)

    # plot
    plt.plot(
        time_ms_new,
        fano_step_new,
        label=get_step_label(m_values[min_idx[0]], r_values[min_idx[1]]),
    )
    plt.plot(time_ms_new, fano_ramp_loop, label=get_ramp_label(beta_loop, sigma_loop))

plt.xlabel("Time (ms)")
plt.ylabel("Fano Factor")
plt.legend()
plt.show()
