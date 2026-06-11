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


# PSTH Function
def compute_psth(spikes, dt):
    return np.mean(spikes, axis=0) / dt


# Function for plotting multiple parameters
def plot_psth(m_r_beta_sigma_values, x0, Rh, Ntrials, T, dt, ratio):
    # Larger Bins
    T_new = T // ratio
    dt_new = dt * ratio
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
        if ratio == 1:
            # Time axis: bin centres in ms
            time_ms = (np.arange(T) + 0.5) * dt * 1000
            psth_step_loop = compute_psth(step_spikes_loop, dt)
            psth_ramp_loop = compute_psth(ramp_spikes_loop, dt)
            # StepModel
            plt.plot(
                time_ms,
                psth_step_loop,
                label=get_step_label(m_loop, r_loop),
            )
            # RampModel
            plt.plot(
                time_ms,
                psth_ramp_loop,
                label=get_ramp_label(beta_loop, sigma_loop),
            )

        else:
            # Time axis: bin centres in ms
            time_ms_new = (np.arange(T_new) + 0.5) * dt_new * 1000
            smoothed_psth_step_loop = compute_psth(
                bin_spikes(step_spikes_loop, ratio), dt_new
            )
            smoothed_psth_ramp_loop = compute_psth(
                bin_spikes(ramp_spikes_loop, ratio), dt_new
            )
            # StepModel
            plt.plot(
                time_ms_new,
                smoothed_psth_step_loop,
                label=get_step_label(m_loop, r_loop),
            )
            # RampModel
            plt.plot(
                time_ms_new,
                smoothed_psth_ramp_loop,
                label=get_ramp_label(beta_loop, sigma_loop),
            )
    plt.xlabel("Time (ms)")
    plt.ylabel("PSTH")
    plt.legend()
    plt.show()


# Parameter Space
m_r_beta_sigma_values = [[50, 10, 0.5, 0.2], [80, 10, 0.1, 0.2], [50, 1000, 0.5, 100]]


# Plot Unsmoothed PSTH
plot_psth(m_r_beta_sigma_values, x0, Rh, Ntrials, T, dt, 1)


# Plot Smoothed PSTH
plot_psth(
    m_r_beta_sigma_values, x0, Rh, Ntrials, T, dt, 10
)  # ratio = 10 (100ms bins for T_new=10)


# Plot Smoothed PSTH at Different Ntrials
Ntrials_values = [30, 1000, 5000]
plt.figure(figsize=(10, 6))
for Nd in Ntrials_values:
    step_loop = StepModel(m=m, r=r, x0=x0, Rh=Rh)
    ramp_loop = RampModel(beta=beta, sigma=sigma, x0=x0, Rh=Rh)
    step_spikes_loop, step_jumps_loop, step_rates_loop = step_loop.simulate(
        Ntrials=Nd, T=T, get_rate=True
    )
    ramp_spikes_loop, ramp_xs_loop, ramp_rates_loop = ramp_loop.simulate(
        Ntrials=Nd, T=T, get_rate=True
    )

    smoothed_psth_step_loop = compute_psth(bin_spikes(step_spikes_loop, ratio), dt_new)
    smoothed_psth_ramp_loop = compute_psth(bin_spikes(ramp_spikes_loop, ratio), dt_new)
    # StepModel
    plt.plot(
        time_ms_new,
        smoothed_psth_step_loop,
        label=f" Ntrials = {Nd}, {get_step_label(m, r)}",
    )
    # RampModel
    plt.plot(
        time_ms_new,
        smoothed_psth_ramp_loop,
        label=f" Ntrials = {Nd}, {get_ramp_label(beta, sigma)}",
    )
plt.xlabel("Time (ms)")
plt.ylabel("PSTH")
plt.legend()
plt.show()


# Parameter sweep to find the most similar parameters between the two models
# We fix the values for the RampModel and vary the parameters of the StepModel
m_values = [10, 30, 50, 70, 100, 150, 200, 300, 400]
r_values = [1, 2, 5, 10, 20, 50]
beta_sigma_values = [[0.5, 0.2], [0.1, 0.2], [0.5, 100]]

best_params = []
for idx, (beta_loop, sigma_loop) in enumerate(beta_sigma_values):
    mse = np.zeros((len(m_values), (len(r_values))))
    ramp_loop = RampModel(beta=beta_loop, sigma=sigma_loop, x0=x0, Rh=Rh)
    ramp_spikes_loop, ramp_xs_loop, ramp_rates_loop = ramp_loop.simulate(
        Ntrials=Ntrials, T=T, get_rate=True
    )
    smoothed_psth_ramp_loop = compute_psth(bin_spikes(ramp_spikes_loop, ratio), dt_new)
    # For in For over the parameters
    for i in range(len(m_values)):
        for j in range(len(r_values)):
            step_i = StepModel(m=m_values[i], r=r_values[j], x0=x0, Rh=Rh)
            step_spikes_i, step_jumps_i, step_rates_i = step_i.simulate(
                Ntrials=Ntrials, T=T, get_rate=True
            )
            smoothed_psth_step_i = np.mean(bin_spikes(step_spikes_i, ratio), axis=0) / (
                dt_new
            )
            mse[i, j] = np.mean((smoothed_psth_step_i - smoothed_psth_ramp_loop) ** 2)
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
    best_params.append((beta_loop, sigma_loop, min_idx))


# Single PSTH figure with all beta/sigma pairs
plt.figure(figsize=(10, 6))
for idx, (beta_loop, sigma_loop, min_idx) in enumerate(best_params):
    # Plot for the most similar parameters
    # create models
    step_new = StepModel(m=m_values[min_idx[0]], r=r_values[min_idx[1]], x0=x0, Rh=Rh)
    ramp_new = RampModel(beta=beta_loop, sigma=sigma_loop, x0=x0, Rh=Rh)
    # simulate spikes
    step_spikes_new, step_jumps_new, step_rates_new = step_new.simulate(Ntrials, T)
    ramp_spikes_new, ramp_xs_new, ramp_rates_new = ramp_new.simulate(Ntrials, T)
    # compute smoothed PSTHs
    psth_step_new = compute_psth(bin_spikes(step_spikes_new, ratio), dt_new)
    psth_ramp_new = compute_psth(bin_spikes(ramp_spikes_new, ratio), dt_new)
    # StepModel label with parameters
    step_label_new = f"StepModel\nm={m_values[min_idx[0]]}, r={r_values[min_idx[1]]}"
    ramp_label_new = f"RampModel\nbeta={beta_loop}, sigma={sigma_loop}"
    # plot
    plt.plot(time_ms_new, psth_step_new, label=step_label_new)
    plt.plot(time_ms_new, psth_ramp_new, label=ramp_label_new)
plt.xlabel("Time (ms)")
plt.ylabel("Firing Rate (Hz)")
plt.legend()
plt.show()
