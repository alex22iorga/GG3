import os

os.environ["OMP_NUM_THREADS"] = str(os.cpu_count())

import numpy as np
import sys
import matplotlib.pyplot as plt
from pathlib import Path
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
Ntrials = 25
T = 100
x0 = 0.2
Rh = 50
dt = 1 / T
m = 50
r = 10
beta = 0.5
sigma = 0.2

# Definition of Larger Bins
ratio = 10
T_new = T // ratio
dt_new = dt * ratio  # 100 ms bins for T_new=10


# Time axis: bin centres in ms
time_ms = (np.arange(T) + 0.5) * dt * 1000  # e.g. 5, 15, ..., 995 ms

# StepModel label with parameters
step_label = f"StepModel\nm={m}, r={r}"
# RampModel label with parameters
ramp_label = f"RampModel\nβ={beta}, σ={sigma}"


# New time axis: bin centres in ms
time_ms_new = (np.arange(T_new) + 0.5) * dt_new * 1000  # e.g. 50, 150, ..., 950 ms


# Spike raster plotting function for multiple trials for StepModel
def plot_raster_step(spikes, jumps):

    # Define parameters for plotting
    Ntrials = spikes.shape[0]
    T = spikes.shape[1]

    plt.figure(figsize=(10, 6))

    # Plot spikes for each trial
    for j in range(Ntrials):
        spike_times = np.where(spikes[j] > 0)[0]
        plt.scatter(
            spike_times / T * 1000,
            np.ones_like(spike_times) * j,
            color="black",
            s=10,
            marker="o",
        )

    # Mark jump times with red X
    for j in range(Ntrials):
        if jumps[j] < T:
            plt.scatter(
                jumps[j] / T * 1000, j, marker="x", color="green", s=100, linewidths=2
            )
    plt.xlabel("Time (ms)")
    plt.ylabel("Trial")
    plt.show()


# Spike raster plotting function for multiple trials for RampModel
def plot_raster_ramp(spikes, xs):

    # Define parameters for plotting
    Ntrials = spikes.shape[0]
    T = spikes.shape[1]

    plt.figure(figsize=(10, 6))
    for j in range(Ntrials):
        spike_times = np.where(spikes[j] > 0)[0]
        plt.scatter(
            spike_times / T * 1000,
            np.ones_like(spike_times) * j,
            color="black",
            s=10,
            marker="o",
        )
    plt.xlabel("Time (ms)")
    plt.ylabel("Trial")
    plt.show()


# Bin down to 100ms bins
def bin_spikes(spikes, bin_size):
    Ntrials, T = spikes.shape
    n_bins = T // bin_size
    return spikes[:, : n_bins * bin_size].reshape(Ntrials, n_bins, bin_size).sum(axis=2)


# PSTH Function
def compute_psth(spikes, dt):
    return np.mean(spikes, axis=0) / dt


# Fano factor = variance / mean
def get_fano(spikes):
    fano = np.where(
        spikes.mean(axis=0) > 0, spikes.var(axis=0) / spikes.mean(axis=0), np.nan
    )
    return fano


# Part 1) Raster Plots
# Plot rasters for both models
for shape in [1, 2, 3, 4, 5]:
    step = StepModel(m=m, r=r, x0=x0, Rh=Rh, isi_gamma_shape=shape)
    ramp = RampModel(beta=beta, sigma=sigma, x0=x0, Rh=Rh, isi_gamma_shape=shape)
    step_spikes, step_jumps, step_rates = step.simulate(
        Ntrials=Ntrials, T=T, get_rate=True
    )
    ramp_spikes, ramp_xs, ramp_rates = ramp.simulate(
        Ntrials=Ntrials, T=T, get_rate=True
    )
    plot_raster_step(step_spikes, step_jumps)
    plot_raster_ramp(ramp_spikes, ramp_xs)


# Part 2) PSTH Plots
# Ramp Model
plt.figure(figsize=(10, 6))
for shape in [1, 2, 3, 4, 5]:
    ramp = RampModel(beta=beta, sigma=sigma, x0=x0, Rh=Rh, isi_gamma_shape=shape)
    ramp_spikes, ramp_xs, ramp_rates = ramp.simulate(
        Ntrials=Ntrials, T=T, get_rate=True
    )
    # Binned Spikes
    spikes_ramp_binned = bin_spikes(ramp_spikes, ratio)
    # Compute smoothed PSTH for both models
    smoothed_psth_ramp = compute_psth(spikes_ramp_binned, dt_new)
    # Plot smoothed PSTH for both models
    plt.plot(
        time_ms_new,
        smoothed_psth_ramp,
        label=f"shape={shape}",
    )
plt.xlabel("Time (ms)")
plt.ylabel("PSTH")
# plt.title("Ramp model: PSTH vs ISI gamma shape")
plt.legend()
plt.show()


# Step Model
plt.figure(figsize=(10, 6))
for shape in [1, 2, 3, 4, 5]:
    step = StepModel(m=m, r=r, x0=x0, Rh=Rh, isi_gamma_shape=shape)
    step_spikes, step_jumps, step_rates = step.simulate(
        Ntrials=Ntrials, T=T, get_rate=True
    )
    # Binned Spikes
    spikes_step_binned = bin_spikes(step_spikes, ratio)
    # Compute smoothed PSTH for both models
    smoothed_psth_step = compute_psth(spikes_step_binned, dt_new)
    # Plot smoothed PSTH for both models
    plt.plot(
        time_ms_new,
        smoothed_psth_step,
        label=f"shape={shape}",
    )
plt.xlabel("Time (ms)")
plt.ylabel("PSTH")
# plt.title("Step model: PSTH vs ISI gamma shape")
plt.legend()
plt.show()


# Part 3) Fano Factor
# Ramp Model
plt.figure(figsize=(10, 6))
for shape in [1, 2, 3, 4, 5]:
    ramp = RampModel(beta=beta, sigma=sigma, x0=x0, Rh=Rh, isi_gamma_shape=shape)
    ramp_spikes, ramp_xs, ramp_rates = ramp.simulate(
        Ntrials=Ntrials, T=T, get_rate=True
    )
    fano = get_fano(bin_spikes(ramp_spikes, ratio))
    plt.plot(time_ms_new, fano, label=f"shape={shape}")
plt.axhline(1, color="k", ls=":", label="Poisson (Fano=1)")
plt.xlabel("Time (ms)")
plt.ylabel("Fano factor")
# plt.title("Ramp model: Fano factor vs ISI gamma shape")
plt.legend()
plt.show()


# Step Model
plt.figure(figsize=(10, 6))
for shape in [1, 2, 3, 4, 5]:
    step = StepModel(m=m, r=r, x0=x0, Rh=Rh, isi_gamma_shape=shape)
    step_spikes, step_jumps, step_rates = step.simulate(
        Ntrials=Ntrials, T=T, get_rate=True
    )
    fano = get_fano(bin_spikes(step_spikes, ratio))
    plt.plot(time_ms_new, fano, label=f"shape={shape}")
plt.axhline(1, color="k", ls=":", label="Poisson (Fano=1)")
plt.xlabel("Time (ms)")
plt.ylabel("Fano factor")
# plt.title("Step model: Fano factor vs ISI gamma shape")
plt.legend()
plt.show()
