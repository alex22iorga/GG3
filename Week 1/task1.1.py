import os

os.environ["OMP_NUM_THREADS"] = str(os.cpu_count())

import numpy as np
import numpy.random as npr
import sys
import matplotlib.pyplot as plt
from pathlib import Path
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
Ntrials = 3000
T = 100
x0 = 0.2
Rh = 50
dt = 1 / T

# Parameter Lists
m_r_list = [(50, 10), (80, 10), (50, 1000)]
beta_sigma_list = [[0.5, 0.2], [0.1, 0.2], [0.5, 100]]


# Fixed parameters for StepModel
m = 50
r = 10

# Fixed parameters for RampModel
beta = 0.5
sigma = 0.2


# Instantiate both models with fixed parameters
step = StepModel(m=m, r=r, x0=x0, Rh=Rh)
ramp = RampModel(beta=beta, sigma=sigma, x0=x0, Rh=Rh)

# Simulate both models
step_spikes, step_jumps, step_rates = step.simulate(Ntrials=Ntrials, T=T, get_rate=True)
ramp_spikes, ramp_xs, ramp_rates = ramp.simulate(Ntrials=Ntrials, T=T, get_rate=True)


# Time axis: bin centres in ms
time_ms = (np.arange(T) + 0.5) * dt * 1000  # e.g. 5, 15, ..., 995 ms


# Spike raster plotting function for multiple trials for StepModel
def plot_raster_step(spikes, jumps):

    # Define parameters for plotting
    Ntrials = spikes.shape[0]
    T = spikes.shape[1]

    plt.figure(figsize=(10, 6))

    # Plot spikes for each trial
    for j in range(min(Ntrials, 30)):
        spike_times = np.where(spikes[j] > 0)[0]
        plt.scatter(
            spike_times / T * 1000,
            np.ones_like(spike_times) * j,
            color="black",
            s=10,
            marker="o",
        )

    # Mark jump times with red X
    for j in range(min(Ntrials, 30)):
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
    for j in range(min(Ntrials, 50)):
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


# Plot rasters for StepModel
for m_loop, r_loop in m_r_list:
    step_loop = StepModel(m=m_loop, r=r_loop, x0=x0, Rh=Rh)
    step_spikes_loop, step_jumps_loop, _ = step_loop.simulate(
        Ntrials=Ntrials, T=T, get_rate=True
    )
    plot_raster_step(step_spikes_loop, step_jumps_loop)

# Plot raster for RampModel
for beta_loop, sigma_loop in beta_sigma_list:
    ramp_loop = RampModel(beta=beta_loop, sigma=sigma_loop, x0=x0, Rh=Rh)
    ramp_spikes_loop, ramp_xs_loop, ramp_rates_loop = ramp_loop.simulate(
        Ntrials=Ntrials, T=T, get_rate=True
    )
    plot_raster_ramp(ramp_spikes_loop, ramp_xs_loop)


# Histogram of jump times for StepModel
plt.figure(figsize=(10, 6))
for m_loop, r_loop in m_r_list:
    step_loop = StepModel(m=m_loop, r=r_loop, x0=x0, Rh=Rh)
    step_spikes_loop, step_jumps_loop, _ = step_loop.simulate(
        Ntrials=Ntrials, T=T, get_rate=True
    )
    plt.hist(
        step_jumps_loop / T * 1000,
        bins=20,
        label=f"m={m_loop}, r={r_loop}",
        alpha=0.7,
    )
plt.xlabel("Jump Time (ms)")
plt.ylabel("Number of Trials in each Jump Time Bin")
plt.xlim(0, 2000)
plt.legend()
plt.show()


# Plot of r_t for several trials for RampModel
plt.figure(figsize=(10, 6))
for j in range(10):  # plot first 10 trials
    plt.plot(time_ms, ramp_rates[j])
plt.xlabel("Time (ms)")
plt.ylabel("Rate (Hz)")
plt.show()


# Plot of x_t for several trials for RampModel
plt.figure(figsize=(10, 6))
for j in range(10):  # plot first 10 trials
    plt.plot(time_ms, ramp_xs[j], label=f"Trial {j + 1}")
plt.xlabel("Time (ms)")
plt.ylabel("Latent Variable x_t")
plt.legend()
plt.show()


# Histogram of when r_t crosses Rh for RampModel
plt.figure(figsize=(10, 6))
for beta_loop, sigma_loop in beta_sigma_list:
    crossing_times = []
    ramp_loop = RampModel(beta=beta_loop, sigma=sigma_loop, x0=x0, Rh=Rh)
    ramp_spikes_loop, ramp_xs_loop, ramp_rates_loop = ramp_loop.simulate(
        Ntrials=Ntrials, T=T, get_rate=True
    )
    for j in range(Ntrials):
        crossing_time = np.where(ramp_xs_loop[j] >= 1)[0]
        if len(crossing_time) > 0:
            crossing_times.append(
                crossing_time[0] / T * 1000
            )  # take the first crossing time
    plt.hist(
        crossing_times,
        bins=20,
        label=f"beta={beta_loop}, sigma={sigma_loop}",
        alpha=0.7,
    )
plt.xlabel("Crossing Time (ms)")
plt.ylabel("Number of Trials")
plt.legend()
plt.show()
