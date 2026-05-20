import numpy as np
import numpy.random as npr
import sys
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from models import StepModel, RampModel


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


# Fixed Parameters for both models
Ntrials = 30
T = 100
x0 = 0.2
Rh = 50

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

# Plot rasters for both models
plot1 = plot_raster_step(step_spikes, step_jumps)
plot2 = plot_raster_ramp(ramp_spikes, ramp_xs)
plt.show()


# Histogram of jump times for StepModel
plt.figure(figsize=(8, 4))
plt.hist(step_jumps / T * 1000, bins=20)
plt.xlabel("Jump Time (ms)")
plt.ylabel("Number of Trials in each Jump Time Bin")
plt.show()


# Plot of r_t for several trials for RampModel
plt.figure(figsize=(10, 5))
for j in range(10):  # plot first 10 trials
    plt.plot(np.arange(T) / T * 1000, ramp_rates[j])
plt.xlabel("Time (ms)")
plt.ylabel("Rate (Hz)")
plt.show()


# Plot of x_t for several trials for RampModel
plt.figure(figsize=(10, 5))
for j in range(10):  # plot first 10 trials
    plt.plot(np.arange(T) / T * 1000, ramp_xs[j], label=f"Trial {j + 1}")
plt.xlabel("Time (ms)")
plt.ylabel("Latent Variable x_t")
plt.legend()
plt.show()


# Histogram of when r_t crosses Rh for RampModel
crossing_times = []
for j in range(Ntrials):
    crossing_time = np.where(ramp_xs[j] >= 1)[0]
    if len(crossing_time) > 0:
        crossing_times.append(
            crossing_time[0] / T * 1000
        )  # take the first crossing time
plt.figure(figsize=(8, 4))
plt.hist(crossing_times, bins=20)
plt.xlabel("Crossing Time (ms)")
plt.ylabel("Number of Trials in each Crossing Time Bin")
plt.show()
