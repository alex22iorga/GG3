import numpy as np
import numpy.random as npr
import sys
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import signal
import seaborn as sns

sys.path.append(str(Path(__file__).resolve().parents[1]))
from models import StepModel, RampModel

# Fixed Parameters for both models
Ntrials = 5000
T = 100
x0 = 0.2
Rh = 50
dt = 1 / T

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

# StepModel label with parameters
step_label = f"StepModel\nm={m}, r={r}"

# RampModel label with parameters
ramp_label = f"RampModel\nβ={beta}, σ={sigma}"


# Plot Unsmoothed PSTH for both models on one figure
plt.figure(figsize=(10, 6))
# StepModel
plt.plot(
    np.arange(T) / (T / 1000),
    np.mean(step_spikes, axis=0) / dt,
    color="blue",
    label=step_label,
)
# RampModel
plt.plot(
    np.arange(T) / (T / 1000),
    np.mean(ramp_spikes, axis=0) / dt,
    color="green",
    label=ramp_label,
)
plt.xlabel("Time (ms)")
plt.ylabel("Firing Rate (Hz)")
plt.legend()
plt.show()


# Windowing function for smoothing PSTH
def smooth_psth(psth, window_size):
    window = signal.windows.boxcar(window_size)
    window = window / window.sum()
    return np.convolve(psth, window, mode="same")


# Use windowing function to smooth the spikes for both models
smoothed_psth_step = smooth_psth(np.mean(step_spikes, axis=0) / dt, window_size=5)
smoothed_psth_ramp = smooth_psth(np.mean(ramp_spikes, axis=0) / dt, window_size=5)


# Plot smoothed PSTH for both models
plt.figure(figsize=(10, 6))
# StepModel
plt.plot(
    np.arange(T) / (T / 1000),
    smoothed_psth_step,
    color="blue",
    label=step_label,
)
# RampModel
plt.plot(
    np.arange(T) / (T / 1000),
    smoothed_psth_ramp,
    color="green",
    label=ramp_label,
)
plt.xlabel("Time (ms)")
plt.ylabel("Firing Rate (Hz)")
plt.legend()
plt.show()


# Parameter sweep to find the most similar parameters between the two models
# We fix the values for the RampModel and vary the parameters of the StepModel
m_values = [10, 30, 50, 70, 100, 150, 200]
r_values = [1, 1.5, 2, 5, 10, 20, 50]
mse = np.zeros((len(m_values), (len(r_values))))


# For in For over the parameters
for i in range(len(m_values)):
    for j in range(len(r_values)):
        step_i = StepModel(m=m_values[i], r=r_values[j], x0=x0, Rh=Rh)
        step_spikes_i, step_jumps_i, step_rates_i = step_i.simulate(
            Ntrials=Ntrials, T=T, get_rate=True
        )
        smoothed_psth_step_i = smooth_psth(
            np.mean(step_spikes_i, axis=0) / dt, window_size=5
        )
        mse[i, j] = np.mean((smoothed_psth_step_i - smoothed_psth_ramp) ** 2)

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
plt.xlabel("r")
plt.ylabel("m")
plt.show()


# Best Parameter
min_idx = np.unravel_index(mse.argmin(), mse.shape)
print(
    f"Best parameters: m={m_values[min_idx[0]]}, r={r_values[min_idx[1]]}, MSE={mse[min_idx]:.4f}"
)

# Plot for the most similar parameters
# create models
step_new = StepModel(m=m_values[min_idx[0]], r=r_values[min_idx[1]], x0=x0, Rh=Rh)
ramp_new = RampModel(beta=beta, sigma=sigma, x0=x0, Rh=Rh)

# simulate spikes
step_spikes_new, step_jumps_new, step_rates_new = step_new.simulate(Ntrials, T)
ramp_spikes_new, ramp_xs_new, ramp_rates_new = ramp_new.simulate(Ntrials, T)

# compute PSTHs
psth_step_new = smooth_psth(np.mean(step_spikes_new, axis=0) / dt, 5)
psth_ramp_new = smooth_psth(np.mean(ramp_spikes_new, axis=0) / dt, 5)


# StepModel label with parameters
step_label_new = f"StepModel\nm={m_values[min_idx[0]]}, r={r_values[min_idx[1]]}"

# plot
plt.figure(figsize=(10, 6))
# StepModel
plt.plot(
    np.arange(T) / (T / 1000),
    psth_step_new,
    color="blue",
    label=step_label_new,
)
# RampModel
plt.plot(
    np.arange(T) / (T / 1000),
    psth_ramp_new,
    color="green",
    label=ramp_label,
)
plt.xlabel("Time (ms)")
plt.ylabel("Firing Rate (Hz)")
plt.legend()
plt.show()
