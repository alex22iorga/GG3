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
dt = 1 / T  # 10 ms bins for T=100

# Fixed parameters for StepModel
m = 50
r = 10

# Fixed parameters for RampModel
beta = 0.5
sigma = 100


# StepModel label with parameters
step_label = f"StepModel\nm={m}, r={r}"

# RampModel label with parameters
ramp_label = f"RampModel\nβ={beta}, σ={sigma}"


# Instantiate both models with fixed parameters
step = StepModel(m=m, r=r, x0=x0, Rh=Rh)
ramp = RampModel(beta=beta, sigma=sigma, x0=x0, Rh=Rh)

# We change the number of bins
ratio = 10  # dt_new = 100 ms bins and dt = 10ms
T_new = T // ratio
dt_new = 1 / T_new  # 100 ms bins for T_new=10

# Simulate both models
step_spikes, step_jumps, step_rates = step.simulate(
    Ntrials=Ntrials, T=T_new, get_rate=True
)
ramp_spikes, ramp_xs, ramp_rates = ramp.simulate(
    Ntrials=Ntrials, T=T_new, get_rate=True
)

mean_step_rate = np.mean(step_spikes, axis=0)
mean_ramp_rate = np.mean(ramp_spikes, axis=0)

var_step_rate = np.var(step_spikes, axis=0)
var_ramp_rate = np.var(ramp_spikes, axis=0)

# Fano factor = variance / mean
fano_step = var_step_rate / (mean_step_rate + 1e-10)
fano_ramp = var_ramp_rate / (mean_ramp_rate + 1e-10)

# Plot Fano factor for both models
plt.figure(figsize=(10, 6))
plt.plot(
    np.arange(T_new) / (T_new / 1000),
    fano_step,
    color="blue",
    label=f"StepModel\nm={m}, r={r}",
)
plt.plot(
    np.arange(T_new) / (T_new / 1000),
    fano_ramp,
    color="green",
    label=f"RampModel\nβ={beta}, σ={sigma}",
)
plt.axhline(1, color="red", linestyle=":", label="Poisson (Fano=1)")
plt.xlabel("Time (ms)")
plt.ylabel("Fano Factor")
# plt.title("Fano Factor for StepModel and RampModel")
plt.legend()
plt.show()


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
            Ntrials=Ntrials, T=T_new, get_rate=True
        )
        mean_step_i = np.mean(step_spikes_i, axis=0)
        var_step_i = np.var(step_spikes_i, axis=0)
        fano_step_i = var_step_i / (mean_step_i + 1e-10)
        plt.plot(
            np.arange(T_new) / (T_new / 1000),
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
            Ntrials=Ntrials, T=T_new, get_rate=True
        )
        mean_ramp_i = np.mean(ramp_spikes_i, axis=0)
        var_ramp_i = np.var(ramp_spikes_i, axis=0)
        fano_ramp_i = var_ramp_i / (mean_ramp_i + 1e-10)
        plt.plot(
            np.arange(T_new) / (T_new / 1000),
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
mse = np.zeros((len(m_values), (len(r_values))))


# For in For over the parameters
for i in range(len(m_values)):
    for j in range(len(r_values)):
        step_i = StepModel(m=m_values[i], r=r_values[j], x0=x0, Rh=Rh)
        step_spikes_i, step_jumps_i, step_rates_i = step_i.simulate(
            Ntrials=Ntrials, T=T_new, get_rate=True
        )
        mean_step_i = np.mean(step_spikes_i, axis=0)
        var_step_i = np.var(step_spikes_i, axis=0)
        fano_step_i = var_step_i / (mean_step_i + 1e-10)
        mse[i, j] = np.mean((fano_step_i - fano_ramp) ** 2)

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
# plt.title("MSE over parameter grid")
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
step_spikes_new, step_jumps_new, step_rates_new = step_new.simulate(Ntrials, T_new)
ramp_spikes_new, ramp_xs_new, ramp_rates_new = ramp_new.simulate(Ntrials, T_new)

# Fano factor for new parameters
mean_step_new = np.mean(step_spikes_new, axis=0)
var_step_new = np.var(step_spikes_new, axis=0)
fano_step_new = var_step_new / (mean_step_new + 1e-10)


# StepModel label with parameters
step_label_new = f"StepModel\nm={m_values[min_idx[0]]}, r={r_values[min_idx[1]]}"

# plot
plt.figure(figsize=(10, 6))
# StepModel
plt.plot(
    np.arange(T_new) / (T_new / 1000),
    fano_step_new,
    color="blue",
    label=step_label_new,
)
# RampModel
plt.plot(
    np.arange(T_new) / (T_new / 1000),
    fano_ramp,
    color="green",
    label=ramp_label,
)
plt.xlabel("Time (ms)")
plt.ylabel("Fano Factor")
plt.legend()
plt.show()
