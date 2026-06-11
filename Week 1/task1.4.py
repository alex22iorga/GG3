import os

os.environ["OMP_NUM_THREADS"] = str(os.cpu_count())

import numpy as np
import sys
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
n_datasets = 1000
Ntrials = 400
T = 100
dt = 1 / T  # 10 ms bins for T=100
Rh = 50

# We change the number of bins
ratio = 10  # dt_new = 100 ms bins and dt = 10ms
T_new = T // ratio
dt_new = 1 / T_new  # 100 ms bins for T_new=10


# PSTH Function
def compute_psth(spikes, dt):
    return np.mean(spikes, axis=0) / dt


# Bin down to 100ms bins
def bin_spikes(spikes, bin_size):
    Ntrials, T = spikes.shape
    n_bins = T // bin_size
    return spikes[:, : n_bins * bin_size].reshape(Ntrials, n_bins, bin_size).sum(axis=2)


# Fano factor = variance / mean
def get_fano(spikes):
    mean = spikes.mean(axis=0)
    variance = spikes.var(axis=0)
    fano = np.full_like(mean, np.nan, dtype=float)
    np.divide(variance, mean, out=fano, where=mean > 0)
    return fano


# Function to evaluate criteria for a spike counter
def get_criteria(spikes):
    # Bin down to 100ms bins and compute Fano factor and PSTH
    spikes_binned = bin_spikes(spikes, ratio)
    fano = get_fano(spikes_binned)
    psth = compute_psth(spikes_binned, dt_new)

    # Criteria counter for StepModel
    count = 0

    # Criterior 1 - Maximum Fano Factor
    max_fano = np.nanmax(fano)
    if max_fano > 1.4:
        count += 1

    # Criterion 2 - PSTH Jump Size
    psth_changes = np.abs(np.diff(psth))
    max_psth_change = np.max(psth_changes) if len(psth_changes) > 0 else 0
    if max_psth_change > 4.1:
        count += 1

    # Criterion 3 -  PSTH Jump Size Compared to Total Change
    if np.sum(psth_changes) > 0:
        comparison = np.max(psth_changes) / np.sum(psth_changes)
        if comparison < 0.3:
            count += 1

    # Criterion 4 - Fano Factor Peakiness (max/mean)
    mean_fano = np.nanmean(fano)
    if mean_fano > 0:
        peakiness = np.nanmax(fano) / mean_fano
        if peakiness > 1.5:
            count += 1

    # Combine criteria
    return "step" if count >= 3 else "ramp"


# Function to simulate for multiple datasets
def simulate(n_datasets):
    correct_step = 0
    correct_ramp = 0
    total_step = 0
    total_ramp = 0
    for _ in range(n_datasets):
        m = np.random.uniform(T / 4, 3 * T / 4)
        r = np.random.uniform(0.5, 6)
        beta = np.random.uniform(0, 4)
        sigma = np.exp(np.random.uniform(np.log(0.04), np.log(4)))
        x0 = np.random.uniform(0, 0.5)

        # Randomly choose which model to use
        true_model = np.random.choice(["step", "ramp"])
        if true_model == "step":
            # Simulate data from the StepModel
            model = StepModel(m=m, r=r, x0=x0, Rh=Rh)
        else:
            # Simulate data from the RampModel
            model = RampModel(beta=beta, sigma=sigma, x0=x0, Rh=Rh)

        # Simulate the model
        spikes, _, _ = model.simulate(Ntrials=Ntrials, T=T, get_rate=True)
        result = get_criteria(spikes)
        if true_model == "step":
            total_step += 1
        else:
            total_ramp += 1
        if result == true_model and true_model == "step":
            correct_step += 1
        elif result == true_model and true_model == "ramp":
            correct_ramp += 1
        else:
            continue

    return correct_step, correct_ramp, total_step, total_ramp


# Run over multiple datasets and count correct identifications
correct_step, correct_ramp, total_step, total_ramp = simulate(n_datasets)
accuracy_step = correct_step / total_step
accuracy_ramp = correct_ramp / total_ramp
accuracy_overall = (correct_step + correct_ramp) / (total_step + total_ramp)
print(f"Accuracy Step: {accuracy_step:.2%}")
print(f"Accuracy Ramp: {accuracy_ramp:.2%}")
print(f"Accuracy Overall: {accuracy_overall:.2%}")
if accuracy_step > 0.7 and accuracy_ramp > 0.7:
    print("The criterion distinguishes both models well.")
elif accuracy_step > 0.7:
    print("The criterion identifies steps well but misclassifies ramps.")
elif accuracy_ramp > 0.7:
    print("The criterion identifies ramps well but misses steps.")
else:
    print("The criterion is not effective for either model.")
