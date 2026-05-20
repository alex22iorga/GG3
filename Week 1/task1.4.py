import numpy as np
import sys
from pathlib import Path
from scipy.optimize import curve_fit

sys.path.append(str(Path(__file__).resolve().parents[1]))
from models import StepModel, RampModel


# Fixed Parameters for both models
n_datasets = 100
Ntrials = 400
T = 100
dt = 1 / T  # 10 ms bins for T=100
Rh = 50

# We change the number of bins
ratio = 10  # dt_new = 100 ms bins and dt = 10ms
T_new = T // ratio
dt_new = 1 / T_new  # 100 ms bins for T_new=10


# Logistic function: L= max value, k=slope, x0=midpoint, b=baseline
def logistic(x, L, k, x0, b):
    return L / (1 + np.exp(-k * (x - x0))) + b


correct_datasets = 0

# Sweeping parameters for StepModel and RampModel
for i in range(n_datasets):
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

    # Simulate both models
    spikes, jumps_or_xs, rates = model.simulate(Ntrials=Ntrials, T=T_new, get_rate=True)

    # Fano factor = variance / mean
    mean = np.mean(spikes, axis=0)
    var = np.var(spikes, axis=0)
    fano = var / (mean + 1e-10)

    # Criteria counter for StepModel
    count = 0

    # Criterion 1 - Mean Fano facto
    mean_fano = np.mean(fano)
    distance_from_poisson = np.abs(mean_fano - 1.0)
    if distance_from_poisson < 0.47:
        count += 1

    # Criterion 2 - standard deviation of fano factor
    ff_std = np.std(fano)
    if ff_std > 0.25:
        count += 1

    # Criterion 3: PSTH Jump Size
    psth = mean / dt_new
    psth_changes = np.abs(np.diff(psth))
    max_psth_change = np.max(psth_changes) if len(psth_changes) > 0 else 0
    if max_psth_change < 5.2:
        count += 1

    # Criterion 4:  PSTH Jump Size Compared to Total Change
    if np.sum(psth_changes) > 0:
        comparison = np.max(psth_changes) / np.sum(psth_changes)
        if comparison < 0.3:
            count += 1

    # Criterion 5: Logistic Function Comparison
    try:
        x = np.arange(len(psth))
        y = psth

        # Initial guesses for logistic parameters
        p0 = [np.max(y) - np.min(y), 1, len(x) / 2, np.min(y)]

        # Fit
        params, covariance = curve_fit(logistic, x, y, p0=p0, maxfev=5000)

        # Predictions
        y_pred = logistic(x, params[0], params[1], params[2], params[3])

        # RMSE
        rmse = np.sqrt(np.mean((y - y_pred) ** 2))

        # Condition
        if rmse < 0.47:
            count += 1
    except Exception:
        pass

    # Combine criteria
    if count >= 3 and true_model == "step":
        correct_datasets += 1
    elif count < 3 and true_model == "ramp":
        correct_datasets += 1

print(f"Correctly identified {correct_datasets} out of {n_datasets} datasets.")

if correct_datasets > n_datasets * 0.7:
    print("The criterion is effective in distinguishing between the two models.")
else:
    print("The criterion is not effective in distinguishing between the two models.")
