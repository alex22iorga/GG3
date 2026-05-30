import numpy as np
import sys
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.special import binom

sys.path.append(str(Path(__file__).resolve().parents[1]))
from models import StepModel

# Fixed Parameters
Ntrials = 3000
T = 100
x0 = 0.2
Rh = 50
dt = 1 / T  # 10 ms bins for T=100
m = 50
r = 10
p = r / (m + r)
m_values = [10, 50, 70, 100]
r_values = [5, 10, 20, 50]
colors = ["blue", "orange", "green", "red"]


# Part a) Plot StepModel

# Instantiate the model
step = StepModel(m=m, r=r, x0=x0, Rh=Rh)
# Simulate the model
step_spikes, step_jumps, step_rates = step.simulate(Ntrials=Ntrials, T=T, get_rate=True)


# Plot rates
plt.figure(figsize=(10, 5))
for i in range(10):  # plot first 10 trials
    plt.plot(np.arange(T) / T * 1000, step_rates[i] / Rh, label=f"Trial {i + 1}")
plt.xlabel("Time (ms)")
plt.ylabel("Discrete Latent Variable x_t")
plt.legend()
plt.show()


# Histogram of Jumping Times
plt.figure(figsize=(10, 5))
step_jumps_valid = step_jumps[~np.isnan(step_jumps)]
plt.hist(step_jumps_valid / T * 1000, bins=20)
plt.xlabel("Jump Time (ms) - StepModel")
plt.ylabel("Number of Trials in each Jump Time Bin")
plt.show()


for m_loop in m_values:
    color_idx_2h = 0
    plt.figure(figsize=(10, 5))
    for r_loop in r_values:
        step_loop = StepModel(m=m_loop, r=r_loop, x0=x0, Rh=Rh)
        step_spikes_loop, step_jumps_loop, step_rates_loop = step_loop.simulate(
            Ntrials=Ntrials, T=T, get_rate=True
        )
        plt.hist(
            step_jumps_loop / T * 1000 if len(step_jumps_loop) > 0 else [],
            bins=20,
            color=colors[color_idx_2h],
            label=f"m={m_loop}, r={r_loop}",
        )
        color_idx_2h += 1
    plt.xlabel("Jump Time (ms) - StepModel")
    plt.ylabel("Number of Trials in each Jump Time Bin")
    plt.legend()
    plt.show()


for r_loop in r_values:
    color_idx_2h = 0
    plt.figure(figsize=(10, 5))
    for m_loop in m_values:
        step_loop = StepModel(m=m_loop, r=r_loop, x0=x0, Rh=Rh)
        step_spikes_loop, step_jumps_loop, step_rates_loop = step_loop.simulate(
            Ntrials=Ntrials, T=T, get_rate=True
        )
        plt.hist(
            step_jumps_loop / T * 1000,
            bins=20,
            color=colors[color_idx_2h],
            label=f"m={m_loop}, r={r_loop}",
        )
        color_idx_2h += 1
    plt.xlabel("Jump Time (ms) - StepModel")
    plt.ylabel("Number of Trials in each Jump Time Bin")
    plt.legend()
    plt.show()


# Part b) 2 States Homogenous MC
# Initial Distribution
pi_2h = np.array([1, 0])
# Transition Matrix
T_matrix_2h = np.array([[1 - 1 / m, 1 / m], [0, 1]])

# Sample s_0 from initial distribution
x_0_samples_2h = np.random.choice(2, size=Ntrials, p=pi_2h)


# Simulate the Markov Chain
def get_s_t_2h(T_matrix_2h, x_0_samples_2h):
    s_t_2h = np.zeros((Ntrials, T), dtype=int)
    s_t_2h[:, 0] = x_0_samples_2h
    for t in range(1, T):
        for i in range(Ntrials):
            s_t_2h[i, t] = np.random.choice(2, p=T_matrix_2h[s_t_2h[i, t - 1]])
    return s_t_2h


s_t_2h = get_s_t_2h(T_matrix_2h, x_0_samples_2h)


# Find Jumping Times
def get_jump_times_2h(s_t_2h):
    jump_times_2h = np.full(Ntrials, np.nan)
    for i in range(Ntrials):
        jump_indices = np.where(s_t_2h[i] == 1)[0]
        if len(jump_indices) > 0:
            jump_times_2h[i] = jump_indices[0]
    return jump_times_2h


jump_times_2h = get_jump_times_2h(s_t_2h)

# Map MC states to the trajectory x_t
x_t_values_2h = np.where(s_t_2h == 1, 1.0, x0)

# Plot x_t
plt.figure(figsize=(10, 5))
for i in range(10):  # plot first 10 trials
    plt.plot(np.arange(T) / T * 1000, x_t_values_2h[i], label=f"Trial {i + 1}")
plt.xlabel("Time (ms) - 2 States Homogenous MC")
plt.ylabel("Discrete Latent Variable x_t")
plt.legend()
plt.show()

# Histogram of Jumping Times
plt.figure(figsize=(10, 5))
jump_times_2h_valid = jump_times_2h[~np.isnan(jump_times_2h)]
plt.hist(jump_times_2h_valid / T * 1000, bins=20)
plt.xlabel("Jump Time (ms) - 2 States Homogenous MC")
plt.ylabel("Number of Trials in each Jump Time Bin")
plt.show()


for m_loop in m_values:
    color_idx_2h = 0
    plt.figure(figsize=(10, 5))
    for r_loop in r_values:
        pi_2h_loop = np.array([1, 0])
        T_matrix_2h_loop = np.array([[1 - 1 / m_loop, 1 / m_loop], [0, 1]])
        x_0_samples_2h_loop = np.random.choice(2, size=Ntrials, p=pi_2h_loop)
        s_t_2h_loop = get_s_t_2h(T_matrix_2h_loop, x_0_samples_2h_loop)
        jump_times_2h_loop = get_jump_times_2h(s_t_2h_loop)
        jump_times_2h_loop_valid = jump_times_2h_loop[~np.isnan(jump_times_2h_loop)]
        plt.hist(
            jump_times_2h_loop_valid / T * 1000,
            bins=20,
            color=colors[color_idx_2h],
            label=f"m={m_loop}, r={r_loop}",
        )
        color_idx_2h += 1
    plt.xlabel("Jump Time (ms) - 2 States Homogenous MC")
    plt.ylabel("Number of Trials in each Jump Time Bin")
    plt.legend()
    plt.show()


for r_loop in r_values:
    color_idx_2h = 0
    plt.figure(figsize=(10, 5))
    for m_loop in m_values:
        pi_2h_loop = np.array([1, 0])
        T_matrix_2h_loop = np.array([[1 - 1 / m_loop, 1 / m_loop], [0, 1]])
        x_0_samples_2h_loop = np.random.choice(2, size=Ntrials, p=pi_2h_loop)
        s_t_2h_loop = get_s_t_2h(T_matrix_2h_loop, x_0_samples_2h_loop)
        jump_times_2h_loop = get_jump_times_2h(s_t_2h_loop)
        jump_times_2h_loop_valid = jump_times_2h_loop[~np.isnan(jump_times_2h_loop)]
        plt.hist(
            jump_times_2h_loop_valid / T * 1000,
            bins=20,
            color=colors[color_idx_2h],
            label=f"m={m_loop}, r={r_loop}",
        )
        color_idx_2h += 1
    plt.xlabel("Jump Time (ms) - 2 States Homogenous MC")
    plt.ylabel("Number of Trials in each Jump Time Bin")
    plt.legend()
    plt.show()


# Part c) (r+1) States Homogenous MC


# Initial Distribution
def get_pi_rh(r):
    pi_rh = []
    for i in range(r + 1):
        if i == 0:
            pi_rh.append(1)
        else:
            pi_rh.append(0)
    pi_rh = np.array(pi_rh)
    return pi_rh


pi_rh = get_pi_rh(r)


# Transition Matrix
def get_T_matrix_rh(m, r):
    p = r / (m + r)
    T_matrix_rh = np.zeros((r + 1, r + 1))
    for i in range(r + 1):
        if i == r:
            T_matrix_rh[i, i] = 1
        else:
            T_matrix_rh[i, i] = 1 - p
            T_matrix_rh[i, i + 1] = p
    return T_matrix_rh


T_matrix_rh = get_T_matrix_rh(m, r)


# Sample s_0 from initial distribution
s_0_samples_rh = np.random.choice(r + 1, size=Ntrials, p=pi_rh)


# Simulate the Markov Chain
def get_s_t_rh(s_0_samples_rh, T_matrix_rh, r):
    s_t_rh = np.zeros((Ntrials, T), dtype=int)
    s_t_rh[:, 0] = s_0_samples_rh
    for t in range(1, T):
        for i in range(Ntrials):
            s_t_rh[i, t] = np.random.choice(r + 1, p=T_matrix_rh[s_t_rh[i, t - 1]])
    return s_t_rh


s_t_rh = get_s_t_rh(s_0_samples_rh, T_matrix_rh, r)


# Find Jumping Times
def get_jump_times_rh(s_t_rh, r):
    jump_times_rh = np.full(Ntrials, np.nan)
    for i in range(Ntrials):
        jump_indices = np.where(s_t_rh[i] == r)[0]
        if len(jump_indices) > 0:
            jump_times_rh[i] = max(
                0, jump_indices[0] - r
            )  # shift by r to align with StepModel jump times
    return jump_times_rh


jump_times_rh = get_jump_times_rh(s_t_rh, r)


# Map MC states to the trajectory x_t
x_t_values_rh = np.where(s_t_rh == r, 1.0, x0)

# Plot x_t
plt.figure(figsize=(10, 5))
for i in range(10):  # plot first 10 trials
    plt.plot(np.arange(T) / T * 1000, x_t_values_rh[i], label=f"Trial {i + 1}")
plt.xlabel("Time (ms) - (r+1) States Homogenous MC")
plt.ylabel("Discrete Latent Variable x_t")
plt.legend()
plt.show()

# Histogram of Jumping Times
plt.figure(figsize=(10, 5))
jump_times_rh_valid = jump_times_rh[~np.isnan(jump_times_rh)]
plt.hist(jump_times_rh_valid / T * 1000, bins=20)
plt.xlabel("Jump Time (ms) - (r+1) States Homogenous MC")
plt.ylabel("Number of Trials in each Jump Time Bin")
plt.show()


for m_loop in m_values:
    color_idx_rh = 0
    plt.figure(figsize=(10, 5))
    for r_loop in r_values:
        pi_rh_loop = get_pi_rh(r_loop)
        T_matrix_rh_loop = get_T_matrix_rh(m_loop, r_loop)
        s_0_samples_rh_loop = np.random.choice(r_loop + 1, size=Ntrials, p=pi_rh_loop)
        s_t_rh_loop = get_s_t_rh(s_0_samples_rh_loop, T_matrix_rh_loop, r_loop)
        jump_times_rh_loop = get_jump_times_rh(s_t_rh_loop, r_loop)
        jump_times_rh_loop_valid = jump_times_rh_loop[~np.isnan(jump_times_rh_loop)]
        plt.hist(
            jump_times_rh_loop_valid / T * 1000,
            bins=20,
            color=colors[color_idx_rh],
            label=f"m={m_loop}, r={r_loop}",
        )
        color_idx_rh += 1
    plt.xlabel("Jump Time (ms) - (r+1) States Homogenous MC")
    plt.ylabel("Number of Trials in each Jump Time Bin")
    plt.legend()
    plt.show()


for r_loop in r_values:
    color_idx_rh = 0
    plt.figure(figsize=(10, 5))
    for m_loop in m_values:
        pi_rh_loop = get_pi_rh(r_loop)
        T_matrix_rh_loop = get_T_matrix_rh(m_loop, r_loop)
        s_0_samples_rh_loop = np.random.choice(r_loop + 1, size=Ntrials, p=pi_rh_loop)
        s_t_rh_loop = get_s_t_rh(s_0_samples_rh_loop, T_matrix_rh_loop, r_loop)
        jump_times_rh_loop = get_jump_times_rh(s_t_rh_loop, r_loop)
        jump_times_rh_loop_valid = jump_times_rh_loop[~np.isnan(jump_times_rh_loop)]
        plt.hist(
            jump_times_rh_loop_valid / T * 1000,
            bins=20,
            color=colors[color_idx_rh],
            label=f"m={m_loop}, r={r_loop}",
        )
        color_idx_rh += 1
    plt.xlabel("Jump Time (ms) - (r+1) States Homogenous MC")
    plt.ylabel("Number of Trials in each Jump Time Bin")
    plt.legend()
    plt.show()


# Part d) 2 States Inhomogenous MC

# Initial Distribution
pi_2ih = np.array([1, 0])

# Sample s_0 from initial distribution
s_0_samples_2ih = np.random.choice(2, size=Ntrials, p=pi_2ih)


# Simulate the Markov Chain
def get_s_t_2ih(m, r, s_0_samples_2ih):
    p = r / (m + r)
    s_t_2ih = np.zeros((Ntrials, T), dtype=int)
    s_t_2ih[:, 0] = s_0_samples_2ih
    T_matrix_2ih = np.zeros((T - 1, 2, 2))
    for t in range(1, T):
        prob_sum = 0
        for s in range(t):
            prob_sum += binom(s + r - 1, s) * (p**r) * ((1 - p) ** (s))
        Pt = binom(t + r - 1, t) * (p**r) * ((1 - p) ** t) / (1 - prob_sum + 1e-10)
        T_matrix_2ih_t = np.array([[1 - Pt, Pt], [0, 1]])
        T_matrix_2ih[t - 1] = T_matrix_2ih_t
        for i in range(Ntrials):
            s_t_2ih[i, t] = np.random.choice(2, p=T_matrix_2ih_t[s_t_2ih[i, t - 1]])
    return T_matrix_2ih, s_t_2ih


T_matrix_2ih, s_t_2ih = get_s_t_2ih(m, r, s_0_samples_2ih)


# Find Jumping Times
def get_jump_times_2ih(s_t_2ih):
    jump_times_2ih = np.full(Ntrials, np.nan)
    for i in range(Ntrials):
        jump_indices = np.where(s_t_2ih[i] == 1)[0]
        if len(jump_indices) > 0:
            jump_times_2ih[i] = jump_indices[0]
    return jump_times_2ih


jump_times_2ih = get_jump_times_2ih(s_t_2ih)


# Map MC states to the trajectory x_t
x_t_values_2ih = np.where(s_t_2ih == 1, 1.0, x0)


# Plot x_t
plt.figure(figsize=(10, 5))
for i in range(10):  # plot first 10 trials
    plt.plot(np.arange(T) / T * 1000, x_t_values_2ih[i], label=f"Trial {i + 1}")
plt.xlabel("Time (ms) - 2 States Inhomogenous MC")
plt.ylabel("Discrete Latent Variable x_t")
plt.legend()
plt.show()

# Histogram of Jumping Times
plt.figure(figsize=(10, 5))
jump_times_2ih_valid = jump_times_2ih[~np.isnan(jump_times_2ih)]
plt.hist(jump_times_2ih_valid / T * 1000, bins=20)
plt.xlabel("Jump Time (ms) - 2 States Inhomogenous MC")
plt.ylabel("Number of Trials in each Jump Time Bin")
plt.show()


for m_loop in m_values:
    color_idx_2ih = 0
    plt.figure(figsize=(10, 5))
    for r_loop in r_values:
        p_loop = r_loop / (m_loop + r_loop)
        pi_2ih_loop = np.array([1, 0])
        s_0_samples_2ih_loop = np.random.choice(2, size=Ntrials, p=pi_2ih_loop)
        T_matrix_2ih_loop, s_t_2ih_loop = get_s_t_2ih(
            m_loop, r_loop, s_0_samples_2ih_loop
        )
        jump_times_2ih_loop = get_jump_times_2ih(s_t_2ih_loop)
        jump_times_2ih_loop_valid = jump_times_2ih_loop[~np.isnan(jump_times_2ih_loop)]
        plt.hist(
            jump_times_2ih_loop_valid / T * 1000,
            bins=20,
            color=colors[color_idx_2ih],
            label=f"m={m_loop}, r={r_loop}",
        )
        color_idx_2ih += 1
    plt.xlabel("Jump Time (ms) - 2 States Inhomogenous MC")
    plt.ylabel("Number of Trials in each Jump Time Bin")
    plt.legend()
    plt.show()


for r_loop in r_values:
    color_idx_2ih = 0
    plt.figure(figsize=(10, 5))
    for m_loop in m_values:
        p_loop = r_loop / (m_loop + r_loop)
        pi_2ih_loop = np.array([1, 0])
        s_0_samples_2ih_loop = np.random.choice(2, size=Ntrials, p=pi_2ih_loop)
        T_matrix_2ih_loop, s_t_2ih_loop = get_s_t_2ih(
            m_loop, r_loop, s_0_samples_2ih_loop
        )
        jump_times_2ih_loop = get_jump_times_2ih(s_t_2ih_loop)
        jump_times_2ih_loop_valid = jump_times_2ih_loop[~np.isnan(jump_times_2ih_loop)]
        plt.hist(
            jump_times_2ih_loop_valid / T * 1000,
            bins=20,
            color=colors[color_idx_2ih],
            label=f"m={m_loop}, r={r_loop}",
        )
        color_idx_2ih += 1
    plt.xlabel("Jump Time (ms) - 2 States Inhomogenous MC")
    plt.ylabel("Number of Trials in each Jump Time Bin")
    plt.legend()
    plt.show()


# Part e) Compare how far away is Pt from 1/m
Pt = []
for t in range(1, T):
    prob_sum = 0
    for s in range(t):
        prob_sum += binom(s + r - 1, s) * (p**r) * ((1 - p) ** (s))
    Pt_value = binom(t + r - 1, t) * (p**r) * ((1 - p) ** t) / (1 - prob_sum + 1e-10)
    Pt.append(Pt_value)
Pt = np.array(Pt)


# Ratio between Pt and 1/m
ratio_Pt = Pt * m

plt.figure(figsize=(10, 5))
plt.plot(np.arange(len(Pt)) / T * 1000, ratio_Pt)
plt.xlabel("Time (ms) - 2 States Inhomogenous MC")
plt.ylabel("Ratio of Pt and 1/m")
plt.show()

# Calculate error
error = np.mean((Pt - 1 / m) ** 2)
ratio = np.sqrt(error) * m
print(ratio)


# Part f) Compare how good are the approximations
def compare_models(m, r):
    p = r / (m + r)
    # StepModel
    step = StepModel(m=m, r=r, x0=x0, Rh=Rh)
    step_spikes, step_jumps, step_rates = step.simulate(
        Ntrials=Ntrials, T=T, get_rate=True
    )

    # 2-State Homogenous
    pi_2h = np.array([1, 0])
    T_matrix_2h = np.array([[1 - 1 / m, 1 / m], [0, 1]])
    x_0_samples_2h = np.random.choice(2, size=Ntrials, p=pi_2h)
    s_t_2h = get_s_t_2h(T_matrix_2h, x_0_samples_2h)
    jump_times_2h = get_jump_times_2h(s_t_2h)

    # (r+1)-State Homogenous
    pi_rh = get_pi_rh(r)
    T_matrix_rh = get_T_matrix_rh(m, r)
    s_0_samples_rh = np.random.choice(r + 1, size=Ntrials, p=pi_rh)
    s_t_rh = get_s_t_rh(s_0_samples_rh, T_matrix_rh, r)
    jump_times_rh = get_jump_times_rh(s_t_rh, r)

    # 2-State Inhomogenous
    pi_2ih = np.array([1, 0])
    s_0_samples_2ih = np.random.choice(2, size=Ntrials, p=pi_2ih)
    T_matrix_2ih, s_t_2ih = get_s_t_2ih(m, r, s_0_samples_2ih)
    jump_times_2ih = get_jump_times_2ih(s_t_2ih)

    # Calculate mean of jump times
    mean_jump_Step = np.mean(step_jumps / T * 1000, axis=0)
    mean_jump_2h = np.nanmean(jump_times_2h / T * 1000, axis=0)
    mean_jump_rh = np.nanmean(jump_times_rh / T * 1000, axis=0)
    mean_jump_2ih = np.nanmean(jump_times_2ih / T * 1000, axis=0)

    # Calculate variance of jump times
    var_jump_Step = np.var(step_jumps / T * 1000, axis=0)
    var_jump_2h = np.nanvar(jump_times_2h / T * 1000, axis=0)
    var_jump_rh = np.nanvar(jump_times_rh / T * 1000, axis=0)
    var_jump_2ih = np.nanvar(jump_times_2ih / T * 1000, axis=0)

    # Calculate variance/mean ratio
    var_mean_ratio_Step = var_jump_Step / mean_jump_Step
    var_mean_ratio_2h = var_jump_2h / mean_jump_2h
    var_mean_ratio_rh = var_jump_rh / mean_jump_rh
    var_mean_ratio_2ih = var_jump_2ih / mean_jump_2ih

    # Return all the calculated metrics
    return (
        mean_jump_Step,
        mean_jump_2h,
        mean_jump_rh,
        mean_jump_2ih,
        var_jump_Step,
        var_jump_2h,
        var_jump_rh,
        var_jump_2ih,
        var_mean_ratio_Step,
        var_mean_ratio_2h,
        var_mean_ratio_rh,
        var_mean_ratio_2ih,
    )


for m_loop in m_values:
    for r_loop in r_values:
        results = compare_models(m=m_loop, r=r_loop)
        print(f"\nm={m_loop}, r={r_loop}:")
        print(
            f"  Means: Step={results[0]:.2f}, 2h={results[1]:.2f}, rh={results[2]:.2f}, 2ih={results[3]:.2f}"
        )
        print(
            f"  Variances: Step={results[4]:.2f}, 2h={results[5]:.2f}, rh={results[6]:.2f}, 2ih={results[7]:.2f}"
        )
        print(
            f"  Fano factors: Step={results[8]:.2f}, 2h={results[9]:.2f}, rh={results[10]:.2f}, 2ih={results[11]:.2f}"
        )
