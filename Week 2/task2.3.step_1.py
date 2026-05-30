import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path
import seaborn as sns

sys.path.append(str(Path(__file__).resolve().parents[1]))
from inference import hmm_expected_states, poisson_logpdf

# Fixed Parameters
Ntrials = 30
T = 100
x0 = 0.2
Rh = 50
dt = 1 / T  # 10 ms bins for T=100
m = 50
r = 10
beta = 0.5
sigma = 0.2
p = r / (m + r)


# Version 1: (r+1) States Homogenous MC


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


# Get fire rates
def get_r_t_rh(x_t_values_rh, Rh):
    r_t_rh = np.zeros((Ntrials, T), dtype=float)
    for i in range(Ntrials):
        for t in range(T):
            r_t_rh[i, t] = Rh if x_t_values_rh[i, t] >= 1 else x_t_values_rh[i, t] * Rh
    return r_t_rh


r_t_rh = get_r_t_rh(x_t_values_rh, Rh)


# Generate spike counts from firing rates
def get_n(r_t_rh):
    n_t_rh = np.zeros((Ntrials, T), dtype=int)
    for i in range(Ntrials):
        for t in range(T):
            # Sample from Poisson distribution with rate r_t * dt
            n_t_rh[i, t] = np.random.poisson(r_t_rh[i, t] * dt)
    return n_t_rh


n_t_rh = get_n(r_t_rh)


# Lambdas for Poisson
def get_lambdas_rh(r, x0, Rh):
    lambdas_rh = np.zeros(r + 1)
    for state in range(r + 1):
        if state == r:
            lambdas_rh[state] = 1.0 * Rh * dt
        else:
            lambdas_rh[state] = x0 * Rh * dt
    return lambdas_rh


lambdas_rh = get_lambdas_rh(r, x0, Rh)


# Find and plot the probability of being on the upper rate level
prob_upper_smooth = np.zeros((Ntrials, T))
prob_upper_filter = np.zeros((Ntrials, T))

plt.figure(figsize=(10, 5))
for i in range(2):
    counts = n_t_rh[i]
    ll = poisson_logpdf(counts=counts, lambdas=lambdas_rh, mask=None)
    posterior_smooth, normaliser_smooth = hmm_expected_states(
        pi0=pi_rh, Ps=T_matrix_rh, ll=ll, filter=False
    )
    posterior_filter, normaliser_filter = hmm_expected_states(
        pi0=pi_rh, Ps=T_matrix_rh, ll=ll, filter=True
    )
    prob_upper_smooth[i] = posterior_smooth[:, r]
    prob_upper_filter[i] = posterior_filter[:, r]

    inferred_jump_time_smooth = np.where(prob_upper_smooth[i] >= 0.5)[0]
    inferred_jump_time_filter = np.where(prob_upper_filter[i] >= 0.5)[0]
    if len(inferred_jump_time_smooth) > 0:
        t_smooth = inferred_jump_time_smooth[0]
        plt.plot(
            inferred_jump_time_smooth[0] / T * 1000,
            prob_upper_smooth[i][t_smooth],
            "o",
            color="orange",
            label="Inferred jump time (Smoothed" if i == 0 else "",
        )
    if len(inferred_jump_time_filter) > 0:
        t_filter = inferred_jump_time_filter[0]
        plt.plot(
            inferred_jump_time_filter[0] / T * 1000,
            prob_upper_filter[i][t_filter],
            "o",
            color="purple",
            label="Inferred jump time (Filter)" if i == 0 else "",
        )
    plt.axvline(
        x=jump_times_rh[i] * dt * 1000,
        color="red",
        label="True jump time" if i == 0 else "",
    )
    plt.plot(
        np.arange(T) / T * 1000,
        prob_upper_smooth[i],
        color="blue",
        label="Upper level Probability (Smoothed)" if i == 0 else "",
    )
    plt.plot(
        np.arange(T) / T * 1000,
        prob_upper_filter[i],
        color="green",
        label="Upper level Probability (Filtered)" if i == 0 else "",
    )
plt.xlabel("Time (ms)")
plt.ylabel("P(upper state)")
plt.title(f"m={m}, r={r}")
plt.legend()
plt.show()


# Heatmap of error
# m and r
def heat_map_m_r(m_values, r_values):
    error1 = np.zeros((len(m_values), (len(r_values))))
    error2 = np.zeros((len(m_values), (len(r_values))))
    for i in range(len(m_values)):
        for j in range(len(r_values)):
            # Sample s_0 from initial distribution
            pi_rh = get_pi_rh(r_values[j])
            T_matrix_rh = get_T_matrix_rh(m_values[i], r_values[j])
            s_0_samples_rh = np.random.choice(r_values[j] + 1, size=Ntrials, p=pi_rh)
            s_t_rh = get_s_t_rh(s_0_samples_rh, T_matrix_rh, r_values[j])
            jump_times_rh = get_jump_times_rh(s_t_rh, r_values[j])
            x_t_values_rh = np.where(s_t_rh == r_values[j], 1.0, x0)
            r_t_rh = get_r_t_rh(x_t_values_rh, Rh)
            n_t_rh = get_n(r_t_rh)
            lambdas_rh = get_lambdas_rh(r_values[j], x0, Rh)
            # Compute errors for multiple trials
            errors_smooth = []
            errors_filter = []
            for w in range(Ntrials):
                counts = n_t_rh[w]
                ll = poisson_logpdf(counts=counts, lambdas=lambdas_rh, mask=None)
                posterior_smooth, normaliser_smooth = hmm_expected_states(
                    pi0=pi_rh, Ps=T_matrix_rh, ll=ll, filter=False
                )
                posterior_filter, normaliser_filter = hmm_expected_states(
                    pi0=pi_rh, Ps=T_matrix_rh, ll=ll, filter=True
                )
                prob_upper_smooth[w] = posterior_smooth[:, r_values[j]]
                prob_upper_filter[w] = posterior_filter[:, r_values[j]]

                inferred_jump_time_smooth = np.where(prob_upper_smooth[w] >= 0.5)[0]
                inferred_jump_time_filter = np.where(prob_upper_filter[w] >= 0.5)[0]
                if len(inferred_jump_time_smooth) > 0 and ~np.isnan(jump_times_rh[w]):
                    error = (
                        abs(inferred_jump_time_smooth[0] - jump_times_rh[w]) * dt * 1000
                    )
                else:
                    error = T * dt * 1000  # = 1000ms, worst case
                errors_smooth.append(error)
                if len(inferred_jump_time_filter) > 0 and ~np.isnan(jump_times_rh[w]):
                    error = (
                        abs(inferred_jump_time_filter[0] - jump_times_rh[w]) * dt * 1000
                    )
                else:
                    error = T * dt * 1000  # = 1000ms, worst case
                errors_filter.append(error)
            error1[i, j] = np.mean(errors_smooth)
            error2[i, j] = np.mean(errors_filter)
    return error1, error2


# m and x0
def heat_map_m_x0(m_values, x0_values):
    error1 = np.zeros((len(m_values), (len(x0_values))))
    error2 = np.zeros((len(m_values), (len(x0_values))))
    for i in range(len(m_values)):
        for j in range(len(x0_values)):
            # Sample s_0 from initial distribution
            pi_rh = get_pi_rh(r)
            T_matrix_rh = get_T_matrix_rh(m_values[i], r)
            s_0_samples_rh = np.random.choice(r + 1, size=Ntrials, p=pi_rh)
            s_t_rh = get_s_t_rh(s_0_samples_rh, T_matrix_rh, r)
            jump_times_rh = get_jump_times_rh(s_t_rh, r)
            x_t_values_rh = np.where(s_t_rh == r, 1.0, x0_values[j])
            r_t_rh = get_r_t_rh(x_t_values_rh, Rh)
            n_t_rh = get_n(r_t_rh)
            lambdas_rh = get_lambdas_rh(r, x0_values[j], Rh)
            # Compute errors for multiple trials
            errors_smooth = []
            errors_filter = []
            for w in range(Ntrials):
                counts = n_t_rh[w]
                ll = poisson_logpdf(counts=counts, lambdas=lambdas_rh, mask=None)
                posterior_smooth, normaliser_smooth = hmm_expected_states(
                    pi0=pi_rh, Ps=T_matrix_rh, ll=ll, filter=False
                )
                posterior_filter, normaliser_filter = hmm_expected_states(
                    pi0=pi_rh, Ps=T_matrix_rh, ll=ll, filter=True
                )
                prob_upper_smooth[w] = posterior_smooth[:, r]
                prob_upper_filter[w] = posterior_filter[:, r]

                inferred_jump_time_smooth = np.where(prob_upper_smooth[w] >= 0.5)[0]
                inferred_jump_time_filter = np.where(prob_upper_filter[w] >= 0.5)[0]
                if len(inferred_jump_time_smooth) > 0 and ~np.isnan(jump_times_rh[w]):
                    error = (
                        abs(inferred_jump_time_smooth[0] - jump_times_rh[w]) * dt * 1000
                    )
                else:
                    error = T * dt * 1000  # = 1000ms, worst case
                errors_smooth.append(error)
                if len(inferred_jump_time_filter) > 0 and ~np.isnan(jump_times_rh[w]):
                    error = (
                        abs(inferred_jump_time_filter[0] - jump_times_rh[w]) * dt * 1000
                    )
                else:
                    error = T * dt * 1000  # = 1000ms, worst case
                errors_filter.append(error)
            error1[i, j] = np.mean(errors_smooth)
            error2[i, j] = np.mean(errors_filter)
    return error1, error2


# x0 and r
def heat_map_x0_r(x0_values, r_values):
    error1 = np.zeros((len(x0_values), (len(r_values))))
    error2 = np.zeros((len(x0_values), (len(r_values))))
    for i in range(len(x0_values)):
        for j in range(len(r_values)):
            # Sample s_0 from initial distribution
            pi_rh = get_pi_rh(r_values[j])
            T_matrix_rh = get_T_matrix_rh(m, r_values[j])
            s_0_samples_rh = np.random.choice(r_values[j] + 1, size=Ntrials, p=pi_rh)
            s_t_rh = get_s_t_rh(s_0_samples_rh, T_matrix_rh, r_values[j])
            jump_times_rh = get_jump_times_rh(s_t_rh, r_values[j])
            x_t_values_rh = np.where(s_t_rh == r_values[j], 1.0, x0_values[i])
            r_t_rh = get_r_t_rh(x_t_values_rh, Rh)
            n_t_rh = get_n(r_t_rh)
            lambdas_rh = get_lambdas_rh(r_values[j], x0_values[i], Rh)
            # Compute errors for multiple trials
            errors_smooth = []
            errors_filter = []
            for w in range(Ntrials):
                counts = n_t_rh[w]
                ll = poisson_logpdf(counts=counts, lambdas=lambdas_rh, mask=None)
                posterior_smooth, normaliser_smooth = hmm_expected_states(
                    pi0=pi_rh, Ps=T_matrix_rh, ll=ll, filter=False
                )
                posterior_filter, normaliser_filter = hmm_expected_states(
                    pi0=pi_rh, Ps=T_matrix_rh, ll=ll, filter=True
                )
                prob_upper_smooth[w] = posterior_smooth[:, r_values[j]]
                prob_upper_filter[w] = posterior_filter[:, r_values[j]]

                inferred_jump_time_smooth = np.where(prob_upper_smooth[w] >= 0.5)[0]
                inferred_jump_time_filter = np.where(prob_upper_filter[w] >= 0.5)[0]
                if len(inferred_jump_time_smooth) > 0 and ~np.isnan(jump_times_rh[w]):
                    error = (
                        abs(inferred_jump_time_smooth[0] - jump_times_rh[w]) * dt * 1000
                    )
                else:
                    error = T * dt * 1000  # = 1000ms, worst case
                errors_smooth.append(error)
                if len(inferred_jump_time_filter) > 0 and ~np.isnan(jump_times_rh[w]):
                    error = (
                        abs(inferred_jump_time_filter[0] - jump_times_rh[w]) * dt * 1000
                    )
                else:
                    error = T * dt * 1000  # = 1000ms, worst case
                errors_filter.append(error)
            error1[i, j] = np.mean(errors_smooth)
            error2[i, j] = np.mean(errors_filter)
    return error1, error2


# x0 and Rh
def heat_map_x0_Rh(x0_values, Rh_values):
    error1 = np.zeros((len(x0_values), (len(Rh_values))))
    error2 = np.zeros((len(x0_values), (len(Rh_values))))
    for i in range(len(x0_values)):
        for j in range(len(Rh_values)):
            # Sample s_0 from initial distribution
            pi_rh = get_pi_rh(r)
            T_matrix_rh = get_T_matrix_rh(m, r)
            s_0_samples_rh = np.random.choice(r + 1, size=Ntrials, p=pi_rh)
            s_t_rh = get_s_t_rh(s_0_samples_rh, T_matrix_rh, r)
            jump_times_rh = get_jump_times_rh(s_t_rh, r)
            x_t_values_rh = np.where(s_t_rh == r, 1.0, x0_values[i])
            r_t_rh = get_r_t_rh(x_t_values_rh, Rh_values[j])
            n_t_rh = get_n(r_t_rh)
            lambdas_rh = get_lambdas_rh(r, x0_values[i], Rh_values[j])
            # Compute errors for multiple trials
            errors_smooth = []
            errors_filter = []
            for w in range(Ntrials):
                counts = n_t_rh[w]
                ll = poisson_logpdf(counts=counts, lambdas=lambdas_rh, mask=None)
                posterior_smooth, normaliser_smooth = hmm_expected_states(
                    pi0=pi_rh, Ps=T_matrix_rh, ll=ll, filter=False
                )
                posterior_filter, normaliser_filter = hmm_expected_states(
                    pi0=pi_rh, Ps=T_matrix_rh, ll=ll, filter=True
                )
                prob_upper_smooth[w] = posterior_smooth[:, r]
                prob_upper_filter[w] = posterior_filter[:, r]

                inferred_jump_time_smooth = np.where(prob_upper_smooth[w] >= 0.5)[0]
                inferred_jump_time_filter = np.where(prob_upper_filter[w] >= 0.5)[0]
                if len(inferred_jump_time_smooth) > 0 and ~np.isnan(jump_times_rh[w]):
                    error = (
                        abs(inferred_jump_time_smooth[0] - jump_times_rh[w]) * dt * 1000
                    )
                else:
                    error = T * dt * 1000  # = 1000ms, worst case
                errors_smooth.append(error)
                if len(inferred_jump_time_filter) > 0 and ~np.isnan(jump_times_rh[w]):
                    error = (
                        abs(inferred_jump_time_filter[0] - jump_times_rh[w]) * dt * 1000
                    )
                else:
                    error = T * dt * 1000  # = 1000ms, worst case
                errors_filter.append(error)
            error1[i, j] = np.mean(errors_smooth)
            error2[i, j] = np.mean(errors_filter)
    return error1, error2


# m and Rh
def heat_map_m_Rh(m_values, Rh_values):
    error1 = np.zeros((len(m_values), (len(Rh_values))))
    error2 = np.zeros((len(m_values), (len(Rh_values))))
    for i in range(len(m_values)):
        for j in range(len(Rh_values)):
            # Sample s_0 from initial distribution
            pi_rh = get_pi_rh(r)
            T_matrix_rh = get_T_matrix_rh(m_values[i], r)
            s_0_samples_rh = np.random.choice(r + 1, size=Ntrials, p=pi_rh)
            s_t_rh = get_s_t_rh(s_0_samples_rh, T_matrix_rh, r)
            jump_times_rh = get_jump_times_rh(s_t_rh, r)
            x_t_values_rh = np.where(s_t_rh == r, 1.0, x0)
            r_t_rh = get_r_t_rh(x_t_values_rh, Rh_values[j])
            n_t_rh = get_n(r_t_rh)
            lambdas_rh = get_lambdas_rh(r, x0, Rh_values[j])
            # Compute errors for multiple trials
            errors_smooth = []
            errors_filter = []
            for w in range(Ntrials):
                counts = n_t_rh[w]
                ll = poisson_logpdf(counts=counts, lambdas=lambdas_rh, mask=None)
                posterior_smooth, normaliser_smooth = hmm_expected_states(
                    pi0=pi_rh, Ps=T_matrix_rh, ll=ll, filter=False
                )
                posterior_filter, normaliser_filter = hmm_expected_states(
                    pi0=pi_rh, Ps=T_matrix_rh, ll=ll, filter=True
                )
                prob_upper_smooth[w] = posterior_smooth[:, r]
                prob_upper_filter[w] = posterior_filter[:, r]

                inferred_jump_time_smooth = np.where(prob_upper_smooth[w] >= 0.5)[0]
                inferred_jump_time_filter = np.where(prob_upper_filter[w] >= 0.5)[0]
                if len(inferred_jump_time_smooth) > 0 and ~np.isnan(jump_times_rh[w]):
                    error = (
                        abs(inferred_jump_time_smooth[0] - jump_times_rh[w]) * dt * 1000
                    )
                else:
                    error = T * dt * 1000  # = 1000ms, worst case
                errors_smooth.append(error)
                if len(inferred_jump_time_filter) > 0 and ~np.isnan(jump_times_rh[w]):
                    error = (
                        abs(inferred_jump_time_filter[0] - jump_times_rh[w]) * dt * 1000
                    )
                else:
                    error = T * dt * 1000  # = 1000ms, worst case
                errors_filter.append(error)
            error1[i, j] = np.mean(errors_smooth)
            error2[i, j] = np.mean(errors_filter)
    return error1, error2


# r and Rh
def heat_map_r_Rh(r_values, Rh_values):
    error1 = np.zeros((len(r_values), (len(Rh_values))))
    error2 = np.zeros((len(r_values), (len(Rh_values))))
    for i in range(len(r_values)):
        for j in range(len(Rh_values)):
            # Sample s_0 from initial distribution
            pi_rh = get_pi_rh(r_values[i])
            T_matrix_rh = get_T_matrix_rh(m, r_values[i])
            s_0_samples_rh = np.random.choice(r_values[i] + 1, size=Ntrials, p=pi_rh)
            s_t_rh = get_s_t_rh(s_0_samples_rh, T_matrix_rh, r_values[i])
            jump_times_rh = get_jump_times_rh(s_t_rh, r_values[i])
            x_t_values_rh = np.where(s_t_rh == r_values[i], 1.0, x0)
            r_t_rh = get_r_t_rh(x_t_values_rh, Rh_values[j])
            n_t_rh = get_n(r_t_rh)
            lambdas_rh = get_lambdas_rh(r_values[i], x0, Rh_values[j])
            # Compute errors for multiple trials
            errors_smooth = []
            errors_filter = []
            for w in range(Ntrials):
                counts = n_t_rh[w]
                ll = poisson_logpdf(counts=counts, lambdas=lambdas_rh, mask=None)
                posterior_smooth, normaliser_smooth = hmm_expected_states(
                    pi0=pi_rh, Ps=T_matrix_rh, ll=ll, filter=False
                )
                posterior_filter, normaliser_filter = hmm_expected_states(
                    pi0=pi_rh, Ps=T_matrix_rh, ll=ll, filter=True
                )
                prob_upper_smooth[w] = posterior_smooth[:, r_values[i]]
                prob_upper_filter[w] = posterior_filter[:, r_values[i]]

                inferred_jump_time_smooth = np.where(prob_upper_smooth[w] >= 0.5)[0]
                inferred_jump_time_filter = np.where(prob_upper_filter[w] >= 0.5)[0]
                if len(inferred_jump_time_smooth) > 0 and ~np.isnan(jump_times_rh[w]):
                    error = (
                        abs(inferred_jump_time_smooth[0] - jump_times_rh[w]) * dt * 1000
                    )
                else:
                    error = T * dt * 1000  # = 1000ms, worst case
                errors_smooth.append(error)
                if len(inferred_jump_time_filter) > 0 and ~np.isnan(jump_times_rh[w]):
                    error = (
                        abs(inferred_jump_time_filter[0] - jump_times_rh[w]) * dt * 1000
                    )
                else:
                    error = T * dt * 1000  # = 1000ms, worst case
                errors_filter.append(error)
            error1[i, j] = np.mean(errors_smooth)
            error2[i, j] = np.mean(errors_filter)
    return error1, error2


# Arrays of values
m_values = [10, 25, 30, 50, 70, 100, 115]
r_values = [1, 3, 5, 7, 9, 10, 12, 15]
x0_values = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
Rh_values = [1, 5, 10, 20, 50, 75, 100, 250, 500, 1000]


# Find errors
error1_m_r, error2_m_r = heat_map_m_r(m_values, r_values)
error1_m_x0, error2_m_x0 = heat_map_m_x0(m_values, x0_values)
error1_x0_r, error2_x0_r = heat_map_x0_r(x0_values, r_values)
error1_x0_Rh, error2_x0_Rh = heat_map_x0_Rh(x0_values, Rh_values)
error1_m_Rh, error2_m_Rh = heat_map_m_Rh(m_values, Rh_values)
error1_r_Rh, error2_r_Rh = heat_map_r_Rh(r_values, Rh_values)


# Plot Heatmaps
plt.figure(figsize=(10, 5))
sns.heatmap(
    error1_m_r,
    annot=True,
    fmt=".5f",
    cmap="viridis",
    xticklabels=[f"{r:.2f}" for r in r_values],
    yticklabels=[f"{m:.2f}" for m in m_values],
)
plt.xlabel("r")
plt.ylabel("m")
plt.title("Error Smoother: m vs r")
plt.show()

plt.figure(figsize=(10, 5))
sns.heatmap(
    error2_m_r,
    annot=True,
    fmt=".5f",
    cmap="viridis",
    xticklabels=[f"{r:.2f}" for r in r_values],
    yticklabels=[f"{m:.2f}" for m in m_values],
)
plt.xlabel("r")
plt.ylabel("m")
plt.title("Error Filter: m vs r")
plt.show()

plt.figure(figsize=(10, 5))
sns.heatmap(
    error1_m_x0,
    annot=True,
    fmt=".5f",
    cmap="viridis",
    xticklabels=[f"{x0:.2f}" for x0 in x0_values],
    yticklabels=[f"{m:.2f}" for m in m_values],
)
plt.xlabel("x0")
plt.ylabel("m")
plt.title("Error Smoother: m vs x0")
plt.show()

plt.figure(figsize=(10, 5))
sns.heatmap(
    error2_m_x0,
    annot=True,
    fmt=".5f",
    cmap="viridis",
    xticklabels=[f"{x0:.2f}" for x0 in x0_values],
    yticklabels=[f"{m:.2f}" for m in m_values],
)
plt.xlabel("x0")
plt.ylabel("m")
plt.title("Error Filter: m vs x0")
plt.show()


plt.figure(figsize=(10, 5))
sns.heatmap(
    error1_x0_r,
    annot=True,
    fmt=".5f",
    cmap="viridis",
    xticklabels=[f"{r:.2f}" for r in r_values],
    yticklabels=[f"{x0:.2f}" for x0 in x0_values],
)
plt.xlabel("r")
plt.ylabel("x0")
plt.title("Error Smoother: x0 vs r")
plt.show()

plt.figure(figsize=(10, 5))
sns.heatmap(
    error2_x0_r,
    annot=True,
    fmt=".5f",
    cmap="viridis",
    xticklabels=[f"{r:.2f}" for r in r_values],
    yticklabels=[f"{x0:.2f}" for x0 in x0_values],
)
plt.xlabel("r")
plt.ylabel("x0")
plt.title("Error Filter: x0 vs r")
plt.show()


plt.figure(figsize=(10, 5))
sns.heatmap(
    error1_x0_Rh,
    annot=True,
    fmt=".3f",
    cmap="viridis",
    xticklabels=[f"{Rh:.2f}" for Rh in Rh_values],
    yticklabels=[f"{x0:.2f}" for x0 in x0_values],
)
plt.xlabel("Rh")
plt.ylabel("x0")
plt.title("Error Smoother: x0 vs Rh")
plt.show()


plt.figure(figsize=(10, 5))
sns.heatmap(
    error2_x0_Rh,
    annot=True,
    fmt=".3f",
    cmap="viridis",
    xticklabels=[f"{Rh:.2f}" for Rh in Rh_values],
    yticklabels=[f"{x0:.2f}" for x0 in x0_values],
)
plt.xlabel("Rh")
plt.ylabel("x0")
plt.title("Error Filter: x0 vs Rh")
plt.show()


plt.figure(figsize=(10, 5))
sns.heatmap(
    error1_m_Rh,
    annot=True,
    fmt=".3f",
    cmap="viridis",
    xticklabels=[f"{Rh:.2f}" for Rh in Rh_values],
    yticklabels=[f"{m:.2f}" for m in m_values],
)
plt.xlabel("Rh")
plt.ylabel("m")
plt.title("Error Smoother: m vs Rh")
plt.show()


plt.figure(figsize=(10, 5))
sns.heatmap(
    error2_m_Rh,
    annot=True,
    fmt=".3f",
    cmap="viridis",
    xticklabels=[f"{Rh:.2f}" for Rh in Rh_values],
    yticklabels=[f"{m:.2f}" for m in m_values],
)
plt.xlabel("Rh")
plt.ylabel("m")
plt.title("Error Filter: m vs Rh")
plt.show()


plt.figure(figsize=(10, 5))
sns.heatmap(
    error1_r_Rh,
    annot=True,
    fmt=".3f",
    cmap="viridis",
    xticklabels=[f"{Rh:.2f}" for Rh in Rh_values],
    yticklabels=[f"{r:.2f}" for r in r_values],
)
plt.xlabel("Rh")
plt.ylabel("r")
plt.title("Error Smoother: r vs Rh")
plt.show()


plt.figure(figsize=(10, 5))
sns.heatmap(
    error2_r_Rh,
    annot=True,
    fmt=".3f",
    cmap="viridis",
    xticklabels=[f"{Rh:.2f}" for Rh in Rh_values],
    yticklabels=[f"{r:.2f}" for r in r_values],
)
plt.xlabel("Rh")
plt.ylabel("r")
plt.title("Error Filter: r vs Rh")
plt.show()
