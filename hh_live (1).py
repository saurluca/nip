# %%
import numpy as np
import matplotlib.pyplot as plt

def euler_integrate(
    derivs,
    x0,
    t,
):

    x = np.empty((len(t), len(x0)))

    x[0] = x0

    for k in range(len(t) - 1):

        dt = t[k + 1] - t[k]
        x[k + 1] = x[k] + dt * derivs(t[k], x[k])

    return x

def RC_derivative(tau, I):
    """f(x, t)"""

    def deriv(t, x):
        dx = - 1 / tau * x + I(t)
        return np.array([dx])

    return deriv

def plot_trajectory(t: np.ndarray, V: np.ndarray, ylab="", xlab="Time (ms)", title: str = ""):
    plt.figure()
    plt.plot(t, V)
    plt.xlabel("Time (ms)")
    plt.ylabel(ylab)
    plt.title(title)
    plt.tight_layout()

dt = 0.1
T = 50
t = np.arange(0.0, T + dt, dt)
tau = 20

amp = 3.0
# step current!
I = lambda t: amp if 10.0 <= t < 20.0 else 0.0
# Initial gating at steady state for V0
y0 = np.array([1.0])
traj = euler_integrate(RC_derivative(tau, I), y0, t)

plot_trajectory(t, traj)

# %%
