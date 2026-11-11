import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


# Input functions
def input_1(t):
    return [1.0, 0.0]


def input_2(t):
    return [0.0, 0.5]


def input_3(t):
    return [1.0, 0.5]


def input_4(t):
    return [1 + 0.2 * np.sin(t), 0.2 + 0.6 * np.cos(t)]


# Robot dynamics
def system_dynamics(state, t, input_fn):
    x, y, theta = state

    v, omega = input_fn(t)

    dxdt = v * np.cos(theta)
    dydt = v * np.sin(theta)
    dthetadt = omega

    return [dxdt, dydt, dthetadt]


# Initial conditions
initial_state = [0.0, 0.0, 0.0]

# Simulation time
t = np.linspace(0, 30, 301)

names = [
    "Case i.1: v=1, w=0",
    "Case i.2: v=0, w=0.5",
    "Case i.3: v=1, w=0.5",
    "Case ii: v=1+0.2sin(t), w=0.2+0.6cos(t)"
]

input_fns = [input_1, input_2, input_3, input_4]


for i in range(4):

    solution = odeint(
        system_dynamics,
        initial_state,
        t,
        args=(input_fns[i],)
    )

    x = solution[:, 0]
    y = solution[:, 1]
    theta = solution[:, 2]

    plt.figure(figsize=(10, 8))
    plt.suptitle(names[i])

    # Top view
    plt.subplot(2, 2, 1)
    plt.plot(x, y, linewidth=2)
    plt.plot(x[0], y[0], 'go', label='start')
    plt.plot(x[-1], y[-1], 'rx', label='end')
    plt.title("Top View")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.axis("equal")
    plt.legend()
    plt.grid(True)

    # x(t)
    plt.subplot(2, 2, 2)
    plt.plot(t, x, linewidth=2)
    plt.title("x(t)")
    plt.xlabel("Time (s)")
    plt.ylabel("x (m)")
    plt.grid(True)

    # y(t)
    plt.subplot(2, 2, 3)
    plt.plot(t, y, linewidth=2)
    plt.title("y(t)")
    plt.xlabel("Time (s)")
    plt.ylabel("y (m)")
    plt.grid(True)

    # theta(t)
    plt.subplot(2, 2, 4)
    plt.plot(t, theta, linewidth=2)
    plt.title("theta(t)")
    plt.xlabel("Time (s)")
    plt.ylabel("theta (rad)")
    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        f"q2_case_{i + 1}.png",
        dpi=150
    )

plt.show()