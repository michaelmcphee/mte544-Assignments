import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Robot parameters
r = 0.10
l = 0.25


# G(theta) from Q3(a)
def G_matrix(theta):
    return (1 / r) * np.array([
        [np.cos(theta + np.pi/2), np.sin(theta + np.pi/2), l],
        [np.cos(theta + 7*np.pi/6), np.sin(theta + 7*np.pi/6), l],
        [np.cos(theta + 11*np.pi/6), np.sin(theta + 11*np.pi/6), l]
    ])


# Wheel speed inputs
def input_1():
    return np.array([-2.0, 1.0, 1.0])


def input_2():
    return np.array([1.0, -1.0, 0.0])


def input_3():
    return np.array([
        2.5,
        2.5 - 5*np.sqrt(3),
        2.5 + 5*np.sqrt(3)
    ])


# Robot dynamics
def system_dynamics(state, t, input_fn):
    x, y, theta = state

    u = input_fn()
    G = G_matrix(theta)

    # u = G(theta) q_dot
    q_dot = np.linalg.solve(G, u)

    return q_dot


# Initial conditions
initial_state = [0.0, 0.0, 0.0]

# Simulation time
t = np.linspace(0, 30, 301)

names = [
    "Case I: u1=-2, u2=1, u3=1",
    "Case II.1: Straight line, 60 degree slope",
    "Case II.2: 2 m diameter circle"
]

input_fns = [input_1, input_2, input_3]


for i in range(3):

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

    if i == 0:
        plt.ylim(-1, 1)

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
    plt.title("θ(t)")
    plt.xlabel("Time (s)")
    plt.ylabel("θ (rad)")
    plt.grid(True)

    if i == 1:
        plt.ylim(-1, 1)

    plt.tight_layout()

    plt.savefig(
        f"q3_case_{i + 1}.png",
        dpi=300,
        bbox_inches='tight'
    )

    plt.show()