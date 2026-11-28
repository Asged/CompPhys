import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

class HarmonicOscillator():
    def __init__(self, xmin, xmax, k=1, kT=0.15):
        self.xmin = xmin
        self.xmax = xmax
        self.k = k
        self.kT = kT
        self.Q = np.sqrt(2*np.pi*self.kT / self.k)
        self.V = lambda x: 0.5*self.k*x**2
        self.P = lambda x: 1 / self.Q * np.exp(-0.5*self.k*x**2 / kT)
        self.sample = None

    def thermal_average(self):
        return quad(lambda x: self.V(x) * self.P(x), self.xmin, self.xmax)[0]

    def metropolis_sample(self, n_steps, delta=0.5, x0=0.0):
        x = x0                  # Walker position, starting at x0
        xs = np.empty(n_steps)  # Array to store sample
        for i in range(n_steps):
            x_proposed = x + np.random.uniform(-delta, delta) # Propose a step
            dV = self.V(x_proposed) - self.V(x)
            if dV <= 0 or np.random.rand() < np.exp(-dV / self.kT): # Accept if V potential decreases or uphill with a probability. The larger dV is the lower the probability
                x = x_proposed
            xs[i] = x

        self.sample = xs
        return xs

    def sample_average(self, xs):
        return np.mean(self.V(xs))

    def thermal_average_V2(self):
        return quad(lambda x: self.V(x)**2*self.P(x), self.xmin, self.xmax)[0]

    def heat_capacity_fluctuation(self):
        V1 = self.thermal_average()
        V2 = self.thermal_average_V2()
        return (V2 - V1**2) / self.kT**2

    def heat_capacity_fluctuation_sample(self, xs):
        V = self.V(xs)
        return (np.mean(V**2) - np.mean(V)**2) / self.kT**2

    def plot(self, ax):
        xs = np.linspace(self.xmin, self.xmax, 500)
        if ax is None:
            fig, ax = plt.subplots()

        if self.sample is not None:
            ax.hist(self.sample, bins=60, range=(self.xmin, self.xmax), density=True)

        ax.plot(xs, self.V(xs), label=f"V(x)")
        ax.plot(xs, self.P(xs), label=f"P(x)")
        ax.axhline(self.kT, color="gray", linestyle="--", label=f"$k_BT$ = {self.kT}")
        ax.set_xlabel("x")
        ax.set_ylabel("V(x)")
        ax.set_title("Harmonic potential k = {self.k}")
        ax.legend()
        return ax