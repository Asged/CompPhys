import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

#region Spring Class from 04A
class Spring():
    def __init__ (self, k=1, x0 =0):
        self.k = k
        self.x0 = x0

    # to be used in the MC
    def potential_energy(self, x):
        return 0.5*self.k*(x-self.x0)**2
    
    # to be used in the MD
    def metropolis_mc(self, kbT, n_steps=200000, delta=0.5, x_init=0.0):
        x = x_init
        energies = np.empty(n_steps)
        positions = np.empty(n_steps)
        for i in range(n_steps):
            x_trial = x + np.random.uniform(-delta, delta)
            dE = self.potential_energy(x_trial) - self.potential_energy(x)
            if dE <= 0 or np.random.rand() < np.exp(-dE / kbT):
                x = x_trial
            
            positions[i] = x
            energies[i] = self.potential_energy(x)
            
        return np.mean(energies), positions

    def force(self, x):
        return -self.k * (x - self.x0)

    def plot_temperature_series(self, temperatures, n_steps=200000, delta=0.5):
        n_plots = len(temperatures)
        fig, axes = plt.subplots(1, n_plots, figsize=(4 * n_plots, 4), sharey=True)
        
        if n_plots == 1:
            axes = [axes]

        x_curve = np.linspace(-3, 3, 200)
        v_curve = self.potential_energy(x_curve)

        for ax, kbT in zip(axes, temperatures):
            v_exact = 0.5 * kbT
            v_mc, positions = self.metropolis_mc(kbT=kbT, n_steps=n_steps, delta=delta)

            # Potential curve and sampled positions histogram
            ax.plot(x_curve, v_curve, color='tab:blue', linewidth=1.5)
            ax.hist(positions, bins=60, density=True, color='tab:blue', alpha=0.7, rwidth=0.8)

            # Formatting to match assignment style
            ax.set_xlim(-3.5, 3.5)
            ax.set_ylim(0, 1.5)
            ax.set_xlabel('x')
            ax.set_ylabel('V')
            ax.set_title(f"Ve={v_exact:.3f} Vmet-MC={v_mc:.3f}", fontsize=11)
            ax.text(0, 1.0, f"kT={kbT}", ha='center', fontsize=9)

        plt.tight_layout()
        plt.show()

#region Double Well from 04B

class DoubleWell():
    def __init__(self, k=1.0, x0=0.0, A=-1.0, x1=1.5, l=0.25, xMin=-5.0, xMax=5.0):
            self.k = k
            self.x0 = x0
            self.A = A
            self.x1 = x1
            self.l = l
            self.xMin = xMin
            self.xMax = xMax

    def potential_energy (self , x):
        return .5*self.k*(x-self.x0)**2 + self.A*np.exp(-(x-self.x1)**2 / self.l**2)
    
    def force(self , x):
        return -self.k * (x - self.x0) + (2 * self.A * (x - self.x1) / (self.l**2)) * np.exp(-((x - self.x1) ** 2) / (self.l**2))

    def integrate(self, xs):
        return np.array([self.potential_energy(xs[0]) - quad(lambda xs: self.force(xs), xs[0], x)[0] for x in xs])

    def boltzmann_weight(self, x, kT):
        return np.exp(-self.potential_energy(x) / kT)

    def direct_integration(self, kT):
        num = quad(lambda x: self.potential_energy(x) * self.boltzmann_weight(x, kT), self.xMin, self.xMax)[0]
        den = quad(lambda x: self.boltzmann_weight(x, kT), self.xMin, self.xMax)[0]
        return num / den

    

    def metropolis_mc(self, kT, n_samples=200000, step_size=0.4):
        x = 0
        E = self.potential_energy(x)
        samples = np.empty(n_samples)
        
        for i in range(n_samples):
            x_prop = x + np.random.rand()
            E_prop = self.potential_energy(x_prop)
            dE = E_prop - E
            
            if dE < 0 or np.random.rand() < np.exp(-dE / kT):
                x, E = x_prop, E_prop
                
            samples[i] = x
        return samples


    def metropolis_mc(self, kT, n_samples=200000, step_size=0.4):
        x = 0.0
        E = self.potential_energy(x)
        samples = np.empty(n_samples)

        for i in range(n_samples):
            x_prop = x + np.random.uniform(-step_size, step_size)
            E_prop = self.potential_energy(x_prop)
            dE = E_prop - E
            if dE < 0 or np.random.rand() < np.exp(-dE / kT):
                x, E = x_prop, E_prop
            samples[i] = x

        avg_E_mc = np.mean(self.potential_energy(samples))
        return samples, avg_E_mc

import numpy as np
import matplotlib.pyplot as plt

class MolecularDynamicsSampler: 
    def __init__(self, system, mass=1.0, dt=0.01, n_steps=50):
        self.system = system      # Instance of DoubleWell (or any system providing a .force() method)
        self.mass = mass          # Particle mass
        self.dt = dt              # Time step size (Δt = 0.01)[cite: 1]
        self.n_steps = n_steps    # Number of steps per thermalization run (50)[cite: 1]

    def sample(self, kT, n_runs=10000):
        samples = np.zeros(n_runs)
        x = getattr(self.system, 'x0', 0.0) # Start oscillator x=x0
        for i in range(n_runs):
            v = np.random.normal(0.0, np.sqrt(kT / self.mass))
            
            # Initial force at current position
            f = self.system.force(x)
            
            # Execute 50 Velocity Verlet steps[cite: 1]
            for _ in range(self.n_steps):
                # 1. Update position: x(t + Δt) = x(t) + v(t)*Δt + 0.5*a(t)*Δt²
                x_new = x + v * self.dt + 0.5 * (f / self.mass) * (self.dt ** 2)
                
                # 2. Compute new force: f(t + Δt)
                f_new = self.system.force(x_new)
                
                # 3. Update velocity: v(t + Δt) = v(t) + 0.5*(a(t) + a(t + Δt))*Δt
                v_new = v + 0.5 * ((f + f_new) / self.mass) * self.dt
                
                # Advance states
                x = x_new
                f = f_new
                v = v_new
            
            # Store the end-point of this 50-step run[cite: 1]
            samples[i] = x
            
        return samples

