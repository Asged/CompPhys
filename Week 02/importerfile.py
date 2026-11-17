import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

class HarmonicPotential():
        def __init__(self, k=1, kT=.15, x_min=-3, x_max=3):
            self.k = k
            self.kT = kT
            self.V = lambda x: 0.5 * self.k * x**2
            self.P = lambda x: np.exp(-self.V(x) / self.kT) / np.sqrt(2*np.pi*self.kT / self.k)
            self.x_min = x_min
            self.x_max = x_max

        def draw_sample(self, n=10_000):
              xs = np.linspace(self.x_min, self.x_max, n)
              envelope = self.P
              self.sample = [x for x in xs if np.random.rand() < envelope(x)]
              return self.sample

        def estimate_thermal_average(self):
             return np.sum(self.V(self.sample)) / len(sample)

        def plot(self, ax, xwidth=0.1,use_sample=False):
            average_V = quad(lambda x: self.V(x)*self.P(x),self.x_min,self.x_max)[0]
            title = f'Vexact={average_V:.5}'
            xs = np.linspace(self.x_min, self.x_max, 100)
            ax.plot(xs, self.V(xs))

            if use_sample:
                xs = np.arange(self.x_min,self.x_max,xwidth)
                bars, xs = np.histogram(self.sample,xs)
                xvals = (xs[:-1] + xs[1:]) / 2
                delta_xvals = (xvals[1] - xvals[0])
                P = bars / np.sum(bars) / delta_xvals
                width = delta_xvals * 0.8
                ax.bar(xvals, P, width=width)

                average_V_sampl = self.estimate_thermal_average()
                title += f' Vsample={average_V_sampl:.3f}'
            else:
                xs = np.linspace(self.x_min, self.x_max, 1000)
                ys = self.P(xs)
                ax.plot(xs,ys)
                        
            ax.set_title(title)
            ax.plot([self.x_min, self.x_max],np.ones(2)*average_V)
