import numpy as np
import matplotlib.pyplot as plt
import math


def normal_distribution(x, mu, sigma):
    return np.exp(-1 * ((x - mu) ** 2) / (2 * (sigma ** 2))) / (math.sqrt(2 * np.pi) * sigma)


mu, sigma = 15, 10


x = np.linspace(mu - 6 * sigma, mu + 6 * sigma, 100)
y = -500*normal_distribution(x, mu, sigma)+30
plt.plot(x, y, 'r', label='mu = 15,sigma = 10')
plt.legend()
plt.grid()
plt.show()
