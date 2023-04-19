import numpy as np
from scipy.stats import norm, laplace, poisson
from matplotlib import pyplot as plt
from math import sqrt
from statistics import mean, median

#лапласса
def norm_distrib():
    #  выбираем 10 нормально распределенных точек
    norm10 = norm.rvs(size=10)
    norm50 = norm.rvs(size=50)
    norm1000 = norm.rvs(size=1000)

    plt.subplot()

    #  устанавливаем точки по оси икс
    x = np.linspace(norm.ppf(0.0005), norm.ppf(0.9995), 100)

    #  плотность распределения:
    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(x, norm.pdf(x), 'r-', lw=2, alpha=0.6, label='norm pdf')

    #  строим гистограмму
    ax1.hist(norm10, density=True, bins='auto', edgecolor='black', alpha=0.5)

    ax2 = plt.subplot(3, 1, 2)
    ax2.plot(x, norm.pdf(x), 'r-', lw=2, alpha=0.6, label='norm pdf')
    ax2.hist(norm50, density=True, bins='auto', edgecolor='black', alpha=0.5)

    ax3 = plt.subplot(3, 1, 3)
    ax3.plot(x, norm.pdf(x), 'r-', lw=2, alpha=0.6, label='norm pdf')
    ax3.hist(norm1000, density=True, bins='auto', edgecolor='black', alpha=0.5)
    plt.show()

def laplas():

    laplace10 = laplace.rvs(scale=1/sqrt(2),size=10)
    laplace50 = laplace.rvs(scale=1/sqrt(2),size=50)
    laplace1000 = laplace.rvs(scale=1/sqrt(2),size=1000)
    plt.subplot()

    x = np.linspace(laplace.ppf(0.0005), laplace.ppf(0.9995), 100)
    ax1 = plt.subplot(3,1,1)
    ax1.plot(x, laplace.pdf(x), 'r-', lw=2, alpha=0.6, label='norm pdf')
    ax1.hist(laplace10, density=True, bins='auto', edgecolor='black', alpha=0.5)
    ax1.title.set_text('laplace')
    ax2 = plt.subplot(3, 1, 2)
    ax2.plot(x, laplace.pdf(x), 'r-', lw=2, alpha=0.6, label='norm pdf')
    ax2.hist(laplace50, density=True, bins='auto', edgecolor='black', alpha=0.5)

    ax3 = plt.subplot(3, 1, 3)
    ax3.plot(x, laplace.pdf(x), 'r-', lw=2, alpha=0.6, label='norm pdf')
    ax3.hist(laplace1000, density=True, bins='auto', edgecolor='black', alpha=0.5)

    plt.show()

def poisson_distrib(mu=10):
    plt.subplot()

    x = np.arange(poisson.ppf(0.001, mu),
                  poisson.ppf(0.999, mu))

    ax1 = plt.subplot(3, 1, 1)
    x_hist = poisson.rvs(mu, size=10)
    ax1.hist(x_hist, density=True, edgecolor='black')
    ax1.plot(x, poisson.pmf(x, mu), 'r-', ms=8, label='poisson pmf')

    ax2 = plt.subplot(3, 1, 2)
    x_hist = poisson.rvs(mu, size=100)
    ax2.hist(x_hist, density=True, edgecolor='black')
    ax2.plot(x, poisson.pmf(x, mu), 'r-', ms=8, label='poisson pmf')


    ax3 = plt.subplot(3, 1, 3)
    x_hist = poisson.rvs(mu, size=1000)
    ax3.hist(x_hist, density=True, edgecolor='black')
    ax3.plot(x, poisson.pmf(x, mu), 'r-', ms=8, label='poisson pmf')


    plt.show()


