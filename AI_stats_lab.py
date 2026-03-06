import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.integrate import quad


# =========================================================
# QUESTION 1 — CDF Probabilities
# =========================================================

def cdf_probabilities():
    analytic_gt5 = math.exp(-5)                
    analytic_lt5 = 1 - math.exp(-5)             
    analytic_interval = math.exp(-3) - math.exp(-7)


    rng = np.random.default_rng(seed=42)
    samples = rng.exponential(scale=1, size=100000)

    simulated_gt5 = np.mean(samples > 5)

    return analytic_gt5, analytic_lt5, analytic_interval, simulated_gt5


# =========================================================
# QUESTION 2 — PDF Validation and Visualization
# =========================================================

def pdf_validation_plot():

    def pdf_function(x):
        return 2 * x * math.exp(-(x ** 2))

    x_values = np.linspace(0, 3, 200)
    y_values = [pdf_function(x) for x in x_values]

    integral_value, _ = quad(lambda x: 2 * x * math.exp(-(x ** 2)), 0, np.inf)

    is_valid_pdf = abs(integral_value - 1) < 1e-3

    plt.figure()
    plt.plot(x_values, y_values)
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title("Candidate PDF: f(x) = 2x e^{-x^2}")
    plt.close()   

    return integral_value, is_valid_pdf


# =========================================================
# QUESTION 3 — Exponential Distribution
# =========================================================

def exponential_probabilities():
    
    analytic_gt5 = math.exp(-5)
    analytic_interval = math.exp(-1) - math.exp(-3)

    rng = np.random.default_rng(seed=42)
    samples = rng.exponential(scale=1, size=100000)

    simulated_gt5 = np.mean(samples > 5)
    simulated_interval = np.mean((samples > 1) & (samples < 3))

    return analytic_gt5, analytic_interval, simulated_gt5, simulated_interval


# =========================================================
# QUESTION 4 — Gaussian Distribution
# =========================================================

def gaussian_probabilities():
   
    mu = 10
    sigma = 2

    z_upper = (12 - mu) / sigma
    z_lower = (8 - mu) / sigma

    analytic_le12 = norm.cdf(z_upper)
    analytic_interval = norm.cdf(z_upper) - norm.cdf(z_lower)

    rng = np.random.default_rng(seed=42)
    samples = rng.normal(mu, sigma, 100000)

    simulated_le12 = np.mean(samples <= 12)
    simulated_interval = np.mean((samples > 8) & (samples < 12))

    return analytic_le12, analytic_interval, simulated_le12, simulated_interval
