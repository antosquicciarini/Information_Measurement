import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from scipy.stats import gaussian_kde
from scipy.stats import norm
from scipy.integrate import quad, simps

# Set random seed for reproducibility
np.random.seed(42)

def expected_value(f, pdf):
    discr = np.linspace(-10, 10, 1000000)
    return simps(f(discr) * pdf(discr), discr)

def differentiable_variance(f, pdf):
    discr = np.linspace(-10, 10, 1000000)
    return simps((f(discr)- expected_value(f, pdf))**2, discr)


# Define a mixture of Gaussians distribution
def mixture_of_gaussians(n_samples):
    mean1, std_dev1 = -2, 0.5
    mean2, std_dev2 = 2, 0.7
    
    weights = [0.6, 0.4]
    
    samples1 = np.random.normal(mean1, std_dev1, int(n_samples * weights[0]))
    samples2 = np.random.normal(mean2, std_dev2, int(n_samples * weights[1]))
    
    return np.concatenate([samples1, samples2])

# Number of samples
n = 100

# Generate samples from the mixture of Gaussians distribution
data = mixture_of_gaussians(n)

# Function to estimate KDE and compute bias and variance
def compute_bias_variance(data, bandwidths):
    true_density = lambda x: 0.6 * norm.pdf(x, loc=-2, scale=0.5) + \
                             0.4 * norm.pdf(x, loc=2, scale=0.7)
    
    biases = []
    variances = []
    
    plt.figure(figsize=(15, 8))
    # Plot the true density
    x_dens = np.linspace(-5, 5, 1000)
    plt.plot(x_dens, true_density(x_dens), label='True PDF', color='blue', linewidth=3 )
    plt.scatter(data, np.zeros_like(data), color='red', alpha=0.1, label=f'n:{n} sampled poins from True PDF',)
    
    for h in bandwidths:
        kde = gaussian_kde(data, bw_method=h) #TOO fast, I was not able to create my solution
        expected_value(kde, true_density)
        bias = np.mean((kde(x_dens) - true_density(x_dens))**2)
        variance = differentiable_variance(kde, true_density)
        
        biases.append(bias)
        variances.append(variance)
        
        # Plot the estimated density
        plt.plot(x_dens, kde(x_dens), label=f'Estimated PDF (h={h:.2f})', alpha=0.6)

    plt.legend()
    #plt.tight_layout()
    plt.grid()
    plt.show()
    
    return biases, variances

# Bandwidth values to test
bandwidths = np.array([0.01, 0.05, 0.1, 0.5])

# Compute bias and variance for different bandwidths
biases, variances = compute_bias_variance(data, bandwidths)

# Plot bias and variance as a function of bandwidth
plt.figure(figsize=(10, 6))
plt.plot(bandwidths, biases, label='Bias^2')
plt.plot(bandwidths, variances, label='Variance')
plt.xlabel('Bandwidth')
plt.title('Bias^2 and Variance of KDE as a function of Bandwidth')
plt.grid()
plt.legend()
plt.show()
