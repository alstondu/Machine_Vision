import numpy as np
from scipy.special import gamma as gamma_function
import math

def normal_inv_gamma(alpha, beta, delta, gamma, mu, sigma):
    """Return the probability density function for the normal
    inverse gamma density at (mu, sigma)
    
    Args:
        alpha: shape of variance
        beta: scale of variance
        delta: mean of mu
        gamma: precision of mu
        mu: normal mean
        sigma: normal standard deviation
    Returns:
        a probability density function
    """
    # You will find scipy.special.gamma useful
    #scipy.special.gamma()
    prior = (math.sqrt(gamma/(2*math.pi))/sigma)*(beta**alpha/gamma_function(alpha))*(1/sigma**2)**(alpha+1)*math.exp((2*beta+gamma*(delta-mu)**2)/(-2*sigma**2))

    return prior
