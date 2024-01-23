import numpy as np
import math
def log_normal(X, mu, sigma):
    """Return log-likelihood of data given parameters"

    Computes the log-likelihood that the data X have been generated
    from the given parameters (mu, sigma) of the one-dimensional
    normal distribution.

    Args:
        X: vector of point samples
        mu: mean
        sigma: standard deviation
    Returns:
        a scalar log-likelihood
    """
    #loglik = (1 / ((2 * math.pi*sigma**2)**(len(X)/2))) * math.exp(-sum((x - mu)**2 for x in X) / (2 * sigma**2))
    loglik = -math.log(2*math.pi)*len(X)/2 - math.log(sigma**2)*len(X)/2 - sum((x - mu)**2 for x in X) / (2 * sigma**2)
    return loglik
 