import numpy as np
import math
def normal(X, mu, sigma):
    """Return likelihood of data given parameters"

    Computes the likelihood that the data X have been generated
    from the given parameters (mu, sigma) of the one-dimensional
    normal distribution.

    Args:
        X: vector of point samples
        mu: mean
        sigma: standard deviation
    Returns:
        a scalar likelihood
    """

    lik = (1 / ((2 * math.pi*sigma**2)**(len(X)/2))) * math.exp(-sum((x - mu)**2 for x in X) / (2 * sigma**2))
    return lik