from torch import distributions


def _dynamically_binarize(x):
    return distributions.Bernoulli(probs=x).sample()