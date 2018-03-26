import numpy as np

# calculate mean score for AVA dataset
def mean_score(scores):
    si = np.arange(1, 11, 1)
    mean = np.sum(si * scores,axis=-1)
    return mean

# calculate standard deviation of scores for AVA dataset
def std_score(scores):
    if len(scores.shape) < 3:
        scores = np.expand_dims(scores, axis=0)
    si = np.ones((scores.shape[0],10)) * np.arange(1, 11, 1)
    mean = mean_score(scores)
    a = (si - mean.T) ** 2
    std = np.sqrt(np.sum((a) * scores,axis=-1))
    return std[0]
