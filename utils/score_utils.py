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

def earth_mover_distance(y_true, y_pred):
    cdf_ytrue = np.cumsum(y_true, axis=-1)
    cdf_ypred = np.cumsum(y_pred, axis=-1)
    samplewise_emd = np.sqrt(np.mean(np.square(np.abs(cdf_ytrue - cdf_ypred)), axis=-1))
    return np.mean(samplewise_emd)
