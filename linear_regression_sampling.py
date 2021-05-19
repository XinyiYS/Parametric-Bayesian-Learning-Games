import numpy as np


# buggy, to be fixed
def reg_vol_sampling(X, sample_size, reg_lambda):

    Z = np.linalg.inv(X.T@ X + reg_lambda * np.identity(X.shape[1]))
    h_is = np.asarray([1 - x @Z @x.T for x in X])
    S = set(np.arange(len(X)))
    while True:
        i = np.random.choice(list(S), 1, p=h_is)
        S.remove(i)
        v = Z @ X[i] / np.sqrt(h_is[i])


        h_is[j] = h_is[j] - np.pow(X[j].T @ v, 2)

        for j in S: pass
        Z = Z + v @ v.T

        if len(S) <=  sample_size:
            return list(S)


from scipy.stats import bernoulli
def fast_reg_vol_sampling(X, sample_size, reg_lambda):
    '''
    Implements the FastRegVol sampling method introduced in "Reverse iterative volume sampling for linear regression"
    
    Efficient and provides an unbiased estimator for the pseudo-inverse, and multiplicative factor of expected loss.
    '''
    if sample_size < 2 * X.shape[1]:
        return reg_vol_sampling(X, sample_size, reg_lambda)

    Z = np.linalg.inv(X.T@ X + reg_lambda * np.identity(X.shape[1]))
    S = set(np.arange(len(X)))

    while True:
        A = 0
        while A != 1:
            i = np.random.choice(list(S), 1)
            h_i = 1 - (X[i] @ Z @ X[i].T).squeeze()
            A = bernoulli.rvs(size=1, p=h_i)
        
        S.remove(int(i))
        Z = Z + (1.0 / h_i) * (Z @ X[i].T @ X[i] @ Z)

        if len(S) <= max(2 * X.shape[1], sample_size):
            return list(S)

def leverage_iid_sampling(X, sample_size, reg_lambda):
    Z = np.linalg.inv(X.T @ X + reg_lambda * np.identity(X.shape[1]))
    h_is = np.asarray([1 - x @Z @x.T for x in X])
    return np.random.choice(list(range(len(X))), sample_size, p=h_is / h_is.sum())
