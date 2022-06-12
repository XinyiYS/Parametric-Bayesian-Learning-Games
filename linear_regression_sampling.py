import numpy as np

def leverage_iid_sampling(X, sample_size, reg_lambda):
    Z = np.linalg.inv(X.T @ X + reg_lambda * np.identity(X.shape[1]))
    h_is = np.asarray([1 - x @Z @x.T for x in X])
    return np.random.choice(list(range(len(X))), sample_size, p=h_is / h_is.sum())
