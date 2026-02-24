import torch
import numpy as np
import pickle
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split

def random_split(data, test_size=0.8, random_state=None):
    if torch.is_tensor(data):
        data = data.numpy()
    train_data, test_data = train_test_split(data, test_size=test_size, random_state=random_state)
    return train_data, test_data

def cvfit(data, n_components, covariance_type='full', 
          n_init=2, test_size=0.80, random_state=None,
          verbose=2, max_iter=250, reg_covar=1e-4):
    train_data, test_data = random_split(data, test_size=test_size, random_state=random_state)
    GMM = GaussianMixture(n_components=n_components, covariance_type=covariance_type, 
                          n_init=n_init, random_state=random_state, verbose=verbose,
                          max_iter=max_iter, reg_covar=reg_covar)
    GMM.fit(train_data)
    log_likelihoods = GMM.score_samples(test_data)
    
    return GMM, log_likelihoods

def fit(data, n_components, covariance_type='full', 
        n_init=2, random_state=None, verbose=2, max_iter=250, reg_covar=1e-4):
    GMM = GaussianMixture(n_components=n_components, covariance_type=covariance_type, 
                          n_init=n_init, random_state=random_state, verbose=verbose,
                          max_iter=max_iter, reg_covar=reg_covar)
    GMM.fit(data)
    log_likelihoods = GMM.score_samples(data)
    
    return GMM, log_likelihoods

def save(filename, GMM):
    with open(filename,'wb') as f:
        pickle.dump(GMM,f)

def load(filename):
    with open(filename, 'rb') as f:
        GMM = pickle.load(f)
    return GMM
