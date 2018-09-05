
import numpy as np
from math import pi

def normal(x, mu, sig):
    sig_2 = sig**2
    Z = 1./np.sqrt(2*pi*sig_2)
    d = (x-mu)**2
    return Z*np.exp(-d/(2*sig_2))
    
def zscore(x):
    return (x-np.mean(x))/np.std(x)