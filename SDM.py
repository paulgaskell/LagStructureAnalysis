
"""
SDM is forecasting method that provides the Bayesian optimal forecast of 
the distance between x and y based on asumptions about the markov process 
that drives lag structure change between t and t+1. 

See Gaskell, McGroarty, and Tiropanis 2015
"""

import numpy as np
from math import pi
from pylab import show, figure
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("imports have completed")

def normal(x, mu, sig):
    sig_2 = sig**2
    Z = 1./np.sqrt(2*pi*sig_2)
    d = (x-mu)**2
    return Z*np.exp(-d/(2*sig_2))
	

class SDM:
    """
    class holding the SDM method
    
    example usage below
    """
    
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
        logger.info("t-series dimensions = {}, {}".format(
                        str(x.shape), str(y.shape)
                        ))
    
    def create_d_surface(self, lags):
        """
        """
        D = np.zeros((len(self.x)-lags, 2, lags))
        logger.info("D dim = {}".format(D.shape))
        for t in range(lags, len(self.x), 1):
            D[t-lags][0] = x[t-lags:t][::-1]-y[t]
            D[t-lags][1] = y[t-lags:t][::-1]-x[t]

        self.lags = lags
        self.D = D 
        return self    
    
    def sdm(self, top_paths=False):
        """
        """
        mu = np.mean(self.D)
        sig = np.std(self.D)
        logger.info("mu={} sig={}".format(mu, sig))
		
        W = np.ones(self.D.shape)/(self.lags*2)
        logger.info("W dim = {}".format(W.shape))
        for t in range(1, len(self.D), 1): 
            if top_paths:
                w = W[t-1]
                for i in range(len(self.D[t-1])):
                    w[i] = W[t-1][i]
                    w[i] = w[i]+np.append(0, W[t-1][i][:-1])
                    w[i] = w[i]+np.append(W[t-1][i][1:], 0)
                w = w/3
            else:
                w = W[t-1]
            
            W[t] = (w+0.00001)*normal(self.D[t], mu, sig)
            W[t] = W[t]/np.sum(W[t])
            print W[t].shape, np.sum(W[t])
            
        self.W = W
        return self 

        
if __name__ == "__main__":

    # EXAMPLE USAGE

    # make an example time series
    x = np.random.normal(0, 1, 200)
    y = np.random.normal(0, .1, 200)
    for n, i in enumerate(x):
        if n < 20:
            y[n] += x[n+1]
        elif n < 70:
            y[n] += x[n-10]
        elif n < 120:
            y[n] += x[n-30]
        elif n < 150:
            y[n] += x[n-35]
        else:
            y[n] += x[n]
    
    y = (y-np.mean(y))/np.std(y)
    
    x = np.append(x, x)
    y = np.append(y, y)
    
    # run the TOP method
    sdm = SDM(x, y)
    sdm = sdm.create_d_surface(40).sdm(top_paths=True)    
    fig = figure()
    ax = fig.add_subplot(211)
    ax1 = fig.add_subplot(212)
    ax.imshow([i[0] for i in sdm.W.T], aspect='auto')
    ax1.imshow([i[1] for i in sdm.W.T], aspect='auto')
    show()
	