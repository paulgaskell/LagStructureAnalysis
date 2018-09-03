
"""
SDM is forecasting method that provides the Bayesian optimal forecast of 
the distance between x and y based on asumptions about the markov process 
that drives lag structure change between t and t+1. 

See Gaskell, McGroarty, and Tiropanis 2015
"""

import logging
import numpy as np

from LSABase import LSABase
from LSAUtils import normal

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("imports have completed")

class SDM(LSABase):
    """
    class holding the SDM method
    """   
    
    def create_d_surface(self, series, lags):
        """
        """
        self.series = series 
        for x, y in self.series:
            logger.info("t-series dimensions = {}, {}".format(
                        str(x.shape), str(y.shape)
                        ))

        D = np.zeros((len(self.series[0][0])-lags, len(self.series), lags))
        logger.info("D dim = {}".format(D.shape))
        for t in range(lags, len(self.series[0][0]), 1):
            for i, (a, b) in enumerate(self.series):
                D[t-lags][i] = a[t-lags:t][::-1]-b[t]
        
        self.lags = lags
        self.D = D 
        return self    
    
    def sdm(self, top_paths=False):
        """
        """
        mu = np.mean(self.D)
        sig = np.std(self.D)
        logger.info("mu={} sig={}".format(mu, sig))
        
        W = np.ones(self.D.shape)/(self.lags*len(self.series))
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
        
        print abs(np.sum(W[1:]*self.D[:-1]))
        self.W = W
        return self 

        
if __name__ == "__main__":

    # EXAMPLE USAGE

    sdm = SDM()
    x, y = sdm.simple_example()
    sdm = sdm.create_d_surface([(x, y)], 40).sdm(top_paths=False)    
    sdm.multi_heatmap().display()