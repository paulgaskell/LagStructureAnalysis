
"""
SDM is forecasting method that provides the Bayesian optimal forecast of 
the distance between x and y based on asumptions about the markov process 
that drives lag structure change between t and t+1. 

See Gaskell, McGroarty, and Tiropanis 2015

TODO:
    graphing 
    TOP needs to change to strips 
    aligning series 
    
    - paths act as regularisation
    - linear translation of normal is normal 
    
"""

import logging
import numpy as np

from LSABase import LSABase
from LSAUtils import normal, zscore

logging.basicConfig(level=logging.WARN)
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
        lag_series = np.zeros(D.shape)
        lead_series = np.zeros((len(self.series[0][0])-lags, len(self.series)))
        logger.info("D dim = {}".format(D.shape))
        for t in range(lags, len(self.series[0][0]), 1):
            for i, (a, b) in enumerate(self.series):
                lag_series[t-lags][i] = a[t-lags:t][::-1]
                lead_series[t-lags][i] = b[t]
                D[t-lags][i] = a[t-lags:t][::-1]-b[t]
        
        self.lag_series = lag_series
        self.lead_series = lead_series 
        self.lags = lags
        self.D = D 
        return self    
    
    def sdm(self, top_paths=False):
        """
        """

        PrD = normal(zscore(self.D), 0, 1)
        W = np.ones(self.D.shape)/(self.lags*len(self.series))
        logger.info("W dim = {}".format(W.shape))
        for t in range(1, len(self.D), 1): 
            w = W[t-1]
            if top_paths:
                for i in range(len(self.D[t-1])):
                    w[i] = W[t-1][i]
                    w[i] = w[i]+np.append(0, W[t-1][i][:-1])
                    w[i] = w[i]+np.append(W[t-1][i][1:], 0)
                w[1:-1] = w[1:-1]/3
                w[0] = w[0]/2
                w[-1] = w[-1]/2
            
            W[t] = (w+0.00001)*PrD[t]
            W[t] = W[t]/np.sum(W[t])
        
        self.error = abs(np.sum(W[:-1]*self.D[1:]))
        self.forecast = np.sum(W[:-1]*self.lag_series[1:], 2)
        logger.info("Error={}".format(self.error))
        self.W = W
        return self 

        
if __name__ == "__main__":

    # EXAMPLE USAGE

    sdm = SDM()
    x, y = sdm.simple_example()
    sdm = sdm.create_d_surface([(x, y), (y, x)], 40).sdm(top_paths=False)    
    
    print sdm.forecast.shape
    print sdm.lead_series[1:].shape
    from pylab import figure, show
    fig = figure()
    ax = fig.add_subplot(121)
    ax.plot(sdm.lead_series[1:])
    ax.plot(sdm.forecast)
    ax1 = fig.add_subplot(122)
    ax1.scatter(sdm.forecast, sdm.lead_series[1:])
    
    sdm.multi_heatmap()
    show()
    

    
    