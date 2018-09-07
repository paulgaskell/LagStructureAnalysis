
"""
SDM is forecasting method that provides the Bayesian optimal forecast of 
the distance between x and y based on asumptions about the markov process 
that drives lag structure change between t and t+1. 

See Gaskell, McGroarty, and Tiropanis 2015
"""

import logging
import numpy as np

from LSABase import LSABase
from LSAUtils import normal, zscore

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


class SDM(LSABase):
    """
    class holding the SDM method
    """   
    
    def create_d_surface(self, series, lags):
        """
        Creates a tensor containing the pairwise distances between the leading
        and the lagged series for each x, y pair in series.
        
        Args:
            series: list of series in (x, y) pairs where x is the lagged series
            lags: number of lags of each to be compared in te model 
        Returns:
            self.lag_series: lags of x for x in series 
            self.lead_series value of y for y in series 
            self.D: N*len(series)*lags shaped tensor of pairwise distances
        """

        self.series = series 
        self.zseries = []
        for x, y in self.series:
            logger.info("t-series dimensions = {}, {}".format(
                        str(x.shape), str(y.shape)
                        ))
            self.zseries.append((zscore(x), zscore(y)))

        D = np.zeros((len(self.series[0][0])-lags, len(self.series), lags))
        lag_series = np.zeros(D.shape)
        lead_series = np.zeros((len(self.series[0][0])-lags, len(self.series)))
        logger.info("D dim = {}".format(D.shape))
        for t in range(lags, len(self.zseries[0][0]), 1):
            for i, (a, b) in enumerate(self.zseries):
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
        Iterates over the tensor calculating the Bayesian optimal estimate
        of the probability distribution that minimise sum(wt*wd)
        
        Args:
            top_paths: boolean indicating wether or not to use the TOP paths
                        structure 
            Returns:
                self.W: probabilty surface formed by using the SDM method 
                self.error: fit of the model sum(D_t*W_t-1) for t in T
                self.forecast: forecast the model gives at t 
        """

        PrD = normal(self.D, 0, 1)
        W = np.ones(self.D.shape)/(self.lags*len(self.series))
        top_structure = np.zeros((len(self.D), len(self.series)))
        sub_structure = np.zeros(self.D.shape)
        logger.info("W dim = {}".format(W.shape))
        for t in range(1, len(self.D), 1): 
            w = W[t-1]
            if top_paths:
                for i in range(len(self.D[t-1])):
                    w[i] = W[t-1][i]
                    w[i][1:] += W[t-1][i][:-1]
                    w[i][:-1] += W[t-1][i][1:]
                w[1:-1] = w[1:-1]/3
                w[0] = w[0]/2
                w[-1] = w[-1]/2
            
            W[t] = (w+0.00001)*PrD[t]
            W[t] = W[t]/np.sum(W[t])
            top_structure[t] = np.sum(W[t], 1)
            sub_structure[t] = (W[t].T/top_structure[t]).T 
            
        self.error = abs(np.sum(W[:-1]*self.D[1:]))
        self.forecast = np.sum(sub_structure[:-1]*self.lag_series[1:], 2)
        logger.info("Error={}".format(self.error))
        self.W = W
        self.top_structure = top_structure
        self.sub_structure = sub_structure
        return self 




    

    
    