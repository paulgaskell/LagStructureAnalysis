"""
The Thermal Optimal Path method (TOP) is an algorthm proposed by Sornette and
Zhou (2007) which attempts to find the optimal lead/lag structure between two 
time-series. The algorithm calculates the lowest average energy path over a 
random energy landscape formed by the distances between the two series. 
"""

import numpy as np
import logging

from LSABase import LSABase
from LSAUtils import zscore 

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
     
class TOP(LSABase):
    """
    class holding the TOP method
    """

    def create_d_surface(self, x, y):
        """
        Create a len(x)*len(y) matrix D containing the pairwise 
        distances (x(i)-y(j))**2 for all i<=len(x), j<=len(y
		
		To make this more efficient we rotate the matrix by 45 degrees and 
		unroll each corner to make a pyramid where each layer has dim 1+t*2
		
		Args:
			x, y: time series being compared 
		Returns:
			self.D: pyramid of pairwise distances 
			self.lags: pyramid of corresponding lags 
        """

        self.x = zscore(x)
        self.y = zscore(y)
    
        logger.info("t-series dimensions = {}, {}".format(
                        str(x.shape), str(y.shape)
                        ))
        
        D = []
        lags = []
        for t in range(0, len(x), 1):
            D.append(np.zeros(t*2+1))
            lags.append(np.zeros(t*2+1))
            for tx in range(0, t+1, 1):
                D[t][tx] = (x[tx]-y[t])**2
                lags[t][tx] = t-tx
            for ty in range(1, t+1, 1):
                D[t][-ty] = (x[t]-y[ty-1])**2
                lags[t][-ty] = -(t-ty+1)

        self.lags = lags
        self.D = D
        return self
        
    def top(self, h):
        """
        Main TOP method - recurse over the matrix starting at point [0,0],
        calculate: 
            x[n][m] = {(x[n-1][m]+x[n-1][m-1]+x[n][m-1]])/3}*exp(-D[n][m]/h)
        For all n, m    
		
        Args:
            h: the hyperparameter controling the flexibility of the TOP lag 
                structure
        Returns:
            self.W: the probability surface formed using the TOP method
            self.h: the hyperperameter used to form W
            self.predicted_lag: the lag structure found by the algorithm 
        """
    
        predicted_lags = []
        W = [np.zeros(len(i)) for i in self.D]
        W[0] += 1
        for t in range(1, len(self.D), 1):
            W[t][1:-1] = W[t-1].copy()
            W[t][:-2] += W[t-1]
            W[t][2:] += W[t-1]
            W[t] += 0.00001
            
            W[t] = W[t]*np.exp(-self.D[t]/h)
            W[t] = W[t]/np.sum(W[t])

            predicted_lags.append(np.sum(W[t]*self.lags[t]))
        
        self.predicted_lags = predicted_lags
        self.W = W 
        self.h = h
        return self 

    
