"""
The Thermal Optimal Path method (TOP) is an algorthm proposed by Sornette and
Zhou (2007) which attempts to find the optimal lead/lag structure between two 
time-series. The algorithm calculates the lowest average energy path over a 
random energy landscape formed by the distances between the two series. 
"""

import numpy as np
from pylab import show, figure
import logging

from LSABase import LSABase

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("imports have completed")
     
class TOP(LSABase):
    """
    class holding the TOP method
    
    example usage below
    """

    def create_d_surface(self, x, y):
        """
        Create a len(x)*len(y) matrix D containing the pairwise 
        distances (x(i)-y(j))**2 for all i<=len(x), j<=len(y)
        """

        self.x = x
        self.y = y
    
        logger.info("t-series dimensions = {}, {}".format(
                        str(x.shape), str(y.shape)
                        ))
        
        D = np.zeros((len(self.x), len(self.y)))
        for n, i in enumerate(self.x):
            for m, j in enumerate(self.y):
                D[n][m] = (i-j)**2
        self.D = D
        
        logger.info("D dimensions = {}".format(str(D.shape)))
        return self
        
    def _recursion(self, g):
        """
        Recursion pattern for applying the OTCP pathways
        """
        
        for g_ in range(0, g+1, 1):
            yield g, g_
            if g != g_: 
                yield g_, g
                            
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
        W = np.ones(self.D.shape)
        for t in range(1, len(self.x), 1):
            Z = 0.
            lag = [0, None]
            for n, m in self._recursion(t):
                crds = [i for i in [(n-1, m), (n-1, m-1), (n, m-1)] if min(i) >= 0]
                G = np.mean([W[i] for i in crds])
                W[n][m] = (G+0.00001)*np.exp(-self.D[n][m]/h)
                Z += W[n][m]
            
            mean_lag = 0.            
            for n, m in self._recursion(t):
                W[n][m] = W[n][m]/Z
                mean_lag += W[n][m]*(n-m)
                    
            predicted_lags.append(mean_lag)
        
        self.predicted_lags = predicted_lags
        self.W = W 
        self.h = h
        return self 

        
if __name__ == "__main__":

    
    # run the TOP method
    top = TOP()
    x, y = top.simple_example()
    top = top.create_d_surface(x, y).top(.5)
    top.simple_heatmap().display()

    