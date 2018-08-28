"""
The Thermal Optimal Path method (TOP) is an algorthm proposed by Sornette and
Zhou (2007) which attempts to find the optimal lead/lag structure between two 
time-series as the lowest average energy path over a random energy landscape
formed by the distances between the two series. 
"""

import numpy as np
import seaborn as sns
from pylab import show
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("imports have completed")


def top_evoopt(top, epochs):
    """Use an evolutionary algorithm to optimise TOP fit
    
    Uses an evolutionary algorithm with proportional selection to 
    find the h which minimises sum(d) - i.e. the h which minimises 
    total distance 
    
    Args:
        top: instance of the top class with calculated pr surface
        epochs: number of training epochs
    Returns:
        h: optimised h
        E: error at h=h
    """

    h = 1.
    error = 1000000
    for i in range(0, epochs+1, 1):
        step = np.random.normal(0, 0.1)
        h_temp = h+step
        if h_temp < 0.1: 
            h_temp = 0.1
        if h_temp > 10:
            h_temp = 10

        top = top.top(h_temp)
        error_temp = sum(sum(top.TMAP*top.D))
        
        if error_temp <= error:
            h = h_temp
            error = error_temp
        
        logger.info("epoch={}, h={}, error={}, step={}".format(
                        i, h, error, step)
                        )
    
    return h, error
     
class TOP:
    """
    class holding the TOP method
    
    example usage below
    """
    
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
        logger.info("t-series dimensions = {}, {}".format(
                        str(x.shape), str(y.shape)
                        ))
    
    def create_d_surface(self):
        """
        Create a len(x)*len(y) matrix D containing the pairwise 
        distances (x(i)-y(j))**2 for all i<=len(x), j<=len(y)
        """
        
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
            self.TMAP: the probability surface formed using the TOP method
            self.h: the hyperperameter used to form TMAP
        """
    
        TMAP = np.ones(self.D.shape)
        for t in range(1, len(self.x), 1):
            Z = 0.
            lag = [0, None]
            for n, m in self._recursion(t):
                crds = [i for i in [(n-1, m), (n-1, m-1), (n, m-1)] if min(i) >= 0]
                G = np.mean([TMAP[i] for i in crds])
                TMAP[n][m] = (G+0.00001)*np.exp(-self.D[n][m]/h)
                Z += TMAP[n][m]
            for n, m in self._recursion(t):
                TMAP[n][m] = TMAP[n][m]/Z
                if TMAP[n][m] > lag[0]:
                    lag[0] = TMAP[n][m]
                    lag[1] = n-m
                
        self.TMAP = TMAP 
        self.h = h
        return self 

        
if __name__ == "__main__":

    # EXAMPLE USAGE

    # make an example time series
    x = np.random.normal(0, 1, 100)
    y = np.random.normal(0, 1, 100)
    for n, i in enumerate(x):
        if n < 10:
            y[n] = x[n]
        elif n < 30:
            y[n] = x[n-10]
        elif n < 70:
            y[n] = x[n+20]
        else:
            y[n] = x[n]
    
    # run the TOP method
    top = TOP(x, y)
    top = top.create_d_surface()    
    h, error = top_evoopt(top, 40)

    top = top.top(h)
    sns.heatmap(top.D)
    sns.heatmap(top.TMAP)    
    show()
    