"""
The Thermal Optimal Path method (TOP) is an algorthm proposed by Sornette and
Zhou (2007) which attempts to find the optimal lead/lag structure between two 
time-series. The algorithm calculates the lowest average energy path over a 
random energy landscape formed by the distances between the two series. 
"""

import numpy as np
from pylab import show, figure
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("imports have completed")
     
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
            self.predicted_lag: the lag structure found by the algorithm 
        """
    
        predicted_lags = []
        TMAP = np.ones(self.D.shape)
        for t in range(1, len(self.x), 1):
            Z = 0.
            lag = [0, None]
            for n, m in self._recursion(t):
                crds = [i for i in [(n-1, m), (n-1, m-1), (n, m-1)] if min(i) >= 0]
                G = np.mean([TMAP[i] for i in crds])
                TMAP[n][m] = (G+0.00001)*np.exp(-self.D[n][m]/h)
                Z += TMAP[n][m]
            
            mean_lag = 0.            
            for n, m in self._recursion(t):
                TMAP[n][m] = TMAP[n][m]/Z
                mean_lag += TMAP[n][m]*(n-m)
                    
            predicted_lags.append(mean_lag)
        
        self.predicted_lags = predicted_lags
        self.TMAP = TMAP 
        self.h = h
        return self 

        
if __name__ == "__main__":

    # EXAMPLE USAGE

    # make an example time series
    x = np.random.normal(0, 1, 200)
    y = np.random.normal(0, .4, 200)
    for n, i in enumerate(x):
        if n < 20:
            y[n] += x[n]
        elif n < 70:
            y[n] += x[n-10]
        elif n < 120:
            y[n] += x[n]
        elif n < 170:
            y[n] += x[n+10]
        else:
            y[n] += x[n]
    
    y = (y-np.mean(y))/np.std(y)
    
    # run the TOP method
    top = TOP(x, y)
    top = top.create_d_surface().top(.5)    
    fig = figure()
    ax = fig.add_subplot(211)
    ax1 = fig.add_subplot(212)
    ax.imshow(top.TMAP, aspect='auto')
    ax1.plot(top.predicted_lags)
    show()
    