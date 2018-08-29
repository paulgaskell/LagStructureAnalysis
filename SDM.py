
"""
Base class of the SDM algorithm

TODO
    labels on graph axes
    examples 
    optimisations for Pr
    OTCP recursion structure 
    max a posteriori error 
    linear regression comparison
    ackage structure - pip installable 
    
"""

import numpy as np
import sys
import logging 
from pylab import show
import matplotlib.pyplot as plt
import seaborn as sns
from functools import partial 

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("package install is complete")

class Examples:
    """
    Making a variety of example t-series for testing/illustration 
    purposes
    """
    
    def __init__(self):
        self.x = None
        self.y = None
    
    def make_y(self):
        self.y = np.random.normal(0, 1, 100)
        return self
    
    def make_x(self, lag):
        self.x = np.append(
                    self.y[lag:]+np.random.normal(0, 2, 100-lag), 
                    np.random.normal(0, 1, lag)
                    )
        return self
        
    def out(self):
        return self.x, self.y

class Pr_surfaces:
    
    def create_pr_surface(self, lag, h):
        """
        Args: x, y, loss, lag, Pr
        Returns: 
            fills d with probabilities calculated as Pr(loss(x, y)), for 
            each lag of x and y
        """
        self.lag = lag
        self.h = h
        self.D = np.zeros((len(self.x)-lag, 2, lag-1))
        self.V = np.zeros(self.D.shape)
		
        loss = lambda x, y: (x-y)**2 
        t = self.lag
        end = len(self.x)
        while t < end:
            self.V[t-lag][0] = self.x[t-lag:t-1]
            self.V[t-lag][1] = -self.x[t-lag:t-1]
            self.D[t-lag] = loss(self.y[t], self.V[t-lag])
            t += 1
			
        self.D = np.exp(-self.D/self.h)
        return self

class Recursions:

    def recursive_bayes_simple(self):		
		
        self.W = np.zeros(self.D.shape)
		
        w = np.ones(self.D[0].shape)
        w = w/np.sum(w)
        for t, i in enumerate(self.D):
            w = (w+0.00001)*i
            w = w/np.sum(w)
            self.W[t] = w
			
        return self

class Results:

    def results(self):
        self.forecast = np.sum(np.sum(self.W*self.V, 1), 1)
        print self.forecast
        self.mae = np.mean(np.abs(self.forecast-self.y[self.lag:]))
        return self
       
    def heatmap(self, surface='W'):
        if surface == 'W':
            s = self.W.copy()
        elif surface == 'D':
            s = self.D.copy()
        elif surface == 'V':
            s = self.V.copy()
        else:
            logger.error("you need to specifiy a plottable surface, W, D, V")

        fig, axn = plt.subplots(2, 1, sharex=True, sharey=True)
        cbar_ax = fig.add_axes([.91, .3, .03, .4])

        for i, ax in enumerate(axn.flat):
            sns.heatmap(
                np.array([j[i] for j in s]).T, 
                ax=ax,
                cbar=i== 0,
                vmin=0, 
                vmax=1,
                cbar_ax=None if i else cbar_ax
                )

        #fig.tight_layout(rect=[0, 0, .9, 1])
        show()
        

class SDM(Pr_surfaces, Recursions, Results):
    def __init__(self, x, y):
        self.x = x
        self.y = y

        try:
            assert len(self.x) == len(self.y)
        except AssertionError as E:
            logger.error("time series must be of equal length")
            sys.exit()
	
        logger.info("tseries loaded lengths {}".format(str(x.shape)))
		
class Optimiser:
    def __init__(self, sdm):
        self.sdm = sdm

    def evo_optimiser(self):
        h = np.random.random()
        err = 1
        t = 0
        while t < 100:
            a = np.random.normal(0, err)
            self.sdm = self.sdm.create_pr_surface(10, h+a).recursive_bayes_simple(
                    ).results()
            
            err_ = self.sdm.mae
            
            if err_ < err:
                h = h+a
                err = err_
            
            print err, h, a, t
            t += 1
            
        return self
			
if __name__ == "__main__":

    x, y = Examples().make_y().make_x(5).out()    
    sdm = SDM(x, y)
    
    # np.exp(-((x-y)**2)/h)
    # np.exp(-(((a+bx)-y)**2)/h)
    
    op = Optimiser(sdm)
    op = op.evo_optimiser()

    op.sdm.heatmap()
	