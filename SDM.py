
"""
Base class of the SDM algorithm
"""

import numpy as np
import sys
import logging 
from pylab import show
import seaborn as sns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SDM:
    def __init__(self, x, y):
        self.x = x
        self.y = y

        try:
            assert len(self.x) == len(self.y)
        except AssertionError as E:
            logger.error("time series must be of equal length")
            sys.exit()
	
        logger.info("tseries loaded lengths {}".format(str(x.shape)))
	
    ####### PROBABILITY SURFACES 
	
    def create_pr_surface(self, lag, loss, Pr):
        """
        Args: x, y, loss, lag, Pr
        Returns: 
            fills d with probabilities calculated as Pr(loss(x, y)), for 
            each lag of x and y
        """
        self.lag = lag
        self.loss = loss
        self.Pr = Pr
        self.D = np.zeros((len(self.x)-lag, ((lag-1)*2)))
        self.V = np.zeros(self.D.shape)
		
        t = self.lag
        end = len(self.x)
        loss = self.loss
        lag = self.lag
        while t < end:
            self.V[t-lag][:lag-1] = self.x[t-lag:t-1]
            self.V[t-lag][lag-1:] = self.y[t-lag:t-1][::-1]
            self.D[t-lag] = loss(self.y[t], self.V[t-lag])
            t += 1
			
        self.D = self.Pr(self.D)
        return self
		
    ######## RECURSION FUNCTIONS 
		
    def recursive_bayes_simple(self):		
		
        self.W = np.zeros(self.D.shape)
		
        w = np.ones(len(self.D[0]))
        w = w/np.sum(w)
        for t, i in enumerate(self.D):
            w = (w+0.00001)*i
            w = w/np.sum(w)
            self.W[t] = w
			
        return self
	
    ######### PLOTTING/RESULTS ANALYSIS 
	
    def results(self):
        self.forecast = np.sum(self.W*self.V, 1)
        self.mae = np.mean(np.abs(self.forecast-self.y[self.lag:]))
       
    def heatmap(self, surface='W'):
        if surface == 'W':
            s = self.W.copy()
        elif surface == 'D':
            s = self.D.copy()
        elif surface == 'V':
            s = self.V.copy()
        else:
            logger.error("you need to specifiy a plottable surface, W, D, V")

        g = sns.heatmap(sdm.W.T)
        show()
			
if __name__ == "__main__":
    sdm = SDM(
            np.random.normal(0, 1, 100), 
            np.random.normal(0, 1, 100)
            )
			
    sdm.create_pr_surface(
        10,
        lambda x, y: x-y,
        lambda x: np.exp(-(x**2)/10)
        ).recursive_bayes_simple().results()

    print(sdm.mae)
    sdm.heatmap()
	