
"""
Base class of the SDM algorithm
"""

import numpy as np
import sys
import logging 
from pylab import show
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Pr_surfaces:
    
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
        self.D = np.zeros((len(self.x)-lag, 4, lag-1))
        self.V = np.zeros(self.D.shape)
		
        t = self.lag
        end = len(self.x)
        loss = self.loss
        lag = self.lag
        while t < end:
            self.V[t-lag][0] = self.x[t-lag:t-1]
            self.V[t-lag][1] = -self.x[t-lag:t-1]
            self.V[t-lag][2] = self.y[t-lag:t-1]
            self.V[t-lag][3] = -self.y[t-lag:t-1]
            self.D[t-lag] = loss(self.y[t], self.V[t-lag])
            t += 1
			
        self.D = self.Pr(self.D)
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

        fig, axn = plt.subplots(4, 1, sharex=True, sharey=True)
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

        fig.tight_layout(rect=[0, 0, .9, 1])
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
		
	
			
if __name__ == "__main__":

    y = np.random.normal(0, 1, 100)
    x = np.append(y[5:], np.random.normal(0, 1, 5))
    
    
    sdm = SDM(x, y)
	
    
    sdm.create_pr_surface(
        10,
        lambda x, y: (x-y)**2,
        lambda x: np.exp(-(x**2))
        ).recursive_bayes_simple().results()

    print(sdm.mae)
    sdm.heatmap()
	