
"""
Base class for LSA 
"""

import numpy as np
from pylab import show, figure
from scipy.stats import linregress
import logging 
from LSAUtils import zscore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Examples:
    def simple_example(self):
        x = np.random.normal(0, 1, 200)
        error = np.random.normal(0, 1, 200)
        y = np.zeros(x.shape)
        a = 7
        b = 6
        logger.info(repr(linregress(x, a+x*b+error)))
        for n, i in enumerate(x):
            if n < 20:
                y[n] = a+x[n+1]*b
            elif n < 70:
                y[n] = a+x[n-10]*b
            elif n < 120:
                y[n] = a+x[n-30]*b
            else:
                y[n] = a+x[n-35]*b

        y = y+error
        return x, y

class Charts:
    def plt_simple_heatmap(self):       
        fig = figure()
        ax = fig.add_subplot(111)
        ax.imshow(self.W, aspect='auto')
        return self 
        
    def plt_multi_heatmap(self):    
        fig = figure()
        for s in range(1, len(self.series)+1, 1):
            ax = fig.add_subplot(len(self.series),1,s)    
            ax.imshow([i[s-1] for i in self.W.T], aspect='auto')
        return self
        
    def plt_predicted_lag_line(self):
        fig = figure()
        ax = fig.add_subplot(111)
        ax.plot(self.predicted_lags)
        return self
    
    def plt_multi_forecast(self):
        
        depth = len(self.series)+1
        axs = []
        
        fig = figure()
        axs.append(fig.add_subplot(depth, 1, 1))
        axs[-1].plot(self.top_structure)
            
        f = 3
        for n, i in enumerate(self.forecast.T):
            axs.append(fig.add_subplot(depth, 2, f))
            axs[-1].plot(self.lead_series.T[n][1:])
            axs[-1].plot(self.forecast.T[n])
            f += 1
            axs.append(fig.add_subplot(depth, 2, f))
            axs[-1].scatter(self.forecast.T[n], self.lead_series.T[n][1:])
            f += 1
        return self
    
    def plt_display(self):
        show()
        
    
class LSABase(Examples, Charts):
    """
    examples, grahps, resultsobject handling 
    """
    
    
    
    
        

    
    
    
    