
"""
Base class for LSA 
"""

import numpy as np
from pylab import show, figure

class Examples:
    def simple_example(self):
        x = np.random.normal(0, 1, 200)
        y = np.random.normal(0, 1, 200)
        for n, i in enumerate(x):
            if n < 20:
                y[n] += x[n+1]
            elif n < 70:
                y[n] += x[n-10]
            elif n < 120:
                y[n] += x[n-30]
            elif n < 150:
                y[n] += x[n-35]
            else:
                y[n] += x[n]
    
        y = (y-np.mean(y))/np.std(y)
        return x, y

class Charts:
    def simple_heatmap(self):       
        fig = figure()
        ax = fig.add_subplot(111)
        ax.imshow(self.W, aspect='auto')
        return self 
        
    def multi_heatmap(self):    
        fig = figure()
        for s in range(1, len(self.series)+1, 1):
            ax = fig.add_subplot(len(self.series),1,s)    
            ax.imshow([i[s-1] for i in self.W.T], aspect='auto')
        return self
        
    def predicted_lag_line(self):
        pass
        
    def display(self):
        show()
        
    
class LSABase(Examples, Charts):
    """
    examples, grahps, resultsobject handling 
    """
    
    
    
    
        

    
    
    
    