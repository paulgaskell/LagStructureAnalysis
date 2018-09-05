
import numpy as np 
from scipy.stats import norm
from LSAUtils import normal
from pylab import figure, show

x = np.random.normal(0, 1, 100)
y = (x)+4
fig = figure()
axs = [
        fig.add_subplot(121),
        fig.add_subplot(122)
        ]
        
#axs[0].scatter(x, norm.pdf(x, 0, 1))
axs[0].scatter(y, norm.pdf(y, np.mean(y), np.std(y)))
axs[1].scatter(x, norm.pdf(x, np.mean(x), np.std(x)))

show()


