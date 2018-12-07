
from TOP import TOP
from SDM import SDM

def test_TOP():
    top = TOP()
    x, y = top.simple_example()
    top = top.create_d_surface(x, y).top(1)
    top.plt_predicted_lag_line().plt_display()
    
def test_SDM():
    sdm = SDM()
    x, y = sdm.simple_example()
    sdm = sdm.create_d_surface([(x, y)], 50).sdm()
    sdm.plt_multi_heatmap().plt_display()

test_TOP()    
test_SDM()
