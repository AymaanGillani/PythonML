from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

style.use('fivethirtyeight')

xs = [1, 2, 3, 4, 5, 6]
ys = [5, 4, 6, 5, 6, 7]

def bestfitslope(xs, ys):
    meanx = mean(xs)
    meany = mean(ys)
    meanxy = mean(x*y for x,y in zip(xs,ys))
    meanx2=mean(x**2 for x in xs)
    m=((meanx*meany)-meanxy)/(meanx**2-meanx2)
    b=meany-m*meanx
    return m,b

m,b=bestfitslope(xs,ys)

def predict(x):
    return m*x+b

def squared_error(ys_original,ys_line):
    return sum((yl-yo)**2 for yl,yo in zip(ys_line,ys_original))

def coef_of_determination(ys_original,ys_line):
    ys_mean_line=[mean(ys_original)for y in ys_original]
    squared_error_regr=squared_error(ys_original,ys_line)
    squared_error_y_mean=squared_error(ys_original,ys_mean_line)
    r2=1-(squared_error_regr/squared_error_y_mean)
    return r2

regression_line=[m*x+b for x in xs]
r_squared=coef_of_determination(ys,regression_line)

print(r_squared)
plt.scatter(xs, ys)
plt.plot(xs,regression_line)
plt.scatter(8,predict(8),c='g')
plt.show()
