"""
Part I, First activity in Parameter Fitting Tutorial
Modified by Katie Eckert from ASTR502 activity written by Sheila Kannappan
June 24, 2015
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import scipy.stats as stats
import numpy.random as npr
import pylab
pylab.ion()

pi=np.pi

#Generating fake data set to start with:
alphatrue=2. # slope
betatrue=5.  # intercept
errs=2.5 # sigma (amplitude of errors)

narr=50. # number of data points
xvals = np.arange(narr) + 1. # xvals range from 1-51
yvals = alphatrue*xvals + betatrue+ npr.normal(0,errs,narr) # yvals 
# what does npr.normal do?

# plot fake data
plt.figure(1) 
plt.clf()
plt.plot(xvals,yvals,'b*',markersize=10)

# determine slope & y-intercept using analytic solution from derivation sheet

#alphaest=(np.mean(xvals)*np.mean(yvals)-np.mean(xvals*yvals)) / \
#   (np.mean(xvals)**2 -np.mean(xvals**2)) #  from derivation sheet

#betaest= ? # calculate estimate of y-intercept 


# what are MLE values of slope and y-intercept?

#print("slope = %f" %alphaest)
#print("y-intercept = %f" %?)

# overplot the best fit solution
#yfitvals=xvals*alphaest+betaest
#plt.plot(xvals,yfitvals,'r')

# compute analytic uncertainties on slope and y-intercept from derivation sheet
#alphaunc=?
#betaunc=?

#print("uncertainty on alpha is %0.7f" % (alphaunc))
#print("uncertainty on beta is %0.7f" % (betaunc))


# solution using python solver np.polyfit
# third parameter is order of fit, 1 for linear
#pfit=np.polyfit(xvals,yvals,1) # returns highest power first 

#print("slope = %f" %pfit[0])
#print("y-intercept = %f" %?)

# Do you get the same result as in analytical case?

# Note that not all problems have analytical solutions like linear regression

# can also obtain errors from the diagonal terms of the covariance
# matrix, which is the inverse of the Hessian matrix given on derivation sheet
# also computed in np.polyfit...

#pfit,covp=np.polyfit(xvals,yvals,1,cov='True') # returns highest power first
# setting cov='True' returns the covariance matrix
# how do we get the errors from it?
#print("slope is %0.7f +- %0.7f" % (pfit[0], np.sqrt(covp[0,0])))
#print("intercept is %0.7f +- %0.7f" % (?, ?)

# are those errors the same as in analytical solution?
# what happens if you increase the number of points?
# what happens if you decrease the number of points?


