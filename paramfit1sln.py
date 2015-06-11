# -*- coding: utf-8 -*-
"""
Created on Thu Feb 06 21:50:47 2014

@author: sheila
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import scipy.stats as stat
import numpy.random as npr
import pylab
pylab.ion()

pi=np.pi

#Generating fake data set to start with:
alphatrue=2. # slope
betatrue=5.  # intercept
errs=2.5 # sigma (amplitude of errors)

narr=100. # number of data points
xvals = np.arange(narr) + 1. # xvals range from 1-51
yvals = alphatrue*xvals + betatrue+ npr.normal(0,errs,narr) # yvals 
# what does npr.normal do? - adds scatter in y-direction from Gaussian distribution 

# plot fake data
plt.figure(1) 
plt.clf()
plt.plot(xvals,yvals,'b*',markersize=10)

#determine slope & y-intercept using analytic solution from derivation sheet

alphaest=(np.mean(xvals)*np.mean(yvals)-np.mean(xvals*yvals)) / \
   (np.mean(xvals)**2 -np.mean(xvals**2)) #  from derivation sheet
betaest= np.mean(yvals)-alphaest*np.mean(xvals) # answer

# what are best values of slope and y-intercept?
print("slope = %f" %alphaest)
print("y-intercept = %f" %betaest) # answer

# overplot the best fit solution
yfitvals=xvals*alphaest+betaest
plt.plot(xvals,yfitvals,'r')

# compute analytic uncertainties on slope and y-intercept from derivation sheet
# slope error - break up into pieces to keep simple
inputa1=np.sum((yvals-(xvals*alphaest+betaest))**2.)
inputa2=np.sum((xvals-np.mean(xvals))**2)
inputa3=1./(narr-2.)
alphaunc=np.sqrt(inputa3*inputa1/inputa2)

inputb1=inputa3*inputa1 
inputb2=(1./narr)
inputb3=(np.mean(xvals)**2)/(np.sum((xvals-np.mean(xvals))**2))
betaunc=np.sqrt(inputb1*(inputb2+inputb3))

print("uncertainty on alpha is %0.7f" % (alphaunc))
print("uncertainty on beta is %0.7f" % (betaunc))



# B: solution using python solver np.polyfit
# third parameter is order of fit, 1 for linear
pfit=np.polyfit(xvals,yvals,1) # returns highest power first 

print("slope = %f" %pfit[0])
print("y-intercept = %f" %pfit[1])

# Do you get the same result as in analytical case?

# Note that not all problems have analytical solutions like linear regression

# can also obtain errors from the diagonal terms of the covariance
# matrix, which is the inverse of the Hessian matrix given on derivation sheet
# also computed in np.polyfit...

pfit,covp=np.polyfit(xvals,yvals,1,cov='True') # returns highest power first
# setting cov='True' returns the covariance matrix
# how do we get the errors from it?

print("slope is %0.7f +- %0.7f" % (pfit[0], np.sqrt(covp[0,0])))
print("intercept is %0.7f +- %0.7f" % (pfit[1], np.sqrt(covp[1,1])))

# Are those errors the same as in analytical solution?
# What happens if you increase the number of points?
# What happens if you decrease the number of points?

print("relative difference in the slope uncertainties as percentage of slope value for N = %f data points is %f" % (narr, 100.*(np.sqrt(covp[0,0])-alphaunc)/alphaest))
print("relative difference in the y-intercept uncertainties as percentage of y-intercept value for N = %f data points is %f" % (narr, 100.*(np.sqrt(covp[1,1])-betaunc)/betaest))
