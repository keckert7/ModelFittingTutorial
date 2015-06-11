"""
Part 2, third activity in Parameter Fitting Tutorial
Modified by Katie Eckert from ASTR502 activity written by Sheila Kannappan
June 24, 2015
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import scipy.stats as stat
import numpy.random as npr
import pylab
pylab.ion()

pi=np.pi

# Generating fake data set (same as in paramfit1.py) to start with:
alphatrue=2. # slope
betatrue=5.  # intercept
errs=2.5 # sigma (smplitude of errors)

narr=50. # number of data points
xvals = np.arange(narr) + 1.
yvals = alphatrue*xvals + betatrue+ npr.normal(0,errs,narr)
plt.figure(1) # plot of fake data
plt.clf()
plt.plot(xvals,yvals,'b*',markersize=10)

# Bayesian numerical solution finding the full
# posterior probability distribution of a grid of models

# setup the grids
gridsize1=1000
gridsize2=100
alphaposs=np.arange(gridsize1) / 100. # what values are we considering?
betaposs=np.arange(gridsize2) / 10.  # and here?

print("min slope is %f and max slope is %f" % (np.min(alphaposs), np.max(alphaposs)))
print("min y-int is %f and max y-int is %f" % (np.min(betaposs), np.max(betaposs)))

# what are our (implicit) priors?
# assuming a flat distribution in slope and y-intercept

# but is that true? check that priors are actually equally spaced: 
# plot lines with the different values of y-int or slope for a line from (0,1)
xx=np.arange(0,1,0.1)  # set up array of dummy values

# test y-intercept spacing
plt.figure(2) 
plt.clf()
for i in range(len(betaposs)):       # loop over all y-int values
    plt.plot(xx,xx+betaposs[i],'b-') # plot lines with different y-int values

plt.xlim(0,1)
plt.ylim(0,1)
plt.title("test y-int prior")
# yes - evenly spaced uniform input distribution

# test slope
plt.figure(3)
plt.clf()
for i in range(len(alphaposs)):       # loop over all slope values
    plt.plot(xx,xx*alphaposs[i],'b-') # plot lines with different slope

plt.xlim(0,0.2) # may need to zoom in
plt.ylim(0,0.2)
plt.title("test slope prior")

# The steeper slopes are sampled more finely weighting them unequally in the grid set.

# To use a flat grid set in slope & y-intercept, we must assume a prior, which is equal to (1+beta^2)^(-3/2) 
# read through (http://jakevdp.github.io/blog/2014/06/14/frequentism-and-bayesianism-4-bayesian-in-python/) for more details on obtaining this prior

# remember the Bayesian likelihood is 
#P(M|D)=P(D|M)*P(M)/P(D)
#P(D|M) is the chi^2 likelihood
#P(M) is the prior
#P(D) is the normalization
# compute log likelihood
#bayesianlike=exp(-1*chisq/2)*prior
#lnbayesianlike=-1*chisq/2 + ln(prior)

# compute likelihoods for all possible models in original setup
lnlikes=np.zeros((gridsize1,gridsize2)) # setup an array to contain those values

for i in xrange(gridsize1):  # loop over all possible values of alpha
    for j in xrange (gridsize2): # loop over all possible values of beta
        modelvals = alphaposs[i]*xvals+betaposs[j] # compute yfit for given model
        resids = (yvals - modelvals) # compute residuals for given grid model
        chisq = np.sum(resids**2 / errs**2) # compute chisq likelihood
        #priorval=1.0                         # uniform prior
        priorval=(1.+betaposs[j]**2)**(-3./2.) # compute prior
        #lnlikes[i,j] = ? + ? 
        lnlikes[i,j]=-1.*chisq/2. + np.log(priorval) # plus a constant, will normalize out later
# now we have a full array of likelihoods computed for each possible model
# what if we want to know the probability distribution for the slope?
# we can find out by marginalizing over the intercept
# this means integrating over likelihoods so we have to take exp(lnlikes)
likes=np.exp(lnlikes)
marginalizedlikes_slope = np.sum(likes,axis=1) / np.sum(likes)
# why do we sum over axis 1 in the numerator, but
# the whole array in the denominator?

# plot the likelihood distribution of slope values
plt.figure(4) # plot of posterior prob. distribution for slope
plt.clf()
plt.plot(alphaposs,marginalizedlikes_slope,'g.',markersize=10)
plt.xlabel("alpha")
plt.ylabel("posterior probability distribution of alpha")
# zoom in on the region of significant probability
# and estimate the error from the graph
#plt.xlim(1.5,2.5)

# now marginalize over the slope to see the posterior distribution in y-intercept
marginalizedlikes_yint = np.sum(likes,axis=0) / np.sum(likes)
plt.figure(5)
plt.clf()
plt.plot(betaposs,marginalizedlikes_yint,'g.',markersize=10)
plt.xlabel("beta")
plt.ylabel("posterior probability distribution of beta")



