"""
Part 2, third activity in Parameter Fitting Tutorial
Modified by Katie Eckert from ASTR502 activity written by Sheila Kannappan
June 24, 2015
"""

import numpy as np
import matplotlib.pyplot as plt
import numpy.random as npr
import pylab
pylab.ion()


# Generating fake data set (same as in paramfit1.py) to start with:
alphatrue=2. # slope
betatrue=5.  # intercept
errs=2.5 # sigma (amplitude of errors)

narr=50. # number of data points
xvals = np.arange(narr) + 1.
yvals = alphatrue*xvals + betatrue+ npr.normal(0,errs,narr)

plt.figure(1) # plot of fake data
plt.clf()
plt.plot(xvals,yvals,'b*',markersize=10)
plt.xlabel('xvalues')
plt.ylabel('yvalues')

# Bayesian numerical solution finding the full
# posterior probability distribution of a grid of models

# Setup the grids over parameter space
gridsize1=1000
gridsize2=100
alphaposs=np.arange(gridsize1) / 100. # what values are we considering?
betaposs=np.arange(gridsize2) / 10.  # and here?

#print("min slope is %f and max slope is %f" % (np.min(alphaposs), ?))
#print("min y-int is %f and max y-int is %f" % (?, np.max(betaposs)))

# What are our implicit priors?


# Check to see that the model space from our choice of grid spacing 
# is uniformly spaced by plotting lines with the different values of
# the y-intercept and slope for a line with x values ranging from (0,1)
#xx=np.arange(0,1,0.1)  # set up array of dummy values

# Test y-intercept spacing
#plt.figure(2) 
#plt.clf()
#plt.subplot(121)
#for i in range(len(betaposs)):       # loop over all y-int values
#    plt.plot(xx,xx+betaposs[i],'b-') # plot lines with different y-int values

#plt.xlim(0,1) # limit to small range
#plt.ylim(0,1) # limit to small range
#plt.xlabel("x values")
#plt.ylabel("y values for several values of y-int (y=x+beta)")
#plt.title("test y-int prior")
# yes - evenly spaced uniform input distribution


# Test slope
#plt.subplot(122)
#for i in range(len(?)):       # loop over all slope values
#    plt.plot(?,?)          # plot lines with different slope values

#plt.xlim(0,1) 
#plt.ylim(0,0.2) # will need to zoom in to distinguish lines of different slope values
#plt.xlabel("x values")
#plt.ylabel("y values for several values of slope (y=alpha*x)")
#plt.title("test slope prior")



# A flat prior in slope amounts to a non-flat prior on the angle or tan(y/x), weighting our fit more heavily to steeper values of slope

# We can them determine a prior that compensates for this unequal spacing in angle
# Read through http://jakevdp.github.io/blog/2014/06/14/frequentism-and-bayesianism-4-bayesian-in-python/ for more details on obtaining this prior
# Note that they have reversed the notation for the slope and y-intercept from our convention: prior is written as (1+slope**2)**(-3./2.)

# remember Bayes's theorem: P(M|D)=P(D|M)*P(M)/P(D)
# P(M|D) is the posterior probability distribution 
# P(D|M) is the likelihood of the data given the model
# P(M) is the prior
# P(D) is the normalization

# To compute the posterior probability distribution:
# postprob=exp(-1*chisq/2)*prior
# lnpostprob=-1*chisq/2 + ln(prior)

# Compute the posterior probability for all possible models with two different priors
#lnpostprob_flat=np.zeros((gridsize1,gridsize2)) # setup an array to contain those values for the flat prior
#lnpostprob_comp=np.zeros((gridsize1,gridsize2)) # setup an array to contain those values for the compensating prior

#for i in xrange(gridsize1):  # loop over all possible values of alpha
#    for j in xrange (gridsize2): # loop over all possible values of beta
#        modelvals = alphaposs[i]*xvals+betaposs[j] # compute yfit for given model
#        resids = (yvals - modelvals) # compute residuals for given grid model
#        chisq = np.sum(resids**2 / errs**2) # compute chisq 
#        priorval_flat=1.  # uniform prior
#        priorval_comp=?   # prior to compensate for unequal spacing of angles
#        lnpostprob_flat[i,j] = (-1./2.)*chisq + np.log(priorval_flat)      
#        lnpostprob_comp[i,j] = ? + ?


# Now we have the full posterior probability distribution computed for 
# each possible model using both priors.

# What if we want to know the posterior probability distribution for the slope?
# We can find out by "marginalizing" over the intercept or integrating over the posterior distribution of the intercept.

# First, we take exp(lnpostprob)
#postprob_flat=np.exp(lnpostprob_flat)
#postprob_comp=np.exp(lnpostprob_comp)

#Second, we sum over the y-intercept parameter and normalize
#marginalizedpprob_flat_slope = np.sum(postprob_flat,axis=1) / np.sum(postprob_flat)
#marginalizedpprob_comp_slope = np.sum(postprob_comp,axis=1) / np.sum(postprob_comp)

# why do we sum over axis 1 in the numerator, but
# the whole array in the denominator?

# Plot the marginalized posterior probability distribution of slope values
#plt.figure(3) 
#plt.clf()
#plt.plot(alphaposs,marginalizedpprob_flat_slope,'g.',markersize=10)
#plt.plot(alphaposs,marginalizedpprob_comp_slope,'g.',markersize=10)
#plt.plot(?,?,'r.',markersize=10)
#plt.xlabel("alpha")
#plt.ylabel("marginalized posterior probability distribution of slope")

# zoom in on the region of significant probability
# and estimate the error from the graph
# Compare your error estimate with the error from paramfit1.py - are they similar?


# Now marginalize over the slope to see the posterior probability distribution in y-intercept
#marginalizedpprob_flat_yint = ?
#marginalizedpprob_comp_yint = ?

#plt.figure(4)
#plt.clf()
#plt.plot(betaposs,marginalizedpprob_flat_yint,'g',markersize='10.')
#plt.plot(betaposs,marginalizedpprob_comp_yint,'r',markersize='10.')
#plt.xlabel("beta")
#plt.ylabel("marginalized posterior probability distribution of y-intercept")



# How do the MLE values of the slope & y-intercept compare with marginalized posterior probability distributions? 

# How does the error on the slope & y-intercept compare with the value from the covariance matrix from paramfit1.py?

# What happens to the values and uncertainties of the slope and y-intercept if you change the number of points in your data (try N=100, N=10)?

# What happens if you change the grid spacing (try slope ranges from 1-10 in steps of 0.1, y-int ranges from 1-10 in steps of 1)? 


