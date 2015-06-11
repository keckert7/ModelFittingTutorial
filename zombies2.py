# -*- coding: utf-8 -*-
"""
Created on Wed May 20 21:00:00 2015

@author: katie
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import scipy.stats as stat
import numpy.random as npr
import pylab
pylab.ion()

pi=np.pi

# these are solutions to the Zombie Bayesian analysis portion of the 
# parameter fitting tutorial
# taking data on the % of humans after a wave of zombiism
# power went out 5 days ago so you are trying to determine when zombies will take over

# 1) read in data and plot
datain=np.loadtxt("percentzombie.txt") # read in text file with data
time=datain[:,0] # load in time
perzombie=datain[:,1] # load in % zombies
perhuman=100-perzombie # calculate % human = (1- % zombie)
xval=np.arange(100) # setup dummy array of xvalues ranging from 0-100

err=0.5 # assume some error on your measurement of time = half a day

# check lengths of arrays
print np.size(time)
print np.size(perhuman)


plt.figure(1) 
plt.clf()
plt.plot(perhuman,time,'b*',markersize=10)
plt.xlabel('% human')
plt.ylabel('time')
plt.xlim(0,100)
plt.ylim(-15,15)


# tryout Bayesian analysis
testslope=np.arange(200)/100.-1
testyint=np.arange(200)/2.-5

print("min/max slope are %f/%f" % (np.min(testslope),np.max(testslope)))
print("min/max y-intercept are %f/%f" % (np.min(testyint),np.max(testyint)))

# remember a flat grid space requires a prior = (1+beta^2)^(-3/2)

lnlikeout=np.zeros((np.size(testslope),np.size(testyint)))
for i in range(np.size(testslope)):
    for j in range(np.size(testyint)):
        modeltime=perhuman*testslope[i]+testyint[j]
        residuals=time-modeltime
        chisq=np.sum((residuals)**2/err**2)
        prior=(1.+testyint[j]**2)**(-3./2.)
        lnlikeout[i,j]=-1.*chisq/2. + np.log(prior)
        
# what is posterior likelihood distribution for time of 0% humans (or y-intercept)?

likeout=np.exp(lnlikeout)
#marginalize over slope values to see y-intercept
postdist_yint=np.sum(likeout,axis=0)/np.sum(likeout)

plt.figure(2)
plt.clf()
plt.plot(testyint,postdist_yint,'r*',markersize=10)
plt.ylim(0,1)
plt.xlim(np.min(testyint),np.max(testyint))
plt.xlabel("time to 0% humans")
plt.ylabel("likelihood")

# zoom in
plt.xlim(-5,5)

# since I am not a Zombie yet, I can place a prior that 0% humans has not occured yet therefore my grid space should start today (0)

testslope2=np.arange(200)/100.-1
testyint2=np.arange(200)/2.

# remember a flat grid space requires a prior = (1+beta^2)^(-3/2)

lnlikeout2=np.zeros((np.size(testslope2),np.size(testyint2)))
for i in range(np.size(testslope2)):
    for j in range(np.size(testyint2)):
        modeltime=perhuman*testslope2[i]+testyint2[j]
        residuals=time-modeltime
        chisq=np.sum((residuals)**2/err**2)
        prior=(1.+testyint2[j]**2)**(-3./2.)
        lnlikeout2[i,j]=-1.*chisq/2. + np.log(prior)

likeout2=np.exp(lnlikeout2)
#marginalize over slope values to see y-intercept
postdist_yint2=np.sum(likeout2,axis=0)/np.sum(likeout2)

plt.plot(testyint2,postdist_yint2,'g.',markersize=10)
plt.ylim(0,1)
