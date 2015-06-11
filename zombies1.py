"""
part I, Second activity in Parameter Fitting Tutorial
Solutions Written by Katie Eckert
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

# these are solutions to the Zombie MLE portion of the 
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

# plot time as function of % human
plt.figure(1) 
plt.clf()
plt.plot(perhuman,time,'b*',markersize=10)
# make labels
plt.xlabel('% human')
plt.ylabel('time')
# set limits to show wider ranges
plt.xlim(0,100)
plt.ylim(-15,15)

# 2: linear MLE solution using python solver np.polyfit
# third parameter is order of fit, 1 for linear
p1=np.polyfit(perhuman,time,1) # returns highest power first 

# print out MLE slope & y-intercept
print("slope = %f" %p1[0])
print("y-intercept = %f" %p1[1])
print("uhoh you are already a zombie!")
# evaluate best fit lines
yval1=xval*p1[0]+p1[1]

plt.plot(xval,yval1,'g') # overplot MLE solution in green
# you can also use np.polyval to spit out the values so you don't have to write out the full function (helpful for higher order functions)
plt.plot(xval,np.polyval(p1,xval),'g+',markersize=10)

# 3. Above we have performed the fit so as to minimize residuals in the 
# ydirection (time in this case), what if we wanted to minimize residuals 
# in the x direction, i.e.,  perform the fit backwards

p1b=np.polyfit(time,perhuman,1)
# in this case slope and y-intercept not direct, need to do some algebra
print("slope = %f" %(1./p1b[0]))
print("y-intercept = %f" %(-1.0*p1b[1]/p1b[0]))
print("when evaluated this way there's still time...")
yval2=xval*(1./p1b[0])-(p1b[1]/p1b[0])
plt.plot(xval,yval2,'r')

# as you can see which way you fit makes a big difference - either there are no humans, or there's one more day until there are no more humans

# 4. Plot the residuals from the fit from part 2 & compute the reduced chi^2 
#value of the fit 
# we have assumed the error is ~ half a day)

plt.figure(2)
plt.clf()
plt.plot(perhuman,time-np.polyval(p1,perhuman),'g*',markersize=15)
plt.xlabel("percent human")
plt.ylabel("residuals in time")
redchisq1=np.sum((time-np.polyval(p1,perhuman))**2/err**2)*(1./(np.size(time)-np.size(p1)-1))
print("Reduced Chi^2 value of linear fit = %f" % (redchisq1))
# how well does our fit describe the data?

# Extra Problems (go on to Bayesian tutorial if running out of time)

# 5. What happens if we use higher order fits?

p2=np.polyfit(perhuman,time,2) #2nd order fit
p3=np.polyfit(perhuman,time,3) #3rd order fit
p4=np.polyfit(perhuman,time,4) #4th order fit

print("2nd order fit: y-intercept = %f" % p2[2])
print("3nd order fit: y-intercept = %f" % p3[3])
print("4nd order fit: y-intercept = %f" % p4[4])
# overplot the higher order fits on figure 1
plt.figure(1)
plt.plot(xval,np.polyval(p2,xval),'m')
plt.plot(xval,np.polyval(p3,xval),'c')
plt.plot(xval,np.polyval(p4,xval),'y')
print("oh wow the time to 0% human based on different order fits can vary a lot")

# over plot residuals on figure 2
plt.figure(2)
plt.plot(perhuman,time-np.polyval(p2,perhuman),'m+',markersize=10)
plt.plot(perhuman,time-np.polyval(p3,perhuman),'c.',markersize=20)
plt.plot(perhuman,time-np.polyval(p4,perhuman),'y^',markersize=10)
#
# but are these fits better?

# 6. compute reduced chi^2 for higher order fits
# (number degrees of freeom = Npoints-nparams-1)
redchisq2=np.sum((time-np.polyval(p2,perhuman))**2/err**2)*(1./(np.size(time)-np.size(p2)-1))
redchisq3=np.sum((time-np.polyval(p3,perhuman))**2/err**2)*(1./(np.size(time)-np.size(p3)-1))
redchisq4=np.sum((time-np.polyval(p4,perhuman))**2/err**2)*(1./(np.size(time)-np.size(p4)-1))

print("reduced Chi^2 values")
print("1st order = %f" %redchisq1)
print("2nd order = %f" %redchisq2)
print("3rd order = %f" %redchisq3)
print("4th order = %f" %redchisq4)

# Even though the reduced Chi^2 value for the linear fit is > 1, the higher order fits are <1, indicating that they are overfitting the data. For N=10 data points, we must make sure not to increase the order of our fit too high.
