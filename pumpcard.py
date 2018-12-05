# compare analytical solution to gekko solution
import numpy as np
import matplotlib.pyplot as plt
from gekko import*
from mpl_toolkits.mplot3d.axes3d import Axes3D

#analytical solution
def phi(x):
	phi = np.cos(x)
	return phi

def psi(x):
	psi = np.sin(2*x)
	return psi

def ua(x,t):
	#u = np.cos(x)*np.cos(3*t) + 1/6*np.sin(2*x)*np.sin(6*t)
	a = 18996.06 # ft/s speed of sound in steel
	c=a
	#u = 1/2*(np.cos(x-c*t)+np.cos(x+c*t)) - 1/(4*c)*(np.cos(2*(x+c*t)) -np.
	cos(2*(x-c*t)))
	u = np.cos(x)*np.cos(a*t) + 1/(2*a)*np.sin(2*x)*np.sin(2*a*t) return u

# define time
tf = .0005
npt = 100# number of points in time 
xf = 2*np.pi
npx = 100# number of points in space

time = np.linspace(0,tf,npt) # time array 
xpos = np.linspace(0,xf,npx) # space array

for i in range(npx): 
	usol = ua(xpos[i],time) 
	if i == 0:
		ustora = usol
	else: 
		ustora = np.vstack([ustora ,usol])

for i in range(npt):
	if i == 0:
		xstor = xpos
	else:
		xstor = np.vstack([xstor ,xpos])

for i in range(npx): 0:
	if i == 0:
		tstor = time
	else:
		tstor = np.vstack([tstor ,time])

xstor = xstor.T

#%%
# create gekko model
m = GEKKO()