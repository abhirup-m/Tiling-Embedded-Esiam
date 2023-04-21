#-----------------------------------------------------
# Implements Kramer-Kronig Transform
#----------------------------------------------------
import numpy as np
from scipy import special
import matplotlib.pyplot as plt
import cmath
import math
from scipy import signal
from scipy import integrate

one=complex(1.0,0.0)
ii=complex(0.0,1.0)
zero=complex(0.0,0.0)


#------------------------------------------------------
#CONVOLUTION FUNCTION
#------------------------------------------------------
#          infty
#y3(t)=int^      y1(tau)*y2(t-tau)
#         -infty

def convolve(y1,y2):
  n=len(y1)
  y1p=np.pad(y1,(n,n),'constant',constant_values=(0,0))
  y2p=np.pad(y2,(n,n),'constant',constant_values=(0,0))
  y3p=signal.convolve(y1p,y2p,mode='same')
  y3=y3p[n:2*n]*step_size
  return y3
#------------------------------------------------------
  

#------------------------------------------------------
#Kramers-Kronig transform
#------------------------------------------------------
def kkt(w,rho,ss):
  wl=w+np.ones(len(w))*ss/2.0
  owl=1.0/wl
  return(convolve(rho,owl))
#------------------------------------------------------


#-----------------------------------------------------
# Everything below is for testing. We can compute the
# Hilbert transform of w+i\eta over a Gaussian density 
# of states, which gives the imaginary and real parts of
# the Green's function. Now we can use the imaginary part,
# compute the real part through KKT, and compare against
# the exact result. This is what is done below.
#------------------------------------------------------



#------------------------------------------------------
def gauss(z,t):
  G=-ii*np.sqrt(np.pi)*special.wofz(z/t)/t
  return G
#------------------------------------------------------


t=1.0
eta=1.0e-03

#Frequency Grid - Uniform from -10 to 10 with step_size=0.001
wm=10.0
step_size=1.0e-03
w=np.arange(start=-wm,stop=wm+step_size,step=step_size,dtype=float)
nw=len(w)
ss=step_size

gamma=w+ii*eta
G=gauss(gamma,t)

rho=-G.imag/np.pi

ReG=kkt(w,rho,ss)
ReG_exact=G.real

import matplotlib.pyplot as plt

plt.plot(w,ReG)
plt.plot(w,ReG_exact)
plt.show()
