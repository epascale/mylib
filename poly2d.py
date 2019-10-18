import numpy as np
from numpy.polynomial.polynomial import polyvander2d

def polyfit2d(x, y, z, deg=2):
  '''
  polyfit2d, fit a 2d polinomial of order deg to a series of data point. 
  for instance, if deg = 2, models z as
    z = par[0] + par[1]*x + par[2]*x**2 + par[3]*y + par[4]*x*y + par[5]*y**2
  
  '''
  #if len(z.shape) != 2 or len(x.shape) != 2 or len(y.shape) != 2 or \
  #   len(z.shape) != 1 or len(x.shape) != 1 or len(y.shape) != 1:
  #  raise ValueError("Error in array dimensions")

  z_ = z.flatten()
  
  A = polyvander2d(x.flatten(), y.flatten(), [deg, deg])
  order = np.array([i+j for i in range(deg+1) for j in range(deg+1)])
  cols = np.where(order <= deg)[0]
  A = np.take(A, cols, axis=1)
  
  par = np.linalg.lstsq(A, z_, rcond=None)
  
  return(par[0])

def poly2d(coeff):
  '''
  poly2d,  a 2d polinomial of order deg. 
  for instance, if deg = 2, models z as
    z = par[0] + par[1]*x + par[2]*x**2 + par[3]*y + par[4]*x*y + par[5]*y**2
    
    coeff is the output of polyfit2d
  '''
  
  def p(x, y, coeff=coeff):
    shape = x.shape
    deg = np.int(0.5*(np.sqrt(1+ 8*len(coeff))-1))-1
    
    A = polyvander2d(x.flatten(), y.flatten(), [deg, deg])
    
    order = np.array([i+j for i in range(deg+1) for j in range(deg+1)])
    cols = np.where(order <= deg)[0]
    A = np.take(A, cols, axis=1)
    
    return np.dot(A, coeff).reshape(shape)

  return p
