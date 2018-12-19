import numpy as np
from scipy.special import j1

def airy(x, norm = 'area'):
  """ Airy 2D function, normalised to area 1 
    
    Parameters
    __________
      x : 			1D array
				positions in units of F*wl
    Returns
    -------
      spectrum:			2D array
				Airy function A, such that A.sum() = 1, w
				when sum is over an infinitely extended array (i.e. x.max() == inf)
  """
    
  xx, yy = np.meshgrid(x, x)

  r = np.pi*np.sqrt(xx**2 + yy**2)+1.0e-10

  airy = (2.0*j1(r)/r)**2
  
  if norm == 'area':
      normalization = (0.25* np.pi * (x[1]-x[0])**2)
  elif norm == 'peak':
      normalization = 1.0
  else:
      print 'normalization method not supported'
  
  return normalization * airy
