import numpy as np


def __F_sta__(f, dts):
  ''' 
    Stability
  '''
  x = 2.0*np.pi*f*dts * 1j
  w = 2*x*(x+6.0)
  w /= x**2 + 6.0*x + 12.0
  return np.abs(w*w.conjugate())

def __F_wm__(f, dt):
  ''' 
    Windowed mean
  '''
  x = 2.0*np.pi*f*dt * 1j
  w = 2.0*(x + 6.0)
  w /= x**2 + 6.0*x + 12.0
  return np.abs(w*w.conjugate())

def __F_wv__(f, dt):
  ''' 
    Windowed variance (RPE)
  '''
  x = 2.0*np.pi*f*dt * 1j
  w = x*(x+np.sqrt(12.0))
  w /= x**2 + 6*x + 12.0
  return np.abs(w*w.conjugate())


def __F_wms__(f, dt, dts):
  ''' 
    Windowed mean stability (PDE)
  '''
  w = __F_sta__(f, dts)*__F_wm__(f, dt)
  
  return w


def get_pde(freq, psd, delta_t, delta_ts):
    F_WMS = __F_wms__(freq, delta_t, delta_ts)
    pde2 = 11.8*np.trapz(psd*F_WMS, x = freq)
    return np.sqrt(pde2)

def get_rpe(freq, psd, delta_t):
    F_WV  = __F_wv__(freq, delta_t)
    rpe2 = 11.8*np.trapz(psd*F_WV, x = freq)
    return np.sqrt(rpe2)