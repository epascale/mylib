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


def get_pde(freq, psd, delta_t=90.0, delta_ts=10.0*3600.0):
    # For Ariel: delta_t = 90s, 300s
    #            delta_ts = 10*3600
    F_WMS = __F_wms__(freq, delta_t, delta_ts)
    pde2 = 11.8*np.trapz(psd*F_WMS, x = freq)
    return np.sqrt(pde2)

def get_rpe(freq, psd, delta_t=0.1):
    F_WV  = __F_wv__(freq, delta_t)
    rpe2 = 11.8*np.trapz(psd*F_WV, x = freq)
    return np.sqrt(rpe2)

def get_mpe_rpe(freq, psd, dt1=90.0, dt2=0.1):
    # dt1 = 90.0, 300.0
    # dt2 = 0.1 
    F_WV  = __F_wv__(freq, dt1) * __F_wm__(freq, dt2)
    rpe2 = 11.8*np.trapz(psd*F_WV, x = freq)
    return np.sqrt(rpe2)

def get_pde_filter(f, dt, dts):
  return __F_wms__(f, dt, dts)

def get_rpe_filter(f, dt):
  return __F_wv__(f, dt)

def get_mpe_filter(f, dt):
  return __F_wm__(f, dt)

def get_rpe_mpe_filter(f, dt1, dt2):
  return __F_wv__(f, dt1) * __F_wm__(f, dt2)
