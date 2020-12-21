import numpy as np
from .zernike import Zernike

class Elliptical(object):
    """
    Generates polynomials that are ortonormal on an elliptical pupil implementing the Gram-Schmidt 
    orthogonalization of Zernike polynomials as described in Section 3 of https://doi.org/10.1117/3.927341.
    
    The orthgonalization is implemented as a Cholesky decomposition of the symmetroc matrix of inner products
    Zernike polynomials as described in Eq. 3-27 of the above reference. 
    
    The ordering of Zernike polynomials used is "noll", and the Zernike polynomials are normalised to unit RMS
    on the circle circumscribing the elliptical aperture. 
    
    It is assumed that elliptical pupil has major and minor axes aligned to the cardinal axes, and it encloses the
    region defined by x**2 + y**2/c**2 <= 1.
    
    Parameters
    ----------
    N : integer
        Number of polynomials to generate in a sequence. I should be chosen to close the Zernike order.
        I.e. N = 1, 3, 6, 10, ... If this is not the case, N is re-calculated to force order closure.
    rho : array like
        the radial coordinate normalised to the interval [0, 1]. Values of rho > 1 are masked.
    phi : array like
        Azimuthal coordinate in radians. Has same shape as phi
    
    c   : the ratio of the semi-axis aligned to the y-axis to the semi-axis aligned to x-axis
    
    Returns
    -------
    out : an instance of Elliptical.
    
    Notes
    -----
    This is implemented as a class, because it will need to implement the methods to rotate the elliptical coefficients
    into Zernike coefficients. This functionality will be added in due course.
    
    Example
    -------
    
    Npt = 1024
    x = np.linspace(-1.1, 1.1, Npt)
    xx, yy = np.meshgrid(x, x)
    c = 0.85
    rho = np.sqrt(xx**2 + yy**2)
    phi = np.arctan2(yy, xx)

    elliptical = Elliptical(10, rho, phi, c)
    
    plt.imshow(elliptical(4))
    
    Y = elliptical()
    plt.imshow(Y[4])
    """
    
    def __init__(self, N, rho, phi, c):
        
        x = rho*np.cos(phi)
        y = rho*np.sin(phi)
        
        mask = x**2 + y**2/c**2 > 1
        
        _rho_ = np.ma.masked_array(data=rho, mask = mask)
        phi = np.arctan2(y, x)
        
        # force closure of order
        k =  np.sqrt(1+8*N) - 1
        n = np.ceil(k/2).astype(np.int)
        Npoly = n*(n+1)//2
        
        self.zer = Zernike(Npoly, _rho_, phi, ordering='noll', normalize=True)()

        self.Czz = np.empty( (len(self.zer), len(self.zer)), dtype = self.zer[0].dtype)

        for i in range(self.Czz.shape[1]):
            for j in range(i, self.Czz.shape[0]):
                self.Czz[j, i] = self.Czz[i, j] = np.ma.mean(self.zer[i]*self.zer[j])
        
        self.Qt = np.linalg.cholesky(self.Czz)
        self.M = np.linalg.inv(self.Qt)
       
        self.Y = []
        for i in range(self.M.shape[0]):
            ser = [self.M[i,j]*self.zer[j] for j in range(i+1)]
            self.Y.append(np.ma.masked_array(ser).sum(axis=0))
     
    def __call__(self, *index):
        if index:
            return self.Y[index[0]]
        else:
            return self.Y
