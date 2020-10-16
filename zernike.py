import numpy as np
from math import factorial as fac
from scipy.special import eval_jacobi as jacobi

class Zernike:
    """
    Generates Zernike polynomials 
    
    Parameters
    ----------
    N : integer
        Number of polynomials to generate in a sequence following the defined 'ordering'
    rho : array like
        the radial coordinate normalised to the interval [0, 1]
    phi : array like
        Azimuthal coordinate in radians. Has same shape as phi
    ordering : string
        Can be either ANSI ordering (ordering='ansi', this is the default), or Noll ordering
        (ordering='noll')
    normalize : bool
        Set to True generates ortho-normal polynomials. Set to False generates orthogonal polynomials
        as described in Laksminarayan & Fleck, Journal of Modern Optics (2011). The radial polynomial 
        is estimated using the Jacobi polynomial expression as in their Equation in Equation 14.
        
    Returns
    -------
    out : an instance of Zernike.
    
    
    """
    def __init__(self, N, rho, phi, ordering='ansi', normalize = False):
        
        
        assert ordering in ('ansi', 'noll'), 'Unrecognised ordering scheme.'
        assert N >= 0, 'N shall be a positive integer'
        
        self.ordering = ordering
        self.N = N
        self.m, self.n = self.__j2mn__()
        
        if normalize:
            self.norm = [np.sqrt(n+1) if m == 0 else np.sqrt(2.0*(n+1)) for m, n in zip(self.m, self.n)] 
        else:
            self.norm = np.ones(self.N, dtype = np.float)
            
        Z = {}
        mask = rho > 1.0
        for n in range(max(self.n)+1):
            Z[n] = {}
            for m in range(-n, 1, 2):
                Z[n][m] = np.ma.masked_array(data=self.__ZradJacobi__(m, n, rho), 
                                             mask = mask, 
                                             fill_value=0.0)
                Z[n][-m] = Z[n][m].view()
                
        self.Zrad = [Z[n][m].view() for m, n in zip(self.m, self.n)]
        
        Z = {0: np.ones_like(phi)}
        for m in range(1, self.m.max()+1):
            Z[m]  = np.cos(m*phi)
            Z[-m] = np.sin(m*phi)
        self.Zphi = [Z[m].view() for m in self.m]
        
        
    def __call__(self, j = None):
        """
        Parameters
        ----------
        j : integer
            Polynomial to return. If set to None, returns all polynomial requested at
            instantiation
        
        Returns
        -------
        out : masked array or list of masked arrays
          if j is set to None, the output is a list of polynomials as masked arrays.
          When j is set to an integer, returns the j-th polynomial as a masked array.
        """
        if j is None:
            return [self.norm[k]*self.Zrad[k]*self.Zphi[k] for k in range(self.N)]
        else:
            return self.norm[j]*self.Zrad[j]*self.Zphi[j]
        
    def __j2mn__(self):
        '''
        Convert index j into azimithal number, m, and radial number, n.
        '''
        n = np.empty(self.N, dtype=int)
        j = np.arange(self.N, dtype=int)
        if self.ordering == 'ansi':
            n = np.ceil((-3.0+np.sqrt(9.0+8.0*j))/2.0).astype(int)
            m = 2*j - n*(n+2)
        elif self.ordering == 'noll':
            index = j + 1
            n = ((0.5 * (np.sqrt(8 * index - 7) - 3)) + 1).astype(int)
            cn = n * (n + 1) / 2 + 1
            m = np.empty(self.N, dtype=int)
            idx = n % 2 == 0
            m[idx] = (index[idx] - cn[idx] + 1) // 2 * 2
            m[~idx] = (index[~idx] - cn[~idx]) // 2 * 2 + 1
            m = (-1)**(index%2)*m 
        else:
            raise NameError("Ordering not supported.")

        return m, n

    def __mn2j__(self):
        '''
        Convert radial and azimuthal numbers, respectively n and m, into index j 
        '''
        if self.ordering == 'ansi':
            return (self.n*(self.n+2) + self.m)//2
        else:
            raise NameError("Ordering not supported.")
        
    def __ZradJacobi__(self, m, n, rho):
        '''
            Computes the radial Zernike polinomial

            Parameters
            ----------
            m : integer
              azimuthal number
            n : integer 
              radian number
            rho : array like 
              Pupil semi-diameter normalised radial coordinates

            Returns
            -------
            R_mn : array like 
              the radial Zernike polinomial with shape identical to rho

        '''

        m = np.abs(m)

        if (n < 0):
            raise ValueError('Invalid parameter: n={:d} should be > 0'.format(n))
        if (m > n):
            raise ValueError('Invalid parameter: n={:d} should be larger than m={:d}'.format(n, m))
        if (n-m)%2: 
            raise ValueError('Invalid parameter: n-m={:d} should be a positive even number.'.format(n-m))

        jpoly = jacobi( (n-m)//2, m, 0.0, (1.0-2.0*rho**2))

        return (-1)**((n-m)//2) * rho**m * jpoly

    def __ZradFactorial__(self, m, n, rho):
        '''
            CURRENTLY NOT USED
            Computes the radial Zernike polinomial

            Parameters
            ----------
            m : integer
              azimuthal number
            n : integer 
              radian number
            rho : array like 
              Pupil semi-diameter normalised radial coordinates

            Returns
            -------
            R_mn : array like 
              the radial Zernike polinomial with shape identical to rho

        '''
        m = np.abs(m)

        if (n < 0):
            raise ValueError('Invalid parameter: n={:d} should be > 0'.format(n))
        if (m > n):
            raise ValueError('Invalid parameter: n={:d} should be larger than m={:d}'.format(n, m))
        if (n-m)%2: 
            raise ValueError('Invalid parameter: n-m={:d} should be a positive even number.'.format(n-m))

        pre_fac = lambda k: (-1.0)**k * fac(n-k) / ( fac(k) * fac( (n+m)//2 - k ) * fac( (n-m)//2 - k ) )

        return sum(pre_fac(k) * rho**(n-2*k) for k in range((n-m)//2+1))



  
