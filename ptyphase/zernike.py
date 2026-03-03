import numpy as np 
from skimage.restoration import unwrap_phase
from scipy.optimize import curve_fit

class SquarePolynomials: 
    """
    A class containing a set of orthonormal square polynomials 
    in Cartesian coordinates from Mahajan and Dai 
    Orthonormal polynomials in wavefront analysis: analytical solution
    J. Opt. Soc. Am. A / Vol. 24, No. 9 / September 2007
    """
    @classmethod
    def project_wavefront(cls, wavefront, coords = None):

        if coords is not None:
            original_shape = wavefront.shape
            wavefront = wavefront[coords[0]:coords[1], coords[2]:coords[3]]
        
        shape_y, shape_x = wavefront.shape
    
        wavefront = unwrap_phase(wavefront)

        # Create coordinate grids
        side_x = np.linspace(-1/np.sqrt(2), 1/np.sqrt(2), shape_x)
        side_y = np.linspace(-1/np.sqrt(2), 1/np.sqrt(2), shape_y)

        X, Y = np.meshgrid(side_x, side_y)
        xdata = [X, Y]

        coeffs = cls.extract_square_coefficients_vectorized(wavefront)
        
        all_results = cls.evaluate_all(xdata, coeffs)
        wavefront = sum(all_results.values())
        
        if coords is not None:
            new_wavefront = np.zeros(original_shape, dtype = complex)
            new_wavefront[coords[0]:coords[1], coords[2]:coords[3]] = wavefront
            
        return new_wavefront
    
    @classmethod
    def get_zernike_wavefront(cls, coefficients, pupil_shape):
    
        shape_y, shape_x = pupil_shape
        
        # Create coordinate grids
        side_x = np.linspace(-1/np.sqrt(2), 1/np.sqrt(2), shape_x)
        side_y = np.linspace(-1/np.sqrt(2), 1/np.sqrt(2), shape_y)
    
        X, Y = np.meshgrid(side_x, side_y)
        xdata = [X, Y]
    
        all_results = cls.evaluate_all(xdata, coefficients)
        new_wavefront = sum(all_results.values())
    
        return new_wavefront

    @classmethod
    def extract_square_coefficients_vectorized(cls, phase):
        """
        More efficient implementation using least squares directly.
        """
        instance = cls()
        # Create coordinate grids
        side_x = np.linspace(-1/np.sqrt(2), 1/np.sqrt(2), phase.shape[1])
        side_y = np.linspace(-1/np.sqrt(2), 1/np.sqrt(2), phase.shape[0])
    
        X1, X2 = np.meshgrid(side_x, side_y)
        
        # Reshape coordinates - note: different format for direct method calls
        xdata = [X1, X2]  # This matches the expected format for the polynomial methods
        
        # Reshape phase data
        p_flat = phase.flatten()
        
        # Build design matrix
        function_names = instance.get_function_list()
        n_terms = len(function_names)
        n_pixels = len(p_flat)
        
        A_matrix = np.zeros((n_pixels, n_terms))
        
        print("Building design matrix...")
        for i in range(n_terms):
            func_name = function_names[i]
            # Evaluate polynomial with unit amplitude
            poly_vals = instance.evaluate(func_name, xdata, A=1.0)
            A_matrix[:, i] = poly_vals.flatten()
        print("Done")
        # Solve using least squares
        coefficients, residuals, rank, s = np.linalg.lstsq(A_matrix, p_flat, rcond=None)
        
        return coefficients.tolist()
    
    @classmethod
    def evaluate(self, function_name, xdata, A):
            """
            Evaluate a specific polynomial function by name.
            
            Parameters:
            -----------
            function_name : str
                Name of the function to evaluate ('S1', 'S2', 'S3', 'S4', 'S5', ...)
            xdata : tuple or list
                Input data as (x, y) coordinates
            A : float
                Amplitude parameter
            
            Returns:
            --------
            numpy.ndarray
                Result of the polynomial evaluation
            """
            
            if hasattr(self, function_name):
                return getattr(self, function_name)(xdata, A)
            else:
                raise ValueError(f"Function {function_name} not found")


    @classmethod
    def evaluate_all(self, xdata, A_values):
        """
        Evaluate all polynomial functions with given amplitude values.
        
        Parameters:
        -----------
        xdata : tuple or list
            Input data as (x, y) coordinates
        A_values : list or array
            Amplitude values for each function [A1, A2, A3, A4, A5]
        
        Returns:
        --------
        dict
            Dictionary with function names as keys and results as values
        """
        if len(A_values) != 29:
            raise ValueError("A_values must contain exactly 29 values")
        
        results = {}
        function_names = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S12', 'S13', 'S14', 'S15', 
                          'S16', 'S17', 'S18', 'S19', 'S20', 'S21', 'S22', 'S23', 'S24', 'S25', 'S26', 'S27', 'S28', 'S29']
        
        for i, func_name in enumerate(function_names):
            results[func_name] = getattr(self, func_name)(xdata, A_values[i])
        
        return results


    def get_function_list(self):
        """Return a list of available polynomial functions."""
        return ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S12', 'S13', 'S14', 'S15', 
                'S16', 'S17', 'S18', 'S19', 'S20', 'S21', 'S22', 'S23', 'S24', 'S25', 'S26', 'S27', 'S28', 'S29']
    
    @staticmethod
    def S1(xdata, A):
        x, y = xdata[0], xdata[1]
        return A * np.ones_like(x)

    @staticmethod
    def S2(xdata, A):
        x, y = xdata[0], xdata[1]
        return A * np.sqrt(6) * x
    
    @staticmethod
    def S3(xdata, A):
        x, y = xdata[0], xdata[1]
        return A * np.sqrt(6) * y

    @staticmethod
    def S4(xdata, A):
        x, y = xdata[0], xdata[1]
        rho2 = x**2 + y**2
        return A * np.sqrt(5/2) * (3*rho2 - 1)

    @staticmethod
    def S5(xdata, A):
        x, y = xdata[0], xdata[1]
        return A * 6 * x * y

    @staticmethod
    def S6(xdata, A):
        x, y = xdata[0], xdata[1]
        return A * 3 * np.sqrt(5/2) * (x**2 - y**2)

    @staticmethod
    def S7(xdata, A):
        x, y = xdata[0], xdata[1]
        rho2 = x**2 + y**2
        return A * np.sqrt(21/31) * (15 * rho2 -7) * y

    @staticmethod
    def S8(xdata, A):
        x, y = xdata[0], xdata[1]
        rho2 = x**2 + y**2
        return A * np.sqrt(21/31) * (15 * rho2 -7) * x

    @staticmethod
    def S9(xdata, A):
        x, y = xdata[0], xdata[1]
        return A * np.sqrt(5/31) * (27 * x**2 - 35 * y**2 + 6) *y 


    @staticmethod
    def S10(xdata, A):
        x, y = xdata[0], xdata[1]
        return A * np.sqrt(5/31) * (35 * x**2 - 27 * y**2 - 6) * x

    @staticmethod
    def S11(xdata, A):
        x, y = xdata[0], xdata[1]
        rho2 = x**2 + y**2
        return A * 1/(2*np.sqrt(67)) * (315 * rho2**2 - 240*rho2 + 31)

    @staticmethod
    def S12(xdata, A):
        x, y = xdata[0], xdata[1]
        rho2 = x**2 + y**2
        return A * 15/(2*np.sqrt(2)) * (x**2 - y**2) * (7*rho2 -3)

    @staticmethod
    def S13(xdata, A):
        x, y = xdata[0], xdata[1]
        rho2 = x**2 + y**2
        return A * np.sqrt(42) * (5 * rho2 -3) * x * y

    @staticmethod
    def S14(xdata, A):
        x, y = xdata[0], xdata[1]
        rho2 = x**2 + y**2
        return A * 3 / (4 * np.sqrt(134)) * (10*(49*x**4 - 36*x**2 * y**2 + 49 * y**4) - 150*rho2 + 11)

    @staticmethod
    def S15(xdata, A):
        x, y = xdata[0], xdata[1]
        return A * 5 * np.sqrt(42) * (x**2 - y**2) * x * y 

    @staticmethod
    def S16(xdata, A):
        x, y = xdata[0], xdata[1]
        rho2 = x**2 + y**2
        return A * np.sqrt(55/1966) * (315*rho2**2 - 280*x**2 - 324*y**2 + 57) * x 

    @staticmethod
    def S17(xdata, A):
        x, y = xdata[0], xdata[1]
        rho2 = x**2 + y**2
        return A * np.sqrt(55/1966) * (315*rho2**2 - 324*x**2 - 280*y**2 + 57) * y 

    @staticmethod
    def S18(xdata, A):
        x, y = xdata[0], xdata[1]
        rho2 = x**2 + y**2
        return A * 0.5 * np.sqrt(3/844397) * (105 * (1023*x**4 + 80*x**2 * y**2 - 943*y**4) - 61075 * x**2 + 39915*y**2 + 4692) * x

    @staticmethod
    def S19(xdata, A):
        x, y = xdata[0], xdata[1]
        rho2 = x**2 + y**2
        return A * 0.5 * np.sqrt(3/844397) * (105 * (943*x**4 - 80*x**2 * y**2 - 1023*y**4) - 39915 * x**2 + 61075*y**2 + 4692) * y

    @staticmethod
    def S20(xdata, A):
        x, y = xdata[0], xdata[1]
        rho2 = x**2 + y**2
        return A * 0.25 * np.sqrt(7/859) * (6 * (693*x**4 - 500*x**2 * y**2 + 525*y**4) - 1810 * x**2 - 450*y**2 + 165) * x

    @staticmethod
    def S21(xdata, A):
        x, y = xdata[0], xdata[1]
        rho2 = x**2 + y**2
        return A * 0.25 * np.sqrt(7/859) * (6 * (525*x**4 - 500*x**2 * y**2 + 693*y**4) - 450 * x**2 - 1810*y**2 + 165) * y

    @staticmethod
    def S22(xdata, A):
        x, y = xdata[0], xdata[1]
        rho2 = x**2 + y**2
        return A * 0.25 * np.sqrt(65/849) * (1155 * rho2**3 - 15 * (91 * x**4 + 198 * x**2 * y**2 + 91*y**4) + 453*rho2 - 31)

    @staticmethod
    def S23(xdata, A):
        x, y = xdata[0], xdata[1]
        rho2 = x**2 + y**2
        return A * np.sqrt(33/3923) * (1575 * rho2**2 - 1820*rho2 + 471) * x * y

    @staticmethod
    def S24(xdata, A):
        x, y = xdata[0], xdata[1]
        rho2 = x**2 + y**2
        return A * 24/4 * np.sqrt(65/1349) * (165 * rho2**2 - 140*rho2 + 27) * (x**2 - y**2)

    @staticmethod
    def S25(xdata, A):
        x, y = xdata[0], xdata[1]
        rho2 = x**2 + y**2
        return A * 7 * np.sqrt(33/2) * (9 * rho2 - 5) * x * y * (x**2 - y**2)

    @staticmethod
    def S26(xdata, A):
        x, y = xdata[0], xdata[1]
        rho2 = x**2 + y**2
        return A * (1 / (8 * np.sqrt(849)) * (42 * (1573 * x**6 - 375 * x**4 * y**2 - 375 * x**2 * y**4 + 1573 * y**6) - 60*(707*x**4 - 225 * x**2 * y**2 + 707 * y**4) + 6045*rho2 - 245))

    @staticmethod
    def S27(xdata, A):
        x, y = xdata[0], xdata[1]
        rho2 = x**2 + y**2
        return A * 1 / (2 * np.sqrt(7846)) * (14 * (2673 * x**4 - 2500 * x**2 * y**2 + 2673 * y**4) - 10290*rho2 + 1305) * x * y 

    @staticmethod
    def S28(xdata, A):
        x, y = xdata[0], xdata[1]
        rho2 = x**2 + y**2
        return A * 21 / (8 * np.sqrt(1349)) * (3146 * x**6 - 2250 * x**4 * y**2 + 2250 *x**2 * y**4 - 3146*y**6 - 1770 *(x**4 - y**4) + 245*(x**2-y**2))

    @staticmethod
    def S29(xdata, A):
        x, y = xdata[0], xdata[1]
        rho2 = x**2 + y**2
        return A * (-13.79189793 + 150.92209099*x**2 + 117.01812058*y**2 - 352.15154565*x**4 - 657.27245247 *x**2 * y**2 - 291.12439892*y**4 + 222.62454035*x**6 + 667.87362106 * x**4 * y**2 +667.87362106 *x**2 * y**4 + 222.62454035* y**6)*y


class RectangularPolynomials:
    """
    A class containing a set of orthonormal rectangular polynomials 
    in Cartesian coordinates from Mahajan and Dai 
    Orthonormal polynomials in wavefront analysis: analytical solution
    J. Opt. Soc. Am. A / Vol. 24, No. 9 / September 2007
    """
    def __init__(self, a=1/np.sqrt(2)):
        """
        Initialize the polynomial functions with parameter 'a',
        which is a parameter of rectangularity (see Fig 4 in the paper)
        half-widths of the rectangle along the x and y axes are a and sqrt(1 − a^2)
        a --> 0 or a --> 1 corresponds to a slit 
        a = 1/sqrt(2) corresponds to the square pupil
        Parameters:
        -----------
        a : float
            Parameter used in the polynomial calculations (default: 1/np.sqrt(2))
        """
        self.a = a

    def R1(self, xdata, A):
        x, y = xdata[0], xdata[1]
        return A*np.ones(xdata[0].shape)

    def R2(self, xdata, A):
        x, y = xdata[0], xdata[1]
        return self.a*np.sqrt(3) * x / self.a

    def R3(self, xdata, A):
        _, y = xdata[0], xdata[1]
        return  A*np.sqrt(3/(1 - self.a**2)) * y

    def R4(self, xdata, A):
        x, y = xdata[0], xdata[1]
        rho2 = x**2 + y**2
        return  A*(np.sqrt(5) /(2 * np.sqrt(1 - self.a**2 + 2*self.a**4)))*(3*rho2 - 1)

    def R5(self, xdata, A):
        x, y = xdata[0], xdata[1]
        return  A* 3 * x * y/(self.a*np.sqrt(1-self.a**2)) 

    def R6(self, xdata, A):
        x, y = xdata[0], xdata[1]
        first_bracket = np.sqrt(5) / (2*self.a**2 * (1-self.a**2) * np.sqrt(1 - 2*self.a**2 + 2*self.a**4))
        second_bracket = 3*(1 - self.a**2)**2 * x**2 - 3*self.a**4 * y**2 - self.a**2 * (1 - 3*self.a**2 + 2*self.a**4)
        return  A * first_bracket * second_bracket

    def R7(self, xdata, A):
        x, y = xdata[0], xdata[1]
        rho2 = x**2 + y**2
        first_bracket = np.sqrt(21)/(2*np.sqrt(27 - 81*self.a**2 + 116*self.a**4 - 62*self.a**6))
        second_bracket = (15*rho2 - 9 + 4*self.a**2)*y 
        return  A * first_bracket * second_bracket

    def R8(self, xdata, A):
        x, y = xdata[0], xdata[1]
        rho2 = x**2 + y**2
        first_bracket = np.sqrt(21)/(2*self.a*np.sqrt(35 - 70*self.a**2 + 62*self.a**4))
        second_bracket = (15*rho2 - 5 - 4*self.a**2)*x 
        return A * first_bracket * second_bracket

    def R9(self, xdata, A):
        x, y = xdata[0], xdata[1]
        num = np.sqrt(5)*np.sqrt((27 - 54*self.a**2 + 62*self.a**4) / (1 - self.a**2))
        denom = 2*self.a**2 * (27 - 81*self.a**2 + 116*self.a**4 - 62*self.a**6)
        first_bracket = num/denom
        second_bracket = 27 * (1 - self.a**2)**2 * x**2 - 35*self.a**4 * y**2 - self.a**2*(9 - 39*self.a**2 + 30*self.a**4) * y
        return A * first_bracket * second_bracket

    def R10(self, xdata, A):
        x, y = xdata[0], xdata[1]
        first_bracket = np.sqrt(5)/(2*self.a**3 * (1 - self.a**2) * np.sqrt(35 - 70*self.a**2 + 62*self.a**4))
        second_bracket = 35 * (1 - self.a**2)**2 * x**2 - 27 * self.a**4 * y**2 - self.a**2 * (21 - 51*self.a**2 + 30*self.a**4) * x
        return A * first_bracket * second_bracket

    def R11(self, xdata, A):
        x, y = xdata[0], xdata[1]
        rho2 = x**2 + y**2
        mu = np.sqrt(9 - 36*self.a**2 + 103*self.a**4 - 134*self.a**6 + 67*self.a**8)
        v = np.sqrt(49 - 196*self.a**2 + 330*self.a**4 - 268*self.a**6 + 134*self.a**8)

        first_bracket = 1/(8*mu)
        second_bracket = 315 * rho2**2 - 30*(7 + 2*self.a**2) * x**2 - 30*(9 - 2*self.a**2) * y**2 + 27 + 16*self.a**2 - 16*self.a**4
        return A * first_bracket * second_bracket

    def R12(self, xdata, A):
        x, y = xdata[0], xdata[1]
        rho2 = x**2 + y**2

        mu = np.sqrt(9 - 36*self.a**2 + 103*self.a**4 - 134*self.a**6 + 67*self.a**8)
        v = np.sqrt(49 - 196*self.a**2 + 330*self.a**4 - 268*self.a**6 + 134*self.a**8)
        n = 9 - 45*self.a**2 + 139*self.a**4 - 237*self.a**6 + 210*self.a**8 - 67*self.a**10

        c1 = 35 * (1 - self.a**2)**2 * (18 - 36*self.a**2 + 67*self.a**4)*x**4
        c2 = 630 * (1 - 2*self.a**2)*(1 - 2*self.a**2 + 2*self.a**4) * x**2 * y**2
        c3 = -35 * self.a**4 * (49 - 98*self.a**2 + 67*self.a**4)*y**4
        c4 = -30 * (1 - self.a**2) * (7 - 10*self.a**2 - 12*self.a**4 + 75*self.a**6 - 67*self.a**8) * x**2
        c5 = -30*self.a**2 * (7 - 77*self.a**2 + 189*self.a**4 - 193*self.a**6 + 67*self.a**8) * y**2
        c6 = self.a**2 * (1 - self.a**2) * (1 - 2*self.a**2) * (70 - 233*self.a**2 + 233*self.a**4)
        return A * 3 * mu/(8*self.a**2*v*n) * (c1 + c2 + c3 + c4 + c5 + c6)

    def R13(self, xdata, A):
        x, y = xdata[0], xdata[1]
        rho2 = x**2 + y**2
        first_bracket = np.sqrt(21) / (2*self.a*np.sqrt(1 - 3*self.a**2 + 4*self.a**4 - 2*self.a**6))
        second_bracket = (5*rho2 - 3)* x * y
        return A * first_bracket * second_bracket

    def R14(self, xdata, A):
        x, y = xdata[0], xdata[1]
        rho2 = x**2 + y**2
        mu = np.sqrt(9 - 36*self.a**2 + 103*self.a**4 - 134*self.a**6 + 67*self.a**8)
        v = np.sqrt(49 - 196*self.a**2 + 330*self.a**4 - 268*self.a**6 + 134*self.a**8)
        n = 9 - 45*self.a**2 + 139*self.a**4 - 237*self.a**6 + 210*self.a**8 - 67*self.a**10
        tau = 1/(128 * v * self.a**4 * (1 - self.a**2)**2)

        bracket1 = 735 * (1 - self.a**2)**4 * x**4 - 540*self.a**4 * (1 - self.a**2)**2 * x**2 * y**2 + 735*self.a**8 * y**4 - 90*self.a**2 * (1 - self.a**2)**3 * (7 - 9*self.a**2)*x**2
        bracket2 = 90*self.a**6 * (1 - self.a**2) * (2 - 9*self.a**2) * y**2 + 3*self.a**4 * (1 - self.a**2)**2 * (21 - 62*self.a**2 + 62*self.a**4)
        return A * 16 * tau * (bracket1 + bracket2)

    def R15(self, xdata, A):
        x, y = xdata[0], xdata[1]
        first_bracket = np.sqrt(21) / (2*self.a**3 * (1 - self.a**2) * np.sqrt(1 - 3*self.a**2 + 4*self.a**4 - 2*self.a**6))
        second_bracket = 5 * (1 - self.a**2)**2 * x**2 - 5*self.a**4 * y**2 - self.a**2 * (3 - 9*self.a**2 + 6*self.a**4) * x * y
        return A * first_bracket * second_bracket
    
    def evaluate(self, function_name, xdata, A):
        """
        Evaluate a specific polynomial function by name.
        
        Parameters:
        -----------
        function_name : str
            Name of the function to evaluate ('R1', 'R2', 'R3', 'R4', 'R5', ...)
        xdata : tuple or list
            Input data as (x, y) coordinates
        A : float
            Amplitude parameter
        
        Returns:
        --------
        numpy.ndarray
            Result of the polynomial evaluation
        """
        if hasattr(self, function_name):
            return getattr(self, function_name)(xdata, A)
        else:
            raise ValueError(f"Function {function_name} not found")
    
    def evaluate_all(self, xdata, A_values):
        """
        Evaluate all polynomial functions with given amplitude values.
        
        Parameters:
        -----------
        xdata : tuple or list
            Input data as (x, y) coordinates
        A_values : list or array
            Amplitude values for each function [A1, A2, A3, A4, A5]
        
        Returns:
        --------
        dict
            Dictionary with function names as keys and results as values
        """
        if len(A_values) != 15:
            raise ValueError("A_values must contain exactly 15 values")
        
        results = {}
        function_names = ['R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'R9', 'R10', 'R11', 'R12', 'R13', 'R14', 'R15']
        
        for i, func_name in enumerate(function_names):
            results[func_name] = getattr(self, func_name)(xdata, A_values[i])
        
        return results
    
    def get_function_list(self):
        """Return a list of available polynomial functions."""
        return ['R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'R9', 'R10', 'R11', 'R12', 'R13', 'R14', 'R15']
    


def extract_rectangular_coefficients(phase, a=1/np.sqrt(2)):
    """
    Extract rectangular polynomial coefficients from a phase map.
    
    Parameters:
    -----------
    phase : numpy.ndarray
        2D phase array to decompose
    a : float
        Rectangularity parameter (default: 1/sqrt(2) for square)
    
    Returns:
    --------
    list
        List of coefficients for R1 through R15
    """
    
    # Create the polynomial object
    rect_poly = RectangularPolynomials(a=a)
    
    # Create coordinate grids
    side_x = np.linspace(-a, a, phase.shape[1])
    side_y = np.linspace(-np.sqrt(1-a**2), np.sqrt(1-a**2), phase.shape[0])
    
    X1, X2 = np.meshgrid(side_x, side_y)
    size = X1.shape
    
    # Reshape coordinates for curve_fit
    x1_1d = X1.reshape((1, np.prod(size)))
    x2_1d = X2.reshape((1, np.prod(size)))
    xdata_curve_fit = np.vstack((x1_1d, x2_1d))
    
    # Reshape phase data
    psize = phase.shape
    p_shape = phase.reshape(np.prod(psize))
    
    # Get list of function names
    function_names = rect_poly.get_function_list()
    
    coefflist_rect = []
    
    # Method 1: Using the evaluate method
    for i in range(len(function_names)):
        func_name = function_names[i]
        
        # Create a wrapper function for curve_fit
        def poly_wrapper(xdata, A):
            X1_reshaped = xdata[0].reshape(size)
            X2_reshaped = xdata[1].reshape(size)  
            xdata_class = [X1_reshaped, X2_reshaped]
            result = rect_poly.evaluate(func_name, xdata_class, A)
            return result.flatten()
            # return rect_poly.evaluate(func_name, xdata, A)
        
        try:
            # Add bounds to prevent unrealistic coefficients
            popt, pcov = curve_fit(poly_wrapper, xdata_curve_fit, p_shape, 
                                 bounds=(-10, 10), maxfev=5000)
            coefflist_rect.append(popt[0])
        except Exception as e:
            print(f"Warning: curve_fit failed for {func_name}: {e}")
            coefflist_rect.append(0.0)  # fallback value
    
    return coefflist_rect


def extract_rectangular_coefficients_v2(phase, a=1/np.sqrt(2)):
    """
    Alternative implementation using direct method access.
    """
    
    # Create the polynomial object
    rect_poly = RectangularPolynomials(a=a)
    
    # Create coordinate grids
    side_x = np.linspace(-a, a, phase.shape[1])
    side_y = np.linspace(-np.sqrt(1-a**2), np.sqrt(1-a**2), phase.shape[0])
    
    X1, X2 = np.meshgrid(side_x, side_y)
    size = X1.shape
    
    # Reshape coordinates for curve_fit
    x1_1d = X1.reshape((1, np.prod(size)))
    x2_1d = X2.reshape((1, np.prod(size)))
    xdata = np.vstack((x1_1d, x2_1d))
    
    # Reshape phase data
    psize = phase.shape
    p_shape = phase.reshape(np.prod(psize))
    
    # Create dictionary mapping similar to your original
    coefdict_rect = {
        1: rect_poly.R1,
        2: rect_poly.R2,
        3: rect_poly.R3,
        4: rect_poly.R4,
        5: rect_poly.R5,
        6: rect_poly.R6,
        7: rect_poly.R7,
        8: rect_poly.R8,
        9: rect_poly.R9,
        10: rect_poly.R10,
        11: rect_poly.R11,
        12: rect_poly.R12,
        13: rect_poly.R13,
        14: rect_poly.R14,
        15: rect_poly.R15
    }
    
    coefflist_rect = []
    
    # Your original loop structure
    for i in range(len(coefdict_rect)):
        popt, pcov = curve_fit(coefdict_rect[i+1], xdata, p_shape)
        coefflist_rect.append(popt[0])
    
    return coefflist_rect

def extract_rectangular_coefficients_vectorized(phase, a=1/np.sqrt(2)):
    """
    More efficient implementation using least squares directly.
    """
    
    # Create the polynomial object
    rect_poly = RectangularPolynomials(a=a)
    
    # Create coordinate grids
    side_x = np.linspace(-a, a, phase.shape[1])
    side_y = np.linspace(-np.sqrt(1-a**2), np.sqrt(1-a**2), phase.shape[0])
    
    X1, X2 = np.meshgrid(side_x, side_y)
    
    # Reshape coordinates - note: different format for direct method calls
    xdata = [X1, X2]  # This matches the expected format for the polynomial methods
    
    # Reshape phase data
    p_flat = phase.flatten()
    
    # Build design matrix
    function_names = rect_poly.get_function_list()
    n_terms = len(function_names)
    n_pixels = len(p_flat)
    
    A_matrix = np.zeros((n_pixels, n_terms))
    
    print("Building design matrix...")
    for i in range(n_terms):
        func_name = function_names[i]
        # Evaluate polynomial with unit amplitude
        poly_vals = rect_poly.evaluate(func_name, xdata, A=1.0)
        A_matrix[:, i] = poly_vals.flatten()
    print("Done")
    # Solve using least squares
    coefficients, residuals, rank, s = np.linalg.lstsq(A_matrix, p_flat, rcond=None)
    
    return coefficients.tolist()


