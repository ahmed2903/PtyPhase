import numpy as np 

class SquarePolynomials: 
    """
    A class containing a set of orthonormal square polynomials 
    in Cartesian coordinates from Mahajan and Dai 
    Orthonormal polynomials in wavefront analysis: analytical solution
    J. Opt. Soc. Am. A / Vol. 24, No. 9 / September 2007
    """
    def __init__(self):
        pass


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
    
