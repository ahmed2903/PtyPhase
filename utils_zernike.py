
import numpy as np 
from .ZernikePolynomials import RectangularPolynomials, SquarePolynomials
from scipy.optimize import curve_fit

def unwrap_phase(phase):
    return torch.atan2(torch.sin(phase), torch.cos(phase))

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


def extract_square_coefficients_vectorized(phase):
    """
    More efficient implementation using least squares directly.
    """
    
    # Create the polynomial object
    square_poly = SquarePolynomials()
    
    # Create coordinate grids
    side_x = np.linspace(-1/np.sqrt(2), 1/np.sqrt(2), phase.shape[1])
    side_y = np.linspace(-1/np.sqrt(2), 1/np.sqrt(2), phase.shape[0])

    X1, X2 = np.meshgrid(side_x, side_y)
    
    # Reshape coordinates - note: different format for direct method calls
    xdata = [X1, X2]  # This matches the expected format for the polynomial methods
    
    # Reshape phase data
    p_flat = phase.flatten()
    
    # Build design matrix
    function_names = square_poly.get_function_list()
    n_terms = len(function_names)
    n_pixels = len(p_flat)
    
    A_matrix = np.zeros((n_pixels, n_terms))
    
    print("Building design matrix...")
    for i in range(n_terms):
        func_name = function_names[i]
        # Evaluate polynomial with unit amplitude
        poly_vals = square_poly.evaluate(func_name, xdata, A=1.0)
        A_matrix[:, i] = poly_vals.flatten()
    print("Done")
    # Solve using least squares
    coefficients, residuals, rank, s = np.linalg.lstsq(A_matrix, p_flat, rcond=None)
    
    return coefficients.tolist()