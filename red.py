import numpy as np

def oversample_2d(x, m_shape):
    """
    Oversampling operator O_{mn} in 2D: zero-pad image x (shape n_shape) to shape m_shape > n_shape
    Pads symmetrically on each axis.
    """
    n1, n2 = x.shape
    m1, m2 = m_shape
    pad1_left = (m1 - n1) // 2
    pad1_right = m1 - n1 - pad1_left
    pad2_left = (m2 - n2) // 2
    pad2_right = m2 - n2 - pad2_left
    return np.pad(x, ((pad1_left, pad1_right), (pad2_left, pad2_right)), mode='constant')

def oversample_adjoint_2d(z, n_shape):
    """
    Adjoint O_{mn}^T in 2D: crop center n_shape region from z (shape m_shape)
    """
    m1, m2 = z.shape
    n1, n2 = n_shape
    start1 = (m1 - n1) // 2
    end1 = start1 + n1
    start2 = (m2 - n2) // 2
    end2 = start2 + n2
    return z[start1:end1, start2:end2]

def fourier_magnitude_projection_2d(v, y, eps=1e-12):
    """
    Projection onto Fourier magnitude constraint in 2D:
    Pi_M(v) = F^{-1}( y * F(v) / |F(v)| )
    
    v, y are 2D arrays.
    """
    V = np.fft.fft2(v)
    V_mag = np.abs(V)
    V_phase = V / (V_mag + eps)
    proj = np.fft.ifft2(y * V_phase)
    return proj

def constraint_projection_2d(x):
    """
    Pi_C: Projection onto constraints in 2D (nonnegativity here)
    """
    return np.maximum(np.real(x), 0)

def red_proximal_step_2d(s, denoiser, lambd, tau):
    """
    One RED proximal step in 2D:
    s_plus = Pi_C( (s + lambda * tau * D(s)) / (1 + lambda * tau) )
    """
    Ds = denoiser(s)
    s_new = (s + lambd * tau * Ds) / (1 + lambd * tau)
    return constraint_projection_2d(s_new)

def RED_ITA_F_2d(y, n_shape, rho, lambd, denoiser, max_iter=100, verbose=True):
    """
    RED-ITA-F algorithm for 2D signals/images
    
    Parameters:
    y: measured Fourier magnitudes (2D array, shape m_shape)
    n_shape: tuple (n1, n2) original image shape
    rho: ADMM penalty parameter
    lambd: regularization parameter for RED prior
    denoiser: callable denoiser function D(x), accepts 2D real array
    max_iter: max iterations
    verbose: print progress
    
    Returns:
    x: reconstructed image (real-valued 2D array)
    """
    m_shape = y.shape
    n1, n2 = n_shape
    m1, m2 = m_shape
    
    # Initialize variables
    x = np.zeros(n_shape)
    z = oversample_2d(x, m_shape).astype(np.complex128)
    u = np.zeros(m_shape, dtype=np.complex128)
    
    tau = (n1 * n2) / (m1 * m2 * rho)
    
    for k in range(max_iter):
        v = z + u
        
        # x-update
        s = (n1 * n2) / (m1 * m2) * np.real(oversample_adjoint_2d(v, n_shape))
        x = red_proximal_step_2d(s, denoiser, lambd, tau)
        
        # Forward transform oversampled x
        x_oversampled = oversample_2d(x, m_shape)
        
        # z-update: relaxed projection
        temp = x_oversampled - u
        proj = fourier_magnitude_projection_2d(temp, y)
        z = (rho / (rho + 1)) * temp + (1 / (rho + 1)) * proj
        
        # u-update (dual variable)
        u = u + z - x_oversampled
        
        if verbose and (k % 10 == 0 or k == max_iter - 1):
            residual = np.linalg.norm(y - np.abs(np.fft.fft2(oversample_2d(x, m_shape))))
            print(f"Iter {k+1}/{max_iter}, Residual: {residual:.5e}")
    
    return x


# Example 2D denoiser: Gaussian smoothing (replace with your favorite)
def example_denoiser_2d(x):
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(x, sigma=1)

# --- Usage Example ---
if __name__ == "__main__":
    # Synthetic 2D signal (image)
    n_shape = (64, 64)
    x_true = np.zeros(n_shape)
    x_true[20:44, 25:39] = 1.0  # rectangular patch
    
    oversampling_factor = 2
    m_shape = (n_shape[0] * oversampling_factor, n_shape[1] * oversampling_factor)
    
    # Generate measurements y = |F O_{mn} x_true|
    x_oversampled = oversample_2d(x_true, m_shape)
    y = np.abs(np.fft.fft2(x_oversampled))
    
    # Parameters
    rho = 1.0
    lambd = 0.1
    max_iter = 100
    
    # Run RED-ITA-F 2D
    x_rec = RED_ITA_F_2d(y, n_shape, rho, lambd, example_denoiser_2d, max_iter)
    
    import matplotlib.pyplot as plt
    plt.subplot(1, 2, 1)
    plt.title("True Image")
    plt.imshow(x_true, cmap='gray')
    plt.colorbar()
    
    plt.subplot(1, 2, 2)
    plt.title("Reconstruction")
    plt.imshow(x_rec.real, cmap='gray')
    plt.colorbar()
    plt.show()
