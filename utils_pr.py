import numpy as np 
from numpy.fft import fftshift, ifftshift, fft2, ifft2
import torch
from scipy.ndimage import zoom 
from joblib import Parallel, delayed
import time
from time import strftime
import functools
import multiprocessing as mp
from tqdm import tqdm 
from .ZernikePolynomials import SquarePolynomials

def time_it(func):
    """Decorator to measure execution time of a function."""
    @functools.wraps(func)  # Preserve function metadata
    def wrapper(*args, **kwargs):
        start_time = time.time()  # Record start time
        result = func(*args, **kwargs)  # Execute the function
        end_time = time.time()  # Record end time
        execution_time = end_time - start_time  # Compute execution time
        print(f"Function '{func.__name__}' executed in {execution_time:.6f} seconds")
        return result  # Return original function result
    return wrapper


def run_reconstruction(args):
    
    run_id, recon_class, iterations, fname, zoom_factor, kwargs = args
    
    recon = recon_class(**kwargs)
    
    recon.prepare(extend = 'double', zoom_factor=zoom_factor)
    recon.iterate(iterations)
    recon.save_reconsturction(f"{fname}_id{run_id}.h5")

def reconstruction_repeats(recon_class, iterations, fname, repeats = 5,  n_jobs=4, zoom_factor=1, **kwargs):
    time_str = strftime("%Y-%m-%d_%H.%M")
    fname += f"_{time_str}"
    
    args_list = [(i, recon_class, iterations, fname, zoom_factor, kwargs) for i in range(repeats)]
 
    with mp.Pool(processes=n_jobs) as pool:
        list(tqdm(pool.imap_unordered(run_reconstruction, args_list), total=repeats, desc="Reconstructions"))
    

    
def calc_obj_freq_bandwidth(lr_psize):
    """
    Calculate the Object bandwidth based on the step size in scanning mode.  
    """
    omega_obj_x, omega_obj_y = 2 * np.pi / lr_psize, 2 * np.pi / lr_psize

    return omega_obj_x, omega_obj_y

def prepare_dims(images, pupil_kins, lr_psize, extend = None, band_multiplier = 1):
    """
    Prepare the dimensions of the high resolution fourier space image. 

    High resolution pixel size in fourier space: 2pi/Lx, 2pi/Ly. 
                Where Lx, Ly is the scan length in x and y. # CHECK ME 
    """
    
    # Convert inputs to numpy arrays
    pupil_kins = np.array(pupil_kins)

    # Low-resolution image shape
    coh_img_dim = images[0].shape
    nx_lr, ny_lr = coh_img_dim

    # Object size == Scan length
    Lx, Ly = coh_img_dim[0] * lr_psize, coh_img_dim[1] * lr_psize

    # High-resolution Fourier pixel size 
    dkx, dky = 2 * np.pi / Lx, 2 * np.pi / Ly

    # Define the extents of the wave vectors
    kx, ky = pupil_kins[:, 0], pupil_kins[:, 1]
    kx_min, kx_max = np.min(kx), np.max(kx)
    ky_min, ky_max = np.min(ky), np.max(ky)
    
    if extend == 'double' :
        range_x = kx_max - kx_min
        range_y = ky_max - ky_min
        
        kx_min = kx_min - range_x/2
        kx_max = kx_max + range_x/2
        
        ky_min = ky_min - range_y/2
        ky_max = ky_max + range_y/2
    elif extend == 'triple':
        range_x = kx_max - kx_min
        range_y = ky_max - ky_min

        kx_min -= 1 * range_x
        kx_max += 1 * range_x
        ky_min -= 1 * range_y
        ky_max += 1 * range_y
    elif extend == 'quadruple':
        range_x = kx_max - kx_min
        range_y = ky_max - ky_min

        kx_min -= 1.5 * range_x
        kx_max += 1.5 * range_x
        ky_min -= 1.5 * range_y
        ky_max += 1.5 * range_y
        
    elif extend == 'quintiple':
        range_x = kx_max - kx_min
        range_y = ky_max - ky_min

        kx_min -= 2 * range_x
        kx_max += 2 * range_x
        ky_min -= 2 * range_y
        ky_max += 2 * range_y

    elif extend == 'sixtuple':
        range_x = kx_max - kx_min
        range_y = ky_max - ky_min

        kx_min -= 2.5 * range_x
        kx_max += 2.5 * range_x
        ky_min -= 2.5 * range_y
        ky_max += 2.5 * range_y
        
    elif extend == 'by_bandwidth':
        # Object bandwidth 
        omega_obj_x, omega_obj_y = calc_obj_freq_bandwidth(lr_psize)
        # Extend the range of kx and ky to fit boundary values
        kx_min = kx_min - omega_obj_x * band_multiplier
        kx_max = kx_max + omega_obj_x * band_multiplier
        
        ky_min = ky_min - omega_obj_y * band_multiplier
        ky_max = ky_max + omega_obj_y * band_multiplier

    return (kx_min,kx_max), (ky_min,ky_max), (dkx,dky)

def init_hr_image(bounds_x, bounds_y, dks):

    kx_min_n, kx_max_n = bounds_x
    ky_min_n, ky_max_n = bounds_y
    
    dkx, dky = dks
    
    kx_sp = np.arange(kx_min_n, kx_max_n, dkx)
    ky_sp = np.arange(ky_min_n, ky_max_n, dky)

    nkx, nky = kx_sp.shape[0], ky_sp.shape[0]
    
    # Round up to the nearest even number
    nkx = int(np.ceil(nkx / 2) * 2)  
    nky = int(np.ceil(nky / 2) * 2) 
        
    # High-resolution Object and Fourier spaces
    hr_obj_image = np.ones((nkx, nky), dtype=complex)
    hr_fourier_image =  np.ones((nkx, nky), dtype=complex) #fftshift(fft2(ifftshift(hr_obj_image)))

    return hr_obj_image, hr_fourier_image


def mask_torch_ctf(outer_size, device=torch.device('cpu')):
    """
    Create a (2N, 2M) array with ones in the center region of size (N, M) and zeros elsewhere.
    
    Parameters:
        outer_size: tuple (2N, 2M) -> total size of the array
        inner_size: tuple (N, M) -> size of the central region filled with ones
    
    Returns:
        mask: (2N, 2M) torch.Tensor
    """
    mask = torch.zeros(outer_size, dtype=torch.float64, device=device)
    
    # Calculate center indices
    N, M = outer_size[0]//2, outer_size[1]//2
    
    start_x, start_y = (outer_size[0] - N) // 2, (outer_size[1] - M) // 2
    end_x, end_y = start_x + N, start_y + M

    # Set central region to ones
    mask[start_x:end_x, start_y:end_y] = 1

    return mask


def make_dims_even(size_tuple):
    """
    Takes a tuple of (x, y) and ensures both values are even.
    If a value is odd, it's increased by 1 to make it even.
    """
    x, y = size_tuple
    
    if x % 2 != 0:
        x += 1
        
    if y % 2 != 0:
        y += 1
        
    return (x, y)

def upsample_image(im, zoom_factor):
    """
    Upsample a single image using zoom.
    """
    return zoom(im, zoom_factor, order=3).astype(complex)

def upsample_images(images, zoom_factor, n_jobs=4):
    """
    Upsample a list of images in parallel.

    Parameters:
        images (list of numpy.ndarray): List of input images.
        zoom_factor (float or tuple): Zoom factor for upsampling.
        n_jobs (int): Number of CPU cores to use. Default is -1 (use all available cores).

    Returns:
        numpy.ndarray: Array of upsampled images.
    """
    # Use joblib to parallelize the upsampling
    up_images = Parallel(n_jobs=n_jobs)(delayed(upsample_image)(im, zoom_factor) for im in images)

    return np.array(up_images)

def downsample_array(arr, new_shape):
    """
    Downsample a 2D array by taking mean of non-overlapping rectangular blocks
    
    Args:
        arr (np.ndarray): Input 2D array
        new_shape (tuple): Desired output shape
    
    Returns:
        np.ndarray: Downsampled array
    """
    # Calculate exact block sizes
    orig_rows, orig_cols = arr.shape
    new_rows, new_cols = new_shape
    
    # Calculate block dimensions
    row_factor = orig_rows // new_rows
    col_factor = orig_cols // new_cols
    
    # Trim array to ensure exact division
    trimmed_arr = arr[:new_rows*row_factor, :new_cols*col_factor]
    
    # Reshape and compute mean
    downsampled = trimmed_arr.reshape(
        new_rows, row_factor, 
        new_cols, col_factor
    ).max(axis=(1, 3))
    
    return downsampled

def pad_to_double(arr):
    
    m, n = arr.shape
    out = np.zeros((2*m, 2*n), dtype=arr.dtype)
    
    start_m = m // 2
    start_n = n // 2
    out[start_m:start_m+m, start_n:start_n+n] = arr

    return out

def pad_to_double_torch(arr):
    
    m, n = arr.shape
    out = torch.zeros((2*m, 2*n), dtype=arr.dtype, device = arr.device)
    
    start_m = m // 2
    start_n = n // 2
    out[start_m:start_m+m, start_n:start_n+n] = arr

    return out
    
@time_it
def pad_to_double_parallel(image_list, n_jobs=8):
    """
    Slices the center of each 2D numpy array in a list,
    keeping only half the size in each dimension.
    
    Args:
        image_list: List of 2D numpy arrays
        
    Returns:
        List of center-sliced 2D numpy arrays, each with half the dimensions
    """
    # Use joblib to parallelize the median filter application
    padded_images = Parallel(n_jobs=n_jobs)(
        delayed(pad_to_double)(image) for image in image_list
    
    )
    return padded_images


def pad_to_shape(image_lr, shape):

    ny, nx = shape
    h, w = image_lr.shape
    
    if h > ny or w > nx:
        raise ValueError("Low-res image is larger than pupil size. Cannot pad.")
    
    # Create zeros
    padded = np.zeros((ny, nx), dtype=image_lr.dtype)
    
    # Compute start indices to center image
    start_y = (ny - h) // 2
    start_x = (nx - w) // 2
    
    # Place low-res image into padded array
    padded[start_y:start_y+h, start_x:start_x+w] = image_lr
    
    return padded

def pad_to_shape_parallel(image_list, shape, n_jobs=8):
    """
    Slices the center of each 2D numpy array in a list,
    keeping only half the size in each dimension.
    
    Args:
        image_list: List of 2D numpy arrays
        
    Returns:
        List of center-sliced 2D numpy arrays, each with half the dimensions
    """
    # Use joblib to parallelize the median filter application
    padded_images = Parallel(n_jobs=n_jobs, backend = 'threading')(
        delayed(pad_to_shape)(image, shape) for image in image_list
    
    )
    return padded_images
    
def get_zernike_wavefront(coefficients, pupil_shape):
    
    shape_y, shape_x = pupil_shape
    square_poly = SquarePolynomials() 
    
    # Create coordinate grids
    side_x = np.linspace(-1/np.sqrt(2), 1/np.sqrt(2), shape_x)
    side_y = np.linspace(-1/np.sqrt(2), 1/np.sqrt(2), shape_y)

    X, Y = np.meshgrid(side_x, side_y)
    xdata = [X, Y]

    all_results = square_poly.evaluate_all(xdata, coefficients)
    new_wavefront = sum(all_results.values())
    
    return new_wavefront

def unwrap_phase(phase):
    return torch.atan2(torch.sin(phase), torch.cos(phase))
