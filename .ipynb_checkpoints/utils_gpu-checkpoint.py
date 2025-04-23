import cupy as cp 
from cupy.fft import fftshift, ifftshift, fft2, ifft2
import torch
from cupyx.scipy.ndimage import zoom
from joblib import Parallel, delayed
import time
import functools

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

def calc_obj_freq_bandwidth(lr_psize):
    """
    Calculate the Object bandwidth based on the step size in scanning mode.  
    """
    omega_obj_x, omega_obj_y = 2 * cp.pi / lr_psize, 2 * cp.pi / lr_psize

    return omega_obj_x, omega_obj_y

def prepare_dims(images, pupil_kins, lr_psize, extend = None, band_multiplier = 1):
    """
    Prepare the dimensions of the high resolution fourier space image. 

    High resolution pixel size in fourier space: 2pi/Lx, 2pi/Ly. 
                Where Lx, Ly is the scan length in x and y. # CHECK ME 
    """
    
    # Convert inputs to numpy arrays
    pupil_kins = cp.array(pupil_kins)

    # Low-resolution image shape
    coh_img_dim = images[0].shape
    nx_lr, ny_lr = coh_img_dim

    # Object size == Scan length
    Lx, Ly = coh_img_dim[0] * lr_psize, coh_img_dim[1] * lr_psize

    

    # High-resolution Fourier pixel size 
    dkx, dky = 2 * cp.pi / Lx, 2 * cp.pi / Ly

    # Define the extents of the wave vectors
    kx, ky = pupil_kins[:, 0], pupil_kins[:, 1]
    kx_min, kx_max = cp.min(kx), cp.max(kx)
    ky_min, ky_max = cp.min(ky), cp.max(ky)

    
    
    if extend == 'double' :
        range_x = kx_max - kx_min
        range_y = ky_max - ky_min
        
        kx_min = kx_min - range_x/2
        kx_max = kx_max + range_x/2
        
        ky_min = ky_min - range_y/2
        ky_max = ky_max + range_y/2

    # FIX ME!!!
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
    
    kx_sp = cp.arange(kx_min_n, kx_max_n, dkx)
    ky_sp = cp.arange(ky_min_n, ky_max_n, dky)

    nkx, nky = kx_sp.shape[0], ky_sp.shape[0]
    
    # Round up to the nearest even number
    nkx = int(cp.ceil(nkx / 2) * 2)  
    nky = int(cp.ceil(nky / 2) * 2) 
        
    # High-resolution Object and Fourier spaces
    hr_obj_image = cp.ones((nkx, nky), dtype=complex)
    hr_fourier_image =  cp.ones((nkx, nky), dtype=complex) #fftshift(fft2(ifftshift(hr_obj_image)))

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


def upsample_images(images, zoom_factor):
    """
    Upsample a batch of images using CuPy (on GPU).

    Parameters:
        images (numpy.ndarray): Input images (N, H, W)
        zoom_factor (float or tuple): Zoom factor.

    Returns:
        cupy.ndarray: Upsampled images.
    """
    # Transfer to GPU
    images_gpu = cp.asarray(images)

    upsampled = []
    for i in range(images_gpu.shape[0]):
        # Apply zoom per image (for now — batch zooming isn't supported natively)
        upsampled_image = zoom(images_gpu[i], zoom_factor, order=3).astype(cp.complex64)
        upsampled.append(upsampled_image)

    return cp.stack(upsampled)


def downsample_array(arr, new_shape):
    """
    Downsample a 2D array by taking mean of non-overlapping rectangular blocks
    
    Args:
        arr (cp.ndarray): Input 2D array
        new_shape (tuple): Desired output shape
    
    Returns:
        cp.ndarray: Downsampled array
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


def pad_to_double(img):
    """
    Pads a 2D CuPy array to double its size, centering the original image.

    Args:
        img (cupy.ndarray): 2D image array

    Returns:
        cupy.ndarray: Padded image
    """
    h, w = img.shape
    pad_h = h // 2
    pad_w = w // 2

    return cp.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)


@time_it
def pad_to_double_batch(image_list):
    """
    Pads a list of 2D NumPy or CuPy arrays on the GPU.

    Args:
        image_list (list of numpy.ndarray or cupy.ndarray)

    Returns:
        cupy.ndarray: Stack of padded images
    """
    # Ensure CuPy arrays
    gpu_images = [cp.asarray(im) for im in image_list]
    padded_images = [pad_to_double(im) for im in gpu_images]
    return cp.stack(padded_images)
