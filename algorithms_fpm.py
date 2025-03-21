import numpy as np 
import torch
from numpy.fft import fftshift, ifftshift, fft2, ifft2

from utils_pr import *


def fourier_ptychography_reconstruction(images, kout_vec, 
                                        pupil_func, lr_psize, 
                                        algorithm='epie', 
                                        alpha= 0.1, beta = 0.1,
                                        num_iter=50, 
                                        hr_obj_image=None, 
                                        hr_fourier_image=None):
    """
    General Fourier Ptychography reconstruction algorithm with multiple phase retrieval methods.

    Parameters:
        images (list of ndarray): List of low-resolution images.
        kout_coords (list of tuples): Coordinates of the incoming wavevector.
        kout_vec (list of tuples): Incident wave vectors (kin) for each image.
        pupil_func (float): Pupil function.
        lr_psize (float): Low-resolution pixel size.
        num_iter (int): Number of iterations for the reconstruction.
        algorithm (str): Algorithm choice ('gerchberg_saxton', 'alternating_projections', 'difference_map').

    Returns:
        object_image (ndarray): Reconstructed high-resolution object.
        fourier_image (ndarray): Reconstructed high-resolution Fourier image.
        pupil_func (ndarray): Recovered pupil function.
    """
    
    algorithms = {
        'difference_map': difference_map,
        'epie': epie,
        'pie': pie,
        'epry': epry,
        'epry_lr': epry_lr,
    }
    
    if algorithm not in algorithms:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    kout_vec = np.array(kout_vec)
    images = np.array(images)

    bounds_x, bounds_y, dks = prepare_dims(images, kout_vec, lr_psize)
    omegas = calc_obj_freq_bandwidth(lr_psize)
        
    return algorithms[algorithm](images, pupil_func, kout_vec, 
                                 bounds_x, bounds_y, dks, 
                                 omegas, num_iter,
                                 alpha= 0.1, beta = 0.1,
                                 hr_obj_image=hr_obj_image, 
                                 hr_fourier_image=hr_fourier_image)

def epry(images, pupil_func, kout_vec, 
                 bounds_x, bounds_y, dks, 
                 omegas, num_iter = 50,
                 alpha= 0.1, beta = 0.1,
                hr_obj_image=None, hr_fourier_image=None):
    
    """
    Embedded Pupil Function Reconvery (EPRy) implementation.
    """

    
    kx_min_n, kx_max_n = bounds_x
    ky_min_n, ky_max_n = bounds_y
    dkx, dky = dks

    if hr_obj_image is None or hr_fourier_image is None:
        hr_obj_image, hr_fourier_image = init_hr_image(bounds_x, bounds_y, dks)
    
    omega_obj_x, omega_obj_y = omegas

    # Shapes
    nx_lr , ny_lr = images[0].shape
    nx_hr , ny_hr = hr_obj_image.shape
    
    for it in range(num_iter):
        print(f"Iteration Number: {it+1}/{num_iter}")
        
        for i, (image, kx_iter, ky_iter) in enumerate(zip(images, kout_vec[:,0], kout_vec[:,1])):
            kx_cidx = round((kx_iter - kx_min_n ) / dkx)
            kx_lidx = round(max(kx_cidx - omega_obj_x/(2*dkx), 0))
            kx_hidx = round(kx_cidx + omega_obj_x/(2*dkx)) + (1 if nx_lr % 2 != 0 else 0)

            
            ky_cidx = round((ky_iter - ky_min_n ) / dky)
            ky_lidx = round(max(ky_cidx - omega_obj_y/(2*dky), 0))
            ky_hidx = round(ky_cidx + omega_obj_y/(2*dky)) + (1 if ny_lr % 2 != 0 else 0)


            pupil_func_patch = pupil_func[kx_lidx:kx_hidx, ky_lidx:ky_hidx] #* CTF
            # Forward propagation: Simulate low-resolution image
            image_FT = hr_fourier_image[kx_lidx:kx_hidx, ky_lidx:ky_hidx] * pupil_func_patch
            #image_FT *= nx_lr/nx_hr 
            
            image_lr = fftshift(ifft2(ifftshift(image_FT)))
            image_lr_update = np.sqrt(image) * np.exp(1j * np.angle(image_lr))
            #image_lr_update *= nx_hr/nx_lr
            image_FT_update = fftshift(fft2(ifftshift(image_lr_update))) #* (1/(pupil_func_patch+1e-23)) #* CTF

            # Update fourier spectrum
            mod_pupil = np.abs(pupil_func_patch)**2  
            weight_fac_pupil = np.conjugate(pupil_func_patch) / (mod_pupil.max() + 1e-23)

            delta_lowres_ft = image_FT_update - image_FT 
            hr_fourier_image[kx_lidx:kx_hidx, ky_lidx:ky_hidx] += delta_lowres_ft * 0.9 * weight_fac_pupil
            

    # Reconstruct the high-resolution Fourier image
    hr_obj_image = fftshift(ifft2(ifftshift(hr_fourier_image)))
    print('done')
    return hr_obj_image, hr_fourier_image, pupil_func

def epry_lr(images, pupil_func, kout_vec, 
                 bounds_x, bounds_y, dks, 
                 omegas, num_iter = 50,
                alpha = 0.1, beta = 0.2,
                hr_obj_image=None, hr_fourier_image=None):
    
    """
    Embedded Pupil Function Reconvery (EPRy) implementation.
    """

    kx_min_n, kx_max_n = bounds_x
    ky_min_n, ky_max_n = bounds_y
    dkx, dky = dks

    omega_obj_x, omega_obj_y = omegas
    
    if hr_obj_image is None or hr_fourier_image is None: 
        hr_obj_image = np.ones_like(images[0]).astype(complex)
        hr_fourier_image = np.ones_like(images[0]).astype(complex)
    
    # Shapes
    nx_lr , ny_lr = images[0].shape
    nx_hr , ny_hr = hr_obj_image.shape
    print(images.shape)
    for it in range(num_iter):
        print(f"Iteration Number: {it+1}/{num_iter}")
        
        for i, (image, kx_iter, ky_iter) in enumerate(zip(images, kout_vec[:,0], kout_vec[:,1])):
            kx_cidx = round((kx_iter - kx_min_n ) / dkx)
            kx_lidx = round(max(kx_cidx - omega_obj_x/(2*dkx), 0))
            kx_hidx = round(kx_cidx + omega_obj_x/(2*dkx)) + (1 if nx_lr % 2 != 0 else 0)

            
            ky_cidx = round((ky_iter - ky_min_n ) / dky)
            ky_lidx = round(max(ky_cidx - omega_obj_y/(2*dky), 0))
            ky_hidx = round(ky_cidx + omega_obj_y/(2*dky)) + (1 if ny_lr % 2 != 0 else 0)


            pupil_func_patch = pupil_func[kx_lidx:kx_hidx, ky_lidx:ky_hidx] #* CTF
            # Forward propagation: Simulate low-resolution image
            image_FT = hr_fourier_image * pupil_func_patch
            image_FT *= nx_lr/nx_hr 
            
            image_lr = fftshift(ifft2(ifftshift(image_FT)))
            image_lr_update = np.sqrt(image) * np.exp(1j * np.angle(image_lr))
            image_lr_update *= nx_hr/nx_lr
            image_FT_update = fftshift(fft2(ifftshift(image_lr_update))) #* (1/(pupil_func_patch+1e-23)) #* CTF

            # Update fourier spectrum
            mod_pupil = np.abs(pupil_func_patch)**2  
            weight_fac_pupil = np.conjugate(pupil_func_patch) / (mod_pupil.max() + 1e-23)
            print(mod_pupil.max())
            weight_fac_pupil *= alpha
            
            delta_lowres_ft = image_FT_update - image_FT 
            hr_fourier_image += delta_lowres_ft * weight_fac_pupil

            # Update Pupil Function 
            mod_obj = np.abs(hr_fourier_image)**2
            weight_factor_obj = np.conjugate(hr_fourier_image) / (mod_obj.max() + 1e-23)
            print(mod_obj.max())
            weight_factor_obj *= beta 
            
            pupil_func[kx_lidx:kx_hidx, ky_lidx:ky_hidx] += weight_factor_obj * delta_lowres_ft
    
    
    # Reconstruct the high-resolution Fourier image
    hr_obj_image = fftshift(ifft2(ifftshift(hr_fourier_image)))
    print('done')
    return hr_obj_image, hr_fourier_image, pupil_func




def pie(images, pupil_func, kout_vec, 
                 bounds_x, bounds_y, dks, 
                 omegas, num_iter = 50,
        alpha = 0.1, beta = 0.1,
       hr_obj_image=None, hr_fourier_image=None):
    
    """
    Ptychographical Iterative Engine (PIE) implementation.
    """

    kx_min_n, kx_max_n = bounds_x
    ky_min_n, ky_max_n = bounds_y
    dkx, dky = dks

    if hr_obj_image or hr_fourier_image is None:
        hr_obj_image, hr_fourier_image = init_hr_image(bounds_x, bounds_y, dks)
    
    omega_obj_x, omega_obj_y = omegas

    # Shapes
    nx_lr , ny_lr = images[0].shape
    nx_hr , ny_hr = hr_obj_image.shape

    # Calculate the zoom factor
    zoom_factor = nx_hr / nx_lr, ny_hr / ny_lr

    # Zoom the lower resolution images
    images = upsample_images(images, zoom_factor, n_jobs = 90)
    print(images.shape[0])
    
    CTF = np.zeros((nx_hr, ny_hr))
    
    for it in range(num_iter):
        print(f"Iteration Number: {it+1}/{num_iter}")
        
        for i, (image, kx_iter, ky_iter) in enumerate(zip(images, kout_vec[:,0], kout_vec[:,1])):
            
            kx_cidx = round((kx_iter - kx_min_n ) / dkx)
            kx_lidx = round(max(kx_cidx - omega_obj_x/(2*dkx), 0))
            kx_hidx = round(kx_cidx + omega_obj_x/(2*dkx)) + (1 if nx_lr % 2 != 0 else 0)
            
            ky_cidx = round((ky_iter - ky_min_n ) / dky)
            ky_lidx = round(max(ky_cidx - omega_obj_y/(2*dky), 0))
            ky_hidx = round(ky_cidx + omega_obj_y/(2*dky)) + (1 if ny_lr % 2 != 0 else 0)
            
            #CTF[kx_lidx:kx_hidx, ky_lidx:ky_hidx] = 1.0
            #CTF = CTF.astype(complex)
            
            # Forward propagation: Simulate low-resolution image
            image_FT = hr_fourier_image #* pupil_func #* CTF
            image_FT *= nx_lr/nx_hr 
            
            image_lr = fftshift(ifft2(ifftshift(image_FT)))
            image_lr_update = np.sqrt(image) * np.exp(1j * np.angle(image_lr))
            image_lr_update *= nx_hr/nx_lr
            image_FT_update = fftshift(fft2(ifftshift(image_lr_update))) #* (1/pupil_func) #* CTF

            # Update fourier spectrum
            #mod_pupil = np.abs(pupil_func)**2  
            #weight_fac_pupil = np.conjugate(pupil_func) / (mod_pupil.max() + 1e-23)

            delta_lowres_ft = image_FT_update - image_FT 
            hr_fourier_image += delta_lowres_ft * 0.9 #* weight_fac_pupil # FIX ME
            

    # Reconstruct the high-resolution Fourier image
    hr_obj_image = fftshift(ifft2(ifftshift(hr_fourier_image)))
    print('done')
    return hr_obj_image, hr_fourier_image, pupil_func, CTF
    
def epie(images, pupil_func, kout_vec, 
                 bounds_x, bounds_y, dks, 
                 omegas, num_iter = 50,
         alpha = 0.1, beta = 0.1,
        hr_obj_image=None, hr_fourier_image=None):
    
    """
    Extended Ptychographical Iterative Engine (EPIE) implementation. 
    """

    kx_min_n, kx_max_n = bounds_x
    ky_min_n, ky_max_n = bounds_y
    dkx, dky = dks

    if hr_obj_image or hr_fourier_image is None:
        hr_obj_image, hr_fourier_image = init_hr_image(bounds_x, bounds_y, dks)

    omega_obj_x, omega_obj_y = omegas

    # Shapes
    nx_lr , ny_lr = images[0].shape
    nx_hr , ny_hr = hr_obj_image.shape

    # Calculate the zoom factor
    zoom_factor = nx_hr / nx_lr, ny_hr / ny_lr

    # Zoom the lower resolution images
    images = upsample_images(images, zoom_factor, n_jobs = 90)
    print(images.shape[0])
    
    CTF = np.zeros((nx_hr, ny_hr))
    
    for it in range(num_iter):
        print(f"Iteration Number: {it+1}/{num_iter}")
        
        for i, (image, kx_iter, ky_iter) in enumerate(zip(images, kout_vec[:,0], kout_vec[:,1])):
            
            kx_cidx = round((kx_iter - kx_min_n ) / dkx)
            kx_lidx = round(max(kx_cidx - omega_obj_x/(2*dkx), 0))
            kx_hidx = round(kx_cidx + omega_obj_x/(2*dkx)) + (1 if nx_lr % 2 != 0 else 0)
            
            ky_cidx = round((ky_iter - ky_min_n ) / dky)
            ky_lidx = round(max(ky_cidx - omega_obj_y/(2*dky), 0))
            ky_hidx = round(ky_cidx + omega_obj_y/(2*dky)) + (1 if ny_lr % 2 != 0 else 0)


            
            #CTF[kx_lidx:kx_hidx, ky_lidx:ky_hidx] = 1.0
            #CTF = CTF.astype(complex)
            
            # Forward propagation: Simulate low-resolution image
            image_FT = hr_fourier_image * pupil_func #* CTF
            image_FT *= nx_lr/nx_hr 
            
            image_lr = fftshift(ifft2(ifftshift(image_FT)))

            
            image_lr_update = np.sqrt(image) * np.exp(1j * np.angle(image_lr))
            image_lr_update *= nx_hr/nx_lr

            image_FT_update = fftshift(fft2(ifftshift(image_lr_update))) #* (1/pupil_func) #* CTF

            # Update fourier spectrum
            #mod_pupil = np.abs(pupil_func)**2  
            #weight_fac_pupil = np.conjugate(pupil_func) / (mod_pupil.max() + 1e-23)

            
            delta_lowres_ft = image_FT_update - image_FT 
            hr_fourier_image += delta_lowres_ft * 0.9 #* weight_fac_pupil  # FIX ME
            
            #mod_obj = np.abs(hr_fourier_image)**2
            #weight_factor_obj = np.conjugate(hr_fourier_image) / (mod_obj.max() + 1e-23)

            #weight_factor_obj = .1 #beta
            #pupil_func += weight_factor_obj * delta_lowres_ft

    # Reconstruct the high-resolution Fourier image
    hr_obj_image = fftshift(ifft2(ifftshift(hr_fourier_image)))
    print('done')
    return hr_obj_image, hr_fourier_image, pupil_func, CTF


def difference_map(images, kout_vec, pupil_func, lr_psize, num_iter=50):
    
    """
    Difference Map algorithm for phase retrieval.
    """
    
    object_estimate = np.ones_like(images[0], dtype=complex)
    beta = 0.9  # Relaxation parameter

    
    for _ in range(num_iter):
        for i, img in enumerate(images):
            fourier_estimate = fftshift(fft2(ifftshift(object_estimate * pupil_func)))
            constraint_fourier = np.sqrt(img) * np.exp(1j * np.angle(fourier_estimate))
            difference = fourier_estimate - constraint_fourier
            fourier_estimate -= beta * difference
            object_estimate = ifftshift(ifft2(fftshift(fourier_estimate))) * pupil_func
    return object_estimate, fftshift(fft2(ifftshift(object_estimate))), pupil_func
