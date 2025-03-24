import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, clear_output


def plot_roi_from_numpy(array, roi=None, name=None, vmin=None, vmax=None, save = False, **kwargs):

    """
    Plots a region of interest from a 2D numpy array

    input: 
        array : numpy array (2D or 3D)
        roi : list, Region of interest, [min vertical axis, max vertical axis, min horizontal axis, max horizontal axis] 
        name: str, name to be used for title and saving the figure
        vmin: float, color map minimum
        vmax: float, color map maximum
        save: bool, if True, saves the plot
        frame: int, frame number if array is 3D 
    """
    if roi is None:
        roi = [0,-1,0,-1]

    if name is None:
        name = "Plot"

    # Plot the pupil
    if len(array.shape) == 3 and 'frame' in kwargs:
        frame = kwargs['frame']
        data_roi = array[frame, roi[0]:roi[1], roi[2]:roi[3]]
    elif len(array.shape) == 2:
        data_roi = array[roi[0]:roi[1], roi[2]:roi[3]]
    else:
        raise ValueError("Either pass in a 2D array or a 3D array with 'frame' in the kwargs.")

    if vmin is None:
        me = np.mean(data_roi) 
        vmin = me - 0.5 *me
    if vmax is None:
        me = np.mean(data_roi) 
        vmax = me + 0.5 *me
        
    plt.figure()
    plt.imshow(data_roi, vmin = vmin, vmax = vmax,cmap='viridis')
    plt.title(name)
    plt.colorbar()
    plt.show()
    
    if save:
        plt.savefig(name)
        
def plot_images_side_by_side(image1, image2, 
                             vmin1= None, vmax1=None, 
                             vmin2= None, vmax2=None, 
                             title1="Image 1", title2="Image 2", cmap1="gray", cmap2="gray", figsize=(10, 5), show = False, save_fname = None):
    """
    Plots two images side by side.

    Parameters:
    - image1: First image (2D numpy array).
    - image2: Second image (2D numpy array).
    - title1: Title for the first image (default: "Image 1").
    - title2: Title for the second image (default: "Image 2").
    - cmap1: Colormap for the first image (default: "gray").
    - cmap2: Colormap for the second image (default: "gray").
    - figsize: Size of the figure (default: (10, 5)).
    """
    # Create a figure and subplots
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    
    if vmin1 is None:
        me = np.mean(image1) 
        vmin1 = me - 0.5 *me
    if vmax1 is None:
        me = np.mean(image1) 
        vmax1 = me + 0.5 *me
    
    if vmin2 is None:
        me = np.mean(image2) 
        vmin2 = me - 0.5 *me
    if vmax2 is None:
        me = np.mean(image2) 
        vmax2 = me + 0.5 *me
    
    # Plot the first image
    im1 = axes[0].imshow(image1, vmin = vmin1, vmax = vmax1, cmap=cmap1)    
    axes[0].set_title(title1)
    #axes[0].axis('off')  # Hide axes
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    
    # Plot the second image
    im2 = axes[1].imshow(image2, vmin = vmin2, vmax = vmax2, cmap=cmap2)
    axes[1].set_title(title2)
    #axes[1].axis('off')  # Hide axes
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Adjust layout and display
    plt.tight_layout()
    if show:
        plt.show()

    if save_fname is not None:
        plt.savefig(save_fname)
        
def initialize_live_plot(hr_obj_image, hr_fourier_obj):
    """
    Initializes the live plot with two subplots: one for amplitude and one for phase.
    
    Returns:
        fig, ax: Matplotlib figure and axes.
        img_amp, img_phase: Image objects for real-time updates.
    """

    # Initialize empty images
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            
    # Initialize the plots with the initial image
    img_amp = axes[0].imshow(np.abs(hr_obj_image), vmin =.2, vmax = 1, cmap='viridis')
    axes[0].set_title("Object Object")
    cbar_amp = plt.colorbar(img_amp, ax=axes[0])

    img_phase = axes[1].imshow(np.abs(hr_fourier_obj), cmap='viridis')
    axes[1].set_title("Fourier Amplitude")
    cbar_phase = plt.colorbar(img_phase, ax=axes[1])

    plt.tight_layout()
    plt.ion()  # Enable interactive mode
    plt.show()

    return fig, axes, img_amp, img_phase

def update_live_plot(img_amp, img_phase, hr_obj_image, hr_fourier_obj, fig):
    """
    Updates the live plot with new amplitude and phase images.

    Args:
        img_amp: Matplotlib image object for amplitude.
        img_phase: Matplotlib image object for phase.
        hr_obj_image: The complex object image to be plotted.
    """
    amplitude_obj = np.abs(hr_obj_image)
    amplitude_ft = np.abs(hr_fourier_obj)
    
    img_amp.set_data(amplitude_obj)  # Normalize for visibility
    img_phase.set_data(amplitude_ft)

    amp_mean = np.mean(amplitude_obj)
    vmin = max(amp_mean - 0.1 * amp_mean, 0)
    vmax = amp_mean + 2 * amp_mean
    img_amp.set_clim(vmin, vmax)

    ft_mean = np.mean(amplitude_ft)
    vmin = ft_mean - 0.1 * ft_mean
    vmax = ft_mean + 2 * ft_mean
    img_phase.set_clim(vmin, vmax)
    
    clear_output(wait=True)
    display(fig)
    fig.canvas.flush_events()