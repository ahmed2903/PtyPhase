import numpy as np 
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.fft as fft


class FourierPtychographyNet(nn.Module):
    def __init__(self, spectrum_size, pupil_size):
        super(FourierPtychographyNet, self).__init__()
        
        self.spectrum_size = spectrum_size
        self.pupil_size = pupil_size

        # Initial guess: upsampled low-resolution image spectrum
        self.spectrum_amp = nn.Parameter(torch.ones(spectrum_size[0], spectrum_size[1], dtype=torch.float32))
        self.spectrum_pha = nn.Parameter(torch.zeros(spectrum_size[0], spectrum_size[1], dtype=torch.float32))
        
        
        self.pupil_amp = nn.Parameter(torch.ones(pupil_size[0], pupil_size[1], dtype=torch.float32))
        self.pupil_pha = nn.Parameter(torch.zeros(pupil_size[0], pupil_size[1], dtype=torch.float32))
        

    def forward(self):
        """ Forward propagation: reconstruct low-resolution complex field """
        # Create complex spectrum and pupil from real and imaginary parts
        spectrum = self.spectrum_amp * torch.exp(1j * self.spectrum_pha)
        
        pupil = self.pupil_amp * torch.exp(1j * self.pupil_pha)
        
        #pupil *= CTF 
        
        
        # Apply pupil function (acting as a low-pass filter)
        filtered_spectrum = spectrum * pupil

        # Inverse Fourier Transform to reconstruct low-resolution image
        low_res_image = fft.fftshift(fft.ifft2(fft.ifftshift(filtered_spectrum)))
        
        return low_res_image
    
    

def train_fourier_ptychography(model, target_images, num_epochs=500, lr=0.01):
    """
    Train the Fourier Ptychography network over multiple measurements.
    
    Parameters:
    - model: Instance of FourierPtychographyNet
    - target_images: List of measured low-resolution images
    - pupil_amps: List of corresponding pupil function amplitudes
    - pupil_phas: List of corresponding pupil function phases
    - num_epochs: Number of epochs to train
    - lr: Learning rate
    """
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True)
    loss_fn = nn.MSELoss()

    target_images = np.array(target_images)
    
    # Ensure target_image is a tensor and has the same size as reconstructed_image
    if not isinstance(target_images, torch.Tensor):
        target_images = torch.tensor(target_images, dtype=torch.float32)
    
    # If target_image is not complex, convert it to complex
    if not target_images.is_complex():
        target_images = target_images.to(dtype=torch.complex64)
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        total_loss = 0  # Accumulate loss over all measurements

        # Iterate over all measurements (different illuminations)
        for i in range(len(target_images)):
            
            #CTF = CTFs[i]
            target_image = target_images[i,:,:]

            # Forward pass for the current measurement
            reconstructed_image = model()

            # Compute loss for this measurement
            loss = loss_fn(torch.abs(reconstructed_image), torch.abs(target_image))
            total_loss += loss  # Accumulate loss
        
        # Backpropagation
        total_loss.backward()
        optimizer.step()

        # Logging
        if epoch % 5 == 0:
            print(f"Epoch [{epoch}/{num_epochs}], Loss: {total_loss.item():.6f}")

    return model
