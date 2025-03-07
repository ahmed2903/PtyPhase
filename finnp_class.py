import numpy as np 
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.fft as fft


class FourierPtychographyNet(nn.Module):
    def __init__(self, img_size):
        super(FourierPtychographyNet, self).__init__()
        
        self.img_size = img_size
        
        # Initial guess: upsampled low-resolution image spectrum
        self.spectrum_amp = nn.Parameter(torch.randn(img_size, img_size, dtype=torch.float32))
        self.spectrum_pha = nn.Parameter(torch.randn(img_size, img_size, dtype=torch.float32))
        
        
        self.pupil_amp = nn.Parameter(torch.randn(img_size, img_size, dtype=torch.float32))
        self.pupil_pha = nn.Parameter(torch.randn(img_size, img_size, dtype=torch.float32))
        

    def forward(self, CTF):
        """ Forward propagation: reconstruct low-resolution complex field """
        # Create complex spectrum and pupil from real and imaginary parts
        spectrum = self.spectrum_amp * torch.exp(1j * self.spectrum_pha)
        
        pupil = self.pupil_amp * torch.exp(1j * self.pupil_pha)
        
        pupil *= CTF 
        
        
        # Apply pupil function (acting as a low-pass filter)
        filtered_spectrum = spectrum * pupil

        # Inverse Fourier Transform to reconstruct low-resolution image
        low_res_image = fft.shift(fft.ifft2(fft.ifftshift(filtered_spectrum)))
        
        return low_res_image
    
    
# Define loss function and optimizer
def train_fourier_ptychography(target_image, num_epochs=500, lr=0.01):
    img_size = target_image.shape[-1]
    
    # Model instance
    model = FourierPtychographyNet(img_size)
    
    # L2 Loss Function
    loss_fn = nn.MSELoss()
    
    # Nesterov-accelerated Gradient Descent (NAG)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True)
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        # Forward pass
        reconstructed_image = model()
        
        # Compute loss
        loss = loss_fn(reconstructed_image, target_image)
        
        # Backpropagation
        loss.backward()
        optimizer.step()
        
        # Logging
        if epoch % 50 == 0:
            print(f"Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.6f}")

    return model

def train_fourier_ptychography(model, target_images, CTFs, num_epochs=500, lr=0.01):
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

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        total_loss = 0  # Accumulate loss over all measurements

        # Iterate over all measurements (different illuminations)
        for i in range(len(target_images)):
            
            CTF = CTFs[i]
            target_image = target_images[i]

            # Forward pass for the current measurement
            reconstructed_image = model(CTF)

            # Compute loss for this measurement
            loss = loss_fn(reconstructed_image, target_image)
            total_loss += loss  # Accumulate loss
        
        # Backpropagation
        total_loss.backward()
        optimizer.step()

        # Logging
        if epoch % 50 == 0:
            print(f"Epoch [{epoch}/{num_epochs}], Loss: {total_loss.item():.6f}")

    return model
