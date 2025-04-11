import numpy as np 
import torch
import torch.nn.functional as F 

class ShrinkWrap():
    """
    Performs the shrinkwrap method for a given threshold and sigma values
    Works for 2D data
    Data is a torch tensor 
    """
    def __init__(self, data, sigma, threshold, kernel_size, device):        
        self.threshold = threshold
        self.sigma = sigma
        self.kernel_size = kernel_size
        self.device = device

        if not isinstance(data, torch.Tensor):

            self.data=torch.tensor(data)
        else: 
            self.data = data
        
        self.shp = self.data.size()

    def gaussian_fill(self):
    
        # Define Gaussian kernel
        self.kernel_size = int(4*self.sigma+1)
        sigma = self.sigma
        x = torch.arange(self.kernel_size, dtype=torch.float32) - self.kernel_size // 2
        self.kernel = torch.exp(-0.5 * (x ** 2) / sigma ** 2).to(self.device)
        self.kernel = self.kernel / torch.sum(self.kernel)
        self.kernel = self.kernel.view(1, self.kernel_size).repeat(self.kernel_size, 1)
        
        self.kernel = self.kernel.unsqueeze(0).unsqueeze(0)
        
        self.kernel = self.kernel
    
    
    def compute_support(self):
        
        self.support = torch.abs(self.data).clone()
        maxvalue = torch.abs(self.support).max()
        self.support = torch.where(self.support<(self.threshold*maxvalue), 0, 1)
        self.support = self.support.float()
        self.support = F.conv2d(self.support.unsqueeze(0).unsqueeze(0), self.kernel, padding=self.kernel_size//2)
        self.support = torch.where(self.support<self.threshold, 0, 1)
        self.support = self.support.squeeze(0).squeeze(0)
    
    def get(self):
    
        self.gaussian_fill()
        self.compute_support()
        self.support = self.support.to(self.device)
        
        return self.support
        

class WeightScheduler:
    def __init__(self, 
                 start_epoch=0, 
                 increase_epochs=10, 
                 plateau_epochs=20,
                 alpha_max=1.0, 
                 gamma=0.99, 
                 min_alpha=0.0):
        """
        TV weight scheduler with ramp-up, plateau, and decay phases.

        Args:
            start_epoch (int): Epoch to start increasing TV weight.
            increase_epochs (int): Number of epochs to ramp up TV weight.
            plateau_epochs (int): Number of epochs to hold alpha_max.
            alpha_max (float): Maximum TV weight value.
            gamma (float): Decay factor per epoch after plateau.
            min_alpha (float): Minimum TV weight.
        """
        self.start_epoch = start_epoch
        self.increase_epochs = increase_epochs
        self.plateau_epochs = plateau_epochs
        self.alpha_max = alpha_max
        self.gamma = gamma
        self.min_alpha = min_alpha

        # Precompute when decay starts
        self.decay_start_epoch = start_epoch + increase_epochs + plateau_epochs

    def get_alpha(self, epoch):
        if epoch < self.start_epoch:
            return 0.0
        
        # Ramp-up phase
        if epoch < self.start_epoch + self.increase_epochs:
            progress = (epoch - self.start_epoch) / self.increase_epochs
            return progress * self.alpha_max
        
        # Plateau phase
        if epoch < self.decay_start_epoch:
            return self.alpha_max
        
        # Decay phase
        decay_epochs = epoch - self.decay_start_epoch
        decayed_alpha = self.alpha_max * (self.gamma ** decay_epochs)
        return max(decayed_alpha, self.min_alpha)