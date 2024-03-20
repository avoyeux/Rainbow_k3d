"""
Just testing to see if I can use the GPU to try to find the bary center using a 3D convolution.
Not sure if the thinking makes sense though.
"""


import os
import torch

import numpy as np

from sparse import COO
from Animation_3D_main import Data as ParentClass



class Convolution3D(ParentClass):
    """
    To do the 3D convolution with PyTorch.
    """

    def __init__(self, gaussian_mean: int | float, gaussian_std: int | float,
                 batch_size: int = 15, kernel_size: int = 20, **kwargs):

        self.batch_size = batch_size
        self.kernel_size = kernel_size
        self.mean = gaussian_mean
        self.std = gaussian_std
        super().__init__(**kwargs)

        self.Data_prepocessing()
        print(f'Data preprocessing done')
        self.Structure()
        print(f'3D convolution finished.')

    def Data_prepocessing(self):
        """
        To change the sparse COO array to a Pytorch tensor object.
        """

        cubes = self.time_cubes_no_duplicate_new_2.todense()

        self.batches = [
            torch.tensor(cubes[i:i + self.batch_size]).float()
            for i in range(0, cubes.shape[0], self.batch_size)
        ]
    
    def Gaussian_kernel(size: int, mean: int | float, std: int | float): 
        """
        Generating a 3D gaussian kernel.
        """

        grid = torch.meshgrid([torch.arange(size) for _ in range(3)], indexing='ij')
        grid = torch.stack(grid, dim=-1).float()
        kernel = torch.exp(-((grid - mean)**2).sum(axis=-1) / (2 * std**2))
        return kernel / kernel.sum()

    def Structure(self):
        """
        Main structure for the convolution code.
        """

        # Checking if the GPU was set up properly
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Using device: ', device)
        
        gaussian_kernel = self.Gaussian_kernel(self.kernel_size, self.mean, self.std).to(device)
        gaussian_kernel = gaussian_kernel.expand(self.cubes_shape[1], 1, self.kernel_size, self.kernel_size, self.kernel_size)

        conv_outputs = []
        for batch in self.batches:
            input_tensor = batch.to(device)
            output = torch.nn.functional.conv3d(input=input_tensor, weight=gaussian_kernel, padding=self.kernel_size//2)
            conv_outputs.append(output.cpu())

        concatenated_output = torch.cat(conv_outputs, dim=0)
        output_cpu = concatenated_output.numpy()

        sparse_output = COO.from_numpy(output_cpu)
        np.save('barycenter_sparse_array.npy', output_cpu)


if __name__=='__main__':
    test = Convolution3D(batch_size=15, kernel_size=30, gaussian_mean=30//2, gaussian_std=30/6,
                         time_interval='24h', time_intervals_no_duplicate=True, cube_version = 'new')
