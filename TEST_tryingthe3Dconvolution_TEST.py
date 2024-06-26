"""
Just testing to see if I can use the GPU to try to find the bary center using a 3D convolution.
Not sure if the thinking makes sense though.
"""


import os
import torch

import numpy as np

from sparse import COO
from typeguard import typechecked
from Animation_3D_main import Data as ParentClass



class Convolution3D(ParentClass):
    """
    To do the 3D convolution with PyTorch.
    """

    @typechecked
    def __init__(self, gaussian_mean: int | float, gaussian_std: int | float,
                 batch_size: int = 15, kernel_size: int = 31, **kwargs):

        self.batch_size = batch_size
        self.kernel_size = kernel_size
        self.mean = gaussian_mean
        self.std = gaussian_std
        super().__init__(**kwargs)

        self.Data_prepocessing()
        print(f'Data preprocessing done', flush=True)
        self.Structure()
        print(f'3D convolution finished.')

    def Data_prepocessing(self):
        """
        To change the sparse COO array to a Pytorch tensor object.
        """

        cubes = self.time_cubes_no_duplicate_new_2.todense()

        self.batches = [
            torch.tensor(cubes[i:i + self.batch_size]).float().unsqueeze(1)
            for i in range(0, cubes.shape[0], self.batch_size)
        ]
    
    def Gaussian_kernel(self, size: int, mean: int | float, std: int | float): 
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
        print('Using device: ', device, flush=True)
        
        gaussian_kernel = self.Gaussian_kernel(self.kernel_size, self.mean, self.std).to(device)
        gaussian_kernel = gaussian_kernel.unsqueeze(0).expand(1, 1, self.kernel_size, self.kernel_size, self.kernel_size)

        conv_outputs = []
        for loop, batch in enumerate(self.batches):
            input_tensor = batch.to(device)
            output = torch.nn.functional.conv3d(input=input_tensor, weight=gaussian_kernel, padding=self.kernel_size//2)
            min_val = output.min()
            max_val = output.max()
            normalized_output = (output - min_val) / (max_val - min_val)
            output = torch.nn.functional.conv3d(input=normalized_output, weight=gaussian_kernel, padding=self.kernel_size//2)
            min_val = output.min()
            max_val = output.max()
            normalized_output = (output - min_val) / (max_val - min_val)
            conv_outputs.append(normalized_output.cpu())
            print(f'batch nb {loop} done.', flush=True)

        concatenated_output = torch.cat(conv_outputs, dim=0)
        output_cpu = concatenated_output.numpy()
        output_cpu = np.squeeze(output_cpu, axis=1)
        output_cpu = (output_cpu * 255).astype('uint8')

        np.save(os.path.join(os.getcwd(), f'barycenterarray_mean{self.mean}_std{self.std}_kernel{self.kernel_size}.npy'), output_cpu)
        print('Saving of files finished', flush=True)


if __name__=='__main__':
    test = Convolution3D(batch_size=15, kernel_size=31, gaussian_mean=31//2, gaussian_std=31/18,
                         time_interval='24h', time_intervals_no_duplicate=True, cube_version = 'new')
