"""
Just to save some small visualisation methods to check if an issue comes from the visualisation code or the data itself.
"""

# IMPORTS
import os
import k3d
import h5py
import typing
import sparse
# Aliases
import numpy as np
import matplotlib.pyplot as plt


class OpenData:
    """
    Parent class just to open the HDF5 data file.
    """

    def __init__(self,
                 hdf5_filepath: str,
                 group_paths: str | list[str],
                 chosen_index: int | list[int] = 0
                 ) -> None:
        
        # Arguments
        self.filepath = hdf5_filepath
        self.group_paths = group_paths if isinstance(group_paths, list) else [group_paths]
        self.chosen_index = chosen_index 

    def run(self):
        # TODO: to open and get the necessary data

        with h5py.File(self.filepath, 'r') as H5PYFile:

            # Get all data
            self.data = self.get_data(H5PYFile)

class Visualise(OpenData):
    """
    To visualise the data given a simple function. Works for the cubes and the interpolations.

    Args:
        OpenData (_type_): the parent class that just opens the HDF5 file and gets the corresponding data (saved in self.data).
    """


    def __init__(self, 
                 **kwargs
                 ) -> None:
        
        # Open file
        super().__init__(**kwargs)

        # Run
        self.run()

    def get_data(self, H5PYFile: h5py.File) -> list[sparse.COO]:
        # TODO: to choose the right data
        data = [
                self.get_COO(H5PYFile, path)[self.chosen_index]
                for path in self.group_paths  #TODO: this is wrong. need to change it so that it also work for the params           
        ]
        return data

    def get_COO(self, H5PYFile: h5py.File, group_path: str) -> sparse.COO:
        """
        To get the sparse.COO object from the corresponding coords and values.

        Args:
            H5PYFile (h5py.File): the file object.
            group_path (str): the path to the group where the data is stored.

        Returns:
            sparse.COO: the corresponding sparse data.
        """

        print(group_path)
        if not 'interpolation' in group_path:
            data_coords = H5PYFile[group_path + '/coords'][...]
            data_data = H5PYFile[group_path + '/values'][...]
            shape = np.max(data_coords, axis=1) + 1
            return sparse.COO(coords=data_coords, data=data_data, shape=shape)
        else:
            data = H5PYFile[group_path + '/treated coords'][...]
            shape = np.max(data, axis=1) + 1
            return sparse.COO(coords=data, data=1, shape=shape)
        
    def random_hexadecimal_colour_generator(self) -> typing.Generator[int, None, None]:
        """
        Generator that yields a random colour value in integer hexadecimal code format.

        Returns:
            typing.Generator[int, None, None]: A generator yielding integer hexadecimal colour codes.

        Yields:
            int:  A random colour in hexadecimal code as an integer.
        """

        while True: yield np.random.randint(0, 0xffffff)
        
    def visualisation(self):
        # TODO: the small visualisation
        
        plot = k3d.plot()
        
        for i, data in enumerate(self.data):
            plot += k3d.voxels(
                voxels=data.todense(),
                color_map=[next(self.random_hexadecimal_colour_generator())],
                name=f'{i}',
                outlines=False
            )
        
        plot.display()

class Params(OpenData):
    #TODO: to get the polynomial parameters.
    

    def __init__(self, 
               **kwargs
               ) -> None:
        
        # Open file
        super().__init__(**kwargs)

        self.run()

    def get_data(self, H5PYFile: h5py.File) -> np.ndarray:
        #TODO: to open the right data

        data = np.stack([
            H5PYFile[path + '/parameters'][...]
            for path in self.group_paths
        ], axis=0)

        result = [None] * len(data)
        for i, slicing in enumerate(data):
            mask = slicing[0, :] == self.chosen_index
            result[i] = slicing[:, mask]
        data = np.stack(result, axis=0)
        print(f'data shape is {data.shape}')
        print(data)
        # data_filter = np.all(np.round(data[:, 0, :]) == self.chosen_index, axis=1)
        # print(data_filter.shape)
        # filtera = data[:, 0, :] == self.chosen_index
        # print(data[filtera, :].shape)
        return data

    def polynomial(self, t: np.ndarray, *coeffs: float) -> np.ndarray:
        
        result = 0
        poly_order = len(coeffs) - 1
        for i in range(poly_order + 1): result += coeffs[i] * t**i
        return result
    
    def print_parameters(self):
        # TODO: to print the parameters so that I can compare them 

        for i, path in enumerate(self.group_paths):
            parameters = self.data[i]
            print(f'the shape of the parameters are {parameters.shape}')
            print(f'For path {os.path.basename(path)} x parameters are {parameters[1]}', flush=True)

    def plotting(self):
        # TODO: for the plotting

        points_nb = 1e4
        t = np.linspace(0, 1, int(points_nb))

        for cube in self.data:
            print(cube.shape)

            x, y, z = [cube[i] for i in range(1, 4)]
            X = self.polynomial(t, *x)
            Y = self.polynomial(t, *y)
            Z = self.polynomial(t, *z)


            plt.plot(t, X)
            plt.xlabel('t')
            plt.ylabel('X')
            plt.show()
            plt.figure()
            plt.plot(t, Y)
            plt.show()
            plt.figure()
            plt.plot(t, Z)
            plt.show()
            break
        
    def plotting_sub(self, ):


    def visualisation(self):
        points_nb = 1e3
        t = np.linspace(0, 1, int(points_nb))
        for cube in self.data:
            x, y, z = [cube[i] for i in range(1, 4)]
            X = self.polynomial(t, *x)
            Y = self.polynomial(t, *y)
            Z = self.polynomial(t, *z)

            cube = np.vstack([X, Y, Z])
            cube = np.rint(np.abs(cube.T))
            cube = np.unique(cube, axis=0).T.astype('uint16')
            shape = np.max(cube, axis=1) + 1
            print(f'cube shape is {cube.shape}')
            print(f'shape is {shape}')
            cube = sparse.COO(coords=cube, data=1, shape=shape).todense()
            plot = k3d.plot()
            plot += k3d.voxels(voxels=cube)
            plot.display()
            break
            


