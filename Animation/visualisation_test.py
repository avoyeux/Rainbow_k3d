"""
Just to save some small visualisation methods to check if an issue comes from the visualisation code or the data itself.
"""

# IMPORTS
import os
import k3d
import h5py
import typing
import scipy.optimize
import sparse
import scipy
# Aliases
import numpy as np
import matplotlib.pyplot as plt


class Visualise:
    """
    Parent class just to open the HDF5 data file.
    """

    def __init__(self,
                 hdf5_filepath: str,
                 group_paths: str | list[str],
                 chosen_index: int | list[int] = 0,
                 recreate_interp: bool = False,
                 nb_points: int = 10**3,
                 saving_plots: bool = False,
                 ) -> None:
        
        # Arguments
        self.filepath = hdf5_filepath
        self.group_paths = group_paths if isinstance(group_paths, list) else [group_paths]
        self.chosen_index = chosen_index 
        self.recreate_interpolation = recreate_interp
        self.nb_points = nb_points
        self.save_plots = saving_plots

        self.run()

    def run(self):
        # TODO: to open and get the necessary data

        with h5py.File(self.filepath, 'r') as H5PYFile:

            # Get the initial cubes and interpolation
            self.data = [
                self.get_COO(H5PYFile, path)[self.chosen_index]
                for path in self.group_paths  #TODO: this is wrong. need to change it so that it also work for the params           
            ]

            if self.recreate_interpolation:
                
                coords = self.data[0].coords.T.astype('float64')
                t = np.empty(coords.shape[0], dtype='float64')
                t[0] = 0
                for i in range(1, coords.shape[0]): t[i] = t[i - 1] + np.sqrt(np.sum([(coords[i, a] - coords[i - 1, a])**2 for a in range(3)]))
                t /= t[-1]  # normalisation 


                # t = np.linspace(0, 1, int(self.nb_points))
                self.parameters = [
                    H5PYFile[path + '/parameters'][...]
                    for path in self.group_paths
                    if 'interpolation' in path
                ]
                results = [None] * len(self.parameters)
                for a, slicing in enumerate(self.parameters):
                    mask = slicing[0, :] == self.chosen_index
                    result = slicing[:, mask]
                    x, y, z = [result[i] for i in range(1, 4)]
                    X = self.polynomial(t, *x)
                    Y = self.polynomial(t, *y)
                    Z = self.polynomial(t, *z)

                    cube = np.vstack([X, Y, Z])
                    # cube = np.rint(np.abs(cube.T)) # TODO: added a .T that needs taking off later
                    # cube = np.unique(cube, axis=0).T.astype('uint16')
                    # shape = np.max(cube, axis=1).astype('uint16') + 1
                    # print(shape)
                    # cube = sparse.COO(coords=cube, data=1, shape=shape)

                    coords = cube.T.astype('float64')
                    t = np.empty(coords.shape[0], dtype='float64')
                    t[0] = 0
                    for i in range(1, coords.shape[0]): t[i] = t[i - 1] + np.sqrt(np.sum([(coords[i, a] - coords[i - 1, a])**2 for a in range(3)]))
                    t /= t[-1]  # normalisation 
                    results[a] = (cube, t)
                self.polynomials = results

    def new_interp(self, x:np.ndarray, t) -> np.ndarray:
        #TODO: to redo the interpolation
        p0 = np.random.rand(7)
        params_x, _ = scipy.optimize.curve_fit(self.polynomial, t, x, p0=p0)

        print(f'the new coefficients are {[f"{param}, " for param in params_x]}')

        t_fine = np.linspace(0, 1, 10**4)
        x = self.polynomial(t_fine, *params_x)
        return x, t_fine

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
        last_len = len(self.data)
        for i, (poly, _) in enumerate(self.polynomials):
            plot += k3d.voxels(
                voxels=poly.todense(), # TODO: this is wrong right now, will need to change it
                color_map=[next(self.random_hexadecimal_colour_generator())],
                name=f'{last_len + i}',
                outlines=False
            )           
        plot.display()

    def polynomial(self, t: np.ndarray, *coeffs: float) -> np.ndarray:
        """
        The n-th order polynomial function

        Args:
            t (np.ndarray): an array representing the cumulative distance initially from 0 to 1.
            coeffs (float): the coeffs for the polynomial in the order a0 + a1 * t** + a2 * t**2 + etc.

        Returns:
            np.ndarray: the polynomial results for the given coeffs and t.
        """

        result = 0
        poly_order = len(coeffs) - 1
        # print(f'order is {poly_order}')
        for i in range(poly_order + 1): result += coeffs[i] * t**i
        return result
    
    def print_parameters(self):
        """
        To print the polynomial coefficients.
        """

        for a, slicing in enumerate(self.parameters):
            mask = slicing[0, :] == self.chosen_index
            result = slicing[:, mask]
            x, y, z = [result[i] for i in range(1, 4)]

            print(f'The saved x parameters are {", ".join([str(param) for param in x])}', flush=True)

    def plotting(self):
        """
        To plot the x, y, z values as a function of t.
        """

        for a, data in enumerate(self.data):
            if not 'interpolation' in self.group_paths[a]:

                coords = data.coords.T.astype('float64')
                print(f'the shape of coords after translation is {coords.shape}')
                t = np.empty(coords.shape[0], dtype='float64')
                t[0] = 0
                for i in range(1, coords.shape[0]): t[i] = t[i - 1] + np.sqrt(np.sum([(coords[i, a] - coords[i - 1, a])**2 for a in range(3)]))
                t /= t[-1]  # normalisation 

                # Polynomial
                x, y, z = self.polynomials[0][0]  #TODO: will need to change that laterS
                plot_distance = self.polynomials[0][1]
                # Cube points
                X, Y, Z = data.coords

                print(f'plot_distance shape is {plot_distance.shape} and x shape is {x.shape}')
                print(f'scatter distance shape is {t.shape} with X shape {X.shape}')
                
                # plt.figure()
                # plt.scatter(Y, Z)
                # plt.show()

                self.plotting_sub(yplot=('x', x), yscatter=('X', X), plot_distance=plot_distance, scatter_distance=t)
                self.plotting_sub(yplot=('y', y), yscatter=('Y', Y), plot_distance=plot_distance, scatter_distance=t)
                self.plotting_sub(yplot=('z', z), yscatter=('Z', Z), plot_distance=plot_distance, scatter_distance=t)
        
    def plotting_sub(
            self,
            plot_distance: np.ndarray,
            scatter_distance: np.ndarray,
            yplot: tuple[str, np.ndarray],
            yscatter: tuple[str, np.ndarray],
            title: str = '',
            ) -> None:
        
        plt.figure(figsize=(10, 10))
        if title != '': plt.title(title)

        if yplot[0]=='x':
            x2, t_fine = self.new_interp(yscatter[1], scatter_distance)
            plt.plot(t_fine, x2, color='orange')

        # # Labels
        # plt.xlabel('t')
        # plt.ylabel(yplot[0])

        # Plot
        plt.scatter(plot_distance, yplot[1], c='red')
        plt.scatter(plot_distance, yscatter[1], c='blue', s=0.5)  # TODO: this was scatter distance before

        # Visualise
        if self.save_plots:
            plt.savefig()
        else:
            plt.show()

        # plt.figure()
        # plt.plot(plot_distance, color='orange')
        # plt.show()


            


