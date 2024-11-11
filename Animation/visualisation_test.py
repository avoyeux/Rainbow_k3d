"""
Just to save some small visualisation methods to check if an issue comes from the visualisation code or the data itself.
"""

# IMPORTS
import re
import os
import k3d
import h5py
import scipy
import typing
import sparse
import typeguard
import scipy.optimize

# Aliases
import numpy as np
import matplotlib.pyplot as plt

# Personal imports
from common import Plot

#TODO: change this code so that the plots automatically shows more info about the data that was chosen.
#TODO: also add the correct interpolation order as a new argument.

class Visualise:
    """
    Parent class just to open the HDF5 data file.
    """

    @typeguard.typechecked
    def __init__(
            self,
            hdf5_filepath: str,
            chosen_index: int | list[int] = 0,
            recreate_interp: bool = True,
            with_feet: bool | str = True,
            interpolation_order: int | list[int] = [4, 6],
            nb_points: int = 10**3,
            saving_plots: bool = False,
            axes_order: list[int] = [2, 1, 0],
    ) -> None:
        """ #TODO: update docstring
        To test the visualisation given only one index index in the data gotten from the HDF5 file.

        Args:
            hdf5_filepath (str): the filepath to the data HDF5 file.
            group_paths (str | list[str]): the HDF5 path to the data that needs to be visualised. 
            chosen_index (int | list[int], optional): the index of the cube that the user wants to visualise. Defaults to 0.
            recreate_interp (bool, optional): deciding to recreate the interpolation on the run. Mainly to compare it 
                to the polynomial gotten directly from the HDF5 file. Defaults to False.
            nb_points (int, optional): the number of points used in the recreation of the polynomial function.
                Defaults to 10**3.
            saving_plots (bool, optional): Deciding to save the 2D plots. Defaults to False.
            axes_order (tuple[int, ...] | None, optional): the order of the axes (compared to (t, x, y, z)) when the
                polynomial was created as it is important in the creation of the polynomial. If the HDF5 filename follows the 
                default pattern (e.g. order0321.h5), this argument doesn't need to be set. Defaults to None.
        """
        
        # Arguments
        self.axes_order = axes_order
        self.filepath = hdf5_filepath
        self.chosen_index = chosen_index 
        self.recreate_interpolation = recreate_interp
        self.nb_points = nb_points
        self.save_plots = saving_plots

        # Group path setup
        if isinstance(with_feet, str):
            feet = ['', ' with feet']
        else:
            feet = [' with feet'] if with_feet else ['']
        self.interpolation_order = interpolation_order if isinstance(interpolation_order, list) else [interpolation_order]
        self.group_paths = [
            f"Time integrated/No duplicates new{option}/Time integration of 24.0 hours{added}"
            for option in feet
            for order in self.interpolation_order
            for added in ['', f'/{order}th order interpolation']
        ]

        # Instance functions
        self.filename_info()

        self.run()

    def filename_info(self):
        #TODO: to get information on the file directly from the filename

        filename_pattern = re.compile(r'''
            sig(?P<feet_sigma>\d{1}e\d+)_
            leg(?P<leg_sigma>\d+)_
            lim(?P<leg_position>\d{1}_\d+)
            (_thisone)?.h5
        ''', re.VERBOSE)
        
        filename_match = filename_pattern.search(os.path.basename(self.filepath))
            
        if filename_match is None:
            self.filename_info = None
            print(f"Filename pattern doesn't match with filename {os.path.basename(self.filepath)}. No data information gotten from the filename")
        else:
            filename_info = filename_match.groupdict()
            filename_info['feet_sigma'] = float('e-'.join(filename_info['feet_sigma'].split('e')))
            filename_info['leg_position'] = float('.'.join(filename_info['leg_position'].split('_')))
            self.filename_info = filename_info

            print("Data information is " + '\n   '.join([f'{key}: {filename_info[key]}' for key in filename_info.keys()]))

    def run(self) -> None:
        """
        To open and get the necessary data from the HDF5 file.
        """

        with h5py.File(self.filepath, 'r') as H5PYFile:

            # Get the initial cubes and interpolation
            self.data = [
                self.get_COO(H5PYFile, path)[self.chosen_index]
                for path in self.group_paths       
            ]

            if self.recreate_interpolation:
                
                coords = self.data[0].coords.T.astype('float64')

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

                    t_fine = np.linspace(0, 1, coords.shape[0])
                    X = self.polynomial(t_fine, *x)
                    Y = self.polynomial(t_fine, *y)
                    Z = self.polynomial(t_fine, *z)

                    cube = np.vstack([X, Y, Z])
                    results[a] = cube
                self.polynomials = results

    def new_interp(self, x: np.ndarray, t: np.ndarray, order: int) -> np.ndarray:
        """
        To fit an array given the data points to fit and the array for which the data points are a function of.

        Args:
            x (np.ndarray): the data points.
            t (np.ndarray): the array for which the data points are a function of (i.e. we have x(t)).

        Returns:
            np.ndarray: the fitted array.
        """

        p0 = np.random.rand(order + 1)
        params_x, _ = scipy.optimize.curve_fit(self.polynomial, t, x, p0=p0)

        # print(f'the new coefficients are {[f"{param}, " for param in params_x]}')

        t_fine = np.linspace(0, 1, x.shape[0])
        x = self.polynomial(t_fine, *params_x)
        return x

    def get_COO(self, H5PYFile: h5py.File, group_path: str) -> sparse.COO:
        """
        To get the sparse.COO object from the corresponding coords and values.

        Args:
            H5PYFile (h5py.File): the file object.
            group_path (str): the path to the group where the data is stored.

        Returns:
            sparse.COO: the corresponding sparse data.
        """

        # Get data
        data_coords = H5PYFile[group_path + '/coords'][...]
        data_data = H5PYFile[group_path + '/values'][...] if not 'interpolation' in group_path else 1
        shape = np.max(data_coords, axis=1) + 1
        return sparse.COO(coords=data_coords, data=data_data, shape=shape)
        
    def visualisation(self) -> None:
        """
        For the 3D visualisation of the data points and the corresponding interpolations.
        """

        # print(f'The len of data is {len(self.data)}')
        
        plot = k3d.plot()
        
        for i, data in enumerate(self.data):
            plot += k3d.voxels(
                # positions = data.coords,
                voxels=data.todense(),
                color_map=[next(self.random_hexadecimal_colour_generator())],
                name=f'{i}',
                outlines=False
            )
        last_len = len(self.data)
        if self.recreate_interpolation:
            for i, poly in enumerate(self.polynomials):
                plot += k3d.voxels(
                    # positions = poly[:, ::100].astype('float32'),
                    voxels=self.visualisation_sub(poly).todense(), # TODO: this is wrong right now, will need to change it
                    color_map=[next(self.random_hexadecimal_colour_generator())],
                    name=f'{last_len + i}',
                    outlines=False
                )           
        plot.display()

    def visualisation_sub(self, data: np.ndarray) -> sparse.COO:
        """
        To reformat a coordinates array to a sparse.COO dense array. 

        Args:
            data (np.ndarray): the coordinates array to reformat to a sparse.COO array.

        Returns:
            sparse.COO: the corresponding sparse.COO array.
        """

        data = np.rint(np.abs(data.T))
        data = np.unique(data, axis=0).T.astype('uint16')
        shape = np.max(data, axis=1) + 1
        # print(f'the shape for the polynomial visualisation is {shape}')
        return sparse.COO(coords=data, shape=shape, data=1)

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
    
    def print_parameters(self) -> None:
        """
        To print the polynomial coefficients.
        """

        for slicing in self.parameters:
            mask = slicing[0, :] == self.chosen_index
            result = slicing[:, mask]
            x, y, z = [result[i] for i in range(1, 4)]

            # print(f'The saved x parameters are {", ".join([str(param) for param in x])}', flush=True)

    def reorder_data(self, data: sparse.COO) -> sparse.COO:
        """
        To reorder a sparse.COO array so that the axes orders change. This is done to change which axis is 'sorted', as the first axis is always 
        sorted (think about the .ravel() function).

        Args:
            data (sparse.COO): the array to be reordered, i.e. swapping the axes ordering.

        Returns:
            sparse.COO: the reordered sparse.COO array.
        """

        new_coords = data.coords[self.axes_order]
        new_shape = [data.shape[i] for i in self.axes_order]
        return sparse.COO(coords=new_coords, data=1, shape=new_shape)  # TODO: this doesn't take into account the values
    
    def plotting(self) -> None:
        """
        To plot the x, y, z values as a function of t.
        """

        c = -1
        for a, data in enumerate(self.data):
            if not 'interpolation' in self.group_paths[a]:
                c += 1

                # reordering the data
                reordered_data = self.reorder_data(data)
                coords = reordered_data.coords.T.astype('float64')
                t = np.empty(coords.shape[0], dtype='float64')
                t[0] = 0
                for i in range(1, coords.shape[0]): t[i] = t[i - 1] + np.sqrt(np.sum([(coords[i, a] - coords[i - 1, a])**2 for a in range(3)]))
                t /= t[-1]  # normalisation 

                t_fine = np.linspace(0, 1, coords.shape[0])

                # Polynomial
                x, y, z = self.polynomials[c]
                # Cube points
                X, Y, Z = data.coords

                #Testing reordering 
                # You have to reorder as you need the same t for the plots to work as each value of t represents a specific point.
                data = self.reorder_data(data)
                X, Y, Z = data.coords
                x, y, z = [(('x', x), ('y', y), ('z', z))[i] for i in self.axes_order]
                
                # plt.figure()
                # plt.scatter(Y, Z)
                # plt.show()
                order = self.interpolation_order[c]
                title = f'For interpolation {order}th order.'
                self.plotting_sub(yplot=x, yscatter=('X', X), t=t, t_fine=t_fine, order=order, title=title)
                self.plotting_sub(yplot=y, yscatter=('Y', Y), t=t, t_fine=t_fine, order=order)
                self.plotting_sub(yplot=z, yscatter=('Z', Z), t=t, t_fine=t_fine, order=order)
        
    def plotting_sub(
            self,
            t: np.ndarray,
            t_fine: np.ndarray,
            order: int,
            yplot: tuple[str, np.ndarray],
            yscatter: tuple[str, np.ndarray],
            title: str = '',
        ) -> None:
        """
        To generate plots given the corresponding arrays.

        Args:
            t (np.ndarray): the cumulative distance array.
            t_fine (np.ndarray): the uniform cumulative distance array.
            yplot (tuple[str, np.ndarray]): the name and array for the polynomials gotten from the HDF5 file.
            yscatter (tuple[str, np.ndarray]): the name and data points for each axis.
            title (str, optional): the title of the plot. Defaults to ''.
        """
        
        plt.figure(figsize=(10, 4))
        if title != '': plt.title(title)
        plt.scatter(t[::30], yscatter[1][::30], c='blue', s=0.3, label=f'Data points for {yplot[0]}-axis', zorder=1)  # TODO: this was scatter distance before
        x2 = self.new_interp(yscatter[1], t, order)
        plt.scatter(t_fine, x2, color='orange', label=f'{order}th order on the go', s=1.3, zorder=2)

        # # Labels
        # plt.xlabel('t')
        # plt.ylabel(yplot[0])

        # Plot
        plt.scatter(t_fine, yplot[1], c='red', label=f'{order}th order polynomial from file', s=1.0, zorder=3)
        plt.axvline(
            x=self.filename_info['leg_position'],
            color='purple',
            linestyle='--',
            linewidth=1,
            label=f"Weight limit of {self.filename_info['leg_position']}",
            zorder=4,
        )
        
        plt.legend()

        # Visualise
        if self.save_plots:
            plt.savefig()
        else:
            plt.show()

        # plt.figure()
        # plt.plot(plot_distance, color='orange')
        # plt.show()


            


