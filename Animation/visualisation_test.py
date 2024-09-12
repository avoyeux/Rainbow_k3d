"""
Just to save some small visualisation methods to check if an issue comes from the visualisation code or the data itself.
"""

# IMPORTS
import k3d
import h5py
import typing
import sparse
# Aliases
import numpy as np



class testing:
    # TODO: for test on a single cube with or without interpolations.


    def __init__(self, 
                 hdf5_filepath: str,
                 group_paths: str | list[str],
                 chosen_index: int = 0,
                 ) -> None:
        
        self.filename = hdf5_filepath
        self.group_paths = group_paths if isinstance(group_paths, list) else [group_paths]
        self.chosen_index = chosen_index

        # Run
        self.get_data()
        self.visualisation()

    def get_data(self):
        # TODO: to open and get the necessary data

        with h5py.File(self.filename, 'r') as H5PYFile:

            # Get all data
            self.data = [
                self.get_COO(H5PYFile, path)[self.chosen_index]
                for path in self.group_paths
            ]

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
