"""
To add an HDF5 group with fake data to be able to visually test codes using the HDF5 protuberance
data.
"""

# IMPORTs
import os
import h5py

# IMPORTs sub
import numpy as np

# IMPORTs personal
from Data.Cubes import DataSaver
from common import Decorators



class AddTestingData(DataSaver):
    """
    To add a new h5py.Group to the HDF5 data file with fake data for testing the results.
    """

    def __init__(
            self,
            filename: str,
            test_resolution: int,
            **kwargs,
        ) -> None:
        """
        To initialise the parent class to be able to access the class attributes and methods.
        This child class is to add a 'TEST data' group to the data file where fake data is saved to
        be able to use in tests.

        Args:
            filename (str): the filename of the HDF5 data file to update.
            test_resolution (int): the number of points in the phi direction when creating the
                Sun's surface. At the end the sun will be represented by 2 * test_resolution**2
                points.
        """

        # PARENT
        super().__init__(filename=filename, processes=0, **kwargs)

        # ATTRIBUTEs new
        self.test_resolution = test_resolution

    @Decorators.running_time
    def add_to_file(self):
        """
        To add or update (if it already exist) the 'TEST data' group to the HDF5 file containing
        all the data.
        """

        with h5py.File(os.path.join(self.paths['save'], self.filename), 'a') as HDF5File:

            test_group_name = 'TEST data'
            if test_group_name in HDF5File:
                group = HDF5File[test_group_name]
            else:
                group = HDF5File.create_group('TEST data')
                group.attrs['description'] =  (
                    "This group only contains data to be used for testing."
                )
            # SUN add
            sun_coords = self.create_sun()
            group = self.add_sun(group, sun_coords, 'Sun sphere')

    def add_sun(self, group: h5py.Group, coords: np.ndarray, name: str) -> h5py.File | h5py.Group:
        """
        To add the fake sun data to the HDF5 file.

        Args:
            group (h5py.Group): the 'TEST data' group pointer.
            coords (np.ndarray): the (x, y, z) cartesian coords of the fake Sun data.
            name (str): the name of the new group pointing to the fake Sun data.

        Returns:
            h5py.File | h5py.Group: the 'TEST data' group.
        """

        sun = {
            'description': (
                "The sun surface as points on the sun surface. Used to test if the visualisation "
                "and re-projection codes are working properly."
            ),
            'raw coords': {
                'data': coords.astype('float32'),
                'unit': 'km',
                'description': (
                    "The cartesian coordinates of the sun's surface. There are "
                    f"{2* self.test_resolution**2} points positioned uniformly on the surface."
                ),
            },
        }
        if name in group: del group[name]
        group = self.add_group(group, sun, name)
        return group
    
    def create_sun(self) -> np.ndarray:
        """
        To find the coords of the Sun's surface. The points delimiting the surface are uniformly
        positioned and there are 2 * N**2 points where N is the number of points chosen when
        initialising the class.

        Returns:
            np.ndarray: the (x, y, z) coords representing the Sun's surface.
        """

        # COORDs spherical
        N = self.test_resolution  # number of points in the theta direction
        phi = np.linspace(0, np.pi, N)  # latitude of the points
        theta = np.linspace(0, 2 * np.pi, 2 * N)  # longitude of the points
        phi, theta = np.meshgrid(phi, theta)  # the subsequent meshgrid  

        # COORDs cartesian in km
        x = self.solar_r * np.sin(phi) * np.cos(theta)
        y = self.solar_r * np.sin(phi) * np.sin(theta)
        z = self.solar_r * np.cos(phi) 
        return np.stack([x.ravel(), y.ravel(), z.ravel()], axis=0)



if __name__=='__main__':

    # RUN
    instance = AddTestingData(filename='sig1e20_leg20_lim0_03.h5', test_resolution=int(1e4))
    instance.add_to_file()