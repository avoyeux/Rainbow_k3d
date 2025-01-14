"""
To add an HDF5 group with fake data to be able to visually test codes using the HDF5 protuberance
data.
"""

# IMPORTs
import os
import h5py
import sparse

# IMPORTs sub
import numpy as np

# IMPORTs personal
from Data.Cubes import DataSaver
from common import Decorators


class AddTestingData(DataSaver):
    # todo docstrings
    def __init__(
            self,
            filename: str,
            test_resolution: int,
            processes: int = 0,
            **kwargs,
        ) -> None:

        # PARENT
        super().__init__(filename=filename, processes=processes, **kwargs)

        # ATTRIBUTEs new
        self.test_resolution = test_resolution

    @Decorators.running_time
    def add_to_file(self):

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