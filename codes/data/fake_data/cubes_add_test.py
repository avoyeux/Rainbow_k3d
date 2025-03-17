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
from data.cubes import DataSaver
from common import Decorators

# ? could the problem come from the definition of nx in the idl codes. Something like taking n and
# ? not n+1 (as the number of discreet values is n+1 or n depending on the problem)

# ! not sure if this code is used any more



class AddTestingData(DataSaver):
    """
    To add a new h5py.Group to the HDF5 data file with fake data for testing the results.
    """

    def __init__(self, filename: str, test_resolution: int, **kwargs) -> None:
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
    def add_to_file(self) -> None:
        """
        To add or update (if it already exist) the 'TEST data' group to the HDF5 file containing
        all the data.
        """

        with h5py.File(os.path.join(self.paths['save'], self.filename), 'a') as HDF5File:
            
            # GET dx
            dx = float(HDF5File['dx'][...])  #type:ignore

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

            # SUN index add
            path_to_copy = "Filtered/No duplicates new"
            no_duplicates_info_info = self.get_path_info(HDF5File[path_to_copy])  #type:ignore
            sun_index, borders = self.sun_to_index(
                data_info=no_duplicates_info_info,
                dx=dx,
                sun_data=sun_coords,
            )
            group = self.add_sun_in_index(group, sun_index.coords, borders, 'Sun index')

    @Decorators.running_time
    def sun_to_index(
            self,
            data_info: dict[str, float | np.ndarray],
            dx: float,
            sun_data: np.ndarray,
        ) -> tuple[sparse.COO, dict[str, dict[str, str | float]]]:

        # ATTRIBUTE create
        time_indexes = np.unique(data_info['coords'][0])

        # BORDERs new
        x_min, y_min, z_min = np.min(sun_data, axis=1) # todo need to change this as too many
        # todo points will be kept.
        x_min: float = x_min if x_min <= data_info['xmin'] else data_info['xmin']
        y_min: float = y_min if y_min <= data_info['ymin'] else data_info['ymin']
        z_min: float = z_min if z_min <= data_info['zmin'] else data_info['zmin']
        _, new_borders = self.create_borders((0, x_min, y_min, z_min))

        # COORDs indexes
        sun_data[0, :] = sun_data[0, :] - data_info['xmin']
        sun_data[1, :] = sun_data[1, :] - data_info['ymin']
        sun_data[2, :] = sun_data[2, :] - data_info['zmin']
        sun_data /= dx
        sun_data = np.round(sun_data).astype('int32')

        # FUSION sun - cubes
        data = np.hstack([
            np.vstack((np.full((1, sun_data.shape[1]), time), sun_data))
            for time in time_indexes
        ])
        data = np.hstack([data_info['coords'], data]).astype('int32')
        # INDEXEs positive
        x_min, y_min, z_min = np.min(sun_data, axis=1).astype(int)
        if x_min < 0: data[1, :] -= x_min
        if y_min < 0: data[2, :] -= y_min
        if z_min < 0: data[3, :] -= z_min

        # COO data
        shape = np.max(data, axis=1) + 1
        values = np.ones(data.shape[1], dtype='uint8')
        data = sparse.COO(coords=data, data=values, shape=shape).astype('uint8')
        return data, new_borders

    def add_sun_in_index(
            self,
            group: h5py.Group,
            data: np.ndarray,
            borders: dict[str, dict[str, str | float]],
            name: str,
        ) -> h5py.Group:

        sun_index = {
            'description': "The Sun surface with data no duplicates new.",
            'coords': {
                'data': data,
                'unit': 'none',
                'description': (
                    "The sun surface with the data from new duplicates new (without feet) as "
                    "indexes. This data is only for testing the visualisation and re-projection "
                    "codes."
                ),
            },
        }
        if name in group: del group[name]
        sun_index |= borders
        self.add_group(group, sun_index, name)
        return group

    def get_path_info(self, group: h5py.Group) -> dict[str, float | np.ndarray]:
        
        data_info = {
            'xmin': group['xmin'][...],  #type:ignore
            'ymin': group['ymin'][...],  #type:ignore
            'zmin': group['zmin'][...],  #type:ignore
            'coords': group['coords'][...],  #type:ignore
        }
        return data_info  #type:ignore

    def add_sun(self, group: h5py.Group, coords: np.ndarray, name: str) -> h5py.Group:
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
                "The Sun surface as points on the sun surface. Used to test if the visualisation "
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
            'values': {
                'data': np.ones(coords.shape[1], dtype='uint8'),
                'unit': 'none',
                'description': (
                    "The value associated to each voxel position. In this case, it's just a 1D "
                    "ndarray of ones."
                )
            }
        }
        if name in group: del group[name]
        self.add_group(group, sun, name)
        return group
    
    @Decorators.running_time
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
    instance = AddTestingData(filename='sig1e20_leg20_lim0_03.h5', test_resolution=int(7e2))
    instance.add_to_file()
