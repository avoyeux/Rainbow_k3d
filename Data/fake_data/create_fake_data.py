"""
To create fake protuberance data with the same formatting than the real data created from the
observations of Dr. Bocchialini, Dr. Soubrie's automatic detection and Dr. Auchere's 3D creation
codes.
"""
# * Impossible to create .save file with Python (IDL property -- another reason to hate them).

# IMPORTs
import os
import h5py
import scipy
import datetime

# IMPORTs alias
import numpy as np

# IMPORTs personal
from common import main_paths, Decorators
from Data.base_hdf5_creator import VolumeInfo, BaseHDF5Protuberance

# todo need to think about how to do cubes one by one to save RAM.
# todo add the choice between creating a new hdf5 file or add to the hdf5 containing the real data.



class CreateFakeData(BaseHDF5Protuberance):
    """
    To create a fake data set for testing in an HDF5.
    """

    def __init__(self, filename: str, sun_resolution: int) -> None:

        # PARENT
        super().__init__()

        # ARGUMENTs
        self.filename = filename
        self.sun_resolution = sun_resolution

        # ATTRIBUTEs setup
        self.paths = self.setup_path()
        self.cube_filepath = os.path.join(self.paths['cubes'], 'cube000.save')
        self.volume = self.basic_data()

    def setup_path(self) -> dict[str, str]:
        """
        Gives the directory paths.

        Returns:
            dict[str, str]: the directory paths.
        """

        # CHECK
        python_codes = main_paths.root_path

        # PATHs keep
        paths = {
            'cubes': os.path.join(python_codes, '..', 'Cubes_karine'),
            'save': os.path.join(python_codes, 'Data', 'fake_data'),
        }
        return paths
    
    def basic_data(self) -> VolumeInfo:
        """
        To get the basic volumetric data from the .save files, i.e. dx, xt_min, yt_min, zt_min.

        Returns:
            VolumeInfo: a dataclass containing the basic volumetric data.
        """

        # DATA
        data = scipy.io.readsav(self.cube_filepath)

        # DATA formatting
        information = VolumeInfo(
            dx=data.dx,
            xt_min=data.xt_min,
            yt_min=data.yt_min,
            zt_min=data.zt_min,
        )
        return information
    
    @Decorators.running_time
    def create_hdf5(self) -> None:

        with h5py.File(os.path.join(self.paths['save'], self.filename), 'w') as HDF5File:
            
            # FOUNDATION of file
            self.file_foundation(HDF5File=HDF5File)

            # TEST group
            test_group_name = 'Test data'
            group = HDF5File.create_group(name=test_group_name)

            # SUN surface add
            sun_surface = self.create_sun_surface()
            self.add_sun(group=group, coords=sun_surface, name='Sun surface')

            # SUN indexes add
            sun_surface, borders = self.to_index_pos(sun_surface)
            self.add_sun_indexes(
                group=group['Sun surface'],  #type: ignore
                coords=sun_surface,
                name='coords',
            )

            # SUN borders add
            for key, value in borders.items():
                self.add_dataset(parent_group=group['Sun surface'], info=value, name=key)



            # CUBE add
            fake_cube = self.fake_cube()
            self.add_fake_cube(group, coords=fake_cube, name='Fake cube')

            # CUBE indexes add
            fake_cube, borders = self.to_index_pos(fake_cube)
            self.add_fake_cube_indexes(
                group=group['Fake cube'],
                coords=fake_cube,
                name='coords',
            )

            # CUBE borders add
            for key, value in borders.items():
                self.add_dataset(parent_group=group['Fake cube'], info=value, name=key)

    def file_foundation(self, HDF5File: h5py.File) -> None:

        # ? should I write the code so that I can put a cube at the same time than an sdo pos
        # METADATA file
        metadata_dict = self.main_metadata()
        self.add_dataset(HDF5File, metadata_dict)

        # RESOLUTION in km
        dx_dict = self.dx_to_dict()
        self.add_dataset(HDF5File, dx_dict, 'dx')

        # COORDs sdo
        sdo_dict = self.fake_sdo_pos()
        self.add_dataset(HDF5File, sdo_dict, 'SDO positions')

        # INDEXEs time
        time_dict = self.fake_time_indexes(len(sdo_dict['data']))
        self.add_dataset(HDF5File, time_dict, 'Time indexes')

        # DATEs
        dates_dict = self.fake_dates(len(sdo_dict['data']))
        self.add_dataset(HDF5File, dates_dict, 'Dates')
    
    def fake_dates(self, nb_of_dates: int) -> dict[str, str | np.ndarray]:

        # DATE first
        first_date = datetime.datetime(2025, 1, 1, 0, 0, 0)

        # POPULATE dates
        dates = np.array([
            (first_date + datetime.timedelta(hours=i)).isoformat()
            for i in range(nb_of_dates)
        ], dtype='S19')

        # INFO dates
        dates_dict = {
            'data': dates,
            'unit': 'none',
            'description': (
                "The fake dates that represent the date for each cube. They are saved as an "
                "ndarray with S19 values (so of type np.bytes_)."
            ),
        }
        return dates_dict

    def fake_sdo_pos(self) -> dict[str, dict[str, str | np.ndarray]]:
        
        # DATA
        coef = 50
        sdo_pos = np.array([
            [0, 0, coef * self.solar_r],
            [0, 0, -coef * self.solar_r],
            [0, coef * self.solar_r, 0],
            [coef * self.solar_r, 0, 0],
        ], dtype='float32')

        # INFO sdo pos
        sdo_dict = {
            'data': sdo_pos,
            'unit': 'km',
            'description': (
                "Fake sdo positions for which I know the resulting image results.\n"
                "The data represents the (t, x, y, z) coordinates for SDO."
            ),
        }
        return sdo_dict

    def add_sun(self, group: h5py.Group, coords: np.ndarray, name: str) -> None:
        """
        To add the fake sun data to the HDF5 file.

        Args:
            group (h5py.Group): the 'TEST data' group pointer.
            coords (np.ndarray): the (x, y, z) cartesian coords of the fake Sun data.
            name (str): the name of the new group pointing to the fake Sun data.
        """

        # INFO sun
        sun = {
            'description': (
                "The Sun surface as points on the sun surface. Used to test if the visualisation "
                "and re-projection codes are working properly."
            ),
            'raw coords': {
                'data': coords.astype('float32'),
                'unit': 'km',
                'description': (
                    "The cartesian coordinates of the sun's surface. The points are positioned "
                    "uniformly on the surface."
                ),
            },
            'values': {
                'data': np.ones(coords.shape[1], dtype='uint8'),
                'unit': 'none',
                'description': (
                    "The value associated to each voxel position. In this case, it's just a 1D "
                    "ndarray of ones."
                ),
            },
        }

        # ADD to file
        if name in group: del group[name]
        self.add_group(group, sun, name)

    def add_sun_indexes(self, group: h5py.Group, coords: np.ndarray, name: str) -> None:

        # INFO sun indexes
        indexes = {
            'data': coords.astype('uint16'),
            'unit': 'none',
            'description': (
                "The index positions of the Sun's surface. The index positions are delimited by "
                "the border values 'xt_min', 'yt_min', 'zt_min'." 
            ),
        }

        # ADD to file
        self.add_dataset(group, indexes, name)
    
    def create_sun_surface(self) -> np.ndarray:
        """
        To find the coords of the Sun's surface. The points delimiting the surface are uniformly
        positioned and there are 2 * N**2 points where N is the number of points chosen when
        initialising the class.

        Returns:
            np.ndarray: the (x, y, z) coords representing the Sun's surface.
        """

        # COORDs spherical
        N = self.sun_resolution  # number of points in the theta direction
        phi = np.linspace(0, np.pi, N)  # latitude of the points
        theta = np.linspace(0, 2 * np.pi, 2 * N)  # longitude of the points
        phi, theta = np.meshgrid(phi, theta)  # the subsequent meshgrid  

        # COORDs cartesian in km
        x = self.solar_r * np.sin(phi) * np.cos(theta)
        y = self.solar_r * np.sin(phi) * np.sin(theta)
        z = self.solar_r * np.cos(phi) 
        return np.stack([x.ravel(), y.ravel(), z.ravel()], axis=0)

    def fake_time_indexes(self, nb_of_values: int) -> dict[str, str | np.ndarray]:
        """
        To create a list of the time indexes used in the data and the corresponding description.

        Args:
            nb_of_values (int): the number of cubes that are to be used.

        Returns:
            dict[str, str | np.ndarray]: the time indexes and its description.
        """
        
        # DATA
        time_indexes = np.arange(0, nb_of_values, dtype=int)

        # INFO time indexes
        time_dict = {
            'data': time_indexes,
            'unit': 'none',
            'description': (
                "The time indexes. In this fake data, it is only an np.arange() from 0 to the "
                "number of points you need considered - 1."
            ),
        }
        return time_dict

    def fake_cube(self) -> np.ndarray:
        """
        To create a fake cube data in cartesian reprojected carrington coordinates.

        Returns:
            np.ndarray: the cube coordinates as (x, y, z) in km.
        """

        # todo need to think about what arguments to add to create differently positioned cubes

        # COORDs range
        x_range = np.arange(0, 100 * self.volume.dx, self.volume.dx)
        y_range = np.arange(0, 100 * self.volume.dx, self.volume.dx)
        z_range = np.arange(0, 100 * self.volume.dx, self.volume.dx)

        # COORDs values
        x_positions = x_range + self.volume.xt_min
        y_positions = y_range + self.volume.yt_min
        z_positions = z_range + self.volume.zt_min

        # FILL VOLUME
        X, Y, Z = np.meshgrid(x_positions, y_positions, z_positions, indexing='ij')
        return np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=0)

    def add_fake_cube(self, group: h5py.Group, coords: np.ndarray, name: str) -> None:
        """
        Creates the description to the fake cube data and adds it to the HDF5 file. 

        Args:
            group (h5py.Group): the HDF5 group where the fake cube data needs to be inserted.
            coords (np.ndarray): the fake cube coords in cartesian reprojected Carrington
                coordinates in km.
            name (str): the name of the new group to be added to the parent group.
        """

        # INFO cube
        cube = {
            'raw coords': {
                'data': coords.astype('float32'),
                'unit': 'km',
                'description': (
                    "The fake cube coordinates in cartesian reprojected Carrington coordinates."
                ),
            },
            'values': {
                'data': np.ones(coords.shape[1], dtype='uint8'),
                'unit': 'none',
                'description': (
                    "The value associated to each voxel position. In this case, it's just a 1D "
                    "ndarray of ones."
                ),
            },
        }

        # ADD to file
        if name in group: del group[name]
        self.add_group(group, cube, name) 

    def add_fake_cube_indexes(self, group: h5py.Group, coords: np.ndarray, name: str) -> None:

        # INFO fake cube indexes
        indexes = {
            'data': coords.astype('uint16'),
            'unit': 'none',
            'description': (
                "The index positions of the fake cube voxels. The indexes are delimiter by the "
                "cube borders, i.e. xt_min, yt_min, zt_min."
            ),
        }

        # ADD to file
        self.add_dataset(group, indexes, name)


if __name__=='__main__':

    instance = CreateFakeData(
        filename='testing.h5',
        sun_resolution=int(1e2),
    )
    instance.create_hdf5()
