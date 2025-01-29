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
from common import Decorators, root_path
from Data.base_hdf5_creator import VolumeInfo, BaseHDF5Protuberance

# todo need to think about how to do cubes one by one to save RAM.



class CreateFakeData(BaseHDF5Protuberance):
    """
    To create a fake data set for testing in an HDF5.
    """

    def __init__(
            self,
            filename: str,
            sun_resolution: int,
            torus_main_radius: float,
            torus_width_radius: float,
            torus_plane: str = 'xy',
            nb_of_cubes: int = 4,
            create_new_hdf5: bool = True,
        ) -> None:

        # PARENT
        super().__init__()

        # ARGUMENTs
        self.filename = filename
        self.sun_resolution = sun_resolution
        self.nb_of_cubes = nb_of_cubes
        self.torus_main_radius = torus_main_radius
        self.torus_width_radius = torus_width_radius
        self.torus_plane = torus_plane
        self.create_new_hdf5 = create_new_hdf5

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

        # PATHs setup
        if self.create_new_hdf5:
            save_path = os.path.join(root_path, 'Data', 'fake_data')
        else:
            save_path = os.path.join(root_path, 'Data')
        
        # PATHs keep
        paths = {
            'cubes': os.path.join(root_path, '..', 'Cubes_karine'),
            'save': save_path,
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
        """
        To create the HDF5 file with the fake data.
        """

        with h5py.File(
            os.path.join(self.paths['save'], self.filename),
            'w' if self.create_new_hdf5 else 'a') as HDF5File:
            
            # FOUNDATION of file
            if self.create_new_hdf5: self.file_foundation(HDF5File=HDF5File)

            # TEST group
            test_group_name = 'Test data'
            if test_group_name in HDF5File: del HDF5File[test_group_name]
            group = HDF5File.create_group(name=test_group_name)

            # SUN surface add
            data = self.fake_4D(self.create_sun_surface())
            self.add_sun(group=group, coords=data, name='Sun surface')

            # SUN indexes add
            data, borders = self.to_index_pos(data)
            self.add_sun_indexes(
                group=group,  #type: ignore
                coords=data,
                name='Sun surface',
            )

            # SUN borders add
            for key, value in borders.items():
                self.add_dataset(parent_group=group['Sun surface'], info=value, name=key)

            # CUBE add
            data = self.fake_4D(self.fake_cube())
            print(f'cube max magnitude is {np.max(np.abs(data))}')

            # CROSS add to cube
            cross = self.fake_4D(self.fake_3d_cross())
            data = np.concatenate([data, cross], axis=1)
            print(f'cube + cross max magnitude is {np.max(np.abs(data))}')
            self.add_fake_cube(group, coords=data, name='Fake cube')

            # CUBE indexes add
            data, borders = self.to_index_pos(data, unique=True)
            self.add_fake_cube_indexes(
                group=group,
                coords=data,
                name='Fake cube',
            )

            # CUBE borders add
            self.add_group(group, borders, name='Fake cube')

            # TORUS add
            data = self.fake_4D(self.fake_torus())
            self.add_fake_torus(group, coords=data, name='Fake torus')
            
            # TORUS indexes add
            data, borders = self.to_index_pos(data)
            self.add_fake_cube_indexes(
                group=group,
                coords=data,
                name='Fake torus',
            )

            # TORUS borders add
            self.add_group(group, borders, name='Fake torus')

    def file_foundation(self, HDF5File: h5py.File) -> None:
        """
        To create the foundation of the HDF5 file, i.e. the metadata, the resolution, the SDO
        positions, the time indexes, and the dates.

        Args:
            HDF5File (h5py.File): the HDF5 file object.
        """

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
        """
        To create the fake dates for the data.

        Args:
            nb_of_dates (int): the number of dates that are to be created.

        Returns:
            dict[str, str | np.ndarray]: the dates and its description.
        """

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
        """
        To create the fake SDO positions for the data.

        Returns:
            dict[str, dict[str, str | np.ndarray]]: the SDO positions and its description.
        """
        
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
        }

        # ADD to file
        if name in group: del group[name]
        self.add_group(group, sun, name)

    def add_sun_indexes(self, group: h5py.Group, coords: np.ndarray, name: str) -> None:
        """
        To add the indexes of the Sun's surface to the HDF5 file.

        Args:
            group (h5py.Group): the HDF5 group where the Sun's surface indexes need to be inserted.
            coords (np.ndarray): the (x, y, z) cartesian coords of the Sun's surface.
            name (str): the name of the new dataset pointing to the Sun's surface indexes.
        """

        # INFO sun indexes
        indexes = {
            'coords': {
                'data': coords.astype('uint16'),
                'unit': 'none',
                'description': (
                    "The index positions of the Sun's surface. The index positions are delimited by "
                    "the border values 'xt_min', 'yt_min', 'zt_min'." 
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
        self.add_group(group, indexes, name)
    
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

    def fake_4D(self, cube: np.ndarray) -> np.ndarray:
        """
        To add the time dimension to the cube data.

        Args:
            cube (np.ndarray): _description_

        Returns:
            np.ndarray: _description_
        """

        # COORDs repeat N times
        repeated_coords = np.tile(cube, (1, self.nb_of_cubes))
        time_row = np.repeat(np.arange(self.nb_of_cubes), cube.shape[1])
        return np.vstack([time_row, repeated_coords])
    
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
        }

        # ADD to file
        if name in group: del group[name]
        self.add_group(group, cube, name)
    
    def fake_3d_cross(self) -> np.ndarray:
        """
        Creates a 3D cross with 6 branches.

        Returns:
            np.ndarray: The coordinates of the 3D cross in cartesian reprojected Carrington
                coordinates.
        """

        # COORDs range
        half_length = 200
        half_thickness = 6

        axis_range = np.arange(
            - half_length * self.volume.dx,
            (half_length + 1) * self.volume.dx,
            self.volume.dx,
        )
        center = np.array([
            self.volume.xt_min + (50 * self.volume.dx),
            self.volume.yt_min + (50 * self.volume.dx),
            self.volume.zt_min + (50 * self.volume.dx),
        ])

        # BRANCHEs 
        x_branch = np.stack([
            axis_range + center[0],
            np.zeros(axis_range.shape) + center[1],
            np.zeros(axis_range.shape) + center[2],
        ], axis=0)  # todo just do + center
        print(f'x_branch max magnitude is {np.max(np.abs(x_branch))}')
        y_branch = np.stack([
            np.zeros(axis_range.shape) + center[0],
            axis_range + center[1],
            np.zeros(axis_range.shape) + center[2],
        ], axis=0)
        z_branch = np.stack([
            np.zeros(axis_range.shape) + center[0],
            np.zeros(axis_range.shape) + center[1],
            axis_range + center[2],
        ], axis=0)

        # THICKNESS setup
        thickness_range = np.arange(
            - half_thickness * self.volume.dx,
            (half_thickness + 1) * self.volume.dx,
            self.volume.dx,
        )
        A, B = np.meshgrid(thickness_range, thickness_range)
        translations = np.stack([A.ravel(), B.ravel()], axis=0).T

        # BRANCHEs
        all_x_branches = np.copy(x_branch)
        all_y_branches = np.copy(y_branch)
        all_z_branches = np.copy(z_branch)
        for (a, b) in translations:
            all_x_branches = np.concatenate([
                all_x_branches,
                self.cross_branch_tickness(indexes=(1, 2), branch=x_branch, vals=(a, b))
            ], axis=1)
            all_y_branches = np.concatenate([
                all_y_branches,
                self.cross_branch_tickness(indexes=(0, 2), branch=y_branch, vals=(a, b))
            ], axis=1)
            all_z_branches = np.concatenate([
                all_z_branches,
                self.cross_branch_tickness(indexes=(0, 1), branch=z_branch, vals=(a, b))
            ], axis=1)

        # DUPLICATEs filtering
        all_branches = np.concatenate([all_x_branches, all_y_branches, all_z_branches], axis=1)
        return all_branches
    
    def cross_branch_tickness(
            self,
            indexes: tuple[int, int],
            branch: np.ndarray,
            vals: tuple[float, float],
        ) -> np.ndarray:
        """
        To add the thickness to the branches of the 3D cross.

        Args:
            indexes (list[int]): 
            branch (np.ndarray): _description_
            vals (tuple[float, float]): _description_

        Returns:
            np.ndarray: _description_
        """

        new_branch = np.copy(branch)
        new_branch[indexes[0]] = branch[indexes[0]] + vals[0]
        new_branch[indexes[1]] = branch[indexes[1]] + vals[1]
        return new_branch

    def add_fake_cube_indexes(self, group: h5py.Group, coords: np.ndarray, name: str) -> None:
        """
        To add the indexes of the fake cube to the HDF5 file.

        Args:
            group (h5py.Group): the HDF5 group where the fake cube indexes need to be inserted.
            coords (np.ndarray): the fake cube coords in cartesian reprojected Carrington.
            name (str): the name of the new dataset pointing to the fake cube indexes.
        """

        # INFO fake cube indexes
        indexes = {
            'coords': {
                'data': coords.astype('uint16'),
                'unit': 'none',
                'description': (
                    "The index positions of the fake cube voxels. The indexes are delimiter by "
                    "the cube borders, i.e. xt_min, yt_min, zt_min."
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
        self.add_group(group, indexes, name)

    def fake_torus(self) -> np.ndarray:
        """
        To create a fake torus data in cartesian reprojected carrington coordinates.

        Returns:
            np.ndarray: the torus coordinates as (x, y, z) in km.
        """

        # COORDs range
        theta = np.linspace(0, 2 * np.pi, self.sun_resolution)
        phi = np.linspace(0, 2 * np.pi, self.sun_resolution)
        theta, phi = np.meshgrid(theta, phi)

        # COORDs cartesian in km
        x = (self.torus_main_radius + self.torus_width_radius * np.cos(theta)) * np.cos(phi)
        y = (self.torus_main_radius + self.torus_width_radius * np.cos(theta)) * np.sin(phi)
        z = self.torus_width_radius * np.sin(theta)
        return np.stack([x.ravel(), y.ravel(), z.ravel()], axis=0)
    
    def add_fake_torus(self, group: h5py.Group, coords: np.ndarray, name: str) -> None:
        """
        To add the fake torus data to the HDF5 file.

        Args:
            group (h5py.Group): the HDF5 group where the fake torus data needs to be inserted.
            coords (np.ndarray): the fake torus coords in cartesian reprojected Carrington
            name (str): the name of the new group to be added to the parent group.
        """

        # INFO torus
        torus = {
            'raw coords': {
                'data': coords.astype('float32'),
                'unit': 'km',
                'description': (
                    "The fake torus coordinates in cartesian reprojected Carrington coordinates."
                ),
            },
        }

        # ADD to file
        if name in group: del group[name]
        self.add_group(group, torus, name)    



if __name__=='__main__':

    instance = CreateFakeData(
        filename='sig1e20_leg20_lim0_03.h5',
        sun_resolution=int(2e3),
        create_new_hdf5=False,
        torus_main_radius=7.8e5,
        torus_width_radius=2e4,
    )
    instance.create_hdf5()
