"""
To create fake protuberance data with the same formatting than the real data created from the
observations of Dr. Bocchialini, Dr. Soubrie's automatic detection and Dr. Auchere's 3D creation
codes.
"""
# * Impossible to create .save file with Python (IDL property -- another reason to hate them).

# IMPORTs
import os
import h5py
import datetime

# IMPORTs alias
import numpy as np

# IMPORTs personal
from common import config, Decorators
from codes.data.base_hdf5_creator import VolumeInfo, BaseHDF5Protuberance
from codes.data.fake_data.base_fake_hdf5 import BaseFakeHDF5



class CreateFakeData(BaseFakeHDF5, BaseHDF5Protuberance):
    """
    To create a fake data set for comparing with the fake and real datasets.
    """

    def __init__(
            self,
            angle_step: float,
            sphere_radius: tuple[float, float],
            torus_radius: tuple[float, float],
            increase_factor: float = 1.,
            filepath: str | None = None,
            compression: bool = True,
            compression_lvl: int = 9,
            create_new_hdf5: bool = True,
            verbose: int | None = None,
            flush: bool | None = None,
        ) -> None:
        """
        To initialise the CreateFakeData class.

        Args:
            angle_step (float): the step in radian between each point for the fake data.
            sphere_radius (tuple[float, float]): the max and min radius of the fake ball in km.
            torus_radius (tuple[float, float]): the main and minor radius of the fake torus in km.
            increase_factor (float, optional): the multiplying factor by which the initial cube
                borders will change. Defaults to 1..
            filepath (str | None, optional): the path where the HDF5 file will be saved. If None,
                the config file information will be used. If create_new_hdf5 is False, the filepath
                needs to be the path to am already existing HDF5 file. Defaults to None.
            compression (bool, optional): deciding to compress the datasets using 'gzip'.
                Defaults to True.
            compression_lvl (int, optional): the 'gzip' compression level. If 'compression' is
                False, this parameter is ignored. Defaults to 9.
            create_new_hdf5 (bool, optional): deciding to store the new data in a new HDF5 file.
                If False, the data will be added to the existing HDF5 file. Defaults to True.
            verbose (int | None, optional): the level of verbosity. If None, the corresponding
                config file value will be used. Defaults to None.
            flush (bool | None, optional): deciding to flush the buffer (print output).
                If None, the corresponding config file value will be used. Defaults to None.
        """

        # CONFIGURATION attributes
        if filepath is None and create_new_hdf5:
            self.filepath = str(config.path.data.fake_cube)
        elif filepath is None:
            self.filepath = str(config.path.data.fusion)
        else:
            self.filepath = filepath
        self.verbose = config.run.verbose if verbose is None else verbose
        self.flush = config.run.flush if flush is None else flush

        # PARENTs
        super().__init__(
            angle_step=angle_step,
            sphere_radius=sphere_radius,
            increase_factor=increase_factor,
            torus_radius=torus_radius,
        )
        BaseHDF5Protuberance.__init__(
            self,
            filename=os.path.basename(self.filepath),
            compression=compression,
            compression_lvl=compression_lvl,
        )

        # CHOICEs
        self.choices = ['Sun']

        # ARGUMENTs
        self.create_new_hdf5 = create_new_hdf5

        # PATHs update
        self.paths['save'] = config.path.dir.data.hdf5

        # ATTRIBUTEs setup
        self.volume = self.basic_data()

    def basic_data(self) -> VolumeInfo:
        """
        To get the basic volumetric data from the .save files, i.e. dx, xt_min, yt_min, zt_min.

        Returns:
            VolumeInfo: a dataclass containing the basic volumetric data.
        """

        # DATA formatting
        information = VolumeInfo(
            dx=self.cube_info.dx,
            xt_min=min(self.cube_info.xt),
            yt_min=min(self.cube_info.yt),
            zt_min=min(self.cube_info.zt),
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

            if 'Sun' in self.choices:

                # SUN surface add
                data = self.fake_sphere_surface()
                if 'Raw' in self.choices: 
                    self.add_sun(group=group, coords=data, name='Sun surface')

                # SUN indexes add
                data, borders = self.to_index_pos(data, unique=True)
                self.add_sun_indexes(
                    group=group,  #type: ignore
                    coords=data,
                    name='Sun surface',
                )

                # SUN borders add
                for key, value in borders.items():
                    self.add_dataset(parent_group=group['Sun surface'], info=value, name=key)

                if self.verbose > 0: print('Sun added.', flush=self.flush)

            if 'Cube' in self.choices:
                # CUBE add
                data = self.fake_cube()

                # CROSS add to cube
                cross = self.fake_3d_cross()
                data = np.concatenate([data, cross], axis=1)
                if 'Raw' in self.choices:
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
                if self.verbose > 0: print('Cube and cross added.', flush=self.flush)

            if 'Torus' in self.choices:
                # TORUS add
                data = self.fake_torus()
                if 'Raw' in self.choices:
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
                if self.verbose > 0: print('Torus added.', flush=self.flush)

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
                'data': 1,
                'unit': 'none',
                'description': "The value associated to each voxel position.",
            },
        }

        # ADD to file
        self.add_group(group, indexes, name)

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
                'data': 1,
                'unit': 'none',
                'description': "The value associated to each voxel position.",
            },
        }

        # ADD to file
        self.add_group(group, indexes, name)
    
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
        angle_step=1.5e-3,
        create_new_hdf5=False,
        torus_radius=(7.8e5, 2e4),
        sphere_radius=(6.95e5, 7e5),
        increase_factor=2.5,  # ! not 100% sur if it works properly, but should be. Will test later.
    )
    instance.create_hdf5()
