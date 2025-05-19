"""
To create the HDF5 data files with the cubes data and metadata.
A lot of different data is saved in the file to make any further manipulation or visualisation more
easy.
"""

# IMPORTS
import os
import re
import glob
import h5py
import scipy
import sunpy
import sparse
import astropy

# IMPORTs alias
import numpy as np
import multiprocessing as mp

# IMPORTs sub
import astropy.io.fits  # ! for the static type checker: make sure it introduced no bugs
import sunpy.coordinates
import astropy.coordinates  # ! for the static type checker: make sure it introduced no bugs
from datetime import datetime
from astropy import units as u

# IMPORTs personal
from codes.data.helpers.all_sdo_dates import AllSDOMetadata
from codes.data.polynomial_fit.base_polynomial_fit import Polynomial
from codes.data.base_hdf5_creator import VolumeInfo, BaseHDF5Protuberance
from common import config, Decorators, CustomDate, DatesUtils, MultiProcessing

# TYPE ANNOTATIONs
import queue
import multiprocessing.shared_memory
from typing import Any, Callable, cast
from sitools2.clients.sdo_data import SdoData
type ManagerQueueProxy = queue.Queue[Any]  # used parent: actual queue type is not known
type SharedMemoryAlias = multiprocessing.shared_memory.SharedMemory

# API public
__all__ = ['DataSaver']

# todo change the code so each cube processing is done in a separate process
# todo descriptions change as dates changed and (e.g. raw data) most cube indexes contain no data



class DataSaver(BaseHDF5Protuberance):
    """
    To create cubes with and/or without feet in an HDF5 file.
    """

    @Decorators.running_time
    def __init__(
            self,
            filename: str | None = None,
            processes: int | None = None,
            integration_time: list[int] = [24],
            polynomial_points: int = int(1e6),
            polynomial_order: list[int] = [3, 4, 5],
            feet_lonlat: tuple[tuple[int | float, int | float], ...] = (
                (-177, 14.5),
                (-163.5, -16.5),
            ),
            feet_sigma: int | float = 1e-4,
            south_leg_sigma: int | float = 5,
            leg_threshold: float = 0.03,
            full: bool = False,
            no_feet: bool = False,
            compression: bool = True,
            compression_lvl: int = 9,
            fake_hdf5: bool = False,
    ) -> None:
        """
        To create the cubes with and/or without feet in an HDF5 file.

        Args:
            filename (str | None, optional): the filename of the file to be saved. When None, gets
                the value from the config file. Defaults to None.
            processes (int | None, optional): the number of processes used in the multiprocessing.
                When None, gets the value from the config file. Defaults to None.
            integration_time (list[int], optional): the time(s) in hours used in the time
                integration of the data. Defaults to [24].
            polynomial_points (int, optional): the number of points used when recreating the
                polynomial gotten from the curve fitting of the data. Defaults to int(1e6).
            polynomial_order (list[int], optional): the order(s) used for the polynomial fit of the
                data. Defaults to [3, 4, 5].
            feet_lonlat (tuple[tuple[int | float, int | float], ...], optional): the positions of
                the feet in re-projected Heliographic Carrington coordinates.
                Defaults to ((-177, 14.5), (-163.5, -16.5)).
            feet_sigma (int | float, optional): the sigma uncertainty in the feet used during the
                curve fitting of the data points. Defaults to 1e-4.
            south_leg_sigma (int | float, optional): the sigma uncertainty in the south leg used
                during the curve fitting of the data points. Defaults to 5.
            leg_threshold (float, optional): the threshold used to filter the data in the feet
                curve fitting. Defaults to 0.03.
            full (bool, optional): deciding to save all the data. In the case when 'full' is True,
                the raw coordinates of the polynomial curve are also saved, where as only the
                indexes that can be directly used as coords in a sparse.COO object.
                Defaults to False.
            no_feet (bool, optional): deciding to not add the feet to the data. Defaults to False.
            compression (bool, optional): deciding to use 'gzip' compression in the HDF5 file.
                Defaults to True.
            compression_lvl (int, optional): the level of compression used in the HDF5 file.
                Defaults to 9.
            fake_hdf5 (bool, optional): deciding to use the fake data to create the HDF5 file.
                Defaults to False.
        """

        # FILENAME setup
        if filename is None:
            if fake_hdf5:
                filename = os.path.basename(config.path.data.fake)
            else:
                filename = os.path.basename(config.path.data.real)

        # MULTIPROCESSING setup
        self.processes = int(config.run.processes if processes is None else processes)

        # PARENT
        super().__init__(filename, compression, compression_lvl)

        # CONSTANTs
        self.sdo_metadata: list[SdoData] = AllSDOMetadata().all_sdo_dates
        self.max_len: int = len(self.sdo_metadata)
        self.nb_processes: int = min(self.processes, self.max_len)
        self.feet_options: list[str] = ['', ' with feet'] if not no_feet else ['']
        self.first_datetime: datetime = cast(datetime, self.sdo_metadata[0].date_obs)

        # ARGUMENTs
        self.integration_time = [time * 3600 for time in integration_time]
        self.polynomial_order = polynomial_order
        self.polynomial_points = polynomial_points
        self.feet_sigma = feet_sigma
        self.south_leg_sigma = south_leg_sigma
        self.leg_threshold = leg_threshold
        self.full = full  # deciding to add the heavy sky coords arrays.
        self.no_feet = no_feet
        self.fake_hdf5 = fake_hdf5

        # PLACEHOLDERs
        self.dx: dict[str, str | float]  # information and value of the spatial resolution
        self.paths: dict[str, str]  # the paths to the different directories
        self.cube_numbers: list[int] # the cube numbers
        self.dates_seconds: list[int]  # the dates in seconds
        self.feet: astropy.coordinates.SkyCoord  # the feet positions
        self.cube_pattern: re.Pattern[str]  # the pattern for the .save cubes
        self.date_pattern: re.Pattern[str]  # the pattern for the STEREO B 30.4nm dates

        # SETUP attributes
        self.setup_attributes(feet_lonlat)

    def setup_path(self) -> dict[str, str]:
        """
        Gives the directory paths as a dictionary.

        Returns:
            dict[str, str]: the directory paths.
        """

        # PATHs keep
        paths = {
            'cubes': config.path.dir.data.cubes.karine,
            'intensities': config.path.dir.data.stereo.int,
            'sdo': config.path.dir.data.sdo,
            'stereo info': config.path.data.stereob_info,
            'save': config.path.dir.data.hdf5,
        }

        # PATHS update
        if self.fake_hdf5: paths['cubes'] = config.path.dir.data.cubes.fake
        return paths
    
    def setup_patterns(self) -> tuple[re.Pattern[str], re.Pattern[str]]:
        """
        The regular expression patterns used.

        Returns:
            tuple[re.Pattern[str], re.Pattern[str]]: the regular expression patterns for the .save
                cubes and the intensities STEREO B 30.4nm.
        """

        # PATTERNs
        cube_pattern = re.compile(r'cube(\d{3})\.save')
        date_pattern = re.compile(
            r'(?P<number>\d{4})_(?P<date>\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2})\.\d{3}\.png'
        )
        return cube_pattern, date_pattern
    
    def setup_attributes(self, feet_lonlat: tuple[tuple[int | float, int | float], ...]) -> None:
        """
        Multiple instance attributes are defined here. Function is only here to not flood the
        __init__ method.

        Args:
            feet_lonlat (tuple[tuple[int | float, int | float], ...]): the longitude and latitude
                positions for the added feet (i.e. ((lon1, lat1), (lon2, lat2))).
        """

        # FEET and PATTERNs
        self.feet = self.setup_feet(feet_lonlat)
        self.cube_pattern, self.date_pattern = self.setup_patterns()
        
        # PATHs create
        self.paths = self.setup_path()

        # FILEPATHs cubes
        filepaths = glob.glob(os.path.join(self.paths['cubes'], 'cube*.save'))

        # NUMBERs cubes
        cube_numbers = [  # * keep this as it is used for the raw data
            int(matched.group(1))
            for filepath in filepaths
            if (matched := self.cube_pattern.match(os.path.basename(filepath))) is not None
        ]
        self.cube_numbers = self._cube_numbers_to_cube_index(sorted(cube_numbers))

        # DATEs in seconds
        self.dates_seconds = [
            self._datetime_to_seconds(date=cast(datetime, metadata.date_obs))
            for metadata in self.sdo_metadata
        ]

    def _datetime_to_seconds(self, date: datetime) -> int:
        """
        To convert a datetime object to seconds since the first datetime.

        Args:
            date (datetime): the datetime object to convert.

        Returns:
            int: the number of seconds since the first datetime.
        """

        return round((date - self.first_datetime).total_seconds())

    @Decorators.running_time
    def _cube_numbers_to_cube_index(self, cube_indexes: list[int]) -> list[int]:
        """
        To convert the cube numbers to the new cube indexes (given the cadence of 1 minute).

        Args:
            cube_indexes (list[int]): the initial cube indexes.

        Returns:
            list[int]: the new cube indexes given the cadence of 1 minute.
        """

        stereo_filepaths = sorted(glob.glob(os.path.join(self.paths['intensities'], '*.png')))

        # INDEX to date
        cube_to_date = {
            int(matched.group('number')): datetime.strptime(
                matched.group('date'),
                '%Y-%m-%dT%H-%M-%S',
            )
            for filepath in stereo_filepaths
            if (matched := self.date_pattern.match(os.path.basename(filepath))) is not None
        }
        
        # DATEs of initial cubes
        cube_dates = [cube_to_date[index] for index in cube_indexes]

        # DATEs to new indexes
        date_to_index: dict[datetime, int] = cast(
            dict[datetime, int],
            {
                meta.date_obs: index
                for index, meta in enumerate(self.sdo_metadata)
            },
        )

        # INDEXEs new
        cube_indexes = [date_to_index[date] for date in cube_dates]
        return cube_indexes

    def setup_feet(
            self,
            lonlat: tuple[tuple[int | float, int | float], ...],
        ) -> astropy.coordinates.SkyCoord:
        """
        Gives the 2 feet positions as an astropy.coordinates.SkyCoord object in Carrington
        Heliographic Coordinates. 

        Args:
            lonlat (tuple[tuple[int | float, int | float], ...]): the longitude and latitude
                positions for the added feet (i.e. ((lon1, lat1), (lon2, lat2))).

        Returns:
            coordinates.SkyCoord: the SkyCoord for the feet.
        """

        # FEET setup
        feet_pos = np.empty((3, 2), dtype='float64')
        feet_pos[0, :] = np.array([lonlat[0][0], lonlat[1][0]])
        feet_pos[1, :] = np.array([lonlat[0][1], lonlat[1][1]])
        feet_pos[2, :] = self.solar_r

        # FEET create
        feet = astropy.coordinates.SkyCoord(
            feet_pos[0, :] * u.deg,  #type:ignore
            feet_pos[1, :] * u.deg,  #type:ignore
            feet_pos[2, :] * u.km,  #type:ignore
            frame=sunpy.coordinates.frames.HeliographicCarrington,
        )
        cartesian_feet = feet.represent_as(astropy.coordinates.CartesianRepresentation)#type:ignore
        feet = astropy.coordinates.SkyCoord(
            cartesian_feet,
            frame=feet.frame,
            representation_type='cartesian',
        )
        return feet
    
    def get_cube_dates_info(self) -> dict[str, dict[str, str | np.ndarray]]:
        """
        Gives the cube numbers and dates information. 

        Returns:
            dict[str, dict[str, str | np.ndarray]]: the data and metadata for the cube numbers and
                dates.
        """

        # METADATA
        cube_numbers_info = {
            'data': np.array(self.cube_numbers).astype('uint16'),
            'unit': 'none',
            'description': (
                "The cubes time indexes (i.e. the first row of the data) where there is volumetric"
                " data. This is only true for the raw data and the filtered one (not for the "
                "integrated data)." 
            ),
        }
        cube_dates_info = {
            'data': np.array([
                cast(datetime, meta.date_obs).strftime("%Y-%m-%dT%H-%M-%S")
                for meta in self.sdo_metadata
            ]).astype('S19'),
            'unit': 'none',
            'description': (
                "The dates of the STEREO B 30.4nm acquisitions. These represent all the possible "
                "dates and are indexed in the same way than the data cubes."
            ),
        }
        
        # SAVE reformat
        information = {
            'Time indexes': cube_numbers_info,
            'Dates': cube_dates_info,
        }
        return information
    
    @Decorators.running_time
    def create(self) -> None:
        """
        Main function that encapsulates the file creation and closing with a with statement.
        """

        with h5py.File(os.path.join(self.paths['save'], self.filename), 'w') as H5PYFile:

            # BORDERs
            cube = scipy.io.readsav(
                os.path.join(self.paths['cubes'], f"cube{self.cube_numbers[0]:03d}.save"),
            )
            self.volume = VolumeInfo(
                dx=float(cube.dx),
                xt_min=float(cube.xt_min),
                yt_min=float(cube.yt_min),
                zt_min=float(cube.zt_min),
            )
            self.dx = self.dx_to_dict()
            values = (cube.xt_min, cube.yt_min, cube.zt_min)
            init_borders = self.create_borders(values)

            # METADATA global
            self.foundation(H5PYFile)

            # DATA raw
            self.raw_group(H5PYFile, init_borders)

            # DATA filtered
            self.filtered_group(H5PYFile, init_borders)

            if not self.fake_hdf5:
                # DATA integrated
                self.integrated_group(H5PYFile, init_borders)

                # DATA polynomial
                self.polynomial_group(H5PYFile)
        
    def foundation(self, H5PYFile: h5py.File) -> None:
        """
        For the main file metadata before getting to the HDF5 datasets and groups.

        Args:
            H5PYFile (h5py.File): the HDF5 file.
        """

        description = (
            "Contains the data cubes for the Solar Rainbow event gotten from the intersection of "
            "masks gotten from SDO and STEREO images.The SDO masks were created from an automatic "
            "code created by Dr. Elie Soubrie, while the STEREO masks where manually created by "
            "Dr. Karine Bocchialini by visual interpretation of the 30.4nm STEREO B acquisitions."
            "\nNew values for the feet where added to help for a curve fitting of the filament. "
            "These were added by looking at the STEREO B 30.4nm images as the ends of the "
            "filament are more visible. Hence, the feet are not actually visible in the initial "
            "masks.\nExplanation on what each HDF5 group or dataset represent is given in the "
            "corresponding 'description' attribute."
        )
        metadata = self.main_metadata()
        metadata['description'] += description

        # METADATA
        meta_info = self.get_cube_dates_info()
        sdo_info = self.get_pos_sdo_info()
        # stereo_info = self.get_pos_stereo_info()

        # RUN information
        run_info = self.get_run_information()

        # UPDATE file
        self.add_dataset(H5PYFile, metadata)
        self.add_dataset(H5PYFile, self.dx, 'dx')
        for key in meta_info.keys(): self.add_dataset(H5PYFile, meta_info[key], key)
        self.add_dataset(H5PYFile, sdo_info, 'SDO positions')
        # self.add_dataset(H5PYFile, stereo_info, 'STEREO B positions')
        self.add_group(H5PYFile, run_info, 'Specific information')

    def get_run_information(self) -> dict[str, dict[str, str | int | float]]:
        """
        Gives information on the thresholds and weights used for the data creation.

        Returns:
            dict[str, dict[str, str | int | float]]: the data and metadata for the run information.
        """

        # RUN information
        feet_sigma = {
            'value': self.feet_sigma,
            'unit': 'none',
            'description': (
                "The sigma uncertainty in the feet used during the curve fitting of the data "
                "points. The lower the value, the higher the weight given to the feet positions."
            ),
        }
        leg_threshold = {
            'value': self.leg_threshold,
            'unit': 'none',
            'description': (
                "The threshold used to filter the data in the feet curve fitting. The lower the "
                "value, the less data is considered to be part of the 'left' leg. Information "
                "used with the leg sigma (as a the weight for the leg can be different than for "
                "the two unique voxel representing the feet)."
            ),
        }
        leg_sigma = {
            'value': self.south_leg_sigma,
            'unit': 'none',
            'description': (
                "The sigma uncertainty in the south leg used during the curve fitting of the data "
                "points. The lower the value, the higher the weight given to the south leg."
            ),
        }

        # INFORMATION formatting
        information = {
            'Feet sigma': feet_sigma,
            'Leg threshold': leg_threshold,
            'Leg sigma': leg_sigma,
        }
        return information
    
    @Decorators.running_time
    def get_pos_sdo_info(self) -> dict[str, str | np.ndarray]:
        """
        Gives the SDO satellite position information in Cartesian Heliocentric Coordinates.

        Returns:
            dict[str, str | np.ndarray]: the data and metadata for the SDO satellite position
        """

        # COORDs
        coordinates = self.get_pos_code(
            data=[metadata.ias_path for metadata in self.sdo_metadata], 
            function=self.get_pos_sdo_sub,
        )

        # DATA formatting
        information = {
            'data': coordinates.astype('float32'),
            'unit': 'km',
            'description': (
                "The position of the SDO satellite during the observations in cartesian "
                "heliocentric coordinates.\nThe shape of the data is (N, 3) where N represents "
                "the time indexes for the data and the 3 the x, y, z position of the satellite."
            ),
        }
        return information
    
    @Decorators.running_time
    def get_pos_stereo_info(self) -> dict[str, str | np.ndarray]:
        """
        Gives the STEREO B satellite position information in Cartesian Heliocentric Coordinates.

        Returns:
            dict[str, str | np.ndarray]: the data and metadata for the STEREO B satellite position
        """

        # ! wrong as don't have the positions for all the cubes given the 1 minute cadence
        # * Not using this method for now till discussed with Dr. Auchere.

        # DATA
        stereo_information = scipy.io.readsav(self.paths['stereo info']).datainfos

        # COORDs
        coordinates = self.get_pos_code(stereo_information, self.get_pos_stereo_sub)

        # DATA formatting
        information = {
            'data': coordinates.astype('float32'),
            'unit': 'km',
            'description': (
                "The position of the STEREO B satellite during the observations in cartesian "
                "heliocentric coordinates.\nThe shape of the data is (413, 3) where 413 "
                "represents the time indexes for the data and the 3 the x, y, z position of the "
                "satellite."
            ),          
        }
        return information

    def get_pos_code(
            self,
            data: np.recarray | list[str],
            function: Callable[[ManagerQueueProxy, ManagerQueueProxy], None],
        ) -> np.ndarray:
        """
        To multiprocess the getting of the positions of the SDO and STEREO B satellites.

        Args:
            data (np.recarray | list[str]): the data information for SDO or STEREO B.
            function (typing.Callable[[ManagerQueueProxy, ManagerQueueProxy], None]): the function
                used for each process to get the position of the satellite.

        Returns:
            np.ndarray: the position of the satellite in cartesian heliocentric coordinates.
        """

        # SETUP multiprocessing
        manager = mp.Manager()
        input_queue = manager.Queue()
        output_queue = manager.Queue()
        for i in range(self.max_len): input_queue.put((i, data[i]))
        for _ in range(self.nb_processes): input_queue.put(None)

        # RUN processes
        processes: list[mp.Process] = cast(list[mp.Process], [None] * self.nb_processes)
        for i in range(self.nb_processes):
            p = mp.Process(target=function, args=(input_queue, output_queue))
            p.start()
            processes[i] = p
        for p in processes: p.join()

        # RESULTs formatting
        coordinates_list: list[np.ndarray] = cast(list[np.ndarray], [None] * self.max_len)
        while not output_queue.empty():
            identifier, result = output_queue.get()
            coordinates_list[identifier] = result
        coordinates: np.ndarray = np.stack(coordinates_list, axis=0)
        return coordinates
    
    @staticmethod
    def get_pos_sdo_sub(input_queue: ManagerQueueProxy, output_queue: ManagerQueueProxy) -> None:
        """
        To get the position of the SDO satellite.

        Args:
            input_queue (ManagerQueueProxy): the input information (list[tuple[int, str]]) for
                identification and SDO information.
            output_queue (ManagerQueueProxy): to save the results outside the function.
        """
        
        while True:
            # CHECK queue
            arguments = input_queue.get()
            if arguments is None: return
            identification, filepath = arguments

            # DATA hdu
            header = astropy.io.fits.getheader(filepath)
            coords = sunpy.coordinates.frames.HeliographicCarrington(
                header['CRLN_OBS'] * u.deg,  #type:ignore
                header['CRLT_OBS'] * u.deg,  #type:ignore
                header['DSUN_OBS'] * u.m,  #type:ignore
                obstime=header['DATE-OBS'],
                observer='self',
            )
            coords = coords.represent_as(astropy.coordinates.CartesianRepresentation)

            # CONVERSION to km
            result = np.array([
                coords.x.to(u.km).value,  #type:ignore
                coords.y.to(u.km).value,  #type:ignore
                coords.z.to(u.km).value,  #type:ignore
            ])
            output_queue.put((identification, result))
        
    @staticmethod
    def get_pos_stereo_sub(
            input_queue: ManagerQueueProxy,
            output_queue: ManagerQueueProxy,
        ) -> None:
        """
        To get the position of the STEREO B satellite.

        Args:
            input_queue (ManagerQueueProxy): the input information (list[tuple[int, str]]) for
                identification and SDO information.
            output_queue (ManagerQueueProxy): to save the results outside the function.
        """

        while True:
            # CHECK queue
            arguments = input_queue.get()
            if arguments is None: return
            identification, information_recarray = arguments
            
            # DATA
            date = CustomDate(information_recarray.strdate)
            stereo_date = (
                f'{date.year}-{date.month}-{date.day}T{date.hour}:{date.minute}:{date.second}'
            )
            coords = sunpy.coordinates.frames.HeliographicCarrington(
                information_recarray.lon * u.deg,  #type:ignore
                information_recarray.lat * u.deg,  #type:ignore
                information_recarray.dist * u.km,  #type:ignore
                obstime=stereo_date,
                observer='self',
            )
            coords = coords.represent_as(astropy.coordinates.CartesianRepresentation)

            # CONVERSION to km
            result = np.array([
                coords.x.to(u.km).value,  #type:ignore
                coords.y.to(u.km).value,  #type:ignore
                coords.z.to(u.km).value,  #type:ignore
            ])
            output_queue.put((identification, result))
    
    @Decorators.running_time
    def raw_group(self, H5PYFile: h5py.File, borders: dict[str, dict[str, str | float]]) -> None:
        """
        To create the initial h5py.Group object; where the raw data (with/without feet and/or in
        Carrington Heliographic Coordinates).

        Args:
            H5PYFile (h5py.File): the opened file pointer.
            borders (dict[str, dict[str, str | float]]): the border info (i.e. x_min, etc.) for the
                given data.
        """

        # DATA
        data = self.raw_cubes()

        # GROUP create
        group = H5PYFile.create_group('Raw')
        group.attrs['description'] = (
            "The filament voxels in sparse COO format (i.e. with a coords and values arrays) of "
            "the initial cubes gotten from Dr. Karine Bocchialini's work.\nFurthermore, the "
            "necessary information to be able to position the filament relative to the Sun are "
            "also available. Both cubes, with or without feet, could be inside this group."
        )

        # GROUP raw cubes
        group = self.add_cube(group, data, 'Raw cubes', borders=borders)
        group['Raw cubes'].attrs['description'] = (
            "The initial voxel data in COO format without the feet for the polynomial."
        )
        group['Raw cubes/values'].attrs['description'] = (
            "The values for each voxel as a 1D numpy array, in the same order than the "
            "coordinates given in the 'coords' group.\nEach specific bit set to 1 in these uint8 "
            "values represent a specific information on the voxel. Hence, you can easily filter "
            "these values given which bit is set to 1.\nThe value information is as follows:\n"
            "-0b" + "0" * 7 + "1 represents the intersection between the STEREO and SDO point of "
            "views regardless if the voxel is a duplicate or not.\n"
            "-0b" + "0" * 6 + "10 represents the no duplicates regions from STEREO's point of "
            "view.\n"
            "-0b" + "0" * 5 + "100 represents the no duplicates regions from SDO's point of "
            "view.\n"
            "-0b" + "0" * 4 + "1000 represents the no duplicates data from STEREO's point of "
            "view. This takes into account if a region has a bifurcation.\n"
            "-0b0001" + "0" * 4 + " represents the no duplicates data from SDO's point of view. "
            "This takes into account if a region has a bifurcation.\n"
            "-0b001" + "0" * 5 + " gives out the feet positions.\n"
            "It is important to note the difference between the second/third bit being set to 1 "
            "or the fourth/fifth bit being set to 1. For the fourth and fifth bit, the definition "
            "for a duplicate is more strict. It also takes into account if, in the same region "
            "(i.e. block of voxels that are in contact), there is a bifurcation. If each branch "
            "of the bifurcation can be duplicates, then both branches are taken out (not true for "
            "the second/third bits as the duplicate treatment is done region by region.\n"
            "Furthermore, it is of course also possible to mix and match to get more varied "
            "datasets, e.g. -0b00011000 represents the no duplicates data."
        )

        if self.full:
            # GROUP raw skycoords
            group = self.add_skycoords(group, data, 'Raw coordinates', borders)
            group['Raw coordinates'].attrs['description'] = (
                'The initial data saved as Carrington Heliographic Coordinates in km.'
            )

        if not self.no_feet:
            # GROUP raw cubes with feet
            data, borders = self.with_feet(data, borders)
            group = self.add_cube(group, data, 'Raw cubes with feet', borders=borders)
            group['Raw cubes with feet'].attrs['description'] = (
                'The initial raw data in COO format with the feet positions added.'
            )
            group['Raw cubes with feet/values'].attrs['description'] = (
                group['Raw cubes/values'].attrs['description']
            )

            if self.full:
                # GROUP raw skycoords with feet
                group = self.add_skycoords(group, data, 'Raw coordinates with feet', borders)
                group['Raw coordinates with feet'].attrs['description'] = (
                    "The initial data with the feet positions added saved as Carrington "
                    "Heliographic Coordinates in km."
                )
    
    @Decorators.running_time
    def filtered_group(
            self,
            H5PYFile: h5py.File,
            borders: dict[str, dict[str, str | float]],
        ) -> None:
        """
        To filter the data and save it with feet.

        Args:
            H5PYFile (h5py.File): the file object.
            borders (dict[str, dict[str, str | float]]): the border information.
        """

        # GROUP create
        group = H5PYFile.create_group('Filtered')
        group.attrs['description'] = (
            "This group is based on the data from the 'Raw' HDF5 group. It is made up of already "
            "filtered data for easier use later but also to be able to add a weight to the feet. "
            "Hence, the polynomial data for each filtered data group is also available."
        )

        # DATA
        data = self.get_COO(H5PYFile, f'Raw/Raw cubes').astype('uint8')

        for option in self.feet_options:
            # ADD all data
            new_borders = borders.copy()
            filtered_data = (data & 0b00000001).astype('uint8')
            if option != '': filtered_data, new_borders = self.with_feet(filtered_data, borders)
            group = self.add_cube(
                group=group,
                data=filtered_data,
                data_name=f'All data{option}',
                values=1 if option=='' else None,
                borders=new_borders,
            )
            group[f'All data{option}'].attrs['description'] = (
                f"All data, i.e. the 0b00000001 filtered data{option}. Hence, the data represents "
                "the intersection between the STEREO and SDO point of views.\n"
                + (
                    f"The feet are saved with a value corresponding to 0b00100000."
                    if option != '' else ''
                )
            )

            # ADD no duplicates
            filtered_data = ((data & 0b00011000) == 0b00011000).astype('uint8')
            if option != '': filtered_data, new_borders = self.with_feet(filtered_data, borders)
            group = self.add_cube(
                group=group,
                data=filtered_data,
                data_name=f'No duplicates{option}',
                values=1 if option=='' else None,
                borders=new_borders,
            )
            group[f'No duplicates{option}'].attrs['description'] = (
                f"The new no duplicates data, i.e. the 0b00011000 filtered data{option}. Hence, "
                "the data represents all the data without any of the duplicates. Even the "
                "bifurcations are taken into account. No duplicates should exist in this "
                "filtering.\n"
                + (
                    f"The feet are saved with a value corresponding to 0b00100000."
                    if option != '' else ''
                )
            )

            # ADD SDO los data
            if option != '': continue
            filtered_data = ((data & 0b10000000) == 0b10000000).astype('uint8')
            group = self.add_cube(
                group=group,
                data=filtered_data,
                data_name=f'SDO line of sight',
                values=1,
                borders=new_borders,
            )
            group[f'SDO line of sight'].attrs['description'] = (
                "The SDO line of sight data, i.e. the 0b01000000 filtered data. Hence, this data "
                "represents what is seen by SDO if represented in 3D inside the space of the "
                "rainbow cube data. The limits of the borders are defined in the .save IDL code "
                "named new_toto.pro created by Dr. Frederic Auchere."
            )

            # ADD STEREO los data
            filtered_data = ((data & 0b01000000) == 0b01000000).astype('uint8')
            group = self.add_cube(
                group=group,
                data=filtered_data,
                data_name=f'STEREO line of sight',
                values=1,
                borders=new_borders,
            )
            group[f'STEREO line of sight'].attrs['description'] = (
                "The STEREO line of sight data, i.e. the 0b01000000 filtered data. Hence, this "
                "data represents what is seen by STEREO if represented in 3D inside the space of "
                "the rainbow cube data. The limits of the borders are defined in the .save IDL "
                "code named new_toto.pro created by Dr. Frederic Auchere."
            )
    
    @Decorators.running_time
    def integrated_group(
            self,
            H5PYFile: h5py.File,
            borders: dict[str, dict[str, str | float]],
        ) -> None:
        """
        To integrate the data and save it inside a specific HDF5 group.

        Args:
            H5PYFile (h5py.File): the HDF5 file.
            borders (dict[str, dict[str, str | float]]): the border information.
        """

        # GROUP setup
        group = H5PYFile.create_group('Time integrated')
        group.attrs['description'] = (
            "This group has already time integrated data for some of the main data filtering.\n"
            "This was created for ease of use when further analysing the structures."
        )

        # OPTIONs
        data_options = [
            f'{data_type}{feet}'
            for data_type in ['All data', 'No duplicates']
            for feet in self.feet_options
        ]

        for option in data_options:
            # GROUP create
            inside_group = group.create_group(option)
            inside_group.attrs['description'] = (
                f"This group only contains {option.lower()} time integrated data.\n"
                f"To get border info, please refer to the Filtered/{option} data group."
                + (
                    "Furthermore, the feet are saved with values equal to 0b00100000."
                    if 'with feet' in option else ''
                )
            )

            # INTEGRATION full
            group_name = 'Full integration'
            data, new_borders = self.full_integration(
                H5PYFile=H5PYFile,
                datapath=f'Filtered/{option}',
                borders=borders,
            )

            # ADD data
            inside_group = self.add_cube(
                group=inside_group,
                data=data,
                data_name=group_name,
                values=None if 'with feet' in option else 1,
                borders=new_borders,
            )
            inside_group[group_name].attrs['description'] = (
                f"This group contains the {option.lower()} data fully integrated."
            )
            
            # INTEGRATION time dependent
            for integration_time in self.integration_time:
                # INTEGRATION setup
                time_hours = int(integration_time / 3600)
                group_name = f'Time integration of {time_hours} hours'

                # DATA
                data, new_borders = self.time_integration(
                    H5PYFile=H5PYFile,
                    datapath=f'Filtered/{option}',
                    time=integration_time,
                    borders=borders, 
                )

                # ADD data
                inside_group = self.add_cube(
                    group=inside_group,
                    data=data,
                    data_name=group_name,
                    values=None if 'with feet' in option else 1,
                    borders=new_borders,
                )
                inside_group[group_name].attrs['description'] = (
                    f"This group contains the {option.lower()} data integrated on {time_hours} "
                    "hours intervals."
                )              

    def full_integration(
            self,
            H5PYFile: h5py.File,
            datapath: str,
            borders: dict[str, dict[str, str | float]],
        ) -> tuple[sparse.COO, dict[str, dict[str, str | float]]]:
        """
        To integrate all the data from a given data group.

        Args:
            H5PYFile (h5py.File): the HDF5 file.
            datapath (str): the path to the data group to be integrated.
            borders (dict[str, dict[str, str | float]]): the border information.

        Returns:
            tuple[sparse.COO, dict[str, dict[str, str | float]]]: the integrated data and the new
                corresponding data borders.
        """
    
        # DATA
        data = self.get_COO(H5PYFile, datapath.removesuffix(' with feet'))

        # INTEGRATION full
        integration = sparse.COO.any(data, axis=0)
        return integration, borders  # ? is the border choice the right one ?

    @Decorators.running_time
    def time_integration(
            self,
            H5PYFile: h5py.File,
            datapath: str,
            time: int,
            borders: dict[str, dict[str, str | float]],
        ) -> tuple[sparse.COO, dict[str, dict[str, str | float]]]:
        """
        Gives the time integration of all the data for a given time interval in seconds.

        Args:
            H5PYFile (h5py.File): the HDF5 file.
            datapath (str): the datapath to the data to be integrated.
            time (int): the integration time (in seconds).
            borders (dict[str, dict[str, str | float]]): the border information.

        Returns:
            tuple[sparse.COO, dict[str, dict[str, str | float]]]: the integration data and the new
                corresponding data borders.
        """

        # DATA
        data = self.get_COO(H5PYFile, datapath.removesuffix(' with feet'))

        # MULTIPROCESSING setup
        manager = mp.Manager()
        input_queue = manager.Queue()
        output_queue = manager.Queue()
        for i in range(self.max_len): input_queue.put((i, data, self.dates_seconds, time))
        for _ in range(self.nb_processes): input_queue.put(None)

        # RUN processes
        processes: list[mp.Process] = cast(list[mp.Process], [None] * self.nb_processes)
        for i in range(self.nb_processes):
            p = mp.Process(target=self.time_integration_sub, args=(input_queue, output_queue))
            p.start()
            processes[i] = p
        for p in processes: p.join()
        
        # RESULTs formatting 
        data_list: list[sparse.COO] = cast(list[sparse.COO], [None] * self.max_len)
        while not output_queue.empty():
            identification, result = output_queue.get()
            data_list[identification] = result
        data: sparse.COO = cast(sparse.COO, sparse.stack(data_list, axis=0).astype('uint8'))

        # BORDERs update
        if 'with feet' in datapath:
            data, new_borders = self.with_feet(data, borders)
        else: 
            new_borders = borders.copy()
        return data, new_borders

    @staticmethod
    def time_integration_sub(
            input_queue: ManagerQueueProxy,
            output_queue: ManagerQueueProxy,
        ) -> None:
        """
        To multiprocess the time integration of the cubes. This does it for each given date.

        Args:
            input_queue (ManagerQueueProxy): the input arguments in a mp.Manager.Queue().
            output_queue (ManagerQueueProxy): the results in a np.Manager.Queue().
        """

        while True:
            # CHECK queue
            arguments = input_queue.get()
            if arguments is None: return
            index, data, dates, integration_time = arguments
            date = dates[index]

            date_min = date - integration_time / 2
            date_max = date + integration_time / 2

            # RESULTs save
            chunk = DataSaver.cube_integration(data, date_max, date_min, dates)
            output_queue.put((index, chunk))

    @staticmethod
    def cube_integration(
            data: sparse.COO,
            date_max: int,
            date_min: int,
            dates: list[int],
        ) -> sparse.COO:
        """
        To integrate the date cubes given the max and min date (in seconds) for the integration
        limits.

        Args:
            data (sparse.COO): the data to integrate.
            date_max (int): the maximum date (in seconds) for he integration.
            date_min (int): the minimum date (in seconds) for the integration.
            dates (list[int]): the dates of the cubes in seconds.

        Returns:
            sparse.COO: the integrated cubes.
        """

        chunk = []
        for date, cube in zip(dates, data):

            if date < date_min:
                continue
            elif date <= date_max:
                chunk.append(cube)
            else:
                break

        # CHECK if empty
        if chunk == []:
            return sparse.COO(coords=[], data=[], shape=data.shape[1:]).astype('uint8')
        elif len(chunk) == 1:
            return chunk[0]
        else:
            COO: sparse.COO = cast(sparse.COO, sparse.stack(chunk, axis=0))
            return sparse.COO.any(COO, axis=0)
        
    def get_COO(self, H5PYFile: h5py.File, group_path: str) -> sparse.COO:
        """
        To get the sparse.COO object from the corresponding coords and values.

        Args:
            H5PYFile (h5py.File): the file object.
            group_path (str): the path to the group where the data is stored.

        Returns:
            sparse.COO: the corresponding sparse data.
        """

        data_coords: np.ndarray = cast(h5py.Dataset, H5PYFile[group_path + '/coords'])[...]
        data_data: np.ndarray = cast(h5py.Dataset, H5PYFile[group_path + '/values'])[...]
        data_shape = np.max(data_coords, axis=1) + 1
        return sparse.COO(coords=data_coords, data=data_data, shape=data_shape)
    
    @Decorators.running_time
    def polynomial_group(self, H5PYFile: h5py.File) -> None:
        """
        To add the polynomial information to the file.

        Args:
            H5PYFile (h5py.File): the file object.
        """
        
        # OPTIONs
        data_options = [
            f'{data_type}{feet}'
            for data_type in ['All data', 'No duplicates']
            for feet in self.feet_options
        ]

        sub_options = [
            f'/Time integration of {int(time / 3600)} hours'
            for time in self.integration_time
        ] + ['/Full integration']
        
        # INTEGRATED data
        main_path_2 = 'Time integrated/'
        for main_option in data_options:
            for sub_option in sub_options:
                group_path = main_path_2 + main_option + sub_option
                data = self.get_COO(H5PYFile, group_path).astype('uint16')

                # ADD polynomial
                self.add_polynomial(cast(h5py.Group, H5PYFile[group_path]), data)

    @Decorators.running_time
    def raw_cubes(self) -> sparse.COO:
        """
        To get the initial raw cubes as a sparse.COO object.

        Returns:
            sparse.COO: the raw cubes.
        """

        # ! dates have changed so the rawCubes list indexes will be wrong too

        # SETUP multiprocessing
        cube_nb = len(self.cube_numbers)
        processes_nb = min(self.processes, cube_nb)
        manager = mp.Manager()
        input_queue = manager.Queue()
        output_queue = manager.Queue()

        # INPUT populate
        for cube_number in self.cube_numbers:
            input_queue.put((
                cube_number,
                os.path.join(self.paths['cubes'], f'cube{cube_number:03d}.save'),
            ))
        for _ in range(processes_nb): input_queue.put(None)

        # RUN processes
        processes: list[mp.Process] = cast(list[mp.Process], [None] * processes_nb)
        for i in range(processes_nb): 
            p = mp.Process(
                target=self.raw_cubes_sub,
                args=(input_queue, output_queue),
            )
            p.start()
            processes[i] = p
        for p in processes: p.join()

        # RESULTs formatting
        rawCubes_list: list[sparse.COO] = cast(list[sparse.COO], [None] * self.max_len)
        while not output_queue.empty():
            identifier, result = output_queue.get()
            rawCubes_list[identifier] = result
        
        # FILL NoneTypes
        rawCubes_list = [
            sparse.COO(
                coords=[],
                data=[],
                shape=rawCubes_list[self.cube_numbers[0]].shape,
            ).astype('uint8')
            if cube is None else cube  
            for cube in rawCubes_list
        ]
        rawCubes: sparse.COO = cast(sparse.COO, sparse.stack(rawCubes_list, axis=0))
        return rawCubes
    
    @staticmethod
    def raw_cubes_sub(input_queue: ManagerQueueProxy, output_queue: ManagerQueueProxy) -> None:
        """
        To get the raw data from each cube.

        Args:
            input_queue (ManagerQueueProxy): the input data (identifier, filepath) for each cube.
            output_queue (ManagerQueueProxy): the output data (identifier, data) for each cube.
        """

        while True:

            # CHECK inputs
            args: tuple[int, str] = input_queue.get()
            if args is None: return
            cube_number, filepath = args

            # DATA import
            data = scipy.io.readsav(filepath)
            cube: np.ndarray = data.cube

            # CHECK keywords
            if ('cube1' in data) and ('cube2' in data):
                # LOS data
                cube1: np.ndarray = data.cube1.astype('uint8') * 0b01000000
                cube2: np.ndarray = data.cube2.astype('uint8') * 0b10000000

                result = (cube + cube1 + cube2).astype('uint8')
            else:
                result = cube.astype('uint8')

            # SAVE
            result = np.transpose(result, (2, 1, 0))
            output_queue.put((cube_number, DataSaver.sparse_data(result)))

    @staticmethod
    def sparse_data(cubes: np.ndarray) -> sparse.COO:
        """
        Changes data to a sparse representation.

        Args:
            cubes (np.ndarray): the initial array.

        Returns:
            sparse.COO: the corresponding sparse COO array.
        """

        sparse_cubes = sparse.COO(cubes)
        # the .to_numpy() method wasn't used as the idx_type argument isn't working properly
        sparse_cubes.coords = sparse_cubes.coords.astype('uint16')  # to save RAM
        return sparse_cubes
    
    def add_skycoords(
            self,
            group: h5py.File | h5py.Group,
            data: sparse.COO,
            data_name: str,
            borders: dict[str, dict[str, str | float]],
        ) -> h5py.File | h5py.Group:
        """
        To add to an h5py.Group, the data and metadata of the Carrington Heliographic Coordinates
        for a corresponding cube index spare.COO object. 
        This takes also into account the border information.

        Args:
            group (h5py.File | h5py.Group): the Group where to add the data information.
            data (sparse.COO): the data that needs to be included in the file.
            data_name (str): the group name to be used in the file.
            borders (dict[str, dict[str, str | float]]): the data border information.

        Returns:
            h5py.File | h5py.Group: the updated group.
        """
        
        # SKYCOORDs setup
        skycoords = self.carrington_skyCoords(data, borders)
        data_list: list[np.ndarray] = cast(list[np.ndarray], [None] * len(skycoords))
        for i, skycoord in enumerate(skycoords):
            x: np.ndarray = skycoord.cartesian.x.value  #type:ignore
            y: np.ndarray = skycoord.cartesian.y.value  #type:ignore
            z: np.ndarray = skycoord.cartesian.z.value  #type:ignore
            cube = np.stack([x, y, z], axis=0)

            # (x, y, z) -> (t, x, y, z)
            time_row = np.full((1, cube.shape[1]), i)
            data_list[i] = np.vstack([time_row, cube])
        result: np.ndarray = np.hstack(data_list).astype('float32')

        # DATA formatting
        raw = {
            'description': "Default",
            'coords': {
                'data': result,
                'unit': 'km',
                'description': (
                    "The t, x, y, z coordinates of the cube voxels.\nThe shape is (4, N) where "
                    "the rows represent t, x, y, z where t the time index (i.e. which cube it "
                    "is), and N the total number of voxels. Furthermore, x, y, z, represent the "
                    "X, Y, Z axis Carrington Heliographic Coordinates.\n"
                ),
            },
        }
        
        # BORDERs add
        raw |= borders

        # ADD group
        self.add_group(group, raw, data_name)
        return group
    
    def add_polynomial(self, group: h5py.Group, data: sparse.COO) -> None:
        """
        To add to an h5py.Group, the polynomial curve and parameters given the data to fit.

        Args:
            group (h5py.Group): the group in which an polynomial group needs to be added.
            data (sparse.COO): the data to fit.

        Returns:
            h5py.Group: the updated group.
        """

        polynomial_kwargs = {
            'data': data, 
            'feet_sigma': self.feet_sigma,
            'south_sigma': self.south_leg_sigma,
            'leg_threshold': self.leg_threshold,
            'processes': self.processes,
            'precision_nb': self.polynomial_points,
            'full': self.full,
        }

        # N-ORDERs 
        for n_order in self.polynomial_order:
            instance = Polynomial(order=n_order, **polynomial_kwargs)

            info = instance.get_information()
            self.add_group(group, info, f'{n_order}th order polynomial')
    
    def with_feet(
            self,
            data: sparse.COO,
            borders: dict[str, dict[str, str | float]],  #type:ignore
        ) -> tuple[sparse.COO, dict[str, dict[str, str | float]]]:
        """
        Adds feet to a given initial cube index data as a sparse.COO object.

        Args:
            data (sparse.COO): the data to add feet to.
            borders (dict[str, dict[str, str | float]]): the initial data borders.

        Returns:
            tuple[sparse.COO, dict[str, dict[str, str | float]]]: the new_data with feet and the
                corresponding borders data and metadata.
        """

        # CHEAT type checking
        borders: dict[str, dict[str, float]] = cast(dict[str, dict[str, float]], borders)

        # COORDs feet
        x = self.feet.cartesian.x.to(u.km).value  #type:ignore
        y = self.feet.cartesian.y.to(u.km).value  #type:ignore
        z = self.feet.cartesian.z.to(u.km).value  #type:ignore
        positions = np.stack([x, y, z], axis=0)

        # BORDERs update
        x_min, y_min, z_min = np.min(positions, axis=1) 
        x_min: float = x_min if x_min <= borders['xt_min']['data'] else borders['xt_min']['data']
        y_min: float = y_min if y_min <= borders['yt_min']['data'] else borders['yt_min']['data']
        z_min: float = z_min if z_min <= borders['zt_min']['data'] else borders['zt_min']['data']
        new_borders = self.create_borders((x_min, y_min, z_min))

        # COORDs feet indexes
        positions[0, :] -= borders['xt_min']['data']
        positions[1, :] -= borders['yt_min']['data']
        positions[2, :] -= borders['zt_min']['data']
        positions /= self.dx['data']
        positions = np.round(positions).astype('int32')

        # CUBEs setup
        init_coords = data.coords
        # (x, y, z) -> (t, x, y, z)
        feet = np.hstack([
            np.vstack((np.full((1, 2), time), positions))
            for time in range(self.max_len)
        ])

        # ADD feet
        init_coords = np.hstack([init_coords, feet]).astype('int32')

        # INDEXEs positive
        x_min, y_min, z_min = np.min(positions, axis=1).astype(int)  
        if x_min < 0: init_coords[1, :] -= x_min
        if y_min < 0: init_coords[2, :] -= y_min  # ? not sure what I meant there
        if z_min < 0: init_coords[3, :] -= z_min

        # COO
        shape = np.max(init_coords, axis=1) + 1
        feet_values = np.repeat(np.array([0b00100000], dtype='uint8'), self.max_len * 2)
        values = np.concatenate([data.data, feet_values], axis=0)
        data = sparse.COO(coords=init_coords, data=values, shape=shape).astype('uint8')
        return data, new_borders

    def carrington_skyCoords(
            self,
            data: sparse.COO,
            borders: dict[str, dict[str, str | float]],  #type:ignore
        ) -> list[astropy.coordinates.SkyCoord]:
        """
        Converts sparse.COO cube index data to a list of corresponding astropy.coordinates.SkyCoord
        objects.

        Args:
            data (sparse.COO): the cube index data to be converted.
            borders (dict[str, dict[str, float]]): the input data border information.

        Returns:
            list[coordinates.SkyCoord]: corresponding list of the coordinates for the cube index
                data. They are in Carrington Heliographic Coordinates. 
        """

        # CHEAT type checking
        borders: dict[str, dict[str, float]] = cast(dict[str, dict[str, float]], borders)
        
        # COORDs
        coords = data.coords.astype('float64')

        # CONVERSION heliocentric in km
        coords[1, :] = coords[1, :] * self.dx['data'] + borders['xt_min']['data']
        coords[2, :] = coords[2, :] * self.dx['data'] + borders['yt_min']['data']
        coords[3, :] = coords[3, :] * self.dx['data'] + borders['zt_min']['data']

        # SETUP multiprocessing
        manager = mp.Manager()
        input_queue = manager.Queue()
        output_queue = manager.Queue()
        shm, coords = MultiProcessing.create_shared_memory(coords)
        for i in range(self.max_len): input_queue.put(i)
        for _ in range(self.nb_processes): input_queue.put(None)

        # RUN processes
        processes: list[mp.Process] = cast(list[mp.Process], [None] * self.nb_processes)
        for i in range(self.nb_processes):
            process = mp.Process(
                target=self.skyCoords_slice,
                args=(coords, input_queue, output_queue),
            )
            process.start()
            processes[i] = process
        for p in processes: p.join()
        shm.unlink()

        # RESULTs formatting
        all_SkyCoords: list[astropy.coordinates.SkyCoord] = cast(
            list[astropy.coordinates.SkyCoord],
            [None] * self.max_len,
        )
        while not output_queue.empty():
            identifier, result = output_queue.get()
            all_SkyCoords[identifier] = result
        return all_SkyCoords
    
    @staticmethod
    def skyCoords_slice(
            coords_dict: dict[str, Any],
            input_queue: ManagerQueueProxy,
            output_queue: ManagerQueueProxy,
        ) -> None:
        """
        To create an astropy.coordinates.SkyCoord object for a singular cube (i.e. for a unique
        time index).

        Args:
            coords_dict (dict[str, Any]): information to find the sparse.COO(data).coords
                multiprocessing.shared_memory.SharedMemory object.
            input_queue (ManagerQueueProxy): multiprocessing.Manager.Queue object used for the
                function inputs.
            output_queue (ManagerQueueProxy): multiprocessing.Manager.Queue object used to extract
                the function results.
        """
        
        # DATA open
        shm, coords = cast(
            tuple[SharedMemoryAlias, np.ndarray],
            MultiProcessing.open_shared_memory(coords_dict)
        )

        while True:
            # CHECK queue
            index = input_queue.get()
            if index is None: break

            # DATA section
            slice_filter: np.ndarray = coords[0, :] == index
            cube: np.ndarray = coords[:, slice_filter]
            
            # COORDs reprojected carrington
            skyCoord = astropy.coordinates.SkyCoord(
                cube[1, :], cube[2, :], cube[3, :], 
                unit=u.km,  #type:ignore
                frame=sunpy.coordinates.frames.HeliographicCarrington,
                representation_type='cartesian'
            )
            
            # SAVE
            output_queue.put((index, skyCoord))
        shm.close()



if __name__=='__main__':

    instance = DataSaver(
        integration_time=[12, 18, 24],
        polynomial_order=[4, 6],
        feet_sigma=20,
        south_leg_sigma=20,
        leg_threshold=0.03,
        full=False,
        no_feet=True,
        compression=True,
        compression_lvl=9,
        fake_hdf5=False,
    )
    instance.create()
