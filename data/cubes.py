"""
To create the HDF5 data files with the cubes data and metadata.
A lot of different data is saved in the file to make any further manipulation or visualisation more
easy.
"""

# IMPORTS
import os
import re
import h5py
import scipy
import sunpy
import sparse
import astropy

# IMPORTs alias
import numpy as np
import multiprocessing as mp

# IMPORTs sub
import sunpy.coordinates
import multiprocessing.queues
from astropy import units as u
from typing import Any, Callable

# IMPORTs personal
from data.get_polynomial import Polynomial
from data.base_hdf5_creator import VolumeInfo, BaseHDF5Protuberance
from common import Decorators, CustomDate, DatesUtils, MultiProcessing, root_path

# todo I could change the code so that one process runs only one cube at once (except for the time
# integration part where I need to only take the data section needed). This will need to change the
# whole fetching, cube creating and saved data structure, so holding it off for now

# todo need to change 'Coords' to 'Coords indexes' when I decide to re-run this code.



class DataSaver(BaseHDF5Protuberance):
    """
    To create cubes with and/or without feet in an HDF5 file.
    """

    @Decorators.running_time
    def __init__(
        self,
        filename: str,
        processes: int, 
        integration_time: int | list[int] = [24],
        polynomial_points: float = 10**6, 
        polynomial_order: int | list[int] = [2, 3, 4],
        feet_lonlat: tuple[tuple[int | float, int | float], ...] = ((-177, 14.5), (-163.5, -16.5)),
        feet_sigma: int | float = 1e-4,
        south_leg_sigma: int | float = 5,
        leg_threshold: float = 0.03,
        only_feet: bool = True,
        full: bool = False,
        fake_hdf5: bool = False, 
    ) -> None:
        """  #TODO: update docstring
        To create the cubes with and/or without feet in an HDF5 file.

        Args:
            filename (str): the filename of the file to be saved.
            processes (int): the number of processes used in the multiprocessed parts
            integration_time (int | list[int], optional): the time or times in hours used in the
                time integration of the data. Defaults to [24].
            polynomial_points (float, optional): the number of points used when recreating the
                polynomial gotten from the curve fitting of the data. Defaults to 10**6.
            polynomial_order (int | list[int], optional): the order or orders used for the
                polynomial that fits the data. Defaults to [4, 5, 6].
            feet_lonlat (tuple[tuple[int, int], ...], optional): the positions of the feet in
                Heliographic Carrington. Defaults to ((-177, 15), (-163, -16)).
            feet_sigma (float, optional): the sigma uncertainty in the feet used during the curve
                fitting of the data points. Defaults to 1e-4.
            full (bool, optional): deciding to save all the data. In the case when 'full' is True,
                the raw coordinates of the polynomial curve are also saved, where as only the
                indexes that can be directly used as coords in a sparse.COO object.
                Defaults to False.
        """

        # PARENT
        super().__init__()

        # Initial attributes
        self.filename = filename
        self.processes = processes
        if isinstance(integration_time, int):
            self.integration_time = [integration_time * 3600]
        else:
            self.integration_time = [time * 3600 for time in integration_time]
        if isinstance(polynomial_order, list):
            self.polynomial_order = polynomial_order
        else:
            self.polynomial_order = [polynomial_order]
        self.polynomial_points = polynomial_points
        self.feet_sigma = feet_sigma
        self.south_leg_sigma = south_leg_sigma
        self.leg_threshold = leg_threshold
        self.feet_options = [' with feet'] if only_feet else ['', ' with feet']
        self.full = full  # deciding to add the heavy sky coords arrays.
        self.fake_hdf5 = fake_hdf5

        # CONSTANTs
        self.max_cube_numbers = 413  # ? kind of weird I hard coded this

        # Attributes setup
        self.feet = self.setup_feet(feet_lonlat)
        self.cube_pattern, self.date_pattern = self.setup_patterns()
        self.dx: dict[str, str | float]  # information and value of the spatial resolution
        self.time_indexes: list[int]  # the time indexes with values inside the cubes

        # Created attributes
        self.setup_attributes()

    def setup_path(self) -> dict[str, str]:
        """
        Gives the directory paths as a dictionary.

        Returns:
            dict[str, str]: the directory paths.
        """

        # SETUP
        main = os.path.join(root_path, '..')

        # PATHs keep
        paths = {
            'main': main,
            'codes': root_path,
            'cubes': os.path.join(main, 'Cubes_karine'),
            'intensities': os.path.join(main, 'STEREO', 'int'),
            'sdo': os.path.join(main, 'sdo'),
            'save': os.path.join(root_path, 'data'),
        }

        # PATHS change
        if self.fake_hdf5:
            paths['cubes'] = os.path.join(root_path, 'data', 'fake_data', 'cubes_fake')
            paths['save'] = os.path.join(root_path, 'data', 'fake_data')
        
        # PATHS check
        for key in ['save']: os.makedirs(paths[key], exist_ok=True)
        return paths
    
    def setup_patterns(self) -> tuple[re.Pattern[str], re.Pattern[str]]:
        """
        The regular expression patterns used.

        Returns:
            tuple[re.Pattern[str], re.Pattern[str]]: the regular expression patterns for the .save
                cubes and the intensities STEREO B 30.4nm.
        """

        # Patterns
        cube_pattern = re.compile(r'cube(\d{3})\.save')
        date_pattern = re.compile(
            r'(?P<number>\d{4})_(?P<date>\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2})\.\d{3}\.png'
        )
        return cube_pattern, date_pattern
    
    def setup_attributes(self):
        """
        Multiple instance attributes are defined here. Function is only here to not flood the
        __init__ method.
        """
        
        # Paths
        self.paths = self.setup_path()

        # Get cubes filepaths
        self.filepaths = sorted([
            os.path.join(self.paths['cubes'], name)
            for name in os.listdir(self.paths['cubes'])
            if self.cube_pattern.match(name)
        ])

        # Get cube numbers
        self.cube_numbers = [
            int(self.cube_pattern.match(os.path.basename(filepath)).group(1))
            for filepath in self.filepaths
        ]

        # Get dates
        date_filenames = sorted(os.listdir(self.paths['intensities']))
        dates: list[str] = [None] * self.max_cube_numbers
        for i, number in enumerate(range(self.max_cube_numbers)):
            for filename in date_filenames:
                filename_match = self.date_pattern.match(filename)
                if filename_match:
                    if int(filename_match.group('number')) == number:
                        dates[i] = filename_match.group('date')
                        break
        self.dates = dates

        # Get pretreated dates
        treated_dates = [CustomDate(self.dates[number]) for number in self.cube_numbers]
        year = treated_dates[0].year
        days_per_month = DatesUtils.days_per_month(year)
        self.dates_seconds = [
            ((
                ((days_per_month[date.month] + date.day) * 24 + date.hour) * 60 + date.minute
            ) * 60 + date.second)
            for date in treated_dates
        ]

    def setup_feet(
            self,
            lonlat: tuple[tuple[int | float, int | float], ...],
        ) -> astropy.coordinates.SkyCoord:
        """
        Gives the 2 feet positions as an astropy.coordinates.SkyCoord object in Carrington
        Heliographic Coordinates. 

        Args:
            lonlat (tuple[tuple[int, int], ...]): the longitude and latitude positions for the
                added feet (i.e. ((lon1, lat1), (lon2, lat2))).

        Returns:
            coordinates.SkyCoord: the SkyCoord for the feet.
        """

        # Setup feet ndarray
        feet_pos = np.empty((3, 2), dtype='float64')
        feet_pos[0, :] = np.array([lonlat[0][0], lonlat[1][0]])
        feet_pos[1, :] = np.array([lonlat[0][1], lonlat[1][1]])
        feet_pos[2, :] = self.solar_r

        # Creating the feet
        feet = astropy.coordinates.SkyCoord(
            feet_pos[0, :] * u.deg,
            feet_pos[1, :] * u.deg,
            feet_pos[2, :] * u.km,
            frame=sunpy.coordinates.frames.HeliographicCarrington,
        )
        cartesian_feet = feet.represent_as(astropy.coordinates.CartesianRepresentation)
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

        # Add metadata
        cube_numbers_info = {
            'data': np.array(self.cube_numbers).astype('uint16'),
            'unit': 'none',
            'description': (
                "The time indexes for the data cubes that are used in this file. This value is "
                "used to filter which dates (using the dates dataset) and where the satellite is "
                "positioned (using the 'SDO positions' and 'STEREO B positions' datasets)."
            ),
        }
        cube_dates_info = {
            'data': np.array(self.dates).astype('S19'),
            'unit': 'none',
            'description': (
                "The dates of the STEREO B 30.4nm acquisitions. These represent all the possible "
                "dates, and as such, to get the specific date for each data cube used the time "
                "index dataset needs to be used (something like Dates[Time indexes] will give you "
                "the right dates if in 0 indexing)."
            ),
        }
        
        # Reformat for ease of use
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
            cube = scipy.io.readsav(self.filepaths[0])
            self.volume = VolumeInfo(
                dx=float(cube.dx),
                xt_min=float(cube.xt_min),
                yt_min=float(cube.yt_min),
                zt_min=float(cube.zt_min),
            )
            self.dx = self.dx_to_dict()
            values = (cube.xt_min, cube.yt_min, cube.zt_min)
            init_borders = self.create_borders(values)

            # Main metadata
            self.foundation(H5PYFile)

            # Raw data and metadata
            self.raw_group(H5PYFile, init_borders)

            # Filtered data and metadata
            self.filtered_group(H5PYFile, init_borders)

            if not self.fake_hdf5:
                # Integration data and metadata
                self.integrated_group(H5PYFile, init_borders)

                # Polynomial data and metadata
                self.polynomial_group(H5PYFile)
        
    def foundation(self, H5PYFile: h5py.File) -> None:
        """
        For the main file metadata before getting to the HDF5 datasets and groups.

        Args:
            H5PYFile (h5py.File): the file.
        """

        description = (
            "Contains the data cubes for the Solar Rainbow event gotten from the intersection of "
            "masks gotten from SDO and STEREO images.The SDO masks were created from an automatic "
            "code created by Dr. Elie Soubrie, while the STEREO masks where manually created by "
            "Dr. Karine Bocchialini by visual interpretation of the 30.4nm STEREO B acquisitions."
            "\nNew values for the feet where added to help for a curve fitting of the filament. "
            "These were added by looking at the STEREO B [...] nm images as the ends of the "
            "filament are more visible. Hence, the feet are not actually visible in the initial "
            "masks.\nExplanation on what each HDF5 group or dataset represent is given in the "
            "corresponding 'description' attribute."
        )
        #TODO: finish explanation here explain that the data is saved as sparse arrays.
        metadata = self.main_metadata()
        metadata['description'] += description

        # METADATA
        meta_info = self.get_cube_dates_info()
        sdo_info = self.get_pos_sdo_info()
        stereo_info = self.get_pos_stereo_info()

        # FILE update
        self.add_dataset(H5PYFile, metadata)
        self.add_dataset(H5PYFile, self.dx, 'dx')
        for key in meta_info.keys(): self.add_dataset(H5PYFile, meta_info[key], key)
        self.add_dataset(H5PYFile, sdo_info, 'SDO positions')
        self.add_dataset(H5PYFile, stereo_info, 'STEREO B positions')
    
    @Decorators.running_time
    def get_pos_sdo_info(self) -> dict[str, str | np.ndarray]:
        """
        Gives the SDO satellite position information in Cartesian Heliocentric Coordinates.

        Returns:
            dict[str, str | np.ndarray]: the data and metadata for the SDO satellite position
        """

        # Get data
        SDO_fits_names = [
            os.path.join(self.paths['sdo'], f'AIA_fullhead_{number:03d}.fits.gz')
            for number in range(self.max_cube_numbers)
        ]

        # Get results
        coordinates = self.get_pos_code(SDO_fits_names, self.get_pos_sdo_sub)

        # Add metadata
        information = {
            'data': coordinates.astype('float32'),
            'unit': 'km',
            'description': (
                "The position of the SDO satellite during the observations in cartesian "
                "heliocentric coordinates.\nThe shape of the data is (413, 3) where 413 "
                "represents the time indexes for the data and the 3 the x, y, z position of the "
                "satellite. To find the right position for the right data cube, you need to use "
                "the 'Time indexes' dataset as they represent which time indexes we have usable "
                "data."
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

        # Get data
        stereo_information = scipy.io.readsav(
            os.path.join(self.paths['main'], 'rainbow_stereob_304.save')
        ).datainfos

        # Get results
        coordinates = self.get_pos_code(stereo_information, self.get_pos_stereo_sub)

        # Add metadata
        information = {
            'data': coordinates.astype('float32'),
            'unit': 'km',
            'description': (
                "The position of the STEREO B satellite during the observations in cartesian "
                "heliocentric coordinates.\nThe shape of the data is (413, 3) where 413 "
                "represents the time indexes for the data and the 3 the x, y, z position of the "
                "satellite. To find the right position for the right data cube, you need to use "
                "the 'Time indexes' dataset as they represent which time indexes we have usable "
                "data."
            ),          
        }
        return information

    def get_pos_code(
            self,
            data: np.recarray | list[str],
            function: Callable[[], np.ndarray]
        ) -> np.ndarray:
        """
        To multiprocess the getting of the positions of the SDO and STEREO B satellites.

        Args:
            data (np.recarray | list[str]): the data information for SDO or STEREO B.
            function (typing.Callable[[], np.ndarray]): the function used for each process to get
                the position of the satellite.

        Returns:
            np.ndarray: the position of the satellite in cartesian heliocentric coordinates.
        """

        # Setup
        nb_processes = min(self.processes, self.max_cube_numbers)
        manager = mp.Manager()
        input_queue = manager.Queue()
        output_queue = manager.Queue()
        for i in range(self.max_cube_numbers): input_queue.put((i, data[i]))
        for _ in range(nb_processes): input_queue.put(None)
        # Run
        processes: list[mp.Process] = [None] * nb_processes
        for i in range(nb_processes):
            p = mp.Process(target=function, args=(input_queue, output_queue))
            p.start()
            processes[i] = p
        for p in processes: p.join()
        # Get results
        coordinates: list[np.ndarray] = [None] * self.max_cube_numbers
        while not output_queue.empty():
            identifier, result = output_queue.get()
            coordinates[identifier] = result
        coordinates = np.stack(coordinates, axis=0)
        return coordinates
    
    @staticmethod
    def get_pos_sdo_sub(input_queue: mp.queues.Queue, output_queue: mp.queues.Queue) -> None:
        """
        To get the position of the SDO satellite.

        Args:
            input_queue (mp.queues.Queue): the input information (list[tuple[int, str]]) for
                identification and SDO information.
            output_queue (mp.queues.Queue): to save the results outside the function.
        """
        
        while True:
            # Get args
            arguments = input_queue.get()
            if arguments is None: return
            identification, filepath = arguments

            # Get file header
            header = astropy.io.fits.getheader(filepath)
            # Get positions
            coords = sunpy.coordinates.frames.HeliographicCarrington(
                header['CRLN_OBS'] * u.deg,
                header['CRLT_OBS'] * u.deg,
                header['DSUN_OBS'] * u.m,
                obstime=header['DATE-OBS'],
                observer='self',
            )
            coords = coords.represent_as(astropy.coordinates.CartesianRepresentation)

            # In km
            result = np.array([
                coords.x.to(u.km).value,
                coords.y.to(u.km).value,
                coords.z.to(u.km).value,
            ])
            output_queue.put((identification, result))
        
    @staticmethod
    def get_pos_stereo_sub(input_queue: mp.queues.Queue, output_queue: mp.queues.Queue) -> None:
        """
        To get the position of the STEREO B satellite.

        Args:
            input_queue (mp.queues.Queue): the input information (list[tuple[int, str]]) for
                identification and SDO information.
            output_queue (mp.queues.Queue): to save the results outside the function.
        """

        while True:
            # Get args
            arguments = input_queue.get()
            if arguments is None: return
            identification, information_recarray = arguments
            
            # Get positions
            date = CustomDate(information_recarray.strdate)
            stereo_date = (
                f'{date.year}-{date.month}-{date.day}T{date.hour}:{date.minute}:{date.second}'
            )
            coords = sunpy.coordinates.frames.HeliographicCarrington(
                information_recarray.lon * u.deg,
                information_recarray.lat * u.deg,
                information_recarray.dist * u.km,
                obstime=stereo_date,
                observer='self',
            )
            coords = coords.represent_as(astropy.coordinates.CartesianRepresentation)

            # In km
            result = np.array([
                coords.x.to(u.km).value,
                coords.y.to(u.km).value,
                coords.z.to(u.km).value,
            ])
            output_queue.put((identification, result))
    
    @Decorators.running_time
    def raw_group(self, H5PYFile: h5py.File, borders: dict[str, dict[str, str | float]]):
        """
        To create the initial h5py.Group object; where the raw data (with/without feet and/or in
        Carrington Heliographic Coordinates).

        Args:
            H5PYFile (h5py.File): the opened file pointer.
            borders (dict[str, dict[str, str | float]]): the border info (i.e. x_min, etc.) for the
                given data.
        """

        # Get data
        data = self.raw_cubes()

        # Setup group
        group = H5PYFile.create_group('Raw')
        group.attrs['description'] = (
            "The filament voxels in sparse COO format (i.e. with a coords and values arrays) of "
            "the initial cubes gotten from Dr. Karine [...]'s work.\n Furthermore, the necessary "
            "information to be able to position the filament relative to the Sun are also "
            "available. Both cubes, with or without feet, are inside this group."
        )

        # Add raw cubes group
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
            # Add raw skycoords group
            group = self.add_skycoords(group, data, 'Raw coordinates', borders)
            group['Raw coordinates'].attrs['description'] = (
                'The initial data saved as Carrington Heliographic Coordinates in km.'
            )

        # Add raw feet
        data, borders = self.with_feet(data, borders)
        group = self.add_cube(group, data, 'Raw cubes with feet', borders=borders)
        group['Raw cubes with feet'].attrs['description'] = (
            'The initial raw data in COO format with the feet positions added.'
        )
        group['Raw cubes with feet/values'].attrs['description'] = (
            group['Raw cubes/values'].attrs['description']
        )

        if self.full:
            # Add raw skycoords with feet
            group = self.add_skycoords(group, data, 'Raw coordinates with feet', borders)
            group['Raw coordinates with feet'].attrs['description'] = (
                "The initial data with the feet positions added saved as Carrington Heliographic "
                "Coordinates in km."
            )
    
    @Decorators.running_time
    def filtered_group(self, H5PYFile: h5py.File, borders: dict[str, dict[str, str | float]]):
        """
        To filter the data and save it with feet.

        Args:
            H5PYFile (h5py.File): the file object.
            borders (dict[str, dict[str, str | float]]): the border information.
        """

        # Setup group
        group = H5PYFile.create_group('Filtered')
        group.attrs['description'] = (
            "This group is based on the data from the 'Raw' HDF5 group. It is made up of already "
            "filtered data for easier use later but also to be able to add a weight to the feet. "
            "Hence, the polynomial data for each filtered data group is also available."
        )

        # Get data
        data = self.get_COO(H5PYFile, f'Raw/Raw cubes').astype('uint8')

        for option in self.feet_options:
            # Add all data
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

            # Add no duplicates init
            filtered_data = ((data & 0b00000110) == 0b00000110).astype('uint8')
            #TODO: the shape of the resulting data is weird, no clue why.
            if option != '': filtered_data, new_borders = self.with_feet(filtered_data, borders)
            group = self.add_cube(
                group=group,
                data=data,
                data_name=f'No duplicates init{option}',
                values=1 if option=='' else None,
                borders=new_borders,
            )
            group[f'No duplicates init{option}'].attrs['description'] = (
                f"The initial no duplicates data, i.e. the 0b00000110 filtered data{option}. "
                "Hence, the data represents all the data without the duplicates, without taking "
                "into account if there are bifurcations in some of the regions. Therefore, some "
                "duplicates might still exist using this filtering.\n"
                + (
                    f"The feet are saved with a value corresponding to 0b00100000."
                    if option != '' else ''
                )
            )

            # Add no duplicates new
            filtered_data = ((data & 0b00011000) == 0b00011000).astype('uint8')
            if option != '': filtered_data, new_borders = self.with_feet(filtered_data, borders)
            group = self.add_cube(
                group=group,
                data=filtered_data,
                data_name=f'No duplicates new{option}',
                values=1 if option=='' else None,
                borders=new_borders,
            )
            group[f'No duplicates new{option}'].attrs['description'] = (
                f"The new no duplicates data, i.e. the 0b00011000 filtered data{option}. Hence, "
                "the data represents all the data without any of the duplicates. Even the "
                "bifurcations are taken into account. No duplicates should exist in this "
                "filtering.\n"
                + (
                    f"The feet are saved with a value corresponding to 0b00100000."
                    if option != '' else ''
                )
            )

            # Add line of sight data
            filtered_data = ((data & 0b10000000) == 0b10000000).astype('uint8')
            if option != '': continue
            group = self.add_cube(
                group=group,
                data=filtered_data,
                data_name=f'SDO line of sight',
                borders=new_borders,
            )
            group[f'SDO line of sight'].attrs['description'] = (
                "The SDO line of sight data, i.e. the 0b01000000 filtered data. Hence, this data "
                "represents what is seen by SDO if represented in 3D inside the space of the "
                "rainbow cube data. The limits of the borders are defined in the .save IDL code "
                "named new_toto.pro created by Dr. Frederic Auchere."
            )
            filtered_data = ((data & 0b01000000) == 0b01000000).astype('uint8')
            group = self.add_cube(
                group=group,
                data=filtered_data,
                data_name=f'STEREO line of sight',
                borders=new_borders,
            )
            group[f'STEREO line of sight'].attrs['description'] = (
                "The STEREO line of sight data, i.e. the 0b01000000 filtered data. Hence, this "
                "data represents what is seen by STEREO if represented in 3D inside the space of "
                "the rainbow cube data. The limits of the borders are defined in the .save IDL "
                "code named new_toto.pro created by Dr. Frederic Auchere."
            )
    
    @Decorators.running_time
    def integrated_group(self, H5PYFile: h5py.File, borders: dict[str, dict[str, str | float]]):
        """
        To integrate the data and save it inside a specific HDF5 group.

        Args:
            H5PYFile (h5py.File): the HDF5 file.
            borders (dict[str, dict[str, str | float]]): the border information.
        """

        # Setup group
        group = H5PYFile.create_group('Time integrated')
        group.attrs['description'] = (
            "This group has already time integrated data for some of the main data filtering.\n"
            "This was created for ease of use when further analysing the structures."
        )

        # Data options with or without feet
        data_options = [
            f'{data_type}{feet}'
            for data_type in ['All data', 'No duplicates new']
            for feet in self.feet_options
        ]

        for option in data_options:
            # Setup group
            inside_group = group.create_group(option)
            inside_group.attrs['description'] = (
                f"This group only contains {option.lower()} time integrated data.\n"
                f"To get border info, please refer to the Filtered/{option} data group."
                + (
                    "Furthermore, the feet are saved with values equal to 0b00100000."
                    if 'with feet' in option else ''
                )
            )

            # For each integration time
            for integration_time in self.integration_time:
                # Setup
                time_hours = round(integration_time / 3600, 1)
                group_name = f'Time integration of {time_hours} hours'

                # Get data
                data, new_borders = self.time_integration(
                    H5PYFile=H5PYFile,
                    datapath=f'Filtered/{option}',
                    time=integration_time,
                    borders=borders, 
                )

                # Setup group
                inside_group = self.add_cube(
                    group=inside_group,
                    data=data,
                    data_name=group_name,
                    borders=new_borders,
                )
                inside_group[group_name].attrs['description'] = (
                    f"This group contains the {option.lower()} data integrated on {time_hours} "
                    "hours intervals."
                )              

    @Decorators.running_time
    def time_integration(
            self,
            H5PYFile: h5py.File,
            datapath: str,
            time: int,
            borders: dict[str, dict[str, str | float]]
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

        # Get data
        data = self.get_COO(H5PYFile, datapath.removesuffix(' with feet'))
        #TODO: will need to make a shared memory object later

        # Setup multiprocessing
        dates_len = len(self.dates_seconds)
        nb_processes = min(self.processes, dates_len)
        manager = mp.Manager()
        input_queue = manager.Queue()
        output_queue = manager.Queue()
        for i, date in enumerate(self.dates_seconds):
            input_queue.put((i, data, date, self.dates_seconds, time))
        for _ in range(nb_processes): input_queue.put(None)
        # Run
        processes = [None] * nb_processes
        for i in range(nb_processes):
            p = mp.Process(target=self.time_integration_sub, args=(input_queue, output_queue))
            p.start()
            processes[i] = p
        for p in processes: p.join()
        # Results
        data = [None] * dates_len
        while not output_queue.empty():
            identification, result = output_queue.get()
            data[identification] = result
        data = sparse.stack(data, axis=0).astype('uint8')

        # If feet
        if 'with feet' in datapath:
            data, new_borders = self.with_feet(data, borders)
        else: 
            new_borders = borders.copy()
        return data, new_borders

    @staticmethod
    def time_integration_sub(input_queue: mp.queues.Queue, output_queue: mp.queues.Queue) -> None:
        """
        To multiprocess the time integration of the cubes. This does it for each given date.

        Args:
            input_queue (mp.queues.Queue): the input arguments in a mp.Manager.Queue().
            output_queue (mp.queues.Queue): the results in a np.Manager.Queue().
        """

        while True:
            arguments = input_queue.get()
            if arguments is None: return

            index, data, date, dates, integration_time = arguments
            date_min = date - integration_time / 2
            #TODO: need to change this so that I can do the same for multiple integration times
            date_max = date + integration_time / 2

            # Result
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

        # Nothing found
        if chunk == []:
            return sparse.COO(coords=[], data=[], shape=data.shape[1:]).astype('uint8')
        elif len(chunk) == 1:
            return chunk[0]
        else:
            chunk = sparse.stack(chunk, axis=0)
            return sparse.COO.any(chunk, axis=0)
        
    def get_COO(self, H5PYFile: h5py.File, group_path: str) -> sparse.COO:
        """
        To get the sparse.COO object from the corresponding coords and values.

        Args:
            H5PYFile (h5py.File): the file object.
            group_path (str): the path to the group where the data is stored.

        Returns:
            sparse.COO: the corresponding sparse data.
        """

        data_coords: np.ndarray = H5PYFile[group_path + '/coords'][...]
        data_data: np.ndarray = H5PYFile[group_path + '/values'][...]
        data_shape = np.max(data_coords, axis=1) + 1
        return sparse.COO(coords=data_coords, data=1, shape=data_shape)
    
    @Decorators.running_time
    def polynomial_group(self, H5PYFile: h5py.File):
        """
        To add the polynomial information to the file.

        Args:
            H5PYFile (h5py.File): the file object.
        """
        
        # Data options with or without feet
        data_options = [
            f'{data_type}{feet}'
            for data_type in ['All data', 'No duplicates new']
            for feet in self.feet_options
        ]

        # main_options = ['All data with feet', 'No duplicates new with feet']
        # #TODO: need to add the new duplicates init when I understand the error
        sub_options = [
            f'/Time integration of {round(time / 3600, 1)} hours'
            for time in self.integration_time
        ]
        
        # Time integration group
        main_path_2 = 'Time integrated/'
        for main_option in data_options:
            for sub_option in sub_options:
                group_path = main_path_2 + main_option + sub_option
                data = self.get_COO(H5PYFile, group_path).astype('uint16')
                self.add_polynomial(H5PYFile[group_path], data)

    @Decorators.running_time
    def raw_cubes(self) -> sparse.COO:
        """
        To get the initial raw cubes as a sparse.COO object.

        Returns:
            sparse.COO: the raw cubes.
        """

        # Setup multiprocessing
        manager = mp.Manager()
        queue = manager.Queue()
        filepaths_nb = len(self.filepaths)
        processes_nb = min(self.processes, filepaths_nb)
        indexes = MultiProcessing.pool_indexes(filepaths_nb, processes_nb)

        # Multiprocessing
        processes: list[mp.Process] = [None] * processes_nb
        for i, index in enumerate(indexes): 
            process = mp.Process(
                target=self.raw_cubes_sub,
                args=(queue, i, self.filepaths[index[0]:index[1] + 1], self.fake_hdf5),
            )
            process.start()
            processes[i] = process
        for p in processes: p.join()
        # Results
        rawCubes = [None] * processes_nb 
        while not queue.empty():
            identifier, result = queue.get()
            rawCubes[identifier] = result
        rawCubes: sparse.COO = sparse.concatenate(rawCubes, axis=0)

        self.time_indexes = list(set(rawCubes.coords[0, :])) 
        return rawCubes
    
    @staticmethod
    def raw_cubes_sub(
            queue: mp.queues.Queue,
            queue_index: int,
            filepaths: list[str],
            fake_hdf5: bool,
        ) -> None:
        """
        To import the cubes in sections as there is a lot of cubes.

        Args:
            queue (mp.queues.Queue): to store the results.
            queue_index (int): to keep the initial ordering
            filepaths (list[str]): the filepaths to open.
            fake_hdf5 (bool): if the data used is the fake hdf5 data.
        """

        cubes = [None] * len(filepaths)
        for i, filepath in enumerate(filepaths):
            cube = scipy.io.readsav(filepath).cube

            if not fake_hdf5:
                # Add line of sight data 
                cube1 = scipy.io.readsav(filepath).cube1.astype('uint8') * 0b01000000
                cube2 = scipy.io.readsav(filepath).cube2.astype('uint8') * 0b10000000

                cubes[i] = (cube + cube1 + cube2).astype('uint8')
            else:
                cubes[i] = cube.astype('uint8')
        cubes = np.stack(cubes, axis=0)
        cubes = np.transpose(cubes, (0, 3, 2, 1))
        cubes = DataSaver.sparse_data(cubes)
        queue.put((queue_index, cubes))

    @staticmethod
    def sparse_data(cubes: np.ndarray) -> sparse.COO:
        """
        Changes data to a sparse representation.

        Args:
            cubes (np.ndarray): the initial array.

        Returns:
            sparse.COO: the corresponding sparse COO array.
        """

        cubes = sparse.COO(cubes)
        # the .to_numpy() method wasn't used as the idx_type argument isn't working properly
        cubes.coords = cubes.coords.astype('uint16')  # to save RAM
        return cubes
    
    def add_cube(
            self,
            group: h5py.File | h5py.Group,
            data: sparse.COO,
            data_name: str,
            values: int | None = None,
            borders: dict[str, dict[str, str | float]] | None = None
        ) -> h5py.File | h5py.Group:
        """ # todo update docstring
        To add to an h5py.Group, the data and metadata of a cube index spare.COO object. This takes
        also into account the border information.

        Args:
            group (h5py.File | h5py.Group): the Group where to add the data information.
            data (sparse.COO): the data that needs to be included in the file.
            data_name (str): the group name to be used in the file.
            borders (dict[str, dict[str, str  |  float]]): the data border information.

        Returns:
            h5py.File | h5py.Group: the updated group.
        """
        
        raw = {
            'description': "Default",
            'coords': {
                'data': data.coords.astype('uint16'),
                'unit': 'none',
                'description': (
                    "The index coordinates of the initial voxels.\nThe shape is (4, N) where the "
                    "rows represent t, x, y, z where t the time index (i.e. which cube it is), "
                    "and N the total number of voxels.\n"
                ),
            },
            'values': {
                'data': data.data if values is None else values, 
                'unit': 'none',
                'description': "The values for each voxel.",
            },
        }
        # Add border info
        if borders is not None: raw |= borders
        
        self.add_group(group, raw, data_name)
        return group
    
    def add_skycoords(
            self,
            group: h5py.File | h5py.Group,
            data: sparse.COO, data_name: str,
            borders: dict[str, dict[str, str | float]]
        ) -> h5py.File | h5py.Group:
        """
        To add to an h5py.Group, the data and metadata of the Carrington Heliographic Coordinates
        for a corresponding cube index spare.COO object. 
        This takes also into account the border information.

        Args:
            group (h5py.File | h5py.Group): the Group where to add the data information.
            data (sparse.COO): the data that needs to be included in the file.
            data_name (str): the group name to be used in the file.
            borders (dict[str, dict[str, str  |  float]] | None, optional): the data border
                information.

        Returns:
            h5py.File | h5py.Group: the updated group.
        """
        
        # Setup skycoords ndarray
        skycoords = self.carrington_skyCoords(data, borders)
        data = [None] * len(skycoords)
        for i, skycoord in enumerate(skycoords):
            x = skycoord.cartesian.x.value
            y = skycoord.cartesian.y.value
            z = skycoord.cartesian.z.value
            cube = np.stack([x, y, z], axis=0)

            # (x, y, z) -> (t, x, y, z)
            time_row = np.full((1, cube.shape[1]), self.time_indexes[i])
            data[i] = np.vstack([time_row, cube])
        data = np.hstack(data).astype('float32')

        raw = {
            'description': "Default",
            'coords': {
                'data': data,
                'unit': 'km',
                'description': (
                    "The t, x, y, z coordinates of the cube voxels.\nThe shape is (4, N) where "
                    "the rows represent t, x, y, z where t the time index (i.e. which cube it "
                    "is), and N the total number of voxels. Furthermore, x, y, z, represent the "
                    "X, Y, Z axis Carrington Heliographic Coordinates.\n"
                ),
            },
        }
        # Add border info
        raw |= borders
        self.add_group(group, raw, data_name)
        return group
    
    def add_polynomial(self, group: h5py.Group, data: sparse.COO) -> None:
        """
        To add to an h5py.Group, the polynomial curve and parameters given the data to fit.

        Args:
            group (h5py.Group): the group in which an polynomial group needs to be added.
            data (sparse.COO): the data to interpolate.

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
        # Loop n-orders
        for n_order in self.polynomial_order:
            instance = Polynomial(order=n_order, **polynomial_kwargs)

            info = instance.get_information()
            self.add_group(group, info, f'{n_order}th order polynomial')
    
    def with_feet(
            self,
            data: sparse.COO,
            borders: dict[str, dict[str, Any]],
        ) -> tuple[sparse.COO, dict[str, dict[str, str | float]]]:
        """
        Adds feet to a given initial cube index data as a sparse.COO object.

        Args:
            data (sparse.COO): the data to add feet to.
            borders (dict[str, dict[str, Any]]): the initial data borders.

        Returns:
            tuple[sparse.COO, dict[str, dict[str, str | float]]]: the new_data with feet and the
                corresponding borders data and metadata.
        """

        # Getting the positions
        x = self.feet.cartesian.x.to(u.km).value
        y = self.feet.cartesian.y.to(u.km).value
        z = self.feet.cartesian.z.to(u.km).value
        positions = np.stack([x, y, z], axis=0)

        # Getting the new borders
        x_min, y_min, z_min = np.min(positions, axis=1) 
        x_min = x_min if x_min <= borders['xt_min']['data'] else borders['xt_min']['data']
        y_min = y_min if y_min <= borders['yt_min']['data'] else borders['yt_min']['data']
        z_min = z_min if z_min <= borders['zt_min']['data'] else borders['zt_min']['data']
        new_borders = self.create_borders((x_min, y_min, z_min))

        # Feet pos inside init data
        positions[0, :] -= borders['xt_min']['data']
        positions[1, :] -= borders['yt_min']['data']
        positions[2, :] -= borders['zt_min']['data']
        positions /= self.dx['data']
        positions = np.round(positions).astype('int32')

        # Setup cubes with feet
        init_coords = data.coords
        # (x, y, z) -> (t, x, y, z)
        feet = np.hstack([
            np.vstack((np.full((1, 2), time), positions))
            for time in self.time_indexes
        ])

        # Add feet 
        init_coords = np.hstack([init_coords, feet]).astype('int32')

        # Indexes to positive values
        x_min, y_min, z_min = np.min(positions, axis=1).astype(int)  
        if x_min < 0: init_coords[1, :] -= x_min
        if y_min < 0: init_coords[2, :] -= y_min  # ? not sure what I meant there
        if z_min < 0: init_coords[3, :] -= z_min

        # Changing to COO 
        shape = np.max(init_coords, axis=1) + 1
        feet_values = np.repeat(np.array([0b00100000], dtype='uint8'), len(self.time_indexes) * 2)
        values = np.concatenate([data.data, feet_values], axis=0)
        data = sparse.COO(coords=init_coords, data=values, shape=shape).astype('uint8')
        return data, new_borders

    def carrington_skyCoords(
            self,
            data: sparse.COO,
            borders: dict[str, dict[str, Any]],
        ) -> list[astropy.coordinates.SkyCoord]:
        """
        Converts sparse.COO cube index data to a list of corresponding astropy.coordinates.SkyCoord
        objects.

        Args:
            data (sparse.COO): the cube index data to be converted.
            borders (dict[str, dict[str, Any]]): the input data border information.

        Returns:
            list[coordinates.SkyCoord]: corresponding list of the coordinates for the cube index
                data. They are in Carrington Heliographic Coordinates. 
        """
        
        # Get coordinates
        coords = data.coords.astype('float64')

        # Heliocentric kilometre conversion
        coords[1, :] = coords[1, :] * self.dx['data'] + borders['xt_min']['data']
        coords[2, :] = coords[2, :] * self.dx['data'] + borders['yt_min']['data']
        coords[3, :] = coords[3, :] * self.dx['data'] + borders['zt_min']['data']

        # SharedMemory
        shm, coords = MultiProcessing.create_shared_memory(coords)
        
        # Multiprocessing
        # Constants
        time_nb = len(self.time_indexes)
        processes_nb = min(self.processes, time_nb)
        # Queues
        manager = mp.Manager()
        input_queue = manager.Queue()
        output_queue = manager.Queue()
        # Setup
        for i, time in enumerate(self.time_indexes): input_queue.put((i, time))
        for _ in range(processes_nb): input_queue.put(None)
        # Run
        processes: list[mp.Process] = [None] * processes_nb
        for i in range(processes_nb):
            process = mp.Process(
                target=self.skyCoords_slice,
                args=(coords, input_queue, output_queue),
            )
            process.start()
            processes[i] = process
        for p in processes: p.join()
        shm.unlink()
        # Results
        all_SkyCoords = [None] * time_nb
        while not output_queue.empty():
            identifier, result = output_queue.get()
            all_SkyCoords[identifier] = result
        return all_SkyCoords
    
    @staticmethod
    def skyCoords_slice(
            coords: dict[str, Any],
            input_queue: mp.queues.Queue,
            output_queue: mp.queues.Queue,
        ) -> None:
        """
        To create an astropy.coordinates.SkyCoord object for a singular cube (i.e. for a unique
        time index).

        Args:
            coords (dict[str, Any]): information to find the sparse.COO(data).coords
                multiprocessing.shared_memory.SharedMemory object.
            input_queue (mp.queues.Queue): multiprocessing.Manager.Queue object used for the
                function inputs.
            output_queue (mp.queues.Queue): multiprocessing.Manager.Queue object used to extract
                the function results.
        """
        
        shm, coords = MultiProcessing.open_shared_memory(coords)

        while True:
            # Setup input
            argument = input_queue.get()
            if argument is None: break
            index, time = argument

            # Data slicing
            slice_filter = coords[0, :] == time
            cube = coords[:, slice_filter]
            
            # Carrington Heliographic Coordinates
            skyCoord = astropy.coordinates.SkyCoord(
                cube[1, :], cube[2, :], cube[3, :], 
                unit=u.km,
                frame=sunpy.coordinates.frames.HeliographicCarrington,
                representation_type='cartesian'
            )
            
            # Saving result
            output_queue.put((index, skyCoord))
        shm.close()



if __name__=='__main__':

    kwargs = dict(
        polynomial_order=[4, 5],
        processes=4,
        feet_sigma=20,
        south_leg_sigma=20,
        leg_threshold=0.03,
        only_feet=False,  # todo this option still isn't setup properly
        full=False,
        fake_hdf5=True,
    )
    
    # Naming setup
    splitted = str(kwargs['feet_sigma']).split('.')
    feet_sig = '1' + 'e' + (str(len(splitted[1])) if len(splitted) > 1 else '')
    thresh = "_".join(string for string in str(kwargs['leg_threshold']).split('.'))
    # instance = DataSaver(
    #     f"sig{feet_sig}_leg{kwargs['south_leg_sigma']}_lim{thresh}_test.h5",
    #     **kwargs,
    # )
    instance = DataSaver(
        "new_fake.h5",
        **kwargs,
    )
    instance.create()
