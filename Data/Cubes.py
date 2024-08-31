"""
To create the HDF5 data files with the cubes data and metadata.
A lot of different data is saved in the file to make any further manipulation or visualisation more easy.
"""

# IMPORTS
import os
import re
import sys
import h5py
import scipy
import sunpy
import typing
import sparse
import datetime

import numpy as np
import multiprocessing as mp

# Submodules import
import sunpy.coordinates
import multiprocessing.queues  # necessary as queues doesn't seem to be in the __init__
from astropy import units as u
from astropy import coordinates

# Personal imports
sys.path.append('..')  #TODO: need to check if I need os.path.join() but I don't see why I should
from Common import MultiProcessing, Decorators

# TODO: I also need to save the cube number and dates in the cube as an attribute 

class DataSaver:
    """
    To create cubes with feet and without feet in an HDF5 format.
    """

    @Decorators.running_time
    def __init__(self,
                 filename: str,
                 processes: int, 
                 interpolation_points: float = 10**6, 
                 interpolation_order: int | list[int] = [3, 6],
                 feet_lonlat: tuple[tuple[int, int], ...] = ((-177, 15), (-163, -16)),
                 foot_weight: float = 0.075,
                 exits_ok: bool = False
                 ) -> None:

        # Initial attributes
        self.filename = filename
        self.processes = processes
        self.interpolation_points = interpolation_points
        self.interpolation_order = interpolation_order if isinstance(interpolation_order, list) else [interpolation_order]
        self.foot_weight = foot_weight
        self.exists = exits_ok

        # Constants
        self.solar_r = 6.96e5  # in km

        # Attributes setup
        self.feet = self.setup_feet(feet_lonlat)
        self.dx: dict[str, any]  # information and value of the spatial resolution
        self.time_indexes: list[int]  # the time indexes with values inside the cubes

        # Created attributes
        self.setup_attributes()

        # Run
        self.create()

    def setup_path(self) -> dict[str, str]:
        """
        Gives the directory paths as a dictionary.

        Returns:
            dict[str, str]: the directory paths.
        """

        # Setup
        main = '/home/avoyeux/old_project/avoyeux'
        if not os.path.exists(main): main = '/home/avoyeux/Documents/avoyeux'
        if not os.path.exists(main): raise ValueError(f"The main path {main} not found.")
        python_codes = os.path.join(main, 'python_codes')

        # Format paths
        paths = {
            'main': main,
            'cubes': os.path.join(main, 'Cubes_karine'),
            'intensities': os.path.join(main, 'STEREO', 'int'),
            'save': os.path.join(python_codes, 'Data'),
        }
        return paths
    
    def setup_attributes(self):
        """
        Multiple instance attributes are defined here. Function is only here to not flood the __init__ method.
        """
        
        # Paths
        self.paths = self.setup_path()

        # Patterns
        cube_pattern = re.compile(r'cube(\d{3})\.save')
        date_pattern = re.compile(r'(?P<number>\d{4})_(?P<date>\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2})\.\d{3}\.png')

        # Get cubes filepaths
        self.filepaths = sorted([
            os.path.join(self.paths['cubes'], name)
            for name in os.listdir(self.paths['cubes'])
            if cube_pattern.match(name)
        ])

        # Get cube numbers
        self.cube_numbers = [
            int(cube_pattern.match(os.path.basename(filepath)).group(1))
            for filepath in self.filepaths
        ]

        # Get dates
        date_filenames = sorted(os.listdir(self.paths['intensities']))
        dates = [None] * len(self.filepaths)
        for i, number in enumerate(self.cube_numbers):
            for filename in date_filenames:
                filename_match = date_pattern.match(filename)
                if filename_match:
                    if int(filename_match.group('number')) == number:
                        dates[i] = filename_match.group('date')
                        break
        self.dates: list[str] = dates
    
    def setup_feet(self, lonlat: tuple[tuple[int, int], ...]) -> coordinates.SkyCoord:
        """
        Gives the 2 feet positions as an astropy.coordinates.SkyCoord object in Carrington Heliographic Coordinates. 

        Args:
            lonlat (tuple[tuple[int, int], ...]): the longitude and latitude positions for the added feet (i.e. ((lon1, lat1), (lon2, lat2))).

        Returns:
            coordinates.SkyCoord: the SkyCoord for the feet.
        """

        # Setup feet ndarray
        feet_pos = np.empty((3, 2), dtype='float64')
        feet_pos[0, :] = np.array([lonlat[0][0], lonlat[1][0]])
        feet_pos[1, :] = np.array([lonlat[0][1], lonlat[1][1]])
        feet_pos[2, :] = self.solar_r

        # Creating the feet
        feet = coordinates.SkyCoord(feet_pos[0, :] * u.deg, feet_pos[1, :] * u.deg, feet_pos[2, :] * u.km,
                                    frame=sunpy.coordinates.frames.HeliographicCarrington)
        cartesian_feet = feet.represent_as(coordinates.CartesianRepresentation)
        return coordinates.SkyCoord(cartesian_feet, frame=feet.frame, representation_type='cartesian')
    
    def get_cube_dates_info(self) -> dict[str, dict[str, str | np.ndarray]]:
        """
        Gives the cube numbers and dates information. 

        Returns:
            dict[str, dict[str, str | np.ndarray]]: the data and metadata for the cube numbers and dates.
        """

        # Add metadata
        cube_numbers_info = {
            'data': np.array(self.cube_numbers).astype('uint16'),
            'unit': 'none',
            'description': (
                "The numbers for the data cubes that are used in this file. While this value doesn't really make sense for a random user, this information was "
                "initially used to get a part of the metadata (like the acquisition time) for the cubes as it relates to the STEREO [...]nm acquisitions. All "
                "being said, the corresponding dates are also saved in this HDF5 file."
            ),
        }
        cube_dates_info = {
            'data': np.array(self.dates).astype('S19'),
            'unit': 'none',
            'description': (
                "The cube dates that are used in this file."
            ),
        }
        
        # Reformat for ease of use
        information = {
            'Numbers': cube_numbers_info,
            'Dates': cube_dates_info,
        }
        return information
    
    def create(self) -> None:
        """
        Main function that encapsulates the file creation and closing with a with statement.
        """

        with h5py.File(os.path.join(self.paths['save'],self.filename), 'a+' if self.exists else 'w') as H5PYFile:

            # Get borders
            cube = scipy.io.readsav(self.filepaths[0])
            values = (cube.dx, cube.xt_min, cube.yt_min, cube.zt_min)
            self.dx, init_borders = self.create_borders(values)

            # Main metadata
            H5PYFile = self.foundation(H5PYFile)

            # Raw data and metadata
            H5PYFile, feet_borders = self.init_group(H5PYFile, init_borders)

            # Filtered data and metadata
            H5PYFile = self.filtered_group(H5PYFile, feet_borders)

            # Interpolation data and metadata
            H5PYFile = self.interpolation_group(H5PYFile)

    def create_borders(self, values: tuple[float, ...]) -> tuple[dict[str, any], dict[str, dict[str, str | float]]]:
        """
        Gives the border information for the data.

        Args:
            values (tuple[float, ...]): the dx, xmin, ymin, zmin value in km.

        Returns:
            tuple[dict[str, any], dict[str, dict[str, any]]]: the data and metadata for the data borders.
        """

        # Data and metadata
        dx = {
            'data': values[0],
            'unit': 'km',
            'description': "The voxel resolution.",
        }
        info = {
            'xmin': {
                'data': values[1],
                'unit': 'km',
                'description': ("The minimum X-axis Carrington Heliographic Coordinates value for each data cube.\n"
                                "The X-axis in Carrington Heliographic Coordinates points towards the First Point of Aries."),
            }, 
            'ymin': {
                'data': values[2],
                'unit': 'km',
                'description': ("The minimum Y-axis Carrington Heliographic Coordinates value for each data cube.\n"
                                "The Y-axis in Carrington Heliographic Coordinates points towards the ecliptic's eastern horizon."),
            },
            'zmin': {
                'data': values[3],
                'unit': 'km',
                'description': ("The minimum Z-axis Carrington Heliographic Coordinates value for each data cube.\n"
                                "The Z-axis in Carrington Heliographic Coordinates points towards Sun's north pole."),
            },
        }
        return dx, info
    
    def foundation(self, H5PYFile: h5py.File) -> h5py.File:
        """
        For the main file metadata before getting to the HDF5 datasets and groups.

        Args:
            H5PYFile (h5py.File): the file.

        Returns:
            h5py.File: the updated file.
        """

        description = (
            "Contains the data cubes for the Solar Rainbow event gotten from the intersection of masks gotten from SDO and STEREO images.The SDO masks "
            "were created from an automatic code created by Dr. Elie Soubrie, while the STEREO masks where manually created by Dr. Karine Bocchialini "
            "by visual interpretation of the [...] nm monochromatic STEREO [...] images.\n"
            "New values for the feet where added to help for a curve fitting of the filament. These were added by looking at the STEREO [...] nm images "
            "as the ends of the filament are more visible. Hence, the feet are not actually visible in the initial masks.\n"
            "The data:\n" 
            "- the voxel values were coded in bits so that they can be easily separated in different categories. These are"
        )
        # TODO: need to finish the explanation here and also explain that the data is saved as sparse arrays.

        info = {
            'author': 'Voyeux Alfred',
            'creationDate': datetime.datetime.now().isoformat(),
            'filename': self.filename,
            'description': description,
        }

        # Get more metadata
        meta_info = self.get_cube_dates_info()

        # Update file
        H5PYFile = self.add_dataset(H5PYFile, info)
        H5PYFile = self.add_dataset(H5PYFile, self.dx, 'dx')
        for key in meta_info.keys(): H5PYFile = self.add_dataset(H5PYFile, meta_info[key], key)
        return H5PYFile
    
    def add_dataset(self, group: h5py.File | h5py.Group, info: dict[str, any], name: str = '') -> h5py.File | h5py.Group:
        """
        Adds a DataSet to a HDF5 group like object.

        Args:
            group (h5py.File | h5py.Group): the HDF5 group like object.
            info (dict[str, any]): the information to add in the DataSet.
            name (str, optional): the name of the DataSet to add. Defaults to '' (then the attributes are directly added to the group).

        Returns:
            h5py.File | h5py.Group: the input group like object with the added DataSet.
        """
        
        # Find data key
        for key, item in info.items(): 
            # print(f'for key {key} type is {type(item)}')
            # print(f'here is the  key {key} with type {type(item)}')
            if not isinstance(item, str): break
        # print(key)

        dataset = group.require_dataset(name, shape=info[key].shape, dtype=info[key].dtype, data=info[key]) if name != '' else group
        for attrs_key, item in info.items():
            if key==attrs_key: continue
            dataset.attrs[attrs_key] = item 
        return group
    
    def add_group_old(self, group: h5py.File | h5py.Group, info: dict[str, any], name: str) -> h5py.File | h5py.Group:
        """
        Adds a group with the corresponding DataSets to a HDF5 group like object.

        Args:
            group (h5py.File | h5py.Group): the HDF5 group like object.
            info (dict[str, str  |  dict[str  |  any]]): the information  and data to add in the group.
            name (str): the name of the group.

        Returns:
            h5py.File | h5py.Group: the input group like object with the added group.
        """

        new_group = group.require_group(name)
        for key, item in info.items():
            if isinstance(item, str): 
                new_group.attrs[key] = item
            elif 'data' in item.keys():
                new_group = self.add_dataset(new_group, item, key)
            else:
                new_group = self.add_group(new_group, item, key)
        return group
    
    def add_group(self, group: h5py.File | h5py.Group, info: dict[str, any], name: str) -> h5py.File | h5py.Group:
        """
        Adds a group with the corresponding DataSets to a HDF5 group like object.

        Args:
            group (h5py.File | h5py.Group): the HDF5 group like object.
            info (dict[str, str  |  dict[str  |  any]]): the information  and data to add in the group.
            name (str): the name of the group.

        Returns:
            h5py.File | h5py.Group: the input group like object with the added group.
        """

        # print(f'the type of the input is {type(info)}')
        if not all(isinstance(value, (str, dict)) for value in info.values()):  #TODO: will need to change this later if I get datasets without attributes
            group = self.add_dataset(group, info, name)
        else:
            new_group = group.require_group(name)
            for key, item in info.items():
                if isinstance(item, str):
                    new_group.attrs[key] = item
                else:
                    new_group = self.add_group(new_group, item, key)
        return group
    
    @Decorators.running_time
    def init_group(self, H5PYFile: h5py.File, borders: dict[str, dict[str, str | float]]) -> tuple[h5py.File, dict[str, dict[str, str | float]]]:
        """
        To create the initial h5py.Group object; where the raw data (with/without feet and/or in Carrington Heliographic Coordinates).

        Args:
            H5PYFile (h5py.File): the opened file pointer.
            borders (dict[str, dict[str, any]]): the border info (i.e. x_min, etc.) for the given data.

        Returns:
            h5py.File: the opened file pointer and the new data border information.
        """

        # Get data
        data = self.raw_cubes()

        # Setup group
        group = H5PYFile.create_group('Raw')
        group.attrs['description'] = (
            "The filament voxels in sparse COO format (i.e. with a coords and values arrays) of the initial cubes gotten from Dr. Karine [...]'s work.\n"
            "Furthermore, the necessary information to be able to position the filament relative to the Sun are also available. "
            "Both cubes, with or without feet, are inside this group."
        )

        # Add raw cubes group
        group = self.add_cube(group, data, 'Raw cubes', borders)
        group['Raw cubes'].attrs['description'] = "The initial voxel data in COO format without the feet for the interpolation."
        group['Raw cubes/values'].attrs['description'] = (
            "The values for each voxel as a 1D numpy array, in the same order than the coordinates given in the 'coords' group.\n"
            "Each specific bit set to 1 in these uint8 values represent a specific information on the voxel. Hence, you can easily filter these values given which "
            "bit is set to 1.\n"
            "The value information is as follows:\n"
            "-0b" + "0" * 7 + "1 represents the intersection between the STEREO and SDO point of views regardless if the voxel is a duplicate or not.\n"
            "-0b" + "0" * 6 + "10 represents the no duplicates regions from STEREO's point of view.\n"
            "-0b" + "0" * 5 + "100 represents the no duplicates regions from SDO's point of view.\n"
            "-0b" + "0" * 4 + "1000 represents the no duplicates data from STEREO's point of view. This takes into account if a region has a bifurcation.\n"
            "-0b0001" + "0" * 4 + " represents the no duplicates data from SDO's point of view. This takes into account if a region has a bifurcation.\n"
            "-0b001" + "0" * 5 + " gives out the feet positions.\n"
            "It is important to note the difference between the second/third bit being set to 1 or the fourth/fifth bit being set to 1. For the fourth and "
            "fifth bit, the definition for a duplicate is more strict. It also takes into account if, in the same region (i.e. block of voxels that are in contact), "
            "there is a bifurcation. If each branch of the bifurcation can be duplicates, then both branches are taken out (not true for the second/third bits as "
            "the duplicate treatment is done region by region.\n"
            "Furthermore, it is of course also possible to mix and match to get more varied datasets, e.g. -0b00011000 represents the no duplicates data."
        )

        # Add raw skycoords group
        group = self.add_skycoords(group, data, 'Raw coordinates', borders)
        group['Raw coordinates'].attrs['description'] = 'The initial data saved as Carrington Heliographic Coordinates in km.'

        # Add raw feet
        data, borders = self.with_feet(data, borders)
        group = self.add_cube(group, data, 'Raw cubes with feet', borders)
        group['Raw cubes with feet'].attrs['description'] = 'The initial raw data in COO format with the feet positions added.'
        group['Raw cubes with feet/values'].attrs['description'] = group['Raw cubes/values'].attrs['description']

        # Add raw skycoords with feet
        group = self.add_skycoords(group, data, 'Raw coordinates with feet', borders)
        group['Raw coordinates with feet'].attrs['description'] = 'The initial data with the feet positions added saved as Carrington Heliographic Coordinates in km.'
        return H5PYFile, borders
    
    @Decorators.running_time
    def filtered_group(self, H5PYFile: h5py.File, borders: dict[str, dict[str, str | float]]) -> h5py.File:
        """
        To filter the data and save it with weighted feet.

        Args:
            H5PYFile (h5py.File): the file object.
            borders (dict[str, dict[str, str  |  float]]): the border information.

        Returns:
            h5py.File: the updated file object.
        """

        # Setup group
        group = H5PYFile.create_group('Filtered')
        group.attrs['description'] = (
            "This group is based on the data from the 'Raw' HDF5 group. It is made up of already filtered data for easier use later but also to be able to add a "
            "weight to the feet. Hence, the interpolation data for each filtered data group is also available."
        )

        # Get data
        data = self.get_COO(H5PYFile, 'Raw/Raw cubes with feet').astype('uint8')

        # Add all data
        filtered_data = self.cubes_filtering(data, 0b00000001)
        group = self.add_cube(group, filtered_data, 'All data with feet', borders)
        group['All data with feet'].attrs['description'] = (
            "All data, i.e. the 0b00000001 filtered data with the feet. Hence, the data represents the intersection between the STEREO and SDO point of views.\n"
            f"The feet are weighted with, for each cube (i.e. for a given time), {self.foot_weight * 100}% of the initial cube pixels."
        )

        # Add no duplicates init
        filtered_data = self.cubes_filtering(data, 0b00000110)  #TODO: the shape of the resulting data is weird, no clue why.
        group = self.add_cube(group, data, 'No duplicates init with feet', borders)
        group['No duplicates init with feet'].attrs['description'] = (
            "The initial no duplicates data, i.e. the 0b00000110 filtered data with feet. Hence, the data represents all the data without the duplicates, without "
            "taking into account if there are bifurcations in some of the regions. Therefore, some duplicates might still exist using this filtering.\n"
            f"The feet are weighted with, for each cube (i.e. for a given time), {self.foot_weight * 100}% of the initial cube pixels."
        )

        # Add no duplicates new
        filtered_data = self.cubes_filtering(data, 0b00011000)
        group = self.add_cube(group, filtered_data, 'No duplicates new with feet', borders)
        group['No duplicates new with feet'].attrs['description'] = (
            "The new no duplicates data, i.e. the 0b00011000 filtered data with feet. Hence, the data represents all the data without any of the duplicates. "
            "Even the bifurcations are taken into account. No duplicates should exist in this filtering.\n"
            f"The feet are weighted with, for each cube (i.e. for a given time), {self.foot_weight * 100}% of the initial cube pixels."
        )
        return H5PYFile
    
    def get_COO(self, H5PYFile: h5py.File, group_path: str) -> sparse.COO:
        """
        To get the sparse.COO object from the corresponding coords and values.

        Args:
            H5PYFile (h5py.File): the file object.
            group_path (str): the path to the group where the data is stored.

        Returns:
            sparse.COO: the corresponding sparse data.
        """

        data_coords = H5PYFile[group_path + '/coords']
        data_data = H5PYFile[group_path + '/values']
        data_shape = np.max(data_coords, axis=1) + 1
        return sparse.COO(coords=data_coords, data=data_data, shape=data_shape)
    
    @Decorators.running_time
    def interpolation_group(self, H5PYFile: h5py.File) -> h5py.File:
        """
        To add the interpolation information to the file.

        Args:
            H5PYFile (h5py.File): the file object.

        Returns:
            h5py.File: the updated file object.
        """

        # All data
        group_path = 'Filtered/All data with feet'
        data = self.get_COO(H5PYFile, group_path).astype('uint16')
        self.add_interpolation(H5PYFile[group_path], data)

        # # No duplicates init  # TODO: need to add it later on when I understand why the data shape is like this (right now there are clearly too many voxels)
        # group_path = 'Filtered/No duplicates init with feet'
        # data = self.get_COO(H5PYFile, group_path).astype('uint16')
        # self.add_interpolation(H5PYFile[group_path], data)

        # No duplicates new
        group_path = 'Filtered/No duplicates new with feet'
        data = self.get_COO(H5PYFile, group_path).astype('uint16')
        self.add_interpolation(H5PYFile[group_path], data)
        return H5PYFile

    def cubes_filtering(self, data: sparse.COO, bit_filter: int) -> sparse.COO:
        """
        To filter the data and add the feet with the appropriate weight.

        Args:
            data (sparse.COO): data to be filtered.
            bit_filter (int): the filter in 8bits format.

        Returns:
            sparse.COO: the filtered data with added weighted feet.
        """
        
        # Filtering data
        filtered_data = ((data & bit_filter) == bit_filter).astype('uint16')

        # Setup feet weight
        voxels_per_cube = sparse.COO.sum(data, axis=(1, 2, 3)).todense()
        voxels_per_foot = np.round(voxels_per_cube * self.foot_weight)[:, None, None, None]
        feet = ((data & 0b00100000) > 0)

        # print(f'the shape of data is {data.shape}')
        # print(f'the shape of filtered data is {filtered_data.shape}')
        # print(f'the shape of feet is {feet.shape}')
        # print(f'the shape of voxel per foot is {voxels_per_foot.shape}')

        # Add feet
        feet = feet * voxels_per_foot  # TODO: this should work but keeping the warning here if I find problems in the visualisation
        return (filtered_data + feet).astype('uint16')

    @Decorators.running_time
    def raw_cubes(self) -> sparse.COO:
        """
        To get the initial raw cubes as a sparse.COO object.

        Returns:
            tuple[sparse.COO, list[int]]: the raw data.
        """

        # print(f'filepaths are {self.filepaths}')
        # Setup multiprocessing
        manager = mp.Manager()
        queue = manager.Queue()
        filepaths_nb = len(self.filepaths)
        processes_nb = min(self.processes, filepaths_nb)
        indexes = MultiProcessing.pool_indexes(filepaths_nb, processes_nb)

        # Multiprocessing
        processes = [None] * processes_nb
        for i, index in enumerate(indexes): 
            process = mp.Process(target=self.raw_cubes_sub, args=(queue, i, self.filepaths[index[0]:index[1] + 1]))
            process.start()
            processes[i] = process
        for p in processes: p.join()
        # Results
        rawCubes = [None] * processes_nb 
        while not queue.empty():
            identifier, result = queue.get()
            rawCubes[identifier] = result
        rawCubes = sparse.concatenate(rawCubes, axis=0)

        self.time_indexes = list(set(rawCubes.coords[0, :])) 
        return rawCubes
    
    @staticmethod
    def raw_cubes_sub(queue: mp.queues.Queue, queue_index: int, filepaths: list[str]) -> None:
        """
        To import the cubes in sections as there is a lot of cubes.

        Args:
            queue (mp.queues.Queue): to store the results.
            queue_index (int): to keep the initial ordering
            filepaths (list[str]): the filepaths to open.
        """

        cubes = [None] * len(filepaths)
        for i, filepath in enumerate(filepaths):
            cube = scipy.io.readsav(filepath).cube

            # Add line of sight data 
            cube1 = scipy.io.readsav(filepath).cube1.astype('uint8') * 0b01000000
            cube2 = scipy.io.readsav(filepath).cube2.astype('uint8') * 0b10000000

            cubes[i] = (cube + cube1 + cube2).astype('uint8')
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

        cubes = sparse.COO(cubes)  # the .to_numpy() method wasn't used as the idx_type argument isn't working properly
        cubes.coords = cubes.coords.astype('uint16')  # to save RAM
        return cubes
    
    def add_cube(self, group: h5py.File | h5py.Group, data: sparse.COO, data_name: str, borders: dict[str, dict[str, str | float]]) -> h5py.File | h5py.Group:
        """
        To add to an h5py.Group, the data and metadata of a cube index spare.COO object. This takes also into account the border information.

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
                'description': ("The index coordinates of the initial voxels.\n"
                                "The shape is (4, N) where the rows represent t, x, y, z where t the time index (i.e. which cube it is), and N the total number "
                                "of voxels.\n"),
            },
            'values': {
                'data': data.data, 
                'unit': 'none',
                'description': "The values for each voxel.",
            },
        }
        # Add border info
        raw |= borders
        group = self.add_group(group, raw, data_name)
        return group
    
    def add_skycoords(self, group: h5py.File | h5py.Group, data: sparse.COO, data_name: str, borders: dict[str, dict[str, str | float]]) -> h5py.File | h5py.Group:
        """
        To add to an h5py.Group, the data and metadata of the Carrington Heliographic Coordinates for a corresponding cube index spare.COO object. 
        This takes also into account the border information.

        Args:
            group (h5py.File | h5py.Group): the Group where to add the data information.
            data (sparse.COO): the data that needs to be included in the file.
            data_name (str): the group name to be used in the file.
            borders (dict[str, dict[str, str  |  float]]): the data border information.

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
                'description': ("The t, x, y, z coordinates of the cube voxels.\n"
                                "The shape is (4, N) where the rows represent t, x, y, z where t the time index (i.e. which cube it is), and N the total number "
                                "of voxels. Furthermore, x, y, z, represent the X, Y, Z axis Carrington Heliographic Coordinates.\n"),
            },
        }
        # Add border info
        raw |= borders
        group = self.add_group(group, raw, data_name)
        return group
    
    def add_interpolation(self, group: h5py.Group, data: sparse.COO) -> None:
        """
        To add to an h5py.Group, the interpolation curve and parameters given the data to fit.

        Args:
            group (h5py.Group): the group in which an interpolation group needs to be added.
            data (sparse.COO): the data to interpolate.

        Returns:
            h5py.Group: the updated group.
        """

        interpolation_kwargs = {
            'data': data, 
            'processes': self.processes,
            'precision_nb': self.interpolation_points,
        }
        # Loop n-orders
        for n_order in self.interpolation_order:
            instance = Interpolation(order=n_order, **interpolation_kwargs)

            info = instance.get_information()
            group = self.add_group(group, info, f'{n_order}th order interpolation')
    
    def with_feet(self, data: sparse.COO, borders: dict[str, dict[str, any]]) -> tuple[sparse.COO, dict[str, dict[str, str | float]]]:
        """
        Adds feet to a given initial cube index data as a sparse.COO object.

        Args:
            data (sparse.COO): the data to add feet to.
            borders (dict[str, dict[str, any]]): the initial data borders.

        Returns:
            tuple[sparse.COO, dict[str, dict[str, str | float]]]: the new_data with feet and the corresponding borders data and metadata.
        """

        # Getting the positions
        x = self.feet.cartesian.x.to(u.km).value
        y = self.feet.cartesian.y.to(u.km).value
        z = self.feet.cartesian.z.to(u.km).value
        positions = np.stack([x, y, z], axis=0)
        # print(f'positions shape and dtype are {positions.shape}, {positions.dtype}')

        # Getting the new borders
        x_min, y_min, z_min = np.min(positions, axis=1) 
        x_min = x_min if x_min <= borders['xmin']['data'] else borders['xmin']['data']
        y_min = y_min if y_min <= borders['ymin']['data'] else borders['ymin']['data']
        z_min = z_min if z_min <= borders['zmin']['data'] else borders['zmin']['data']
        _, new_borders = self.create_borders((0, x_min, y_min, z_min))

        # Feet pos inside init data
        positions[0, :] = positions[0, :] - borders['xmin']['data']
        positions[1, :] = positions[1, :] - borders['ymin']['data']
        positions[2, :] = positions[2, :] - borders['zmin']['data']
        positions /= self.dx['data']
        positions = np.round(positions).astype('int32')

        # print(f'positions index max values are {np.max(positions, axis=1)}')
        # print(f'positions index min values are {np.min(positions, axis=1)}')

        # Setup cubes with feet
        init_coords = data.coords
        # (x, y, z) -> (t, x, y, z)
        feet = np.hstack([
            np.vstack((np.full((1, 2), time), positions))
            for time in self.time_indexes
        ])
        # print(f'the initial data shape is {data.shape}')  # good
        # print(f'feet shape {feet.shape}')  # missing feet for 3 cubes...
        # print(f'feet max values are {np.max(feet, axis=1)}')
        # print(f'len of time_indexes is {len(self.time_indexes)}')
        # Add feet 
        init_coords = np.hstack([init_coords, feet]).astype('int32')  # TODO: will change it to uint16 when I am sure that it is working as intended

        shape = np.max(init_coords, axis=1) + 1
        # print(f'the initial shape of init_coords is {shape}')

        # Indexes to positive values
        x_min, y_min, z_min = np.min(positions, axis=1).astype(int)  
        # print(f'the minimum indexes are {x_min}, {y_min}, {z_min}')
        if x_min < 0: init_coords[1, :] -= x_min
        if y_min < 0: init_coords[2, :] -= y_min
        if z_min < 0: init_coords[3, :] -= z_min

        # Changing to COO 
        shape = np.max(init_coords, axis=1) + 1
        print(f'the shape when saving is {shape}')
        feet_values = np.repeat(np.array([0b00100000], dtype='uint8'), len(self.time_indexes) * 2)
        values = np.concatenate([data.data.astype('uint8'), feet_values], axis=0)
        data = sparse.COO(coords=init_coords, data=values, shape=shape).astype('uint8')
        return data, new_borders
        
    def carrington_skyCoords(self, data: sparse.COO, borders: dict[str, dict[str, any]]) -> list[coordinates.SkyCoord]:
        """
        Converts sparse.COO cube index data to a list of corresponding astropy.coordinates.SkyCoord objects.

        Args:
            data (sparse.COO): the cube index data to be converted.
            borders (dict[str, dict[str, any]]): the input data border information.

        Returns:
            list[coordinates.SkyCoord]: corresponding list of the coordinates for the cube index data. They are in Carrington Heliographic Coordinates. 
        """
        
        # Get coordinates
        coords = data.coords.astype('float64')

        # Heliocentric kilometre conversion
        coords[1, :] = coords[1, :] * self.dx['data'] + borders['xmin']['data']
        coords[2, :] = coords[2, :] * self.dx['data'] + borders['ymin']['data']
        coords[3, :] = coords[3, :] * self.dx['data'] + borders['zmin']['data']

        # SharedMemory
        shm, coords = MultiProcessing.shared_memory(coords)
        
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
        processes = [None] * processes_nb
        for i in range(processes_nb):
            process = mp.Process(target=self.skyCoords_slice, args=(coords, input_queue, output_queue))
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
    def skyCoords_slice(coords: dict[str, any], input_queue: mp.queues.Queue, output_queue: mp.queues.Queue) -> None:
        """
        To create an astropy.coordinates.SkyCoord object for a singular cube (i.e. for a unique time index).

        Args:
            coords (dict[str, any]): information to find the sparse.COO(data).coords multiprocessing.shared_memory.SharedMemory object.
            input_queue (mp.queues.Queue): multiprocessing.Manager.Queue object used for the function inputs.
            output_queue (mp.queues.Queue): multiprocessing.Manager.Queue object used to extract the function results.
        """
        
        shm = mp.shared_memory.SharedMemory(name=coords['name'])
        coords = np.ndarray(coords['shape'], dtype=coords['dtype'], buffer=shm.buf)

        while True:
            # Setup input
            argument = input_queue.get()
            if argument is None: break
            index, time = argument

            # Data slicing
            slice_filter = coords[0, :] == time
            cube = coords[:, slice_filter]
            
            # Carrington Heliographic Coordinates
            skyCoord = coordinates.SkyCoord(
                cube[1, :], cube[2, :], cube[3, :], 
                unit=u.km,
                frame=sunpy.coordinates.frames.HeliographicCarrington,
                representation_type='cartesian'
            )
            
            # Saving result
            output_queue.put((index, skyCoord))
        shm.close()
    

class Interpolation:
    """
    To get the fit curve position voxels and the corresponding n-th order polynomial parameters.
    """

    def __init__(self, data: sparse.COO, order: int, processes: int, precision_nb: int | float = 10**6) -> None:
        """
        Initialisation of the Interpolation class. Using the get_information() instance method, you can get the curve position voxels and the 
        corresponding n-th order polynomial parameters with their explanations inside a dict[str, str | dict[str, str | np.ndarray]].

        Args:
            data (sparse.COO): the data for which a fit is needed.
            order (int): the polynomial order for the fit.
            processes (int): the number of processes for multiprocessing.
            precision_nb (int | float, optional): the number of points used in the fitting. Defaults to 10**6.
        """

        # Arguments 
        self.data = data.coords[[0, 3, 2, 1]]  # TODO: as weirdly in the initial setup it doesn't work
        self.shape = [data.shape[i] for i in [0, 3, 2, 1]]
        self.poly_order = order
        self.processes = processes
        self.precision_nb = precision_nb

        # New attributes
        self.params_init = np.random.rand(order + 1)  # the initial (random) polynomial coefficients
    
    def get_information(self) -> dict[str, str | dict[str, str | np.ndarray]]:
        """
        To get the information and data for the interpolation and corresponding parameters (i.e. polynomial coefficients) ndarray. The explanations for these two 
        arrays are given inside the dict in this method.

        Returns:
            dict[str, str | dict[str, str | np.ndarray]]: the data and metadata for the interpolation and corresponding polynomial coefficients.
        """

        # Get data
        interpolations, parameters = self.get_data()

        # No duplicates uint16
        treated_interpolations = self.no_duplicates_data(interpolations)
        
        # Save information
        information = {
            'description': f"The interpolation curve with the corresponding parameters of the {self.poly_order}th order polynomial for each cube.",
            'raw coords': {
                'data': interpolations.astype('float32'),
                'unit': 'none',
                'description': (
                    "The index positions of the fitting curve for the corresponding data. The shape of this data is (4, N) where the rows represent (t, x, y, z). "
                    "Furthermore, the index positions are saved as floats, i.e. if you need to visualise it as voxels, then an np.round() and np.unique() is needed."
                ),
            },
            'treated coords': {
                'data': treated_interpolations,
                'unit': 'none',
                'description': (
                    "The index positions of the fitting curve for the corresponding data. The shape of this data is (4, N) where the rows represent (t, x, y, z). "
                    "This data set is treated, i.e. the coords here can directly be used in a sparse.COO object as the indexes are uint type and the duplicates "
                    "are already taken out."
                ),
            },
            'parameters': {
                'data': parameters.astype('float32'),
                'unit': 'none',
                'description': (
                    "The constants of the polynomial for the feet. The shape is (4, total number of constants) where the 4 represents t, x, y, z. " 
                    "Moreover, the constants are in order a0, a1, ... where the polynomial is a0 + a1*x + a2*x**2 ..."
                ),
            },
        }
        return information
    
    @Decorators.running_time
    def no_duplicates_data(self, data: np.ndarray) -> np.ndarray:
        """
        To get no duplicates uint16 voxel positions from a float type data.

        Args:
            data (np.ndarray): the data to treat.

        Returns:
            np.ndarray: the corresponding treated data.
        """

        # Setup multiprocessing
        manager = mp.Manager()
        output_queue = manager.Queue()
        processes_nb = min(self.processes, self.time_len)
        indexes = MultiProcessing.pool_indexes(self.time_len, processes_nb)
        shm, data = MultiProcessing.shared_memory(data)
        # Run
        processes = [None] * processes_nb
        for i, index in enumerate(indexes):
            p = mp.Process(target=self.no_duplicates_data_sub, args=(data, output_queue, index, i))
            p.start()
            processes[i] = p
        for p in processes: p.join()
        shm.unlink()
        # Results
        interpolations = [None] * processes_nb
        while not output_queue.empty():
            identifier, result = output_queue.get()
            interpolations[identifier] = result
        interpolations = np.concatenate(interpolations, axis=1)
        return interpolations.astype('uint16')

    @staticmethod
    def no_duplicates_data_sub(data: dict[str, any], queue: mp.queues.Queue, index: tuple[int, int], position: int) -> None:
        """
        To multiprocess the no duplicates uint16 voxel positions treatment.

        Args:
            data (dict[str, any]): the information to get the data from a multiprocessing.shared_memory.SharedMemory() object.
            queue (mp.queues.Queue): the output queue.
            index (tuple[int, int]): the indexes to slice the data properly.
            position (int): the position of the process to concatenate the result in the right order.
        """
        
        # Open SharedMemory
        shm = mp.shared_memory.SharedMemory(name=data['name'])
        data = np.ndarray(data['shape'], data['dtype'], shm.buf)

        # Select data
        data_filters = (data[0, :] >= index[0]) & (data[0, :] <= index[1])
        data = np.copy(data[:, data_filters])
        shm.close()

        # No duplicates indexes
        data = np.rint(np.abs(data.T))
        data = np.unique(data, axis=0).T.astype('uint16')
        queue.put((position, data))
        
    @Decorators.running_time
    def get_data(self) -> tuple[np.ndarray, np.ndarray]:
        """
        To get the interpolation and corresponding polynomial coefficients. The interpolation in this case is the voxel positions of the curve fit as a sparse.COO
        coords array (i.e. shape (4, N) where N the number of non zeros).

        Returns:
            tuple[np.ndarray, np.ndarray]: the interpolation and parameters array, both with shape (4, N) (not the same value for N of course).
        """

        # Constants
        self.time_indexes = list(set(self.data[0, :]))  # TODO: might get this from outside the class so that it is not computed twice or more
        self.time_len = len(self.time_indexes)
        process_nb = min(self.processes, self.time_len)

        # Shared memory
        shm, data = MultiProcessing.shared_memory(self.data)

        # Multiprocessing
        manager = mp.Manager()
        input_queue = manager.Queue()
        output_queue = manager.Queue()
        # Setup input
        for i, time in enumerate(self.time_indexes): input_queue.put((i, time))
        for _ in range(process_nb): input_queue.put(None)
        # Run
        processes = [None] * process_nb
        kwargs = {
            'data': data,
            'input_queue': input_queue,
            'output_queue': output_queue,
        }
        kwargs_sub = {
            'params_init': self.params_init,
            'shape': self.shape,
            'nth_order_polynomial': self.generate_nth_order_polynomial(),
            'precision_nb': self.precision_nb,
        }
        for i in range(process_nb):
            p = mp.Process(target=self.get_data_sub, kwargs={'kwargs_sub': kwargs_sub, **kwargs})
            p.start()
            processes[i] = p
        for p in processes: p.join()
        shm.unlink()
        # Results
        parameters = [None] * self.time_len
        interpolations = [None] * self.time_len
        while not output_queue.empty():
            identifier, interp, params = output_queue.get()
            interpolations[identifier] = interp
            parameters[identifier] = params
        interpolations = np.concatenate(interpolations, axis=1)
        print(f'interpolation shape is {interpolations.shape}')
        parameters = np.concatenate(parameters, axis=1)
        print(f'parameters shape is {parameters.shape}')
        return interpolations, parameters

    @staticmethod
    def get_data_sub(data: dict[str, any], input_queue: mp.queues.Queue, output_queue: mp.queues.Queue, kwargs_sub: dict[str, any]) -> None:
        """
        Static method to multiprocess the curve fitting creation.

        Args:
            data (dict[str, any]): the data information to access the multiprocessing.shared_memory.SharedMemory() object.
            input_queue (mp.queues.Queue): the input_queue for each process.
            output_queue (mp.queues.Queue): the output_queue to save the results.
            kwargs_sub (dict[str, any]): the kwargs for the polynomial_fit function.
        """
        
        # Open SharedMemory
        shm = mp.shared_memory.SharedMemory(name=data['name'])
        data = np.ndarray(data['shape'], dtype=data['dtype'], buffer=shm.buf)

        while True:
            args = input_queue.get()
            if args is None: break

            index, time_index = args
            time_filter = data[0, :] == time_index
            section = data[1:, time_filter]

            # Check if enough points for interpolation
            nb_parameters = len(kwargs_sub['params_init'])
            if nb_parameters >= section.shape[1]:
                print(f'For cube index {index}, not enough points for interpolation (shape {section.shape})', flush=True)
                result = np.empty((4, 0))  # TODO: This is wrong, just want to check the output shapes if there are in COO format.
                params = np.empty((4, 0))
                output_queue.put((index, result, params))
            else:
                # Ax swap for easier manipulation
                section = np.stack(section, axis=1)

                # Get cumulative distance
                t = np.empty(section.shape[0])
                t[0] = 0
                for i in range(1, section.shape[0]): t[i] = t[i - 1] + np.linalg.norm(section[i] - section[i - 1])
                t /= t[-1]  # normalisation 

                # Get results
                kwargs = {
                    'section': section,
                    't': t,
                    'time_index': time_index,
                }
                result, params = Interpolation.polynomial_fit(**kwargs, **kwargs_sub)

                # Save results
                output_queue.put((index, result, params))
        shm.close()

    @staticmethod
    def polynomial_fit(section: np.ndarray, t: np.ndarray, time_index: int, params_init: np.ndarray, shape: tuple[int, ...], precision_nb: float,
                       nth_order_polynomial: typing.Callable[[np.ndarray, tuple[int | float, ...]], np.ndarray]
                       ) -> tuple[np.ndarray, np.ndarray]:
        """
        To get the polynomial fit of a data cube.

        Args:
            section (np.ndarray): the data cube to be fitted.
            t (np.ndarray): the polynomial variable. In our case the cumulative distance.
            time_index (int): the time index for the cube that is being fitted.
            params_init (np.ndarray): the initial (random) polynomial coefficients.
            shape (tuple[int, ...]): the shape of the inputted data cube. 
            precision_nb (float): the number of points used in the polynomial when saved.
            nth_order_polynomial (typing.Callable[[np.ndarray, tuple[int  |  float, ...]], np.ndarray]): the n-th order polynomial function.

        Returns:
            tuple[np.ndarray, np.ndarray]: the polynomial position voxels and the corresponding coefficients.
        """

        # Get params
        x, y, z = [section[:, i] for i in range(3)]
        params_x, _ = scipy.optimize.curve_fit(nth_order_polynomial, t, x, p0=params_init)
        params_y, _ = scipy.optimize.curve_fit(nth_order_polynomial, t, y, p0=params_init)
        params_z, _ = scipy.optimize.curve_fit(nth_order_polynomial, t, z, p0=params_init)
        params = np.vstack([params_x, params_y, params_z]).astype('float32')
        # params = np.stack([params_x, params_y, params_z], axis=0)  # shape (3, n_order + 1)

        # Get curve
        t_fine = np.linspace(-0.5, 1.5, precision_nb)
        x = nth_order_polynomial(t_fine, *params_x)
        y = nth_order_polynomial(t_fine, *params_y)
        z = nth_order_polynomial(t_fine, *params_z)
        data = np.vstack([x, y, z]).astype('float32')

        # Cut outside init data
        conditions_upper = (data[0, :] >= shape[1] - 1) | (data[1, :] >= shape[2] - 1) | (data[2, :] >= shape[3] - 1)
        # TODO: the top code line is wrong as I am taking away 1 pixel but for now it is just to make sure no problem arouses from floats. will need to see what to do later
        conditions_lower = np.any(data < 0, axis=0)  # as floats can be a little lower than 0
        conditions = conditions_upper | conditions_lower
        data = data[:, ~conditions]

        # No duplicates
        unique_data = np.unique(data, axis=1)

        # Recreate format
        time_row = np.full((1, unique_data.shape[1]), time_index)
        unique_data = np.vstack([time_row, unique_data]).astype('float32')
        time_row = np.full((1, params.shape[1]), time_index)
        params = np.vstack([time_row, params]).astype('float32')
        return unique_data[[0, 3, 2, 1]], params[[0, 3, 2, 1]]  # TODO: will need to change this if I cancel the ax swapping in cls.__init__
    
    def generate_nth_order_polynomial(self) -> typing.Callable[[np.ndarray, tuple[int | float, ...]], np.ndarray]:
        """
        To generate a polynomial function given a polynomial order.

        Returns:
            typing.Callable[[np.ndarray, tuple[int | float, ...]], np.ndarray]: the polynomial function.
        """
        
        def nth_order_polynomial(t: np.ndarray, *coeffs: int | float) -> np.ndarray:
            """
            Polynomial function given a 1D ndarray and the polynomial coefficients. The polynomial order is defined before hand.

            Args:
                t (np.ndarray): the 1D array for which you want the polynomial results.

            Returns:
                np.ndarray: the polynomial results.
            """

            # Initialisation
            result = 0

            # Calculating the polynomial
            for i in range(self.poly_order + 1): result += coeffs[i] * t**i
            return result
        return nth_order_polynomial



if __name__=='__main__':

    DataSaver('testing_new.h5', processes=50)    

