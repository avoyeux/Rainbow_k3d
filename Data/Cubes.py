"""
To create the HDF5 data files for the cubes.
"""

# IMPORTS
import os
import re
import h5py
import glob
import sunpy
import sparse
import astropy

import numpy as np
import multiprocessing as mp

from scipy.io import readsav
from datetime import datetime
from astropy import units as u
from astropy import coordinates

# Personal imports
from ..Common import Decorators, MultiProcessing


class SavingTest:
    """
    To create cubes with feet and without feet in an HDF5 format.
    """

    def __init__(self, filename: str, processes: int, feet_lonlat: tuple[tuple[int, int], ...] = ((-177, 15), (-163, -16))) -> None:

        # Initial attributes
        self.filename = filename
        self.processes = processes

        # Constants
        self.feet_lonlat = feet_lonlat
        self.solar_r = 6.96e5

        # Attributes setup
        self.dx: float
        self.x_min: float
        self.y_min: float
        self.z_min: float

        # Created attributes
        self.paths = self.setup_path()
        self.cubes = self.fetch_cubes()

        # Run
        self.create()

    def setup_path(self) -> dict[str, str]:
        """
        Gives the directory paths.

        Returns:
            dict[str, str]: the directory paths.
        """

        main = '/home/avoyeux/Documents/avoyeux/python_codes'
        paths = {
            'main': main,
            'cubes': os.path.join(main, '..', 'Cubes_karine'),
            'save': os.path.join(main, 'Data'),
        }
        return paths
    
    def create(self) -> None:
        """
        Creates the HDF5 file.
        """

        with h5py.File(os.path.join(self.paths['save'],self.filename), 'w') as H5PYFile:

            # Setup filepaths
            pattern = re.compile(r'cube(\d{3})\.save')
            self.filepaths = [
                os.path.join(self.paths['cubes'], name)
                for name in os.listdir(self.paths['cubes'])
                if pattern.match(name)
            ]

            # Main metadata
            cube = readsav(self.filepaths[0])
            values = (cube.dx, cube.xt_min, cube.yt_min, cube.zt_min)
            dx, init_borders = self.create_borders(values)
            H5PYFile = self.foundation(H5PYFile, dx)

            # Setup multiprocessing
            manager = mp.Manager()
            queue = manager.Queue()
            indexes = MultiProcessing.pool_indexes(len(self.filepaths), self.processes)
            # Multiprocessing
            processes = [None] * len(indexes)
            for i, index in enumerate(indexes): 
                process = mp.Process(target=self.rawCubes, args=(queue, i, index))
                process.start()
                processes[i] = process
            for p in process: p.join()
            # Results
            rawCubes = [None] * len(indexes)
            while not queue.empty():
                identifier, result = queue.get()
                rawCubes[identifier] = result
            rawCubes = sparse.concatenate(rawCubes, axis=0)

            # Raw metadata
            H5PYFile = self.init_group(H5PYFile, rawCubes, init_borders)

    def rawCubes(self, queue: mp.queues.Queue, queue_index: int, index: tuple[int, int]) -> None:
        """
        To import the cubes in sections as there is a lot of cubes.

        Args:
            queue (mp.queues.Queue): to store the results.
            queue_index (int): to keep the initial ordering
            index (tuple[int, int]): the unique index sections used by each process. 
        """

        cubes = [readsav(filepath).cube.astype('uint8') for filepath in self.filepaths[index[0]:index[1] + 1]]
        cubes = np.stack(cubes, axis=0)
        cubes = np.transpose(cubes, (0, 3, 2, 1))
        cubes = self.sparse_data(cubes)
        queue.put((queue_index, cubes))  #TODO: this might not work, need to check if it does.

    def sparse_data(self, cubes: np.ndarray) -> sparse.COO:
        """
        Changes data to a sparse representation.

        Args:
            cubes (np.ndarray): the initial array.

        Returns:
            sparse.COO: the corresponding sparse COO array.
        """

        cubes = sparse.COO(cubes)  # the .to_numpy() method wasn't used as the idx_type argument isn't working properly
        cubes.coords = cubes.coords.astype('uint16')  # to save memory
        return cubes
    
    def foundation(self, H5PYFile: h5py.File, dx: dict[str, any]) -> h5py.File:
        """
        For the main information before getting to the HDF5 datasets and groups.

        Returns:
            h5py.File: the HDF5 file.
        """

        description = ("Contains the data cubes for the Solar Rainbow event gotten from the intersection of masks gotten from SDO and STEREO images. The SDO masks were created from an automatic "
        "code created by Dr. Elie Soubrie, while the STEREO masks where manually created by Dr. Karine [...] by visual interpretation of the [...] nm monochromatic STEREO [...] images.\n"
        "New values for the feet where added to help for a curve fitting of the filament. These were added by looking at the STEREO [...] nm images as the ends of the filament are more visible. "
        "Hence, the feet are not actually visible in the initial masks.\n"
        "The data:\n" 
        "- the voxel values were coded in bits so that they can be easily separated in different categories. These are")
        # TODO: need to finish the explanation here and also explain that the data is saved as sparse arrays.

        info = {
            'author': 'Voyeux Alfred',
            'creationDate': datetime.now().isoformat(),
            'filename': self.filename,
            'description': description,
        }

        # Update file
        H5PYFile = self.add_dataset(H5PYFile, info)
        H5PYFile = self.add_dataset(H5PYFile, dx, 'dx')
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
        
        dataset = group.require_dataset(name, data=info['data']) if name != '' else group
        for key, item in info.items():
            if key=='data': continue
            dataset.attrs[key] = item 
        return group
    
    def add_group(self, group: h5py.File | h5py.Group, info: dict[str, any | dict[str, any]], name: str) -> h5py.File | h5py.Group:
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

    def create_borders(self, values: tuple[float, ...]) -> tuple[dict[str, any], dict[str, dict[str, any]]]:
        """
        Gives the border information for the data.

        Args:
            values (tuple[float, ...]): the dx, xmin, ymin, zmin value in km.

        Returns:
            tuple[dict[str, any], dict[str, dict[str, any]]]: the data and metadata for the data borders.
        """

        # TODO: thinking of adding the cube shape here but most likely an np.max() would do the trick as the borders should represent (x, y, z) -> (0, 0, 0)

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
    
    def add_feet(self, H5PYFile: h5py.File, data: sparse.COO, dx: dict[str, any], borders: dict[str, dict[str, any]]):
        
        # Carrington Heliographic
        skycoords = self.carrington_skyCoords(data, dx, borders)

        # Creating the feet
        feet = coordinates.SkyCoord(self.feet_pos[0, :] * u.deg,
                                    self.feet_pos[1, :] * u.deg,
                                    self.feet_pos[2, :] * u.km,
                                    frame=sunpy.coordinates.frames.HeliographicCarrington)
        cartesian_feet = feet.represent_as(coordinates.CartesianRepresentation)
        feet = coordinates.SkyCoord(cartesian_feet, frame=feet.frame, representation_type='cartesian')

        # Multiprocessing
        # Constants
        skycoords_nb = len(skycoords)
        processes_nb = min(self.processes, skycoords_nb)
        # Setup
        manager = mp.Manager()
        input_queue = manager.Queue()
        output_queue = manager.Queue()
        # For positioning
        for i, skycoord in enumerate(skycoords): input_queue.put((i, skycoord))
        for _ in range(processes_nb): input_queue.put(None)
        # Run
        processes = []
        for i in range(processes_nb):
            process = mp.Process(target=self.concatenate_skycoords, args=(input_queue, output_queue, feet))
            process.start()
            processes[i] = process
        for p in processes: p.join()
        # Results
        skycoords = [None] * skycoords_nb
        while not output_queue.empty():
            identifier, result = output_queue.get()
            skycoords[identifier] = result

    
    def with_feet(self, data: sparse.COO, dx: dict[str, any], borders: dict[str, dict[str, any]]) -> sparse.COO:

        # Setup feet ndarray
        feet_pos = np.empty((3, 2), dtype='float64')
        feet_pos[0, :] = np.array([self.feet_lonlat[0][0], self.feet_lonlat[1][0]])
        feet_pos[1, :] = np.array([self.feet_lonlat[0][1], self.feet_lonlat[1][1]])
        feet_pos[2, :] = self.solar_r

        # Creating the feet
        feet = coordinates.SkyCoord(feet_pos[0, :] * u.deg, feet_pos[1, :] * u.deg, feet_pos[2, :] * u.km,
                                    frame=sunpy.coordinates.frames.HeliographicCarrington)
        cartesian_feet = feet.represent_as(coordinates.CartesianRepresentation)
        feet = coordinates.SkyCoord(cartesian_feet, frame=feet.frame, representation_type='cartesian')

        # Getting the positions
        x = feet.cartesian.x.to(u.km).value
        y = feet.cartesian.y.to(u.km).value
        z = feet.cartesian.z.to(u.km).value
        positions = np.stack([x, y, z], axis=0)

        # Getting the new borders
        _, x_min, y_min, z_min = np.min(positions, axis=1) 
        x_min = x_min if x_min <= borders['xmin']['data'] else borders['xmin']['data']
        y_min = y_min if y_min <= borders['ymin']['data'] else borders['ymin']['data']
        z_min = z_min if z_min <= borders['zmin']['data'] else borders['zmin']['data']
        _, new_borders = self.create_borders((0, x_min, y_min, z_min))

        # Feet pos inside init data
        positions[0, :] = positions[0, :] - borders['xmin']['data']
        positions[1, :] = positions[1, :] - borders['ymin']['data']
        positions[2, :] = positions[2, :] - borders['zmin']['data']
        positions /= dx['data']
        positions = np.round(positions).astype('uint16') 

        # Setup cubes with feet
        init_coords = data.coords
        time_indexes = list(set(init_coords[0, :]))
        # (x, y, z) -> (t, x, y, z)
        index_row = np.repeat(time_indexes, 2)  # as there are 2 feet
        feet = np.vstack([index_row, feet])  # shape (4, len(time_indexes) * 2)

        # Add feet 
        init_coords = np.hstack([coords, feet]).astype('int32')  # TODO: will change it to uint16 when I am sure that it is working as intended

        # Indexes to positive values
        _, x_min, y_min, z_min = np.min(positions, axis=1).astype(int)  
        if x_min < 0: init_coords[1, :] -= x_min
        if y_min < 0: init_coords[2, :] -= y_min
        if z_min < 0: init_coords[3, :] -= z_min

        # Changing to COO 
        shape = np.max(init_coords, axis=1) + 1
        feet_values = np.repeat(np.array([0b00100000], dtype='uint8'), len(time_indexes))  #TODO: need to add the values information in the HDF5 file info
        values = np.concatenate([data.data.astype('uint8'), feet_values], axis=0)
        data = sparse.COO(coords=init_coords, data=values, shape=shape)
        return data  

    def concatenate_skycoords(self, input_queue: mp.queues.Queue, output_queue: mp.queues.Queue, feet: coordinates.SkyCoord):

        #TODO: this function might become useless
        
        while True:
            args = input_queue.get()
            if args is None: return
            identifier, skycoord = args

            concatenation = coordinates.concatenate(skycoord, feet)
            output_queue.put((identifier, concatenation))


        

    def carrington_skyCoords(self, data: sparse.COO, dx: dict[str, any], borders: dict[str, dict[str, any]]) -> list[coordinates.SkyCoord]:
        
        # Get coordinates
        coords = data.coords.astype('float64')

        # Heliocentric kilometre conversion
        coords[1, :] = coords[1, :] * dx['data'] + borders['xmin']['data']
        coords[2, :] = coords[2, :] * dx['data'] + borders['ymin']['data']
        coords[3, :] = coords[3, :] * dx['data'] + borders['zmin']['data']
        time_indexes = list(set(coords[0, :]))

        # SharedMemory
        shm, coords = MultiProcessing.shared_memory(coords)
        
        # Multiprocessing
        # Constants
        time_nb = len(time_indexes)
        processes_nb = min(self.processes, time_nb)
        # Queues
        manager = mp.Manager()
        input_queue = manager.Queue()
        output_queue = manager.Queue()
        # Setup
        for i, time in enumerate(time_indexes): input_queue.put((i, time))
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
    
    def skyCoords_slice(self, coords: dict[str, any], input_queue: mp.queues.Queue, output_queue: mp.queues.Queue) -> None:
        
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
            skyCoord = coordinates.SkyCoord(cube[1, :], cube[2, :], cube[3, :], 
                                                    unit=u.km,
                                                    frame=sunpy.coordinates.frames.HeliographicCarrington,
                                                    representation_type='cartesian')
            
            # Saving result
            output_queue.put((index, skyCoord))
        shm.close()

    def add_Cubes(self, group: h5py.File | h5py.Group, data: sparse.COO, data_name: str, borders: dict[str, dict[str, any]]) -> h5py.File | h5py.Group:
        
        raw = {
            'description': "Default",
            'coords': {
                'data': data.coords,
                'unit': 'none',
                'description': ("The index coordinates of the initial voxels.\n"
                                "The shape is (4, N) where the rows represent t, x, y, z where t the time index (i.e. which cube it is), and N the total number"
                                "of voxels.\n"),
            },
            'value': {
                'data': data.data, 
                'unit': 'none',
                'description': "The values for each voxel.",
            },
        }
        # Add border info
        raw |= borders
        group = self.add_group(group, raw, data_name)
        return group

    def add_skycoords(self, group: h5py.File | h5py.Group, skycoords: list[coordinates.SkyCoord], data_name: str) -> h5py.File | h5py.Group:

        # TODO: stopped as I need to change the code completly to keep the voxel values with the coords
        pass



    def init_group(self, H5PYFile: h5py.File, data: sparse.COO, dx: dict[str, any], borders: dict[str, dict[str, any]]) -> h5py.File:

        # Group setup
        group = H5PYFile.create_group('L0 data')
        group.attrs['description'] = ("The filament voxels in sparse COO format (i.e. with a coords and values arrays) of the initial cubes gotten from Dr. Karine [...]'s work."
        "Furthermore, the necessary information to be able to position the filament relative to the Sun are also available.\n"
        "Both cubes, with or without feet, are inside this group.")

        # Add raw cubes group
        group = self.add_Cubes(group, data, 'Raw data', borders)
        group['Raw data'].attrs['description'] = "The initial voxel data in COO format without the feet for the interpolation."



        # TODO: to make sure that the extremes for the data cubes with feet are set up properly, might as well change everything to skycoords from the get go
        skycoords = self.carrington_skyCoords(data, dx, borders)



        





        # Add feet to raw cubes
        data = self.with_feet(data, dx, borders) # TODO: this is wrong
        ###################    # TODO: I need to make sure that the value for the data points is saved even when using SKYCOORDS positions.
        ### RAWSKYCOORD ###
        ###################

        rawSkyCoord = {
            'description': ("The voxel positions in Carrington Heliographic Coordinates without the feet for the interpolation."),
            'coords': {
                'data': rawCubes.coords,
                'unit': 'km',
                'description': ("The index coordinates of the initial voxels.\n"
                                "The shape is (4, N) where the rows represent t, x, y, z where t the time index (i.e. which cube it is), and N the total number"
                                "of voxels.\n"
                                "To get the positions with respect to the Sun, then dx, x_min, y_min, z_min need to be taken into account."),
            },
            'value': {
                'data': rawCubes.data, 
                'unit': 'none',
                'description': "The values for each voxel.",
            },
        }
        # Add border info
        raw |= borders

        ###################
        ####### FEET ######
        ###################

        with_feet = {
            'description': "The raw initial data with the added feet for the interpolation.",
            'coords': {
                'data': None, # TODO: need to decide how to add the feet data
                'unit': 'none',
                'description': ("The index coordinates of the initial voxels.\n"
                                "To get the positions with respect to the Sun, then dx, x_min, y_min, z_min need to be taken into account."),
            },
            'value': {
                'data': None, #TODO: need to create that data
                'unit': 'none',
                'description': 'The values for each pixel',
            },
        }


        # Adding the info together
        info = {
            'description': description,
            'raw': raw,
            'withFeet': None, 
        }

        H5PYFile = self.add_group(H5PYFile, info, 'InitCubes')
        return H5PYFile

    


























class Cubes:
    """
    To create cubes with feet and without feet in an HDF5 format.
    """

    def __init__(self, filename: str) -> None:

        # Initial attributes
        self.filename = filename

        # Constants
        self.solar_r = 6.96e5

        # Attributes setup
        self.dx: float
        self.x_min: float
        self.y_min: float
        self.z_min: float

        # Created attributes
        self.paths = self.setup_path()
        self.cubes = self.fetch_cubes()

        # Run
        self.create()

    def setup_path(self) -> dict[str, str]:

        main = '/home/avoyeux/Documents/avoyeux/python_codes'
        paths = {
            'main': main,
            'cubes': os.path.join(main, '..', 'Cubes_karine'),
            'save': os.path.join(main, 'Data'),
        }
        return paths
    
    def create(self) -> None:
        """
        Creates the HDF5 file.
        """

        with h5py.File(os.path.join(self.paths['save'],self.filename), 'w') as H5PYFile:

            # Main metadata
            file_info = self.foundation()
            H5PYFile = self.add_dataset(H5PYFile, file_info)

            # Raw metadata
            raw_info = self.init_group()
            H5PYFile = self.add_group(H5PYFile, raw_info, 'InitCubes')

    def foundation(self) -> dict[str, str]:
        """
        For the main information before getting to the HDF5 datasets and groups.

        Returns:
            dict[str, str]: the main information.
        """

        description = ("Contains the data cubes for the Solar Rainbow event gotten from the intersection of masks gotten from SDO and STEREO images. The SDO masks were created from an automatic "
        "code created by Dr. Elie Soubrie, while the STEREO masks where manually created by Dr. Karine [...] by visual interpretation of the [...] nm monochromatic STEREO [...] images.\n"
        "New values for the feet where added to help for a curve fitting of the filament. These were added by looking at the STEREO [...] nm images as the ends of the filament are more visible. "
        "Hence, the feet are not actually visible in the initial masks.\n"
        "The data:\n" 
        "- the voxel values were coded in bits so that they can be easily separated in different categories. These are")
        # TODO: need to finish the explanation here and also explain that the data is saved as sparse arrays.

        info = {
            'author': 'Voyeux Alfred',
            'creationDate': datetime.now().isoformat(),
            'filename': self.filename,
            'description': description,
        }
        return info
    
    def add_group(self, group: h5py.File | h5py.Group, info: dict[str, str | dict[str, any]], name: str) -> h5py.File | h5py.Group:
        """
        Adds a group with the corresponding DataSets to a HDF5 group like object.

        Args:
            group (h5py.File | h5py.Group): the HDF5 group like object.
            info (dict[str, str  |  dict[str  |  any]]): the information  and data to add in the group.
            name (str): the name of the group.

        Returns:
            h5py.File | h5py.Group: the input group like object with the added group.
        """

        new_group = group.create_group(name)
        for key, item in info.items():
            if isinstance(item, str): 
                new_group.attrs[key] = item
            elif 'data' in item.keys():
                new_group = self.add_dataset(new_group, item, key)
            else:
                new_group = self.add_group(new_group, item, key)
        return group
    
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
        
        dataset = group.create_dataset(name, data=info['data']) if name != '' else group
        for key, item in info.items():
            if key=='data': continue
            dataset.attrs[key] = item 
        return group
    
    def fetch_cubes(self) -> sparse.COO:
        # Find files
        pattern = re.compile(r'cube(\d{3})\.save')
        names = [name for name in os.listdir(self.paths['cubes']) if pattern.match(name)]

        # Opening the first cube
        cube = readsav(os.path.join(self.paths['cubes'], names[0]))
        self.dx = cube.dx
        self.x_min = cube.xt_min
        self.y_min = cube.yt_min
        self.z_min = cube.zt_min

        # Opening the cubes
        cubes = np.stack([readsav(os.path.join(self.paths['cubes'], name)).cube.astype('uint8') for name in names], axis=0)
        cubes = np.transpose(cubes, (0, 3, 2, 1)) # as readsav reads the axes in a different order.
        cubes = self.sparse_data(cubes)
        return cubes

    def init_group(self) -> dict[str, str | dict[str, any]]:
        
        description = ("The filament voxels in sparse COO format (i.e. with a coords and values arrays) of the initial cubes gotten from Dr. Karine [...]'s work."
        "Furthermore, the necessary information to be able to position the filament relative to the Sun are also available.\n"
        "Both cubes, with or without feet, are inside this group.")

        raw = {
            'description': "The raw initial data without the feet for the interpolation.",
            'coords': {
                'data': self.cubes.coords,
                'unit': 'none',
                'description': ("The index coordinates of the initial voxels.\n"
                                "To get the positions with respect to the Sun, then dx, x_min, y_min, z_min need to be taken into account."),
            },
            'value': {
                'data': self.cubes.data, 
                'unit': 'none',
                'description': "The values for each voxel.",
            },
        }

        with_feet = {
            'description': "The raw initial data with the added feet for the interpolation.",
            'coords': {
                'data': None, # TODO: need to decide how to add the feet data
                'unit': 'none',
                'description': ("The index coordinates of the initial voxels.\n"
                                "To get the positions with respect to the Sun, then dx, x_min, y_min, z_min need to be taken into account."),
            },
            'value': {
                'data': None, #TODO: need to create that data
                'unit': 'none',
                'description': 'The values for each pixel',
            },
        }

        dx = {
            'data': self.dx,
            'unit': 'km',
            'description': "The voxel resolution.",
        }
        x_min = {
            'data': self.x_min,
            'unit': 'km',
            'description': ("The minimum X-axis Carrington Heliographic Coordinates value for each data cube.\n"
                            "The X-axis in Carrington Heliographic Coordinates points towards the First Point of Aries."),
        } # TODO: need to check what the x, y, z axis are actually called to be able to fill up the description.
        y_min = {
            'data': self.y_min,
            'unit': 'km',
            'description': ("The minimum Y-axis Carrington Heliographic Coordinates value for each data cube.\n"
                            "The Y-axis in Carrington Heliographic Coordinates points towards the ecliptic's eastern horizon."),
        }
        z_min = {
            'data': self.z_min,
            'unit': 'km',
            'description': ("The minimum Z-axis Carrington Heliographic Coordinates value for each data cube.\n"
                            "The Z-axis in Carrington Heliographic Coordinates points towards Sun's north pole."),
        }
        info = {
            'description': description,
            'raw': raw,
            'withFeet': None, 
            'dx': dx,
            'x_min': x_min,
            'y_min': y_min,
            'z_min': z_min,
        }
        return info

    def sparse_data(self, cubes: np.ndarray) -> sparse.COO:
        """
        Changes data to a sparse representation.

        Args:
            cubes (np.ndarray): the initial array.

        Returns:
            sparse.COO: the corresponding sparse COO array.
        """

        cubes = sparse.COO(cubes)  # the .to_numpy() method wasn't used as the idx_type argument isn't working properly
        cubes.coords = cubes.coords.astype('uint16')  # to save memory
        return cubes
    


if __name__=='__main__':

    Cubes('testing3.h5')
    

