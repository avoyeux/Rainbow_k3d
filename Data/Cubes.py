"""
To create the HDF5 data files for the cubes.
"""

# IMPORTS
import os
import re
import h5py
import sys
import sunpy
import sparse

import numpy as np
import multiprocessing as mp

# Submodules import
import sunpy.coordinates
import multiprocessing.queues  # necessary as queues doesn't seem to be in the __init__
from scipy.io import readsav
from datetime import datetime
from astropy import units as u
from astropy import coordinates

# Personal imports
sys.path.append(os.path.join('..'))
from Common import MultiProcessing



class SavingTest:
    """
    To create cubes with feet and without feet in an HDF5 format.
    """

    def __init__(self, filename: str, processes: int, feet_lonlat: tuple[tuple[int, int], ...] = ((-177, 15), (-163, -16))) -> None:

        # Initial attributes
        self.filename = filename
        self.processes = processes

        # Constants
        self.solar_r = 6.96e5  # in km

        # Attributes setup
        self.feet = self.setup_feet(feet_lonlat)
        self.dx: dict[str, any]  # information and value of the spatial resolution
        self.time_indexes: list[int]  # the time indexes with values inside the cubes

        # Created attributes
        self.paths = self.setup_path()

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
    
    def setup_feet(self, lonlat: tuple[tuple[int, int], ...]) -> coordinates.SkyCoord:
        """
        Gives the 2 feet positions as an astropy.coordinates.SkyCoord object in Carrington Heliographic Coordinates. 

        Args:
            lonlat (tuple[tuple[int, int], ...]): _description_

        Returns:
            coordinates.SkyCoord: _description_
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
    
    def create(self) -> None:
        """
        Main function that encapsulates the file creation and closing with a with statement.
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
            self.dx, init_borders = self.create_borders(values)
            H5PYFile = self.foundation(H5PYFile)

            # Setup multiprocessing
            manager = mp.Manager()
            queue = manager.Queue()
            filepaths_nb = len(self.filepaths)
            processes_nb = min(self.processes, filepaths_nb)
            indexes = MultiProcessing.pool_indexes(filepaths_nb, processes_nb)
            # Multiprocessing
            processes = [None] * processes_nb
            for i, index in enumerate(indexes): 
                process = mp.Process(target=self.rawCubes, args=(queue, i, index))
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
    
    def foundation(self, H5PYFile: h5py.File) -> h5py.File:
        """
        For the main file metadata before getting to the HDF5 datasets and groups.

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
        H5PYFile = self.add_dataset(H5PYFile, self.dx, 'dx')
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
        
        dataset = group.require_dataset(name, shape=info['data'].shape, dtype=info['data'].dtype, data=info['data']) if name != '' else group
        for key, item in info.items():
            if key=='data': continue
            dataset.attrs[key] = item 
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

        new_group = group.require_group(name)
        for key, item in info.items():
            if isinstance(item, str): 
                new_group.attrs[key] = item
            elif 'data' in item.keys():
                new_group = self.add_dataset(new_group, item, key)
            else:
                new_group = self.add_group(new_group, item, key)
        return group

    def create_borders(self, values: tuple[float, ...]) -> tuple[dict[str, any], dict[str, dict[str, str | float]]]:
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
    
    def init_group(self, H5PYFile: h5py.File, data: sparse.COO, borders: dict[str, dict[str, any]]) -> h5py.File:
        """
        To create the initial h5py.Group object; where the raw data (with/without feet and/or in Carrington Heliographic Coordinates).

        Args:
            H5PYFile (h5py.File): the opened file pointer.
            data (sparse.COO): the initial raw cubes data.
            borders (dict[str, dict[str, any]]): the border info (i.e. x_min, etc.) for the given data.

        Returns:
            h5py.File: the opened file pointer.
        """

        # Group setup
        group = H5PYFile.create_group('L0 data')
        group.attrs['description'] = (
            "The filament voxels in sparse COO format (i.e. with a coords and values arrays) of the initial cubes gotten from Dr. Karine [...]'s work.\n"
            "Furthermore, the necessary information to be able to position the filament relative to the Sun are also available. "
            "Both cubes, with or without feet, are inside this group."
        )

        # Add raw cubes group
        group = self.add_cube(group, data, 'Raw cube', borders)
        group['Raw data'].attrs['description'] = "The initial voxel data in COO format without the feet for the interpolation."

        # Add raw skycoords group
        group = self.add_skycoords(group, data, 'Raw coordinates', borders)
        group['Raw coordinates'].attrs['description'] = 'The initial data saved as Carrington Heliographic Coordinates in km.'

        # Add raw feet
        data, borders = self.with_feet(data, borders)
        group = self.add_cube(group, data, 'Raw cubes with feet', borders)
        group['Raw with feet'].attrs['description'] = 'The initial raw data in COO format with the feet positions added.'

        # Add raw skycoords with feet
        group = self.add_skycoords(group, data, 'Raw coordinates with feet', borders)
        group['Raw coordinates with feet'].attrs['description'] = 'The initial data with the feet positions added saved as Carrington Heliographic Coordinates in km.'
        return H5PYFile
    
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
                'data': data.coords,
                'unit': 'none',
                'description': ("The index coordinates of the initial voxels.\n"
                                "The shape is (4, N) where the rows represent t, x, y, z where t the time index (i.e. which cube it is), and N the total number "
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
        data = np.hstack(data)

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
        positions = np.round(positions).astype('uint16') 

        # Setup cubes with feet
        init_coords = data.coords
        time_indexes = list(set(init_coords[0, :]))
        # (x, y, z) -> (t, x, y, z)
        feet = np.hstack([
            np.vstack((np.full((1, 2), time), positions))
            for time in time_indexes
        ])

        # Add feet 
        init_coords = np.hstack([init_coords, feet]).astype('int32')  # TODO: will change it to uint16 when I am sure that it is working as intended

        # Indexes to positive values
        x_min, y_min, z_min = np.min(positions, axis=1).astype(int)  
        if x_min < 0: init_coords[1, :] -= x_min
        if y_min < 0: init_coords[2, :] -= y_min
        if z_min < 0: init_coords[3, :] -= z_min

        # Changing to COO 
        shape = np.max(init_coords, axis=1) + 1
        feet_values = np.repeat(np.array([0b00100000], dtype='uint8'), len(time_indexes) * 2)  #TODO: need to add the values information in the HDF5 file info
        values = np.concatenate([data.data.astype('uint8'), feet_values], axis=0)
        data = sparse.COO(coords=init_coords, data=values, shape=shape)
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
    
    def skyCoords_slice(self, coords: dict[str, any], input_queue: mp.queues.Queue, output_queue: mp.queues.Queue) -> None:
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
    


if __name__=='__main__':

    SavingTest('testing10.h5', processes=10)    

