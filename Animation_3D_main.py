"""
Imports of the data, preprocessing and creation of the 3D k3d visualisation class to then be used in a jupyter notebook.
"""

# Imports
import os
import re
import k3d
import time
import threading

import numpy as np
import ipywidgets as widgets

from glob import glob
from time import sleep
from astropy.io import fits
from scipy.io import readsav
from scipy.ndimage import label
from typeguard import typechecked
from IPython.display import display
from sparse import COO, stack, concatenate
from multiprocessing import shared_memory, Process, Manager
# Personal imports
from common_alf import Decorators

from skimage.morphology import skeletonize_3d
from scipy.ndimage import binary_fill_holes


class CustomDate:
    """
    To separate the year, month, day, hour, minute, second from a string dateutil.parser.parser
    doesn't work in this case. 
    """

    @typechecked
    def __init__(self, date_str: str | bytes):
        self.year = None
        self.month = None
        self.day = None
        self.hour = None
        self.minute = None
        self.second = None

        if isinstance(date_str, str):
            self.Parse_date_str(date_str=date_str)
        elif isinstance(date_str, bytes):
            self.Parse_date_bytes(date_str=date_str)

    def Parse_date_str(self, date_str: str):
        """
        Separating a string in the format YYYY-MM-DDThh-mm-ss to get the different time attributes.
        """

        date_part, time_part = date_str.split("T")
        self.year, self.month, self.day = map(int, date_part.split("-"))
        self.hour, self.minute, self.second = map(int, time_part.split("-"))
    
    def Parse_date_bytes(self, date_str: bytes):
        """
        Separating a bytestring in the format YYYY/MM/DD hh:mm:ss to get the different date attributes.
        """

        date_part, time_part = date_str.split(b' ')
        self.year, self.month, self.day = map(int, date_part.split(b"/"))
        self.hour, self.minute, self.second = map(int, time_part.split(b':'))


class Data:
    """
    To upload and manipulate the data to then be inputted in the k3d library for 3D animations.
    """

    @Decorators.running_time  # gives the start and the end time of the function
    @typechecked  # checks the type for the inputs at runtime 
    def __init__(self, everything: bool = False, both_cubes: str | bool = 'Karine', sun: bool = False, 
                 all_data: bool = False, duplicates: bool = False, no_duplicate: bool = False, line_of_sight: bool = False, 
                 trace_data: bool = False, trace_no_duplicate: bool = False, time_intervals_all_data: bool = False, 
                 time_intervals_no_duplicate: bool = False, time_interval: str | int = 1, heliographic_grid_degrees: int | float = 15, 
                 fov_center: tuple[int | float, int | float, int | float] | str = 'cubes', sun_texture_resolution: int = 960,
                 sdo_pov: bool = False, stereo_pov: bool = False, batch_number: int = 10, make_screenshots: bool = False, cube_version: str = 'old',
                 convolution_3d: bool = False, conv_treshold: int = 125, polynomials: bool = False, skeleton: bool = False, 
                 html_snapshot: bool = False):
        
        # Arguments
        self.first_cube = False
        self.second_cube = False
        self.cube_version_0 = False
        self.cube_version_1 = False
        if 'old' in cube_version:
            self.cube_version_0 = True
        elif 'new' in cube_version:
            self.cube_version_1 = True
        elif 'both' in cube_version:
            self.cube_version_0 = True
            self.cube_version_1 = True
        else:
            raise ValueError(f'"cube_version" needs to be "old", "new" or "both".')

        if isinstance(both_cubes, str):
            if 'alf' in both_cubes.lower():
                self.first_cube = True
            elif 'kar' in both_cubes.lower():
                self.second_cube = True
            else:
                raise ValueError('Wrong string value for argument both_cubes. It has to contain "alf" or "kar".')
        elif isinstance(both_cubes, bool):
            self.first_cube = True
            self.second_cube = (both_cubes or everything)

        if isinstance(fov_center, tuple):
            self.fov_center = fov_center  # position the camera aims at
        elif 'cub' in fov_center.lower():
            self.fov_center = True  # using the cubes center as the fov_center
        elif 'stereo' in fov_center.lower():
            self.fov_center = False  # using the stereo fov center as the fov center
        else:
            raise ValueError("If 'fov_center' a string, needs to have 'cub' or 'stereo' inside it.")

        self.sun = (sun or everything)  # choosing to plot the Sun
        self.all_data = (all_data or everything)  # choosing to plot all the data (i.e. data containing the duplicates)
        self.duplicates = (duplicates or everything)  # for the duplicates data (i.e. from SDO and STEREO)
        self.no_duplicate = (no_duplicate or everything) # for the no duplicates data
        self.line_of_sight = (line_of_sight or everything)  # for the line of sight data 
        self.trace_data = (trace_data or everything) # for the trace of all_data
        self.trace_no_duplicate = (trace_no_duplicate or everything)  # same for no_duplicate
        self.time_intervals_all_data = (time_intervals_all_data or everything)  # for the data integration over time for all_data
        self.time_intervals_no_duplicate = (time_intervals_no_duplicate or everything)  # same for no_duplicate
        self.time_interval = time_interval  # time interval in hours (if 'int' or 'h' in str), days (if 'd' in str), minutes (if 'min' in str)  
        self._sun_texture_resolution = sun_texture_resolution  # choosing the Sun's texture resolution
        self.stereo_pov = stereo_pov  # following the STEREO pov
        self.sdo_pov = sdo_pov  # same for the SDO pov 
        self._batch_number = batch_number  # number of batches for the I/O bound tasks
        self.make_screenshots = make_screenshots  # creating screenshots when clicking play
        self._heliographic_grid_degrees = heliographic_grid_degrees  # heliographic grid steps in degrees
        self.convolution = convolution_3d
        self.conv_treshold = conv_treshold
        self.polynomials = polynomials
        self.skeleton = skeleton
        self.html_snapshot = html_snapshot

        # Instance attributes set when running the class
        self.paths = None  # dictionary containing all the path names and the corresponding path
        self._cube_names_all = None  # all the cube names that will be used
        self._cube_names_1 = None  # cube names for the first set 
        self._cube_names_2 = None  # for the second set
        self.cube_numbers_all = None  # sorted list of the number corresponding to each used cube
        self._cube_numbers_1 = None  # same but only for the first set
        self._cube_numbers_2 = None  # same for the second set
        self.dates_all = None  # list of the date with .year, .month, etc for all the cubes
        self.dates_1 = None  # same for the first set of cubes
        self.dates_2 = None  # same for the second set of cubes 
        self.cubes_shape = None  # the shape of cube 1 or cube 2 if cube 2 is also chosen
        self.cubes_all_data_1 = None  # (sparse or not) boolean 4D array of all the data for the first set
        self.cubes_all_data_2 = None  # same for the second set 
        self.cubes_lineofsight_STEREO_1 = None  # (sparse or not) boolean 4D array of the line of sight data seen from STEREO for the first set
        self.cubes_lineofsight_STEREO_2 = None  # same for the second set
        self.cubes_lineofsight_SDO_1 = None  # same for SDO and the first set
        self.cubes_lineofsight_SDO_2 = None  # same for the second set
        self.cubes_no_duplicate_1 = None  # (sparse or not) boolean 4D array for the no duplicates for the first set
        self.cubes_no_duplicate_2 = None  # same for the second set
        self.cubes_no_duplicates_STEREO_1 = None  # same seen from STEREO for the first set
        self.cubes_no_duplicates_STEREO_2 = None  # same for the second set
        self.cubes_no_duplicates_SDO_1 = None  # same seen from SDO for the first set
        self.cubes_no_duplicates_SDO_2 = None  # same for the second set 
        self.trace_cubes_1 = None  # boolean 3D array of all the data for the first set 
        self.trace_cubes_2 = None  # same for the second set
        self.trace_cubes_no_duplicate_1 = None  # same  for the no duplicate data for the first set
        self.trace_cubes_no_duplicate_2 = None  # same for the second set 
        self.time_cubes_all_data_1 = None  # (sparse or not) boolean 4D array of the integration of all_data over time_interval for the first set
        self.time_cubes_all_data_2 = None  # same for the second set
        self.time_cubes_no_duplicate_1 = None  # same for the no duplicate for the first set
        self.time_cubes_no_duplicate_2 = None  # same for the second set 
        self.radius_index = None  # radius of the Sun in grid units
        self.sun_center = None  # position of the Sun's center [x, y, z] in the grid
        self._texture_height = None  # height in pixels of the input texture image
        self._texture_width = None  # width in pixels of the input texture image 
        self._sun_texture = None  # Sun's texture image after some visualisation treatment
        self.sun_points = None  # positions of the pixels for the Sun's texture
        self._sun_texture_x = None  # 1D array with values corresponding to the height texture image indexes and array indexes to the theta direction position
        self._sun_texture_y = None  # same for width and phi direction
        self.hex_colours = None  # 1D array with values being integer hex colours and indexes being the position in the Sun's surface
        self._pattern_int = None  # re.compile pattern of the 'int' STEREO .png filenames
        self._all_filenames = None  # all the 'int' filenames for both cubes
        self._days_per_month = None  # list of the number of day per each month in a normal year
        self._length_dx = None  # x direction unit voxel size in km
        self._length_dy = None  # same for the y direction
        self._length_dz = None  # same for the z direction
        self.STEREO_pos = None  # array giving the position of the STEREO satellite for each time step
        self.SDO_pos = None  # same for SDO

        # Functions
        self.Paths()
        self.Names()
        self.Sun_pos()
        self.Dates_all_data_sets()            
        self.Choices()
        self.Complete_sparse_arrays()

        # Deleting the private class attributes
        self.Attribute_deletion()

    def Prepocessing_polynomial_data(self):
        """
        Code to get the .npy barycenter filenames
        """

        pattern = re.compile(r'''poly_
                             (?P<datatype>[a-zA-Z0-9]+)_
                             (lim_(?P<conv_limit>\d+)_)?
                             order(?P<order>\d+)
                             \.npy''', re.VERBOSE)
        
        files_dataNmatches = []
        for filename in os.listdir(self.paths['polynomials']):
            filename_match = pattern.match(filename)

            if filename_match:
                file_data = np.load(os.path.join(self.paths['polynomials'], filename)).astype('uint16')
                file_data = file_data[[0, 3, 2, 1]]
                file_data = COO(coords=file_data, data=1, shape=self.cubes_shape)
                files_dataNmatches.append((filename_match, file_data))
            else:
                print(f"\033[92mPolynomial array filename {filename} doesn't match the usual pattern. \033[0m")
        self.polynomials_matchesNdata = files_dataNmatches

    def Paths(self):
        """
        Input and output paths manager.
        """

        main_path = '../'
        self.paths = {
            'Main': main_path,
            'Cubes': os.path.join(main_path, 'Cubes'),
            'Cubes_karine': os.path.join(main_path, 'Cubes_karine'),
            'Textures': os.path.join(main_path, 'Textures'),
            'Intensities': os.path.join(main_path, 'STEREO', 'int'),
            'SDO': os.path.join(main_path, 'sdo'),
            'polynomials': os.path.join(main_path, 'curveFitArrays'),
            }    
    
    def Choices(self):
        """
        To choose what is computed and added depending on the arguments chosen.
        """

        if self.time_intervals_all_data or self.time_intervals_no_duplicate: self.Time_interval()

        if self.first_cube:
            self.dates_1 = self.Dates_n_times(self._cube_numbers_1)
            self.cubes_all_data_1, self.cubes_no_duplicate_init_1, self.cubes_no_duplicates_init_STEREO_1, self.cubes_no_duplicates_init_SDO_1, \
            self.cubes_no_duplicate_new_1, self.cubes_no_duplicates_new_STEREO_1, self.cubes_no_duplicates_new_SDO_1, \
            self.trace_cubes_1, self.trace_cubes_no_duplicate_1, self.time_cubes_all_data_1, self.time_cubes_no_duplicate_init_1, \
            self.time_cubes_no_duplicate_new_1, self.cubes_lineofsight_STEREO_1, self.cubes_lineofsight_SDO_1 \
            = self.Processing_data(self.paths['Cubes'], self._cube_names_1, self.dates_1)
            
        if self.second_cube:
            self.dates_2 = self.Dates_n_times(self._cube_numbers_2)
            self.cubes_all_data_2, self.cubes_no_duplicate_init_2, self.cubes_no_duplicates_init_STEREO_2, self.cubes_no_duplicates_init_SDO_2, \
            self.cubes_no_duplicate_new_2, self.cubes_no_duplicates_new_STEREO_2, self.cubes_no_duplicates_new_SDO_2, \
            self.trace_cubes_2, self.trace_cubes_no_duplicate_2, self.time_cubes_all_data_2, self.time_cubes_no_duplicate_init_2, \
            self.time_cubes_no_duplicate_new_2, self.cubes_lineofsight_STEREO_2, self.cubes_lineofsight_SDO_2 \
            = self.Processing_data(self.paths['Cubes_karine'], self._cube_names_2, self.dates_2)

        if self.sun:
            self.Sun_texture()
            self.Sun_points()
            self.Colours_1D()
                         
        if self.stereo_pov:
            self.STEREO_stats()
        elif self.sdo_pov:
            self.SDO_stats()

        if not self.fov_center: self.STEREO_pov_center()

        if self.convolution: self.Conv3d_results()

        if self.polynomials: self.Prepocessing_polynomial_data()

    def Names(self):
        """
        To get the filenames of all the cubes.
        """

        # Setting the cube name pattern (only cube{:03d}.save files are kept)
        pattern = re.compile(r'cube(\d{3})\.save')

        # The cube names
        cube_names = []

        if self.first_cube:
            cube_names.extend([cube_name for cube_name in os.listdir(self.paths['Cubes']) \
                      if pattern.match(cube_name)])
            self._cube_names_1 = sorted(cube_names)  # first data set
            self._cube_numbers_1 = [int(pattern.match(cube_name).group(1)) for cube_name in self._cube_names_1]

        if self.second_cube:
            cube_names_2 = [cube_name for cube_name in os.listdir(self.paths['Cubes_karine']) \
                               if pattern.match(cube_name)] 
            cube_names.extend(cube_names_2)
            self._cube_names_2 = sorted(cube_names_2) # second set
            self._cube_numbers_2 = [int(pattern.match(cube_name).group(1)) for cube_name in self._cube_names_2]

        if not self.make_screenshots:
            self._cube_names_all = [cube_name for cube_name in set(cube_names)]
            self._cube_names_all.sort()

            # Getting the corresponding cube_numbers 
            self.cube_numbers_all = [int(pattern.match(cube_name).group(1)) for cube_name in self._cube_names_all]
        else:
            self.cube_numbers_all = np.arange(0, 413) # all the possible numbers for the gifs

    def Dates_all_data_sets(self):
        """
        To get the dates and times corresponding to all the used cubes.
        To do so images where both numbers are in the filename are used.
        """

        self._pattern_int = re.compile(r'\d{4}_(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2})\.\d{3}\.png')
        self._all_filenames = sorted(glob(os.path.join(self.paths['Intensities'], '*.png')))

        if not self.make_screenshots:
            # Getting the corresponding filenames 
            filenames = []
            for number in self.cube_numbers_all: 
                for filepath in self._all_filenames:
                    filename = os.path.basename(filepath)
                    if filename[:4] == f'{number:04d}':
                        filenames.append(filename)
                        break
            self.dates_all = [CustomDate(self._pattern_int.match(filename).group(1)) for filename in filenames]
        else:
            self.dates_all = [CustomDate(self._pattern_int.match(os.path.basename(filename)).group(1)) 
                            for filename in self._all_filenames]

    def Dates_n_times(self, cube_numbers):
        """
        To get the dates and times corresponding to each cube sets. 
        To do so images where the cube number is in the filename is used.
        """

        # Getting the corresponding filenames 
        filenames = []
        for number in cube_numbers:
            for filepath in self._all_filenames:
                filename = os.path.basename(filepath)
                if filename[:4] == f'{number:04d}':
                    filenames.append(filename)
                    break
        dates = [CustomDate(self._pattern_int.match(filename).group(1)) for filename in filenames]
        return dates
    
    @Decorators.running_time
    def Processing_data(self, cubes_path, cube_names, dates):
        """
        Downloading and processing the data depending on the arguments chosen.
        """

        # Multiprocessing initial I/O bound tasks
        IO_processes = []
        manager = Manager()
        queue = manager.Queue()
        total_length = len(cube_names)
        step = int(np.ceil(total_length / self._batch_number))

        for i in range(self._batch_number):
            if not (i==self._batch_number - 1):
                IO_processes.append(Process(target=self.Cubes, args=(queue, i, cubes_path, cube_names[step * i:step * (i + 1)])))
                if self.line_of_sight:
                    IO_processes.append(Process(target=self.Cubes_lineofsight_STEREO, args=(queue, self._batch_number + i, cubes_path, cube_names[step * i:step * (i + 1)])))
                    IO_processes.append(Process(target=self.Cubes_lineofsight_SDO, args=(queue, 2 * self._batch_number + i, cubes_path, cube_names[step * i:step * (i + 1)])))
            else:
                IO_processes.append(Process(target=self.Cubes, args=(queue, i, cubes_path, cube_names[step * i:])))
                if self.line_of_sight:
                    IO_processes.append(Process(target=self.Cubes_lineofsight_STEREO, args=(queue, self._batch_number + i, cubes_path, cube_names[step * i:])))
                    IO_processes.append(Process(target=self.Cubes_lineofsight_SDO, args=(queue, 2 * self._batch_number + i, cubes_path, cube_names[step * i:]))) 
        # Ordering the results gotten from the I/O bound tasks
        results = [None for _ in range(self._batch_number * 3)]
        for p in IO_processes: p.start()
        for p in IO_processes: p.join()

        while not queue.empty():
            identifier, result = queue.get()
            results[identifier] = result
        cubes = concatenate(results[:self._batch_number], axis=0)
        print(f'The cubes shape is {cubes.shape}')
        if self.line_of_sight:
            STEREO = concatenate(results[self._batch_number:self._batch_number * 2], axis=0)
            SDO = concatenate(results[self._batch_number * 2:], axis=0)

        self.cubes_shape = cubes.shape
        print(f'CUBES - {round(cubes.nbytes/ 2**20,3)}Mb')

        # CPU bound processes 
        processes = []
        results = [None for _ in range(14)]
        if self.line_of_sight:
            results[-2] = STEREO
            results[-1] = SDO
        
        # Shared memory
        sparse_data = cubes.data
        sparse_coords = cubes.coords
        # Shared memory properties
        self._sparse_data_shape = sparse_data.shape
        self._sparse_coords_shape = sparse_coords.shape
        self.sparse_data_dtype = sparse_data.dtype
        self.sparse_coords_dtype = sparse_coords.dtype
        # Shared memory object and np.ndarray
        shm_data = self.Shared_memory(sparse_data, self.sparse_data_dtype)
        shm_coords = self.Shared_memory(sparse_coords, self.sparse_coords_dtype)
        self._shm_data_name, self._shm_coords_name = shm_data.name, shm_coords.name

        # Separating the data
        if self.all_data:
            processes.append(Process(target=self.Cubes_all_data, args=(queue,)))
        if self.no_duplicate:
            if self.cube_version_0:
                processes.append(Process(target=self.Cubes_no_duplicate_init, args=(queue,)))
            if self.cube_version_1:
                processes.append(Process(target=self.Cubes_no_duplicate_new, args=(queue,)))
        if self.duplicates:
            if self.cube_version_0:
                processes.append(Process(target=self.Cubes_STEREO_no_duplicate_init, args=(queue,)))
                processes.append(Process(target=self.Cubes_SDO_no_duplicate_init, args=(queue,)))
            if self.cube_version_1:
                processes.append(Process(target=self.Cubes_STEREO_no_duplicate_new, args=(queue,)))
                processes.append(Process(target=self.Cubes_SDO_no_duplicate_new, args=(queue,)))
        if self.trace_data:
            processes.append(Process(target=self.Cubes_trace, args=(queue,)))
        if self.trace_no_duplicate:
            processes.append(Process(target=self.Cubes_trace_no_duplicate, args=(queue,)))
        if self.time_intervals_all_data or self.time_intervals_no_duplicate:
            self._days_per_month = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
            date = self.dates_all[0]
            if (date.year % 4 == 0 and date.year % 100 !=0) or (date.year % 400 == 0):  # Only works if the year doesn't change
                self._days_per_month[2] = 29  # for leap years
        if self.time_intervals_all_data:
            processes.append(Process(target=self.Cubes_time_chunks, args=(queue, dates)))

        if self.time_intervals_no_duplicate:
            processes.append(Process(target=self.Cubes_time_chunks_no_duplicate_init, args=(queue, dates)))
            processes.append(Process(target=self.Cubes_time_chunks_no_duplicate_new, args=(queue, dates)))

        for p in processes: p.start()
        for p in processes: p.join()

        shm_data.close()
        shm_data.unlink()
        shm_coords.close()
        shm_coords.unlink()

        # Ordering and saving the results
        while not queue.empty():
            identifier, result = queue.get()
            results[identifier] = result
        return results
    
    def Sparse_data(self, cubes):
        """
        To make the voxel positions np.ndarrays less memory heavy by taking into account that they are sparse arrays.
        """

        cubes = COO(cubes)  # the .to_numpy() method wasn't used as the idx_type argument isn't working properly
        cubes.coords = cubes.coords.astype('uint16')  # to save memory
        return cubes
    
    def Cubes(self, queue, queue_index, path, names):
        """
        To import the cubes in sections as it is mainly an I/O bound task.
        """

        cubes = [readsav(os.path.join(path, cube_name)).cube.astype('uint8') for cube_name in names]
        cubes = np.array(cubes)  
        cubes = np.transpose(cubes, (0, 3, 2, 1))
        cubes = self.Sparse_data(cubes)
        queue.put((queue_index, cubes))

    def Cubes_lineofsight_STEREO(self, queue, queue_index, path, names):
        """
        To trace the cubes for the line of sight STEREO data. Also imported in sections as it is mainly an I/O bound task.
        """

        cubes_lineofsight_STEREO = [readsav(os.path.join(path, cube_name)).cube1.astype('uint8') for cube_name in names]  
        cubes_lineofsight_STEREO = np.array(cubes_lineofsight_STEREO)  # line of sight seen from STEREO 
        cubes_lineofsight_STEREO = np.transpose(cubes_lineofsight_STEREO, (0, 3, 2, 1))
        cubes_lineofsight_STEREO = self.Sparse_data(cubes_lineofsight_STEREO)
        queue.put((queue_index, cubes_lineofsight_STEREO))
        
    def Cubes_lineofsight_SDO(self, queue, queue_index, path, names):
        """
        To trace the cubes for the line of sight SDO data. Also imported in sections as it is mainly an I/O bound task.
        """

        cubes_lineofsight_SDO = [readsav(os.path.join(path, cube_name)).cube2.astype('uint8') for cube_name in names]
        cubes_lineofsight_SDO = np.array(cubes_lineofsight_SDO)  # line of sight seen from SDO
        cubes_lineofsight_SDO = np.transpose(cubes_lineofsight_SDO, (0, 3, 2, 1))
        cubes_lineofsight_SDO = self.Sparse_data(cubes_lineofsight_SDO)
        queue.put((queue_index, cubes_lineofsight_SDO))

    def Shared_memory(self, array, dtype):
        """
        Creating a shared memory space given an input np.ndarray.
        """

        shm = shared_memory.SharedMemory(create=True, size=array.nbytes)
        shared_array = np.ndarray(array.shape, dtype=dtype, buffer=shm.buf)
        np.copyto(shared_array, array)
        return shm 
    
    def Shared_array_reconstruction(self):
        """
        To reconstruct the shared COO array as it is separated in a data and a coords np.ndarray.
        """

        shm_data = shared_memory.SharedMemory(name=self._shm_data_name)
        shm_coords = shared_memory.SharedMemory(name=self._shm_coords_name)

        data = np.ndarray(self._sparse_data_shape, dtype=self.sparse_data_dtype, buffer=shm_data.buf)
        coords = np.ndarray(self._sparse_coords_shape, dtype=self.sparse_coords_dtype, buffer=shm_coords.buf)
        cubes = COO(coords=coords, data=data, shape=self.cubes_shape)
        cubes = COO.copy(cubes)  # had to add a .copy() as it wasn't working properly
        shm_data.close()
        shm_coords.close()
        return cubes

    def Cubes_all_data(self, queue):
        """
        To create the cubes for all the data.
        """

        cubes = self.Shared_array_reconstruction()
        cubes_all_data = self.Sparse_data(cubes & 0b00000001).astype('uint8')
        queue.put((0, cubes_all_data))
    
    def Cubes_no_duplicate_init(self, queue):
        """
        To create the cubes for the no duplicate data.
        """

        cubes = self.Shared_array_reconstruction()
        cubes_no_duplicate = self.Sparse_data((cubes & 0b00000110) == 6).astype('uint8')  # no  duplicates 
        queue.put((1, cubes_no_duplicate))

    def Cubes_STEREO_no_duplicate_init(self, queue):
        """
        To create the cubes for the no duplicate data from STEREO.
        """

        cubes = self.Shared_array_reconstruction()
        cubes_no_duplicates_STEREO = self.Sparse_data((cubes & 0b00000010) > 0).astype('uint8')
        queue.put((2, cubes_no_duplicates_STEREO))
    
    def Cubes_SDO_no_duplicate_init(self, queue):
        """
        To create the cubes for the no duplicate data from SDO.
        """

        cubes = self.Shared_array_reconstruction()
        cubes_no_duplicates_SDO = self.Sparse_data((cubes & 0b00000100) > 0).astype('uint8')
        queue.put((3, cubes_no_duplicates_SDO))
        
    def Cubes_no_duplicate_new(self, queue):
        """
        To create the cubes for the no duplicate data.
        """

        cubes = self.Shared_array_reconstruction()
        cubes_no_duplicate = self.Sparse_data((cubes & 0b00011000) == 24).astype('uint8')  # no  duplicates 
        queue.put((4, cubes_no_duplicate))

    def Cubes_STEREO_no_duplicate_new(self, queue):
        """
        To create the cubes for the no duplicate data from STEREO.
        """

        cubes = self.Shared_array_reconstruction()
        cubes_no_duplicates_STEREO = self.Sparse_data((cubes & 0b00001000) > 0).astype('uint8')
        queue.put((5, cubes_no_duplicates_STEREO))
    
    def Cubes_SDO_no_duplicate_new(self, queue):
        """
        To create the cubes for the no duplicate data from SDO.
        """

        cubes = self.Shared_array_reconstruction()
        cubes_no_duplicates_SDO = self.Sparse_data((cubes & 0b00010000) > 0).astype('uint8')
        queue.put((6, cubes_no_duplicates_SDO))

    def Cubes_trace(self, queue):
        """
        To create the cubes of the trace of all the data.
        """

        cubes = self.Shared_array_reconstruction()
        trace_cube = COO.any(cubes, axis=0).astype('uint8')
        queue.put((7, trace_cube))

    def Cubes_trace_no_duplicate(self, queue):
        """
        To create the cubes of the trace of the no duplicate.
        """

        cubes = self.Shared_array_reconstruction()
        cubes_no_duplicate = self.Sparse_data((cubes & 0b00000110) == 6).astype('uint8')
        trace_cube_no_duplicate = COO.any(cubes_no_duplicate, axis=0).astype('uint8')
        queue.put((8, trace_cube_no_duplicate))

    def Cubes_time_chunks(self, queue, dates):
        """
        To create the cubes for the time integrations for all data.
        """

        cubes = self.Shared_array_reconstruction()
        time_cubes_all_data = []
        cubes_all_data = self.Sparse_data(cubes & 0b00000001).astype('uint8')
        for date in self.dates_all:
            date_seconds = (((self._days_per_month[date.month] + date.day) * 24 + date.hour) * 60 + date.minute) * 60 + date.second

            date_min = date_seconds - self.time_interval / 2
            date_max = date_seconds + self.time_interval / 2      
            time_cubes_all_data.append(self.Time_chunks(dates, cubes_all_data, date_max, date_min)) 
        
        time_cubes_all_data = stack(time_cubes_all_data, axis=0)
        time_cubes_all_data = self.Sparse_data(time_cubes_all_data).astype('uint8')
        queue.put((9, time_cubes_all_data))

    def Cubes_time_chunks_no_duplicate_init(self, queue, dates):
        """
        To create the cubes for the time integrations for the no duplicates.
        """

        cubes = self.Shared_array_reconstruction()
        time_cubes_no_duplicate = [] 
        cubes_no_duplicate = self.Sparse_data((cubes & 0b00000110) == 6).astype('uint8')
        for date in self.dates_all:
            date_seconds = (((self._days_per_month[date.month] + date.day) * 24 + date.hour) * 60 + date.minute) * 60 + date.second

            date_min = date_seconds - self.time_interval / 2
            date_max = date_seconds + self.time_interval / 2
            time_cubes_no_duplicate.append(self.Time_chunks(dates, cubes_no_duplicate, date_max, date_min))
        
        time_cubes_no_duplicate = stack(time_cubes_no_duplicate, axis=0)
        time_cubes_no_duplicate = self.Sparse_data(time_cubes_no_duplicate).astype('uint8')
        queue.put((10, time_cubes_no_duplicate))

    def Cubes_time_chunks_no_duplicate_new(self, queue, dates):
        """
        To create the cubes for the time integrations for the no duplicates.
        """

        cubes = self.Shared_array_reconstruction()
        time_cubes_no_duplicate = [] 
        cubes_no_duplicate = self.Sparse_data((cubes & 0b00011000) == 24).astype('uint8')
        for date in self.dates_all:
            date_seconds = (((self._days_per_month[date.month] + date.day) * 24 + date.hour) * 60 + date.minute) * 60 + date.second

            date_min = date_seconds - self.time_interval / 2
            date_max = date_seconds + self.time_interval / 2
            time_cubes_no_duplicate.append(self.Time_chunks(dates, cubes_no_duplicate, date_max, date_min))
        
        time_cubes_no_duplicate = stack(time_cubes_no_duplicate, axis=0)
        time_cubes_no_duplicate = self.Sparse_data(time_cubes_no_duplicate).astype('uint8')
        queue.put((11, time_cubes_no_duplicate))

    def Time_interval(self):
        """
        Checking the date interval type an assigning a value to be used in another function to get the min and max date.
        """

        if isinstance(self.time_interval, int):
            time_delta = self.time_interval * 3600
        elif isinstance(self.time_interval, str):
            number = re.search(r'\d+', self.time_interval)
            number = int(number.group())
            self.time_interval = self.time_interval.lower()
            if 'min' in self.time_interval: 
                time_delta = number * 60
            elif 'h' in self.time_interval:
                time_delta = number * 3600
            elif 'd' in self.time_interval:
                time_delta = number * 3600 * 24
            else:
                raise ValueError("Time_interval string value not supported. Add 'min' for minutes, 'h' for hours or 'd' for days.")
        else:
            raise TypeError('Problem for time interval. Typeguard should have already raised an error.')
        self.time_interval = time_delta

    def Time_chunks(self, dates, cubes, date_max, date_min):
        """
        To select the data in the time chunk given the data chosen for the integration.
        """

        chunk = []
        for date2, data2 in zip(dates, cubes):
            date_seconds2 = (((self._days_per_month[date2.month] + date2.day) * 24 + date2.hour) * 60 + date2.minute) * 60 \
                + date2.second
            
            if date_seconds2 < date_min:
                continue
            elif date_seconds2 <= date_max:
                chunk.append(data2)
            else:
                break
        if len(chunk) == 0:  # i.e. if nothing was found
            return COO(np.zeros((cubes.shape[1], cubes.shape[2], cubes.shape[3])))
        elif len(chunk) == 1:
            return chunk[0]
        else:
            chunk = stack(chunk, axis=0)
            return COO.any(chunk, axis=0)
          
    def Complete_sparse_arrays(self): #TODO: need to change this as it takes around 2 seconds to run
        """
        To reformat the sparse arrays so that the number of values is equal to the total number of cube numbers used.
        """

        if self.first_cube:
            cubes_lineofsight_STEREO_1 = []
            cubes_lineofsight_SDO_1 = []
            cubes_all_data_1 = []
            cubes_no_duplicate_init_1 = []
            cubes_no_duplicates_init_STEREO_1 = []
            cubes_no_duplicates_init_SDO_1 = []
            cubes_no_duplicate_new_1 = []
            cubes_no_duplicates_new_STEREO_1 = []
            cubes_no_duplicates_new_SDO_1 = []
            index = -1
            for number in self.cube_numbers_all:
                if number in self._cube_numbers_1:
                    index += 1
                    cubes_lineofsight_STEREO_1.append(self.cubes_lineofsight_STEREO_1[index] if self.line_of_sight else None)
                    cubes_lineofsight_SDO_1.append(self.cubes_lineofsight_SDO_1[index] if self.line_of_sight else None)
                    cubes_all_data_1.append(self.cubes_all_data_1[index] if self.all_data else None)
                    if self.cube_version_0:
                        cubes_no_duplicate_init_1.append(self.cubes_no_duplicate_init_1[index] if self.no_duplicate else None)
                        cubes_no_duplicates_init_STEREO_1.append(self.cubes_no_duplicates_init_STEREO_1[index] if self.duplicates else None)
                        cubes_no_duplicates_init_SDO_1.append(self.cubes_no_duplicates_init_SDO_1[index] if self.duplicates else None)
                    if self.cube_version_1:
                        cubes_no_duplicate_new_1.append(self.cubes_no_duplicate_new_1[index] if self.no_duplicate else None)
                        cubes_no_duplicates_new_STEREO_1.append(self.cubes_no_duplicates_new_STEREO_1[index] if self.duplicates else None)
                        cubes_no_duplicates_new_SDO_1.append(self.cubes_no_duplicates_new_SDO_1[index] if self.duplicates else None)
                else:
                    cubes_lineofsight_STEREO_1.append(None)
                    cubes_lineofsight_SDO_1.append(None)
                    cubes_all_data_1.append(None)
                    cubes_no_duplicate_init_1.append(None)
                    cubes_no_duplicates_init_STEREO_1.append(None)
                    cubes_no_duplicates_init_SDO_1.append(None)
                    cubes_no_duplicate_new_1.append(None)
                    cubes_no_duplicates_new_STEREO_1.append(None)
                    cubes_no_duplicates_new_SDO_1.append(None)
            self.cubes_lineofsight_STEREO_1 = cubes_lineofsight_STEREO_1
            self.cubes_lineofsight_SDO_1 = cubes_lineofsight_SDO_1
            self.cubes_all_data_1 = cubes_all_data_1
            self.cubes_no_duplicate_init_1 = cubes_no_duplicate_init_1
            self.cubes_no_duplicates_init_STEREO_1 = cubes_no_duplicates_init_STEREO_1
            self.cubes_no_duplicates_init_SDO_1 = cubes_no_duplicates_init_SDO_1
            self.cubes_no_duplicate_new_1 = cubes_no_duplicate_new_1
            self.cubes_no_duplicates_new_STEREO_1 = cubes_no_duplicates_new_STEREO_1
            self.cubes_no_duplicates_new_SDO_1 = cubes_no_duplicates_new_SDO_1

        if self.second_cube:
            cubes_lineofsight_STEREO_2 = []
            cubes_lineofsight_SDO_2 = []
            cubes_all_data_2 = []
            cubes_no_duplicate_init_2 = []
            cubes_no_duplicates_init_STEREO_2 = []
            cubes_no_duplicates_init_SDO_2 = []
            cubes_no_duplicate_new_2 = []
            cubes_no_duplicates_new_STEREO_2 = []
            cubes_no_duplicates_new_SDO_2 = []
            index = -1
            for number in self.cube_numbers_all:
                if number in self._cube_numbers_2:
                    index += 1
                    cubes_lineofsight_STEREO_2.append(self.cubes_lineofsight_STEREO_2[index] if self.line_of_sight else None)
                    cubes_lineofsight_SDO_2.append(self.cubes_lineofsight_SDO_2[index] if self.line_of_sight else None)
                    cubes_all_data_2.append(self.cubes_all_data_2[index] if self.all_data else None)
                    if self.cube_version_0:
                        cubes_no_duplicate_init_2.append(self.cubes_no_duplicate_init_2[index] if self.no_duplicate else None)
                        cubes_no_duplicates_init_STEREO_2.append(self.cubes_no_duplicates_init_STEREO_2[index] if self.duplicates else None)
                        cubes_no_duplicates_init_SDO_2.append(self.cubes_no_duplicates_init_SDO_2[index] if self.duplicates else None)
                    if self.cube_version_1:
                        cubes_no_duplicate_new_2.append(self.cubes_no_duplicate_new_2[index] if self.no_duplicate else None)
                        cubes_no_duplicates_new_STEREO_2.append(self.cubes_no_duplicates_new_STEREO_2[index] if self.duplicates else None)
                        cubes_no_duplicates_new_SDO_2.append(self.cubes_no_duplicates_new_SDO_2[index] if self.duplicates else None)
                else:
                    cubes_lineofsight_STEREO_2.append(None)
                    cubes_lineofsight_SDO_2.append(None)
                    cubes_all_data_2.append(None)
                    cubes_no_duplicate_init_2.append(None)
                    cubes_no_duplicates_init_STEREO_2.append(None)
                    cubes_no_duplicates_init_SDO_2.append(None)
                    cubes_no_duplicate_new_2.append(None)
                    cubes_no_duplicates_new_STEREO_2.append(None)
                    cubes_no_duplicates_new_SDO_2.append(None)
            self.cubes_lineofsight_STEREO_2 = cubes_lineofsight_STEREO_2
            self.cubes_lineofsight_SDO_2 = cubes_lineofsight_SDO_2
            self.cubes_all_data_2 = cubes_all_data_2
            self.cubes_no_duplicate_init_2 = cubes_no_duplicate_init_2
            self.cubes_no_duplicates_init_STEREO_2 = cubes_no_duplicates_init_STEREO_2
            self.cubes_no_duplicates_init_SDO_2 = cubes_no_duplicates_init_SDO_2
            self.cubes_no_duplicate_new_2 = cubes_no_duplicate_new_2
            self.cubes_no_duplicates_new_STEREO_2 = cubes_no_duplicates_new_STEREO_2
            self.cubes_no_duplicates_new_SDO_2 = cubes_no_duplicates_new_SDO_2

    def Sun_pos(self):
        """
        To find the Sun's radius and the center position in the cubes reference frame.
        """

        # Reference data 
        if self.first_cube:
            first_cube_name = os.path.join(self.paths['Cubes'], self._cube_names_1[0])
        else:
            first_cube_name = os.path.join(self.paths['Cubes_karine'], self._cube_names_2[0])

        # Initial data values
        solar_r = 6.96e5
        first_cube = readsav(first_cube_name)
        self._length_dx = first_cube.dx
        self._length_dy = first_cube.dy
        self._length_dz = first_cube.dz
        x_min = first_cube.xt_min
        y_min = first_cube.yt_min
        z_min = first_cube.zt_min

        # The Sun's radius
        self.radius_index = solar_r / self._length_dx  # TODO: need to change this if dx!=dy!=dz.

        # The Sun center's position
        x_index = x_min / self._length_dx 
        y_index = y_min / self._length_dy 
        z_index = z_min / self._length_dz 
        self.sun_center = np.array([0 - x_index, 0 - y_index, 0 - z_index])
        print(f'sun center coords are {self.sun_center}')

    def Sun_texture_old(self):
        """
        To create upload the Sun's texture and do the usual treatment so that the contrasts are more visible.
        A logarithmic intensity treatment is done with a saturation of really high an really low intensities.
        """

        # Importing AIA 33.5nm synoptics map
        hdul = fits.open(os.path.join(self.paths['Textures'], 'syn304_20120723.005959_20120723_010115_n5euA.fts_20120723_005845_n5euB.fts.fts'))
        image = hdul[0].data  # (960, 1920) monochromatic image

        # Image shape
        self._texture_height, self._texture_width = image.shape
        print(self._texture_height)
        print(self._texture_width)

        # Image treatment
        lower_cut = np.nanpercentile(image, 0.5)
        upper_cut = np.nanpercentile(image, 99.99)
        image[image < lower_cut] = lower_cut
        image[image > upper_cut] = upper_cut

        # Adding a white longitude latitude white grid each 15 degrees:
        nb_of_grids_lat = np.arange(1, 180 / self._heliographic_grid_degrees, 1)
        nb_of_grids_lon = np.arange(1, 360 / self._heliographic_grid_degrees + 1, 1)
        # grid_index_lat = self._heliographic_grid_degrees * nb_of_grids_lat / hdul[0].header['CDELT1']
        # grid_index_lon = self._heliographic_grid_degrees * nb_of_grids_lon / hdul[0].header['CDELT1']
        grid_index_lat = self._heliographic_grid_degrees * nb_of_grids_lat / 0.25
        grid_index_lon = self._heliographic_grid_degrees * nb_of_grids_lon / 0.25

        for lat in grid_index_lat: image[int(lat) - 1, :] = np.nanmax(image) * 1.0
        for lon in grid_index_lon: image[:, int(lon) - 1] = np.nanmax(image) * 1.0

        # Replacing nan values to the lower_cut 
        nw_image = np.where(np.isnan(image), lower_cut, image)  # TODO: would need to change the nan values to the interpolation for the pole
        nw_image = np.flip(nw_image, axis=0)

        # Changing values to a logarithmic scale
        self._sun_texture = np.log(nw_image)

    def Sun_texture(self):
        self._texture_height, self._texture_width = 720, 1440

        image = np.ones((self._texture_height, self._texture_width))


        # Adding a white longitude latitude white grid each 15 degrees:
        nb_of_grids_lat = np.arange(1, 180 / self._heliographic_grid_degrees, 1)
        nb_of_grids_lon = np.arange(1, 360 / self._heliographic_grid_degrees + 1, 1)

        grid_index_lat = self._heliographic_grid_degrees * nb_of_grids_lat / 0.25
        grid_index_lon = self._heliographic_grid_degrees * nb_of_grids_lon / 0.25

        for lat in grid_index_lat: image[int(lat) - 1, :] = 0
        for lon in grid_index_lon: image[:, int(lon) - 1] = 0
        nw_image = np.flip(image, axis=0)
        self._sun_texture = nw_image

    def Sun_points(self):
        """
        Creates a spherical cloud of points that represents the pixels on the Sun's surface.
        """

        # Initialisation
        N = self._sun_texture_resolution  # number of points in the theta direction
        phi = np.linspace(0, np.pi, N)  # latitude of the points
        theta = np.linspace(0, 2 * np.pi, 2 * N)  # longitude of the pointss
        phi, theta = np.meshgrid(phi, theta)  # the subsequent meshgrid

        # Conversion to cartesian coordinates
        x = self.radius_index * np.sin(phi) * np.cos(theta) + self.sun_center[0]
        y = self.radius_index * np.sin(phi) * np.sin(theta) + self.sun_center[1]
        z = self.radius_index * np.cos(phi) + self.sun_center[2] 

        # Creation of the position of the spherical cloud of points
        self.sun_points = np.array([x, y, z], dtype='float32').T

        # The corresponding image indexes to get the colors
        self._sun_texture_x = np.linspace(0, self._texture_height - 1, self.sun_points.shape[0], dtype='uint16')
        self._sun_texture_y = np.linspace(0, self._texture_width - 1, self.sun_points.shape[1], dtype='uint16')

    def Colours_1D(self):
        """
        Creates a 1D array of the integer Hex pixel values (0x000000 format) of a 2D sun texture image.
        """

        x_indices = self._sun_texture_x[:, np.newaxis]
        y_indices = self._sun_texture_y[np.newaxis, :]

        colours = self._sun_texture[x_indices, y_indices].flatten()
        normalized_colours = (colours - np.min(colours)) / (np.max(colours) - np.min(colours))
        blue_val = (normalized_colours * 255).astype('int')  # TODO: change to uint8, but might create problems if I remember correctly 
        self.hex_colours = (blue_val << 16) + (blue_val << 8) + blue_val
        self.hex_colours = self.hex_colours.astype('uint32')

    def STEREO_stats(self):
        """
        To save the information needed to find the position of STEREO.
        """

        data = readsav(os.path.join(self.paths['Main'], 'rainbow_stereob_304.save')).datainfos

        # Multiprocessing initial I/O bound tasks
        IO_processes = []
        manager = Manager()
        queue = manager.Queue()
        total_length = len(self.cube_numbers_all)
        step = int(np.ceil(total_length / self._batch_number))
        # Preping the processes
        for i in range(self._batch_number):
            if not (i==self._batch_number - 1):
                IO_processes.append(Process(target=self.STEREO_coords, args=(queue, i, data, self.cube_numbers_all[step * i:step * (i + 1)])))
            else:
                IO_processes.append(Process(target=self.STEREO_coords, args=(queue, i, data, self.cube_numbers_all[step * i:])))
        # Running the processes
        for p in IO_processes: p.start()
        for p in IO_processes: p.join()

        # Ordering the results
        results = [None for _ in range(self._batch_number)]
        while not queue.empty():
            identifier, result = queue.get()
            results[identifier] = result
        self.STEREO_pos = np.concatenate(results, axis=0) 

    def STEREO_coords(self, queue, i, data, numbers):
        """
        To get the position of STEREO given the fits filepath.
        Done like this as the computation is I/O bound and so the paths are separated in sections for multiprocessing.
        """

        from astropy import units as u
        from astropy.coordinates import CartesianRepresentation
        from sunpy.coordinates.frames import  HeliographicCarrington

        stereo_pos = []
        for number in numbers:
            stereo_lon = data[number].lon
            stereo_lat = data[number].lat
            stereo_dsun = data[number].dist
            stereo_date = data[number].strdate

            stereo_date = CustomDate(stereo_date)
            stereo_date = f'{stereo_date.year}-{stereo_date.month}-{stereo_date.day}T{stereo_date.hour}:{stereo_date.minute}:{stereo_date.second}'

            hec_coords = HeliographicCarrington(stereo_lon * u.deg, stereo_lat * u.deg, stereo_dsun * u.km,
                                                obstime=stereo_date, observer='self')
            hec_coords = hec_coords.represent_as(CartesianRepresentation)

            Xhec = hec_coords.x.value
            Yhec = hec_coords.y.value
            Zhec = hec_coords.z.value

            xpos_index = Xhec / self._length_dx
            ypos_index = Yhec / self._length_dy
            zpos_index = Zhec / self._length_dz

            stereo_pos.append(self.sun_center + np.array([xpos_index, ypos_index, zpos_index])) 
        queue.put((i, np.array(stereo_pos)))       
    
    def STEREO_pov_center(self):
        """
        To get the cartesian coordinates of the center of STEREO's field of view.        
        """

        stereo_data = readsav(os.path.join(self.paths['Main'], 'rainbow_stereob_304.save'))
        stereo_latcen = stereo_data.latcen
        stereo_loncen = stereo_data.loncen

        x = self.radius_index * np.sin(stereo_latcen / 180 * np.pi + np.pi / 2) * np.cos(stereo_loncen / 180 * np.pi) + self.sun_center[0]
        y = self.radius_index * np.sin(stereo_latcen / 180 * np.pi + np.pi / 2) * np.sin(stereo_loncen / 180 * np.pi) + self.sun_center[1]
        z = self.radius_index * np.cos(stereo_latcen / 180 * np.pi + np.pi / 2) + self.sun_center[2] 
        self.stereo_pov_center = np.array([x, y, z])

    def SDO_stats(self):
        """
        To save the information needed to find the position of SDO.
        """

        SDO_fits_names = [os.path.join(self.paths['SDO'], f'AIA_fullhead_{number:03d}.fits.gz')
                           for number in self.cube_numbers_all]

        # Multiprocessing initial I/O bound tasks
        IO_processes = []
        manager = Manager()
        queue = manager.Queue()
        total_length = len(SDO_fits_names)
        step = int(np.ceil(total_length / self._batch_number))
        # Preping the processes
        for i in range(self._batch_number):
            if not (i==self._batch_number - 1):
                IO_processes.append(Process(target=self.SDO_coords, args=(queue, i, SDO_fits_names[step * i:step * (i + 1)])))
            else:
                IO_processes.append(Process(target=self.SDO_coords, args=(queue, i, SDO_fits_names[step * i:])))
        # Running the processes
        for p in IO_processes: p.start()
        for p in IO_processes: p.join()

        # Ordering the results
        results = [None for _ in range(self._batch_number)]
        while not queue.empty():
            identifier, result = queue.get()
            results[identifier] = result
        self.SDO_pos = np.concatenate(results, axis=0)

    def SDO_coords(self, queue, i, paths):
        """
        To get the position of SDO given the fits filepath.
        Done like this as the computation is I/O bound and so the paths are separated in sections for multiprocessing.
        """

        from astropy import units as u
        from astropy.coordinates import CartesianRepresentation
        from sunpy.coordinates.frames import  HeliographicCarrington

        SDO_pos = []
        for fits_name in paths:
            header = fits.getheader(fits_name)

            hec_coords = HeliographicCarrington(header['CRLN_OBS']* u.deg, header['CRLT_OBS'] * u.deg, 
                                                header['DSUN_OBS'] * u.m, obstime=header['DATE-OBS'], 
                                                observer='self')
            hec_coords = hec_coords.represent_as(CartesianRepresentation)

            Xhec = hec_coords.x.value
            Yhec = hec_coords.y.value
            Zhec = hec_coords.z.value

            xpos_index = Xhec / (1000 * self._length_dx)
            ypos_index = Yhec / (1000 * self._length_dy)
            zpos_index = Zhec / (1000 * self._length_dz)

            SDO_pos.append(self.sun_center + np.array([xpos_index, ypos_index, zpos_index])) 
        queue.put((i, np.array(SDO_pos)))

    def Conv3d_results(self):
        """
        just testing the result gotten from the convolution
        """

        data = np.load(os.path.join('..', 'test_conv3d_array', 'conv3dRainbow.npy')).astype('uint8')

        binary_data = data > self.conv_treshold
        self.cubes_convolution = self.Sparse_data(binary_data)

        if self.skeleton:
            # Multiprocessing initial I/O bound tasks
            nb_of_batches = 6
            IO_processes = []
            manager = Manager()
            queue = manager.Queue()

            step = int(np.ceil(self.cubes_shape[0] / nb_of_batches))
            # Preping the processes
            for i in range(nb_of_batches):
                if not (i==nb_of_batches - 1):
                    IO_processes.append(Process(target=self.Skeleton_loop, args=(queue, i, binary_data[step * i:step * (i + 1)])))
                else:
                    IO_processes.append(Process(target=self.Skeleton_loop, args=(queue, i, binary_data[step * i:])))
            
            # Running the processes
            for p in IO_processes: p.start()
            for p in IO_processes: p.join()

            # Ordering the results
            results = [None for _ in range(nb_of_batches)]
            while not queue.empty():
                identifier, result = queue.get()
                results[identifier] = result
            cubes_barycenter = np.concatenate(results, axis=0)
            self.cubes_skeleton = self.Sparse_data(cubes_barycenter)

    def Skeleton_loop(self, queue, i, data):
        skeletons = []
        for cube in data:
            skeleton = skeletonize_3d(cube)
            skeletons.append(skeleton)
        skeletons = np.array(skeletons)
        queue.put((i, skeletons))

    def Attribute_deletion(self):
        """
        To delete some of the attributes that are not used in the inherited class. Done to save some RAM.
        """

        # Private attributes 
        del self._sun_texture_resolution, self._cube_names_all, self._cube_names_1, self._cube_names_2, self._cube_numbers_1, self._cube_numbers_2
        del self._texture_height, self._texture_width, self._sun_texture, self._sun_texture_x, self._sun_texture_y, self._batch_number
        del self._pattern_int, self._all_filenames, self._days_per_month, self._length_dy, self._length_dz


class K3dAnimation(Data):
    """
    Creates the corresponding k3d animation to then be used in a Jupyter notebook file.
    """

    @typechecked
    def __init__(self, compression_level: int = 9, plot_height: int = 1260, sleep_time: int | float = 2, 
                 camera_fov: int | float | str = 0.23, camera_zoom_speed: int | float = 0.7, trace_opacity: int | float = 0.1, 
                 screenshot_scale: int | float = 2, screenshot_sleep: int | float = 5,  screenshot_version: str = 'vtest', 
                 camera_pos: tuple[int | float, int | float, int | float] | None = None, up_vector: tuple[int, int, int] = (0, 0, 1), 
                 visible_grid: bool = False, **kwargs):
        
        super().__init__(**kwargs)

        # Arguments
        self.compression_level = compression_level  # the compression level of the data in the 3D visualisation
        self.plot_height = plot_height  # the height in pixels of the plot (initially it was 512)
        self.sleep_time = sleep_time  # sets the time between each frames (in seconds)
        self.camera_zoom_speed = camera_zoom_speed  # zoom speed of the camera 
        self.trace_opacity = trace_opacity  # opacity factor for all the trace voxels
        self.screenshot_scale = screenshot_scale  # the 'resolution' of the screenshot 
        self.screenshot_sleep = screenshot_sleep  # sleep time between each screenshot as synchronisation time is needed
        self.version = screenshot_version  # to save the screenshot with different names if multiple screenshots need to be saved
        self.camera_pos = camera_pos  # position of the camera multiplied by 1au
        self.up_vector = up_vector  # up vector for the camera
        self.visible_grid = visible_grid  # setting the grid to be visible or not
        
        if camera_fov=='sdo':
            self.camera_fov = self.Fov_for_SDO()
        elif camera_fov=='stereo':
            self.camera_fov = 0.26
        elif isinstance(camera_fov, (int, float)):
            self.camera_fov = camera_fov
        else:
            raise ValueError('When "camera_fov" a string, needs to be `sdo` or `stereo`.')

        # Instance attributes set when running the class
        self.plot = None  # k3d plot object
        self.plot_alldata_set1 = None  # voxels plot of the all data for set 1
        self.plot_alldata_set2 = None  # same for set 2 
        self.plot_dupli_STEREO_init_set1 = None  # voxels plot of the no duplicates seen from STEREO for set 1
        self.plot_dupli_STEREO_new_set1 = None  # voxels plot of the no duplicates seen from STEREO for set 1
        self.plot_dupli_STEREO_init_set2 = None  # same for set 2
        self.plot_dupli_STEREO_new_set2 = None  # same for set 2
        self.plot_dupli_SDO_init_set1 = None  # same seen from SDO for set 1
        self.plot_dupli_SDO_new_set1 = None  # same seen from SDO for set 1
        self.plot_dupli_SDO_init_set2 = None  # same for set 2
        self.plot_dupli_SDO_new_set2 = None  # same for set 2
        self.plot_dupli_init_set1 = None  # voxels plot of the no duplicate for set 1
        self.plot_dupli_new_set1 = None  # voxels plot of the no duplicate for set 1
        self.plot_dupli_init_set2 = None  # same for set 2
        self.plot_dupli_new_set2 = None  # same for set 2
        self.plot_los_STEREO_new_set1 = None  # voxels plot of the line of sight from STEREO for set 1 
        self.plot_los_STEREO_new_set1 = None  # voxels plot of the line of sight from STEREO for set 1 
        self.plot_los_STEREO_init_set2 = None  # same for set 2
        self.plot_los_STEREO_new_set2 = None  # same for set 2
        self.plot_los_SDO_init_set1 = None  # same seen from SDO for set 1
        self.plot_los_SDO_new_set1 = None  # same seen from SDO for set 1
        self.plot_los_SDO_init_set2 = None  # same for set 2
        self.plot_los_SDO_new_set2 = None  # same for set 2
        self.plot_interv_init_set1 = None  # voxels plot for the time integration of all data for set 1
        self.plot_interv_new_set1 = None  # voxels plot for the time integration of all data for set 1
        self.plot_interv_init_set2 = None  # same for set 2
        self.plot_interv_new_set2 = None  # same for set 2
        self.plot_interv_dupli_init_set1 = None  # same for the no duplicates data and set 1
        self.plot_interv_dupli_new_set1 = None  # same for the no duplicates data and set 1
        self.plot_interv_dupli_init_set2 = None  # same for set 2
        self.plot_interv_dupli_new_set2 = None  # same for set 2
        self.play_pause_button = None  # Play/Pause widget initialisation
        self.time_slider = None  # time slider widget
        self.date_dropdown = None  # Date dropdown widget to show the date
        self.time_link = None  # JavaScript Link between the two widgets
        self.date_text = None  # list of strings giving the text associated to each time frames 

        # Making the animation
        self.Update_paths()
        if self.time_intervals_all_data or self.time_intervals_no_duplicate: self.Time_interval_string()
        self.Date_strings()
        self.Animation()

    @typechecked
    @classmethod
    def The_usual(cls, version: int, data: str = 'no_duplicate', **classkwargs):
        """
        Gives the usual arguments used when making screenshots for a given point of view and data type.
        """

        if version==0:
            kwargs = {'sun': True, 'sdo_pov': True, 'fov_center':'stereo', 'camera_fov': 'sdo', 'up_vector': (0, 0, 1), 
                      'make_screenshots': True, 'screenshot_version': 'v0', 'screenshot_scale': 2, 
                      'sun_texture_resolution': 1920, 'both_cubes': 'kar'}
        elif version==1:
            kwargs = {'sun': True, 'stereo_pov': True, 'fov_center': 'stereo', 'camera_fov': 'stereo', 'up_vector': (0, 0, 1), 
                      'make_screenshots': True, 'screenshot_version': 'v1', 'screenshot_scale': 1, 
                      'sun_texture_resolution': 1920, 'both_cubes': 'kar'}        
        elif version==2:
            kwargs = {'sun': True, 'fov_center': 'cubes', 'camera_pos': (-0.7, 0.7, 0), 'up_vector': (0, 0, 1),
                      'make_screenshots': True, 'screenshot_version': 'v2', 'screenshot_scale': 1,
                      'sun_texture_resolution': 1920, 'both_cubes': 'kar'}
        elif version==3:
            kwargs = {'sun': True, 'fov_center': 'cubes', 'camera_pos': (0, 0, 1) , 'up_vector': (-1, 0, 0), 
                      'make_screenshots': True, 'screenshot_version': 'v3', 'screenshot_scale': 1, 
                      'sun_texture_resolution': 1920, 'both_cubes': 'kar'}     
        else:
            raise ValueError(f"The integer 'version' needs to have value going from 0 to 3, not {version}.")

        if 'intervals_no' in data:
            kwargs['time_intervals_no_duplicate'] = True
        elif 'intervals_all' in data:
            kwargs['time_intervals_all_data'] = True
        elif 'no_dupli' in data:
            kwargs['no_duplicate'] =True
        elif 'all' in data:
            kwargs['all_data'] = True
        else:
            raise ValueError(f"String '{data}' is not yet supported for argument 'data'.")
        combined_kwargs = {**kwargs, **classkwargs}  # so that the attributes can also be manually changed if needed be
        return cls(**combined_kwargs)
    
    def Update_paths(self):
        """
        Updating the paths of the parent class to be able to save screenshots.
        """

        if self.make_screenshots:
            self.paths['Screenshots'] = os.path.join(self.paths['Main'], 'texture_screenshots')
            os.makedirs(self.paths['Screenshots'], exist_ok=True)

    def Fov_for_SDO(self):
        """
        To get the same FOV than SDO when the fov_center parameter is the Sun.
        """

        hdul = fits.open(os.path.join(self.paths['SDO'], 'AIA_fullhead_000.fits.gz'))
        image_shape = np.array(hdul[0].data).shape
        Total_fov_in_degrees = image_shape[0] * hdul[0].header['CDELT1'] / 3600
        hdul.close()
        return Total_fov_in_degrees / 3

    def Full_array(self, sparse_cube):
        """
        To recreate a full 3D np.array from a sparse np.ndarray representing a 3D volume.
        If the initial value is None, returns an empty np.ndarray with the right shape.
        """

        if sparse_cube:
            cube = sparse_cube.todense()
            return cube.astype('uint8')
        else:
            return np.zeros((self.cubes_shape[1], self.cubes_shape[2], self.cubes_shape[3]), dtype='uint8')

    def Camera_params(self):
        """
        Camera visualisation parameters.
        """
 
        self.plot.camera_auto_fit = False
        self.plot.camera_fov = self.camera_fov  # FOV in degrees
        self.plot.camera_zoom_speed = self.camera_zoom_speed  # it was zooming too quickly (default=1.2)
        
        # Point to look at, i.e. initial rotational reference
        if isinstance(self.fov_center, tuple):
            self._camera_reference = self.fov_center
        elif self.fov_center:
            self._camera_reference = np.array([self.cubes_shape[3], self.cubes_shape[2], self.cubes_shape[1]]) / 2
        else:
            self._camera_reference = self.stereo_pov_center
        
        if self.stereo_pov:
            self.plot.camera = [self.STEREO_pos[0, 0], self.STEREO_pos[0, 1], self.STEREO_pos[0, 2],
                                self._camera_reference[0], self._camera_reference[1], self._camera_reference[2],
                                self.up_vector[0], self.up_vector[1], self.up_vector[2]] # up vector
        elif self.sdo_pov:
            self.plot.camera = [self.SDO_pos[0, 0], self.SDO_pos[0, 1], self.SDO_pos[0, 2],
                                self._camera_reference[0], self._camera_reference[1], self._camera_reference[2],
                                self.up_vector[0], self.up_vector[1], self.up_vector[2]]  # up vector
        else:
            au_in_solar_r = 215  # 1 au in solar radii
            distance_to_sun = au_in_solar_r * self.radius_index 

            if not self.camera_pos:
                print("no 'camera_pos', setting default values.")
                self.camera_pos = (-1, -0.5, 0)
            self.plot.camera = [self._camera_reference[0] + self.camera_pos[0] * distance_to_sun, 
                                self._camera_reference[1] + self.camera_pos[1] * distance_to_sun,
                                self._camera_reference[2] + self.camera_pos[2] * distance_to_sun,
                                self._camera_reference[0], self._camera_reference[1], self._camera_reference[2],
                                self.up_vector[0], self.up_vector[1], self.up_vector[2]]  # up vector

    def Update_voxel(self, change):
        """
        Updates the plots depending on which time frame you want to be shown. 
        Also creates the screenshots if it is set to True.
        """

        if self.stereo_pov:
            self.plot.camera = [self.STEREO_pos[change['new'], 0], self.STEREO_pos[change['new'], 1], self.STEREO_pos[change['new'], 2],
                                self._camera_reference[0], self._camera_reference[1], self._camera_reference[2],
                                0, 0, 1]
            sleep(0.2) 
        elif self.sdo_pov:
            self.plot.camera = [self.SDO_pos[change['new'], 0], self.SDO_pos[change['new'], 1], self.SDO_pos[change['new'], 2],
                                self._camera_reference[0], self._camera_reference[1], self._camera_reference[2],
                                0, 0, 1]
            sleep(0.2)
        
        if self.polynomials:
            for index, plot in enumerate(self.plots_polynomials):
                _, data = self.polynomials_matchesNdata[index]
                plot.voxels = self.Full_array(data[change['new']]).T

        if self.all_data:
            if self.first_cube: self.plot_alldata_set1.voxels = self.Full_array(self.cubes_all_data_1[change['new']])
            if self.second_cube: self.plot_alldata_set2.voxels = self.Full_array(self.cubes_all_data_2[change['new']])
        if self.duplicates:
            if self.first_cube:
                if self.cube_version_0:
                    self.plot_dupli_STEREO_init_set1.voxels = self.Full_array(self.cubes_no_duplicates_init_STEREO_1[change['new']])
                    self.plot_dupli_SDO_init_set1.voxels = self.Full_array(self.cubes_no_duplicates_init_SDO_1[change['new']])
                if self.cube_version_1:
                    self.plot_dupli_STEREO_new_set1.voxels = self.Full_array(self.cubes_no_duplicates_new_STEREO_1[change['new']])
                    self.plot_dupli_SDO_new_set1.voxels = self.Full_array(self.cubes_no_duplicates_new_SDO_1[change['new']])
            if self.second_cube:
                if self.cube_version_0:
                    self.plot_dupli_STEREO_init_set2.voxels = self.Full_array(self.cubes_no_duplicates_init_STEREO_2[change['new']])
                    self.plot_dupli_init_SDO_set2.voxels = self.Full_array(self.cubes_no_duplicates_init_SDO_2[change['new']])
                     
                if self.cube_version_1:
                    self.plot_dupli_STEREO_new_set2.voxels = self.Full_array(self.cubes_no_duplicates_new_STEREO_2[change['new']])
                    self.plot_dupli_new_SDO_set2.voxels = self.Full_array(self.cubes_no_duplicates_new_SDO_2[change['new']])
        if self.no_duplicate:
            if self.first_cube:
                if self.cube_version_0: self.plot_dupli_init_set1.voxels = self.Full_array(self.cubes_no_duplicate_init_1[change['new']])
                if self.cube_version_1: self.plot_dupli_new_set1.voxels = self.Full_array(self.cubes_no_duplicate_new_1[change['new']]) 
            if self.second_cube:
                if self.cube_version_0: self.plot_dupli_init_set2.voxels = self.Full_array(self.cubes_no_duplicate_init_2[change['new']]) 
                if self.cube_version_1: self.plot_dupli_new_set2.voxels = self.Full_array(self.cubes_no_duplicate_new_2[change['new']]).T

        if self.line_of_sight:
            if self.first_cube:    
                self.plot_los_STEREO_set1.voxels = self.Full_array(self.cubes_lineofsight_STEREO_1[change['new']])
                self.plot_los_SDO_set1.voxels = self.Full_array(self.cubes_lineofsight_SDO_1[change['new']])
            if self.second_cube:
                self.plot_los_STEREO_set2.voxels = self.Full_array(self.cubes_lineofsight_STEREO_2[change['new']])
                self.plot_los_SDO_set2.voxels = self.Full_array(self.cubes_lineofsight_SDO_2[change['new']])

        if self.time_intervals_all_data:
            if self.first_cube: self.plot_interv_set1.voxels = self.Full_array(self.time_cubes_all_data_1[change['new']])
                
            if self.second_cube:
                if self.cube_version_0: self.plot_interv_init_set2.voxels = self.Full_array(self.time_cubes_all_data_init_2[change['new']]) 
                if self.cube_version_1: self.plot_interv_new_set2.voxels = self.Full_array(self.time_cubes_all_data_new_2[change['new']])
                          
        if self.time_intervals_no_duplicate:
            if self.first_cube:
                if self.cube_version_0: self.plot_interv_dupli_init_set1.voxels = self.Full_array(self.time_cubes_no_duplicate_init_1[change['new']])
                if self.cube_version_1: self.plot_interv_dupli_new_set1.voxels = self.Full_array(self.time_cubes_no_duplicate_new_1[change['new']])
            if self.second_cube:
                if self.cube_version_0: self.plot_interv_dupli_init_set2.voxels = self.Full_array(self.time_cubes_no_duplicate_init_2[change['new']])
                if self.cube_version_1: self.plot_interv_dupli_new_set2.voxels = self.Full_array(self.time_cubes_no_duplicate_new_2[change['new']]).T
       
        if self.skeleton: self.plot_skeleton.voxels = self.Full_array(self.cubes_skeleton[change['new']])
        if self.convolution: self.plot_convolution.voxels = self.Full_array(self.cubes_convolution[change['new']])
        if self.make_screenshots: self.Screenshot_making()

        if self.html_snapshot:
            sleep(4)
            with open(f"snapshot_date{self.date_text[change['new']]}.html", "w") as f:
                f.write(self.plot.get_snapshot())

    def Play(self):
        """
        Params for the play button.
        """
        
        if self.play_pause_button.value and self.time_slider.value < len(self.cube_numbers_all) - 1:
            self.time_slider.value += 1
            threading.Timer(self.sleep_time, self.Play).start()  # where you also set the sleep() time.
                
        else:
            self.play_pause_button.description = 'Play'
            self.play_pause_button.icon = 'play'

    def Screenshot_making(self):
        """
        To create a screenshot of the plot. A sleep time was added as the screenshot method relies
        on asynchronus traitlets mechanism.
        """

        import base64

        self.plot.fetch_screenshot()
        sleep(self.screenshot_sleep)

        screenshot_png = base64.b64decode(self.plot.screenshot)
        if self.time_intervals_no_duplicate:
            screenshot_name = f'nodupli_interval{self.time_interval}_{self.date_text[self.time_slider.value]}_{self.version}.png'
        elif self.time_intervals_all_data:
            screenshot_name = f'alldata_interval{self.time_interval}_{self.date_text[self.time_slider.value]}_{self.version}.png'
        elif self.no_duplicate:
            screenshot_name = f'nodupli_{self.time_slider.value:03d}_{self.date_text[self.time_slider.value]}_{self.version}.png'
        elif self.all_data:
            screenshot_name = f'alldata_{self.time_slider.value:03d}_{self.date_text[self.time_slider.value]}_{self.version}.png'
        else:
            raise ValueError("The screenshot name for that type of data still hasn't been created.")
        
        screenshot_namewpath = os.path.join(self.paths['Screenshots'], screenshot_name)
        with open(screenshot_namewpath, 'wb') as f:
            f.write(screenshot_png)

    def Play_pause_handler(self, change):
        """
        Changes the play button to pause when it is clicked.
        """

        if change['new']:  # if clicked play
            self.Play()
            self.play_pause_button.description = 'Pause'
            self.play_pause_button.icon = 'pause'
        else:  
            pass

    def Date_strings(self):
        """
        Uses the dates for the files to create a corresponding string list.
        """

        self.date_text = [f'{date.year}-{date.month:02d}-{date.day:02d}_{date.hour:02d}h{date.minute:02d}min'
                          for date in self.dates_all]

    def Time_interval_string(self):
        """
        To change self.time_interval to a string giving a value in day, hours or minutes.
        """

        if self.time_interval < 60:
            self.time_interval = f'{self.time_interval}s'
        elif self.time_interval < 3600:
            time_interval = f'{self.time_interval // 60}min'
            if self.time_interval % 60 != 0:
                self.time_interval = time_interval + f'{self.time_interval % 60}s'
            else:
                self.time_interval = time_interval
        elif self.time_interval < 3600 * 24:
            time_interval = f'{self.time_interval // 3600}h'
            if self.time_interval % 3600 != 0:
                self.time_interval = time_interval + f'{(self.time_interval % 3600) // 60}min'
            else: 
                self.time_interval = time_interval
        elif self.time_interval < 3600 * 24 * 3.1:
            time_interval = f'{self.time_interval // (3600 * 24)}days'
            if self.time_interval % (3600 * 24) != 0:
                self.time_interval = time_interval + f'{(self.time_interval % 3600 * 24) // 3600}h'
            else:
                self.time_interval = time_interval
        else:
            raise ValueError("Time interval is way too large")
    
    def Random_hexadecimal_color_generator(self):
        """
        Generator that yields a color value in integer hexadecimal code format.
        """

        while True:
            yield np.random.randint(0, 0xffffff)

    def Animation(self):
        """
        Creates the 3D animation using k3d. 
        """
        
        # Initialisation of the plot
        self.plot = k3d.plot(grid_visible=self.visible_grid)  # plot with no axes. If a dark background is needed then background_color=0x000000
        self.plot.height = self.plot_height  
            
        # Adding the camera specific parameters
        self.Camera_params()
        
        if self.polynomials:
            self.plots_polynomials = []
            for (pattern, data) in self.polynomials_matchesNdata:
                data = self.Full_array(data[0])
                limit_val = f"_lim{pattern.group('conv_limit')}" if pattern.group('conv_limit') else None
                plot = k3d.voxels(data.T, opacity=0.95, compression_level=self.compression_level, color_map=[next(self.Random_hexadecimal_color_generator())], 
                                  name=f"fit_order{pattern.group('order')}_data: {limit_val if limit_val else ''}_{pattern.group('datatype')}")
                self.plots_polynomials.append(plot)
                self.plot += plot

        # Adding the SUN!!!
        if self.sun:
            self.plot += k3d.points(positions=self.sun_points, point_size=3.5, colors=self.hex_colours, shader='flat',
                                    name='SUN', compression_level=self.compression_level)
        
        # Just for testing some stuff
        # self.plot += k3d.points(positions=np.array([0,0,0]), name='000', opacity=1, compression_level=self.compression_level, point_size=5)
        # self.plot += k3d.points(positions=self.sun_center, name='suncenter', opacity=1, compression_level=self.compression_level, point_size=5)
        # self.plot += k3d.points(positions=np.array([0, 0, 10]), name='001', opacity=1, compression_level=self.compression_level, point_size=5)
        # self.plot += k3d.points(positions=np.array([10, 0, 0]), name='100', opacity=1, compression_level=self.compression_level, point_size=5)
        # self.plot += k3d.points(positions=np.array([0, 10, 0]), name='010', opacity=1, compression_level=self.compression_level, point_size=5)
        # data = np.zeros((30, 30, 30))
        # data[20, 0, 0] = 1
        # self.plot += k3d.voxels(data.T, opacity=1, compression_level=self.compression_level, name='voxelinxdirec')

        # Adding the different data sets (i.e. with or without duplicates)
        if self.all_data:  #old color color_map=[0x90ee90]
            if self.first_cube:
                data = self.Full_array(self.cubes_all_data_1[0])
                self.plot_alldata_set1 = k3d.voxels(data, outlines=False, opacity=0.5, compression_level=self.compression_level,
                                                     color_map=[0x0000ff], name='Set1: all data')
                self.plot += self.plot_alldata_set1
            if self.second_cube:
                data = self.Full_array(self.cubes_all_data_2[0])
                self.plot_alldata_set2 = k3d.voxels(data, outlines=False, opacity=0.3, compression_level=self.compression_level,
                                                     color_map=[0xff6e00], name='Set2: all data')
                self.plot += self.plot_alldata_set2           
       
        if self.duplicates:
            if self.first_cube:
                if self.cube_version_0:
                    data = self.Full_array(self.cubes_no_duplicates_init_STEREO_1[0])
                    self.plot_dupli_init_STEREO_set1 = k3d.voxels(data, compression_level=self.compression_level, outlines=True, 
                                            color_map=[0xff0000], name='aSet1: no duplicates from SDO')
                    data = self.Full_array(self.cubes_no_duplicates_init_SDO_1[0])
                    self.plot_dupli_init_SDO_set1 = k3d.voxels(data, compression_level=self.compression_level, outlines=True, 
                                            color_map=[0x0000ff], name='aSet1: no duplicates from STEREO')
                    self.plot += self.plot_dupli_init_STEREO_set1
                    self.plot += self.plot_dupli_init_SDO_set1
                if self.cube_version_1:
                    data = self.Full_array(self.cubes_no_duplicates_new_STEREO_1[0])
                    self.plot_dupli_new_STEREO_set1 = k3d.voxels(data, compression_level=self.compression_level, outlines=True, 
                                            color_map=[0xff0000], name='Set1: no duplicates from SDO')
                    data = self.Full_array(self.cubes_no_duplicates_new_SDO_1[0])
                    self.plot_dupli_new_SDO_set1 = k3d.voxels(data, compression_level=self.compression_level, outlines=True, 
                                            color_map=[0x0000ff], name='Set1: no duplicates from STEREO')
                    self.plot += self.plot_dupli_new_STEREO_set1
                    self.plot += self.plot_dupli_new_SDO_set1
            if self.second_cube:
                if self.cube_version_0:
                    data = self.Full_array(self.cubes_no_duplicates_init_STEREO_2[0])
                    self.plot_dupli_init_STEREO_set2 = k3d.voxels(data, compression_level=self.compression_level, outlines=True, 
                                            color_map=[0xff0000], name='aSet2: no duplicates from SDO')
                    data = self.Full_array(self.cubes_no_duplicates_init_SDO_2[0])
                    self.plot_dupli_init_SDO_set2 = k3d.voxels(data, compression_level=self.compression_level, outlines=True, 
                                            color_map=[0x0000ff], name='aSet2: no duplicates from STEREO')
                    self.plot += self.plot_dupli_init_STEREO_set2
                    self.plot += self.plot_dupli_init_SDO_set2
                if self.cube_version_1:
                    data = self.Full_array(self.cubes_no_duplicates_new_STEREO_2[0])
                    self.plot_dupli_new_STEREO_set2 = k3d.voxels(data, compression_level=self.compression_level, outlines=True, 
                                            color_map=[0xff0000], name='Set2: no duplicates from SDO')
                    data = self.Full_array(self.cubes_no_duplicates_new_SDO_2[0])
                    self.plot_dupli_new_SDO_set2 = k3d.voxels(data, compression_level=self.compression_level, outlines=True, 
                                            color_map=[0x0000ff], name='Set2: no duplicates from STEREO')
                    self.plot += self.plot_dupli_new_STEREO_set2
                    self.plot += self.plot_dupli_new_SDO_set2
               
        if self.no_duplicate:
            if self.first_cube:
                if self.cube_version_0:
                    data = self.Full_array(self.cubes_no_duplicate_init_1[0])
                    self.plot_dupli_init_set1 = k3d.voxels(data, compression_level=self.compression_level, outlines=True, 
                                            color_map=[0x0000ff], opacity=0.3, name='aSet1: no duplicates')
                    self.plot += self.plot_dupli_init_set1
                if self.cube_version_1:
                    data = self.Full_array(self.cubes_no_duplicate_new_1[0])
                    self.plot_dupli_new_set1 = k3d.voxels(data, compression_level=self.compression_level, outlines=True, 
                                            color_map=[0x0000ff], opacity=0.3, name='Set1: no duplicates')
                    self.plot += self.plot_dupli_new_set1
            if self.second_cube:
                if self.cube_version_0:
                    data = self.Full_array(self.cubes_no_duplicate_init_2[0])
                    self.plot_dupli_init_set2 = k3d.voxels(data, compression_level=self.compression_level, outlines=True, 
                                            color_map=[0xff0000], opacity=0.7, name='aSet2: no duplicates')
                    self.plot += self.plot_dupli_init_set2
                if self.cube_version_1:
                    data = self.Full_array(self.cubes_no_duplicate_new_2[0])
                    self.plot_dupli_new_set2 = k3d.voxels(data.T, compression_level=self.compression_level, outlines=True, 
                                            color_map=[0x0000ff], opacity=0.2, name='Set2: no duplicates')
                    self.plot += self.plot_dupli_new_set2

        if self.line_of_sight:
            if self.first_cube:
                data = self.Full_array(self.cubes_lineofsight_STEREO_1[0])
                self.plot_los_STEREO_set1 = k3d.voxels(data, compression_level=self.compression_level, outlines=True, 
                                        color_map=[0x0000ff], name='aSet1: seen from Stereo')
                data = self.Full_array(self.cubes_lineofsight_SDO_1[0])
                self.plot_los_SDO_set1 = k3d.voxels(data, compression_level=self.compression_level, outlines=True, 
                                        color_map=[0xff0000], name='aSet1: seen from SDO')
                self.plot += self.plot_los_STEREO_set1
                self.plot += self.plot_los_SDO_set1

            if self.second_cube:       
                data = self.Full_array(self.cubes_lineofsight_STEREO_2[0])
                self.plot_los_STEREO_set2 = k3d.voxels(data, compression_level=self.compression_level, outlines=True, 
                                        color_map=[0x0000ff], name='aSet2: seen from Stereo')
                data = self.Full_array(self.cubes_lineofsight_SDO_2[0])
                self.plot_los_SDO_set2 = k3d.voxels(data, compression_level=self.compression_level, outlines=True, 
                                        color_map=[0xff0000], name='aSet2: seen from SDO')
                self.plot += self.plot_los_STEREO_set2
                self.plot += self.plot_los_SDO_set2

        if self.time_intervals_all_data:
            if self.first_cube:
                data = self.Full_array(self.time_cubes_all_data_1[0])
                self.plot_interv_set1 = k3d.voxels(data, compression_level=self.compression_level, outlines=False, 
                                        color_map=[0xff6666],opacity=1, name=f'Set1: all data for {self.time_interval}')
                self.plot += self.plot_interv_set1
            if self.second_cube:
                if self.cube_version_0:
                    data = self.Full_array(self.time_cubes_all_data_init_2[0])
                    self.plot_interv_init_set2 = k3d.voxels(data, compression_level=self.compression_level, outlines=False, 
                                            color_map=[0xff6666],opacity=1, name=f'Set2: all data for {self.time_interval}')
                    self.plot += self.plot_interv_init_set2    
                if self.cube_version_1:
                    data = self.Full_array(self.time_cubes_all_data_new_2[0])
                    self.plot_interv_new_set2 = k3d.voxels(data, compression_level=self.compression_level, outlines=False, 
                                            color_map=[0xff6666],opacity=1, name=f'Set2: all data for {self.time_interval}')
                    self.plot += self.plot_interv_new_set2         
       
        if self.time_intervals_no_duplicate:
            if self.first_cube:
                if self.cube_version_0:
                    data = self.Full_array(self.time_cubes_no_duplicate_init_1[0])
                    self.plot_interv_dupli_init_set1 = k3d.voxels(data, compression_level=self.compression_level, outlines=False,
                                            color_map=[0x0000ff], opacity=0.35, name=f'aSet1: no duplicate for {self.time_interval}')
                    self.plot += self.plot_interv_dupli_init_set1
                if self.cube_version_1:
                    data = self.Full_array(self.time_cubes_no_duplicate_new_1[0])
                    self.plot_interv_dupli_new_set1 = k3d.voxels(data, compression_level=self.compression_level, outlines=False,
                                            color_map=[0x0000ff], opacity=0.35, name=f'Set1: no duplicate for {self.time_interval}')
                    self.plot += self.plot_interv_dupli_new_set1
            if self.second_cube:
                if self.cube_version_0:
                    data = self.Full_array(self.time_cubes_no_duplicate_init_2[0])
                    self.plot_interv_dupli_init_set2 = k3d.voxels(data, compression_level=self.compression_level, outlines=True,
                                            color_map=[0x0000ff], opacity=0.35, name=f'aSet2: no duplicate for {self.time_interval}')
                    self.plot += self.plot_interv_dupli_init_set2          
                if self.cube_version_1:
                    data = self.Full_array(self.time_cubes_no_duplicate_new_2[0])
                    self.plot_interv_dupli_new_set2 = k3d.voxels(data.T, compression_level=self.compression_level, outlines=True,
                                            color_map=[0xff0000], opacity=0.35, name=f'Set2: no duplicate for {self.time_interval}')
                    self.plot += self.plot_interv_dupli_new_set2           

        if self.trace_data:
            if self.first_cube:
                data = self.Full_array(self.trace_cubes_1)
                self.plot += k3d.voxels(data, compression_level=self.compression_level, outlines=False, color_map=[0xff6666],
                                    opacity=self.trace_opacity, name='Set1: total trace')
            if self.second_cube:
                data = self.Full_array(self.trace_cubes_2)
                self.plot += k3d.voxels(data, compression_level=self.compression_level, outlines=False, color_map=[0xff6666],
                                    opacity=self.trace_opacity, name='Set2: total trace')
        
        if self.trace_no_duplicate:
            if self.first_cube:
                data = self.Full_array(self.trace_cubes_no_duplicate_1)
                self.plot += k3d.voxels(data, compression_level=self.compression_level, outlines=False, 
                                    color_map=[0xff6666], opacity=self.trace_opacity, name='Set1: no duplicates trace')
            if self.second_cube:
                data = self.Full_array(self.trace_cubes_no_duplicate_2)
                self.plot += k3d.voxels(data, compression_level=self.compression_level, outlines=False, 
                                    color_map=[0xff6666], opacity=self.trace_opacity, name='Set2: no duplicates trace')
    
        if self.skeleton:
            data = self.Full_array(self.cubes_skeleton[0])
            self.plot_skeleton = k3d.voxels(data, compression_level=self.compression_level, outlines=False, color_map=[0xff6e00], opacity=1,
                                              name='barycenter for new no dupliactes' )
            self.plot += self.plot_skeleton

        if self.convolution:
            data = self.Full_array(self.cubes_convolution[0])
            self.plot_convolution = k3d.voxels(data, compression_level=self.compression_level, outlines=False, color_map=[0xff6e00], opacity=0.5,
                                        name='conv3d')
            self.plot += self.plot_convolution

        # Adding a play/pause button
        self.play_pause_button = widgets.ToggleButton(value=False, description='Play', icon='play')

        # Set up the time slider and the play/pause button
        self.time_slider = widgets.IntSlider(min=0, max=len(self.cube_numbers_all)-1, description='Frame:')
        self.date_dropdown = widgets.Dropdown(options=self.date_text, description='Date:')
        self.time_slider.observe(self.Update_voxel, names='value')
        self.time_link= widgets.jslink((self.time_slider, 'value'), (self.date_dropdown, 'index'))
        self.play_pause_button.observe(self.Play_pause_handler, names='value')

        # Display
        display(self.plot, self.time_slider, self.date_dropdown, self.play_pause_button)
        if self.make_screenshots:
            self.plot.screenshot_scale = self.screenshot_scale
            self.Screenshot_making()


class VoxelVelocities(Data):
    """
    To try and calculate the plasma velocities by measuring the relative speed of voxel 'regions' (not really regions as actual regions vary to much in size).
    What needs to be taken into account:
        - only for images that are right next to each other.
        - need to decide if I take the center of the region to be the border of said region or the center.
        - need to decide a maximum velocity for which the regions are clearly not the same.
        - need only positive (or negative depending on the reference) velovity values.
        - need to set a minimum size for a given region.
        - need to choose a small set as the whole thing will most likely give a lot of false values.
        - need to test with averaging when I have something that works.
        - maybe also test time integrations. 


    possible image numbers: 
        - 77 to 81  (23T04-06 to 23T04-36) good
        - 96 to 98 (23T07-16 to 23T07-36) meh
        - 272 to 276 (24T12-36 to 24T13-16) really good
        - 310 to 317 (24T18-56 to 24T20-06) really good
        - 353 to 357 (25T02-06 to 25T02-46) good if you use all_data
    """

    def __init__(self, **kwargs): 
        # Default arguments
        default_kwargs = {
            'cube_version': 'new',
            'time_intervals_no_duplicate': True,
        }

        default_kwargs.update(kwargs)
        super().__init__(**default_kwargs)

    def Regions_preprocessing(self):
        """
        Function to get the baricenter of the time cunks
        """
        pass

# Basic tests
if __name__ == '__main__':
    # K3dAnimation(everything=True, both_cubes='alf')
    pass