"""
Imports of the data, preprocessing and creation of the 3D k3d visualisation class to then be used in a jupyter notebook.
"""

# Imports
from __future__ import annotations
import os
import re
import k3d

import numpy as np
import ipywidgets as widgets

from glob import glob
from time import sleep
from threading import Timer
from astropy.io import fits
from scipy.io import readsav
from astropy import units as u
from typeguard import typechecked
from IPython.display import display
from sparse import COO, stack, concatenate
from skimage.morphology import skeletonize_3d
from multiprocessing import Process, Manager
from multiprocessing.queues import Queue as QUEUE
from multiprocessing.shared_memory import SharedMemory
from astropy.coordinates import CartesianRepresentation
from sunpy.coordinates.frames import  HeliographicCarrington

from .Common import Decorators, MultiProcessing

class CustomDate:
    """
    To separate the year, month, day, hour, minute, second if a string dateutil.parser.parser doesn't work. 
    """

    @typechecked
    def __init__(self, date_str: str | bytes):
        self.year: int
        self.month: int
        self.day: int
        self.hour: int
        self.minute: int
        self.second: int

        if isinstance(date_str, str):
            self.parse_date_str(date_str=date_str)
        elif isinstance(date_str, bytes):
            self.parse_date_bytes(date_str=date_str)

    def parse_date_str(self, date_str: str) -> None:
        """
        Separating a string in the format YYYY-MM-DDThh-mm-ss to get the different time attributes.
        """

        date_part, time_part = date_str.split("T")
        self.year, self.month, self.day = map(int, date_part.split("-"))
        self.hour, self.minute, self.second = map(int, time_part.split("-"))
    
    def parse_date_bytes(self, date_str: bytes) -> None:
        """
        Separating a bytestring in the format YYYY/MM/DD hh:mm:ss to get the different date attributes.
        """

        date_part, time_part = date_str.split(b' ')
        self.year, self.month, self.day = map(int, date_part.split(b"/"))
        self.hour, self.minute, self.second = map(int, time_part.split(b':'))


# @ClassDecorator(Decorators.running_time)
class Data:
    """
    To upload and manipulate the data to then be inputted in the k3d library for 3D animations.
    """

    @Decorators.running_time  
    @typechecked  
    def __init__(self, everything: bool = False, sun: bool = False, 
                 all_data: bool = False, duplicates: bool = False, no_duplicates: bool = False, line_of_sight: bool = False, 
                 trace_data: bool = False, trace_no_duplicates: bool = False, time_intervals_all_data: bool = False, 
                 time_intervals_no_duplicates: bool = False, time_interval: str | int = 1, heliographic_grid_degrees: int | float = 15, 
                 fov_center: tuple[int | float, int | float, int | float] | str = 'cubes', sun_texture_resolution: int = 960,
                 sdo_pov: bool = False, stereo_pov: bool = False, processes: int = 10, make_screenshots: bool = False, cube_version: str = 'old',
                 convolution_3d: bool = False, conv_treshold: int = 125, polynomials: bool = False, skeleton: bool = False, 
                 html_snapshot: bool = False):
        
        # Arguments
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
        self.no_duplicates = (no_duplicates or everything) # for the no duplicates data
        self.line_of_sight = (line_of_sight or everything)  # for the line of sight data 
        self.trace_data = (trace_data or everything) # for the trace of all_data
        self.trace_no_duplicates = (trace_no_duplicates or everything)  # same for no_duplicate
        self.time_intervals_all_data = (time_intervals_all_data or everything)  # for the data integration over time for all_data
        self.time_intervals_no_duplicates = (time_intervals_no_duplicates or everything)  # same for no_duplicate
        self.time_interval = time_interval  # time interval in hours (if 'int' or 'h' in str), days (if 'd' in str), minutes (if 'min' in str)  
        self._sun_texture_resolution = sun_texture_resolution  # choosing the Sun's texture resolution
        self.stereo_pov = stereo_pov  # following the STEREO pov
        self.sdo_pov = sdo_pov  # same for the SDO pov 
        self._processes = processes  # number of processes
        self.make_screenshots = make_screenshots  # creating screenshots when clicking play
        self._heliographic_grid_degrees = heliographic_grid_degrees  # heliographic grid steps in degrees
        self.convolution = convolution_3d
        self.conv_treshold = conv_treshold
        self.polynomials = polynomials
        self.skeleton = skeleton
        self.html_snapshot = html_snapshot

        # Instance attributes set when running the class
        self.paths: dict[str, str]  # contains all the path names and the corresponding path
        self._cube_names: list[str] # all the cube names that will be used
        self.cube_numbers_all: list[int]  # all the cube numbers or numbers from 0 to 412 if make_screenshots=True
        self._cube_numbers: list[int]  # same but only the existing cube numbers
        self.dates: list[CustomDate] # list of the date with .year, .month, etc for all the cubes
        self.cubes_shape: tuple[int]  # the shape of the cubes
        self.cubes_all_data: list[COO | None] | COO # sparse boolean 4D array of all the data 
        self.cubes_lineofsight_STEREO: list[COO | None] | COO  # sparse boolean 4D array of the line of sight data seen from STEREO 
        self.cubes_lineofsight_SDO: list[COO | None] | COO  # same for SDO 
        self.cubes_no_duplicates_init: list[COO | None] | COO  # sparse boolean 4D array for the first method no duplicates cubes 
        self.cubes_no_duplicates_new: list[COO | None] | COO # same for the new method
        self.cubes_no_duplicates_STEREO_init: list[COO | None] | COO  # same seen from STEREO for the first method
        self.cubes_no_duplicates_STEREO_new: list[COO | None] | COO  # same for the new method
        self.cubes_no_duplicates_SDO_init: list[COO | None] | COO  # same seen from SDO  for the first method
        self.cubes_no_duplicates_SDO_init: list[COO | None] | COO # same for the new method
        self.trace_cubes: COO  # sparse boolean 3D array of all the data  
        self.trace_cubes_no_duplicates_init: COO  # same  for the no duplicate data first method
        self.trace_cubes_no_duplicates_new: COO  # same  for the new method
        self.time_cubes_all_data: list[COO | None] | COO  # sparse boolean 4D array of the integration of all_data over time_interval 
        self.time_cubes_no_duplicates: list[COO | None] | COO  # same for the no duplicate 
        self.radius_index: float  # radius of the Sun in grid units
        self.sun_center: np.ndarray # position of the Sun's center [x, y, z] in the grid
        self.sun_points: np.ndarray  # positions of the pixels for the Sun's texture
        self._days_per_month: list[int]  # number of days per each month in a year
        self._length_dx: int  # x direction unit voxel size in km
        self._length_dy: int  # same for the y direction
        self._length_dz: int # same for the z direction
        self.STEREO_pos: np.ndarray  # gives the position of the STEREO satellite for each time step
        self.SDO_pos: np.ndarray # same for SDO

        # Functions
        self.paths_creation()
        self.names()
        self.sun_pos()
        self.dates_data()            
        self.choices()

        # Deleting the private class attributes
        self.attribute_deletion()

    def paths_creation(self) -> None:
        """Input and output paths manager.
        """

        main_path = '../'
        self.paths = {
            'Main': main_path,
            'Cubes_karine': os.path.join(main_path, 'Cubes_karine'),
            'Textures': os.path.join(main_path, 'Textures'),
            'Intensities': os.path.join(main_path, 'STEREO', 'int'),
            'SDO': os.path.join(main_path, 'sdo'),
            'polynomials': os.path.join(main_path, 'curveFitArrays'),
            'screenshots': os.path.join(main_path, 'screenshots'),
            }    
    
    def choices(self) -> None:
        """
        To choose what is computed and added depending on the arguments chosen.
        """

        if self.time_intervals_all_data or self.time_intervals_no_duplicates: self.Time_interval()

        cubes_all_data, cubes_no_duplicates_init, cubes_no_duplicates_STEREO_init, cubes_no_duplicates_SDO_init, \
        cubes_no_duplicates_new, cubes_no_duplicates_STEREO_new, cubes_no_duplicates_SDO_new, trace_cubes, \
        trace_cubes_no_duplicates_init, trace_cubes_no_duplicates_new, time_cubes_all_data, time_cubes_no_duplicates_init, \
        time_cubes_no_duplicates_new, cubes_lineofsight_STEREO, cubes_lineofsight_SDO = self.Processing_data()

        if self.sun:
            sun_texture, dimensions = self.Sun_texture()
            sun_texture_x, sun_texture_y = self.Sun_points(dimensions)
            self.Colours_1D(sun_texture, sun_texture_x, sun_texture_y)
                         
        if self.stereo_pov:
            self.STEREO_stats()
        elif self.sdo_pov:
            self.SDO_stats()

        if not self.fov_center: self.STEREO_pov_center()

        cubes_convolution, cubes_skeleton = self.Conv3d_results() if self.convolution else None, None

        if self.polynomials: self.Preprocessing_polynomial_data()

        self.cubes_all_data, self.cubes_no_duplicates_init, self.cubes_no_duplicates_STEREO_init, self.cubes_no_duplicates_SDO_init, \
        self.cubes_no_duplicates_new, self.cubes_no_duplicates_STEREO_new, self.cubes_no_duplicates_SDO_new, self.trace_cubes, \
        self.trace_cubes_no_duplicates_init, self.trace_cubes_no_duplicates_new, self.time_cubes_all_data, self.time_cubes_no_duplicates_init, \
        self.time_cubes_no_duplicates_new, self.cubes_lineofsight_STEREO, self.cubes_lineofsight_SDO, self.cubes_convolution, self.cubes_skeleton \
        = cubes_all_data, cubes_no_duplicates_init, cubes_no_duplicates_STEREO_init, cubes_no_duplicates_SDO_init, \
        cubes_no_duplicates_new, cubes_no_duplicates_STEREO_new, cubes_no_duplicates_SDO_new, trace_cubes, trace_cubes_no_duplicates_init, \
        trace_cubes_no_duplicates_new, time_cubes_all_data, time_cubes_no_duplicates_init, time_cubes_no_duplicates_new, \
        cubes_lineofsight_STEREO, cubes_lineofsight_SDO, cubes_convolution, cubes_skeleton
        
        if self.make_screenshots: self.Complete_sparse_arrays()

    def names(self) -> None:
        """
        To get the filenames of all the cubes.
        """

        # Setting the cube name pattern (only cube{:03d}.save files are kept)
        pattern = re.compile(r'cube(\d{3})\.save')

        # The cube names
        cube_names = [cube_name for cube_name in os.listdir(self.paths['Cubes_karine']) if pattern.match(cube_name)] 
        self._cube_names = sorted(cube_names) 
        self._cube_numbers = [int(pattern.match(cube_name).group(1)) for cube_name in self._cube_names] 
        self.cube_numbers_all = self._cube_numbers if not self.make_screenshots else np.arange(0, 413) # TODO: need to check noth these attributes as I have kindoff mixed them up...

    def dates_data(self) -> None:
        """
        To get the dates and times corresponding to all the used cubes.
        To do so images where both numbers are in the filename are used.
        """

        pattern_int = re.compile(r'\d{4}_(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2})\.\d{3}\.png')
        filepaths = sorted(glob(os.path.join(self.paths['Intensities'], '*.png')))

        if not self.make_screenshots:
            # Getting the corresponding filenames 
            filenames = []
            for number in self._cube_numbers: 
                for filepath in filepaths:
                    filename = os.path.basename(filepath)
                    if filename[:4] == f'{number:04d}':
                        filenames.append(filename)
                        break
            self.dates = [CustomDate(pattern_int.match(filename).group(1)) for filename in filenames]
        else:
            self.dates = [CustomDate(pattern_int.match(os.path.basename(filename)).group(1)) for filename in filepaths]
    
    @Decorators.running_time
    def Processing_data(self) -> list[np.ndarray]:
        """
        Downloading and processing the data depending on the arguments chosen.
        """

        # Multiprocessing initial I/O bound tasks
        manager = Manager()
        queue = manager.Queue()
        indexes = MultiProcessing.Pool_indexes(len(self._cube_names), self._processes)

        processes = [Process(target=self.Cubes, args=(queue, i, index)) for i, index in enumerate(indexes)]
        if self.line_of_sight: 
            for i, index in enumerate(indexes):
                processes.append(Process(target=self.Cubes_lineofsight_STEREO, args=(queue, self._processes + i, index)))
                processes.append(Process(target=self.Cubes_lineofsight_SDO, args=(queue, 2 * self._processes + i, index)))
        for p in processes: p.start()
        for p in processes: p.join()

        # Ordering the results gotten from the I/O bound tasks
        results = [None] * (self._processes * 3)
        while not queue.empty():
            identifier, result = queue.get()
            results[identifier] = result
        cubes = concatenate(results[:self._processes], axis=0)
        print(f'The cubes shape is {cubes.shape}')
        if self.line_of_sight:
            STEREO = concatenate(results[self._processes:self._processes * 2], axis=0)
            SDO = concatenate(results[self._processes * 2:], axis=0)

        self.cubes_shape = cubes.shape
        print(f'CUBES - {round(cubes.nbytes/ 2**20,3)}Mb')

        # CPU bound processes 
        processes = []
        results = [None] * 15
        if self.line_of_sight:
            results[-2] = STEREO
            results[-1] = SDO
        
        # Shared memory object and corresponding np.ndarray info
        shm_data, self._sparse_data = self.Shared_memory(cubes.data)
        shm_coords, self._sparse_coords = self.Shared_memory(cubes.coords)

        # Separating the data
        if self.all_data: processes.append(Process(target=self.Cubes_all_data, args=(queue,)))
        if self.no_duplicates:
            if self.cube_version_0: processes.append(Process(target=self.Cubes_no_duplicates_init, args=(queue,)))
            if self.cube_version_1: processes.append(Process(target=self.Cubes_no_duplicates_new, args=(queue,)))
        if self.duplicates:
            if self.cube_version_0:
                processes.append(Process(target=self.Cubes_STEREO_no_duplicates_init, args=(queue,)))
                processes.append(Process(target=self.Cubes_SDO_no_duplicates_init, args=(queue,)))
            if self.cube_version_1:
                processes.append(Process(target=self.Cubes_STEREO_no_duplicates_new, args=(queue,)))
                processes.append(Process(target=self.Cubes_SDO_no_duplicates_new, args=(queue,)))

        if self.trace_data: processes.append(Process(target=self.Cubes_trace, args=(queue,)))
        if self.trace_no_duplicates: 
            if self.cube_version_0: processes.append(Process(target=self.Cubes_trace_no_duplicates_init, args=(queue,)))
            if self.cube_version_1: processes.append(Process(target=self.Cubes_trace_no_duplicates_new, args=(queue,)))

        if self.time_intervals_all_data or self.time_intervals_no_duplicates:
            self._days_per_month = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
            date = self.dates[0]
            if (date.year % 4 == 0 and date.year % 100 !=0) or (date.year % 400 == 0): self._days_per_month[2] = 29  # for leap years

        if self.time_intervals_all_data: processes.append(Process(target=self.Cubes_time_chunks, args=(queue,)))

        if self.time_intervals_no_duplicates:
            processes.append(Process(target=self.Cubes_time_chunks_no_duplicates_init, args=(queue,)))
            processes.append(Process(target=self.Cubes_time_chunks_no_duplicates_new, args=(queue,)))

        for p in processes: p.start()
        for p in processes: p.join()

        shm_data.unlink()
        shm_coords.unlink()

        # Getting, ordering and saving the results
        while not queue.empty():
            identifier, result = queue.get()
            results[identifier] = result
        return results
    
    def Sparse_data(self, cubes: np.ndarray) -> COO:
        """
        Changes data to a sparse representation.

        Args:
            cubes (np.ndarray): the initial array.

        Returns:
            sparse.COO: the corresponding sparse COO array.
        """

        cubes = COO(cubes)  # the .to_numpy() method wasn't used as the idx_type argument isn't working properly
        cubes.coords = cubes.coords.astype('uint16')  # to save memory
        return cubes
    
    def Cubes(self, queue: QUEUE, queue_index: int, index: tuple[int, int]) -> None:
        """
        To import the cubes in sections as it is mainly an I/O bound task.
        """

        cubes = [readsav(os.path.join(self.paths['Cubes_karine'], cube_name)).cube.astype('uint8') for cube_name in self._cube_names[index[0]:index[1] + 1]]
        cubes = np.stack(cubes, axis=0)
        cubes = np.transpose(cubes, (0, 3, 2, 1))
        cubes = self.Sparse_data(cubes)
        queue.put((queue_index, cubes))

    def Cubes_lineofsight_STEREO(self, queue: QUEUE, queue_index: int, index: tuple[int, int]) -> None:
        """
        To trace the cubes for the line of sight STEREO data. Also imported in sections as it is mainly an I/O bound task.
        """

        cubes_lineofsight_STEREO = [readsav(os.path.join(self.paths['Cubes_karine'], cube_name)).cube1.astype('uint8') for cube_name in self._cube_names[index[0]:index[1] + 1]]  
        cubes_lineofsight_STEREO = np.array(cubes_lineofsight_STEREO)  # line of sight seen from STEREO 
        cubes_lineofsight_STEREO = np.transpose(cubes_lineofsight_STEREO, (0, 3, 2, 1))
        cubes_lineofsight_STEREO = self.Sparse_data(cubes_lineofsight_STEREO)
        queue.put((queue_index, cubes_lineofsight_STEREO))
        
    def Cubes_lineofsight_SDO(self, queue: QUEUE, queue_index: int, index: tuple[int, int]) -> None:
        """
        To trace the cubes for the line of sight SDO data. Also imported in sections as it is mainly an I/O bound task.
        """

        cubes_lineofsight_SDO = [readsav(os.path.join(self.paths['Cubes_karine'], cube_name)).cube2.astype('uint8') for cube_name in self._cube_names[index[0]:index[1] + 1]]
        cubes_lineofsight_SDO = np.array(cubes_lineofsight_SDO)  # line of sight seen from SDO
        cubes_lineofsight_SDO = np.transpose(cubes_lineofsight_SDO, (0, 3, 2, 1))
        cubes_lineofsight_SDO = self.Sparse_data(cubes_lineofsight_SDO)
        queue.put((queue_index, cubes_lineofsight_SDO))

    def Shared_memory(self, data: np.ndarray) -> tuple[SharedMemory, dict[str, any]]:
        """
        Creating a shared memory space given an input np.ndarray.
        """

        # Initialisations
        shm = SharedMemory(create=True, size=data.nbytes)
        info = {
            'shm.name': shm.name,
            'data.shape': data.shape,
            'data.dtype': data.dtype,
        }
        shared_array = np.ndarray(info['data.shape'], dtype=info['data.dtype'], buffer=shm.buf)
        np.copyto(shared_array, data)
        shm.close()
        return shm, info
    
    def Shared_array_reconstruction(self) -> COO:
        """
        To reconstruct the shared COO array as it is separated in a data and a coords np.ndarray.
        """

        shm_data = SharedMemory(name=self._sparse_data['shm.name'])
        shm_coords = SharedMemory(name=self._sparse_coords['shm.name'])

        data = np.ndarray(self._sparse_data['data.shape'], dtype=self._sparse_data['data.dtype'], buffer=shm_data.buf)
        coords = np.ndarray(self._sparse_coords['data.shape'], dtype=self._sparse_coords['data.dtype'], buffer=shm_coords.buf)
        cubes = COO(coords=coords, data=data, shape=self.cubes_shape)
        cubes = COO.copy(cubes)  # had to add a .copy() as it wasn't working properly
        shm_data.close()
        shm_coords.close()
        return cubes

    def Cubes_all_data(self, queue: QUEUE) -> None:
        """
        To create the cubes for all the data.
        """

        cubes = self.Shared_array_reconstruction()
        cubes_all_data = self.Sparse_data(cubes & 0b00000001).astype('uint8')
        queue.put((0, cubes_all_data))
    
    def Cubes_no_duplicates_init(self, queue: QUEUE) -> None:
        """
        To create the cubes for the no duplicate data.
        """

        cubes = self.Shared_array_reconstruction()
        cubes_no_duplicate = self.Sparse_data(cubes == 0b00000110).astype('uint8')  # no  duplicates 
        queue.put((1, cubes_no_duplicate))

    def Cubes_STEREO_no_duplicates_init(self, queue: QUEUE) -> None:
        """
        To create the cubes for the no duplicate data from STEREO.
        """

        cubes = self.Shared_array_reconstruction()
        cubes_no_duplicates_STEREO = self.Sparse_data((cubes & 0b00000010) > 0).astype('uint8')
        queue.put((2, cubes_no_duplicates_STEREO))
    
    def Cubes_SDO_no_duplicates_init(self, queue: QUEUE) -> None:
        """
        To create the cubes for the no duplicate data from SDO.
        """

        cubes = self.Shared_array_reconstruction()
        cubes_no_duplicates_SDO = self.Sparse_data((cubes & 0b00000100) > 0).astype('uint8')
        queue.put((3, cubes_no_duplicates_SDO))
        
    def Cubes_no_duplicates_new(self, queue: QUEUE) -> None:
        """
        To create the cubes for the no duplicate data.
        """

        cubes = self.Shared_array_reconstruction()
        cubes_no_duplicate = self.Sparse_data(cubes == 0b00011000).astype('uint8')  # no  duplicates 
        queue.put((4, cubes_no_duplicate))

    def Cubes_STEREO_no_duplicates_new(self, queue: QUEUE) -> None:
        """
        To create the cubes for the no duplicate data from STEREO.
        """

        cubes = self.Shared_array_reconstruction()
        cubes_no_duplicates_STEREO = self.Sparse_data((cubes & 0b00001000) > 0).astype('uint8')
        queue.put((5, cubes_no_duplicates_STEREO))
    
    def Cubes_SDO_no_duplicates_new(self, queue: QUEUE) -> None:
        """
        To create the cubes for the no duplicate data from SDO.
        """

        cubes = self.Shared_array_reconstruction()
        cubes_no_duplicates_SDO = self.Sparse_data((cubes & 0b00010000) > 0).astype('uint8')
        queue.put((6, cubes_no_duplicates_SDO))

    def Cubes_trace(self, queue: QUEUE) -> None:
        """
        To create the cubes of the trace of all the data.
        """

        cubes = self.Shared_array_reconstruction()
        trace_cube = COO.any(cubes, axis=0).astype('uint8')
        queue.put((7, trace_cube))

    def Cubes_trace_no_duplicates_init(self, queue: QUEUE) -> None:
        """
        To create the cubes of the trace of the no duplicate.
        """

        cubes = self.Shared_array_reconstruction()
        cubes_no_duplicate = self.Sparse_data(cubes == 0b00000110)
        trace_cube_no_duplicate = COO.any(cubes_no_duplicate, axis=0).astype('uint8')
        queue.put((8, trace_cube_no_duplicate))

    def Cubes_trace_no_duplicates_new(self, queue: QUEUE) -> None:
        """
        To create the cubes of the trace of the no duplicate.
        """

        cubes = self.Shared_array_reconstruction()
        cubes_no_duplicate = self.Sparse_data(cubes == 0b00011000)
        trace_cube_no_duplicate = COO.any(cubes_no_duplicate, axis=0).astype('uint8')
        queue.put((9, trace_cube_no_duplicate))

    def Cubes_time_chunks(self, queue: QUEUE) -> None:
        """
        To create the cubes for the time integrations for all data.
        """

        cubes = self.Shared_array_reconstruction()
        time_cubes_all_data = []
        cubes_all_data = self.Sparse_data(cubes & 0b00000001).astype('uint8')
        for date in self.dates:
            date_seconds = (((self._days_per_month[date.month] + date.day) * 24 + date.hour) * 60 + date.minute) * 60 + date.second

            date_min = date_seconds - self.time_interval / 2
            date_max = date_seconds + self.time_interval / 2      
            time_cubes_all_data.append(self.Time_chunks(cubes_all_data, date_max, date_min)) 
        
        time_cubes_all_data = stack(time_cubes_all_data, axis=0)
        time_cubes_all_data = self.Sparse_data(time_cubes_all_data).astype('uint8')
        queue.put((10, time_cubes_all_data))

    def Cubes_time_chunks_no_duplicates_init(self, queue: QUEUE) -> None:
        """
        To create the cubes for the time integrations for the no duplicates.
        """

        cubes = self.Shared_array_reconstruction()
        time_cubes_no_duplicate = [] 
        cubes_no_duplicate = self.Sparse_data(cubes == 0b00000110).astype('uint8')
        for date in self.dates:
            date_seconds = (((self._days_per_month[date.month] + date.day) * 24 + date.hour) * 60 + date.minute) * 60 + date.second

            date_min = date_seconds - self.time_interval / 2
            date_max = date_seconds + self.time_interval / 2
            time_cubes_no_duplicate.append(self.Time_chunks(cubes_no_duplicate, date_max, date_min))
        
        time_cubes_no_duplicate = stack(time_cubes_no_duplicate, axis=0)
        time_cubes_no_duplicate = self.Sparse_data(time_cubes_no_duplicate).astype('uint8')
        queue.put((11, time_cubes_no_duplicate))

    def Cubes_time_chunks_no_duplicates_new(self, queue: QUEUE) -> None:
        """
        To create the cubes for the time integrations for the no duplicates.
        """

        cubes = self.Shared_array_reconstruction()
        time_cubes_no_duplicate = [] 
        cubes_no_duplicate = self.Sparse_data(cubes == 0b00011000).astype('uint8')
        for date in self.dates:
            date_seconds = (((self._days_per_month[date.month] + date.day) * 24 + date.hour) * 60 + date.minute) * 60 + date.second

            date_min = date_seconds - self.time_interval / 2
            date_max = date_seconds + self.time_interval / 2
            time_cubes_no_duplicate.append(self.Time_chunks(cubes_no_duplicate, date_max, date_min))
        
        time_cubes_no_duplicate = stack(time_cubes_no_duplicate, axis=0)
        time_cubes_no_duplicate = self.Sparse_data(time_cubes_no_duplicate).astype('uint8')
        queue.put((12, time_cubes_no_duplicate))

    def Time_interval(self) -> None:
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

    def Time_chunks(self, cubes: COO, date_max: int, date_min: int) -> COO:
        """
        To select the data in the time chunk given the data chosen for the integration.
        """

        chunk = []
        for date2, data2 in zip(self.dates, cubes):
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
          
    def Complete_sparse_arrays(self) -> None: 
        """
        To reformat the sparse arrays so that the number of values is equal to the total number of cube numbers used.
        """

        numbers_len = len(self.cube_numbers_all)

        cubes_lineofsight_STEREO = [None] * numbers_len
        cubes_lineofsight_SDO = [None] * numbers_len
        cubes_all_data = [None] * numbers_len
        cubes_no_duplicates_init = [None] * numbers_len
        cubes_no_duplicates_STEREO_init = [None] * numbers_len
        cubes_no_duplicates_SDO_init = [None] * numbers_len
        cubes_no_duplicates_new = [None] * numbers_len
        cubes_no_duplicates_STEREO_new = [None] * numbers_len
        cubes_no_duplicates_SDO_new = [None] * numbers_len
        index = -1

        for i, number in enumerate(self.cube_numbers_all):
            if number in self._cube_numbers:
                index += 1
                if self.all_data: cubes_all_data[i] = self.cubes_all_data[index]
                if self.line_of_sight: 
                    cubes_lineofsight_STEREO[i] = self.cubes_lineofsight_STEREO[index]
                    cubes_lineofsight_SDO[i] = self.cubes_lineofsight_SDO[index] 
                if self.cube_version_0:
                    if self.no_duplicates: cubes_no_duplicates_init[i] = self.cubes_no_duplicates_init[index]
                    if self.duplicates:
                        cubes_no_duplicates_STEREO_init[i] = self.cubes_no_duplicates_STEREO_init[index]
                        cubes_no_duplicates_SDO_init[i] = self.cubes_no_duplicates_SDO_init[index]
                if self.cube_version_1:
                    if self.no_duplicates: cubes_no_duplicates_new[i] = self.cubes_no_duplicates_new[index] 
                    if self.duplicates:
                        cubes_no_duplicates_STEREO_new[i] = self.cubes_no_duplicates_STEREO_new[index]
                        cubes_no_duplicates_SDO_new[i] = self.cubes_no_duplicates_SDO_new[index]
        self.cubes_lineofsight_STEREO = cubes_lineofsight_STEREO
        self.cubes_lineofsight_SDO = cubes_lineofsight_SDO
        self.cubes_all_data = cubes_all_data
        self.cubes_no_duplicates_init = cubes_no_duplicates_init
        self.cubes_no_duplicates_STEREO_init = cubes_no_duplicates_STEREO_init
        self.cubes_no_duplicates_SDO_init = cubes_no_duplicates_SDO_init
        self.cubes_no_duplicates_new = cubes_no_duplicates_new
        self.cubes_no_duplicates_STEREO_new = cubes_no_duplicates_STEREO_new
        self.cubes_no_duplicates_SDO_new = cubes_no_duplicates_SDO_new

    def sun_pos(self) -> None:
        """
        To find the Sun's radius and the center position in the cubes reference frame.
        """

        # Reference data 
        first_cube_name = os.path.join(self.paths['Cubes_karine'], self._cube_names[0])

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

    def Sun_texture(self) -> tuple[np.ndarray, tuple[int, int]]:
        """
        For adding a carrington grid to the sun.
        """

        texture_height, texture_width = 720, 1440
        image = np.ones((texture_height, texture_width)).astype('uint8')

        # Adding a black longitude latitude grid-line  each 15 degrees:
        nb_of_grids_lat = np.arange(1, 180 / self._heliographic_grid_degrees, 1)
        nb_of_grids_lon = np.arange(1, 360 / self._heliographic_grid_degrees + 1, 1)

        grid_index_lat = self._heliographic_grid_degrees * nb_of_grids_lat / 0.25
        grid_index_lon = self._heliographic_grid_degrees * nb_of_grids_lon / 0.25

        for lat in grid_index_lat: image[int(lat) - 1, :] = 0
        for lon in grid_index_lon: image[:, int(lon) - 1] = 0
        # nw_image = np.flip(image, axis=0)
        return image, (texture_height, texture_width)

    def Sun_points(self, dimensions: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
        """
        Creates a spherical cloud of points that represents the pixels on the Sun's surface.
        """

        # Initialisation
        height, width = dimensions
        N = self._sun_texture_resolution  # number of points in the theta direction
        phi = np.linspace(0, np.pi, N)  # latitude of the points
        theta = np.linspace(0, 2 * np.pi, 2 * N)  # longitude of the points
        phi, theta = np.meshgrid(phi, theta)  # the subsequent meshgrid

        # Conversion to cartesian coordinates
        x = self.radius_index * np.sin(phi) * np.cos(theta) + self.sun_center[0]
        y = self.radius_index * np.sin(phi) * np.sin(theta) + self.sun_center[1]
        z = self.radius_index * np.cos(phi) + self.sun_center[2] 

        print(f"x, y and z shapes for the sun are {x.shape}, {y.shape}, {z.shape}.")

        # Creation of the position of the spherical cloud of points
        self.sun_points = np.array([z.ravel(), y.ravel(), x.ravel()], dtype='float32').T
        print(f'sun_points shape is {self.sun_points.shape}')

        # The corresponding image indexes to get the colors
        sun_texture_x = np.linspace(0, height - 1, N, dtype='uint16')
        sun_texture_y = np.linspace(0, width - 1, N * 2, dtype='uint16')
        return sun_texture_x, sun_texture_y
    
    def Colours_1D(self, sun_texture: np.ndarray, sun_texture_x: np.ndarray, sun_texture_y: np.ndarray) -> None:
        """
        Creates a 1D array of the integer Hex pixel values (0x000000 format) of a 2D sun texture image.
        """

        x_indices = sun_texture_x[:, np.newaxis]
        y_indices = sun_texture_y[np.newaxis, :]

        colours = sun_texture[x_indices, y_indices].flatten()
        normalized_colours = (colours - np.min(colours)) / (np.max(colours) - np.min(colours))
        blue_val = (normalized_colours * 255).astype('int')  # TODO: change to uint8, but might create problems if I remember correctly 
        hex_colours = (blue_val << 16) + (blue_val << 8) + blue_val
        self.hex_colours = hex_colours.astype('uint32')
        print(f'hex_colours shape is {self.hex_colours.shape}')

    def STEREO_stats(self) -> None:
        """
        To save the information needed to find the position of STEREO.
        """

        data = readsav(os.path.join(self.paths['Main'], 'rainbow_stereob_304.save')).datainfos

        # Multiprocessing initialisation
        manager = Manager()
        queue = manager.Queue()
        indexes = MultiProcessing.Pool_indexes(len(self.cube_numbers_all), self._processes)

        processes = [Process(target=self.STEREO_coords, args=(queue, i, data, index)) for i, index in enumerate(indexes)]
        for p in processes: p.start()
        for p in processes: p.join()

        # Ordering the results
        results = [None] * self._processes
        while not queue.empty():
            identifier, result = queue.get()
            results[identifier] = result
        self.STEREO_pos = np.concatenate(results, axis=0) 

    def STEREO_coords(self, queue: QUEUE, i: int, data, data_index: tuple[int, int]) -> None:  # TODO: need to check the type() of .datainfos in this case
        """
        To get the position of STEREO given the fits filepath.
        Done like this as the computation is I/O bound and so the paths are separated in sections for multiprocessing.
        """

        from astropy import units as u
        from astropy.coordinates import CartesianRepresentation
        from sunpy.coordinates.frames import  HeliographicCarrington

        stereo_pos = [None] * (data_index[1] - data_index[0] + 1)
        for index, number in enumerate(self.cube_numbers_all[data_index[0]:data_index[1] + 1]):
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

            stereo_pos[index] = self.sun_center + np.array([xpos_index, ypos_index, zpos_index]) 
        queue.put((i, np.array(stereo_pos)))       
    
    def STEREO_pov_center(self) -> None:
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

    def SDO_stats(self) -> None:
        """
        To save the information needed to find the position of SDO.
        """

        SDO_fits_names = [os.path.join(self.paths['SDO'], f'AIA_fullhead_{number:03d}.fits.gz') for number in self.cube_numbers_all]

        # Multiprocessing initialisation
        manager = Manager()
        queue = manager.Queue()
        indexes = MultiProcessing.Pool_indexes(len(SDO_fits_names), self._processes)

        # Processes 
        processes = [
            Process(target=self.SDO_coords, args=(queue, i, SDO_fits_names[index[0]:index[1] + 1]))
            for i, index in enumerate(indexes)
        ]
        for p in processes: p.start()
        for p in processes: p.join()

        # Ordering the results
        results = [None] * self._processes
        while not queue.empty():
            identifier, result = queue.get()
            results[identifier] = result
        self.SDO_pos = np.concatenate(results, axis=0)

    def SDO_coords(self, queue: QUEUE, i: int, paths: list[str]) -> None:
        """
        To get the position of SDO given the fits filepath.
        Done like this as the computation is I/O bound and so the paths are separated in sections for multiprocessing.
        """

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

    def Conv3d_results(self) -> tuple[COO, COO | None]:
        """
        just testing the result gotten from the convolution
        """

        data = np.load(os.path.join('..', 'test_conv3d_array', 'conv3dRainbow.npy')).astype('uint8')

        binary_data = data > self.conv_treshold
        cubes_convolution = self.Sparse_data(binary_data)

        if self.skeleton:
            # Multiprocessing initialisation
            nb_of_batches = 6
            manager = Manager()
            queue = manager.Queue()
            indexes = MultiProcessing.Pool_indexes(self.cubes_shape[0], nb_of_batches)

            # Defining and running the processes
            processes = [Process(target=self.Skeleton_loop, args=(queue, i, binary_data[index[0]:index[1] + 1])) for i, index in enumerate(indexes)]
            for p in processes: p.start()
            for p in processes: p.join()

            # Ordering the results
            results = [None] * nb_of_batches
            while not queue.empty():
                identifier, result = queue.get()
                results[identifier] = result
            cubes_barycenter = np.concatenate(results, axis=0)
            cubes_skeleton = self.Sparse_data(cubes_barycenter)
            return cubes_convolution, cubes_skeleton
        return cubes_convolution, None

    def Skeleton_loop(self, queue: QUEUE, i: int, data: np.ndarray) -> None:
        """
        Creating the skeleton of each row of a 4D np.ndarray.
        """

        skeletons = [None] * len(data)
        for index, cube in enumerate(data):
            skeleton = skeletonize_3d(cube)
            skeletons[index] = skeleton
        skeletons = np.array(skeletons)
        queue.put((i, skeletons))

    def Preprocessing_polynomial_data(self) -> None:
        """
        Code to get the .npy barycenter filenames.
        """

        pattern = re.compile(r'''poly_
                             (?P<datatype>[a-zA-Z0-9]+)_
                             (lim_(?P<conv_limit>\d+)_)?
                             order(?P<order>\d+)_''', re.VERBOSE)
        filenames = os.listdir(self.paths['polynomials'])
        
        files_dataNmatches = [] 
        for filename in filenames:
            filename_match = pattern.match(filename)

            if filename_match:
                # Loading the data
                file_data = np.load(os.path.join(self.paths['polynomials'], filename))

                # Setting up the multiprocessing
                times = np.unique(file_data[0, :]).astype('uint16')  # Getting all the unique time values to then do a multiprocessed np.unique for each time value
                manager = Manager()
                queue = manager.Queue()
                indexes = MultiProcessing.Pool_indexes(len(times), self._processes)
                shm, file_data = self.Shared_memory(file_data[[0, 3, 2, 1]])

                processes = [
                    Process(target=self.Preprocessing_polynomial_data_sub, args=(queue, file_data, i, (times[index[0]], times[index[1]])))
                    for i, index in enumerate(indexes)
                ]
                for p in processes: p.start()
                for p in processes: p.join()

                results = [None] * self._processes
                while not queue.empty():
                    identifier, result = queue.get()
                    results[identifier] = result
                files_dataNmatches.append((filename_match, np.concatenate(results, axis=1)))
                shm.unlink()
            else:
                print(f"\033[92mPolynomial array filename {filename} doesn't match the usual pattern. \033[0m")
        self.polynomials_matchesNdata = files_dataNmatches

    def Preprocessing_polynomial_data_sub(self, queue: QUEUE, data: dict[str, any], position: int, index: tuple[int, int]) -> None:
        """
        To multiprocess the preprocessing for the polynomial data as the np.unique() function is quite slow.
        """

        # Opening the shared memory
        shm = SharedMemory(name=data['shm.name'])
        data = np.ndarray(data['data.shape'], dtype=data['data.dtype'], buffer=shm.buf)

        filters = (data[0, :] >= index[0]) & (data[0, :] <= index[1])
        data = np.copy(data[:, filters])
        shm.close()
        
        data = np.rint(np.abs(data.T))
        data = np.unique(data, axis=0).T.astype('uint16')
        data = COO(coords=data, data=1, shape=self.cubes_shape)
        queue.put((position, data))

    def attribute_deletion(self) -> None:
        """
        To delete some of the attributes that are not used in the inherited class. Done to save some RAM.
        """

        # Private attributes 
        del self._sun_texture_resolution, self._cube_names, self._cube_numbers, self._processes, self._days_per_month, self._length_dx, self._length_dy, self._length_dz

# @ClassDecorator(Decorators.running_time)
class K3dAnimation(Data):
    """
    Creates the corresponding k3d animation to then be used in a Jupyter notebook file.
    """

    @typechecked
    def __init__(self, compression_level: int = 9, plot_height: int = 1260, sleep_time: int | float = 2, 
                 camera_fov: int | float | str = 0.23, camera_zoom_speed: int | float = 0.7, 
                 screenshot_scale: int | float = 2, screenshot_sleep: int | float = 5,  screenshot_version: str = 'vtest', 
                 camera_pos: tuple[int | float, int | float, int | float] | None = None, up_vector: tuple[int, int, int] = (0, 0, 1), 
                 visible_grid: bool = False, outlines: bool = False,  **kwargs):
        
        super().__init__(**kwargs)

        # Arguments
        self.plot_height = plot_height  # the height in pixels of the plot (initially it was 512)
        self.sleep_time = sleep_time  # sets the time between each frames (in seconds)
        self.camera_zoom_speed = camera_zoom_speed  # zoom speed of the camera 
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
        
        self.kwargs = {
            'compression_level': compression_level, # the compression level of the data in the 3D visualisation
            'outlines': outlines,
        }

        # Instance attributes set when running the class
        self.plot: k3d.plot  # plot object
        self.plot_alldata: k3d.voxels  # voxels plot of the all data 
        self.plot_dupli_STEREO_init: k3d.voxels  # voxels plot of the no duplicates seen from STEREO for the first method
        self.plot_dupli_STEREO_new: k3d.voxels  # same for the second method
        self.plot_dupli_SDO_init: k3d.voxels  # same for SDO for the first method
        self.plot_dupli_SDO_new: k3d.voxels  # same for the second method
        self.plot_dupli_init: k3d.voxels  # voxels plot of the no duplicate for the first method
        self.plot_dupli_new: k3d.voxels  # same for the second method
        self.plot_los_STEREO_init: k3d.voxels  # voxels plot of the line of sight from STEREO for the first method
        self.plot_los_STEREO_new: k3d.voxels  # same for the second method
        self.plot_los_SDO_init: k3d.voxels  # same seen from SDO for the first method
        self.plot_los_SDO_new: k3d.voxels  # same for the second method
        self.plot_interv_init: k3d.voxels  # voxels plot for the time integration of all data for the first method
        self.plot_interv_new: k3d.voxels  # same for the second method
        self.plot_interv_dupli_init: k3d.voxels  # same for the no duplicates data for the first method
        self.plot_interv_dupli_new: k3d.voxels  # same for the second method
        self.play_pause_button: widgets.ToggleButton  # Play/Pause widget initialisation
        self.time_slider: widgets.IntSlider # time slider widget
        self.date_dropdown: widgets.Dropdown  # Date dropdown widget to show the date
        self.time_link: widgets.jslink  # JavaScript Link between the two widgets
        self.date_text: list[str]  # gives the text associated to each time frames 

        # Making the animation
        if self.make_screenshots: self.Update_paths()
        if self.time_intervals_all_data or self.time_intervals_no_duplicates: self.Time_interval_string()
        self.Date_strings()
        self.Animation()

    @typechecked
    @classmethod
    def The_usual(cls, version: int, data: str = 'no_duplicate', **classkwargs) -> K3dAnimation:
        """
        Gives the usual arguments used when making screenshots for a given point of view and data type.
        """

        if version==0:
            kwargs = {'sun': True, 'sdo_pov': True, 'fov_center':'stereo', 'camera_fov': 'sdo', 'up_vector': (0, 0, 1), 
                      'make_screenshots': True, 'screenshot_version': 'v0', 'screenshot_scale': 2, 
                      'sun_texture_resolution': 1920}
        elif version==1:
            kwargs = {'sun': True, 'stereo_pov': True, 'fov_center': 'stereo', 'camera_fov': 'stereo', 'up_vector': (0, 0, 1), 
                      'make_screenshots': True, 'screenshot_version': 'v1', 'screenshot_scale': 1, 
                      'sun_texture_resolution': 1920}        
        elif version==2:
            kwargs = {'sun': True, 'fov_center': 'cubes', 'camera_pos': (-0.7, 0.7, 0), 'up_vector': (0, 0, 1),
                      'make_screenshots': True, 'screenshot_version': 'v2', 'screenshot_scale': 1,
                      'sun_texture_resolution': 1920}
        elif version==3:
            kwargs = {'sun': True, 'fov_center': 'cubes', 'camera_pos': (0, 0, 1) , 'up_vector': (-1, 0, 0), 
                      'make_screenshots': True, 'screenshot_version': 'v3', 'screenshot_scale': 1, 
                      'sun_texture_resolution': 1920}     
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
    
    def Update_paths(self) -> None:
        """
        Updating the paths of the parent class to be able to save screenshots.
        """

        self.paths['Screenshots'] = os.path.join(self.paths['Main'], 'texture_screenshots')
        os.makedirs(self.paths['Screenshots'], exist_ok=True)

    def Fov_for_SDO(self) -> int | float:
        """
        To get the same FOV than SDO when the fov_center parameter is the Sun.
        """

        hdul = fits.open(os.path.join(self.paths['SDO'], 'AIA_fullhead_000.fits.gz'))
        image_shape = np.array(hdul[0].data).shape
        Total_fov_in_degrees = image_shape[0] * hdul[0].header['CDELT1'] / 3600
        hdul.close()
        return Total_fov_in_degrees / 3

    def Full_array(self, sparse_cube: COO) -> np.ndarray:
        """
        To recreate a full 3D np.array from a sparse np.ndarray representing a 3D volume.
        If the initial value is None, returns an empty np.ndarray with the right shape.
        """

        if sparse_cube:
            cube = sparse_cube.todense()
            return cube.astype('uint8')
        else:
            return np.zeros((self.cubes_shape[1], self.cubes_shape[2], self.cubes_shape[3]), dtype='uint8')

    def Camera_params(self) -> None:
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

    def Update_voxel(self, change: dict[str, any]) -> None:
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
                plot.voxels = self.Full_array(data[change['new']])

        if self.all_data: self.plot_alldata.voxels = self.Full_array(self.cubes_all_data[change['new']])

        if self.duplicates:
            if self.cube_version_0:
                self.plot_dupli_STEREO_init.voxels = self.Full_array(self.cubes_no_duplicates_STEREO_init[change['new']])
                self.plot_dupli_SDO_init.voxels = self.Full_array(self.cubes_no_duplicates_SDO_init[change['new']])
            if self.cube_version_1:
                self.plot_dupli_STEREO_new.voxels = self.Full_array(self.cubes_no_duplicates_STEREO_new[change['new']])
                self.plot_dupli_SDO_new.voxels = self.Full_array(self.cubes_no_duplicates_SDO_new[change['new']])

        if self.no_duplicates:
            if self.cube_version_0: self.plot_dupli_init.voxels = self.Full_array(self.cubes_no_duplicates_init[change['new']])
            if self.cube_version_1: self.plot_dupli_new.voxels = self.Full_array(self.cubes_no_duplicates_new[change['new']])

        if self.line_of_sight:
            self.plot_los_STEREO_set1.voxels = self.Full_array(self.cubes_lineofsight_STEREO[change['new']])
            self.plot_los_SDO_set1.voxels = self.Full_array(self.cubes_lineofsight_SDO[change['new']])
 
        if self.time_intervals_all_data:
            if self.cube_version_0: self.plot_interv_init.voxels = self.Full_array(self.time_cubes_all_data_init[change['new']]) 
            if self.cube_version_1: self.plot_interv_new.voxels = self.Full_array(self.time_cubes_all_data_new[change['new']])
                          
        if self.time_intervals_no_duplicates:
            if self.cube_version_0: self.plot_interv_dupli_init.voxels = self.Full_array(self.time_cubes_no_duplicates_init[change['new']])
            if self.cube_version_1: self.plot_interv_dupli_new.voxels = self.Full_array(self.time_cubes_no_duplicates_new[change['new']])
       
        if self.skeleton: self.plot_skeleton.voxels = self.Full_array(self.cubes_skeleton[change['new']])
        if self.convolution: self.plot_convolution.voxels = self.Full_array(self.cubes_convolution[change['new']])
        if self.make_screenshots: self.Screenshot_making()

        if self.html_snapshot:
            sleep(4)
            with open(f"snapshot_date{self.date_text[change['new']]}.html", "w") as f:
                f.write(self.plot.get_snapshot())

    def Play(self) -> None:
        """
        Params for the play button.
        """
        
        if self.play_pause_button.value and self.time_slider.value < len(self.cube_numbers_all) - 1:
            self.time_slider.value += 1
            Timer(self.sleep_time, self.Play).start()  # where you also set the sleep() time.
                
        else:
            self.play_pause_button.description = 'Play'
            self.play_pause_button.icon = 'play'

    def Screenshot_making(self) -> None:
        """
        To create a screenshot of the plot. A sleep time was added as the screenshot method relies
        on asynchronus traitlets mechanism.
        """

        import base64

        self.plot.fetch_screenshot()
        sleep(self.screenshot_sleep)

        screenshot_png = base64.b64decode(self.plot.screenshot)
        if self.time_intervals_no_duplicates:
            screenshot_name = f'nodupli_interval{self.time_interval}_{self.date_text[self.time_slider.value]}_{self.version}.png'
        elif self.time_intervals_all_data:
            screenshot_name = f'alldata_interval{self.time_interval}_{self.date_text[self.time_slider.value]}_{self.version}.png'
        elif self.no_duplicates:
            screenshot_name = f'nodupli_{self.time_slider.value:03d}_{self.date_text[self.time_slider.value]}_{self.version}.png'
        elif self.all_data:
            screenshot_name = f'alldata_{self.time_slider.value:03d}_{self.date_text[self.time_slider.value]}_{self.version}.png'
        else:
            raise ValueError("The screenshot name for that type of data still hasn't been created.")
        
        screenshot_namewpath = os.path.join(self.paths['Screenshots'], screenshot_name)
        with open(screenshot_namewpath, 'wb') as f:
            f.write(screenshot_png)

    def Play_pause_handler(self, change: dict[str, any]) -> None:
        """
        Changes the play button to pause when it is clicked.
        """

        if change['new']:  # if clicked play
            self.Play()
            self.play_pause_button.description = 'Pause'
            self.play_pause_button.icon = 'pause'

    def Date_strings(self) -> None:
        """
        Uses the dates for the files to create a corresponding string list.
        """

        self.date_text = [f'{date.year}-{date.month:02d}-{date.day:02d}_{date.hour:02d}h{date.minute:02d}min' for date in self.dates]

    def Time_interval_string(self) -> None:
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
    
    def Random_hexadecimal_color_generator(self) -> iter[int]:
        """
        Generator that yields a color value in integer hexadecimal code format.
        """

        while True:
            yield np.random.randint(0, 0xffffff)

    def Animation(self) -> None:
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
                limit_val = f"_lim{pattern.group('conv_limit')}" if pattern.group('conv_limit') else None
                plot = k3d.voxels(self.Full_array(data[0]), opacity=0.95, color_map=[next(self.Random_hexadecimal_color_generator())], 
                                  name=f"fit_order{pattern.group('order')}_data: {limit_val if limit_val else ''}_{pattern.group('datatype')}", **self.kwargs)
                self.plots_polynomials.append(plot)
                self.plot += plot

        # Adding the SUN!!!
        if self.sun:
            self.plot += k3d.points(positions=self.sun_points, point_size=3.5, colors=self.hex_colours, shader='flat', name='SUN',
                                    compression_level=self.kwargs['compression_level'])

        # Adding the different data sets (i.e. with or without duplicates)
        if self.all_data:  #old color color_map=[0x90ee90]
            self.plot_alldata = k3d.voxels(self.Full_array(self.cubes_all_data[0]), opacity=0.5, color_map=[0x0000ff], name='All data', **self.kwargs)
            self.plot += self.plot_alldata      
       
        if self.duplicates:
            if self.cube_version_0:
                self.plot_dupli_STEREO_init = k3d.voxels(self.Full_array(self.cubes_no_duplicates_STEREO_init[0]), color_map=[0xff0000], 
                                                         name='M1: no duplicates from SDO', **self.kwargs)
                self.plot_dupli_SDO_init = k3d.voxels(self.Full_array(self.cubes_no_duplicates_SDO_init[0]), color_map=[0x0000ff], name='M1: no duplicates from STEREO', 
                                                      **self.kwargs)
                self.plot += self.plot_dupli_STEREO_init + self.plot_dupli_SDO_init
            if self.cube_version_1:
                self.plot_dupli_STEREO_new = k3d.voxels(self.Full_array(self.cubes_no_duplicates_STEREO_new[0]), color_map=[0xff0000], name='M2: no duplicates from SDO',
                                                        **self.kwargs)
                self.plot_dupli_SDO_new = k3d.voxels(self.Full_array(self.cubes_no_duplicates_SDO_new[0]), color_map=[0x0000ff], name='M2: no duplicates from STEREO',
                                                     **self.kwargs)
                self.plot += self.plot_dupli_STEREO_new + self.plot_dupli_SDO_new
               
        if self.no_duplicates:
            if self.cube_version_0:
                self.plot_dupli_init = k3d.voxels(self.Full_array(self.cubes_no_duplicates_init[0]), color_map=[0x0000ff], opacity=0.3, name='M1: no duplicates',
                                                  **self.kwargs)
                self.plot += self.plot_dupli_init
            if self.cube_version_1:
                self.plot_dupli_new = k3d.voxels(self.Full_array(self.cubes_no_duplicates_new[0]), color_map=[0x0000ff], opacity=0.3, name='M2: no duplicates',
                                                 **self.kwargs)
                self.plot += self.plot_dupli_new

        if self.line_of_sight:
            self.plot_los_STEREO = k3d.voxels(self.Full_array(self.cubes_lineofsight_STEREO[0]), color_map=[0x0000ff], name='Seen from Stereo', **self.kwargs)
            self.plot_los_SDO = k3d.voxels(self.Full_array(self.cubes_lineofsight_SDO[0]), color_map=[0xff0000], name='Seen from SDO', **self.kwargs)
            self.plot += self.plot_los_STEREO + self.plot_los_SDO

        if self.time_intervals_all_data:
            if self.cube_version_0:
                self.plot_interv_init = k3d.voxels(self.Full_array(self.time_cubes_all_data_init[0]), color_map=[0xff6666], opacity=1, 
                                                   name=f'M1: all data for {self.time_interval}', **self.kwargs)
                self.plot += self.plot_interv_init   
            if self.cube_version_1:
                self.plot_interv_new = k3d.voxels(self.Full_array(self.time_cubes_all_data_new[0]), color_map=[0xff6666],opacity=1, 
                                                  name=f'M2: all data for {self.time_interval}', **self.kwargs)
                self.plot += self.plot_interv_new        
       
        if self.time_intervals_no_duplicates:
            if self.cube_version_0:
                self.plot_interv_dupli_init = k3d.voxels(self.Full_array(self.time_cubes_no_duplicates_init[0]), color_map=[0x0000ff], opacity=0.35,
                                                         name=f'M1: no duplicate for {self.time_interval}', **self.kwargs)
                self.plot += self.plot_interv_dupli_init
            if self.cube_version_1:
                self.plot_interv_dupli_new = k3d.voxels(self.Full_array(self.time_cubes_no_duplicates_new[0]), color_map=[0x0000ff], opacity=0.35,
                                                        name=f'M2: no duplicate for {self.time_interval}', **self.kwargs)
                self.plot += self.plot_interv_dupli_new      

        if self.trace_data:
            self.plot += k3d.voxels(self.Full_array(self.trace_cubes), color_map=[0xff6666], opacity=0.4, name='Total trace', **self.kwargs)
        
        if self.trace_no_duplicates:
            if self.cube_version_0:
                self.plot += k3d.voxels(self.Full_array(self.trace_cubes_no_duplicates_init), color_map=[0xff6666], opacity=0.4,
                                        name='M1: no duplicates trace', **self.kwargs)
            if self.cube_version_1:
                self.plot += k3d.voxels(self.Full_array(self.trace_cubes_no_duplicates_new), color_map=[0xff6666], opacity=0.4, 
                                        name='M2: no duplicates trace', **self.kwargs)
    
        if self.skeleton:
            self.plot_skeleton = k3d.voxels(self.Full_array(self.cubes_skeleton[0]), color_map=[0xff6e00], opacity=1, name='barycenter for the no dupliactes',
                                            **self.kwargs)
            self.plot += self.plot_skeleton

        if self.convolution:
            self.plot_convolution = k3d.voxels(self.Full_array(self.cubes_convolution[0]), color_map=[0xff6e00], opacity=0.5, name='conv3d', **self.kwargs)
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