"""Trying to see if I can fit a n-th order polynomial in the convolved (i.e. smooth banana shape)
Rainbow protuberance.
"""

# Imports 
import os
import re

import numpy as np
import matplotlib.pyplot as plt

from glob import glob
from queue import Queue
from typing import Callable
from astropy.io import fits
from scipy.io import readsav
from threading import Thread
from astropy import units as u
from typeguard import typechecked
from scipy.optimize import curve_fit
from scipy.ndimage import binary_erosion
from sparse import COO, stack, concatenate
from skimage.morphology import skeletonize
from multiprocessing import Process, Manager
from multiprocessing.queues import Queue as QUEUE
from multiprocessing.shared_memory import SharedMemory
from astropy.coordinates import CartesianRepresentation, SkyCoord
from astropy.coordinates import concatenate as astro_concatenate
from sunpy.coordinates.frames import HeliographicCarrington

from Animation_3D_main import CustomDate
from Common import Decorators, MultiProcessing, ClassDecorator


class BarycenterCreation:

    @typechecked
    def __init__(self, datatype: str | list = 'raw', polynomial_order: int | list | tuple = 3, integration_time: int | str = '24h', 
                 multiprocessing: bool = True, multiprocessing_multiplier: int = 2, threads: int = 5, multiprocessing_raw: int = 5,
                 feet_pos: tuple[tuple[int | float, int | float], ...] = ((-177, 15), (-163, -16)), foot_width: int | float = 3, 
                 solar_r: int | float = 6.96e5, verbose: int = 1, flush: bool = False, saving_with_feet: bool = False):
        
        # Arguments
        self.datatype = datatype if isinstance(datatype, list) else [datatype]  # what data is fitted - 'raw', 'rawContours'
        self.n = polynomial_order if not isinstance(polynomial_order, int) else [polynomial_order]  # the polynomial order
        self.time_interval_int = integration_time
        self.time_interval_str = integration_time
        self.multiprocessing = multiprocessing
        self.multiprocessing_multiplier = multiprocessing_multiplier
        self.threads = threads
        self.multiprocessing_raw = multiprocessing_raw
        self.feet_lonlat = feet_pos
        self.foot_width = foot_width
        self.solar_r = solar_r  #TODO: keep in mind that astropy seems to be using 6.957e5
        self.verbose = verbose
        self.flush = flush
        self.saving_feet = saving_with_feet
        
        # Attributes
        self.paths: dict[str, str]  # the different folder paths
        self.polynomial_list = [self.Generate_nth_order_polynomial(order)[0] for order in self.n]
        self.params_list = [np.random.rand(order + 1) for order in self.n]

        # Functions
        self.Paths()
        self.Time_interval()
        self.Loader()

    def Paths(self) -> None:
        """
        Path creation.
        """

        main_path = os.path.join(os.getcwd(), '..')
        self.paths = {
            'main': main_path, 
            'cubes': os.path.join(main_path, 'Cubes_karine'),
            'intensities': os.path.join(main_path, 'STEREO', 'int'),
            'cubes_feet': os.path.join(main_path, 'Cubes_karine_feet'),
            'save': os.path.join(main_path, 'curveFitArrays_with_feet'),
        }
        os.makedirs(self.paths['save'], exist_ok=True)
        os.makedirs(self.paths['cubes_feet'], exist_ok=True)
    
    def Time_interval(self) -> None:
        """
        Checking the date interval type an assigning a value to be used in another function to get the min and max date.
        """

        if isinstance(self.time_interval_int, int):
            time_delta = self.time_interval_int * 3600
        elif isinstance(self.time_interval_int, str):
            number = re.search(r'\d+', self.time_interval_int)
            number = int(number.group())
            self.time_interval_int = self.time_interval_int.lower()
            if 'min' in self.time_interval_int: 
                time_delta = number * 60
            elif 'h' in self.time_interval_int:
                time_delta = number * 3600
            elif 'd' in self.time_interval_int:
                time_delta = number * 3600 * 24
            else:
                raise ValueError("Time_interval string value not supported. Add 'min' for minutes, 'h' for hours or 'd' for days.")
        else:
            raise TypeError('Problem for time interval. Typeguard should have already raised an error.')
        self.time_interval_int = time_delta

    def Time_chunks(self, dates, cubes: np.ndarray | dict, date_max: int, date_min: int, shared_array: bool = False) -> COO:
        """
        To select the data in the time chunk given the data chosen for the integration.
        """

        if shared_array:
            shm = SharedMemory(name=cubes['shm.name'])
            coords = np.ndarray(cubes['data.shape'], dtype=cubes['data.dtype'], buffer=shm.buf)
            cubes = COO(coords=coords, data=1, shape=cubes['cubes.shape'])

        chunk = []
        for date, data in zip(dates, cubes):
            date_seconds = (((self.days_per_month[date.month] + date.day) * 24 + date.hour) * 60 + date.minute) * 60 + date.second
            
            if date_seconds < date_min:
                continue
            elif date_seconds <= date_max:
                chunk.append(data)
            else:
                break
        
        if shared_array: shm.close()

        if len(chunk) == 0:  # i.e. if nothing was found
            return COO(np.zeros(cubes.shape))
        elif len(chunk) == 1:
            return chunk[0]
        else:
            chunk = stack(chunk, axis=0)
            return COO.any(chunk, axis=0)
          
    def Generate_nth_order_polynomial(self, n: int = 3) -> tuple[Callable[[np.ndarray, any], np.ndarray], int]:
        """
        Creating a one variable n-th order polynomial.
        """
        
        def nth_order_polynomial(t: np.ndarray, *coeffs: int | float) -> np.ndarray:
            """
            The n-th order polynomial.
            """

            # Initialisation
            result = 0
            
            # Calculating the polynomial
            for order in range(n + 1): result += coeffs[order] * t**order 
            return result
        
        nb_coeffs = n + 1
        return nth_order_polynomial, nb_coeffs
    
    def Polynomial_derivative(self,  t: np.ndarray, coeffs: np.ndarray | list) -> np.ndarray:
        """
        Outputs the derivative of a n-th order polynomial given it's constants in order (i.e. order is a_0 + a_1 * t + ...)
        """

        # Initialisation
        n_order = len(coeffs) - 1
        result = 0

        for order in range(1, n_order + 1): result += order * coeffs[order] * t**[order-1]
        return result

    @Decorators.running_time
    def Loader(self) -> None:
        """
        To add the chosen data to the Main_structure function.
        The function is here to create polynomials for different data inputs. 
        """

        # Multiprocessing set up
        multiprocessing = True if (self.multiprocessing and len(self.datatype) > 1) else False
        processes = []

        # Loading the raw data
        data_raw = self.Getting_IDL_cubes()
        if multiprocessing: shm_raw, data_raw = MultiProcessing.shared_memory(data_raw)

        for datatype in self.datatype:
            kwargs = {
                'data': data_raw,
                'datatype': datatype,
                      }
            
            if 'Contours' in datatype: kwargs['contour'] = True

            if multiprocessing:
                processes.append(Process(target=self.Main_structure, kwargs=kwargs))
            else:
                self.Main_structure(**kwargs)

        if not multiprocessing: return

        for p in processes: p.start()
        for p in processes: p.join()
        shm_raw.unlink()

    @Decorators.running_time
    def File_opening_threads(self, filepaths: str) -> list[np.ndarray]:
        """
        Opening the .save files via threads
        """
        
        # Threading initialisation
        queue = Queue()
        threads = [None] * self.threads
        results = [None] * len(filepaths)

        # Creating a set number of threads and starting them
        for i in range(self.threads):
            t = Thread(target=self.File_reader, args=(queue, results))
            t.start()
            threads[i] = t

        # Populating the queue
        for index, filepath in enumerate(filepaths): queue.put((index, filepath))
        for _ in range(self.threads): queue.put(None)  # For braking the while loop inside the File_reader() function.
        queue.join()

        # Joining the threads
        for t in threads: t.join()
        return results

    def File_reader(self, queue: Queue[tuple[int, str]], results: list[tuple[int, str]]) -> None:
        """
        Opens a .save file and saves it in a queue.Queue().
        """

        while True:
            # Getting the queue inputs
            item = queue.get()
            if item is None: queue.task_done(); break
            
            # Setting up the file reading
            index, filepath = item
            try:
                data = readsav(filepath).cube.astype('uint8')
                results[index] = data
            except Exception as e:
                raise TypeError(f'Filename {os.path.basename(filepath)} read error: {e}')
            finally:
                queue.task_done()

    def Time_integration(self, dates: list, data: np.ndarray | dict, queue: QUEUE = None, index: int | None = None, step: int | None = None, 
                         shared_array: bool = False, last: bool = False) -> None | COO:
        """
        To integrate the cubes for a given time interval and the file dates in seconds.
        """

        # If multiprocessed the parent function.
        if shared_array:
            if not last:
                dates_section = dates[step * index:step * (index + 1)]
            else: 
                dates_section = dates[step * index:]
        else:
            dates_section = dates

        time_cubes_no_duplicate = []
        for date in dates_section:
            date_seconds = (((self.days_per_month[date.month] + date.day) * 24 + date.hour) * 60 + date.minute) * 60 + date.second

            date_min = date_seconds - self.time_interval_int / 2
            date_max = date_seconds + self.time_interval_int / 2
            time_cubes_no_duplicate.append(self.Time_chunks(dates=dates, cubes=data, date_max=date_max, date_min=date_min, shared_array=shared_array))
        time_cubes_no_duplicate = stack(time_cubes_no_duplicate, axis=0).astype('uint8')

        if not shared_array: return time_cubes_no_duplicate
        queue.put((index, time_cubes_no_duplicate))

    @Decorators.running_time
    def Getting_IDL_cubes(self) -> np.ndarray:
        """
        Opening the .save data files to get the 'raw' cube and doing the time integration.
        """

        # The filename patterns
        pattern_cubes = re.compile(r'cube(\d{3})\.save')  # for the cubes
        pattern_int = re.compile(r'\d{4}_(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2})\.\d{3}\.png')  # for the intensities

        # All filenames for the intensities
        all_filenames = sorted(glob(os.path.join(self.paths['intensities'], '*.png')))

        # Getting the cube names and opening the cubes
        cube_names = [cube_name for cube_name in os.listdir(self.paths['cubes']) if pattern_cubes.match(cube_name)]
        cube_names = sorted(cube_names) 
        self.cube_numbers = [int(pattern_cubes.match(cube_name).group(1)) for cube_name in cube_names]

        cubes = self.File_opening_threads([os.path.join(self.paths['cubes'], cube_name)
                                           for cube_name in cube_names])
        cubes = np.array(cubes, dtype='uint8')  
        cubes_no_duplicate = ((cubes & 0b00011000) == 24).astype('uint8')

        # Setting up a sparse data array
        cubes_no_duplicate = COO(cubes_no_duplicate)

        # Getting the corresponding filenames 
        filenames = []
        for number in self.cube_numbers: 
            for filepath in all_filenames:
                filename = os.path.basename(filepath)
                if filename[:4] == f'{number:04d}': filenames.append(filename); break
        dates = [CustomDate(pattern_int.match(filename).group(1)) for filename in filenames]
        self.days_per_month = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        date = dates[0]
        if (date.year % 4 == 0 and date.year % 100 !=0) or (date.year % 400 == 0):  # Only works if the year doesn't change
            self.days_per_month[2] = 29  # for leap years
        
        # Setting up the multiprocessing if needed
        multiprocessing = True if (self.multiprocessing and self.multiprocessing_raw > 1) else False

        if multiprocessing:

            coords = cubes_no_duplicate.coords
            shm = SharedMemory(create=True, size=coords.nbytes)
            shared_array = np.ndarray(coords.shape, dtype=coords.dtype, buffer=shm.buf)
            np.copyto(shared_array, coords)
            shm.close()

            cubes_no_duplicate = {
                'shm.name': shm.name,
                'data.shape': coords.shape,
                'cubes.shape': cubes_no_duplicate.shape,
                'data.dtype': coords.dtype,
                'cubes.dtype': cubes_no_duplicate.dtype,
            }

            # Multiprocessing initialisation
            processes = []
            manager = Manager()
            queue = manager.Queue()
    
            step = int(np.ceil(len(dates) / self.multiprocessing_raw))

            kwargs = {
                'queue': queue,
                'dates': dates,
                'data': cubes_no_duplicate,
                'step': step,
                'shared_array': True,
                }

            for i in range(self.multiprocessing_raw):
                if i != self.multiprocessing_raw - 1:
                    processes.append(Process(target=self.Time_integration, kwargs={'index': i, **kwargs}))
                else:
                    processes.append(Process(target=self.Time_integration, kwargs={'index': i, 'last': True, **kwargs}))
            
            for p in processes: p.start()
            for p in processes: p.join()

            results = [None] * self.multiprocessing_raw

            while not queue.empty():
                identifier, result = queue.get()
                results[identifier] = result
            results = concatenate(results, axis=0).astype('uint8')
        else:
            results = self.Time_integration(dates=dates, data=cubes_no_duplicate, shared_array=False)

        if self.verbose > 1: print(f'initial cubes - shape:{results.shape} - nnz:{results.nnz} - size:{round(results.nbytes / 2 ** 20, 2)}Mb')

        results, data_info = self.cubes_feet(results)

        if self.verbose> 0: print(f'cubes with feet - shape:{results.shape} - nnz:{results.nnz} - size:{round(results.nbytes / 2 ** 20, 2)}Mb')
        if self.saving_feet: self.saving_cubes(results, data_info)

        if multiprocessing: shm.unlink()
        return results.astype('uint16').todense()
    
    def saving_cubes(self, data:COO, data_info: dict[str, any]) -> None:
        """To save the cubes when the feet have been added.

        Args:
            data (COO): _description_

        Returns:
            _type_: _description_
        """

        coords = data.coords.astype('uint16')
        x_min = np.array(data_info['x_min'])
        y_min = np.array(data_info['y_min'])
        z_min = np.array(data_info['z_min'])
        shape = data_info['shape']
        dx = np.array([self.dx])
        unit = np.array(['km'])

        filename = f'sparse_cubes_with_feet.npz'
        np.savez(os.path.join(self.paths['cubes_feet'], filename), data=coords, x_min=x_min, y_min=y_min, z_min=z_min, shape=shape, dx=dx, unit=unit)

        np.savez_compressed(os.path.join(self.paths['cubes_feet'], 'compressed_' + filename), data=coords, x_min=x_min, y_min=y_min, z_min=z_min, 
                            shape=shape, dx=dx, unit=unit)
        
        if self.verbose > 0: print(f'File {filename} saved - main coords shape:{coords.shape} - dtype:{coords.dtype} - size:{round(coords.nbytes / 2 ** 20, 2)}Mb')

    @Decorators.running_time
    def cubes_feet(self, data: COO) -> COO:
        """Adding the feet to the cubes so that the interpolation is forced to pass through there. 
        Also, I would also need to define the feet as the end of the interpolation.

        Args:
            data (COO): _description_

        Returns:
            COO: _description_
        """

        # Set up to determine the number of values needed for the feet
        sum_per_slice = COO.sum(data, axis=(1, 2, 3)).todense()
        weight_per_foot = np.round(sum_per_slice / 18)  # number of points per foot for each slice as I want each feet to represent 5% of the final total data.

        # The data coords in heliographic cartesian coordinates
        skycoords = self.carrington_skycoords(data)
        
        # Getting the corresponding angle if dx is at one solar radius
        # d_theta = np.arccos(1 - dx**2 / (2 * solar_r**2))  
        d_theta = 2 * np.sin(self.dx / (2 * self.solar_r)) * 1.1  # * 1.1 just to make sure I have enough points as the foot rectangle is curved in cartesian space

        cubes_coords = [None] * len(weight_per_foot)
        for i, nb_points in enumerate(weight_per_foot):
            # Setting up the SkyCoord object
            foot_skycoords_0 = self.foot_grid_make(self.feet_lonlat[0], nb_points=nb_points, d_theta=d_theta)
            foot_skycoords_1 = self.foot_grid_make(self.feet_lonlat[1], nb_points=nb_points, d_theta=d_theta)
            skycoord_concat = astro_concatenate([skycoords[i], foot_skycoords_0, foot_skycoords_1])

            # Getting the position np.ndarray
            x = skycoord_concat.cartesian.x.value
            y = skycoord_concat.cartesian.y.value
            z = skycoord_concat.cartesian.z.value
            cube = np.stack([x, y, z], axis=0)

            # Setting up the 4D coords array
            time_row = np.full((1, cube.shape[1]), i)
            cubes_coords[i] = np.vstack((time_row, cube))
        cubes_coords = np.hstack(cubes_coords)

        # Set up to get the voxel coords
        _, x_min, y_min, z_min = np.min(cubes_coords, axis=1)
        cubes_coords[1, :] = (cubes_coords[1, :] - x_min) / self.dx
        cubes_coords[2, :] = (cubes_coords[2, :] - y_min) / self.dx
        cubes_coords[3, :] = (cubes_coords[3, :] - z_min) / self.dx
        cubes_coords = np.round(cubes_coords).astype('uint16')

        # Cleaning up any pixel that have the same position (if there are any)
        cubes_coords = np.unique(cubes_coords, axis=1).astype('uint16')

        # Creation of the corresponding COO object
        shape = np.max(cubes_coords, axis=1) + 1

        # Saving the data 'metadata'
        data_info = {
            'shape': shape,
            'x_min': x_min,
            'y_min': y_min,
            'z_min': z_min,
        }
        return COO(coords=cubes_coords, data=1, shape=shape), data_info

    def foot_grid_make(self, foot_lonlat: tuple[int | float], nb_points: int, d_theta: float):
        """To create the positions of the voxel in carrington space for each foot

        Args:
            foot_pos (_type_): _description_
        """

        d_deg = np.rad2deg(d_theta)
        # print(f'd_deg is {d_deg}')

        lon_values = np.arange(foot_lonlat[0] - self.foot_width / 2, foot_lonlat[0] + self.foot_width / 2 + d_deg*0.1, d_deg)
        lat_values = np.arange(foot_lonlat[1] - self.foot_width / 2, foot_lonlat[1] + self.foot_width / 2 + d_deg*0.1, d_deg)

        # print(f'lon, lat shape is {lon_values.shape}, {lat_values.shape}')
        # print(f'number of points is {nb_of_points}')

        # Setting 68% of the values to be in the middle quarter of the grid
        gaussian_std = self.foot_width / 2
        gaussian_distribution = np.exp( - ((lon_values[:, None] - foot_lonlat[0])**2 + (lat_values[None, :] - foot_lonlat[1])**2) / (2 * gaussian_std**2))
        
        # print(f'initial gaussian distribution max is {gaussian_distribution.max()}')
        # Resetting the gaussian so the sum of its values is equal to nb_of_points
        gaussian_distribution = nb_points * gaussian_distribution / np.sum(gaussian_distribution)

        # Changing the 2D distribution to a 3D representation of ones or zeros to be able to only use COO.coords later
        max_val = np.round(gaussian_distribution).max().astype(int)

        gaussian_distribution_3D = np.zeros((gaussian_distribution.shape[0], gaussian_distribution.shape[1], max_val), dtype=int)
        for value in range(max_val): gaussian_distribution_3D[:, :, value] = (gaussian_distribution >= value + 1)

        # Swapping the distribution values to the corresponding carrington heliographic coordinates
        foot_positions = COO(gaussian_distribution_3D).coords
        foot_positions[0, :] = np.deg2rad(foot_positions[0, :] * d_deg + foot_lonlat[0])
        foot_positions[1, :] = np.deg2rad(foot_positions[1, :] * d_deg + foot_lonlat[1])
        foot_positions[2, :] = foot_positions[2, :] * self.dx + self.solar_r

        # print(f'gaussian 3D distribution shape is {gaussian_distribution_3D.shape}')

        # Creating the corresponding skycoord object
        coords = SkyCoord(foot_positions[0, :] * u.rad, foot_positions[1, :] * u.rad, foot_positions[2, :] * u.km, frame=HeliographicCarrington)
        cartesian = coords.represent_as(CartesianRepresentation)
        return SkyCoord(cartesian, frame=coords.frame, representation_type='cartesian')

    @Decorators.running_time
    def carrington_skycoords(self, data: COO) -> list[SkyCoord]:
        """
        To get the cartesian coordinates of every points in the cubes.
        """

        cubes_sparse_coords = data.coords.astype('float64')

        # Initialisation in cartesian carrington heliographic coordinates
        first_cube = readsav(os.path.join(self.paths['cubes'], f'cube{self.cube_numbers[0]:03d}.save'))
        self.dx = first_cube.dx  # in km
        self.x_min = first_cube.xt_min  # in km
        y_min = first_cube.yt_min
        z_min = first_cube.zt_min

        cubes_sparse_coords[1, :] = cubes_sparse_coords[1, :] * self.dx + self.x_min
        cubes_sparse_coords[2, :] = cubes_sparse_coords[2, :] * self.dx + y_min
        cubes_sparse_coords[3, :] = cubes_sparse_coords[3, :] * self.dx + z_min

        # Creating the SkyCoord object for each 3D slice
        time_indexes = list(set(cubes_sparse_coords[0, :]))
        sky_coords_list = [None] * len(time_indexes)
        for i, time in enumerate(time_indexes):
            time_filter = cubes_sparse_coords[0, :] == time
            sparse_coords_section = cubes_sparse_coords[:, time_filter]
            sky_coords = SkyCoord(sparse_coords_section[1, :], sparse_coords_section[2, :], sparse_coords_section[3, :], 
                                unit=u.km, frame=HeliographicCarrington, representation_type='cartesian')
            sky_coords_list[i] = sky_coords
        return sky_coords_list

    @Decorators.running_time
    def Main_structure(self, datatype: str, data: np.ndarray | dict, contour: bool = False) -> None:

        # Defining the reference if using shared arrays
        multiprocessing = True if (self.multiprocessing and len(self.n) > 1) else False
        kwargs = {
            'datatype': datatype,
            'contour': contour,
        }

        if multiprocessing:
            if not isinstance(data, dict): 
                shm, data = MultiProcessing.shared_memory(data)  
            else:
                shm = None

            processes = [Process(target=self.Time_multiprocessing, kwargs={'index': i, 'data': data, **kwargs}) for i in range(len(self.n))]
            for p in processes: p.start()
            for p in processes: p.join()

            if shm is not None: shm.unlink()
        else:
            if isinstance(data, dict):
                shm = SharedMemory(name=data['shm.name'])
                data = np.ndarray(data['data.shape'], dtype=data['data.dtype'], buffer=shm.buf)
                data = np.copy(data)

                shm.close()          
            for i in range(len(self.n)): self.Time_multiprocessing(index=i, data=data, **kwargs)

    def Time_multiprocessing(self, data: np.ndarray | dict, datatype: str, index: int, contour: bool = False) -> None:
        """
        The for loop on the time axis.
        """

        # Initialisation of the multiprocessing
        multiprocessing = True if (self.multiprocessing and self.multiprocessing_multiplier > 1) else False

        if multiprocessing:
            if not isinstance(data, dict): 
                shm, data = MultiProcessing.shared_memory(data)
                indexes = MultiProcessing.pool_indexes(data['data.shape'][0], self.multiprocessing_multiplier)
            else: 
                shm = None
                indexes = MultiProcessing.pool_indexes(data.shape[0], self.multiprocessing_multiplier)

            # Initialisation
            processes = []
            manager = Manager()
            queue = manager.Queue()

            # Step an multiprocessing kwargs
            kwargs = {
                'queue': queue,
                'data': data,
                'contour': contour,
                'poly_index': index,
            }

            processes = [Process(target=self.Time_loop, kwargs={'index': i, 'data_index': index, **kwargs}) for i, index in enumerate(indexes)]
            for p in processes: p.start()
            for p in processes: p.join()

            if shm is not None: shm.unlink()

            results = [None] * self.multiprocessing_multiplier
            while not queue.empty():
                identifier, result = queue.get()
                results[identifier] = result
            results = np.concatenate(results, axis=1)
        else:
            results = self.Time_loop(data=data, data_index=(0, data.shape[0] - 1), contour=contour, poly_index=index)

        # Saving the data in .npy files
        filename = f'poly_{datatype}_order{self.n[index]}_{self.time_interval_str}.npy'
        np.save(os.path.join(self.paths['save'], filename), results.astype('float64'))

        if self.verbose > 0:
            print(f'File {filename} saved - shape:{results.shape} - dtype:{results.dtype} - size:{round(results.nbytes / 2 ** 20, 2)}Mb', flush=self.flush)

    def Time_loop(self, data: np.ndarray | dict, poly_index: int, data_index: tuple[int, int], index: int | None = None, queue: None | QUEUE = None,
                  contour: bool = False) -> None | np.ndarray:
        """
        The for loop on the time axis.
        """

        if isinstance(data, dict):
            shm = SharedMemory(name=data['shm.name'])
            data = np.ndarray(data['data.shape'], dtype=data['data.dtype'], buffer=shm.buf)
        else: 
            shm = None

        # Initialisation of the result list
        results = [None] * (data_index[1] - data_index[0] + 1)
        for time in range(data_index[0], data_index[1] + 1): 
            # Selecting a slice of the data
            section = data[time]

            if contour:
                # Prepocessing to only keep the contours
                eroded_data = binary_erosion(section)
                section = section.astype('uint8') - eroded_data.astype('uint8')

            # Setting up the data
            x, y, z = np.where(section)
            points = np.stack((x, y, z), axis=1)

            # Calculating the cumulative distance
            t = np.zeros(points.shape[0])
            for i in range(1, points.shape[0]): t[i] = t[i-1] + np.linalg.norm(points[i] - points[i - 1])
            t /= t[-1]  # normalisation
            t += 1

            results[time - data_index[0]] = self.Fitting_polynomial(time=time, t=t, data=points, index=poly_index, first_time_index=data_index[0])
        results = np.concatenate(results, axis=1)

        if shm is not None: shm.close()
        if queue is None: return results  # if no multiprocessing
        queue.put((index, results))

    def Fitting_polynomial(self, time: int, t: np.ndarray, data: np.ndarray | dict, index: int, first_time_index: int = 0) -> np.ndarray:
            """
            Where the fitting of the curve actually takes place.
            """

            # Getting the data ready
            x, y, z = data[:, 0], data[:, 1], data[:, 2]
            polynomial = self.polynomial_list[index]
            params = self.params_list[index]

            # Finding the best parameters
            params_x, _ = curve_fit(polynomial, t, x, p0=params)
            params_y, _ = curve_fit(polynomial, t, y, p0=params)          
            params_z, _ = curve_fit(polynomial, t, z, p0=params)     

            # Getting the curve
            t_fine = np.linspace(0.5, 2.5, 10**6)
            x = polynomial(t_fine, *params_x)
            y = polynomial(t_fine, *params_y)
            z = polynomial(t_fine, *params_z)

            # Taking away duplicates 
            data = np.vstack((x, y, z)).T.astype('float64')

            # Cutting away the values that are outside the initial cube shape
            conditions_upper = (data[:, 0] >= 318) | (data[:, 1] >= 225) | (data[:, 2] >= 185) 
            conditions_lower = np.any(data < 0, axis=1)
            conditions = conditions_upper | conditions_lower
            data = data[~conditions]       

            unique_data = np.unique(data, axis=0).T
            time_row = np.full((1, unique_data.shape[1]), time + first_time_index)
            unique_data = np.vstack((time_row, unique_data)).astype('float64')

            params = np.array([params_x, params_y, params_z], dtype='float64')  # array of shape (3, n_order + 1)
            return unique_data


# @ClassDecorator(Decorators.running_time)
class OrthographicalProjection:
    """
    Does the 2D projection of a 3D volume.
    Used to recreate what is seen by SDO when looking at the cube, especially the curve fits.
    Also, while not yet implemented, will also add the envelop around the projection.
    """

    @typechecked
    def __init__(self, data: np.ndarray | COO | None = None, processes: int = 0, saving_data: bool = False, time_interval: str = '24h',
                 saving_plots: bool = False, saved_data: bool = False, saving_filename: str = 'k3d_projection_cubes.npy',
                 plot_choices: str | list[str] = ['polar', 'sdo image', 'no duplicate', 'envelop', 'polynomial']):
        
        # Data arguments
        if isinstance(data, COO):
            data = data.coords
        elif data.shape[0] not in [3, 4]:
            data = COO(data).coords

        # Arguments
        self.time_interval = time_interval
        self.multiprocessing = True if processes > 1 else False
        self.processes = processes
        self.saving_filename = os.path.splitext(saving_filename)[0] + f'_{self.time_interval}.npy'
        self.solar_r = 6.96e5  # in km
        self.plot_choices = self.Plot_choices(plot_choices if isinstance(plot_choices, list) else [plot_choices])

        # Input print
        if data is not None: print(f"The input data shape is {data.shape} with size {round(data.nbytes / 2**20, 2)}Mb.")

        # Functions
        self.Paths()
        self.Important_attributes()
        self.Choices(data, saving_data, saving_plots, saved_data)

    def Plot_choices(self, plot_choices: list[str]) -> dict[str, bool]:
        """
        To check the values given for the plot_choices argument
        """

        # Initialisation of the possible choices
        possibilities = ['polar', 'cartesian', 'sdo image', 'no duplicate', 'envelop', 'polynomial']
        plot_choices_kwargs = {key: False for key in possibilities}

        for key in plot_choices: 
            if key in possibilities: 
                plot_choices_kwargs[key] = True
            else: 
                raise ValueError(f"Value for the 'plot_choices' argument not recognised.") 
            
        if 'envelop' in plot_choices.keys(): plot_choices_kwargs['polar'] = True
        return plot_choices_kwargs

    def Paths(self) -> None:
        """
        Function where the filepaths dictionary is created.
        """

        main_path = os.path.join(os.getcwd(), '..')
        self.paths = {
            'main': main_path, 
            'SDO': os.path.join(main_path, 'sdo'),
            'cubes': os.path.join(main_path, 'Cubes_karine'),
            'curve fit': os.path.join(main_path, 'curveFitArrays'),
            'save': os.path.join(main_path, 'projection_results'),
        }
        os.makedirs(self.paths['save'], exist_ok=True)
    
    def Important_attributes(self) -> None:
        """
        To create the values of some of the class attributes.
        """

        # Initialisation
        cube_pattern = re.compile(r'cube(\d{3})\.save')

        # Important lists
        self.numbers = sorted([
            int(cube_pattern.match(cube_name).group(1))
            for cube_name in os.listdir(self.paths['cubes'])
            if cube_pattern.match(cube_name)
        ])
        self.SDO_filepaths = [os.path.join(self.paths['SDO'], f'AIA_fullhead_{number:03d}.fits.gz')
                              for number in self.numbers]
    
    def Choices(self, data: np.ndarray | None, saving_data: bool, saving_plots: bool, saved_data: bool) -> None:
        """
        To call the right class functions depending on the class arguments.
        The function is here just to not overpopulate __init__.
        """

        if not saved_data:
            self.SDO_pos()
            data = self.Cartesian_pos(data)
            data = self.Matrix_rotation(data)
            if saving_data: self.Saving_data(data)
        else:
            data = np.load(os.path.join(self.paths['save'], self.saving_filename))

        if saving_plots: self.Plotting(data)  

    def Cartesian_pos(self, data: np.ndarray) -> np.ndarray:
        """
        To calculate the heliographic cartesian positions of some of the objects.
        """

        cubes_sparse_coords = data.astype('float64')

        # Initialisation
        first_cube = readsav(os.path.join(self.paths['cubes'], f'cube{self.numbers[0]:03d}.save'))
        dx = first_cube.dx  # in km
        dy = first_cube.dy
        dz = first_cube.dz
        x_min = first_cube.xt_min  # in km
        y_min = first_cube.yt_min
        z_min = first_cube.zt_min
        cubes_sparse_coords[1, :] = (cubes_sparse_coords[1, :] * dx + x_min) 
        cubes_sparse_coords[2, :] = (cubes_sparse_coords[2, :] * dy + y_min) 
        cubes_sparse_coords[3, :] = (cubes_sparse_coords[3, :] * dz + z_min)
   
        self.heliographic_cubes_origin = np.array([x_min, y_min, z_min], dtype='float64')
        return cubes_sparse_coords

    def SDO_pos(self) -> None:
        """
        To get the position of the SDO satellite.
        """
        # Multithreading initialisation
        if self.multiprocessing:
            manager = Manager()
            queue = manager.Queue()
            indexes = MultiProcessing.pool_indexes(len(self.SDO_filepaths), self.processes)

            processes = [Process(target=self.SDO_pos_sub, kwargs={'queue':queue, 'index': index, 'data_index': data_index}) for index, data_index in enumerate(indexes)]
            for p in processes: p.start()
            for p in processes: p.join()

            results = [None] * self.processes
            while not queue.empty():
                identifier, result = queue.get()
                results[identifier] = result
            self.sdo_pos = [one_result for sub_results in results for one_result in sub_results]
        else:
            self.sdo_pos = self.SDO_pos_sub((0, len(self.SDO_filepaths) - 1))
    
    def SDO_pos_sub(self, data_index: tuple[int, int], queue: QUEUE | None = None, index: int | None = None) -> None:
        """
        For the multithreading when opening the files
        """
        
        sdo_pos = []
        for filepath in self.SDO_filepaths[data_index[0]:data_index[1] + 1]:
            header = fits.getheader(filepath)
            hec_coords = HeliographicCarrington(header['CRLN_OBS']* u.deg, header['CRLT_OBS'] * u.deg, 
                                                header['DSUN_OBS'] * u.km, obstime=header['DATE-OBS'], observer='self')
            hec_coords = hec_coords.represent_as(CartesianRepresentation)

            Xhec = hec_coords.x.value 
            Yhec = hec_coords.y.value 
            Zhec = hec_coords.z.value
            sdo_pos.append((Xhec, Yhec, Zhec))
        
        if self.multiprocessing: queue.put((index, sdo_pos))
        return sdo_pos

    def Matrix_rotation(self, data: np.ndarray) -> np.ndarray:
        """
        To rotate the matrix so that it aligns with the satellite pov
        """

        if self.multiprocessing:
            # Initialisation of the multiprocessing
            manager = Manager()
            queue = manager.Queue()
            shm, data = MultiProcessing.shared_memory(data.astype('float64'))

            # Setting up each process
            indexes = MultiProcessing.pool_indexes(len(self.numbers), self.processes)
            kwargs = {
                'queue': queue, 
                'data': data,
            }
            processes = [Process(target=self.Time_loop, kwargs={'index': i, 'data_index': index_tuple, **kwargs}) for i, index_tuple in enumerate(indexes)]
            for p in processes: p.start()
            for p in processes: p.join()
            
            shm.unlink()

            # Getting the results 
            results = [None] * self.processes
            while not queue.empty():
                identifier, result = queue.get()
                results[identifier] = result
            results = [projection_matrix for sublist in results for projection_matrix in sublist]
        else:
            results = self.Time_loop(data=data, data_index=(0, len(self.numbers) - 1))

        # Ordering the final result so that it is a np.ndarray
        start_index = 0
        total_nb_vals = sum(arr.shape[1] for arr in results)
        final_results = np.empty((4, total_nb_vals), dtype='float64')
        for t, result in enumerate(results):
            nb_columns = result.shape[1]
            final_results[0, start_index: start_index + nb_columns] = t
            final_results[1:4, start_index: start_index + nb_columns] = result
            start_index += nb_columns
        return final_results

    def Time_loop(self, data: np.ndarray | dict, data_index: tuple[int, int], index: int = 0, queue: QUEUE | None = None) -> None | list[np.ndarray]:
        """
        Loop over the time indexes so that I can multiprocess if needed be.
        """

        if self.multiprocessing:
            shm = SharedMemory(name=data['shm.name'])
            data = np.ndarray(data['data.shape'], dtype=data['data.dtype'], buffer=shm.buf)

        result_list = []
        for time in range(data_index[0], data_index[1] + 1):
            data_filter = data[0, :] == time
            result = data[1:4, data_filter]
            center = np.array([0, 0, 0]).reshape(3, 1)  # TODO: need to change this when the code works. This is only to see where the sun center is
            result = np.column_stack((result, center))  # adding the sun center to see if the translation is correct
            satelitte_pos = self.sdo_pos[time]

            # Centering the SDO pov on the Sun center
            sun_center = np.array([0, 0, 0])
            # Defining the view direction vector
            viewing_direction = sun_center - satelitte_pos
            viewing_direction_norm = viewing_direction / np.linalg.norm(viewing_direction)

            # Defining the up vector and the normal vector to the projection
            up_vector = np.array([0, 0, 1])
            target_axis = np.array([0, 0, 1])

            # Axis of rotation and angle
            rotation_axis = np.cross(viewing_direction_norm, target_axis)
            cos_theta = np.dot(viewing_direction_norm, target_axis)
            theta = np.arccos(cos_theta)

            # Corresponding rotation matrix
            rotation_matrix = self.Rodrigues_rotation(rotation_axis, -theta)

            # up_vector in the new rotated matrix
            up_vector_rotated = np.dot(rotation_matrix, up_vector)
            theta = np.arctan2(up_vector_rotated[0], up_vector_rotated[1])
            up_rotation_matrix = self.Rodrigues_rotation(target_axis, theta)
            
            # Final rotation matrix
            rotation_matrix = np.matmul(up_rotation_matrix, rotation_matrix)

            result = [np.matmul(rotation_matrix, point) for point in result.T]
            result = np.stack(result, axis=-1)
            result_list.append(result)
        if not self.multiprocessing: return result_list
        shm.close()
        queue.put((index, result_list))
    
    def Rodrigues_rotation(self, axis: np.ndarray, angle: float) -> np.ndarray:
        """
        The Rodrigues's rotation formula to rotate matrices.
        """

        # Normalisation
        axis = axis / np.linalg.norm(axis)

        matrix = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        return np.eye(3) + np.sin(angle) * matrix + (1 - np.cos(angle)) * (matrix @ matrix)

    def Saving_data(self, data: np.ndarray) -> None:
        """
        Function to save the reprojection arrays.
        The arrays saved have shape (4, n) as the whole 3D projection is saved for each time step (maybe the viewpoint depth might be useful later on).
        """

        np.save(os.path.join(self.paths['save'], self.saving_filename), data.astype('float64'))

        # STATS print
        print(f"Saved array. Array shape is {data.shape} with size {round(data.nbytes / 2**20, 2)}Mb.")

    def Plotting(self, data: np.ndarray) -> None:
        """
        Function to plot the data.
        """

        if self.multiprocessing:
            shm, data = MultiProcessing.shared_memory(data) 
            indexes = MultiProcessing.pool_indexes(len(self.numbers), self.processes)
            processes = [Process(target=self.Plotting_sub, kwargs={'data': data, 'data_index': index_tuple}) for index_tuple in indexes]
            for p in processes: p.start()
            for p in processes: p.join()
            shm.unlink()
        else:
            self.Plotting_sub(data=data, data_index=(0, len(self.numbers) - 1))

    def Plotting_sub(self, data: np.ndarray | dict, data_index: tuple[int, int]) -> None:
        """
        To be able to multiprocess the plotting.
        """
        
        if self.multiprocessing:
            shm = SharedMemory(name=data['shm.name'])
            data = np.ndarray(data['data.shape'], dtype=data['data.dtype'], buffer=shm.buf)

        if self.plot_choices['envelop']: self.Envelop_preprocessing()

        for time in range(data_index[0], data_index[1] + 1):
            data_filter  = data[0, :] == time
            result = data[1:4, data_filter]

            # Voxel positions
            x, y, _ = result
            image_nb = self.numbers[time]

            if self.plot_choices['sdo image']: self.SDO_image(index=time)

            if self.plot_choices['cartesian']:
                # SDO projection plotting
                plt.figure(figsize=(5, 5))
                plt.scatter(x / self.solar_r, y / self.solar_r, s=0.7)
                plt.title(f'SDO POV for image nb{image_nb}')
                plt.xlabel('Solar X [au]')
                plt.ylabel('Solar Y [au]')
                plot_name = f'sdoprojection_{image_nb:03d}_{self.time_interval}.png'
                plt.savefig(os.path.join(self.paths['save'], plot_name), dpi=500)
                plt.close()

            if self.plot_choices['polar']:
                # Changing to polar coordinates
                r = np.sqrt(x**2 + y**2)
                theta = np.arctan2(y, x) - np.pi / 2
                theta = np.where(theta < 0, theta + 2 * np.pi, theta)
                theta = 2 * np.pi - theta  # clockwise
                theta = np.where(theta >= 2 * np.pi, theta - 2 * np.pi, theta)  # modulo 2pi
                theta = np.degrees(theta) 

                # SDO polar projection plotting
                plt.figure(figsize=(12, 5))
                plt.scatter(theta, r / 10**3, s=0.7)
                plt.xlim(245, 295)
                plt.ylim(700, 870)
                plt.title(f'SDO polar projection: {image_nb}')
                plt.xlabel('Polar angle [degrees]')
                plt.ylabel('Radial distance [Mm]')
                plot_name = f'sdopolarprojection_{image_nb:03d}_{self.time_interval}.png'
                plt.savefig(os.path.join(self.paths['save'], plot_name), dpi=500)
                plt.close()

            print(f'Plot nb{image_nb} finished.')

    def Envelop_preprocessing(self):
        """
        Opens the two png images of the envelop in polar coordinates. Then, treats the data to use it in
        the polar plots.
        """

        pass

    def SDO_image(self, index: int):
        """
        To open the SDO image data, preprocess it and return it as an array for use in plots.
        """

        image = fits.getdata(self.SDO_filepaths[index], 1)
        pass # TODO: will do it later as I need to take into account CRPIX1 and CRPIX2 but also conversion image to plot values

class TESTING_STUFF:
    """
    Just to test the rotation matrix to see if it is doing it's job properly.
    """
    
    @typechecked
    def __init__(self, coords: np.ndarray | None = None, data: np.ndarray | None = None, camera_position: np.ndarray | tuple | list = (1, 1, 1), 
                 center_of_interest: np.ndarray | tuple | list = (0, 0, 0), camera_fov: int | float | str = '1deg'):
    
        # Data arguments
        self.data = self.Random_data()

        # Camera arguments
        self.center_of_interest = np.stack(center_of_interest, axis=0) if isinstance(center_of_interest, (tuple, list)) else center_of_interest
        self.camera_position = np.stack(camera_position, axis=0) if isinstance(camera_position, (tuple, list)) else camera_position  # TODO: need to check for when the camera is a list but only the x, y, z, value
        self.camera_fov = camera_fov

        # Functions
        # self.Checks()  # checks if the arguments had the expected shapes
        self.Paths()
        self.Important_attributes()

    def Random_data(self) -> np.ndarray:
        """
        making data up
        """

        data = np.zeros((4, 220 * 3))
        data[0, ::3] = np.arange(0, 220)
        data[1, ::3] = 1
        data[2, ::3] = 1
        data[3, ::3] = 1
        data[0, 1::3] = np.arange(0, 220)
        data[1, 1::3] = 0
        data[2, 1::3] = 0
        data[3, 1::3] = 0
        data[0, 2::3] = np.arange(0, 220)
        data[1, 2::3] = 1
        data[2, 2::3] = 0
        data[3, 2::3] = 0
        # print(f'random data shape is {data.shape}')
        return data

    def Paths(self) -> None:
        """
        Function where the filepaths dictionary is created.
        """

        main_path = os.path.join(os.getcwd(), '..')
        self.paths = {
            'main': main_path, 
            'SDO': os.path.join(main_path, 'sdo'),
            'cubes': os.path.join(main_path, 'Cubes_karine'),
            'curve fit': os.path.join(main_path, 'curveFitArrays'),
        }
    
    def Matrix_rotation(self) -> list[np.ndarray]:
        """
        To rotate the matrix so that it aligns with the satellite pov
        """
        data_list = []
        satelitte_poss = np.array([
            [5, 5, 5],
            [8, 8, 8],
            [1, 0, 0],
            [0, 1, 0], 
            [10, 20, 30],
        ])
        for i, test_time in enumerate([0, 5, 40, 90, 200]):
            # data_filter = self.cartesian_data_coords[0, :] == test_time
            data_filter = self.data[0, :] == test_time

            # data = self.cartesian_data_coords[1:4, data_filter]
            data = self.data[1:4, data_filter]
            # satelitte_pos = self.sdo_pos[test_time]
            satelitte_pos = satelitte_poss[i]
            print(f'satellite pos is {satelitte_pos}')
            # print(f'data is {data} with shape {data.shape}')
            # print(f'satellite_pos shape is {satelitte_pos.shape}')
            sun_center = np.array([0, 0, 0])

            viewing_direction = sun_center - satelitte_pos
            viewing_direction_norm = viewing_direction / np.linalg.norm(viewing_direction)

            # Axis to center on the pov
            target_axis = np.array([0, 0, 1])

            # Axis of rotation
            rotation_axis = np.cross(viewing_direction_norm, target_axis)

            # Angle of rotation
            cos_theta = np.dot(viewing_direction_norm, target_axis)
            theta = np.arccos(cos_theta)

            # Corresponding rotation matrix
            rotation_matrix = self.Rodrigues_rotation(rotation_axis, theta)


            # As the data has original shape (3, n) for each cube I need to change it to (n,3) for the rotation
            data = data.T

            # print(f'data transpose shape is {data.shape}')
            # print(f'rotation matrix shape is {rotation_matrix.shape}')
            nw_data = [np.matmul(rotation_matrix, point) for point in data]
            nw_data = np.stack(nw_data, axis=-1)
            # print(f"new_data shape is {nw_data.shape}")
            data = nw_data
            # rotated_data = np.dot(data, rotation_matrix.T)
            # data = rotated_data.T
            data_list.append(data)
        return data_list
    
    def Rodrigues_rotation(self, axis: np.ndarray, angle: float) -> np.ndarray:
        """
        The Rodrigues's rotation formula to rotate matrices.
        """

        # Normalisation
        axis = axis / np.linalg.norm(axis)

        matrix = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        return np.eye(3) + np.sin(angle) * matrix + (1 - np.cos(angle)) * (matrix @ matrix)

    def Important_attributes(self) -> None:
        """
        To create the values of some of the class attributes.
        """

        # Initialisation
        cube_pattern = re.compile(r'cube(\d{3})\.save')

        # Class attributes
        self.numbers = sorted([
            int(cube_pattern.match(cube_name).group(1))
            for cube_name in os.listdir(self.paths['cubes'])
            if cube_pattern.match(cube_name)
        ])
        self.SDO_filepaths = [os.path.join(self.paths['SDO'], f'AIA_fullhead_{number:03d}.fits.gz')
                              for number in self.numbers]

    def SDO_pos(self) -> None:
        """
        To get the important info out of the SDO data.
        """
        
        # Opening the fits to get the data
        sdo_pos = []
        for filepath in self.SDO_filepaths:
            header = fits.getheader(filepath)

            hec_coords = HeliographicCarrington(header['CRLN_OBS']* u.deg, header['CRLT_OBS'] * u.deg, 
                                                header['DSUN_OBS'] * u.m, obstime=header['DATE-OBS'], 
                                                observer='self')
            hec_coords = hec_coords.represent_as(CartesianRepresentation)

            Xhec = hec_coords.x.value / (1000)
            Yhec = hec_coords.y.value / (1000)
            Zhec = hec_coords.z.value / (1000)

            sdo_pos.append((Xhec, Yhec, Zhec))
        self.sdo_pos = sdo_pos
    
    def Saving_arrays(self, data: np.ndarray):
        """
        Function to save the reprojection arrays.
        The arrays saved have shape (4, n) as the whole 3D projection is saved for each time step (maybe the viewpoint depth might be useful later on).
        """

        results = self.Matrix_rotation(data)
        filename = "reprojection_sdo_pov.npy"
        np.save(os.path.join(self.paths['save'], filename), results)



if __name__=='__main__':
    BarycenterCreation(datatype=['raw'], 
                         polynomial_order=[6],
                         integration_time='24h',
                         multiprocessing=False,
                         multiprocessing_multiplier=6, 
                         multiprocessing_raw=2,
                         saving_with_feet=True, 
                         verbose=2)
    
    # interpolations_filepath = os.path.join(os.getcwd(), '..', 'curveFitArrays', 'poly_raw_order6_24h.npy')
    # data = np.load(interpolations_filepath)
    # data = data[[0, 3, 2, 1]]  # TODO: will need to change this when I recreate the polynomial arrays taking into account the axis swapping whe opening a .save

    # Projection_test = OrthographicalProjection(data=data, saved_data=True, processes=12, saving_plots=True, time_interval='24h')