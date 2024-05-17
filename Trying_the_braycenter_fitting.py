"""
Trying to see if I can fit a n-th order polynomial in the convolved (i.e. smooth banana shape)
Rainbow protuberance.
"""

# Imports 
import os
import numpy as np

from astropy.io import fits
from astropy import units as u
from astropy.coordinates import SkyCoord
from typeguard import typechecked
from scipy.optimize import curve_fit
from skimage.morphology import skeletonize
from astropy.coordinates import CartesianRepresentation
from sunpy.coordinates.frames import  HeliographicCarrington

from common_alf import MathematicalEquations, Decorators


class BarycenterCreation:

    @typechecked
    def __init__(self, n_oder: int = 3, conv_threshold: int = 125, test_time_index: int = 200):
        
        pass

    def Main_structure(self):
        """
        Class' main function.
        """

        # Importing the data
        cubes = np.load(os.path.join(self.path, 'barycenter_array_4.npy')).astype('uint8')
        binary_cubes = cubes > self.conv_thresholds
        # Testing for a specific value
        test_section = binary_cubes[self.test_time_index]
        x, y, z = np.where(test_section==1)
        w = np.ones(len(x))
        coords = (x, y, z)

        # Creating the n-th order cartesian polynomial
        nth_order_polynomial, nb_coeffs = MathematicalEquations.Generate_nth_order_polynomial(self.n_order)

        # Initial random guess for the polynomial coefficients
        constants = np.random.rand(nb_coeffs)

        params, params_covariance = curve_fit(nth_order_polynomial, coords, w, p0=constants)

        # Setting up the index grid to get the polynomial 'solutions'
        x_range = np.arange(0, cubes.shape[1])
        y_range = np.arange(0, cubes.shape[2])
        z_range = np.arange(0, cubes.shape[3])
        X, Y, Z = np.meshgrid(x_range, y_range, z_range, indexing='ij')

        # Creating the results cube
        mesh_coords = X, Y, Z
        result_cube = nth_order_polynomial(mesh_coords, *params)
        
        # Normalisation of the results
        result_cube = (result_cube - np.min(result_cube)) / (np.max(result_cube - np.min(result_cube)))
        result_cube = (np.abs(result_cube) * 255).astype('uint16')
        np.save(os.path.join(self.path, f'polynomial_results_convlim{self.conv_thresholds}.npy'), result_cube)

from scipy.ndimage import binary_erosion
from multiprocessing import Process, shared_memory, Manager
from scipy.io import readsav
import re
from glob import glob
from Animation_3D_main import CustomDate
import threading
import queue as Queue
from sparse import COO, stack, concatenate

class BarycenterCreation_4:

    @typechecked
    def __init__(self, datatype: str | list = 'surface', conv_threshold: int | list | tuple = 125, polynomial_order: int | list | tuple = 3, 
                 integration_time: int | str = '24h', multiprocessing: bool = True, multiprocessing_multiplier: int = 2, threads: int = 5,
                 multiprocessing_raw: int = 5):
        
        # Arguments
        self.datatype = datatype if not isinstance(datatype, str) else [datatype]  # what data is fitted - 'conv3dAll', 'conv3dSkele', 'raw', 'rawContours'
        self.conv_thresholds = conv_threshold if not isinstance(conv_threshold, int) else [conv_threshold]  # minimum threshold to convert the conv3d data to binary
        self.n = polynomial_order if not isinstance(polynomial_order, int) else [polynomial_order]  # the polynomial order
        self.time_interval = integration_time
        self.multiprocessing = multiprocessing
        self.multiprocessing_multiplier = multiprocessing_multiplier
        self.threads = threads
        self.multiprocessing_raw = multiprocessing_raw

        # Attributes
        self.paths = None  # the different folder paths
        self.polynomial_list = [self.Generate_nth_order_polynomial(order)[0] for order in self.n]
        self.params_list = [np.random.rand(order + 1) for order in self.n]

        # Functions
        self.Paths()
        self.Time_interval()
        self.Loader()

    def Paths(self):
        """
        Path creation.
        """

        self.paths = {
            'main': os.path.join(os.getcwd(), '..', 'test_conv3d_array'), 
            'cubes': os.path.join(os.getcwd(), '..', 'Cubes_karine'),
            'intensities': os.path.join(os.getcwd(), '..', 'STEREO', 'int'),
                     }
    
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

    def Time_chunks(self, dates, cubes: np.ndarray | dict, date_max, date_min, shared_array: bool = False):
        """
        To select the data in the time chunk given the data chosen for the integration.
        """

        if shared_array:
            shm = shared_memory.SharedMemory(name=cubes['shm.name'])
            coords = np.ndarray(cubes['data.shape'], dtype=cubes['data.dtype'], buffer=shm.buf)
            cubes = COO(coords=coords, data=1, shape=cubes['cubes.shape'])

        chunk = []
        for date2, data2 in zip(dates, cubes):
            date_seconds2 = (((self.days_per_month[date2.month] + date2.day) * 24 + date2.hour) * 60 + date2.minute) * 60 \
                + date2.second
            
            if date_seconds2 < date_min:
                continue
            elif date_seconds2 <= date_max:
                chunk.append(data2)
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
          
    def Generate_nth_order_polynomial(self, n: int = 3):
        """
        Creating a one variable n-th order polynomial.
        """

        def nth_order_polynomial(t, *coeffs):
            """
            The n-th order polynomial.
            """

            # Initialisation
            result = 0

            # Calculating the polynomial
            for order in range(n + 1):
                result += coeffs[order] * t**order 
            return result
        
        nb_coeffs = n + 1
        return nth_order_polynomial, nb_coeffs
    
    def Shared_memory(self, data: np.ndarray):
        """
        Creating a shared memory object for storing a numpy array.
        Only saves a dictionary with the necessary information to access the shared array.
        """

        shm = shared_memory.SharedMemory(create=True, size=data.nbytes)
        shared_array = np.ndarray(data.shape, dtype=data.dtype, buffer=shm.buf)
        np.copyto(shared_array, data)

        info = {
            'shm.name': shm.name,
            'data.shape': data.shape,
            'data.dtype': data.dtype,
        }
        shm.close()
        return shm, info

    @Decorators.running_time
    def Loader(self):
        """
        To add the chosen data to the Main_structure function.
        The function is here to create polynomials for different data inputs. 
        """

        # Multiprocessing set up
        shm_conv = None
        shm_raw = None
        multiprocessing = True if (self.multiprocessing and len(self.datatype) > 1) else False
        processes = []

        # Pre-loading the convolution data if needed
        conv_name_list = ['conv3dAll', 'conv3dSkele']
        if any(item in conv_name_list for item in self.datatype):
            data_conv = np.load(os.path.join(self.paths['main'], 'conv3dRainbow.npy')).astype('uint8')

            if multiprocessing: shm_conv, data_conv = self.Shared_memory(data_conv)

        # Pre-loading if raw data needed
        raw_name_list = ['raw', 'rawContours']
        if any(item in raw_name_list for item in self.datatype):
            data_raw = self.Getting_IDL_cubes()

            if multiprocessing: shm_raw, data_raw = self.Shared_memory(data_raw)

        for datatype in self.datatype:
            kwargs = {
                'datatype': datatype,
                'shared_array': multiprocessing,
                      }
            if 'Contours' in datatype:
                kwargs['contour'] = True
            elif 'Skele' in datatype:
                kwargs['skeleton'] = True

            kwargs['data'] = data_conv if 'conv3d' in datatype else data_raw

            if multiprocessing:
                processes.append(Process(target=self.Main_structure, kwargs=kwargs))
            else:
                self.Main_structure(**kwargs)

        for p in processes: p.start()
        for p in processes: p.join()

        if multiprocessing: 
            if shm_conv is not None: shm_conv.unlink()
            if shm_raw is not None: shm_raw.unlink()

    @Decorators.running_time
    def File_opening_threads(self, filepaths: str):
        """
        Opening the .save files via threads
        """
        
        # Threading initialisation
        queue = Queue.Queue()
        threads = []
        lock = threading.Lock()
        results = []

        # Creating a set number of threads and starting them
        for _ in range(self.threads):
            t = threading.Thread(target=self.File_reader, args=(queue, lock, results))
            t.start()
            threads.append(t)

        # Populating the queue
        for index, filepath in enumerate(filepaths): queue.put((index, filepath))
        for _ in range(self.threads): queue.put(None)  # For braking the while loop inside the File_reader() function.
        queue.join()

        # Joining the threads
        for t in threads: t.join()
        results.sort(key=lambda x: x[0])
        return [result[1] for result in results]

    def File_reader(self, queue, lock, results: list):
        """
        Opens a .save file and saves it in a queue.Queue().
        """

        while True:
            # Getting the queue inputs
            item = queue.get()
            if item is None: 
                queue.task_done()
                break
            
            # Setting up the file reading
            index, filepath = item
            try:
                data = readsav(filepath).cube.astype('uint8')
                with lock: results.append((index, data))
            except Exception as e:
                raise TypeError(f'Filename {os.path.basename(filepath)} read error: {e}')
            finally:
                queue.task_done()

    def Time_integration(self, dates: list, data: np.ndarray | dict, queue = None, index: int | None = None, step: int | None = None, shared_array: bool = False, last: bool = False):
        """
        To integrate the cubes for a given a time interval and the file dates in seconds.
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

            date_min = date_seconds - self.time_interval / 2
            date_max = date_seconds + self.time_interval / 2
            time_cubes_no_duplicate.append(self.Time_chunks(dates=dates, cubes=data, date_max=date_max, date_min=date_min, shared_array=shared_array))
        time_cubes_no_duplicate = stack(time_cubes_no_duplicate, axis=0).astype('uint8')

        if shared_array: 
            queue.put((index, time_cubes_no_duplicate))
        else:
            return time_cubes_no_duplicate
    

    @Decorators.running_time
    def Getting_IDL_cubes(self):
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
        cube_numbers = [int(pattern_cubes.match(cube_name).group(1)) for cube_name in cube_names]

        cubes = self.File_opening_threads([os.path.join(self.paths['cubes'], cube_name)
                                           for cube_name in cube_names])
        cubes = np.array(cubes, dtype='uint8')  
        cubes_no_duplicate = ((cubes & 0b00011000) == 24).astype('uint8')

        # Setting up a sparse data array
        cubes_no_duplicate = COO(cubes_no_duplicate)

        # Getting the corresponding filenames 
        filenames = []
        for number in cube_numbers: 
            for filepath in all_filenames:
                filename = os.path.basename(filepath)
                if filename[:4] == f'{number:04d}':
                    filenames.append(filename)
                    break
        dates = [CustomDate(pattern_int.match(filename).group(1)) for filename in filenames]
        self.days_per_month = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        date = dates[0]
        if (date.year % 4 == 0 and date.year % 100 !=0) or (date.year % 400 == 0):  # Only works if the year doesn't change
            self.days_per_month[2] = 29  # for leap years
        
        # Setting up the multiprocessing if needed
        multiprocessing = True if (self.multiprocessing and self.multiprocessing_raw > 1) else False

        if multiprocessing:

            coords = cubes_no_duplicate.coords
            shm = shared_memory.SharedMemory(create=True, size=coords.nbytes)
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

            results = [None for _ in range(self.multiprocessing_raw)]
            index = -1

            while not queue.empty():
                identifier, result = queue.get()
                results[identifier] = result
                index += 1
            results = concatenate(results, axis=0).astype('uint8')
            results = results.todense()

        else:
            results = self.Time_integration(dates=dates, data=cubes_no_duplicate, shared_array=False)
            results = results.todense()

        if multiprocessing: shm.unlink()
        return results

    @Decorators.running_time
    def Main_structure(self, datatype: str, data: np.ndarray | dict, shared_array: bool = False, contour: bool = False, skeleton: bool = False):
        """
        Class' main function.
        """

        # Defining the reference if using shared arrays
        if shared_array:
            shm = shared_memory.SharedMemory(name=data['shm.name'])
            data = np.ndarray(data['data.shape'], dtype=data['data.dtype'], buffer=shm.buf)

        # Defining the conv_threshold list if the right datatype
        conv_thresholds = [] if 'conv3dSkele' not in datatype else self.conv_thresholds

        if len(conv_thresholds) != 0:
            # Creating a list of arrays depending on the thresholds chosen
            data = [data > threshold for threshold in conv_thresholds]  # WARNING: This could take a lot of RAM
        elif shared_array:
            data = [np.copy(data)]
        else:
            data = [data]
        
        if shared_array: shm.close()

        # Setting up the multiprocessing 
        multiprocessing = True if (self.multiprocessing and len(data) > 1) else False
        processes = []
 
        if multiprocessing:
            processes = [Process(target=self.Processing_4D_binary, args=(datatype, binary_cube, contour, skeleton, 
                                                                         conv_thresholds[conv_index] if conv_thresholds else None)) 
                         for conv_index, binary_cube in enumerate(data)]
        else:
            for conv_index, binary_cube in enumerate(data): 
                self.Processing_4D_binary(datatype, binary_cube, contour, skeleton, conv_thresholds[conv_index] if conv_thresholds else None)

        for p in processes: p.start()
        for p in processes: p.join()

    def Processing_4D_binary(self, datatype: str, data: np.ndarray | None, contour: bool = False, skeleton: bool = False, conv_threshold: int | None = None):
        """
        The processing of one 4D binary data cube
        """

        # Multiprocessing set-up
        multiprocessing = True if (self.multiprocessing and len(self.n) > 1) else False
        processes = []

        if multiprocessing: shm, data = self.Shared_memory(data)

        kwargs = {
            'data': data,
            'datatype': datatype,
            'contour': contour,
            'skeleton': skeleton,
            'shared_array': multiprocessing,
        }

        if multiprocessing:
            processes = [Process(target=self.Time_multiprocessing, kwargs={'index': i, **kwargs}) for i in range(len(self.n))]
        else:
            for i in range(len(self.n)): self.Time_multiprocessing(index=i, **kwargs)

        for p in processes: p.start()
        for p in processes: p.join()
        
        if multiprocessing: shm.unlink()

    def Time_multiprocessing(self, data: np.ndarray | dict, datatype: str,  shared_array: bool, index: int, conv_threshold: int | None = None, 
                  contour: bool = False, skeleton: bool = False):
        """
        The for loop on the time axis.
        """

        # Initialisation of the multiprocessing
        multiprocessing = True if (self.multiprocessing and self.multiprocessing_multiplier > 1) else False

        if multiprocessing:
            if not shared_array: shm, data = self.Shared_memory(data)

            # Initialisation
            processes = []
            manager = Manager()
            queue = manager.Queue()

            # Step an multiprocessing kwargs
            data_shape_0 = data['data.shape'][0] if isinstance(data, dict) else data.shape[0]
            step = int(np.ceil(data_shape_0 / self.multiprocessing_multiplier))
            kwargs = {
                'queue': queue,
                'step': step,
                'data': data,
                'shared_array': True,
                'contour': contour,
                'skeleton': skeleton,
                'poly_index': index,
            }
            # Preping the processes
            for i in range(self.multiprocessing_multiplier):
                if not (i==self.multiprocessing_multiplier - 1):
                    processes.append(Process(target=self.Time_loop, kwargs={'index': i, **kwargs}))
                else:
                    processes.append(Process(target=self.Time_loop, kwargs={'index': i, 'last': True, **kwargs}))

            for p in processes: p.start()
            for p in processes: p.join()

            if not shared_array: shm.unlink()

            results = [None for _ in range(self.multiprocessing_multiplier)]
            while not queue.empty():
                identifier, result = queue.get()
                results[identifier] = result
            results = np.concatenate(results, axis=1)
        else:
            results = self.Time_loop(data=data, shared_array=shared_array, contour=contour, skeleton=skeleton, poly_index=index)

        # Saving the data in .npy files
        if conv_threshold is not None:
            added_string = f'_lim_{conv_threshold}'
        else:
            added_string = ''
        filename = f'poly_{datatype}' + added_string + f'_order{self.n[index]}.npy'
        np.save(os.path.join(self.paths['main'], filename), results.astype('uint16'))

        print(f'File {filename} saved with shape {results.shape}', flush=True)

    def Time_loop(self, poly_index: int, index: int | None = None, queue: None = None, step: int | None = None, last: bool = False, data: np.ndarray | None = None, shared_array: bool = False,
                  contour: bool = False, skeleton: bool = False):
        """
        The for loop on the time axis.
        """

        if shared_array:
            shm = shared_memory.SharedMemory(name=data['shm.name'])
            data = np.ndarray(data['data.shape'], dtype=data['data.dtype'], buffer=shm.buf)

        # If multiprocessing this part
        if step:
            first_index = step * index
            if not last:
                data = data[first_index:step * (index + 1)]
            else:
                data = data[first_index:]
        else:
            first_index = 0

        # Initialisation of the result list
        results = []
        for time in range(data.shape[0]): # TODO: need to change the order so that not so many processes are opened and closed
            # Selecting a slice of the data
            section = data[time]

            if contour:
                # Prepocessing to only keep the contours
                eroded_data = binary_erosion(section)
                section = section.astype('uint8') - eroded_data.astype('uint8')
            elif skeleton:
                # Preprocessing to only fit the skeleton
                section = skeletonize(section)

            # Setting up the data
            x, y, z = np.where(section)
            points = np.stack((x, y, z), axis=1)

            # Calculating the cumulative distance
            t = np.zeros(points.shape[0])
            for i in range(1, points.shape[0]):
                t[i] = t[i-1] + np.linalg.norm(points[i] - points[i - 1])
            t /= t[-1]  # normalisation
            t += 1

            results.append(self.Fitting_polynomial(time=time, t=t, data=points, index=poly_index, first_time_index=first_index))
        results = np.concatenate(results, axis=1)

        if shared_array: shm.close()
        if not step: return results  # if no multiprocessing
        queue.put((index, results))

    def Fitting_polynomial(self, time: int, t: np.ndarray, data: np.ndarray | dict, index: int, first_time_index: int = 0):
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
            t_fine = np.linspace(0.5, 2.5, 10**4)
            x = polynomial(t_fine, *params_x)
            y = polynomial(t_fine, *params_y)
            z = polynomial(t_fine, *params_z)

            # Taking away duplicates 
            data = np.vstack((x, y, z)).T.astype('uint16')

            # Cutting away the values that are outside the initial cube shape
            conditions = (data[:, 0] >= 320) | (data[:, 1] >= 226) | (data[:, 2] >= 186)
            data = data[~conditions]        
            unique_data = np.unique(data, axis=0).T
            time_row = np.full((1, unique_data.shape[1]), time + first_time_index)
            unique_data = np.vstack((time_row, unique_data)).astype('uint16')
            return unique_data


class OrthographicalProjection:
    """
    Does the 2D projection of a 3D volume.
    Used to recreate what is seen by SDO when looking at the cube, especially the curve fits.
    """

    @typechecked
    def __init__(self, coords: np.ndarray | None = None, data: np.ndarray | None = None, camera_position: np.ndarray | tuple | list = (1, 1, 1), 
                 center_of_interest: np.ndarray | tuple | list = (0, 0, 0), camera_fov: int | float | str = '1deg'):
    
        # Data arguments

        if data is not None:
            if data.shape[0] not in [3, 4]: 
                self.data = COO(data)
            else:
                self.data = data
        elif coords is not None:
            self.data = coords
        else:
            raise ValueError("'coords' or 'data' needed to get the 2D projection.")

        # Camera arguments
        self.center_of_interest = np.stack(center_of_interest, axis=0) if isinstance(center_of_interest, (tuple, list)) else center_of_interest
        self.camera_position = np.stack(camera_position, axis=0) if isinstance(camera_position, (tuple, list)) else camera_position  # TODO: need to check for when the camera is a list but only the x, y, z, value
        self.camera_fov = camera_fov

        # Functions
        # self.Checks()  # checks if the arguments had the expected shapes
        self.Paths()
        self.Important_attributes()
        self.Cartesian_pos()
        self.SDO_pos()

    # def Checks(self):
    #     """
    #     Checking if the inputs got the expected values/shapes.
    #     """

    #     if (len(self.camera_position.shape) == 2) and (self.camera_position.shape[1] == 3):
    #         pass
    #     elif (len(self.camera_position.shape == 1)) and (self.camera_position.shape[0] == 3):
    #         pass
    #     else:
    #         raise ValueError(f"'camera_position' needs to be a list/tuple of arrays or an array of shape (3,) or (n, 3). Not {self.camera_position}")
        
    def Paths(self):
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

    # def Recentering_the_data(self):
    #     """
    #     Putting the center of interest as the initial origin of the data to simply the translation matrix 
    #     creation.
    #     """

    #     if self.data.shape[0] == 3:
    #         self.data -= self.center_of_interest
    #     elif self.data.shape[0] == 4:
    #         pass

    #     # TODO: need to finish this part too

    # @typechecked
    # def Ortho_projection(self, right: int | float, left: int | float, bottom: int | float, top: int | float, 
    #                      near: int | float, far: int | float, x: np.ndarray, y: np.ndarray, z: np.ndarray):
    #     """
    #     The translation matrix.
    #     """

    #     x_proj = (2 * x - right - left) / (right - left)
    #     y_proj = (2 * y - top - bottom) / (top - bottom)
    #     z_proj = (-2 * z - far - near) / (far - near)

    #     return x_proj, y_proj, z_proj

    # def Ortho_projection_params(self):
    #     """
    #     To get all the parameters needed for the orthographical projection translation matrix.
    #     """

    #     # Re-centering the data
    #     if len(self.center_of_interest.shape) != 2 and self.center_of_interest.shape[0] == 3:
    #         self.center_of_interest = np.stack([
    #             self.center_of_interest 
    #             for loop in range(self.data.shape[0])
    #         ], axis=0)
    #     elif len(self.center_of_interest.shape) > 2:
    #         raise ValueError('The value of center_of_interest is wrong.')
    #     for time in range(self.data.shape[0]):
    #         filters = self.cartesian_data_coords[0, :] == time
    #         section = self.cartesian_data_coords[filters].astype('int32')
    #         section[1:4, :] -= self.center_of_interest[time, :]
    #         self.cartesian_data_coords[filters] = section

    #     # The box size gotten by looking at the Coronal Monsoon paper
    #     lat_min = (245 - 270) * u.deg  # TODO: need to check with sir Auchere. Most probably wrong
    #     lat_max = (295 - 270) * u.deg
    #     radius_min = 0 * u.km
    #     radius_max = radius_min + (self.solar_r + 3.4 * 10**3) / self.solar_r * u.km
    #     lon_min = 165 * u.deg
    #     lon_max = 225 * u.deg

    #     # Setting the max and min coords in cartesian coordinates that encompasses the whole sun with the protuberance
    #     cartesian_max = np.array([radius_max, radius_max, radius_max])
    #     cartesian_min = - cartesian_max


    #     # TODO: most probably don't need this code anymore
    
    def Matrix_rotation(self):
        """
        To rotate the matrix so that it aligns with the satellite pov
        """
        data_list = []
        for test_time in [0, 5, 40, 90, 200]:
            data_filter = self.cartesian_data_coords[0, :] == test_time

            data = self.cartesian_data_coords[1:4, data_filter]
            print(f'data shape is {data.shape}')
            center = np.array([0, 0, 0]).reshape(3, 1)
            data = np.column_stack((data, center))  # adding the sun center to see if the translation is correct
            print(data.shape)
            satelitte_pos = self.sdo_pos[test_time]
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

            print(f'data transpose shape is {data.shape}')
            print(f'rotation matrix shape is {rotation_matrix.shape}')
            nw_data = []
            for point in data:
                # print(f'point shape is {point.shape}')

                new_point = np.matmul(rotation_matrix, point)
                # print(f'new_point shape is {new_point.shape}')
                nw_data.append(new_point)
            nw_data = np.stack(nw_data, axis=-1)
            print(f"new_data shape is {nw_data.shape}")
            data = nw_data
            # rotated_data = np.dot(data, rotation_matrix.T)
            # data = rotated_data.T
            data_list.append(data)
        return data_list
    
    def Rodrigues_rotation(self, axis, angle):
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

    def Important_attributes(self):
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

    def Cartesian_pos(self):
        """
        To calculate the heliographic cartesian positions of some of the objects.
        """

        cubes_sparse_coords = np.copy(self.data).astype('float64')

        # Initialisation
        self.solar_r = 6.96e5  # in km
        first_cube = readsav(os.path.join(self.paths['cubes'], f'cube{self.numbers[0]:03d}.save'))
        dx = first_cube.dx  # in km
        print(f'dx is {dx}')
        print(f'one cube shape is {np.array(first_cube.cube).shape}')
        dy = first_cube.dy
        dz = first_cube.dz
        x_min = first_cube.xt_min  # in km
        y_min = first_cube.yt_min
        z_min = first_cube.zt_min
        # print(f'cubes_sparse_coords shape is {cubes_sparse_coords.shape}')
        cubes_sparse_coords[1, :] = (cubes_sparse_coords[1, :] * dx + x_min) 
        cubes_sparse_coords[2, :] = (cubes_sparse_coords[2, :] * dy + y_min) 
        cubes_sparse_coords[3, :] = (cubes_sparse_coords[3, :] * dz + z_min)
        # print(f'cubes_sparse_coords shape is {cubes_sparse_coords.shape}')
        print(f'x max and min {round(np.max(cubes_sparse_coords[1, :]))}  {round(np.min(cubes_sparse_coords[1, :]))}')
        print(f'y max and min {round(np.max(cubes_sparse_coords[2, :]))}  {round(np.min(cubes_sparse_coords[2, :]))}')
        print(f'z max and min {round(np.max(cubes_sparse_coords[3, :]))}  {round(np.min(cubes_sparse_coords[3, :]))}')

        self.cartesian_data_coords = cubes_sparse_coords
        self.heliographic_cubes_origin = np.array([x_min, y_min, z_min]) 
    
    # def Main_structure(self):
    #     """
    #     Where the main functions are added together to make the code.
    #     """


    def SDO_pos(self):
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




class TESTING_STUFF:
    
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

    def Random_data(self):
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

    def Paths(self):
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

    
    def Matrix_rotation(self):
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
            nw_data = []
            for point in data:
                # print(f'point shape is {point.shape}')

                new_point = np.matmul(rotation_matrix, point)
                # print(f'new_point shape is {new_point.shape}')
                nw_data.append(new_point)
            nw_data = np.stack(nw_data, axis=-1)
            # print(f"new_data shape is {nw_data.shape}")
            data = nw_data
            # rotated_data = np.dot(data, rotation_matrix.T)
            # data = rotated_data.T
            data_list.append(data)
        return data_list
    
    def Rodrigues_rotation(self, axis, angle):
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

    def Important_attributes(self):
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

    # def Cartesian_pos(self):
    #     """
    #     To calculate the heliographic cartesian positions of some of the objects.
    #     """

    #     cubes_sparse_coords = np.copy(self.data).astype('float64')

    #     # Initialisation
    #     self.solar_r = 6.96e5  # in km
    #     first_cube = readsav(os.path.join(self.paths['cubes'], f'cube{self.numbers[0]:03d}.save'))
    #     dx = first_cube.dx  # in km
    #     dy = first_cube.dy
    #     dz = first_cube.dz
    #     x_min = first_cube.xt_min  # in km
    #     y_min = first_cube.yt_min
    #     z_min = first_cube.zt_min
    #     print(f'cubes_sparse_coords shape is {cubes_sparse_coords.shape}')
    #     cubes_sparse_coords[1, :] = (cubes_sparse_coords[1, :] * dx + x_min) 
    #     cubes_sparse_coords[2, :] = (cubes_sparse_coords[2, :] * dy + y_min) 
    #     cubes_sparse_coords[3, :] = (cubes_sparse_coords[3, :] * dz + z_min)
    #     print(f'cubes_sparse_coords shape is {cubes_sparse_coords.shape}')

    #     self.cartesian_data_coords = cubes_sparse_coords
    #     self.heliographic_cubes_origin = np.array([x_min, y_min, z_min]) 

    def SDO_pos(self):
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




if __name__=='__main__':
    BarycenterCreation_4(datatype=['conv3dAll'], 
                         polynomial_order=[2],
                         integration_time='24h',
                         multiprocessing=True,
                         multiprocessing_multiplier=15, 
                         multiprocessing_raw=6)