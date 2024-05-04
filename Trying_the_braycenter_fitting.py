"""
Trying to see if I can fit a n-th order polynomial in the convolved (i.e. smooth banana shape)
Rainbow protuberance.
"""

# Imports 
import os
import numpy as np

from typeguard import typechecked
from scipy.optimize import curve_fit
from skimage.morphology import skeletonize_3d

from common_alf import MathematicalEquations, Decorators



class BarycenterCreation:

    @typechecked
    def __init__(self, n_oder: int = 3, conv_threshold: int = 125, test_time_index: int = 200):
        
        # Arguments
        self.n_order = n_oder
        self.conv_thresholds = conv_threshold
        self.test_time_index = test_time_index

        # Attributes
        self.path = None  # the path to the cubes and where to save the result 

        # Functions
        self.Paths()
        self.Main_structure()

    def Paths(self):
        """
        Path creation.
        """

        self.path = os.path.join(os.getcwd(), '..', 'test_conv3d_array')

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

from sklearn.decomposition import PCA
from scipy.interpolate import BSpline, make_interp_spline


class BarycenterCreation_2:

    @typechecked
    def __init__(self, n_oder: int = 3, conv_threshold: int = 125, test_time_index: int = 200):
        
        # Arguments
        self.n_order = n_oder
        self.conv_thresholds = conv_threshold
        self.test_time_index = test_time_index

        # Attributes
        self.path = None  # the path to the cubes and where to save the result 

        # Functions
        self.Paths()
        self.Main_structure()

    def Paths(self):
        """
        Path creation.
        """

        self.path = os.path.join(os.getcwd(), '..', 'test_conv3d_array')

    def Main_structure(self):
        """
        Class' main function.
        """

        # Importing the data
        cubes = np.load(os.path.join(self.path, 'barycenter_array_4.npy')).astype('uint8')
        binary_cubes = cubes > self.conv_thresholds
        
        # Testing for a specific value
        test_section = binary_cubes[self.test_time_index]
        test_section = skeletonize_3d(test_section)

        print(f'skeleton shape is {test_section.shape}')
        x, y, z = np.where(test_section)
        points = np.stack((x, y, z), axis=1)

        # Finding the principal component
        pca = PCA(n_components=1)
        projection = pca.fit_transform(points)
        sorted_indices = np.argsort(projection[:, 0])
        sorted_points = points[sorted_indices]

        # Fit a B-spline
        t = np.linspace(0, 1, len(sorted_points))
        x, y, z = sorted_points[:, 0], sorted_points[:, 1], sorted_points[:, 2]
        spl_x = make_interp_spline(t, x, k=self.n_order)
        spl_y = make_interp_spline(t, y, k=self.n_order)
        spl_z = make_interp_spline(t, z, k=self.n_order)

        # Evaluate the spline for a smooth curve
        t_fine = np.linspace(0, 1, 1*10**5)
        smooth_x = spl_x(t_fine)
        smooth_y = spl_y(t_fine)
        smooth_z = spl_z(t_fine)

        # Recreating the curve
        points = np.vstack((smooth_x, smooth_y, smooth_z)).T.astype('uint16')
        unique_points = np.unique(points, axis=0).T
        
        print(f'the shape of the output is {unique_points.shape}')
        np.save(os.path.join(self.path, f'polynomial_results_convlim{self.conv_thresholds}_skele.npy'), unique_points)


from scipy.interpolate import splprep, splev


class BarycenterCreation_3:

    @typechecked
    def __init__(self, n_oder: int = 3, conv_threshold: int = 125, test_time_index: int = 200,
                 smooth: float = 0.5):
        
        # Arguments
        self.n_order = n_oder
        self.smooth = smooth
        self.conv_thresholds = conv_threshold
        self.test_time_index = test_time_index

        # Attributes
        self.path = None  # the path to the cubes and where to save the result 

        # Functions
        self.Paths()
        self.Main_structure()

    def Paths(self):
        """
        Path creation.
        """

        self.path = os.path.join(os.getcwd(), '..', 'test_conv3d_array')

    @Decorators.running_time
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
        points = np.stack((x, y, z), axis=1)

        # Finding the principal component
        tck, u = splprep([x, y, z], s=self.smooth, k=self.n_order)
        new_points = np.array(splev(np.linspace(0, 1, 1*10**5), tck))

        print(f'new_points shape is {new_points.shape}')
        new_points = new_points.T.astype('uint16')
        unique_points = np.unique(new_points, axis=0).T

        print(f'the shape of unique_points is{unique_points.shape}')
        np.save(os.path.join(self.path, f'polynomial_results_convlim{self.conv_thresholds}.npy'), unique_points)


from scipy.ndimage import binary_erosion
from multiprocessing import Process, Manager, shared_memory
from scipy.io import readsav
import re
from glob import glob
from Animation_3D_main import CustomDate
import threading
import queue as Queue


class BarycenterCreation_4:

    @typechecked
    def __init__(self, datatype: str | list = 'surface', conv_threshold: int | list | tuple = 125, polynomial_order: int | list | tuple = 3, 
                 integration_time: int | str = '24h', multiprocessing: bool = True, threads: int = 5):
        
        # Arguments
        self.datatype = datatype if not isinstance(datatype, str) else [datatype]  # what data is fitted - 'conv3dContours', 'conv3dAll', 'raw', 'rawContours'
        self.conv_thresholds = conv_threshold if not isinstance(conv_threshold, int) else [conv_threshold]  # minimum threshold to convert the conv3d data to binary
        self.n = polynomial_order if not isinstance(polynomial_order, int) else [polynomial_order]  # the polynomial order
        self.time_interval = integration_time
        self.multiprocessing = multiprocessing
        self.threads = threads

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

    def Time_chunks(self, dates, cubes, date_max, date_min):
        """
        To select the data in the time chunk given the data chosen for the integration.
        """

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
        if len(chunk) == 0:  # i.e. if nothing was found
            return np.zeros((cubes.shape[1], cubes.shape[2], cubes.shape[3]))
        elif len(chunk) == 1:
            return chunk[0]
        else:
            chunk = np.stack(chunk, axis=0)
            return np.any(chunk, axis=0)
          
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
        multiprocessing = True if (self.multiprocessing and len(self.datatype) > 1) else False
        processes = []

        # Pre-loading the convolution data if needed
        conv_name_list = ['conv3dAll', 'conv3dSkele', 'conv3dContours']
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
            shm_conv.unlink()
            shm_raw.unlink()

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
        return results[:, 1]

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
        date = date[0]
        if (date.year % 4 == 0 and date.year % 100 !=0) or (date.year % 400 == 0):  # Only works if the year doesn't change
            self.days_per_month[2] = 29  # for leap years
        
        # Creating the time_cubes integrations together
        time_cubes_no_duplicate = [] 
        for date in dates:
            date_seconds = (((self.days_per_month[date.month] + date.day) * 24 + date.hour) * 60 + date.minute) * 60 + date.second

            date_min = date_seconds - self.time_interval / 2
            date_max = date_seconds + self.time_interval / 2
            time_cubes_no_duplicate.append(self.Time_chunks(dates, cubes_no_duplicate, date_max, date_min))
        
        time_cubes_no_duplicate = np.stack(time_cubes_no_duplicate, axis=0)
        return time_cubes_no_duplicate.astype('uint8')
    
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
        conv_thresholds = [] if 'conv3d' not in datatype else self.conv_thresholds

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
            processes = [Process(target=self.Processing_4D_binary, args=(datatype, binary_cube, contour, skeleton, conv_thresholds)) 
                         for binary_cube in data]
        else:
            for binary_cube in data: self.Processing_4D_binary(datatype, binary_cube, contour, skeleton, conv_thresholds)

        for p in processes: p.start()
        for p in processes: p.join()

    def Processing_4D_binary(self, datatype: str, data: np.ndarray | None, contour: bool = False, skeleton: bool = False, conv_thresholds: list = []):
        """
        The processing of one 4D binary data cube
        """

        # Initialisation of the list with all the results
        all_unique_points = [[] for _ in range(len(self.n))]

        # For loop through the time index
        for time in range(data.shape[0]):
            # Selecting a slice of the data
            section = data[time]

            if contour:
                # Prepocessing to only keep the contours
                eroded_data = binary_erosion(section)
                section = section.astype('uint8') - eroded_data.astype('uint8')
            elif skeleton:
                # Preprocessing to only fit the skeleton
                section = skeletonize_3d(section)

            # Setting up the data
            x, y, z = np.where(section)
            points = np.stack((x, y, z), axis=1)

            # Calculating the cumulative distance
            t = np.zeros(points.shape[0])
            for i in range(1, points.shape[0]):
                t[i] = t[i-1] + np.linalg.norm(points[i] - points[i - 1])
            t /= t[-1]  # normalisation
            t += 1

            # Setting up the multiprocessing
            multiprocessing = True if (self.multiprocessing and len(self.n) > 1) else False
            processes = []
            manager = Manager()
            queue = manager.Queue()
            time_results = [None for _ in range(len(self.n))]

            if multiprocessing:
                # Setting up the shared memory
                shm = shared_memory.SharedMemory(create=True, size=points.nbytes)
                shared_array = np.ndarray(points.shape, dtype=points.dtype, buffer=shm.buf)
                np.copyto(shared_array, points)
                points = {
                    'shm.name': shm.name,
                    'data.shape': points.shape,
                    'data.dtype': points.dtype,
                    }
                shm.close()
                processes = [Process(target=self.Fitting_polynomial, args=(time, t, points, index, multiprocessing, queue)) 
                             for index in range(len(self.n))]
            else:
                time_results = [self.Fitting_polynomial(time=time, t=t, points=points, index=index)
                                for index in range(len(self.n))]

            # Running the multiprocessing 
            for p in processes: p.start()
            for p in processes: p.join()

            if multiprocessing: shm.unlink()
            
            # Getting the results from the queue 
            while not queue.empty():
                indentifier, result = queue.get()
                time_results[indentifier] = result
            
            # Populating the total results list
            for i in range(len(self.n)): all_unique_points[i].append(time_results[i])
            
        # Saving the data in .npy files
        for i, n in enumerate(self.n):
            if len(conv_thresholds) != 0:
                added_string = 'lim' + '_'.join(f'{threshold}' for threshold in conv_thresholds)
            else:
                added_string= ''
            
            # Preparing the filename and data and saving it
            filename = f'poly_{datatype}_' + added_string + f'_order{n}.npy'
            data_array = np.concatenate(all_unique_points[i], axis=1)
            np.save(os.path.join(self.paths['main'], filename), data_array.astype('uint16'))

            print(f'File {filename} saved with shape {data_array.shape}', flush=True)

    def Fitting_polynomial(self, time: int, t: np.ndarray, data: np.ndarray | dict, index, shared_array: bool = False, queue = None):
            """
            Where the fitting of the curve actually takes place.
            """

            if shared_array: 
                shm = shared_memory.SharedMemory(name=data['shm.name'])
                data = np.ndarray(data['data.shape'], dtype=data['data.dtype'], buffer=shm.buf)

            # Getting the data ready
            x, y, z = data[:, 0], data[:, 1], data[:, 2]
            if shared_array: shm.close()
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
            time_row = np.full((1, unique_data.shape[1]), time)
            data_points = np.vstack((time_row, data_points))

            if queue:
                # Getting the results out of the child process
                queue.put((index, unique_data.astype('uint16')))
            else:
                return unique_data


if __name__=='__main__':

    # BarycenterCreation_2(n_oder=3, conv_threshold=40, test_time_index=84)
    # BarycenterCreation_3(n_oder=3, conv_threshold=40, test_time_index=84, smooth=1)
    # BarycenterCreation_4(conv_threshold=10, polynomial_order=4)

    BarycenterCreation_4(datatype=['conv3dContours', 'conv3dSkele'], 
                         conv_threshold=[10, 20], 
                         polynomial_order=[4, 5],
                         integration_time='24h',
                         multiprocessing=True)