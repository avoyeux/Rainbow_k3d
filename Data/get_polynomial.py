"""
To create the polynomial fit given the 3D protuberance voxels, but also to use the polynomial fit
parameters to recreate the curve.
In the child class, you can recreate the polynomial fit starting form the Sun's surface (if 
possible) and with a given number of points in the curve.
"""

# IMPORTS
import h5py
import scipy
import typing
import sparse

# IMPORTs alias
import numpy as np
import multiprocessing as mp

# IMPORTs personal
from common import Decorators, MultiProcessing, root_path
from Projection.projection_dataclasses import CubeInformation, HDF5GroupPolynomialInformation



class Polynomial:
    """
    To get the fit curve position voxels and the corresponding n-th order polynomial parameters.
    """

    axes_order = [0, 3, 2, 1]

    def __init__(
            self, 
            data: sparse.COO, 
            order: int, 
            feet_sigma: int | float,
            south_sigma: int | float,
            leg_threshold: float,
            processes: int, 
            precision_nb: int | float = 10**6, 
            full: bool = False,
        ) -> None:
        """ #TODO:update docstring
        Initialisation of the Polynomial class. Using the get_information() instance method, you
        can get the curve position voxels and the corresponding n-th order polynomial parameters
        with their explanations inside a dict[str, str | dict[str, str | np.ndarray]].

        Args:
            data (sparse.COO): the data for which a fit is needed.
            order (int): the polynomial order for the fit.
            feet_sigma (float): the sigma uncertainty value used for the feet when fitting the
                data.
            processes (int): the number of processes for multiprocessing.
            precision_nb (int | float, optional): the number of points used in the fitting.
                Defaults to 10**6.
        """

        # Arguments 
        self.data = self.reorder_data(data)
        self.poly_order = order
        self.feet_sigma = feet_sigma
        self.south_sigma = south_sigma
        self.leg_threshold = leg_threshold
        self.processes = processes
        self.precision_nb = precision_nb
        self.full = full

        # New attributes
        self.params_init = np.random.rand(order + 1)  # initial (random) polynomial coefficients
    
    def reorder_data(self, data: sparse.COO) -> sparse.COO:
        """
        To reorder a sparse.COO array so that the axes orders change. This is done to change which
        axis is 'sorted', as the first axis is always sorted (think about the .ravel() function).

        Args:
            data (sparse.COO): the array to be reordered, i.e. swapping the axes ordering.

        Returns:
            sparse.COO: the reordered sparse.COO array.
        """

        new_coords = data.coords[Polynomial.axes_order]
        new_shape = [data.shape[i] for i in Polynomial.axes_order]
        return sparse.COO(coords=new_coords, data=data.data, shape=new_shape)

    def get_information(self) -> dict[str, str | dict[str, str | np.ndarray]]:
        """
        To get the information and data for the polynomial and corresponding parameters (i.e.
        polynomial coefficients) ndarray. The explanations for these two arrays are given inside
        the dict in this method.

        Returns:
            dict[str, str | dict[str, str | np.ndarray]]: the data and metadata for the
                polynomial and corresponding polynomial coefficients.
        """

        # Get data
        polynomials, parameters = self.get_data()

        # No duplicates uint16
        treated_polynomials = self.no_duplicates_data(polynomials)
        
        # Save information
        information = {
            'description': (
                "The polynomial curve with the corresponding parameters of the "
                f"{self.poly_order}th order polynomial for each cube."
            ),

            'coords': {
                'data': treated_polynomials,
                'unit': 'none',
                'description': (
                    "The index positions of the fitting curve for the corresponding data. The "
                    "shape of this data is (4, N) where the rows represent (t, x, y, z). This "
                    "data set is treated, i.e. the coords here can directly be used in a "
                    "sparse.COO object as the indexes are uint type and the duplicates are "
                    "already taken out."
                ),
            },
            'parameters': {
                'data': parameters.astype('float32'),
                'unit': 'none',
                'description': (
                    "The constants of the polynomial for the fitting. The shape is (4, total "
                    "number of constants) where the 4 represents t, x, y, z. Moreover, the "
                    "constants are in order a0, a1, ... where the polynomial is "
                    "a0 + a1*x + a2*x**2 ..."
                ),
            },
        }
        if self.full:
            raw_coords = {
                'raw_coords': {
                    'data': polynomials.astype('float32'),
                    'unit': 'none',
                    'description': (
                        "The index positions of the fitting curve for the corresponding data. The "
                        "shape of this data is (4, N) where the rows represent (t, x, y, z). "
                        "Furthermore, the index positions are saved as floats, i.e. if you need "
                        "to visualise it as voxels, then an np.round() and np.unique() is needed."
                    ),
                },
            }
            information |= raw_coords 
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
        shm, data = MultiProcessing.create_shared_memory(data)
        # Run
        processes = [None] * processes_nb
        for i, index in enumerate(indexes):
            p = mp.Process(target=self.no_duplicates_data_sub, args=(data, output_queue, index, i))
            p.start()
            processes[i] = p
        for p in processes: p.join()
        shm.unlink()
        # Results
        polynomials = [None] * processes_nb
        while not output_queue.empty():
            identifier, result = output_queue.get()
            polynomials[identifier] = result
        polynomials = np.concatenate(polynomials, axis=1)
        return polynomials.astype('uint16')

    @staticmethod
    def no_duplicates_data_sub(
            data: dict[str, any],
            queue: mp.queues.Queue,
            index: tuple[int, int],
            position: int,
        ) -> None:
        """
        To multiprocess the no duplicates uint16 voxel positions treatment.

        Args:
            data (dict[str, any]): the information to get the data from a
                multiprocessing.shared_memory.SharedMemory() object.
            queue (mp.queues.Queue): the output queue.
            index (tuple[int, int]): the indexes to slice the data properly.
            position (int): the position of the process to concatenate the result in the right
                order.
        """
        
        # Open SharedMemory
        shm, data = MultiProcessing.open_shared_memory(data)

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
        To get the polynomial and corresponding polynomial coefficients. The polynomial in
        this case is the voxel positions of the curve fit as a sparse.COO coords array (i.e. shape
        (4, N) where N the number of non zeros).

        Returns:
            tuple[np.ndarray, np.ndarray]: the polynomial and parameters array, both with shape
                (4, N) (not the same value for N of course).
        """

        # CONSTANTS
        self.time_indexes = list(set(self.data.coords[0, :]))
        #TODO: might get this from outside the class so that it is not computed twice or more
        self.time_len = len(self.time_indexes)
        process_nb = min(self.processes, self.time_len)

        # Setting up weights as sigma (0 to 1 with 0 being infinite weight)
        sigma = self.data.data.astype('float64')
        print(
            f'The maximum value found even before the filtering and everything is {np.max(sigma)}'
        )
        print(f'The number of non 1 values are {np.sum(sigma > 2)}', flush=True)

        # Shared memory
        shm_coords, coords = MultiProcessing.create_shared_memory(
            data=self.data.coords.astype('float64'),
        )
        shm_sigma, sigma = MultiProcessing.create_shared_memory(sigma)

        # Multiprocessing
        manager = mp.Manager()
        input_queue = manager.Queue()
        output_queue = manager.Queue()
        # QUEUE input
        for i, time in enumerate(self.time_indexes): input_queue.put((i, time))
        for _ in range(process_nb): input_queue.put(None)
        # Run
        processes = [None] * process_nb
        kwargs = {
            'coords': coords,
            'sigma': sigma,
            'input_queue': input_queue,
            'output_queue': output_queue,
        }
        kwargs_sub = {
            'params_init': self.params_init,
            'shape': self.data.shape,
            'nth_order_polynomial': self.generate_nth_order_polynomial(),
            'precision_nb': self.precision_nb,
            'feet_sigma': self.feet_sigma,
            'south_sigma': self.south_sigma,
            'leg_threshold': self.leg_threshold,
        }
        for i in range(process_nb):
            p = mp.Process(target=self.get_data_sub, kwargs={'kwargs_sub': kwargs_sub, **kwargs})
            p.start()
            processes[i] = p
        for p in processes: p.join()
        # Unlink shared memories
        shm_coords.unlink()
        shm_sigma.unlink()
        # Results
        parameters: list[np.ndarray] = [None] * self.time_len
        polynomials: list[np.ndarray] = [None] * self.time_len
        while not output_queue.empty():
            identifier, interp, params = output_queue.get()
            polynomials[identifier] = interp
            parameters[identifier] = params
        polynomials: np.ndarray = np.concatenate(polynomials, axis=1)
        parameters: np.ndarray = np.concatenate(parameters, axis=1)
        return polynomials, parameters

    @staticmethod
    def get_data_sub(
            coords: dict[str, any],
            sigma: dict[str, any],
            input_queue: mp.queues.Queue,
            output_queue: mp.queues.Queue,
            kwargs_sub: dict[str, any],
        ) -> None:
        """
        Static method to multiprocess the curve fitting creation.

        Args:
            coords (dict[str, any]): the coordinates information to access the
                multiprocessing.shared_memory.SharedMemory() object.
            sigma (dict[str, any]): the weights (here sigma) information to access the
                multiprocessing.shared_memory.SharedMemory() object.
            input_queue (mp.queues.Queue): the input_queue for each process.
            output_queue (mp.queues.Queue): the output_queue to save the results.
            kwargs_sub (dict[str, any]): the kwargs for the polynomial_fit function.
        """
        
        # Open shared memories
        shm_coords, coords = MultiProcessing.open_shared_memory(coords)
        shm_sigma, sigma = MultiProcessing.open_shared_memory(sigma)
        
        while True:
            # Get arguments
            args = input_queue.get()
            if args is None: break
            index, time_index = args

            # Filter data
            time_filter = coords[0, :] == time_index
            coords_section = coords[1:, time_filter]
            sigma_section = sigma[time_filter]

            # Check if enough points for polynomial
            nb_parameters = len(kwargs_sub['params_init'])
            if nb_parameters >= sigma_section.shape[0]:
                print(
                    f"For cube index {index}, not enough points for polynomial (shape "
                    f"{coords_section.shape})",
                    flush=True,
                )
                result = np.empty((4, 0)) 
                params = np.empty((4, 0))
            else:
                # Get cumulative distance
                t = np.empty(sigma_section.shape[0], dtype='float64')
                t[0] = 0
                for i in range(1, sigma_section.shape[0]): 
                    t[i] = t[i - 1] + np.sqrt(np.sum([
                        (coords_section[a, i] - coords_section[a, i - 1])**2 
                        for a in range(3)
                    ]))
                t /= t[-1]  # normalisation 

                # Get results
                kwargs = {
                    'coords': coords_section,
                    'sigma': sigma_section,
                    't': t,
                    'time_index': time_index,
                }
                result, params = Polynomial.polynomial_fit(**kwargs, **kwargs_sub)

            # Save results
            output_queue.put((index, result, params))
        # Close shared memories
        shm_coords.close()
        shm_sigma.close()

    @staticmethod
    def polynomial_fit(
            coords: np.ndarray,
            sigma: np.ndarray,
            t: np.ndarray,
            time_index: int,
            params_init: np.ndarray,
            shape: tuple[int, ...],
            precision_nb: int,
            nth_order_polynomial: typing.Callable[
                [np.ndarray, tuple[int | float, ...]],
                np.ndarray
            ],
            feet_sigma: int | float,
            south_sigma: int | float,
            leg_threshold: float,
        ) -> tuple[np.ndarray, np.ndarray]:
        """ #TODO: update docstring
        To get the polynomial fit of a data cube.

        Args:
            coords (np.ndarray): the coordinates ndarray to be fitted.
            sigma (np.ndarray): the corresponding weights (here as sigma) for the coordinates
                array.
            t (np.ndarray): the polynomial variable. In our case the cumulative distance.
            time_index (int): the time index for the cube that is being fitted.
            params_init (np.ndarray): the initial (random) polynomial coefficients.
            shape (tuple[int, ...]): the shape of the inputted data cube. 
            precision_nb (float): the number of points used in the polynomial when saved.
            nth_order_polynomial (typing.Callable[
                [np.ndarray, tuple[int  |  float, ...]],
                np.ndarray
                ]): the n-th order polynomial function.
            feet_sigma (float): the sigma position uncertainty used for the feet.

        Returns:
            tuple[np.ndarray, np.ndarray]: the polynomial position voxels and the corresponding
                coefficients.
        """

        # Setting up polynomial weights
        feet_mask = sigma > 2
        beginning_mask = t < leg_threshold
        #TODO: testing a mask at the beginning of last axis (now the x-axis) to force the curve to
        # pass through the left leg.
        t_mask = beginning_mask & ~feet_mask

        # Try to get params
        params = Polynomial.scipy_curve_fit(
            polynomial=nth_order_polynomial,
            t=t,
            t_mask=t_mask,
            coords=coords,
            params_init=params_init,
            sigma=sigma,
            feet_mask=feet_mask,
            feet_sigma=feet_sigma,
            south_sigma=south_sigma,
        )

        # Get curve
        params_x, params_y, params_z = params
        t_fine = np.linspace(-0.5, 1.5, precision_nb)
        x = nth_order_polynomial(t_fine, *params_x)
        y = nth_order_polynomial(t_fine, *params_y)
        z = nth_order_polynomial(t_fine, *params_z)
        data = np.vstack([x, y, z]).astype('float64')

        # Cut outside init data
        conditions_upper = (
            (data[0, :] >= shape[1] - 1) |
            (data[1, :] >= shape[2] - 1) |
            (data[2, :] >= shape[3] - 1)
        )
        #TODO: the top code line is wrong as I am taking away 1 pixel but for now it is just to
        # make sure no problem arouses from floats. will need to see what to do later
        conditions_lower = np.any(data < 0, axis=0)  # as floats can be a little lower than 0
        conditions = conditions_upper | conditions_lower
        data = data[:, ~conditions]

        # No duplicates
        unique_data = np.unique(data, axis=1)

        # Recreate format
        time_row = np.full((1, unique_data.shape[1]), time_index)
        unique_data = np.vstack([time_row, unique_data]).astype('float64')
        time_row = np.full((1, params.shape[1]), time_index)
        params = np.vstack([time_row, params]).astype('float64')
        return unique_data[Polynomial.axes_order], params[Polynomial.axes_order]
        #TODO: will need to change this if I cancel the ax swapping in cls.__init__

    @staticmethod
    def scipy_curve_fit(
            polynomial: typing.Callable[[np.ndarray, tuple[int | float, ...]], np.ndarray],
            t: np.ndarray,
            t_mask: np.ndarray,
            coords: np.ndarray,
            params_init: np.ndarray,
            sigma: np.ndarray,
            feet_mask: np.ndarray,
            feet_sigma: int | float,
            south_sigma: int | float,
        ) -> np.ndarray:
        """
        To try a polynomial curve fitting using scipy.optimize.curve_fit(). If scipy can't converge
        on a solution due to the feet weight, then the feet weight is divided by 4 (i.e. the
        corresponding sigma is multiplied by 4) and the fitting is tried again.
        #TODO: update docstring
        Args:
            polynomial (typing.Callable[[np.ndarray, tuple[int  |  float, ...]], np.ndarray]): the
                function that outputs the n_th order polynomial function results.
            t (np.ndarray): the cumulative distance.
            coords (np.ndarray): the (x, y, z) coords of the data points.
            params_init (np.ndarray): the initial (random) polynomial parameters.
            sigma (np.ndarray): the standard deviation for each data point (i.e. can be seen as the
                inverse of the weight).
            mask (np.ndarray): the mask representing the feet position.
            feet_sigma (float): the value of sigma given for the feet. This value is quadrupled
                every time a try fails.

        Returns:
            np.ndarray: the coefficients (params_x, params_y, params_z) of the polynomial.
        """

        kwargs = {
            'polynomial': polynomial,
            't': t, 
            't_mask': t_mask,
            'coords': coords,
            'params_init': params_init,
            'sigma': sigma,
            'feet_mask': feet_mask,
            'south_sigma': south_sigma,
        }
        try: 
            sigma[feet_mask] = feet_sigma
            x, y, z = coords
            params_x, _ = scipy.optimize.curve_fit(polynomial, t, x, p0=params_init, sigma=sigma)
            sigma[~feet_mask] = 20
            sigma[t_mask] = south_sigma
            params_y, _ = scipy.optimize.curve_fit(polynomial, t, y, p0=params_init, sigma=sigma)
            params_z, _ = scipy.optimize.curve_fit(polynomial, t, z, p0=params_init, sigma=sigma)
            params = np.vstack([params_x, params_y, params_z]).astype('float64')
        
        except Exception:
            # Changing feet value
            feet_sigma *= 4
            print(
                "\033[1;31mThe curve_fit didn't work. Multiplying the value of the feet by 4, "
                f"i.e. value is {feet_sigma}.\033[0m",
                flush=True,
            )
            params = Polynomial.scipy_curve_fit(feet_sigma=feet_sigma, **kwargs)

        finally:
            print(
                f"\033[92mThe curve_fit worked with feet values equal to {feet_sigma}.\033[0m",
                flush=True,
            )
            return params

    def generate_nth_order_polynomial(
            self,
        ) -> typing.Callable[[np.ndarray, tuple[int | float, ...]], np.ndarray]:
        """
        To generate a polynomial function given a polynomial order.

        Returns:
            typing.Callable[[np.ndarray, tuple[int | float, ...]], np.ndarray]: the polynomial
                function.
        """
        
        def nth_order_polynomial(t: np.ndarray, *coeffs: int | float) -> np.ndarray:
            """
            Polynomial function given a 1D ndarray and the polynomial coefficients. The polynomial
            order is defined before hand.

            Args:
                t (np.ndarray): the 1D array for which you want the polynomial results.

            Returns:
                np.ndarray: the polynomial results.
            """

            # Initialisation
            result: np.ndarray = 0

            # Calculating the polynomial
            for i in range(self.poly_order + 1): result += coeffs[i] * t**i
            return result
        return nth_order_polynomial


class GetPolynomial:
    """
    Finds the polynomial fit parameters given the data_type and integration time to consider.
    Recreates the fit given a specified number of points. the fit is done for t going from 0 to 1
    (i.e. the fit doesn't really go outside the data voxels themselves).
    """

    def __init__(
            self,
            polynomial_order: int,
            integration_time: int,
            number_of_points: int | float,
            data_type: str = 'No duplicates new with feet',
        ) -> None:
        """
        Initialise the class so that the pointer to the polynomial parameters is created (given
        the specified data type, integration time and polynomial order).
        After having finished using the class, the .close() method needs to be used to close the 
        HDF5 pointer.

        Args:
            polynomial_order (int): the polynomial order to consider.
            integration_time (int): the integration time to consider for the choosing of the 
                polynomial fit.
            number_of_points (int | float): the number of points to use when getting the polynomial
                fit positions.
            data_type (str, optional): the data type to consider when looking for the corresponding
                polynomial fit. Defaults to 'No duplicates new with feet'.
        """

        # ATTRIBUTES
        self.t_fine = np.linspace(0, 1, int(number_of_points)) 
        self.data_type = data_type
        self.integration_time = integration_time
        self.order = polynomial_order
        
        # POINTERs
        self.file, self.polynomial_info = self.get_group_pointer()

    def get_group_pointer(self) -> tuple[h5py.File, HDF5GroupPolynomialInformation]:
        """ # todo update the docstring
        Opens the HDF5 file and returns the pointer to the file and the needed polynomial feet
        parameters.

        Returns:
            tuple[h5py.File, h5py.Dataset]: the pointer to the HDF5 file and to the polynomial fit
                parameters.
        """

        # PATHs file and dataset
        filepath = root_path + '/Data/sig1e20_leg20_lim0_03.h5'
        group_path = (
            'Time integrated/' + 
            self.data_type +
            f'/Time integration of {self.integration_time}.0 hours'
        )

        # FILE read
        H5PYFile = h5py.File(filepath, 'r')
        return H5PYFile, HDF5GroupPolynomialInformation(H5PYFile[group_path], self.order)

    def get_params(self, cube_index: int) -> np.ndarray:
        """ # todo update the docstring
        To filter the polynomial fit parameters to keep only the parameters for a given 'cube'
        (i.e. for a given time index).

        Args:
            cube_index (int): the index of the cube to consider. The index here represents the
                time index in the data itself and not the number representing the cube (e.g. not 10
                in cube0010.save).

        Returns:
            np.ndarray: the (x, y, z) parameters of the polynomial fit for a given time.
        """

        mask = (self.polynomial_info.coords[0, :] == cube_index)
        return self.polynomial_info.coords[1:4, mask].astype('float64')

    def nth_order_polynomial(self, t: np.ndarray, *coeffs: int | float) -> np.ndarray:
        """
        Polynomial function given a 1D ndarray and the polynomial coefficients. The polynomial
        order is defined before hand.

        Args:
            t (np.ndarray): the 1D array for which you want the polynomial results.
            coeffs (int | float): the coefficient(s) for the polynomial in the order a_0 + a_1 * t
                a_2 * t**2 + ...

        Returns:
            np.ndarray: the polynomial results.
        """

        # INIT
        result: np.ndarray = 0

        # POLYNOMIAL
        for i in range(self.order + 1): result += coeffs[i] * t**i
        return result

    def get_polynomial(self, cube_index: int) -> np.ndarray:
        """
        Gives the polynomial fit coordinates with a certain number of points. The fit here is
        defined for t in [0, 1].

        Args:
            cube_index (int): the index of the cube to consider. The index here represents the
                time index in the data itself and not the number representing the cube (e.g. not 10
                in cube0010.save).

        Returns:
            np.ndarray: the polynomial fit coordinates.
        """

        # PARAMs
        params = self.get_params(cube_index)

        # COORDs
        return self.get_coords(self.t_fine, params)

    def get_coords(self, t_fine: np.ndarray, params: np.ndarray) -> np.ndarray:
        """
        Gives the coordinates of the polynomial fit given the polynomial parameters defined for a
        given cumulative distance 't_fine'.

        Args:
            t_fine (np.ndarray): the cumulative distance to consider. Usually a np.linspace().
            params (np.ndarray): the (x, y, z) parameters of the polynomial fit.

        Returns:
            np.ndarray: the coordinates of the polynomial fit.
        """

        # PARAMs
        params_x, params_y, params_z = params

        # COORDs
        x = self.nth_order_polynomial(t_fine, *params_x)
        y = self.nth_order_polynomial(t_fine, *params_y)
        z = self.nth_order_polynomial(t_fine, *params_z)
        return np.stack([x, y, z], axis=0)
    
    def close(self):
        """
        To close the HDF5 file pointer.
        To be used when not needing to compute more polynomial fit positions.
        """

        self.file.close()


class GetCartesianProcessedPolynomial(GetPolynomial):
    """
    To process the polynomial fit positions so that the final result is a curve with a set number
    of points defined from the Sun's surface. If not possible, then the fit stops at a predefined
    distance. The number of points in the resulting stays the same.
    """

    def __init__(
            self,
            polynomial_order: int,
            integration_time: int,
            number_of_points: int | float,
            dx: float,
            data_type: str = 'No duplicates new with feet',
        ) -> None:
        """ # todo update docstring
        To process the polynomial fit positions so that the final result is a curve with a set
        number of points defined from the Sun's surface. If not possible, then the fit stops at a
        predefined distance. The number of points in the resulting stays the same.

        Args:
            polynomial_order (int): the order of the polynomial fit to consider.
            integration_time (int): the integration time to consider when choosing the polynomial
                fit parameters.
            number_of_points (int | float): the number of positions to consider in the final
                polynomial fit.
            borders (dict[str, float]): the cube borders to consider to be able to get the final
                heliocentric cartesian positions of the polynomial fit.
        """

        # PARENT
        super().__init__(
            polynomial_order=polynomial_order,
            integration_time=integration_time,
            number_of_points=0,
            data_type=data_type,
        )

        # EXTRAPOLATION polynomial
        self.t_fine = np.linspace(-0.2, 1.4, int(1e4))

        # ATTRIBUTEs
        self.dx = dx
        self.solar_r = 6.96e5  # in km
        self.number_of_points = number_of_points

    def to_cartesian(self, data: np.ndarray) -> np.ndarray:
        """
        To calculate the heliographic cartesian positions given a ndarray of index positions.

        Args:
            data (np.ndarray): the index positions.

        Returns:
            np.ndarray: the corresponding heliocentric cartesian positions.
        """

        # COORDs cartesian
        data[0, :] = data[0, :] * self.dx + self.polynomial_info.xt_min
        data[1, :] = data[1, :] * self.dx + self.polynomial_info.yt_min
        data[2, :] = data[2, :] * self.dx + self.polynomial_info.zt_min
        return data

    def reprocessed_polynomial(self, cube_index: int) -> CubeInformation:
        """
        To create the polynomial fit to firstly find the polynomial fit limits to consider. From
        there the polynomial fit positions are recalculated keeping the new limits into
        consideration and the final number of points needed.

        Args:
            cube_index (int): the index of the cube to consider. The index here represents the
                time index in the data itself and not the number representing the cube (e.g. not 10
                in cube0010.save).

        Returns:
            np.ndarray: the polynomial fit positions in the initial cube index positions. Important
                to note that these new 'cube indexes' can also be negative as the limits have been
                re-computed.
        """

        # PARAMs polynomial
        params = self.get_params(cube_index)

        # COORDs cartesian
        coords = self.get_coords(self.t_fine, params)
        coords = self.to_cartesian(coords)
        
        # FILTER inside the Sun
        distance_sun_center = np.sqrt(coords[0]**2 + coords[1]**2 + coords[2]**2)  # in km
        sun_filter = distance_sun_center < self.solar_r

        # FILTER far from Sun
        x_filter = (coords[0] < - self.solar_r * 1.27) | (coords[0] > - self.solar_r * 0.92)
        y_filter = (coords[1] < - self.solar_r * 0.5) | (coords[1] > -self.solar_r * 0.05)
        z_filter = (coords[2] < - self.solar_r * 0.28) | (coords[2] > self.solar_r * 0.3)
        
        # FILTERs combine
        to_filter = (x_filter | y_filter | z_filter | sun_filter)
        new_t = self.t_fine[~to_filter]

        # RANGE filtered polynomial
        t_fine = np.linspace(np.min(new_t), np.max(new_t), int(self.number_of_points))

        # COORDs new        
        coords = self.get_coords(t_fine, params)

        # DATA reformatting
        information = CubeInformation(
            order=self.order,
            xt_min=self.polynomial_info.xt_min,
            yt_min=self.polynomial_info.yt_min,
            zt_min=self.polynomial_info.zt_min,
            coords=coords,
        )
        return information
