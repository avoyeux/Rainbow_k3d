"""
To create the polynomial fit given the 3D protuberance voxels, but also to use the polynomial fit
parameters to recreate the curve.
"""

# IMPORTS
import h5py
import scipy
import sparse

# IMPORTs alias
import numpy as np
import multiprocessing as mp

# IMPORTs sub
from typing import Any, Callable

# IMPORTs personal
from common import Decorators, MultiProcessing
from codes.projection.projection_dataclasses import HDF5GroupPolynomialInformation

# PLACEHOLDERs type annotation
LockProxy = Any
ValueProxy = Any
QueueProxy = Any



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
            precision_nb: int = int(1e6), 
            full: bool = False,
        ) -> None:
        """
        Initialisation of the Polynomial class. Using the get_information() instance method, you
        can get the curve position voxels and the corresponding n-th order polynomial parameters
        with their explanations inside a dict[str, str | dict[str, str | np.ndarray]].

        Args:
            data (sparse.COO): the data for which a fit is needed.
            order (int): the polynomial order for the fit.
            feet_sigma (int | float): the sigma uncertainty value used for the feet when fitting
                the data.
            south_sigma (int | float): the sigma uncertainty value used for the leg when fitting
                the data.
            leg_threshold (float): the threshold value for defining which points correspond to the
                legs.
            processes (int): the number of processes for multiprocessing.
            precision_nb (int, optional): the number of points used in the fitting.
                Defaults to int(1e6).
            full (bool, optional): choosing to save the cartesian positions of the polynomial fit.
        """

        # ARGUMENTs 
        self.data = self.reorder_data(data)
        self.poly_order = order
        self.feet_sigma = feet_sigma
        self.south_sigma = south_sigma
        self.leg_threshold = leg_threshold
        self.processes = processes
        self.precision_nb = precision_nb
        self.full = full

        # ATTRIBUTEs new
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

        # DATA
        polynomials, parameters = self.get_data()

        # AXEs swap  # * will need to change this if I cancel the ax swapping in cls.__init__
        parameters = parameters[Polynomial.axes_order]
        polynomials = polynomials[Polynomial.axes_order]

        # NO DUPLICATEs
        treated_polynomials = self.no_duplicates_data(polynomials)
        
        # DATA formatting
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

        # MULTIPROCESSING setup
        manager = mp.Manager()
        lock = manager.Lock()
        output_queue = manager.Queue()
        value = manager.Value('i', self.time_len)
        processes_nb = min(self.processes, self.time_len)
        shm, data = MultiProcessing.create_shared_memory(data)

        # RUN processes
        processes: list[mp.Process] = [None] * processes_nb
        for i in range(processes_nb):
            p = mp.Process(
                target=self.no_duplicates_data_sub,
                args=(data, lock, value, output_queue),
            )
            p.start()
            processes[i] = p
        for p in processes: p.join()
        shm.unlink()

        # RESULTs formatting
        polynomials = [None] * self.time_len
        while not output_queue.empty():
            identifier, result = output_queue.get()
            polynomials[identifier] = result
        polynomials: np.ndarray = np.concatenate(polynomials, axis=1).astype('uint16')
        return polynomials

    @staticmethod
    def no_duplicates_data_sub(
            data: dict[str, Any],
            lock: LockProxy,
            value: ValueProxy,
            output_queue: QueueProxy,
        ) -> None:
        """  # todo change docstring
        To multiprocess the no duplicates uint16 voxel positions treatment.

        Args:
            data (dict[str, Any]): the information to get the data from a
                multiprocessing.shared_memory.SharedMemory() object.
            queue (mp.queues.Queue): the output queue.
            index (tuple[int, int]): the indexes to slice the data properly.
            position (int): the position of the process to concatenate the result in the right
                order.
        """
        
        # DATA open
        shm, array = MultiProcessing.open_shared_memory(data)

        while True:
            # CHECK input
            with lock:
                index = int(value.value) - 1
                if index < 0: break
                value.value -= 1

            # DATA filtering
            data_filters = (array[0, :] == index)
            data = np.copy(array[1:, data_filters]).astype('float32')

            # NO DUPLICATEs
            data = np.rint(np.abs(data))
            data = np.unique(data, axis=1).astype('uint16')

            # SAVE results
            output_queue.put((index, Polynomial.format_result(data, index)))

        # CLOSE shared memory
        shm.close()
        
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

        # CONSTANTs
        self.time_len = self.data.coords[0, :].max() + 1
        process_nb = min(self.processes, self.time_len)

        # Setting up weights as sigma (0 to 1 with 0 being infinite weight)
        sigma = self.data.data.astype('float64')

        # MULTIPROCESSING setup
        shm_coords, coords = MultiProcessing.create_shared_memory(
            data=self.data.coords.astype('float64'),
        )
        shm_sigma, sigma = MultiProcessing.create_shared_memory(sigma)
        manager = mp.Manager()
        lock = manager.Lock()
        value = manager.Value('i', self.time_len)
        # * tried 'i' but clearly didn't give a signed integer
        output_queue = manager.Queue()
        
        # INPUTs
        kwargs = {
            'coords': coords,
            'sigma': sigma,
            'lock': lock,
            'value': value,
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

        # RUN processes
        processes: list[mp.Process] = [None] * process_nb
        for i in range(process_nb):
            p = mp.Process(target=self.get_data_sub, kwargs={'kwargs_sub': kwargs_sub, **kwargs})
            p.start()
            processes[i] = p
        for p in processes: p.join()
        shm_coords.unlink()
        shm_sigma.unlink()

        # RESULTs formatting
        parameters = [None] * self.time_len
        polynomials = [None] * self.time_len
        while not output_queue.empty():
            identifier, interp, params = output_queue.get()
            polynomials[identifier] = interp
            parameters[identifier] = params
        polynomials: np.ndarray = np.concatenate(polynomials, axis=1)
        parameters: np.ndarray = np.concatenate(parameters, axis=1)
        return polynomials, parameters

    @staticmethod
    def get_data_sub(
            coords: dict[str, Any],
            sigma: dict[str, Any],
            lock: LockProxy,
            value: ValueProxy,
            output_queue: QueueProxy,
            kwargs_sub: dict[str, Any],
        ) -> None:
        """ # todo change docstring
        Static method to multiprocess the curve fitting creation.

        Args:
            coords (dict[str, Any]): the coordinates information to access the
                multiprocessing.shared_memory.SharedMemory() object.
            sigma (dict[str, Any]): the weights (here sigma) information to access the
                multiprocessing.shared_memory.SharedMemory() object.
            input_queue (mp.queues.Queue): the input_queue for each process.
            output_queue (mp.queues.Queue): the output_queue to save the results.
            kwargs_sub (dict[str, Any]): the kwargs for the polynomial_fit function.
        """
        
        # DATA open
        shm_coords, coords = MultiProcessing.open_shared_memory(coords)
        shm_sigma, sigma = MultiProcessing.open_shared_memory(sigma)
        
        while True:
            # CHECK input
            with lock:
                index = int(value.value) - 1
                if index < 0: break
                value.value -= 1

            # DATA filtering
            time_filter = coords[0, :] == index
            coords_section = coords[1:, time_filter]
            sigma_section = sigma[time_filter]

            # CHECK if enough points
            nb_parameters = len(kwargs_sub['params_init'])
            if nb_parameters >= sigma_section.shape[0]:
                print(
                    f"For cube index {index}, not enough points for polynomial (shape "
                    f"{coords_section.shape})",
                    flush=True,
                )
                result = np.empty((3, 0)) 
                params = np.empty((3, 0))
            else:
                # DISTANCE cumulative
                t = np.empty(sigma_section.shape[0], dtype='float64')
                t[0] = 0
                for i in range(1, sigma_section.shape[0]): 
                    t[i] = t[i - 1] + np.sqrt(np.sum([
                        (coords_section[a, i] - coords_section[a, i - 1])**2 
                        for a in range(3)
                    ]))
                t /= t[-1]  # normalisation 

                # RESULTs formatting
                kwargs = {
                    'coords': coords_section,
                    'sigma': sigma_section,
                    't': t,
                }
                result, params = Polynomial.polynomial_fit(**kwargs, **kwargs_sub)

            # SAVE results
            output_queue.put((
                index,
                Polynomial.format_result(result, index),
                Polynomial.format_result(params, index),
            ))
        
        shm_coords.close()
        shm_sigma.close()

    @staticmethod
    def format_result(data: np.ndarray, time_index: int) -> np.ndarray:
        """
        To format the data so that it has shape (4, N) where 4 represents (t, x, y, z).

        Args:
            data (np.ndarray): the data of shape (3, N).
            time_index (int): the time index corresponding to the data.

        Returns:
            np.ndarray: the reformatted data.
        """

        # DATA formatting
        time_row = np.full((1, data.shape[1]), time_index)
        data = np.vstack([time_row, data]).astype('float32')
        return data

    @staticmethod
    def polynomial_fit(
            coords: np.ndarray,
            sigma: np.ndarray,
            t: np.ndarray,
            params_init: np.ndarray,
            shape: tuple[int, ...],
            precision_nb: int,
            nth_order_polynomial: Callable[
                [np.ndarray, tuple[int | float, ...]],
                np.ndarray
            ],
            feet_sigma: int | float,
            south_sigma: int | float,
            leg_threshold: float,
        ) -> tuple[np.ndarray, np.ndarray]:
        """
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
            feet_sigma (int | float): the sigma position uncertainty used for the feet.
            south_sigma (int | float): the sigma uncertainty for the leg values.
            leg_threshold (float): the threshold value for defining which points correspond to the
                legs.

        Returns:
            tuple[np.ndarray, np.ndarray]: the polynomial position voxels and the corresponding
                coefficients.
        """

        # WEIGHTs
        feet_mask = sigma > 2  # ! this might create some problems
        beginning_mask = t < leg_threshold
        t_mask = beginning_mask & ~feet_mask

        # PARAMs
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

        # CURVE create
        params_x, params_y, params_z = params
        t_fine = np.linspace(-0.5, 1.5, precision_nb)
        x = nth_order_polynomial(t_fine, *params_x)
        y = nth_order_polynomial(t_fine, *params_y)
        z = nth_order_polynomial(t_fine, *params_z)
        data = np.vstack([x, y, z]).astype('float64')

        # BORDERs cut
        conditions_upper = (  # todo change this when the other changes are finished
            (data[0, :] >= shape[1] - 1) |
            (data[1, :] >= shape[2] - 1) |
            (data[2, :] >= shape[3] - 1)
        )
        # * the top code line is wrong as I am taking away 1 pixel but for now it is just to
        # * make sure no problem arouses from floats. will need to see what to do later
        conditions_lower = np.any(data < 0, axis=0)  # as floats can be a little lower than 0
        conditions = conditions_upper | conditions_lower
        data = data[:, ~conditions]
        unique_data = np.unique(data, axis=1)
        return unique_data, params

    @staticmethod
    def scipy_curve_fit(
            polynomial: Callable[[np.ndarray, tuple[int | float, ...]], np.ndarray],
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

        Args:
            polynomial (typing.Callable[[np.ndarray, tuple[int  |  float, ...]], np.ndarray]): the
                function that outputs the n_th order polynomial function results.
            t (np.ndarray): the cumulative distance.
            t_mask (np.ndarray): the mask representing the south leg position.
            coords (np.ndarray): the (x, y, z) coords of the data points.
            params_init (np.ndarray): the initial (random) polynomial parameters.
            sigma (np.ndarray): the standard deviation for each data point (i.e. can be seen as the
                inverse of the weight).
            feet_mask (np.ndarray): the mask representing the feet position.
            feet_sigma (int | float): the value of sigma given for the feet. This value is
                quadrupled every time a try fails.
            south_sigma (int | float): the value of sigma given for the south leg.

        Returns:
            np.ndarray: the coefficients (params_x, params_y, params_z) of the polynomial.
        """

        # DATA formatting
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
            # FITTING
            sigma[feet_mask] = feet_sigma
            x, y, z = coords
            params_x, _ = scipy.optimize.curve_fit(polynomial, t, x, p0=params_init, sigma=sigma)
            sigma[~feet_mask] = 20
            sigma[t_mask] = south_sigma
            params_y, _ = scipy.optimize.curve_fit(polynomial, t, y, p0=params_init, sigma=sigma)
            params_z, _ = scipy.optimize.curve_fit(polynomial, t, z, p0=params_init, sigma=sigma)
            params = np.vstack([params_x, params_y, params_z]).astype('float64')
        
        except Exception:
            # FITTING failed
            feet_sigma *= 4
            print(
                "\033[1;31mThe curve_fit didn't work. Multiplying the value of the feet by 4, "
                f"i.e. value is {feet_sigma}.\033[0m",
                flush=True,
            )
            params = Polynomial.scipy_curve_fit(feet_sigma=feet_sigma, **kwargs)

        finally:
            return params

    def generate_nth_order_polynomial(
            self,
        ) -> Callable[[np.ndarray, tuple[int | float, ...]], np.ndarray]:
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


class GetPolynomialFit:
    """
    Finds the polynomial fit parameters given the data_type and integration time to consider.
    Recreates the fit given a specified number of points. the fit is done for t going from 0 to 1
    (i.e. the fit doesn't really go outside the data voxels themselves).
    """

    def __init__(
            self,
            filepath: str,
            polynomial_order: int,
            integration_time: int,
            number_of_points: int,
            data_type: str = 'No duplicates',
            with_fake_data: bool = False,
        ) -> None:
        """ # todo update docstring
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
                polynomial fit. Defaults to 'No duplicates'.
        """

        # ATTRIBUTES
        self.filepath = filepath
        self.t_fine = np.linspace(0, 1, number_of_points) 
        self.data_type = data_type
        self.integration_time = integration_time
        self.order = polynomial_order
        self.with_fake_data = with_fake_data
        
        # POINTERs
        self.file, self.polynomial_info = self.get_group_pointer()

    def get_group_pointer(self) -> tuple[h5py.File, HDF5GroupPolynomialInformation]:
        """
        To get the HDF5 file pointer and the HDF5GroupPolynomialInformation containing the dataset
        pointer.

        Returns:
            tuple[h5py.File, HDF5GroupPolynomialInformation]: the HDF5 file pointer and the pointer
                to the dataset containing the polynomial fit parameters.
        """

        # PATH group
        init_path = 'Real/' if self.with_fake_data else ''
        group_path = (
            init_path +
            'Time integrated/' + 
            self.data_type +
            f'/Time integration of {self.integration_time}.0 hours'
        )

        # FILE read
        H5PYFile = h5py.File(self.filepath, 'r')
        return H5PYFile, HDF5GroupPolynomialInformation(H5PYFile[group_path], self.order) #type:ignore

    def get_params(self, cube_index: int) -> np.ndarray:
        """
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
        return self.get_coords(params)

    def get_coords(self, params: np.ndarray) -> np.ndarray:
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
        x = self.nth_order_polynomial(self.t_fine, *params_x)
        y = self.nth_order_polynomial(self.t_fine, *params_y)
        z = self.nth_order_polynomial(self.t_fine, *params_z)
        return np.stack([x, y, z], axis=0)
    
    def close(self):
        """
        To close the HDF5 file pointer.
        To be used when not needing to compute more polynomial fit positions.
        """

        self.file.close()
