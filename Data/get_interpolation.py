"""
To get the polynomial fit parameters and recreate the corresponding curve given the data type and
the interpolation order to consider.
In the child class, you can recreate the polynomial fit starting form the Sun's surface (if 
possible) and with a given number of points in the curve.
"""

# IMPORTS
import os
import h5py

# IMPORTs alias
import numpy as np



class GetInterpolation:
    """
    Finds the polynomial fit parameters given the data_type and integration time to consider.
    Recreates the fit given a specified number of points. the fit is done for t going from 0 to 1
    (i.e. the fit doesn't really go outside the data voxels themselves).
    """

    def __init__(
            self,
            interpolation_order: int,
            integration_time: int,
            number_of_points: int | float,
            data_type: str = 'No duplicates new with feet',
        ) -> None:
        """
        Initialise the class so that the pointer to the polynomial parameters is created (given
        the specified data type, integration time and interpolation order).
        After having finished using the class, the .close() method needs to be used to close the 
        HDF5 pointer.

        Args:
            interpolation_order (int): the polynomial order to consider.
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
        self.order = interpolation_order
        
        # POINTERs
        self.file, self.data_reference = self.get_data_pointer()

    def get_data_pointer(self) -> tuple[h5py.File, h5py.Dataset]:
        """
        Opens the HDF5 file and returns the pointer to the file and the needed polynomial feet
        parameters.

        Returns:
            tuple[h5py.File, h5py.Dataset]: the pointer to the HDF5 file and to the polynomial fit
                parameters.
        """

        # CHECK path
        main_path = '/home/avoyeux/Documents/'
        if not os.path.exists(main_path): main_path = '/home/avoyeux/old_project/'

        # PATHs file and dataset
        filepath = main_path + 'avoyeux/python_codes/Data/sig1e20_leg20_lim0_03.h5'
        dataset_path = (
            'Time integrated/' + 
            self.data_type +
            f'/Time integration of {self.integration_time}.0 hours' +
            f'/{self.order}th order interpolation/parameters'
        )

        # FILE read
        H5PYFile = h5py.File(filepath, 'r')
        return H5PYFile, H5PYFile[dataset_path]

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

        mask = (self.data_reference[0, :] == cube_index)
        return self.data_reference[1:4, mask].astype('float64')

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
        result = 0

        # POLYNOMIAL
        for i in range(self.order + 1): result += coeffs[i] * t**i
        return result

    def get_interpolation(self, cube_index: int) -> np.ndarray:
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


class GetCartesianProcessedInterpolation(GetInterpolation):
    """
    To process the polynomial fit positions so that the final result is a curve with a set number
    of points defined from the Sun's surface. If not possible, then the fit stops at a predefined
    distance. The number of points in the resulting stays the same.
    """

    def __init__(
            self,
            interpolation_order: int,
            integration_time: int,
            number_of_points: int | float,
            borders: dict[str, float],
            data_type: str = 'No duplicates new with feet',
        ) -> None:
        """
        To process the polynomial fit positions so that the final result is a curve with a set
        number of points defined from the Sun's surface. If not possible, then the fit stops at a
        predefined distance. The number of points in the resulting stays the same.

        Args:
            interpolation_order (int): the order of the polynomial fit to consider.
            integration_time (int): the integration time to consider when choosing the polynomial
                fit parameters.
            number_of_points (int | float): the number of positions to consider in the final
                polynomial fit.
            borders (dict[str, float]): the cube borders to consider to be able to get the final
                heliocentric cartesian positions of the polynomial fit.
        """

        # PARENT
        super().__init__(
            interpolation_order=interpolation_order,
            integration_time=integration_time,
            number_of_points=0,
            data_type=data_type,
        )

        # EXTRAPOLATION polynomial
        self.t_fine = np.linspace(-0.2, 1.4, int(1e4))

        # ATTRIBUTEs
        self.solar_r = 6.96e5  # in km
        self.borders = borders
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
        data[0, :] = data[0, :] * self.borders['dx'] + self.borders['xmin']
        data[1, :] = data[1, :] * self.borders['dx'] + self.borders['ymin']
        data[2, :] = data[2, :] * self.borders['dx'] + self.borders['zmin']
        return data

    def reprocessed_interpolation(self, cube_index: int) -> np.ndarray:
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

        # PARAMs interpolation
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

        # RANGE filtered interpolation
        t_fine = np.linspace(np.min(new_t), np.max(new_t), int(self.number_of_points))

        # COORDs new        
        coords = self.get_coords(t_fine, params)
        return coords
