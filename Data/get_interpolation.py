"""
To get the interpolation parameters and recreate the corresponding curve given the data type and
the interpolation orders to consider. #TODO: update docstring
This code is only for one cube, i.e. the cube index to get the interpolation parameters need to
be specified.
"""

# IMPORTS
import os
import h5py

# IMPORTs alias
import numpy as np



class GetInterpolation:

    def __init__(
            self,
            interpolation_order: int,
            integration_time: int,
            number_of_points: int | float,
            data_type: str = 'No duplicates new with feet'
        ) -> None:

        # ATTRIBUTES
        #TODO: need to remake the interpolations as 0 to 1 for t doesnt mean the range between both
        # feet. Just adding the feet everytime so that 0 and 1 represent the feet should be fine.
        self.t_fine = np.linspace(0, 1, int(number_of_points)) 
        self.data_type = data_type
        self.integration_time = integration_time
        self.order = interpolation_order
        
        # RUN
        self.file, self.data_reference = self.get_data_pointer()

    def get_data_pointer(self) -> tuple[h5py.File, h5py.Dataset]:

        # PATH
        main_path = '/home/avoyeux/Documents/'
        if not os.path.exists(main_path): main_path = '/home/avoyeux/old_project/'
        filepath = main_path + 'avoyeux/python_codes/Data/sig1e20_leg20_lim0_03.h5'
        dataset_path = (
            'Time integrated/' + 
            self.data_type +
            f'/Time integration of {self.integration_time}.0 hours' +
            f'/{self.order}th order interpolation/parameters'
        )

        H5PYFile = h5py.File(filepath, 'r')
        return H5PYFile, H5PYFile[dataset_path]

    def get_params(self, cube_index: int) -> np.ndarray:
        #to get the curve params
        # the cube position in the data (not its value with max 412).
        mask = (self.data_reference[0, :] == cube_index)
        return self.data_reference[1:4, mask].astype('float64')

    def nth_order_polynomial(self, t: np.ndarray, *coeffs: int | float) -> np.ndarray:
        """
        Polynomial function given a 1D ndarray and the polynomial coefficients. The polynomial
        order is defined before hand.

        Args: #TODO: update docstring
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
        # the cube position in the data (not its value with max 412).

        # ARGs check
        if t_fine is None: t_fine = self.t_fine

        # PARAMs
        params = self.get_params(cube_index)

        # COORDs
        return self.get_coords(self.t_fine, params)

    def get_coords(self, t_fine: np.ndarray, params: np.ndarray) -> np.ndarray:

        params_x, params_y, params_z = params

        # COORDs
        x = self.nth_order_polynomial(t_fine, *params_x)
        y = self.nth_order_polynomial(t_fine, *params_y)
        z = self.nth_order_polynomial(t_fine, *params_z)
        return np.stack([x, y, z], axis=0)
    
    def close(self):

        self.file.close()


class GetCartesianProcessedInterpolation(GetInterpolation):

    def __init__(
            self,
            interpolation_order: int,
            integration_time: int,
            number_of_points: int | float,
            borders: dict[str, float], #TODO: need to check type 
            data_type: str = 'No duplicates new with feet',
        ) -> None:

        # PARENT
        super().__init__(
            interpolation_order=interpolation_order,
            integration_time=integration_time,
            number_of_points=0,
            data_type=data_type,
        )

        # EXTRAPOLATION polynomial
        self.t_fine = np.linspace(-0.2, 1.4, int(5e4))

        # ATTRIBUTEs
        self.solar_r = 6.96e5  # in km
        self.borders = borders  #TODO: should I get the borders myself from the opened file?
        self.number_of_points = number_of_points

    def to_cartesian(self, data: np.ndarray) -> np.ndarray:
        """
        To calculate the heliographic cartesian positions given a ndarray of index positions.

        Args:
            data (np.ndarray): the index positions.
            data_info (dict[str, np.ndarray]): the data information.

        Returns:
            np.ndarray: the corresponding heliographic cartesian positions.
        """

        # Initialisation
        data[0, :] = data[0, :] * self.borders['dx'] + self.borders['xmin']
        data[1, :] = data[1, :] * self.borders['dx'] + self.borders['ymin']
        data[2, :] = data[2, :] * self.borders['dx'] + self.borders['zmin']
        return data

    def reprocessed_interpolation(self, cube_index: int) -> np.ndarray:

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
        # print(f'min and max of t_fine are {np.min(t_fine), np.max(t_fine)}')

        # COORDs new        
        coords = self.get_coords(t_fine, params)
        return coords  #TODO: remember that these coords are negative and positive cube indexes
