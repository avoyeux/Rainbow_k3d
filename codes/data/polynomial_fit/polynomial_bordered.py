"""
To redefine the polynomial coordinates so that the range of the fit doesn't pass through the Sun's
surface.
Furthermore, the coordinates (gotten from the computing of the polynomial fit) that are too far
from the initial borders of the cube are also removed.
"""

# IMPORTs alias
import numpy as np

# IMPORTs personal
from codes.projection.format_data import CubeInformation
from codes.data.polynomial_fit.base_polynomial_fit import GetPolynomialFit



class ProcessedBorderedPolynomialFit(GetPolynomialFit):
    """
    To process the polynomial fit positions so that the final result is a curve with a set number
    of points defined from the Sun's surface. If not possible, then the fit stops at a predefined
    distance. The number of points in the resulting stays the same.
    """

    def __init__(
            self,
            filepath: str,
            group_path: str,
            polynomial_order: int,
            number_of_points: int,
            dx: float,
        ) -> None:
        """
        To process the polynomial fit positions so that the final result is a curve with a set
        number of points defined from the Sun's surface. If not possible, then the fit stops at a
        predefined distance. The number of points in the resulting stays the same.

        Args:
            filepath (str): the path to the polynomial fit data.
            group_path (str): the path to the group in the HDF5 file.
            polynomial_order (int): the order of the polynomial fit to consider.
            number_of_points (int): the number of positions to consider in the final polynomial
                fit.
            dx (float): the voxel resolution in km.
        """

        # PARENT
        super().__init__(
            filepath=filepath,
            group_path=group_path,
            polynomial_order=polynomial_order,
            number_of_points=0,
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
            CubeInformation: the information for the reprocessed polynomial fit.
        """

        # PARAMs polynomial
        params = self.get_params(cube_index)

        # COORDs cartesian
        coords = self.get_coords(params)
        coords = self.to_cartesian(coords)
        
        # FILTER inside the Sun
        distance_sun_center = np.sqrt(coords[0]**2 + coords[1]**2 + coords[2]**2)  # in km
        sun_filter = distance_sun_center < self.solar_r

        # FILTER far from Sun
        x_filter = (coords[0] < - self.solar_r * 1.30) | (coords[0] > - self.solar_r * 0.90)
        y_filter = (coords[1] < - self.solar_r * 0.7) | (coords[1] > -self.solar_r * 0.02)
        z_filter = (coords[2] < - self.solar_r * 0.3) | (coords[2] > self.solar_r * 0.5)
        
        # FILTERs combine
        to_filter = (x_filter | y_filter | z_filter | sun_filter)
        new_t = self.t_fine[~to_filter]

        # RANGE filtered polynomial
        self.t_fine = np.linspace(np.min(new_t), np.max(new_t), self.number_of_points)

        # COORDs new        
        coords = self.get_coords(params)

        # DATA reformatting
        information = CubeInformation(
            order=self.order,
            xt_min=self.polynomial_info.xt_min,
            yt_min=self.polynomial_info.yt_min,
            zt_min=self.polynomial_info.zt_min,
            coords=coords,
        )
        return information
