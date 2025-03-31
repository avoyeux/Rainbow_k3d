"""
To store the parent class containing the reprojection methods.
"""

# IMPORTs alias
import numpy as np

# IMPORTs personal
from codes.projection.helpers.projection_dataclasses import (
    CubeInformation, CubePointer, UniqueCubePointer, FakeCubePointer, ProjectedCube
)



class BaseReprojection:
    """
    To store the base methods used for the reprojection of data cubes from cartesian coordinates
    to polar coordinates in the SDO image reference frame.
    """

    def __init__(self) -> None:
        """
        To initialise the class containing the reprojection methods.
        The initialisation is empty as the class is only used as a parent class.
        """

        pass

    def cartesian_pos(self, data: CubeInformation, dx: float) -> CubeInformation:
        """
        To calculate the heliographic cartesian positions given a ndarray of index positions.

        Args:
            data (CubeInformation): the heliographic cartesian positions of the protuberance.

        Returns:
            CubeInformation: the heliographic cartesian positions.
        """

        data.coords[0, :] = data.coords[0, :] * dx + data.xt_min
        data.coords[1, :] = data.coords[1, :] * dx + data.yt_min
        data.coords[2, :] = data.coords[2, :] * dx + data.zt_min
        return data

    def matrix_rotation(self, data: np.ndarray, sdo_pos: np.ndarray) -> tuple[np.ndarray, float]:
        """ 
        Gives the cartesian positions of the voxels in an orthonormal coordinates system centred on
        SDO's position and with the new z-axis pointing to the Sun's center.

        Args:
            data (np.ndarray): the (x, y, z) coordinates of the data voxels in heliocentric
                cartesian coordinates.
            sdo_pos (np.ndarray): the position of the SDO satellite in heliocentric cartesian
                coordinates.

        Returns:
            tuple[np.ndarray, float]: the voxel coordinates in the new reference frame, with the
                normalisation constant of the Z-axis (later needed to calculate the projected polar
                radius from the disk center to each voxel).
        """

        # DATA open
        x, y, z = data
        a, b, c = - sdo_pos.astype('float64')
        sign = a / abs(a)

        # CONSTANTs normalisation
        new_N_x = 1 / np.sqrt(1 + b**2 / a**2 + ((a**2 + b**2) / (a * c))**2)
        new_N_y = a * c / np.sqrt(a**2 + b**2)
        new_N_z = 1 /  np.sqrt(a**2 + b**2 + c**2)

        # COORDS new
        new_x = 1 / new_N_x + sign * new_N_x * (x + y * b / a - z * (a**2 + b**2) / (a * c))
        new_y = 1 / new_N_y + sign * new_N_y * (-x * b / (a * c) + y / c)
        new_z = 1 / new_N_z + sign * new_N_z * (x * a + y * b + z * c)
        
        # DATA return
        coords = np.stack([new_x, new_y, new_z], axis=0)
        return coords, new_N_z
    
    def get_polar_image(self, data: tuple[np.ndarray, float]) -> np.ndarray:
        """ 
        Gives the polar coordinates in SDO's image reference frame of the protuberance voxels.

        Args:
            data (tuple[np.ndarray, float]): the heliocentric cartesian positions of the
                protuberance voxels.

        Returns:
            np.ndarray: (r, theta) of the voxels in polar coordinates centred on the disk center as
                seen from SDO and with theta starting from the projected solar north pole.
        """
        
        # DATA open
        coords, z_norm = data
        x, y, z = coords

        # IMAGE polar coordinates
        rho_polar = np.arccos(z / np.sqrt(x**2 + y**2 + z**2))
        theta_polar = (y / np.abs(y)) * np.arccos(x / np.sqrt(x**2 + y**2))
        theta_polar = np.rad2deg((theta_polar + 2 * np.pi) % (2 * np.pi))

        # UNITs to km
        rho_polar = np.tan(rho_polar) / z_norm  # ? why did I put this here ?
        return np.stack([rho_polar, theta_polar], axis=0)
    
    def get_angles(self, coords: np.ndarray) -> np.ndarray:
        """ 
        Gives the angle between the polynomial fit and the SDO image plane. 

        Args:
            coords (np.ndarray): the coordinates of the voxels in heliocentric cartesian
                coordinates.

        Returns:
            np.ndarray: the angles between the coordinates (for b_{n+1} - b_{n}) and SDO's image
                plane. Information needed to correct the velocities seen in 2D in SDO's image.
        """
        
        x, y, z = coords

        # DIRECTIONS a_n = b_{n+1} - b{n}
        x_direction = x[1:] - x[:-1]
        y_direction = y[1:] - y[:-1]
        z_direction = z[1:] - z[:-1]

        # ANGLE rho - image plane
        theta_spherical = np.arccos(
            z_direction / np.sqrt(x_direction**2 + y_direction**2 + z_direction**2)
        )
        theta_spherical -= np.pi / 2
        return theta_spherical
    
    def get_polar_image_angles(self, data: tuple[np.ndarray, float]) -> np.ndarray:
        """ 
        Gives the polar coordinates (r, theta) in the created SDO image (i.e. centred on the disk
        center and with theta starting from the north pole direction). Furthermore, the angle of
        the polynomial fit relative to the SDO image plane is also computed.

        Args:
            data (tuple[np.ndarray, float]): the voxel position in heliocentric cartesian
                coordinates.

        Returns:
            np.ndarray: (r, theta, angle) in the SDO image reference frame.
        """

        # todo add an explanation in the equation .md file.

        # DATA open
        coords, _ = data

        # ANGLES
        angles = self.get_angles(coords)  # todo need to change it so that I take the middle point

        # POLAR pos
        rho_polar, theta_polar = self.get_polar_image(data)
        return np.stack([rho_polar[:-1], theta_polar[:-1], angles], axis=0)
    
    def format_cube(
            self,
            data: CubePointer | UniqueCubePointer | FakeCubePointer,
            dx: float,
            index: int,
            name: str,
            colour: str,
            sdo_pos: np.ndarray,
        ) -> ProjectedCube:
        """
        To format the cube data for the projection.

        Args:
            data (CubePointer | UniqueCubePointer | FakeCubePointer): the data cube to be
                formatted.
            dx (float): the pixel size of the data cube in km.
            index (int): the index of the corresponding real data cube.
            name (str): the name of the data used as a label in the final plot.
            colour (str): the colour of the data cube for the plot.
            sdo_pos (np.ndarray): the position of the SDO satellite.

        Returns:
            ProjectedCube: the formatted and reprojected data cube.
        """

        # CUBE formatting
        cube = CubeInformation(
            xt_min=data.xt_min,
            yt_min=data.yt_min,
            zt_min=data.zt_min,
            coords=data[index],
        )
        cube = self.cartesian_pos(cube, dx=dx)
        cube = self.get_polar_image(self.matrix_rotation(data=cube.coords, sdo_pos=sdo_pos))
        
        # PROJECTION formatting
        return ProjectedCube(data=cube, name=name, colour=colour)
