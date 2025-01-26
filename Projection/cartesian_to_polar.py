"""
To change an image to the corresponding polar representation using skimage.transform.warp_polar().
For now, this code only works in a specific case looking at an AIA FITS image to find the solar
disk's center and get the corresponding image defined inside borders in polar coordinate values.
"""

# IMPORTs
import os
import sunpy
import astropy
import skimage
import astropy 

# IMPORTS alias
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u

# IMPORTs sub
import astropy.io.fits
import skimage.transform
import sunpy.coordinates
from typing import Any
from dataclasses import dataclass, field

# IMPORTs personal
from common import root_path
from Projection.polynomial_projection_dataclasses import ImageBorders

# ANNOTATIONs alias
AstropyFitsHeaderType = Any



@dataclass(slots=True, repr=False, eq=False)
class ImageInfo:
    """
    To store the information needed for the image processing.
    """

    # DATA
    image: np.ndarray
    sdo_pos: np.ndarray
    sun_radius: float  # ? should I take solar_r or RSUN_REF from the header?

    # IMAGE properties
    image_borders: ImageBorders
    sun_center: tuple[float, float]
    resolution_km: float

    # PLACEHOLDERs
    resolution_angle: float = field(init=False)
    max_index: float = field(init=False)

    def __post_init__(self):
        
        self.resolution_angle = 360 / (2 * np.pi * self.sun_radius / (self.resolution_km * 1e3))
        self.max_index = max(self.image_borders.radial_distance) * 1e3 / self.resolution_km


@dataclass(slots=True, frozen=True, repr=False, eq=False)
class PolarImageInfo:
    """
    To store the information gotten by creating the polar image.
    """

    # DATA
    image: np.ndarray
    sdo_pos: np.ndarray

    # IMAGE properties
    resolution_km: float
    resolution_angle: float


class CartesianToPolar:
    """
    To change an SDO FITS image to the polar representation centred on the Sun's disk.
    The outputted image borders depends on the specified borders in polar coordinates (r, theta).
    """

    def __init__(
            self,
            filepath: str,
            borders: ImageBorders,
            direction: str = 'anticlockwise',
            theta_offset: int | float = 0,
            plot: bool = False,
            **kwargs,
        ) -> None:
        """
        To change an SDO fits image to the polar representation centred on the Sun's disk.
        The image borders will depend on the specified method.
        The result of the processing is gotten from the .coordinates_cartesian_to_polar() after
        initialisation of the class.

        Args:
            filepath (str): path to the SDO AIA FITS file.
            borders (ImageBorders): the borders for the polar SDO image plot.
            direction (str, optional): the direction of the polar angle.
                Defaults to 'anticlockwise'.
            theta_offset (int | float, optional): the offset needed to get the wanted polar angle.
                Defaults to 0.
            plot (bool, optional): Rough plot to see if the code works properly. Defaults to False.
        """

        # ATTRIBUTEs
        self.filepath = filepath
        self.borders = borders
        self.direction = direction
        self.theta_offset = theta_offset
        self.plot = plot
        self.kwargs = kwargs

        # RUN
        self._initial_checks()
        self.paths = self._paths_setup()
        self.image_info = self._open_data()

    @classmethod
    def get_polar_image(
            cls,
            filepath: str,
            borders: ImageBorders,
            direction: str = 'anticlockwise',
            theta_offset: int | float = 0,
            **kwargs,
        ) -> PolarImageInfo:
        """
        Class method to get the processed polar SDO image without needing to initialise the class
        first.

        Args:
            filepath (str): path to the SDO AIA FITS file.
            borders (ImageBorders): the borders for the polar SDO image plot.
            direction (str, optional): the direction of the polar angle.
                Defaults to 'anticlockwise'.
            theta_offset (int | float, optional): the offset needed to get the wanted polar angle.
                Defaults to 0.

        Returns:
            PolarImageInfo: the processed polar SDO image with some corresponding information.
        """

        instance = cls(
            filepath=filepath,
            borders=borders,
            direction=direction,
            theta_offset=theta_offset,
            **kwargs,
        )
        return instance.coordinates_cartesian_to_polar()

    def _initial_checks(self) -> None:
        """
        To check the direction string attribute to make sure it is recognised.

        Raises:
            ValueError: if the direction string attribute is wrong.
        """

        # CHECK direction
        direction_options = ['clockwise', 'anticlockwise']
        if self.direction not in direction_options:
            raise ValueError(
                f"'{self.direction} not in permitted options. "
                f"You need to choose between {', '.join(direction_options)}."
            )

    def _paths_setup(self) -> dict[str, str]:
        """
        To get a dictionary with the needed directory paths.

        Raises:
            ValueError: if the main_path isn't recognised.

        Returns:
            dict[str, str]: contains all the needed directory or filepaths.
        """

        # PATHs formatting
        paths = {'sdo': os.path.join(root_path, 'sdo')}
        return paths
    
    def _open_data(self) -> ImageInfo:
        """
        To open the FITS file, get and compute the necessary info for the image processing.

        Returns:
            ImageInfo: the needed information for the image processing.
        """
        
        # OPEN fits
        index = 0 if 'AIA' in self.filepath else 1
        hdul = astropy.io.fits.open(self.filepath)
        header = hdul[index].header

        # DATA formatting
        data_info = ImageInfo(
            image=hdul[index].data,
            sdo_pos=self.carrington_to_cartesian(header),
            sun_radius=header['RSUN_REF'],
            image_borders=self.borders,
            sun_center=(header['Y0_MP'], header['X0_MP']),
            resolution_km=((
                (np.tan(np.deg2rad(header['CDELT1'] / 3600) / 2) * header['DSUN_OBS']) * 2
            ) / 1e3), # in km       
        )
        hdul.close()
        return data_info
    
    def carrington_to_cartesian(self, header: AstropyFitsHeaderType) -> np.ndarray:
        """
        To convert the Carrington coordinates to the Cartesian one.

        Args:
            header (AstropyFitsHeaderType): the header of the SDO FITS file.

        Returns:
            np.ndarray: the corresponding Cartesian coordinates.
        """

        # COORDs frame
        coords = sunpy.coordinates.frames.HeliographicCarrington(
            header['CRLN_OBS'] * u.deg,
            header['CRLT_OBS'] * u.deg,
            header['DSUN_OBS'] * u.m,
            obstime=header['DATE-OBS'],
            observer='self',
        )
        coords = coords.represent_as(astropy.coordinates.CartesianRepresentation)

        # In km
        result = np.array([
            coords.x.to(u.km).value,
            coords.y.to(u.km).value,
            coords.z.to(u.km).value,
        ])
        return result

    def _slice_image(self, image: np.ndarray, dx: float, d_theta: float) -> np.ndarray:
        """
        Slicing the polar image so that the image borders are the same that the ones specified in
        .__init__().

        Args:
            image (np.ndarray): the polar image.
            dx (float): the image resolution in km.
            d_theta (float): the image resolution in degrees.

        Returns:
            np.ndarray: the sliced polar image.
        """

        # DISTANCE radial
        min_radial_index = round(min(self.borders.radial_distance) * 1e3 / dx)

        # ANGLE polar
        max_polar_index = round(max(self.borders.polar_angle) / d_theta)
        min_polar_index = round(min(self.borders.polar_angle) / d_theta)
        return image[min_polar_index:max_polar_index + 1, min_radial_index:]

    def coordinates_cartesian_to_polar(self) -> PolarImageInfo:
        """
        To change the image from a cartesian representation to the corresponding polar one.

        Returns:
            PolarImageInfo: the polar image with some corresponding information.
        """

        # SHAPE image
        theta_nb_pixels = round(360 / self.image_info.resolution_angle)
        radial_nb_pixels = round(self.image_info.max_index)

        # RE-CALCULATION dx and dtheta (round() was used).
        new_d_theta = 360 / theta_nb_pixels  
        new_dx = max(self.borders.radial_distance) * 1e3 / radial_nb_pixels

        # POLAR image
        image = skimage.transform.warp_polar(
            image=self.image_info.image,
            center=self.image_info.sun_center,
            output_shape=(theta_nb_pixels, radial_nb_pixels),
            radius=self.image_info.max_index,
        )

        # RESULT plot
        if self.plot: self.plot_result(image)

        # CORRECTIONs
        if self.theta_offset != 0: image = self._rotate_polar(image, new_d_theta)
        if self.direction == 'clockwise': image = np.flip(image, axis=0)

        # DATA formatting
        polar_image_info = PolarImageInfo(
            image=self._slice_image(image, new_dx, new_d_theta).T,
            sdo_pos=self.image_info.sdo_pos,
            resolution_km=new_dx,
            resolution_angle=new_d_theta,
        )
        return polar_image_info
    
    def _rotate_polar(self, polar_image: np.ndarray, d_theta: float) -> np.ndarray:
        """
        To rotate the image so that the theta angle angle starts where you want it to (given the
        specified theta offset).

        Args:
            polar_image (np.ndarray): the polar SDO image.
            d_theta (float): the polar image resolution in degrees.

        Returns:
            np.ndarray: the corresponding rotated polar image.
        """
        
        shift = round(self.theta_offset / d_theta)
        return np.roll(polar_image, shift=-shift, axis=0)

    def plot_result(self, image: np.ndarray) -> None:
        """
        To plot the polar image to see if the code works properly.

        Args:
            image (np.ndarray): the polar image.
        """

        # CUTs
        lower_cut = np.nanpercentile(image, 2)
        higher_cut = np.nanpercentile(image, 99.99)

        # SATURATION
        image[image < lower_cut] = lower_cut
        image[image > higher_cut] = higher_cut
        
        if not np.any(image == 0):
            image = np.log(image.T)

            plt.figure(figsize=(18, 10))
            plt.imshow(image, interpolation='none')
            ax = plt.gca()
            ax.minorticks_on()
            ax.set_aspect('auto')
            plt.savefig(f'testing{np.round(np.random.random(1), 5)}.png', dpi=800)
            plt.close()
            print('done')