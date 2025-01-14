"""
To change an image to the corresponding polar representation using skimage.transform.warp_polar().
For now, this code only works in a specific case looking at an AIA FITS image to find the solar
disk's center and get the corresponding image defined inside borders in polar coordinate values.
"""

# IMPORTs
import os
import skimage
import astropy 

# IMPORTS alias
import numpy as np

# IMPORTs sub
import astropy.io.fits
import skimage.transform

import matplotlib.pyplot as plt


class CartesianToPolar:
    """
    To change an SDO FITS image to the polar representation centred on the Sun's disk.
    The outputted image borders depends on the specified borders in polar coordinates (r, theta).
    """

    def __init__(
            self,
            filepath: str,
            borders: dict[str, tuple[int, int]],
            direction: str = 'anticlockwise',
            theta_offset: int | float = 0,
            **kwargs,
        ) -> None:
        """
        To change an SDO fits image to the polar representation centred on the Sun's disk.
        The image borders will depend on the specified method.
        The result of the processing is gotten from the .coordinates_cartesian_to_polar() after
        initialisation of the class.

        Args:
            filepath (str): path to the SDO AIA FITS file.
            borders (dict[str, tuple[int, int]]): the final borders for the polar SDO image.
            direction (str, optional): the direction of the polar angle.
                Defaults to 'anticlockwise'.
            theta_offset (int | float, optional): the offset needed to get the wanted polar angle.
                Defaults to 0.
        """

        # ATTRIBUTEs
        self.filepath = filepath
        self.borders = borders
        self.direction = direction
        self.theta_offset = theta_offset
        self.kwargs = kwargs

        # RUN
        self._initial_checks()
        self.paths = self._paths_setup()
        self.data_info = self._open_data()

    @classmethod
    def get_polar_image(
            cls,
            filepath: str,
            borders: dict[str, tuple[int, int]],
            direction: str = 'anticlockwise',
            theta_offset: int | float = 0,
            **kwargs,
        ) -> dict[str | dict[str, float | np.ndarray], float]:
        """
        Class method to get the processed polar SDO image without needing to initialise the class
        first.

        Args:
            filepath (str): path to the SDO AIA FITS file.
            borders (dict[str, tuple[int, int]]): the final borders for the polar SDO image.
            direction (str, optional): the direction of the polar angle.
                Defaults to 'anticlockwise'.
            theta_offset (int | float, optional): the offset needed to get the wanted polar angle.
                Defaults to 0.

        Returns:
            dict[str | dict[str, float | np.ndarray], float]: contains the final polar image inside
                the specified borders, with the initial dx and dtheta used and the new image's dx
                and dtheta.
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

        # CHECK path
        main_path = '/home/avoyeux/Documents/avoyeux/'
        if not os.path.exists(main_path): main_path = '/home/avoyeux/old_project/avoyeux/'
        if not os.path.exists(main_path):
            raise ValueError(f"\033[1;31mThe main path {main_path} not found.")

        # PATHs save
        paths = {'sdo': os.path.join(main_path, 'sdo')}
        return paths
    
    def _open_data(self) -> dict[str, any]:
        """
        To open the FITS file, get and compute the necessary info for the image processing.

        Returns:
            dict[str, any]: information needed for the processing, e.g. the image, sun center.
        """
        
        # OPEN fits
        index = 0 if 'AIA' in self.filepath else 1
        hdul = astropy.io.fits.open(self.filepath)
        header = hdul[index].header

        # DATA organise
        data_info = {
            'image': hdul[index].data,
            'center': (header['Y0_MP'], header['X0_MP']),
            'dx': ((
                (np.tan(np.deg2rad(header['CDELT1'] / 3600) / 2) * header['DSUN_OBS']) * 2
            ) / 1e3),  # in km
        }
        sun_radius = header['RSUN_REF']
        sun_perimeter = 2 * np.pi * sun_radius
        data_info['d_theta'] = 360 / (sun_perimeter / (data_info['dx'] * 1e3))
        data_info['max index'] = max(self.borders['radial distance']) * 1e3 / data_info['dx']
        hdul.close()
        return data_info

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
        min_radial_index = round(min(self.borders['radial distance']) * 1e3 / dx)

        # ANGLE polar
        max_polar_index = round(max(self.borders['polar angle']) / d_theta)
        min_polar_index = round(min(self.borders['polar angle']) / d_theta)
        return image[min_polar_index:max_polar_index + 1, min_radial_index:]

    def coordinates_cartesian_to_polar(self) -> dict[str | dict[str, float | np.ndarray], float]:
        """
        To change the image from a cartesian representation to the corresponding polar one.

        Returns:
            dict[str | dict[str, float | np.ndarray], float]: the final processed image with some
                needed information to be able to properly locate the position of the final image.
        """

        # SHAPE image
        theta_nb_pixels = round(360 / self.data_info['d_theta'])
        radial_nb_pixels = round(self.data_info['max index'])

        # RE-CALCULATION dx and dtheta (round() was used).
        new_d_theta = 360 / theta_nb_pixels  
        new_dx = max(self.borders['radial distance']) * 1e3 / radial_nb_pixels

        # POLAR image
        image = skimage.transform.warp_polar(
            image=self.data_info['image'],
            center=self.data_info['center'],
            output_shape=(theta_nb_pixels, radial_nb_pixels),
            radius=self.data_info['max index'],
        )
  
        # TEST plotting
        # lower_cut = np.nanpercentile(image, 2)
        # higher_cut = np.nanpercentile(image, 99.99)

        # # SATURATION
        # image[image < lower_cut] = lower_cut
        # image[image > higher_cut] = higher_cut
        
        # if not np.any(image == 0):
        #     image = np.log(image.T)

        #     plt.figure(figsize=(18, 10))
        #     plt.imshow(image, interpolation='none')
        #     ax = plt.gca()
        #     ax.minorticks_on()
        #     ax.set_aspect('auto')
        #     plt.savefig(f'testing{np.round(np.random.random(1), 5)}.png', dpi=800)
        #     plt.close()
        #     print('done')

        # CORRECTIONs
        if self.theta_offset != 0: image = self._rotate_polar(image, new_d_theta)
        if self.direction == 'clockwise': image = np.flip(image, axis=0)

        # SAVE data
        info = {
            'image': {
                'data': self._slice_image(image, new_dx, new_d_theta).T,
                'dx': new_dx,
                'd_theta': new_d_theta,
            },
            'dx': self.data_info['dx'],
            'd_theta': self.data_info['d_theta'],
        }
        return info
    
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
