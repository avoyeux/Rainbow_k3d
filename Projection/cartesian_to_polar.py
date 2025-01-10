"""
To change a cartesian image to a polar one. Still work in progress.
"""

import os
import skimage
import astropy 

import numpy as np

import astropy.io.fits
import skimage.transform



class CartesianToPolar:
    """
    To change coordinates, or an image, from the cartesian representation to polar coordinates.
    """

    def __init__(
            self,
            filepath: str,
            borders: dict[str, tuple[int, int]],
            direction: str = 'anticlockwise',
            theta_offset: int | float = 0,
            channel_axis: None | int = None,
            **kwargs,
        ) -> None:

        # Attributes
        self.filepath = filepath
        self.borders = borders
        self.direction = direction
        self.theta_offset = theta_offset
        self.channel_axis = channel_axis

        self.kwargs = kwargs

        # Setup
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
            channel_axis: None | int = None,
            **kwargs,
        ) -> dict[str | dict[str, float | np.ndarray], float]:

        instance = cls(
            filepath=filepath,
            borders=borders,
            direction=direction,
            theta_offset=theta_offset,
            channel_axis=channel_axis,
            **kwargs,
        )
        return instance._coordinates_cartesian_to_polar()

    def _initial_checks(self) -> None:

        # Direction keyword argument check
        direction_options = ['clockwise', 'anticlockwise']
        if self.direction not in direction_options:
            raise ValueError(
                f"'{self.direction} not in permitted options. "
                f"You need to choose between {', '.join(direction_options)}."
            )

    def _paths_setup(self) -> dict[str, str]:

        # Check main path
        main_path = '/home/avoyeux/Documents/avoyeux/'
        if not os.path.exists(main_path): main_path = '/home/avoyeux/old_project/avoyeux/'
        if not os.path.exists(main_path):
            raise ValueError(f"\033[1;31mThe main path {main_path} not found.")

        paths = {
            'sdo': os.path.join(main_path, 'sdo'),
        }
        return paths
    
    def _open_data(self) -> dict[str, any]:
        
        index = 0 if 'AIA' in self.filepath else 1
        hdul = astropy.io.fits.open(self.filepath)
        header = hdul[index].header

        # ORGANISE data
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
        """To cut the image so that the bounds are the same than for the inputted borders"""

        # Radial distance section
        min_radial_index = round(min(self.borders['radial distance']) * 1e3 / dx)

        # Polar angle section
        max_polar_index = round(max(self.borders['polar angle']) / d_theta)
        min_polar_index = round(min(self.borders['polar angle']) / d_theta)
        return image[min_polar_index:max_polar_index + 1, min_radial_index:]

    def _coordinates_cartesian_to_polar(self) -> dict[str | dict[str, float | np.ndarray], float]:
        """
        To change the cartesian coordinates to the polar ones.
        """

        # Setup image shape depending on dx and dtheta
        theta_nb_pixels = round(360 / self.data_info['d_theta'])
        radial_nb_pixels = round(max(self.borders['radial distance']) * 1e3 / self.data_info['dx'])

        # Re-calculating dx and dtheta as round() needed to be used.
        new_d_theta = 360 / theta_nb_pixels  
        new_dx = max(self.borders['radial distance']) * 1e3 / radial_nb_pixels

        image = skimage.transform.warp_polar(
            image=self.data_info['image'],
            center=self.data_info['center'],
            output_shape=(theta_nb_pixels, radial_nb_pixels),
            channel_axis=self.channel_axis,
            radius=self.data_info['max index'],
        )

        # Corrections
        if self.theta_offset != 0: image = self._rotate_polar(image, new_d_theta)
        if self.direction == 'clockwise': image = np.flip(image, axis=0)
        info = {
            'image': {
                'data': self._slice_image(image, new_dx, new_d_theta).T,
                'dx': new_dx,
                'd_theta': new_d_theta,
            },
            'dx': self.data_info['dx'],
            'd_theta': self.data_info['d_theta'],
        }
        # print(f"dx is {info['dx']}")
        # print(f"d_theta is {info['d_theta']}", flush=True)
        return info
    
    def _rotate_polar(self, polar_image: np.ndarray, d_theta: float) -> np.ndarray:
        
        shift = round(self.theta_offset / d_theta)
        return np.roll(polar_image, shift=-shift, axis=0)



if __name__ == '__main__':

    CartesianToPolar(
        image_nb=1,
        borders= {
            'radial distance': (690, 870),
            'polar angle': (245, 295),
        },
        direction='clockwise',
        theta_offset=90,
        channel_axis=None,
    )
