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
            image_nb: int,
            output_shape: tuple[int, int],
            borders: dict[str, tuple[int, int]],
            direction: str = 'anticlockwise',
            theta_offset: int | float = 0,
            channel_axis: None | int = None,
            **kwargs,
        ) -> None:

        # Attributes
        self.image_nb = image_nb
        self.output_shape = output_shape
        self.borders = borders
        self.direction = direction
        self.theta_offset = theta_offset
        self.channel_axis = channel_axis

        self.kwargs = kwargs

        # Setup
        self._initial_checks()
        self.paths = self._paths_setup()
        self.data = self._open_data()

    @classmethod
    def get_polar_image(
            cls,
            image_nb: int,
            output_shape: tuple[int, int],
            borders: dict[str, tuple[int, int]],
            direction: str = 'anticlockwise',
            theta_offset: int | float = 0,
            channel_axis: None | int = None,
            **kwargs,
        ) -> np.ndarray:

        instance = cls(
            image_nb=image_nb,
            output_shape=output_shape,
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
            raise ValueError(f"'{self.direction} not in permitted options. You need to choose between {', '.join(direction_options)}.")

    def _paths_setup(self) -> dict[str, str]:

        # Check main path
        main_path = '/home/avoyeux/Documents/avoyeux/'
        if not os.path.exists(main_path): main_path = '/home/avoyeux/old_project/avoyeux/'
        if not os.path.exists(main_path): raise ValueError(f"\033[1;31mThe main path {main_path} not found.")

        paths = {
            'sdo': os.path.join(main_path, 'sdo'),
        }
        return paths
    
    def _open_data(self) -> dict[str, any]:

        filename = f"AIA_fullhead_{self.image_nb:03d}.fits.gz"
        hdul = astropy.io.fits.open(os.path.join(self.paths['sdo'], filename))
        header = hdul[0].header

        data_info = {
            'image': hdul[0].data,
            'center': (header['X0_MP'], header['Y0_MP']),
            'sun radius': header['RSUN_REF'],
            'dx': (np.tan(np.deg2rad(header['CDELT1'] / 3600) / 2) * header['DSUN_OBS']) * 2,  # CUNIT is 'arcsec'
        }
        print(f"dx is {data_info['dx']}")
        data_info['max index'] = max(self.borders['radial distance']) * 1e6 / data_info['dx']
        data_info['min index'] = min(self.borders['radial distance']) * 1e6 / data_info['dx']
        print(f"the max index is {data_info['max index']}")
        hdul.close()
        return data_info

    def _slice_image(self, image: np.ndarray) -> np.ndarray:
        """To cut the image so that the bounds are the same than for the inputted borders"""

        # Radial distance section
        new_dx = max(self.borders['radial distance']) * 1e6 / image.shape[1]
        min_radial_index = round(min(self.borders['radial distance']) * 1e6 / new_dx)

        # Polar angle section
        d_theta = 360 / image.shape[0]
        max_polar_index = round(max(self.borders['polar angle']) / d_theta)
        min_polar_index = round(min(self.borders['polar angle']) / d_theta)
        return image[min_polar_index:max_polar_index + 1, min_radial_index:]

    def _coordinates_cartesian_to_polar(self) -> np.ndarray:
        """
        To change the cartesian coordinates to the polar ones.
        """

        print(f"_coord: max of image is {np.max(self.data['image'])}")
        image = skimage.transform.warp_polar(
            image=self.data['image'],
            center=self.data['center'],
            output_shape=self.output_shape,
            channel_axis=self.channel_axis,
            radius=self.data['max index'],  #TODO: most likely an error here
        )
        print(f'_coord0: max of image is {np.max(image)}')

        # Corrections
        if self.theta_offset != 0:
            image = self._rotate_polar(image, self.theta_offset)
            print(f"_coord1: max of image is {np.max(image)}")
        if self.direction == 'clockwise':
            image = np.flip(image, axis=0)
            print(f"_coord2: max of image is {np.max(image)}")
        
        return self._slice_image(image)
    
    def _rotate_polar(self, polar_image: np.ndarray, angle: int | float) -> np.ndarray:
        
        d_theta = 360 / polar_image.shape[0]  #TODO: need to check if it is the right ax
        shift = round(angle / d_theta)
        return np.roll(polar_image, shift=shift, axis=0)
    


if __name__ == '__main__':

    CartesianToPolar(
        image_nb=1,
        output_shape=(1_000, 1_000),
        borders= {
            'radial distance': (690, 870),
            'polar angle': (245, 295),
        },
        direction='clockwise',
        theta_offset=90,
        channel_axis=None,
    )