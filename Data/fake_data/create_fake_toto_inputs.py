"""
To create fake FITS files based on the original data to see if Dr. Auchere's new_toto.pro code
works as intended.
"""

# todo I also need to create fake .mat files to convert to .save files to make sure that the 
# re-projection problem doesn't stem from my code.

# IMPORTs
import os

# IMPORTs alias
import numpy as np

# IMPORTs sub
import PIL.Image
import matplotlib.pyplot as plt
from astropy.io import fits
from dataclasses import dataclass, field

# IMPORTs personal
from common import root_path, Decorators



@dataclass(slots=True, repr=False, eq=False)
class SdoHeaderInformation:
    """
    To store the information from the SDO FITS header.
    """

    sun_center: tuple[float, float]
    resolution_arcsec: float
    d_sun: float
    dx: float = field(init=False)

    def __post_init__(self) -> None:
        # PIXEL size km
        self.dx = ((
            (np.tan(np.deg2rad(self.resolution_arcsec / 3600) / 2) * self.d_sun) * 2
        ) / 1e3)
        print(f'dx: {self.dx}')


class CreateFakeTotoInputs:
    """
    To create the fake data inputs (i.e. FITs + PNGs) for Dr. Auchere's new_toto.pro code.
    """

    def __init__(self, sphere_radius: float, fake_len: int = 413) -> None:

        self.sphere_radius = sphere_radius
        self.fake_len = fake_len

        # RUN
        self.paths = self.paths_setup()
        self.create_stereo_fake_data()
        self.create_sdo_fake_data()


    def paths_setup(self) -> dict[str, str]:
        """
        Gives the paths to the different needed directories.

        Returns:
            dict[str, str]: the paths to the different directories.
        """

        # PATHs setup
        main_path = os.path.join(root_path, '..')

        # PATHs formatting
        paths = {
            'main': main_path,
            'stereo files': os.path.join(main_path, 'STEREO', 'masque_karine'),
            'sdo files': os.path.join(main_path, 'sdo'),
            'save fits': os.path.join(root_path, 'Data/fake_data/fits'),
            'save png': os.path.join(root_path, 'Data/fake_data/png'),
        }

        # PATHs create
        for key in ['save fits', 'save png']: os.makedirs(paths[key], exist_ok=True)
        return paths

    @Decorators.running_time
    def create_stereo_fake_data(self) -> None:
        """
        To create the fake data for the STEREO masks.
        """

        real_image = PIL.Image.open(os.path.join(self.paths['stereo files'], 'frame0000.png'))
        fake_image = np.zeros(real_image.size, dtype=np.uint8)
        real_image.close()

        for index in range(self.fake_len):
            image = PIL.Image.fromarray(fake_image)
            image.save(os.path.join(self.paths['save png'], f'frame{index:04d}.png'))

    @Decorators.running_time
    def create_sdo_fake_data(self) -> None:
        """
        To create the fake data for the SDO masks.
        """

        for index in range(self.fake_len):
            
            sdo_file_name = f'AIA_fullhead_{index:03d}.fits.gz'
            sdo_hdul = fits.open(
                os.path.join(self.paths['sdo files'], sdo_file_name),
                mode='readonly',
            )
            fake_hdul = fits.HDUList([hdu.copy() for hdu in sdo_hdul])

            # HDU populate
            fake_hdul = self.create_sun_image_sdo(fake_hdul)

            # SAVE hdul
            fake_hdul.writeto(os.path.join(self.paths['save fits'], sdo_file_name), overwrite=True)

    def create_sun_image_sdo(self, hdul: fits.HDUList) -> fits.HDUList:
        """
        To create the sun image in the SDO FITS file.

        Args:
            hdul (fits.HDUList): the HDUList of the SDO FITS file.

        Returns:
            fits.HDUList: the HDUList of the SDO FITS file with the sun image.
        """
        
        # INFO formatting
        hdu = hdul[0] 
        sdo_info = SdoHeaderInformation(
            sun_center=(hdu.header['Y0_MP'], hdu.header['X0_MP']),
            resolution_arcsec=hdu.header['CDELT1'],
            d_sun=hdu.header['DSUN_OBS'],
        )
        
        # COORDs sun
        x, y = self.circle(sdo_info.dx)

        # SUN IMAGE creation
        image = np.zeros(hdu.data.shape, dtype=np.uint8)
        x_indexes = (x / sdo_info.dx) + sdo_info.sun_center[0]
        y_indexes = (y / sdo_info.dx) + sdo_info.sun_center[1]

        # INDEXEs positive integers
        x_indexes = np.round(x_indexes).astype(int)
        y_indexes = np.round(y_indexes).astype(int)

        # SAVE image
        image[y_indexes, x_indexes] = 1
        # self.plot(image)
        hdul[0].data = image
        return hdul

    def plot(self, image: np.ndarray) -> None:
        """
        To plot the result for a visual check.

        Args:
            image (np.ndarray): the image to plot.
        """

        plt.figure()
        plt.imshow(image, cmap='gray', interpolation='none')
        plt.colorbar()
        plt.savefig('test.png', dpi=800)
        plt.close()

    def circle(self, resolution: float) -> np.ndarray:
        """
        To create the cartesian coordinates of a circle.

        Args:
            resolution (float): the resolution of the SDO image in km.

        Returns:
            np.ndarray: the cartesian coordinates of the circle.
        """

        # COORDs polar
        r = np.arange(0, self.sphere_radius, resolution / 10)
        theta = np.linspace(0, 2 * np.pi, len(r))
        r, theta = np.meshgrid(r, theta)

        # COORDs cartesian
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return np.stack([x.ravel(), y.ravel()], axis=0)



if __name__=='__main__':

    CreateFakeTotoInputs(sphere_radius=6.96e5)