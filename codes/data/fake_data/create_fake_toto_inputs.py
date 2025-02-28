"""
To create fake FITS files based on the original data to see if Dr. Auchere's new_toto.pro code
works as intended.
"""

# IMPORTs
import os

# IMPORTs alias
import numpy as np
import multiprocessing as mp

# IMPORTs sub
import PIL.Image
import matplotlib.pyplot as plt
from typing import Any
from astropy.io import fits
from dataclasses import dataclass, field

# IMPORTs personal
from common import root_path, Decorators

# PLACEHOLDERs type annotation
LockProxy = Any
ValueProxy = Any



@dataclass(slots=True, repr=False, eq=False)
class SdoHeaderInformation:
    """
    To store the information from the SDO FITS header.
    """
    
    # CONSTANTs
    d_sun: float  # distance to sun in m
    resolution_arcsec: float  # pixel resolution in arcsec
    sun_center: tuple[float, float]  # the center of the sun in the image in pixels
    
    # PLACEHOLDERs
    dx: float = field(init=False)

    def __post_init__(self) -> None:
        # PIXEL size km
        self.dx = ((
            (np.tan(np.deg2rad(self.resolution_arcsec / 3600) / 2) * self.d_sun) * 2
        ) / 1e3)


class CreateFakeTotoInputs:
    """
    To create the fake data inputs (i.e. FITs + PNGs) for Dr. Auchere's new_toto.pro code.
    """

    def __init__(self, sphere_radius: float, fake_len: int = 413, processes: int = 4) -> None:
        """
        To create the fake data inputs for Dr. Auchere's new_toto.pro code.

        Args:
            sphere_radius (float): the radius of the sphere in km.
            fake_len (int, optional): the number of fake data to create. Defaults to 413.
        """

        # ATTRIBUTEs
        self.fake_len = fake_len
        self.sphere_radius = sphere_radius
        self.processes = processes
        
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
            'sdo files': os.path.join(main_path, 'sdo'),  # todo change if config.yml is updated
            'save fits': os.path.join(root_path, 'data/fake_data/fits'),
            'save png': os.path.join(root_path, 'data/fake_data/png'),
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

        # MULTIPROCESSING setup
        processes_nb = min(self.processes, self.fake_len)

        if processes_nb > 1:
            manager = mp.Manager()
            lock = manager.Lock()
            value = manager.Value('i', 0)

            # MULTIPROCESSING run
            processes: list[mp.Process] = [None] * processes_nb
            for i in range(processes_nb):
                p = mp.Process(
                    target=self.sdo_fake_data_multiprocessing,
                    kwargs={'value': value, 'lock': lock},
                )
                p.start()
                processes[i] = p
            for p in processes: p.join()
            manager.shutdown()
        else:
            # NO MULTIPROCESSING
            for i in range(self.fake_len): self.create_sun_image_sdo(index=i)
            
    def sdo_fake_data_multiprocessing(self, value: ValueProxy, lock: LockProxy) -> None:
        """
        To create the fake data for the SDO masks in multiprocessing.

        Args:
            value (ValueProxy): A value proxy from the multiprocessing.Manager class.
            lock (LockProxy): A lock proxy from the multiprocessing.Manager class.
        """

        while True:
            # COUNTER value
            with lock:
                index = value.value
                if index >= self.fake_len: return
                value.value += 1
            
            self.create_sun_image_sdo(index)
        
    def create_sun_image_sdo(self, index: int) -> None:
        """
        To create the fake fits file image.

        Args:
            index (int): the index of the SDO FITS file.
        """

        # HDU open
        sdo_file_name = f'AIA_fullhead_{index:03d}.fits.gz'
        sdo_hdul = fits.open(
            os.path.join(self.paths['sdo files'], sdo_file_name),
            mode='readonly',
        )
        fake_hdul = fits.HDUList([hdu.copy() for hdu in sdo_hdul])
        
        # INFO formatting
        hdu = fake_hdul[0] 
        sdo_info = SdoHeaderInformation(
            sun_center=(hdu.header['X0_MP'], hdu.header['Y0_MP']),
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
        fake_hdul[0].data = image

        # SAVE hdul
        fake_hdul.writeto(os.path.join(self.paths['save fits'], sdo_file_name), overwrite=True)
        print(f'SAVED - {sdo_file_name}', flush=True)

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
        r = np.arange(0, self.sphere_radius, resolution / 8)
        theta = np.linspace(0, 2 * np.pi, len(r))
        r, theta = np.meshgrid(r, theta)

        # COORDs cartesian
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return np.stack([x.ravel(), y.ravel()], axis=0)



if __name__=='__main__':

    CreateFakeTotoInputs(sphere_radius=6.96e5, processes=10)
