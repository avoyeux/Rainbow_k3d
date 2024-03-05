"""
To do some stats on the voxels from Animation_3D_main.py.
It is general stats to try and find a periodicity in the total volume of the pixels, on plasma "rain"
velocity and so on.
"""

# Imports 
import os
import re

import numpy as np
import pandas as pd

from PIL import Image
from glob import glob
from pathlib import Path
from astropy.io import fits
from sparse import COO, stack
from typeguard import typechecked
from Animation_3D_main import Data, CustomDate



class Stats(Data):
    """
    To do some initial stats.
    """

    @typechecked
    def __init__(self, everything: bool = True, edges: bool = False,  **kwargs):

        super().__init__(all_data=True, no_duplicate=True, 
                         both_cubes='kar', make_screenshots=True, cube_version='both', 
                         time_intervals_all_data=True, time_intervals_no_duplicate=True, duplicates=True,
                         **kwargs)
        if not edges:
            self.Structure()
        else:
            self.Finding_edges()

    def Pre_processing(self, cubes):
        """
        To process the data so that even the None values become COO matrices
        """

        if isinstance(cubes, list):
            shape = (self.cubes_shape[1], self.cubes_shape[2], self.cubes_shape[3]) 
            print(f'the initial cube shape is {shape}')
            empty_coo = COO(data=[], coords=[], shape=shape)
            for loop in range(len(cubes)):
                if cubes[loop] is None:
                    cubes[loop] = empty_coo
            result = stack(cubes, axis=0)
            print(f'the shape of the concatenate is {result.shape}')
            return result
        else:
            print(f"the initial cube ain't a list. it's shape is {cubes.shape}")
            return cubes

    def Calculations(self, cubes):
        """
        Has the different calculations done on a given cube set
        """
        
        cubes = self.Pre_processing(cubes)
        summing = COO.sum(cubes, axis=(1, 2, 3))
        return summing.todense()
    
    def Structure(self):
        """
        Creation of the data dictionary and hence the corresponding .csv file.
        """

        data_dict = {'dates (seconds)': self.Dates_sub(),
                     'all_data_kar': self.Calculations(self.cubes_all_data_2),
                     'no_duplicates_stereo_kar_old': self.Calculations(self.cubes_no_duplicates_init_STEREO_2),
                     'no_duplicates_sdo_kar_old': self.Calculations(self.cubes_no_duplicates_init_SDO_2),
                     'no_duplicates_kar_old': self.Calculations(self.cubes_no_duplicate_init_2),
                     'no_duplicates_stereo_kar_new': self.Calculations(self.cubes_no_duplicates_new_STEREO_2),
                     'no_duplicates_sdo_kar_new': self.Calculations(self.cubes_no_duplicates_new_SDO_2),
                     'no_duplicates_kar_new': self.Calculations(self.cubes_no_duplicate_new_2),
                     f'interval_{self.time_interval}_all_data_kar': self.Calculations(self.time_cubes_all_data_2),
                     f'interval_{self.time_interval}_no_duplicates_kar': self.Calculations(self.time_cubes_no_duplicate_2)}
        
        df = pd.DataFrame(data_dict)
        df.to_csv('../STATS/k3d_volumes.csv', index=False)


    def Dates_sub(self):
        """
        To get the dates of every cube in seconds starting from the first instance, i.e 00:00:00 on the 23-07-2012.
        """

        path = '../STEREO/avg/'
        avg_paths = [filepath for filepath in sorted(Path(path).glob('*.png'))]
        avg_pattern = re.compile(r'''(?P<number>\d{4})_
                                 (?P<date>\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2})''', re.VERBOSE)
        
        dates = []
        for path in avg_paths:
            pattern = avg_pattern.match(os.path.basename(path))

            if pattern:
                date = CustomDate(pattern.group('date'))
                dates.append(self.Date_to_seconds(date))
            else:
                raise ValueError(f"avg filename {path} didn't match")
            
        
        return np.array(dates)

    def Date_to_seconds(self, date):
        """
        To change a date in day, hour, minute, seconds format to a seconds format.
        """

        return (((date.day - 23) * 24 + date.hour) * 60 + date.minute) * 60 + date.second
    
    def Finding_edges(self):
        """
        Code to find the edges to then make the filament representation smaller.
        """

        cubes = self.Pre_processing(self.cubes_all_data_2)
        total_cube = COO.any(cubes, axis=0)
        cube = total_cube.todense()

        ax1_cube = np.any(cube, axis=(1, 2)).astype('uint8')
        ax2_cube = np.any(cube, axis=(0, 2)).astype('uint8')
        ax3_cube = np.any(cube, axis=(0, 1)).astype('uint8')

        ax1_cube_first = np.flatnonzero(ax1_cube)[0]
        ax1_cube_last = np.flatnonzero(ax1_cube)[-1]
        ax2_cube_first = np.flatnonzero(ax2_cube)[0]
        ax2_cube_last = np.flatnonzero(ax2_cube)[-1]
        ax3_cube_first = np.flatnonzero(ax3_cube)[0]
        ax3_cube_last = np.flatnonzero(ax3_cube)[-1]

        print(f'the cube shape is {cube.shape}')
        print(f'ax1_cube type is {ax1_cube.dtype}')
        print(f'the first and last for axis 1 are {ax1_cube_first}, {ax1_cube_last} shape {ax1_cube.shape}')
        print(f'the first and last for axis 2 are {ax2_cube_first}, {ax2_cube_last} shape {ax2_cube.shape}')
        print(f'the first and last for axis 3 are {ax3_cube_first}, {ax3_cube_last} shape {ax3_cube.shape}')

        solar_r = 6.96e5 
        # xt_min = -solar_r*1.25
        # xt_max = -solar_r*0.95
        # yt_min = -solar_r*0.5
        # yt_max = solar_r*0.0
        # zt_min = -solar_r*0.32
        # zt_max = solar_r*0.32
        xt_min = -solar_r * 1.258
        xt_max = -solar_r * 0.943
        yt_min = -solar_r * 0.454
        yt_max = -solar_r * 0.072
        zt_min = -solar_r * 0.263
        zt_max = solar_r * 0.277

        len_ax1 = abs(xt_min - xt_max)
        len_ax2 = abs(yt_min - yt_max)
        len_ax3 = abs(zt_min - zt_max)

        nw_xt_min = xt_min + self._length_dx * (ax3_cube_first - 1)
        nw_xt_max = xt_min + self._length_dx * (ax3_cube_last + 1)
        # nw_xt_max = xt_max - self._length_dx * (cube.shape[0] - (ax1_cube_last + 1)) 
        nw_yt_min = yt_min + self._length_dx * (ax2_cube_first - 1)
        nw_yt_max = yt_min + self._length_dx * (ax2_cube_last + 1)
        # nw_yt_max = yt_max - self._length_dx * (cube.shape[1] - (ax2_cube_last + 1)) 
        nw_zt_min = zt_min + self._length_dx * (ax1_cube_first - 1)
        nw_zt_max = zt_min + self._length_dx * (ax1_cube_last + 1)
        # nw_zt_max = zt_max - self._length_dx * (cube.shape[2] - (ax3_cube_last + 1))

        print(f'the new x min and max are {nw_xt_min/solar_r} and {nw_xt_max/solar_r}') 
        print(f'the new y min and max are {nw_yt_min/solar_r} and {nw_yt_max/solar_r}')
        print(f'the new z min and max are {nw_zt_min/solar_r} and {nw_zt_max/solar_r}')


class MaskStats:
    """
    To save some mask stats.
    """

    def __init__(self):

        # Functions
        self.Paths()
        self.Numbers()
        self.Saving()

    def Paths(self):
        """
        For the paths to the masks and to save the csv stats file.
        """

        main_path = os.path.join(os.getcwd(), '..')

        self.paths = {'Main': main_path,
                      'SDO_fits': os.path.join(main_path, 'sdo'),
                      'STEREO_masks': os.path.join(main_path, 'STEREO', 'masque_karine'),
                      'STATS': os.path.join(main_path, 'STATS')}
        os.makedirs(self.paths['STATS'], exist_ok=True)
    
    def Numbers(self):
        """
        To get the numbers of the files (using a pattern) to match the data correctly.
        """

        pattern = re.compile(r'''AIA_fullhead_(\d{3}).fits.gz''')
        STEREO_paths = glob(os.path.join(self.paths['SDO_fits'], '*.fits.gz'))

        self.numbers = sorted([int(pattern.match(os.path.basename(path)).group(1)) for path in STEREO_paths])

    def Downloads(self):
        """
        To get the data from the corresponding masks.
        """

        SDO_surfaces = []
        STEREO_surfaces = []
        STEREO_dlon = 0.075
        for nb in self.numbers:
            SDO_hdul = fits.open(os.path.join(self.paths['SDO_fits'], f'AIA_fullhead_{nb:03d}.fits.gz'))
            SDO_surfaces.append(np.sum(SDO_hdul[0].data) * SDO_hdul[0].header['CDELT1']**2)
            SDO_hdul.close()

            STEREO_path = os.path.join(self.paths['STEREO_masks'], f'frame{nb:04d}.png')
            if os.path.exists(STEREO_path):
                print(f'STEREO path nb{nb} found.')
                image = np.mean(Image.open(STEREO_path), axis=2)  # as the png has 3 channels
                all_white = image.size * 255  # image in uint8
                STEREO_surfaces.append((all_white - np.sum(image)) / 255 * (STEREO_dlon**2))  # as the mask is when image==0
            else:
                print(f'STEREO path nb{nb} not found.')
                STEREO_surfaces.append(np.nan)
        return SDO_surfaces, STEREO_surfaces

    def Saving(self):
        """
        To save the data in a csv file.
        """

        sdo_surface, stereo_surface = self.Downloads()

        data = {'Image nb': self.numbers, 'STEREO mask': stereo_surface, 'SDO mask': sdo_surface}
        df = pd.DataFrame(data)
        df.to_csv(os.path.join(self.paths['STATS'], 'k3d_mask_area.csv'), index=False)



if __name__=='__main__':
    # Stats(time_interval='20min')
    # # MaskStats()

    Stats(time_interval='20min')
