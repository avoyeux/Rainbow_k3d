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
    def __init__(self, everything: bool = True, **kwargs):

        super().__init__(everything, both_cubes=True, make_screenshots=True, cube_version='both', **kwargs)
        self.Structure()

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
                     'all_data_alf': self.Calculations(self.cubes_all_data_1),
                     'all_data_kar': self.Calculations(self.cubes_all_data_2),
                     'no_duplicates_stereo_alf_old': self.Calculations(self.cubes_no_duplicates_init_STEREO_1),
                     'no_duplicates_stereo_kar_old': self.Calculations(self.cubes_no_duplicates_init_STEREO_2),
                     'no_duplicates_sdo_alf_old': self.Calculations(self.cubes_no_duplicates_init_SDO_1),
                     'no_duplicates_sdo_kar_old': self.Calculations(self.cubes_no_duplicates_init_SDO_2),
                     'no_duplicates_alf_old': self.Calculations(self.cubes_no_duplicate_init_1),
                     'no_duplicates_kar_old': self.Calculations(self.cubes_no_duplicate_init_2),
                     'no_duplicates_stereo_alf_new': self.Calculations(self.cubes_no_duplicates_new_STEREO_1),
                     'no_duplicates_stereo_kar_new': self.Calculations(self.cubes_no_duplicates_new_STEREO_2),
                     'no_duplicates_sdo_alf_new': self.Calculations(self.cubes_no_duplicates_new_SDO_1),
                     'no_duplicates_sdo_kar_new': self.Calculations(self.cubes_no_duplicates_new_SDO_2),
                     'no_duplicates_alf_new': self.Calculations(self.cubes_no_duplicate_new_1),
                     'no_duplicates_kar_new': self.Calculations(self.cubes_no_duplicate_new_2),
                     f'interval_{self.time_interval}_all_data_alf': self.Calculations(self.time_cubes_all_data_1),
                     f'interval_{self.time_interval}_all_data_kar': self.Calculations(self.time_cubes_all_data_2),
                     f'interval_{self.time_interval}_no_duplicates_alf': self.Calculations(self.time_cubes_no_duplicate_1),
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
                STEREO_surfaces.append(0)
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
    Stats(time_interval='20min')
    # MaskStats()
