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

from pathlib import Path
from sparse import COO, concatenate, stack
from typeguard import typechecked
from Animation_3D_main import Data, CustomDate


class Stats(Data):
    """
    To do some initial stats.
    """

    @typechecked
    def __init__(self, everything: bool = True, **kwargs):

        super().__init__(everything, both_cubes=True, **kwargs)
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
                     'no_duplicates_stereo_alf': self.Calculations(self.cubes_no_duplicates_STEREO_1),
                     'no_duplicates_stereo_kar': self.Calculations(self.cubes_no_duplicates_STEREO_2),
                     'no_duplicates_sdo_alf': self.Calculations(self.cubes_no_duplicates_SDO_1),
                     'no_duplicates_sdo_kar': self.Calculations(self.cubes_no_duplicates_SDO_2),
                     'no_duplicates_alf': self.Calculations(self.cubes_no_duplicate_1),
                     'no_duplicates_kar': self.Calculations(self.cubes_no_duplicate_2),
                     f'interval_{self.time_interval}_all_data_alf': self.Calculations(self.time_cubes_all_data_1),
                     f'interval_{self.time_interval}_all_data_kar': self.Calculations(self.time_cubes_all_data_2),
                     f'interval_{self.time_interval}_no_duplicates_alf': self.Calculations(self.time_cubes_no_duplicate_1),
                     f'interval_{self.time_interval}_no_duplicates_kar': self.Calculations(self.time_cubes_no_duplicate_2)}
        
        df = pd.DataFrame(data_dict)
        df.to_csv('../k3d_stats.csv', index=False)


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
                date = CustomDate.parse_date(pattern.group('date'))
                dates.append(self.Date_to_seconds(date))
            else:
                raise ValueError(f"avg filename {path} didn't match")
        return np.array(dates)

    def Date_to_seconds(self, date):
        """
        To change a date in day, hour, minute, seconds format to a seconds format.
        """

        return (((date.day - 23) * 24 + date.hour) * 60 + date.minute) * 60 + date.second
    

if __name__=='__main__':
    Stats(time_interval='20min')
