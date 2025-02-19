"""
To check the results gotten from the animation code.
"""

# IMPORTs
import os

# IMPORTs alias
import numpy as np
import pandas as pd

# IMPORTs personal
from animation.animation_code import Setup
from animation.animation_dataclasses import *



class DataPrint(Setup):


    def __init__(self, **kwargs):

        super().__init__(**kwargs)

        self.sdo_pos = self.cubes.sdo_pos * self.constants.dx

    def format_outputs(self) -> list[str]:
        """
        Format the outputs to be printed.

        Returns:
            list[str]: the formatted outputs.
        """

        time_indexes = self.constants.time_indexes
        string_sdo = self.as_string(self.sdo_pos)
        string_stereo = self.as_string(self.sdo_pos)

        formatted_list = [
            f"time{time_indexes[i]:03d} - sdo: ({string_sdo[i]}), stereo: ({string_stereo[i]})."
            for i in range(len(time_indexes))
        ]
        return formatted_list

    def as_string(self, data: np.ndarray) -> list[str]:
        """
        Convert the data to a list of strings.

        Args:
            data (np.ndarray): the data to be converted.

        Returns:
            list[str]: the converted data.
        """
         
        string_list = [
            f"{', '.join(map(str, map(lambda x: round(x, 2), values.tolist())))}"
            for values in data
        ]
        return string_list
    
    def __str__(self) -> str:
        """
        Return the formatted outputs as a string.

        Returns:
            str: the formatted outputs as a string.
        """

        formatted_list = self.format_outputs()
        return '\n'.join(formatted_list)
    
    def to_csv(self) -> None:

        # SETUP
        path_save = os.path.join(self.paths['code'], 'tests', 'sdo_pos_animation.csv')

        # DATA
        data = {
            'time': self.constants.time_indexes,
            'x_pos': [x[0] for x in self.sdo_pos],
            'y_pos': [y[1] for y in self.sdo_pos],
            'z_pos': [z[2] for z in self.sdo_pos],
        }
        df = pd.DataFrame(data)

        # SAVE
        df.to_csv(path_save, index=False)



if __name__=='__main__':

    instance = DataPrint(
        filename='data.h5',
        sun=False,
        all_data=False,
        no_duplicate=False,
        all_data_integration=False,
        no_duplicate_integration=False,
        line_of_sight_SDO=False,
        line_of_sight_STEREO=False,
        pov_sdo=True,
        pov_stereo=True,
        polynomial=False,
        polynomial_order=4,
        with_feet=False,
        only_fake_data=False,
    )
    print(instance)
    instance.to_csv()
    instance.close()
    print('FINISHED')