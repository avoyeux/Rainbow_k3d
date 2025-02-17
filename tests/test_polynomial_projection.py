"""
To check the sdo positions.
"""

# IMPORTs
import os
import h5py

# IMPORTs alias
import numpy as np

# IMPORTs sub
from typing import Any

# IMPORTs personal
from common import Decorators
from common.server_connection import SSHMirroredFilesystem
from projection.polynomial_projection import OrthographicalProjection

# PLACEHOLDERs type annotation
QueueProxy = Any



class PrintValuesProjection(OrthographicalProjection):

    @Decorators.running_time
    def __init__(self, **kwargs):

        super().__init__(**kwargs)

        self.run()

    def data_setup(
            self,
            input_queue: QueueProxy | None = None,
            index_list: np.ndarray | None = None,
        ) -> None:
        """
        Override the setup just to fetch the constants.

        Args:
            input_queue (QueueProxy | None, optional): placeholder. Defaults to None.
            index_list (np.ndarray | None, optional): placeholder. Defaults to None.
        """

        with h5py.File(os.path.join(self.paths['data'], self.filename), 'r') as H5PYFile:
            self.constants = self.get_global_constants(H5PYFile)

    def get_used_sdo_positions(self) -> list[tuple[float, float, float]]:
        """
        Get the sdo positions used in the projection.

        Returns:
            list[tuple[float, float, float]]: the sdo positions.
        """

        filepaths = [
            self.sdo_timestamps[self.constants.dates[time_index].decode('utf8')[:-3]]
            for time_index in self.constants.time_indexes
        ]
        print(f'time indexes are {self.constants.time_indexes}')
        print(f'filepaths are {filepaths}')

        # SERVER fetch
        if self.in_local: filepaths = SSHMirroredFilesystem.remote_to_local(filepaths)

        sdo_positions = [
            tuple([
                round(float(x), 2)
                for x in self.sdo_image(path).sdo_pos
            ])
            for path in filepaths
        ]
        print(f'sdo_positions are {sdo_positions}')

        if self.in_local: SSHMirroredFilesystem.cleanup()
        return sdo_positions

    def as_string(self, sdo_positions: list[tuple[float, float, float]]) -> list[str]:
        """
        Convert the sdo positions to a list of strings.

        Args:
            sdo_positions (list[tuple[float, float, float]]): the sdo positions.

        Returns:
            list[str]: the converted sdo positions.
        """

        list_str = [
            ', '.join(map(str, sdo_pos))
            for sdo_pos in sdo_positions
        ]
        return list_str

    @Decorators.running_time    
    def format_outputs(self) -> list[str]:
        """
        Format the outputs to be printed.

        Returns:
            list[str]: the formatted outputs.
        """

        sdo_positions = self.as_string(self.get_used_sdo_positions())
        formatted_list = [
            f"time{self.constants.time_indexes[i]:03d} - sdo: ({sdo_positions[i]})."
            for i in range(len(self.constants.time_indexes))
        ]
        return formatted_list

    def __str__(self) -> str:
        """
        Return the formatted outputs as a string.

        Returns:
            str: the formatted outputs as a string.
        """

        formatted_list = self.format_outputs()
        return '\n'.join(formatted_list)

    def to_csv(self) -> None:

        pass



if __name__=='__main__':
    instance = PrintValuesProjection(
        filename='data.h5',
        verbose=2,
        processes=1,
        polynomial_order=[4],
        data_type='No duplicates',
        plot_choices=[
            'polar',
        ],
        fake_hdf5=False,
        flush=True,
    )
    print(instance)
    print('Done.')
