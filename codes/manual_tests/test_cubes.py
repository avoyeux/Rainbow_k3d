"""
This is to get the sdo position values from the AIA images to check if I am using the wrong ones
for the volumetric data.
"""

# IMPORTs
import os
import re
import glob
import sunpy
import astropy

# IMPORTs alias
import numpy as np
import pandas as pd
import multiprocessing as mp

# IMPORTs sub
import sunpy.coordinates
from typing import Any
from astropy.io import fits
from astropy import units as u

# IMPORTs personal
from common import Decorators, root_path

# PLACEHOLDERs type annotation
QueueProxy = Any



class OpenAIAImages:
    """
    To get the sdo positions from the AIA images.
    """

    @Decorators.running_time
    def __init__(self, processes: int = 5) -> None:
        """
        To get the sdo positions from the AIA images.

        Args:
            processes (int, optional): the number of processes to use. Defaults to 5.
        """
        
        # ATTRIBUTEs
        self.processes = processes

        self.paths = self.setup_paths()
        self.filepaths = sorted(glob.glob(os.path.join(self.paths['data'], '*.fits.gz')))
        self.nb_files = len(self.filepaths)
        self.file_numbers = self.get_file_numbers()

        # RUN
        self.sdo_positions = self.multiprocessing()
        print(f'self.sdo_positions shape is {self.sdo_positions.shape}')

    def setup_paths(self) -> dict[str, str]:
        """
        Setup the paths for the data and save directories.

        Returns:
            dict[str, str]: the paths for the data and save directories. 
        """

        # PATHs formatting
        main_path = os.path.join(root_path, '..')
        paths = {
            'main': main_path,
            'data': os.path.join(main_path, 'sdo'),
            'save': os.path.join(root_path, 'tests'),
        }

        # CHECKs
        for key in ['save']: os.makedirs(paths[key], exist_ok=True)
        return paths
    
    @Decorators.running_time
    def multiprocessing(self) -> np.ndarray:
        """
        To multiprocess the opening of the AIA images and processing of the sdo positions.

        Returns:
            np.ndarray: the sdo positions.
        """

        # MULTIPROCESSING
        nb_processes = min(self.processes, self.nb_files)
        processes: list[mp.Process] = [None] * nb_processes
        # ARGUMENTs
        manager = mp.Manager()
        input_queue = manager.Queue()
        output_queue = manager.Queue()
        for i, path in enumerate(self.filepaths): input_queue.put((i, path))
        for _ in range(nb_processes): input_queue.put(None)
        # RUN
        for i in range(nb_processes):
            p = mp.Process(target=self.open_images, args=(input_queue, output_queue,))
            p.start()
            processes[i] = p
        for p in processes: p.join()
        # RESULTs
        results: list[np.ndarray] = [None] * self.nb_files
        while not output_queue.empty():
            identifier, result = output_queue.get()
            results[identifier] = result
        # RESULTs formatting
        return np.stack(results, axis=0)

    def open_images(self, input_queue: QueueProxy, output_queue: QueueProxy) -> None:
        """
        To open the AIA images and get the sdo positions.

        Args:
            input_queue (QueueProxy): the process input data.
            output_queue (QueueProxy): the sdo positions and corresponding identifier.
        """

        while True:
            # CHECKs
            arg = input_queue.get()
            if arg is None: return
            i, path = arg

            # OPEN
            header = fits.getheader(path, ext=0)
            coords = sunpy.coordinates.frames.HeliographicCarrington(
                header['CRLN_OBS'] * u.deg,
                header['CRLT_OBS'] * u.deg,
                header['DSUN_OBS'] * u.m,
                obstime=header['DATE-OBS'],
                observer='self',
            )
            coords = coords.represent_as(astropy.coordinates.CartesianRepresentation)

            # CONVERSION
            result = np.array([
                coords.x.to(u.km).value,
                coords.y.to(u.km).value,
                coords.z.to(u.km).value,
            ])
            output_queue.put((i, np.round(result, 2)))

    def get_file_numbers(self) -> list[int]:
        """
        To get the file numbers from the AIA images.

        Raises:
            ValueError: if the file number could not be parsed.

        Returns:
            list[int]: the file numbers.
        """

        # SETUP pattern
        pattern = re.compile(r'AIA_fullhead_(?P<number>\d{3})\.fits\.gz')

        # NUMBERs file id
        file_numbers: list[int] = [None] * self.nb_files
        for i, path in enumerate(self.filepaths):

            search = pattern.search(path)
            if search is not None:
                file_numbers[i] = int(search.group('number'))
            else:
                raise ValueError(f'Could not parse the file number from: {path}')
        return file_numbers

    def __str__(self) -> str:
        """
        To format the sdo positions for printing.

        Returns:
            str: the formatted sdo positions.
        """

        formatted_list = [
            f'time{i:03d} - sdo: ({", ".join(map(str, self.sdo_positions[i]))}).'
            for i in self.file_numbers
        ]
        return '\n'.join(formatted_list)

    @Decorators.running_time
    def to_csv(self) -> None:
        """
        To save the sdo positions to a csv file.
        """

        # DATA
        data = {
            'time': self.file_numbers,
            'x_pos': self.sdo_positions[:, 0],
            'y_pos': self.sdo_positions[:, 1],
            'z_pos': self.sdo_positions[:, 2],
        }
        df = pd.DataFrame(data)

        # SAVE
        df.to_csv(os.path.join(self.paths['save'], 'sdo_pos_AIA.csv'), index=False)



if __name__ == '__main__':

    instance = OpenAIAImages(
        processes=10,
    )
    instance.to_csv()
    print(instance)
