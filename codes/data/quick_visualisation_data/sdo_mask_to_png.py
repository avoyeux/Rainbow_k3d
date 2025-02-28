"""
To save the SDO masks as png for quick visualisation.
"""

# IMPORTs
import os
import glob

# IMPORTs alias
import numpy as np
import multiprocessing as mp

# IMPORTs sub
from typing import Any
from astropy.io import fits
import matplotlib.pyplot as plt

# IMPORTs personal
from common import config, Decorators

# PLACEHOLDERs type annotation
QueueProxy = Any



class SdoMasksToPng:
    """
    To save the sdo images contained in FITS files as PNG images.
    """

    @Decorators.running_time
    def __init__(self, processes: int | None = None) -> None:
        """
        Saving the SDO masks as PNG images.

        Args:
            processes (int | None, optional): number of processes to use. None uses the process
                number from the config file. Defaults to None.
        """

        # ARGUMENTs
        if processes is None:
            self.processes: int = config.run.processes
        else:
            self.processes = processes

        # ATTRIBUTEs setup
        self.paths = self.paths_setup()

        # RUN
        self.multiprocess()

    def paths_setup(self) -> dict[str, str]:
        """
        Setting up the needed paths as a dict.

        Returns:
            dict[str, str]: the paths to the needed directories.
        """

        # PATHs formatting
        paths = {
            'sdo': config.path.dir.data.sdo,
            'png': os.path.join(config.path.dir.data.sdo, 'png'),
        }

        # PATHs create
        for key in ['png']: os.makedirs(paths[key], exist_ok=True)
        return paths
    
    def multiprocess(self) -> None:
        """
        Multiprocessing to save the SDO masks as PNG images.
        """

        # FILEPATHs
        filepaths = glob.glob(os.path.join(self.paths['sdo'], '*.fits.gz'))
        filepaths_len = len(filepaths)

        # MULTIPROCESSING setup
        manager = mp.Manager()
        input_queue = manager.Queue()
        nb_processes = min(self.processes, filepaths_len)
        for path in filepaths: input_queue.put(path)
        for _ in range(nb_processes): input_queue.put(None)
        
        # RUN multiprocessing
        processes: list[mp.Process] = [None] * nb_processes
        for i in range(nb_processes):
            p = mp.Process(target=self.save_png, args=(input_queue,))
            p.start()
            processes[i] = p
        for p in processes: p.join()

    def save_png(self, input_queue: QueueProxy) -> None:
        """
        Saving the SDO masks as PNG images.

        Args:
            input_queue (QueueProxy): the queue to get the filepaths from.
        """

        while True:
            
            # INPUTs
            filepath = input_queue.get()
            if filepath is None: break
            filename = os.path.basename(filepath)

            # DATA
            image: np.ndarray = fits.getdata(filepath, 0).astype('uint8')

            # PLOT
            png_name = filename.replace('.fits.gz', '.png')
            plt.figure(figsize=(10, 10))
            plt.imshow(image, cmap='gray', interpolation='none', origin='lower')
            plt.axis('off')
            plt.savefig(
                fname=os.path.join(self.paths['png'], png_name),
                dpi=200,
            )
            plt.close()

            # PROGRESS
            print(f'SAVED - {png_name}', flush=True)



if __name__ == '__main__':

    # RUN
    SdoMasksToPng(processes=14)
