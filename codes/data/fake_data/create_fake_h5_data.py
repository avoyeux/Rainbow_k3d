"""
To create fake .h5 data to then be converted to .save data to then be used by my code.
Should be able to tell if there is any actual projection errors from the totality of my code.
"""

# IMPORTs
import os
import h5py
import sparse

# IMPORTs alias
import numpy as np
import multiprocessing as mp

# IMPORTs sub
from typing import Any

# IMPORTs personal
from common import config, Decorators
from codes.data.fake_data.base_fake_hdf5 import BaseFakeHDF5

# PLACEHOLDERs type annotation
LockProxy = Any
ValueProxy = Any



class FakeData(BaseFakeHDF5):
    """
    Creates fake h5 data to then be converted to .save file to finally be used by my re-projection
    code.
    """

    def __init__(
            self,
            nb_of_points: int,
            sphere_radius: tuple[float, float],
            nb_of_cubes: int,
            increase_factor: float,
            processes: int | None = None,
            flush: bool | None = None,
            verbose: int | None = None,
        ) -> None:
        """
        To create fake h5 data to then be converted to .save data.

        Args:
            nb_of_points (int): the number of points to be used in spherical phi direction.
            sphere_radius (tuple[float, float]): the min and max radius of the fake sphere in km.
            nb_of_cubes (int): the number of cubes to create (and hence the final number of hdf5
                files created).
            increase_factor (float): the cube border increase factor.
            processes (int | None, optional): the number of processes used in the multiprocessing.
                When None, it uses the config file value. Defaults to None.
            flush (bool | None, optional): deciding to flush the buffer each time there is a print.
                When None, it uses the config file value. Defaults to None.
            verbose (int | None, optional): the verbosity level. When None, it uses the config file
                value. Defaults to None.
        """

        # PARENT
        super().__init__(
            nb_of_points=nb_of_points,
            sphere_radius=sphere_radius,
            increase_factor=increase_factor,
        )

        # CONFIGURATION attributes
        self.processes = config.run.processes if processes is None else processes
        self.flush = config.run.flush if flush is None else flush
        self.verbose = config.run.verbose if verbose is None else verbose

        # ATTRIBUTEs
        self.nb_of_cubes = nb_of_cubes
        self.multiprocessing = True if self.processes > 1 else False

        # PATH update
        self.paths['save'] = config.path.dir.fake.h5
        os.makedirs(self.paths['save'], exist_ok=True)

        # RUN
        self.create_fake_data()

    @Decorators.running_time
    def create_fake_data(self) -> None:
        """
        To create the fake mat data and save it to .h5 files.
        """

        # COORDs
        sphere_surface = self.fake_sphere_surface()
        sphere_surface = self.to_index(sphere_surface)[[2, 1, 0]]  # ? why the changes in axes
        # added this to follow the same shape pattern as the original save files. 

        # ARRAY 3d
        shape = np.max(sphere_surface, axis=1) + 1
        array = sparse.COO(coords=sphere_surface, data=1, shape=shape).todense()

        # MULTI-PROCESSING
        if self.multiprocessing:
            # MULTI-PROCESSING setup
            nb_processes = min(self.processes, self.nb_of_cubes)
            manager = mp.Manager()
            counter = manager.Value('i', 0)
            lock = manager.Lock()

            # MUTLI-PROCESSING run
            processes: list[mp.Process] = [None] * nb_processes
            for i in range(nb_processes):
                p =  mp.Process(target=self.sub_create_fake_data, args=(array, lock, counter))
                p.start()
                processes[i] = p
            for p in processes: p.join()

        else:
            # HDF5 save
            for i in range(self.nb_of_cubes):
                filepath = os.path.join(self.paths['save'], f'cube{i:03d}.h5')
                self.create_h5(filepath=filepath, data=array)

    def sub_create_fake_data(self, data: np.ndarray, lock: LockProxy, counter: ValueProxy) -> None:
        """
        To create the .h5 files using multiprocessing.

        Args:
            data (np.ndarray): the data to save in the .h5 files.
            lock (LockProxy): multiprocessing.Manager.Lock.
            counter (ValueProxy): the counter to keep track of the number of cubes saved.
        """

        while True:

            # COUNTER value
            with lock:
                if counter.value >= self.nb_of_cubes: return
                i = counter.value
                counter.value += 1
            
            # HDF5 save
            filepath = os.path.join(self.paths['save'], f'cube{i:03d}.h5')
            self.create_h5(filepath=filepath, data=data)

    def create_h5(self, filepath: str, data: np.ndarray) -> None:
        """
        To create the .h5 file given the desired filepath and data.

        Args:
            filepath (str): the filepath to the new hdf5 file.
            data (np.ndarray): the data to save in the hdf5 file.
        """

        with h5py.File(filepath, 'w') as f:
            # DATA formatting
            f.create_dataset('cube', data=data, compression='gzip', compression_opts=9)
            f.create_dataset('dx', data=self.cube_info.dx)
            f.create_dataset('xt_min', data=min(self.cube_info.xt))
            f.create_dataset('yt_min', data=min(self.cube_info.yt))
            f.create_dataset('zt_min', data=min(self.cube_info.zt))
            f.create_dataset('xt_max', data=max(self.cube_info.xt))
            f.create_dataset('yt_max', data=max(self.cube_info.yt))
            f.create_dataset('zt_max', data=max(self.cube_info.zt))
        if self.verbose > 0: print(f"Saved {os.path.basename(filepath)}", flush=self.flush)



if __name__=='__main__':
    FakeData(
        sphere_radius=(7.6e5, 8e5),
        nb_of_points=int(1e3),
        nb_of_cubes=413,
        increase_factor=1.3,
    )
