"""
To create fake .h5 data to then be converted to .save data to then be used by my code.
Should be able to tell if there is any actual projection errors from the totality of my code.
"""

# IMPORTs
import os
import h5py
import scipy
import sparse

# IMPORTs alias
import numpy as np
import multiprocessing as mp

# IMPORTs sub
import multiprocessing.queues
from typing import Any
from dataclasses import dataclass, field

# IMPORTs personal
from common import root_path, Decorators

# PLACEHOLDERs type annotation
LockProxy = Any
ValueProxy = Any



@dataclass(slots=True, repr=False, eq=False)
class CubeBorders:
    """
    To store the border information of the cube, be it in cartesian, in spherical or in indexes.
    """

    # CUBEs info
    dx: float
    xt_min: float
    xt_max: float
    yt_min: float
    yt_max: float
    zt_min: float
    zt_max: float
    increase_factor: float
    
    # BORDERs spherical
    r_min: float = field(init=False)
    r_max: float = field(init=False)
    theta_min: float = field(init=False)
    theta_max: float = field(init=False)
    phi_min: float = field(init=False)
    phi_max: float = field(init=False)

    # BORDERs indexes
    max_x_index: int = field(init=False)
    max_y_index: int = field(init=False)
    max_z_index: int = field(init=False)

    def __post_init__(self) -> None:

        # BORDERs increase
        self.border_increase()

        # BORDERs spherical
        self.r_min = np.sqrt(self.xt_min**2 + self.yt_min**2 + self.zt_min**2)
        self.r_max = np.sqrt(self.xt_max**2 + self.yt_max**2 + self.zt_max**2)
        self.theta_min = np.arccos(self.zt_min / self.r_min)
        self.theta_max = np.arccos(self.zt_max / self.r_max)
        self.phi_min = np.arctan2(self.yt_min, self.xt_min)
        self.phi_max = np.arctan2(self.yt_max, self.xt_max)

        # BORDERs indexes
        self.max_x_index = int((self.xt_max - self.xt_min) // self.dx)
        self.max_y_index = int((self.yt_max - self.yt_min) // self.dx)
        self.max_z_index = int((self.zt_max - self.zt_min) // self.dx)

    def border_increase(self) -> None:
        """
        To increase the borders of the cube.
        """

        # BORDERs range
        x_range = self.xt_max - self.xt_min
        y_range = self.yt_max - self.yt_min
        z_range = self.zt_max - self.zt_min

        # BORDERs increase
        self.xt_min -= x_range * (self.increase_factor - 1) / 2
        self.yt_min -= y_range * (self.increase_factor - 1) / 2
        self.zt_min -= z_range * (self.increase_factor - 1) / 2
        self.xt_max += x_range * (self.increase_factor - 1) / 2
        self.yt_max += y_range * (self.increase_factor - 1) / 2
        self.zt_max += z_range * (self.increase_factor - 1) / 2

class FakeData:
    """
    Creates fake mat data to then be converted to .save file to finally be used by my re-projection
    code.
    """

    def __init__(
            self,
            radius: float,
            nb_of_points: int,
            nb_of_cubes: int,
            increase_factor: float,
            processes: int,
        ) -> None:
        """
        To create the fake sun surface data indexes in mat files.

        Args:
            radius (float): the radius of the sphere in km.
            nb_of_points (int): the initial resolution in spherical coordinates of the sphere.
            nb_of_cubes (int): the number of cubes to create.
            increase_factor (float): the coefficient in the increase (or decrease) of the cube
                size.
        """

        # ATTRIBUTEs
        self.radius = radius
        self.nb_of_cubes = nb_of_cubes
        self.nb_of_points = nb_of_points
        self.increase_factor = increase_factor
        self.processes = processes
        self.multiprocessing = True if processes > 1 else False

        # RUN
        self.paths = self.path_setup()
        self.defaults = self.get_defaults()
        self.create_fake_data()

    def path_setup(self) -> dict[str, str]:
        """
        To give the paths to the different needed directories.

        Returns:
            dict[str, str]: the paths to the different directories.
        """

        # PATHs setup
        main_path = os.path.join(root_path, '..')

        # PATHs formatting
        paths = {
            'main': main_path,
            'cubes': os.path.join(main_path, 'Cubes_karine'),
            'save h5': os.path.join(root_path, 'Data/fake_data/h5'),
        }

        # PATHs create
        for key in ['save h5']: os.makedirs(paths[key], exist_ok=True)
        return paths

    def get_defaults(self) -> CubeBorders:
        """
        To get the usual values of the cube resolution and borders.

        Returns:
            CubeBorders: the usual values of the cube resolution and borders.
        """

        first_cube = scipy.io.readsav(os.path.join(self.paths['cubes'], 'cube000.save'))
        info = CubeBorders(
            dx=float(first_cube.dx),
            increase_factor=self.increase_factor,
            xt_min=float(first_cube.xt_min),
            yt_min=float(first_cube.yt_min),
            zt_min=float(first_cube.zt_min),
            xt_max=float(first_cube.xt_max),
            yt_max=float(first_cube.yt_max),
            zt_max=float(first_cube.zt_max),
        )
        return info

    @Decorators.running_time    
    def create_sphere_surface(self) -> np.ndarray:
        """
        To get the cartesian coordinates of the sphere surface.

        Returns:
            np.ndarray: the cartesian coordinates of the sphere surface.
        """
        
        # COORDs spherical
        phi = np.linspace(self.defaults.phi_min, self.defaults.phi_max, self.nb_of_points * 2)
        theta = np.linspace(self.defaults.theta_min, self.defaults.theta_max, self.nb_of_points)
        phi, theta = np.meshgrid(phi, theta)

        # COORDs cartesian
        x = self.radius * np.sin(theta) * np.cos(phi)
        y = self.radius * np.sin(theta) * np.sin(phi)
        z = self.radius * np.cos(theta)
        return np.stack((x.ravel(), y.ravel(), z.ravel()), axis=0)

    @Decorators.running_time
    def to_index(self, coords: np.ndarray) -> np.ndarray:
        """
        To convert the cartesian coordinates to cube indexes.

        Args:
            coords (np.ndarray): the cartesian coordinates.

        Returns:
            np.ndarray: the cube indexes.
        """

        # COORDs to cube
        coords[0] -= self.defaults.xt_min
        coords[1] -= self.defaults.yt_min
        coords[2] -= self.defaults.zt_min

        # BIN values
        coords //= self.defaults.dx

        # FILTER values
        x_filter = (coords[0] >= 0) & (coords[0] < self.defaults.max_x_index)
        y_filter = (coords[1] >= 0) & (coords[1] < self.defaults.max_y_index)
        z_filter = (coords[2] >= 0) & (coords[2] < self.defaults.max_z_index)
        coords = coords[:, x_filter & y_filter & z_filter]

        # UNIQUE values
        coords = np.unique(coords.astype(int), axis=1)
        return coords

    @Decorators.running_time
    def create_fake_data(self) -> None:
        """
        To create the fake mat data and save it to .h5 files.
        """

        # COORDs
        sphere_surface = self.create_sphere_surface()
        sphere_surface = self.to_index(sphere_surface)[[2, 1, 0]]
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
                p =  mp.Process(target=self.create_h5, args=(array, lock, counter))
                p.start()
                processes[i] = p
            for p in processes: p.join()

        else:
            # HDF5 save
            for i in range(self.nb_of_cubes):
                filename = os.path.join(self.paths['save h5'], f'cube{i:03d}.h5')

                with h5py.File(filename, 'w') as f:
                    # DATA formatting
                    f.create_dataset('cube', data=array, compression='gzip')
                    f.create_dataset('dx', data=self.defaults.dx)
                    f.create_dataset('xt_min', data=self.defaults.xt_min)
                    f.create_dataset('yt_min', data=self.defaults.yt_min)
                    f.create_dataset('zt_min', data=self.defaults.zt_min)
                    f.create_dataset('xt_max', data=self.defaults.xt_max)
                    f.create_dataset('yt_max', data=self.defaults.yt_max)
                    f.create_dataset('zt_max', data=self.defaults.zt_max)
                print(f"Saved {filename}")

    def create_h5(self, data: np.ndarray, lock: LockProxy, counter: ValueProxy) -> None:
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
            filename = os.path.join(self.paths['save h5'], f'cube{i:03d}.h5')

            with h5py.File(filename, 'w') as f:
                # DATA formatting
                f.create_dataset('cube', data=data, compression='gzip')
                f.create_dataset('dx', data=self.defaults.dx)
                f.create_dataset('xt_min', data=self.defaults.xt_min)
                f.create_dataset('yt_min', data=self.defaults.yt_min)
                f.create_dataset('zt_min', data=self.defaults.zt_min)
                f.create_dataset('xt_max', data=self.defaults.xt_max)
                f.create_dataset('yt_max', data=self.defaults.yt_max)
                f.create_dataset('zt_max', data=self.defaults.zt_max)
            print(f"Saved {filename}")



if __name__=='__main__':
    FakeData(radius=7e5, nb_of_points=int(1e3), nb_of_cubes=413, increase_factor=1.2, processes=12)
