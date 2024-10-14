"""
To get the projection of the polynomial interpolation inside the envelop. 
This is done to get the angle between the projection seen by SDO and the 3D polynomial representation of the data.
"""

# Imports
import os
import h5py
import typeguard

# Aliases
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt

# Sub imports
import multiprocessing.queues

# Personal imports 
from Common import MultiProcessing, Decorators, Plot

#TODO: need to see what the problem in the visualisation angle is. Seems off when comparing to the possibly erroneous 3D visualisation.
#TODO: I need to also add the sdo image itself.

class OrthographicalProjection:
    """
    Does the 2D projection of a 3D volume.
    Used to recreate what is seen by SDO when looking at the cube, especially the curve fits.
    Also, while not yet implemented, will also add the envelop around the projection.
    """

    @Decorators.running_time
    @typeguard.typechecked
    def __init__(
            self,
            processes: int = 0,
            integration_time_hours: int = 24,
            saving_plots: bool = False,
            filename: str = 'order0321.h5',
            data_type : str = 'No duplicates new with feet',
            polynomial_order: int | list[int] = [4, 5, 6],
            plot_choices: str | list[str] = ['polar', 'sdo image', 'no duplicate', 'envelop', 'polynomial'], 
            verbose: int = 1,
            flush: bool = False
        ) -> None:

        # Arguments
        self.integration_time = integration_time_hours
        self.multiprocessing = True if processes > 1 else False
        self.processes = processes
        self.filename = filename
        self.data_type = data_type
        self.polynomial_order = sorted(polynomial_order) if isinstance(polynomial_order, list) else [polynomial_order]
        self.solar_r = 6.96e5  # in km
        self.plot_choices = self.plot_choices_creation(plot_choices if isinstance(plot_choices, list) else [plot_choices])
        self.verbose = verbose
        self.flush = flush

        # Paths setup
        self.paths = self.path_setup()

        # Get interpolations data
        self.data_info = self.data_setup()

        self.heliographic_cubes_origin = np.array([self.data_info['xmin'], self.data_info['ymin'], self.data_info['zmin']], dtype='float32')

        # Plotting
        if saving_plots: self.plotting()

    def path_setup(self) -> dict[str, str]:
        """
        To get the paths to the needed directories and files.

        Raises:
            ValueError: if the main path is not found.

        Returns:
            dict[str, str]: the needed paths.
        """

        # Check main path
        main_path = '/home/avoyeux/Documents/avoyeux/'
        if not os.path.exists(main_path): main_path = '/home/avoyeux/old_project/avoyeux/'
        if not os.path.exists(main_path): raise ValueError(f"\033[1;31mThe main path {main_path} not found.")
        code_path = os.path.join(main_path, 'python_codes')
        # Save paths
        paths = {
            'main': main_path,
            'codes': code_path,
            'data': os.path.join(code_path, 'Data', self.filename),
            'save': os.path.join(main_path, 'Work_done', 'Projections_new')
        }
        os.makedirs(paths['save'], exist_ok=True)
        return paths

    def data_setup(self) -> dict[str, np.ndarray]:
        """
        To get the necessary data as a dictionary.

        Returns:
            dict[str, np.ndarray]: all the data gotten from the HDF5 file.
        """
        
        # Open file
        with h5py.File(self.paths['data'], 'r') as H5PYFile:

            # Get default data information
            dx = H5PYFile['dx'][...]
            indexes = H5PYFile['Time indexes'][...]
            all_dates = H5PYFile['Dates'][...]
            dates = [all_dates[i].decode('utf-8') for i in indexes]
            sdo_pos = H5PYFile['SDO positions'][indexes]
            
            # Get data borders
            border_path = 'Filtered/' + self.data_type
            xmin = H5PYFile[border_path + '/xmin'][...]
            ymin = H5PYFile[border_path + '/ymin'][...]
            zmin = H5PYFile[border_path + '/zmin'][...]

            # Get group path
            group_path = (
                'Time integrated/' 
                + self.data_type 
                + f'/Time integration of {self.integration_time}.0 hours' 
            )

            cubes = H5PYFile[group_path + '/coords'][...]
            interpolations = [
                H5PYFile[group_path + f'/{polynomial_order}th order interpolation/raw_coords'][...]
                for polynomial_order in self.polynomial_order
            ]

        # Formatting data
        data = {
            'dx': dx,
            'dates': dates,
            'xmin': xmin,
            'ymin': ymin,
            'zmin': zmin,
            'sdo_pos': sdo_pos,
        }
        data['cubes'] = self.matrix_rotation(self.cartesian_pos(cubes, data), data)
        data['interpolations'] = np.stack([
            self.matrix_rotation(self.cartesian_pos(interpolation, data), data)
            for interpolation in interpolations
        ], axis=0)

        # Data info prints
        if self.verbose > 0:
            print(
                "CUBES INFO - "
                f"shape: {cubes.shape} - "
                f"size: {round(cubes.nbytes / 2**20, 2)}Mb."
            ) 
            print(
                "POLY INFO - "
                f"shape: {data['interpolations'].shape} - "
                f"size:{round(data['interpolations'].nbytes / 2**20, 2)}Mb.",
                flush=self.flush,
            )
        return data

    def plot_choices_creation(self, plot_choices: list[str]) -> dict[str, bool]:
        """
        Creating a dictionary that chooses what to plot.

        Args:
            plot_choices (list[str]): the list of the choices made in the plotting.

        Raises:
            ValueError: if the plotting choice string is not recognised.

        Returns:
            dict[str, bool]: decides what will be plotted later on.
        """

        # Initialisation of the possible choices
        possibilities = ['polar', 'cartesian', 'cube', 'sdo image', 'envelop', 'interpolations']
        plot_choices_kwargs = {
            key: False 
            for key in possibilities
        }

        for key in plot_choices: 
            if key in possibilities: 
                plot_choices_kwargs[key] = True
            else: 
                raise ValueError(f"Value for the 'plot_choices' argument not recognised.") 
            
        if 'envelop' in plot_choices: plot_choices_kwargs['polar'] = True
        return plot_choices_kwargs

    @Decorators.running_time
    def cartesian_pos(self, data: np.ndarray, data_info: dict[str, np.ndarray]) -> np.ndarray:
        """
        To calculate the heliographic cartesian positions given a ndarray of index positions.

        Args:
            data (np.ndarray): the index positions.
            data_info (dict[str, np.ndarray]): the data information.

        Returns:
            np.ndarray: the corresponding heliographic cartesian positions.
        """

        cubes_sparse_coords = data.astype('float32')

        # Initialisation
        cubes_sparse_coords[1, :] = (cubes_sparse_coords[1, :] * data_info['dx'] + data_info['xmin']) 
        cubes_sparse_coords[2, :] = (cubes_sparse_coords[2, :] * data_info['dx'] + data_info['ymin']) 
        cubes_sparse_coords[3, :] = (cubes_sparse_coords[3, :] * data_info['dx'] + data_info['zmin'])
        return cubes_sparse_coords

    @Decorators.running_time
    def matrix_rotation(self, data: np.ndarray, data_info: dict[str, np.ndarray]) -> np.ndarray:
        """
        To rotate the matrix so that it aligns with the satellite pov.

        Args:
            data (np.ndarray): the matrix to be rotated with shape (4, N) where 4 is t, x, y, z.
            data_info (dict[str, np.ndarray]): the data information.

        Returns:
            np.ndarray: the rotated matrix.
        """

        if self.multiprocessing:
            # Multiprocessing setup
            manager = mp.Manager()
            queue = manager.Queue()
            shm, data = MultiProcessing.shared_memory(data.astype('float32'))
            indexes = MultiProcessing.pool_indexes(len(data_info['dates']), self.processes)
            # Run
            processes = [None] * len(indexes)
            for i, index_tuple in enumerate(indexes):
                p = mp.Process(target=self.time_loop, kwargs={
                    'index': i,
                    'data_index': index_tuple,
                    'queue': queue,
                    'data': data,
                    'multiprocessing': self.multiprocessing,
                    'sdo_pos': data_info['sdo_pos'],
                })
                p.start()
                processes[i] = p
            for p in processes: p.join()
            shm.unlink()

            # Getting the results 
            results = [None] * len(indexes)
            while not queue.empty():
                identifier, result = queue.get()
                results[identifier] = result
            results = [projection_matrix for sublist in results for projection_matrix in sublist]
        else:
            results = self.time_loop(data=data, data_index=(0, len(data_info['dates']) - 1))

        # Ordering the final result so that it is a np.ndarray
        start_index = 0
        total_nb_vals = sum(arr.shape[1] for arr in results)
        final_results = np.empty((4, total_nb_vals), dtype='float32')
        for t, result in enumerate(results):
            nb_columns = result.shape[1]
            final_results[0, start_index: start_index + nb_columns] = t
            final_results[1:4, start_index: start_index + nb_columns] = result
            start_index += nb_columns
        return final_results

    @staticmethod
    def time_loop(
            data: np.ndarray | dict,
            data_index: tuple[int, int],
            sdo_pos: np.ndarray,
            multiprocessing: bool,
            index: int = 0,
            queue: mp.queues.Queue | None = None,
        ) -> list[np.ndarray] | None:
        """
        To rotate a given section of the total data.

        Args:
            data (np.ndarray | dict): the total data.
            data_index (tuple[int, int]): the data section indexes.
            sdo_pos (np.ndarray): the position of SDO.
            multiprocessing (bool): choosing to multiprocess or not.
            index (int, optional): the index to identify the result. Defaults to 0.
            queue (mp.queues.Queue | None, optional): queue to save the results if multiprocessing. Defaults to None.

        Returns:
            list[np.ndarray] | None: _description_
        """

        if multiprocessing:
            shm = mp.shared_memory.SharedMemory(name=data['name'])
            data = np.ndarray(data['shape'], dtype=data['dtype'], buffer=shm.buf)

        result_list = []
        for time in range(data_index[0], data_index[1] + 1):
            data_filter = data[0, :] == time
            result = data[1:4, data_filter]
            center = np.array([0, 0, 0]).reshape(3, 1)  # TODO: need to change this when the code works. This is only to see where the sun center is
            result = np.column_stack((result, center))  # adding the sun center to see if the translation is correct
            satelitte_pos = sdo_pos[time]

            # Centering the SDO pov on the Sun center
            sun_center = np.array([0, 0, 0])
            # Defining the view direction vector
            viewing_direction = sun_center - satelitte_pos
            viewing_direction_norm = viewing_direction / np.linalg.norm(viewing_direction)

            # Defining the up vector and the normal vector to the projection
            up_vector = np.array([0, 0, 1])
            target_axis = np.array([0, 0, 1])

            # Axis of rotation and angle
            rotation_axis = np.cross(viewing_direction_norm, target_axis)
            cos_theta = np.dot(viewing_direction_norm, target_axis)
            theta = np.arccos(cos_theta)

            # Corresponding rotation matrix
            rotation_matrix = OrthographicalProjection.rodrigues_rotation(rotation_axis, -theta)

            # up_vector in the new rotated matrix
            up_vector_rotated = np.dot(rotation_matrix, up_vector)
            theta = np.arctan2(up_vector_rotated[0], up_vector_rotated[1])
            up_rotation_matrix = OrthographicalProjection.rodrigues_rotation(target_axis, theta)
            
            # Final rotation matrix
            rotation_matrix = np.matmul(up_rotation_matrix, rotation_matrix)

            result = [np.matmul(rotation_matrix, point) for point in result.T]
            result = np.stack(result, axis=-1)
            result_list.append(result)
        if not multiprocessing: return result_list
        shm.close()
        queue.put((index, result_list))

    @staticmethod 
    def rodrigues_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
        """
        The Rodrigues's rotation formula to rotate matrices.
        """

        # Normalisation
        axis = axis / np.linalg.norm(axis)

        matrix = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        return np.eye(3) + np.sin(angle) * matrix + (1 - np.cos(angle)) * (matrix @ matrix)

    @Decorators.running_time
    def plotting(self) -> None:
        """
        Parent function to plot the data.
        """

        if self.multiprocessing:
            # Multiprocessing setup
            shm_cubes, cubes = MultiProcessing.shared_memory(self.data_info['cubes']) 
            shm_interp, interpolations = MultiProcessing.shared_memory(self.data_info['interpolations'])
            indexes = MultiProcessing.pool_indexes(len(self.data_info['dates']), self.processes)
            
            # Arguments
            kwargs = {
                'cubes': cubes,
                'interpolations': interpolations,
                'polynomial_orders': self.polynomial_order,
                'multiprocessing': True,
                'dates': self.data_info['dates'],
                'solar_r': self.solar_r,
                'integration_time': self.integration_time,
                'paths': self.paths,
                'plot_choices': self.plot_choices,
                'verbose': self.verbose,
                'flush': self.flush,
            }

            # Run
            processes = [None] * len(indexes)
            for i, index in enumerate(indexes):
                p = mp.Process(target=self.plotting_sub, kwargs={'data_index': index, **kwargs})
                p.start()
                processes[i] = p
            for p in processes: p.join()
            shm_cubes.unlink()
            shm_interp.unlink()
        else:
            # Arguments
            kwargs = {
                'cubes': self.data_info['cubes'],
                'interpolations': self.data_info['interpolations'],
                'polynomial_orders': self.polynomial_order,
                'multiprocessing': False,
                'dates': self.data_info['dates'],
                'solar_r': self.solar_r,
                'integration_time': self.integration_time,
                'paths': self.paths,
                'plot_choices': self.plot_choices,
                'verbose': self.verbose,
                'flush': self.flush,
            }
            self.plotting_sub(data_index=(0, len(self.numbers) - 1), **kwargs)

    @staticmethod
    def plotting_sub(
            cubes: dict[str, any],
            interpolations: dict[str, any],
            multiprocessing: bool,
            dates: list[str],
            solar_r: float,
            integration_time: int,
            paths: dict[str, str],
            plot_choices: dict[str, bool],
            polynomial_orders: list[int],
            data_index: tuple[int, int],
            verbose: int,
            flush: bool,
        ) -> None:
        """
        To be able to multiprocess the plotting.
        """
        
        if multiprocessing:
            # Multiprocessing setup
            shm_cubes = mp.shared_memory.SharedMemory(name=cubes['name'])
            shm_interpolations = mp.shared_memory.SharedMemory(name=interpolations['name'])
            cubes = np.ndarray(cubes['shape'], dtype=cubes['dtype'], buffer=shm_cubes.buf)
            interpolations = np.ndarray(interpolations['shape'], dtype=interpolations['dtype'], buffer=shm_interpolations.buf)

        if plot_choices['envelop']: OrthographicalProjection.envelop_preprocessing()  #TODO: need to add this part

        for time in range(data_index[0], data_index[1] + 1):
            # Filtering data
            cubes_filter  = cubes[0, :] == time
            cube = cubes[1:, cubes_filter]

            indexes = np.stack([
                np.nonzero(interpolations[i, 0, :] == time)[0]
                for i in range(interpolations.shape[0])
            ], axis=0)  
            
            positions = np.stack([
                np.stack([
                    interpolations[nb, i, indexes[nb]]
                    for i in range(1, 4)
                ], axis=0)
                for nb in range(interpolations.shape[0])
            ], axis=0)

            # Voxel positions
            x_cube, y_cube, _ = cube
            x_interp, y_interp = [positions[:, i] for i in range(2)]

            # Data info
            date = dates[time]

            # if self.plot_choices['sdo image']: self.sdo_image(index=time)  #TODO: need to add this part too

            plot_kwargs = {
                0: {
                    's': 1,
                    'color': 'blue',
                    'zorder': 1,
                    'label': 'Data points',
                },
                1: {
                    's': 2,
                    'zorder': 2,
                },
            }

            if plot_choices['cartesian']:
                # SDO projection plotting
                plt.figure(figsize=(5, 5))
                if plot_choices['cube']: plt.scatter(x_cube / solar_r, y_cube / solar_r, **plot_kwargs[0])
                if plot_choices['interpolations']: 
                    for i in range(interpolations.shape[0]): 
                        plt.scatter(
                            x_interp[i] / solar_r, y_interp[i] / solar_r,
                            label=f'{polynomial_orders[i]}th order polynomial',
                            **plot_kwargs[1],
                        )
                plt.title(f'SDO POV for - {date}')
                plt.xlabel('Solar X [au]')
                plt.ylabel('Solar Y [au]')
                plt.legend()
                plot_name = f'sdoprojection_{date}_{integration_time}h.png'
                plt.savefig(os.path.join(paths['save'], plot_name), dpi=500)
                plt.close()

            if plot_choices['polar']:
                # Changing to polar coordinates
                r_cube, theta_cube = OrthographicalProjection.to_polar(x_cube, y_cube)

                # SDO polar projection plotting
                plt.figure(figsize=(12, 5))
                if plot_choices['cube']: plt.scatter(theta_cube, r_cube / 10**3, **plot_kwargs[0])
                if plot_choices['interpolations']: 
                    r_interp, theta_interp = OrthographicalProjection.to_polar(x_interp, y_interp)
                    for i in range(interpolations.shape[0]):
                        plt.scatter(
                            theta_interp[i],
                            r_interp[i] / 10**3,
                            label=f'{polynomial_orders[i]}th order polynomial',
                            color=f'#{next(Plot.random_hexadecimal_int_color_generator()):06x}',
                            **plot_kwargs[1],
                        )
                plt.xlim(245, 295)
                plt.ylim(700, 870)
                plt.title(f'SDO polar projection - {date}')
                plt.xlabel('Polar angle [degrees]')
                plt.ylabel('Radial distance [Mm]')
                plt.legend()
                plot_name = f'sdopolarprojection_{date}_{integration_time}h.png'
                plt.savefig(os.path.join(paths['save'], plot_name), dpi=500)
                plt.close()

            if verbose > 1: print(f'SAVED - filename:{plot_name}', flush=flush)
        # Closing shared memories
        if multiprocessing: shm_cubes.close(); shm_interpolations.close()

    @staticmethod
    def to_polar(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Changes cartesian coordinates to polar.

        Args:
            x (np.ndarray): the x-axis cartesian coordinates.
            y (np.ndarray): the y-axis cartesian coordinates.

        Returns:
            tuple[np.ndarray, np.ndarray]: the r and theta polar coordinate values.
        """
         
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x) - np.pi / 2
        theta = np.where(theta < 0, theta + 2 * np.pi, theta)
        theta = 2 * np.pi - theta  # clockwise
        theta = np.where(theta >= 2 * np.pi, theta - 2 * np.pi, theta)  # modulo 2pi
        theta = np.degrees(theta) 
        return r, theta

    @staticmethod
    def envelop_preprocessing(self):
        """
        Opens the two png images of the envelop in polar coordinates. Then, treats the data to use it in
        the polar plots.
        """

        pass

    # def sdo_image(self, index: int):
    #     """
    #     To open the SDO image data, preprocess it and return it as an array for use in plots.
    #     """

    #     image = fits.getdata(self.sdo_filepaths[index], 1)
    #     pass # TODO: will do it later as I need to take into account CRPIX1 and CRPIX2 but also conversion image to plot values


if __name__=='__main__':
    OrthographicalProjection(
        verbose=2,
        processes=10,
        polynomial_order=[4, 5, 6],
        saving_plots=True,
        plot_choices=['polar', 'cartesian', 'cube', 'interpolations'],
        flush=True,
    )