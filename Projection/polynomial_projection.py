"""
To get the projection of the polynomial interpolation inside the envelope. 
This is done to get the angle between the projection seen by SDO and the 3D polynomial representation of the data.
"""

# Imports
import os
import h5py
import astropy
import typeguard

# Aliases
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt

# Sub imports
import multiprocessing.queues

# Personal imports 
from common import MultiProcessing, Decorators, Plot
from extract_envelope import Envelope

#TODO: IMPORTANT: there is clearly a problem in the reprojection of the curve made by Dr. Auchere for the Rainbow paper. Need to check this first (print the initial mask with the interpolation and stuff.)
#TODO: IMPORTANT: I need to also add the sdo image itself.
#TODO: MINOR: need to update the code for when there is only one process used (i.e. no multiprocessing)

class OrthographicalProjection:
    """
    Does the 2D projection of a 3D volume.
    Used to recreate what is seen by SDO when looking at the cube. Done to recreate the plots made by Dr. Auchere in his 'the coronal Monsoon' paper.
    """

    @Decorators.running_time
    @typeguard.typechecked
    def __init__(
            self,
            processes: int = 0,
            integration_time_hours: int = 24,
            filename: str = 'sig1e20_leg20_lim0_03_thisone.h5',
            data_type : str = 'No duplicates new',
            with_feet: bool = True,
            polynomial_order: int | list[int] = [4, 5, 6],
            plot_choices: str | list[str] = ['polar', 'sdo image', 'no duplicate', 'envelope', 'polynomial'], 
            verbose: int = 1,
            flush: bool = False
        ) -> None:
        """
        Reprojection of the computed 3D volume to recreate a 2D image of what is seen from SDO's POV. This is done to recreate the analysis of Dr. Auchere's paper; 'The coronal Monsoon'.

        Args:
            processes (int, optional): the number of parallel processes used in the multiprocessing. Defaults to 0.
            integration_time_hours (int, optional): the integration time used for the data to be reprojected. Defaults to 24.
            filename (str, optional): the filename of the HDF5 containing all the relevant 3D data. Defaults to 'sig1e20_leg20_lim0_03_thisone.h5'.
            data_type (str, optional): the data type of the 3D object to be reprojected (e.g. All data). Defaults to 'No duplicates new'.
            with_feet (bool, optional): deciding to use the data with or without added feet. Defaults to True.
            polynomial_order (int | list[int], optional): the order(s) of the polynomial function(s) that represent the fitting of the integrated 3D volume. Defaults to [4, 5, 6].
            plot_choices (str | list[str], optional): the main choices that the user wants to be in the reprojection. The possible choices are:
                ['polar', 'cartesian', 'cube', 'sdo image', 'envelope', 'interpolations']. Defaults to ['polar', 'sdo image', 'no duplicate', 'envelope', 'polynomial'].
            verbose (int, optional): gives the verbosity in the outputted prints. The higher the value, the more prints. Starts at 0 for no prints. Defaults to 1.
            flush (bool, optional): used in the 'flush' kwarg of the print() class. Decides to force the print buffer to be emptied (i.e. forces the prints). Defaults to False.
        """

        # Arguments
        feet = ' with feet' if with_feet else ''
        self.integration_time = integration_time_hours
        self.multiprocessing = True if processes > 1 else False
        self.processes = processes
        self.filename = filename
        self.data_type = data_type + feet
        self.foldername = filename.split('.')[0] + ''.join(feet.split(' '))
        self.polynomial_order = sorted(polynomial_order) if isinstance(polynomial_order, list) else [polynomial_order]
        self.plot_choices = self.plot_choices_creation(plot_choices if isinstance(plot_choices, list) else [plot_choices])
        self.verbose = verbose
        self.flush = flush

        # Important constants
        self.solar_r = 6.96e5  # in km
        self.projection_borders = {
            'radial distance': (690, 870),  # in Mm
            'polar angle': (245, 295),  # in degrees
        }
        Envelope.borders = {**Envelope.borders, **self.projection_borders} 

        # Paths setup
        self.paths = self.path_setup()

        # Get interpolations data
        self.data_info = self.data_setup()

        self.heliographic_cubes_origin = np.array([self.data_info['xmin'], self.data_info['ymin'], self.data_info['zmin']], dtype='float32')

        # Plotting
        self.plotting()

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
            'sdo': os.path.join(main_path, 'sdo'),
            'envelope': os.path.join(main_path, 'Work_done', 'Envelope'),
            'save': os.path.join(main_path, 'Work_done', self.foldername),
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
            dx = H5PYFile['dx'][...][0]
            all_dates = H5PYFile['Dates'][...]
            indexes = H5PYFile['Time indexes'][...]
            sdo_pos = H5PYFile['SDO positions'][indexes]
            dates = [all_dates[i].decode('utf-8') for i in indexes]
            
            # Get data borders
            border_path = 'Filtered/' + self.data_type
            xmin = H5PYFile[border_path + '/xmin'][...]
            ymin = H5PYFile[border_path + '/ymin'][...]
            zmin = H5PYFile[border_path + '/zmin'][...]

            # Get group path
            time_integrated_path = (
                'Time integrated/' 
                + self.data_type 
                + f'/Time integration of {self.integration_time}.0 hours' 
            )

            # Get data
            no_duplicates = H5PYFile['Filtered/' + self.data_type + '/coords'][...]
            cubes = H5PYFile[time_integrated_path + '/coords'][...]
            interpolations = [
                H5PYFile[time_integrated_path + f'/{polynomial_order}th order interpolation/raw_coords'][...]
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
        data['no_duplicates'] = self.matrix_rotation(self.cartesian_pos(no_duplicates, data), data)
        data['interpolations'] = [
            self.matrix_rotation(self.cartesian_pos(interpolation, data), data)
            for interpolation in interpolations
        ]

        # Data info prints
        if self.verbose > 0:
            print(
                "CUBES INFO - "
                f"shape: {cubes.shape} - "
                f"size: {round(cubes.nbytes / 2**20, 2)}Mb.",
                flush=self.flush,
            ) #TODO: will need to add a print for the no duplicates too and maybe also the interpolation. Might just write a class especially for stuff like that in the 
            #common package
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
        possibilities = ['polar', 'cartesian', 'cube', 'sdo image', 'envelope', 'interpolations']
        plot_choices_kwargs = {
            key: False 
            for key in possibilities
        }

        for key in plot_choices: 
            if key in possibilities: 
                plot_choices_kwargs[key] = True
            else: 
                raise ValueError(f"Value for the 'plot_choices' argument not recognised.") 
            
        if 'envelope' in plot_choices: plot_choices_kwargs['polar'] = True
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
            shm, data = MultiProcessing.create_shared_memory(data.astype('float32'))
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
            list[np.ndarray] | None: if not multiprocessing, the list of the rotated 3D volumes. Each index value is for a different time.
        """

        if multiprocessing:
            shm = mp.shared_memory.SharedMemory(name=data['name'])
            data = np.ndarray(data['shape'], dtype=data['dtype'], buffer=shm.buf)

        result_list = []
        for time in range(data_index[0], data_index[1] + 1):
            data_filter = data[0, :] == time
            result = data[1:4, data_filter]
            satelitte_pos = sdo_pos[time]

            # Add Sun center as reference
            # center = np.array([0, 0, 0]).reshape(3, 1)  
            # result = np.column_stack((result, center))
            
            # Centering SDO pov on Sun center
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

            # Format results
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

        Args:
            axis (np.ndarray): the axis that needs rotating.
            angle (float): the rotation angle in degrees.

        Returns:
            np.ndarray: the rotated axis.
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
        To plot the reprojection of the 3D volumes.
        """

        #TODO: maybe add a print that prints all the information in self.data_info

        # Get d_theta in the reprojection
        d_theta = self.data_info['dx'] * 360 / (2 * np.pi * self.solar_r)

        if self.plot_choices['envelope']:
            envelope_data = Envelope.get(
                polynomial_order=6,
                number_of_points=1e5,
                plot=True,
            )
        else:
            envelope_data = None

        # Init kwargs setup
        colour_generator = Plot.different_colours(omit=['white', 'red'])
        global_kwargs = {
            'polynomial_orders': self.polynomial_order,
            'multiprocessing': self.multiprocessing,
            'dates': self.data_info['dates'],
            'solar_r': self.solar_r,
            'integration_time': self.integration_time,
            'paths': self.paths,
            'plot_choices': self.plot_choices,
            'colours': [next(colour_generator) for _ in self.polynomial_order],
            'dx': self.data_info['dx'],
            'd_theta': d_theta,
            'envelope_data': envelope_data,
            'projection_borders': self.projection_borders,
            'verbose': self.verbose,
            'flush': self.flush,
        }

        if self.multiprocessing:
            # Multiprocessing setup
            shm_cubes, cubes = MultiProcessing.create_shared_memory(self.data_info['cubes']) 
            shm_no_duplicates, no_duplicates = MultiProcessing.create_shared_memory(self.data_info['no_duplicates'])
            shm_interp, interpolations = MultiProcessing.create_shared_memory(self.data_info['interpolations'])
            indexes = MultiProcessing.pool_indexes(len(self.data_info['dates']), self.processes)

            # Arguments
            multip_kwargs = {
                'cubes': cubes,
                'no_duplicates': no_duplicates,
                'interpolations': interpolations,
            }

            # Run
            processes = [None] * len(indexes)
            for i, index in enumerate(indexes):
                p = mp.Process(
                    target=self.plotting_sub,
                    kwargs={
                        'data_index': index,
                        **global_kwargs,
                        **multip_kwargs,
                    },
                )
                p.start()
                processes[i] = p
            for p in processes: p.join()
            shm_cubes.unlink()
            shm_no_duplicates.unlink()
            shm_interp.unlink()
        else:
            # Arguments
            kwargs = {
                'cubes': self.data_info['cubes'],
                'no_duplicates': self.data_info['no_duplicates'],
                'interpolations': self.data_info['interpolations'],
            }
            self.plotting_sub(
                data_index=(0, len(self.numbers) - 1),
                **global_kwargs,
                **kwargs,
            )

    @staticmethod
    def plotting_sub(  #TODO: add the curves with and without feet
            cubes: dict[str, any] | np.ndarray,
            no_duplicates: dict[str, any] | np.ndarray,
            interpolations: dict[str, any] | list[np.ndarray],
            multiprocessing: bool,
            dates: list[str],
            solar_r: float,
            integration_time: int,
            paths: dict[str, str],
            plot_choices: dict[str, bool],
            polynomial_orders: list[int],
            data_index: tuple[int, int],
            colours: list[int],
            envelope_data: None | tuple[tuple[np.ndarray, np.ndarray], list[tuple[np.ndarray, np.ndarray]]],
            dx: float,
            d_theta: float,
            projection_borders: dict[str, tuple[int, int]],
            verbose: int,
            flush: bool,
        ) -> None:
        """
        To be able to multiprocess the plotting.
        """

        # Ordering plot kwargs
        plot_kwargs = {
            'interpolation': {
                's': 2,
                'zorder': 2,
            },
            'envelope': {
                'linestyle': '--',
                'alpha': 0.9,
                'zorder': 4,
            },
            'image': {
                'extent': (
                    245,
                    295,
                    690,
                    870,
                ),
                'alpha': 0.5,
                'origin': 'lower',
                'zorder': 0,
            },
            'contour': {
                'linewidth': 0.6,
                'alpha': 1,
                'zorder': 3,
            },
        }
        
        if multiprocessing:
            shm_cubes, cubes = MultiProcessing.open_shared_memory(cubes)
            shm_no_duplicates, no_duplicates = MultiProcessing.open_shared_memory(no_duplicates)
            shm_interpolations, interpolations = MultiProcessing.open_shared_memory(interpolations)

        if envelope_data is not None:
            middle_t_curve, envelope_y_x_curve = envelope_data

        for time in range(data_index[0], data_index[1] + 1):
            # Filtering data
            cubes_filter  = cubes[0, :] == time
            cube = cubes[1:, cubes_filter]

            no_duplicates_filter = no_duplicates[0, :] == time
            no_duplicate = no_duplicates[1:, no_duplicates_filter]

            positions = [None] * len(interpolations)
            for nb in range(len(interpolations)):
                indexes = np.nonzero(interpolations[nb][0, :] == time)[0]
                positions[nb] = [
                    interpolations[nb][i, indexes]
                    for i in range(1, 4)
                ]

            # Voxel positions
            x_cube, y_cube, _ = cube
            x_no_duplicate, y_no_duplicate, _ = no_duplicate
            x_interp, y_interp = [
                [
                    position[i]
                    for position in positions
                ]
                for i in range(2)
            ]

            # Data info
            date = dates[time]

            # if self.plot_choices['sdo image']: self.sdo_image(index=time)  #TODO: need to add this part too

            if plot_choices['cartesian']:

                plt.figure(figsize=(5, 5))
                if plot_choices['cube']: plt.scatter(x_cube / solar_r, y_cube / solar_r, **plot_kwargs[0])
                if plot_choices['interpolations']: 
                    for i in range(len(interpolations)): 
                        plt.scatter(
                            x_interp[i] / solar_r, y_interp[i] / solar_r,
                            label=f'{polynomial_orders[i]}th order polynomial',
                            color=colours[i],
                            **plot_kwargs['interpolation'],
                        )
                plt.title(f'SDO POV for - {date}')
                plt.xlabel('Solar X [au]')
                plt.ylabel('Solar Y [au]')
                plt.legend(loc='upper right')
                plot_name = f'sdoprojection_{date}_{integration_time}h.png'
                plt.savefig(os.path.join(paths['save'], plot_name), dpi=500)
                plt.close()

                if verbose > 1: print(f'SAVED - filename:{plot_name}')

            if plot_choices['polar']:

                # Changing to polar coordinates
                r_cube, theta_cube = OrthographicalProjection.to_polar(x_cube, y_cube)
                r_no_duplicate, theta_no_duplicate = OrthographicalProjection.to_polar(x_no_duplicate, y_no_duplicate)

                # SDO polar projection plotting
                plt.figure(figsize=(12, 5))
                if plot_choices['envelope']: 

                    plt.plot(middle_t_curve[0], middle_t_curve[1], color='black', label='Middle path', **plot_kwargs['envelope'])
                    
                    envelope = envelope_y_x_curve[0]
                    plt.plot(envelope[0], envelope[1], color='grey', label='Envelope', **plot_kwargs['envelope'])
                    for envelope in envelope_y_x_curve[1:]: plt.plot(envelope[0], envelope[1], color='grey', **plot_kwargs['envelope'])

                if plot_choices['cube']: 

                    # Get image and contours
                    lines, image = OrthographicalProjection.cube_contour(
                        polar_theta=theta_cube,
                        polar_r=r_cube,
                        d_theta=d_theta,
                        dx=dx,
                        projection_borders=projection_borders,
                    )

                    # Plot
                    line = lines[0]
                    plt.plot(line[1], line[0], color='red', label='time integrated contours', **plot_kwargs['contour'])
                    for line in lines[1:]: plt.plot(line[1], line[0], color='red', **plot_kwargs['contour'])
                    # plt.imshow(image, **plot_kwargs['image']) 

                    lines, image = OrthographicalProjection.cube_contour(
                        polar_theta=theta_no_duplicate,
                        polar_r=r_no_duplicate,
                        d_theta=d_theta,
                        dx=dx,
                        projection_borders=projection_borders,
                    )

                    if lines is not None:
                        # plt.imshow(image, **plot_kwargs['image'])

                        line = lines[0]
                        plt.plot(line[1], line[0], color='orange', label='no duplicate contours', **plot_kwargs['contour'])
                        for line in lines: plt.plot(line[1], line[0], color='orange', **plot_kwargs['contour'])
                    
                    # plt.scatter(theta_cube, r_cube / 10**3, **plot_kwargs[0])
                if plot_choices['interpolations']: 
                    for i in range(len(interpolations)):
                        # Get polar positions
                        r_interp, theta_interp = OrthographicalProjection.to_polar(x_interp[i], y_interp[i])

                        # Plot
                        plt.scatter(
                            theta_interp,
                            r_interp / 10**3,
                            label=f'{polynomial_orders[i]}th order polynomial',
                            color=colours[i],
                            **plot_kwargs['interpolation'],
                        )
                plt.xlim(245, 295)
                plt.ylim(690, 870)
                plt.gca().set_aspect('auto')
                plt.title(f'SDO polar projection - {date}')
                plt.xlabel('Polar angle [degrees]')
                plt.ylabel('Radial distance [Mm]')
                plt.legend(loc='upper right')
                plot_name = f'sdopolarprojection_{date}_{integration_time}h.png'
                plt.savefig(os.path.join(paths['save'], plot_name), dpi=500)
                plt.close()

            if verbose > 1: print(f'SAVED - filename:{plot_name}', flush=flush)
        # Closing shared memories
        if multiprocessing: shm_cubes.close(); shm_no_duplicates.close(); shm_interpolations.close()

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
    def cube_contour(
            polar_theta: np.ndarray,
            polar_r: np.ndarray,
            d_theta: float,
            dx: float,
            projection_borders: dict[str, tuple[int, int]],
        ) -> list[tuple[list[float], list[float]]]:
        """
        To get the contours of the data cube when it is seen by SDO and using dx (i.e. the voxel size) as the pixel size.
        The contours positions are given in polar coordinates units (i.e. radial distance and polar angle).

        Args:
            polar_theta (np.ndarray): the polar angle positions of the cube's voxels.
            polar_r (np.ndarray): the radial distance positions of the cube's voxels.
            d_theta (float): dx in polar angle degree units.
            dx (float): the length size of one of the cube's voxels in km.
            projection_borders (dict[str, tuple[int, int]]): the borders of the plot and hence also of the corresponding image.

        Returns:
            list[tuple[list[float], list[float]]]: the lines of the contours of the image and the actual image.
        """

        polar_theta -= projection_borders['polar angle'][0]
        polar_r = (polar_r / 1e3) - projection_borders['radial distance'][0]  # in Mm

        # Initialising the image
        image_shape = (
            int((projection_borders['radial distance'][1] - projection_borders['radial distance'][0]) * 1e3 / dx),
            int((projection_borders['polar angle'][1] - projection_borders['polar angle'][0]) / d_theta),
        )
        image = np.zeros(image_shape)

        # Filtering the pixels outside the final image
        filters = (polar_theta >= 0) & (polar_r >= 0)
        polar_theta = polar_theta[filters]  # As some cube values are outside the plot box
        polar_r = polar_r[filters]  # the values in the range of the plot in Mm

        # Binning the data
        polar_theta //= d_theta
        polar_r //= (dx / 1e3)  # in Mm in the range of the plot. i.e. 0 is the plot border
        
        # Filtering the duplicates
        polar = np.stack([polar_r, polar_theta], axis=0)
        indexes = np.unique(polar, axis=1).astype('int64')

        # If no pixels found in the image
        if len(indexes[0]) == 0: return None, None  #TODO: will need to make this a bit cleaner

        # Populating the image
        image[indexes[0], indexes[1]] = 1  

        # Get contours
        lines = Plot.contours(image)

        # Get corresponding polar coordinates
        nw_lines = [None] * len(lines)
        for i, line in enumerate(lines):
            nw_lines[i] = ((
                [projection_borders['radial distance'][0] + (value * dx) / 1e3 for value in line[0]],
                [projection_borders['polar angle'][0] + (value * d_theta) for value in line[1]],
            ))
        return nw_lines, image

    def sdo_image(self, image_number: int):
        """
        To open the SDO image data, preprocess it and return it as an array for use in plots.
        """

        index = ...

        image = astropy.io.fits.getdata(self.sdo_filepaths[index], 1)
        pass #TODO: will do it later as I need to take into account CRPIX1 and CRPIX2 but also conversion image to plot values


if __name__ == '__main__':
    OrthographicalProjection(
        filename='sig1e20_leg20_lim0_03_thisone.h5',
        with_feet=True,
        verbose=2,
        processes=48,
        polynomial_order=[4],
        plot_choices=['polar', 'cube', 'interpolations', 'envelope'],
        flush=True,
    )





