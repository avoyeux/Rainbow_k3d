"""
To get the projection of the polynomial interpolation inside the envelop. 
This is done to get the angle between the projection seen by SDO and the 3D polynomial representation of the data.
"""

# Imports
import os
import re
import sys
import h5py
import sparse
import typeguard

# Aliases
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt

# Sub imports
import multiprocessing.queues

# Personal imports 
sys.path.append('..')
from Common import MultiProcessing, Decorators



#TODO: should I also add the cube itself in the projection as a background?
#TODO: I need to also add the sdo image itself.
class OrthographicalProjection:
    """
    Does the 2D projection of a 3D volume.
    Used to recreate what is seen by SDO when looking at the cube, especially the curve fits.
    Also, while not yet implemented, will also add the envelop around the projection.
    """

    @typeguard.typechecked
    def __init__(
            self,
            data_path: str,
            processes: int = 0,
            saving_data: bool = False,
            integration_time_hours: int = 24,
            saving_plots: bool = False,
            filename: str = 'order0321.h5',
            data_type : str = 'No duplicates new with feet',
            polynomial_order: int = 5,
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
        self.polynomial_order = polynomial_order
        self.solar_r = 6.96e5  # in km
        self.plot_choices = self.plot_choices_creation(plot_choices if isinstance(plot_choices, list) else [plot_choices])
        self.verbose = verbose
        self.flush = flush

        # Paths setup
        self.paths = self.path_setup()

        # Get interpolation data
        self.data_info = self.data_setup()

        # Get plot choices
        self.plot_kwargs = self.plot_choices_creation()

        # Plotting
        if saving_plots: self.plotting(self.data_info['interpolation'])

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
        if not os.path.exist(main_path): main_path = '/home/avoyeux/old_project/avoyeux/'
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

            # Get group path
            group_path = (
                'Time integrated/' 
                + self.data_type 
                + f'/Time integration of {self.integration_time}.0 hours' 
                + f'/{self.polynomial_order}th order interpolation'
            )

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

            interpolation = self.get_COO(H5PYFile, group_path)  
            #TODO: it should be better to just redo the polynomial function using the parameters but then I need to be able to stop
            #TODO: the polynomial exactly at the feet positions.
            #TODO: will need to add the cube data here 
        
        data = {
            'dx': dx,
            'dates': dates,
            'xmin': xmin,
            'ymin': ymin,
            'zmin': zmin,
            'sdo pos': sdo_pos,
            'interpolation': interpolation,
        }

        if self.verbose > 0: 
            print(
                f"POLY INFO - sparse shape:{interpolation.coords.shape} "
                f"- dense shape:{interpolation.shape} "
                f"- size:{round(interpolation.nbytes / 2**20, 2)}Mb.",
                flush=self.flush,
            )
        return data
    
    def get_COO(self, H5PYFile: h5py.File, group_path: str) -> sparse.COO:
        """
        To get the sparse.COO object from the corresponding coords and values.

        Args:
            H5PYFile (h5py.File): the file object.
            group_path (str): the path to the group where the data is stored.

        Returns:
            sparse.COO: the corresponding sparse data.
        """

        # Get data
        data_coords = H5PYFile[group_path + '/coords'][...]
        data_data = H5PYFile[group_path + '/values'][...] if not 'interpolation' in group_path else 1
        shape = np.max(data_coords, axis=1) + 1
        return sparse.COO(coords=data_coords, data=data_data, shape=shape)

    def plot_choices_creation(self, plot_choices: list[str]) -> dict[str, bool]:
        """
        To check the values given for the plot_choices argument.
        """  #TODO: finish this docstring

        # Initialisation of the possible choices
        possibilities = ['polar', 'cartesian', 'sdo image', 'no duplicate', 'envelop', 'polynomial']
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

    # def path(self) -> None:
    #     """
    #     Function where the filepaths dictionary is created.
    #     """

    #     main_path = os.path.join(os.getcwd(), '..')
    #     self.paths = {
    #         'main': main_path, 
    #         'SDO': os.path.join(main_path, 'sdo'),
    #         'cubes': os.path.join(main_path, 'Cubes_karine'),
    #         'curve fit': os.path.join(main_path, 'curveFitArrays'),
    #         'envelope': os.path.join(main_path, 'Projection', 'Envelope'),
    #         'save': os.path.join(main_path, 'Projection' ,'Projection_results'),
    #     }
    #     os.makedirs(self.paths['save'], exist_ok=True)

    @Decorators.running_time
    def cartesian_pos(self, data: np.ndarray) -> np.ndarray:
        """
        To calculate the heliographic cartesian positions of some of the objects.
        """

        cubes_sparse_coords = data.astype('float64')

        # Initialisation
        cubes_sparse_coords[1, :] = (cubes_sparse_coords[1, :] * self.data_info['dx'] + self.data_info['x_min']) 
        cubes_sparse_coords[2, :] = (cubes_sparse_coords[2, :] * self.data_info['dx'] + self.data_info['y_min']) 
        cubes_sparse_coords[3, :] = (cubes_sparse_coords[3, :] * self.data_info['dx'] + self.data_info['z_min'])
   
        self.heliographic_cubes_origin = np.array([self.data_info['x_min'], self.data_info['y_min'], self.data_info['z_min']], dtype='float64')
        return cubes_sparse_coords

    @Decorators.running_time
    def matrix_rotation(self, data: np.ndarray) -> np.ndarray:
        """
        To rotate the matrix so that it aligns with the satellite pov
        """

        if self.multiprocessing:
            # Initialisation of the multiprocessing
            manager = Manager()
            queue = manager.Queue()
            shm, data = MultiProcessing.shared_memory(data.astype('float64'))

            # Setting up each process
            indexes = MultiProcessing.pool_indexes(len(self.numbers), self.processes)
            kwargs = {
                'queue': queue, 
                'data': data,
            }
            processes = [Process(target=self.time_loop, kwargs={'index': i, 'data_index': index_tuple, **kwargs}) for i, index_tuple in enumerate(indexes)]
            for p in processes: p.start()
            for p in processes: p.join()
            
            shm.unlink()

            # Getting the results 
            results = [None] * self.processes
            while not queue.empty():
                identifier, result = queue.get()
                results[identifier] = result
            results = [projection_matrix for sublist in results for projection_matrix in sublist]
        else:
            results = self.time_loop(data=data, data_index=(0, len(self.numbers) - 1))

        # Ordering the final result so that it is a np.ndarray
        start_index = 0
        total_nb_vals = sum(arr.shape[1] for arr in results)
        final_results = np.empty((4, total_nb_vals), dtype='float64')
        for t, result in enumerate(results):
            nb_columns = result.shape[1]
            final_results[0, start_index: start_index + nb_columns] = t
            final_results[1:4, start_index: start_index + nb_columns] = result
            start_index += nb_columns
        return final_results

    def time_loop(self, data: np.ndarray | dict, data_index: tuple[int, int], index: int = 0, queue: QUEUE | None = None) -> None | list[np.ndarray]:
        """
        Loop over the time indexes so that I can multiprocess if needed be.
        """

        if self.multiprocessing:
            shm = mp.shared_memory.SharedMemory(name=data['shm.name'])
            data = np.ndarray(data['data.shape'], dtype=data['data.dtype'], buffer=shm.buf)

        result_list = []
        for time in range(data_index[0], data_index[1] + 1):
            data_filter = data[0, :] == time
            result = data[1:4, data_filter]
            center = np.array([0, 0, 0]).reshape(3, 1)  # TODO: need to change this when the code works. This is only to see where the sun center is
            result = np.column_stack((result, center))  # adding the sun center to see if the translation is correct
            satelitte_pos = self.sdo_pos[time]

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
            rotation_matrix = self.rodrigues_rotation(rotation_axis, -theta)

            # up_vector in the new rotated matrix
            up_vector_rotated = np.dot(rotation_matrix, up_vector)
            theta = np.arctan2(up_vector_rotated[0], up_vector_rotated[1])
            up_rotation_matrix = self.rodrigues_rotation(target_axis, theta)
            
            # Final rotation matrix
            rotation_matrix = np.matmul(up_rotation_matrix, rotation_matrix)

            result = [np.matmul(rotation_matrix, point) for point in result.T]
            result = np.stack(result, axis=-1)
            result_list.append(result)
        if not self.multiprocessing: return result_list
        shm.close()
        queue.put((index, result_list))
    
    def rodrigues_rotation(self, axis: np.ndarray, angle: float) -> np.ndarray:
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

    def saving_data(self, data: np.ndarray) -> None:
        """
        Function to save the reprojection arrays.
        The arrays saved have shape (4, n) as the whole 3D projection is saved for each time step (maybe the viewpoint depth might be useful later on).
        """

        np.save(os.path.join(self.paths['save'], self.saving_filename), data.astype('float64'))

        # STATS print
        if self.verbose > 0: print(f"SAVING - filename:{self.saving_filename} - shape:{data.shape} - nbytes:{round(data.nbytes / 2**20, 2)}Mb.", flush=self.flush)

    @Decorators.running_time
    def plotting(self, data: np.ndarray) -> None:
        """
        Function to plot the data.
        """

        if self.multiprocessing:
            shm, data = MultiProcessing.shared_memory(data) 
            indexes = MultiProcessing.pool_indexes(len(self.numbers), self.processes)
            processes = [Process(target=self.plotting_sub, kwargs={'data': data, 'data_index': index_tuple}) for index_tuple in indexes]
            for p in processes: p.start()
            for p in processes: p.join()
            shm.unlink()
        else:
            self.plotting_sub(data=data, data_index=(0, len(self.numbers) - 1))

    def plotting_sub(self, data: np.ndarray | dict, data_index: tuple[int, int]) -> None:
        """
        To be able to multiprocess the plotting.
        """
        
        if self.multiprocessing:
            shm = SharedMemory(name=data['shm.name'])
            data = np.ndarray(data['data.shape'], dtype=data['data.dtype'], buffer=shm.buf)

        if self.plot_choices['envelop']: self.envelop_preprocessing()

        for time in range(data_index[0], data_index[1] + 1):
            data_filter  = data[0, :] == time
            result = data[1:4, data_filter]

            # Voxel positions
            x, y, _ = result
            image_nb = self.numbers[time]

            # if self.plot_choices['sdo image']: self.sdo_image(index=time)

            if self.plot_choices['cartesian']:
                # SDO projection plotting
                plt.figure(figsize=(5, 5))
                plt.scatter(x / self.solar_r, y / self.solar_r, s=0.7)
                plt.title(f'SDO POV for image nb{image_nb}')
                plt.xlabel('Solar X [au]')
                plt.ylabel('Solar Y [au]')
                plot_name = f'sdoprojection_{image_nb:03d}_{self.time_interval}.png'
                plt.savefig(os.path.join(self.paths['save'], plot_name), dpi=500)
                plt.close()

            if self.plot_choices['polar']:
                # Changing to polar coordinates
                r = np.sqrt(x**2 + y**2)
                theta = np.arctan2(y, x) - np.pi / 2
                theta = np.where(theta < 0, theta + 2 * np.pi, theta)
                theta = 2 * np.pi - theta  # clockwise
                theta = np.where(theta >= 2 * np.pi, theta - 2 * np.pi, theta)  # modulo 2pi
                theta = np.degrees(theta) 

                # SDO polar projection plotting
                plt.figure(figsize=(12, 5))
                plt.scatter(theta, r / 10**3, s=0.7)
                plt.xlim(245, 295)
                plt.ylim(700, 870)
                plt.title(f'SDO polar projection: {image_nb}')
                plt.xlabel('Polar angle [degrees]')
                plt.ylabel('Radial distance [Mm]')
                plot_name = f'sdopolarprojection_{image_nb:03d}_{self.time_interval}.png'
                plt.savefig(os.path.join(self.paths['save'], plot_name), dpi=500)
                plt.close()

            if self.verbose > 1: print(f'SAVED - filename:{plot_name}', flush=self.flush)

    def envelop_preprocessing(self):
        """
        Opens the two png images of the envelop in polar coordinates. Then, treats the data to use it in
        the polar plots.
        """

        pass

    def sdo_image(self, index: int):
        """
        To open the SDO image data, preprocess it and return it as an array for use in plots.
        """

        image = fits.getdata(self.sdo_filepaths[index], 1)
        pass # TODO: will do it later as I need to take into account CRPIX1 and CRPIX2 but also conversion image to plot values