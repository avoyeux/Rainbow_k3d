"""
To get the projection of the polynomial interpolation inside the envelope. 
This is done to get the angle between the projection seen by SDO and the 3D polynomial representation of the data.
"""

# testing branch creation

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
from extract_envelope import Envelope
from cartesian_to_polar import CartesianToPolar
from common import Decorators, Plot, SSHMirroredFilesystem

#TODO: need to check the memory consumption of my code. seems to be huge af. probably need the np.unique earlier and keep the multiprocessing in one step for each image. 
#TODO: MINOR: need to update the code for when there is only one process used (i.e. no multiprocessing)

class OrthographicalProjection:
    """
    Does the 2D projection of a 3D volume.
    Used to recreate recreate the plots made by Dr. Auchere in his 'the coronal Monsoon' paper.
    """

    @Decorators.running_time
    @typeguard.typechecked
    def __init__(
            self,
            processes: int = 1,
            integration_time_hours: int = 24,
            filename: str = 'sig1e20_leg20_lim0_03.h5',
            data_type : str = 'No duplicates new',
            with_feet: bool = True,
            polynomial_order: int | list[int] = [4, 5, 6],
            plot_choices: str | list[str] = ['polar', 'sdo image', 'no duplicate', 'envelope', 'polynomial'], 
            verbose: int = 1,
            flush: bool = False
        ) -> None:
        """
        Re-projection of the computed 3D volume to recreate a 2D image of what is seen from SDO's POV.
        This is done to recreate the analysis of Dr. Auchere's paper; 'The coronal Monsoon'.

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
        self.processes = processes if processes > 1 else 1
        self.filename = filename
        self.data_type = data_type + feet
        self.foldername = filename.split('.')[0] + ''.join(feet.split(' ')) + 'testing'
        self.polynomial_order = sorted(polynomial_order) if isinstance(polynomial_order, list) else [polynomial_order]
        self.plot_choices = self.plot_choices_creation(plot_choices if isinstance(plot_choices, list) else [plot_choices])
        self.in_local = True
        self.verbose = verbose
        self.flush = flush

        # Important constants
        self.solar_r = 6.96e5  # in km
        self.projection_borders = {
            'radial distance': (690, 870),  # in Mm
            'polar angle': (245, 295),  # in degrees
        }
        Envelope.borders = {**Envelope.borders, **self.projection_borders} 

        # SETUP paths
        self.paths = self.path_setup()
        # GLOBAL data
        self.global_data = self.global_information()
        # Get sdo image paths
        self.sdo_timestamps = self.SDO_image_finder()

        # RUN code
        self.multiprocessing_all()

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
        if not os.path.exists(main_path): 
            main_path = '/home/avoyeux/old_project/avoyeux/'
            self.in_local = False
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

    def global_information(self) -> dict[str, tuple[list, list] | dict[str, list[str] | dict[str, int | float | str]]]:

        # GET envelope
        if self.plot_choices['envelope']:
            envelope_data = Envelope.get(
                polynomial_order=6,
                number_of_points=1e5,
                plot=True,
            )
        else:
            envelope_data = None

        # GET plot colours
        colour_generator = Plot.different_colours(omit=['white', 'red'])
        colours = [
            next(colour_generator)
            for _ in self.polynomial_order
        ]

        # SETUP kwargs for final plots
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
                    self.projection_borders['polar angle'][0],
                    self.projection_borders['polar angle'][1],
                    self.projection_borders['radial distance'][0],
                    self.projection_borders['radial distance'][1],
                ),
                'interpolation': 'none',
                'cmap': 'gray',
                'aspect': 'auto',
                'alpha': 1,
                'origin': 'lower',
                'zorder': 0,
            },
            'contour': {
                'linewidth': 0.8,
                'alpha': 1,
                'zorder': 3,
            },
            'colours': colours,
        }
        global_data = {
            'envelope': envelope_data,
            'plot': plot_kwargs,
        }
        return global_data
    
    @Decorators.running_time
    def multiprocessing_all(self) -> None:
        """
        Setups the multiprocessing of the whole code.
        """

        # STATS data
        with h5py.File(self.paths['data'], 'r') as H5PYFile: indexes = H5PYFile['Time indexes'][...]

        # INFO multiprocessing
        data_len = len(indexes)
        nb_processes = min(self.processes, data_len)
        # SETUP multiprocessing
        processes = [None] * nb_processes
        manager = mp.Manager()
        input_queue = manager.Queue()
        for i in range(data_len): input_queue.put(i)
        for _ in range(nb_processes): input_queue.put(None)
        # RUN multiprocessing
        for i in range(nb_processes):
            p = mp.Process(
                target=self.data_setup,
                kwargs={
                    'input_queue': input_queue,
                },
            )
            p.start()
            processes[i] = p
        for p in processes: p.join()

        if self.in_local: SSHMirroredFilesystem.cleanup()

    def data_setup(
            self,
            input_queue: mp.queues.Queue,
        ) -> None:
        """
        """

        # CONNECTION to server
        if self.in_local: self.connection = SSHMirroredFilesystem(verbose=2)
        
        # Open file
        with h5py.File(self.paths['data'], 'r') as H5PYFile:

            # DATA paths
            time_integrated_path = (
                'Time integrated/' 
                + self.data_type 
                + f'/Time integration of {self.integration_time}.0 hours' 
            )

            # GLOBAL data
            dx = H5PYFile['dx'][0]
            # BORDERS
            border_path = 'Filtered/' + self.data_type
            xmin = H5PYFile[border_path + '/xmin'][...]
            ymin = H5PYFile[border_path + '/ymin'][...]
            zmin = H5PYFile[border_path + '/zmin'][...]

            # POINTERS
            indexes = H5PYFile['Time indexes']
            dates = H5PYFile['Dates']
            sdo_pos_list = H5PYFile['SDO positions']
            no_duplicates = H5PYFile['Filtered/' + self.data_type + '/coords'] 
            cubes = H5PYFile[time_integrated_path + '/coords']
            interpolations = [
                H5PYFile[time_integrated_path + f'/{polynomial_order}th order interpolation/raw_coords']
                for polynomial_order in self.polynomial_order
            ]
            
            while True:
                # INFO process 
                process = input_queue.get()
                if process is None: break

                # DATA
                index = indexes[process]
                date = dates[index].decode('utf8')
                sdo_pos = sdo_pos_list[index]
                no_duplicate = self.filter_data(no_duplicates, process)
                cube = self.filter_data(cubes, process)
                interpolation = [
                    self.filter_data(interp, process)
                    for interp in interpolations
                ]

                # Formatting data
                data = {
                    'dx': dx,
                    'time index': index,
                    'date': date,
                    'xmin': xmin,
                    'ymin': ymin,
                    'zmin': zmin,
                    'sdo_pos': sdo_pos,
                }

                sun_perimeter = 2 * np.pi * self.solar_r
                data['d_theta'] = 360 / (sun_perimeter / data['dx'])
                data['cube'] = self.matrix_rotation(self.cartesian_pos(cube, data), data['sdo_pos'])
                data['no_duplicate'] = self.matrix_rotation(self.cartesian_pos(no_duplicate, data), data['sdo_pos'])
                data['interpolation'] = [
                    self.matrix_rotation(self.cartesian_pos(interp, data), data['sdo_pos'])
                    for interp in interpolation
                ]

                print(f"d_theta and dx are {data['d_theta']}, {data['dx']}")
                self.plotting(data)
        
        if self.in_local: self.connection.close()

    def filter_data(
            self,
            data: np.ndarray,
            process: int,
        ) -> np.ndarray:
        
        data_filter = (data[0, :] == process)
        return data[1:4, data_filter]

    def cartesian_pos(
            self,
            data: np.ndarray,
            data_info: dict[str, int | float | list | np.ndarray],
        ) -> np.ndarray:
        """
        To calculate the heliographic cartesian positions given a ndarray of index positions.

        Args:
            data (np.ndarray): the index positions.
            data_info (dict[str, np.ndarray]): the data information.

        Returns:
            np.ndarray: the corresponding heliographic cartesian positions.
        """

        cubes_sparse_coords = data.astype('float64')

        # Initialisation
        cubes_sparse_coords[0, :] = (cubes_sparse_coords[0, :] * data_info['dx'] + data_info['xmin']) 
        cubes_sparse_coords[1, :] = (cubes_sparse_coords[1, :] * data_info['dx'] + data_info['ymin']) 
        cubes_sparse_coords[2, :] = (cubes_sparse_coords[2, :] * data_info['dx'] + data_info['zmin'])
        return cubes_sparse_coords

    def matrix_rotation(
            self,
            data: np.ndarray,
            sdo_pos: np.ndarray,
        ) -> np.ndarray:

        # Get data
        x, y, z = data

        ####### Explained in markdown file ####### 
        a, b, c = - sdo_pos.astype('float64')
        sign = a / abs(a)

        # Normalisation constants
        new_N_x = 1 / np.sqrt(1 + b**2 / a**2 + ((a**2 + b**2) / (a * c))**2)
        new_N_y = a * c / np.sqrt(a**2 + b**2)
        new_N_z = 1 /  np.sqrt(a**2 + b**2 + c**2)

        # Get new coordinates
        new_x = 1 / new_N_x + sign * new_N_x * (x + y * b / a - z * (a**2 + b**2) / (a * c))
        new_y = 1 / new_N_y + sign * new_N_y * (-x * b / (a * c) + y / c)
        new_z = 1 / new_N_z + sign * new_N_z * (x * a + y * b + z * c)

        rho_polar = np.arccos(new_z / np.sqrt(new_x**2 + new_y**2 + new_z**2))
        theta_polar = (new_y / np.abs(new_y)) * np.arccos(new_x / np.sqrt(new_x**2 + new_y**2))
        theta_polar = np.rad2deg((theta_polar + 2 * np.pi) % (2 * np.pi))
        ##########################################

        # Changing units to km
        rho_polar = np.tan(rho_polar) / new_N_z
        return np.stack([rho_polar, theta_polar], axis=0)

    @Decorators.running_time
    def plotting(
            self,
            data_info: dict[str, int | float | list | np.ndarray]
        ) -> None:
        """
        """

        if self.global_data['envelope'] is not None: middle_t_curve, envelope_y_x_curve = self.global_data['envelope']

        # Voxel positions
        r_cube, theta_cube = data_info['cube']
        r_no_duplicate, theta_no_duplicate = data_info['no_duplicate']
        x_interp, y_interp = [
            [
                interp[i]
                for interp in data_info['interpolation']
            ]
            for i in range(2)
        ]

        # SDO polar projection plotting
        plt.figure(figsize=(14, 5))
        if self.plot_choices['envelope']: 

            plt.plot(middle_t_curve[0], middle_t_curve[1], color='black', label='Middle path', **self.global_data['plot']['envelope'])
            
            envelope = envelope_y_x_curve[0]
            plt.plot(envelope[0], envelope[1], color='grey', label='Envelope', **self.global_data['plot']['envelope'])
            for envelope in envelope_y_x_curve[1:]: plt.plot(envelope[0], envelope[1], color='grey', **self.global_data['plot']['envelope'])

        if self.plot_choices['sdo image']: 
            # SDO mask
            filepath = os.path.join(self.paths['sdo'], f"AIA_fullhead_{data_info['time index']:03d}.fits.gz")
            polar_image_info = self.sdo_image(filepath)
            image = np.zeros(polar_image_info['image']['data'].shape)
            image[polar_image_info['image']['data'] > 0] = 1

            # Get contours
            lines = self.image_contour(
                image=image,
                dx=polar_image_info['image']['dx'],
                d_theta=polar_image_info['image']['d_theta'],
            )

            # Plot contours
            line = lines[0]
            plt.plot(line[1], line[0], color='green', label='SDO mask contours', **self.global_data['plot']['contour'])
            for line in lines[1:]: plt.plot(line[1], line[0], color='green', **self.global_data['plot']['contour'])

            # SDO image
            filepath = self.sdo_timestamps[data_info['date'][:-3]]
            if not os.path.exists(filepath): filepath = self.connection.mirror(filepath, strip_level=1)
            sdo_image_info = self.sdo_image(filepath)

            plt.imshow(self.sdo_image_treatment(sdo_image_info['image']['data']), **self.global_data['plot']['image'])

            if self.plot_choices['cube']: 
                image_shape = (
                    int((self.projection_borders['radial distance'][1] - self.projection_borders['radial distance'][0]) * 1e3 / sdo_image_info['dx']),
                    int((self.projection_borders['polar angle'][1] - self.projection_borders['polar angle'][0]) / sdo_image_info['d_theta']),
                )

                # CONTOURS plot
                self.plot_contours(
                    rho=r_cube,
                    theta=theta_cube,
                    d_theta=data_info['d_theta'],
                    dx=data_info['dx'],
                    image_shape=image_shape,
                    color='red',
                    label='time integrated contours',
                )
                self.plot_contours(
                    rho=r_no_duplicate,
                    theta=theta_no_duplicate,
                    d_theta=data_info['d_theta'],
                    dx=data_info['dx'],
                    image_shape=image_shape,
                    color='orange',
                    label='no duplicate contours',
                )
                
        if self.plot_choices['interpolations']: 
            for i in range(len(data_info['interpolation'])):
                # Get polar positions
                r_interp, theta_interp = x_interp[i], y_interp[i]

                # Plot
                plt.scatter(
                    theta_interp,
                    r_interp / 10**3,
                    label=f'{self.polynomial_order[i]}th order polynomial',
                    color=self.global_data['plot']['colours'][i],
                    **self.global_data['plot']['interpolation'],
                )

        plt.xlim(self.projection_borders['polar angle'][0], self.projection_borders['polar angle'][1])
        plt.ylim(self.projection_borders['radial distance'][0], self.projection_borders['radial distance'][1])
        ax = plt.gca()
        ax.minorticks_on()
        ax.set_aspect('auto')
        plt.title(f"SDO polar projection - {data_info['date']}")
        plt.xlabel('Polar angle [degrees]')
        plt.ylabel('Radial distance [Mm]')
        plt.legend(loc='upper right')
        plot_name = f"sdopolarprojection_{data_info['date']}_{self.integration_time}h.png"
        plt.savefig(os.path.join(self.paths['save'], plot_name), dpi=500)
        plt.close()

        if self.verbose > 1: 
            print(f"the image nb is {data_info['time index']}")
            print(f'SAVED - filename: {plot_name}', flush=self.flush)

    def plot_contours(
            self,
            rho: np.ndarray,
            theta: np.ndarray,
            d_theta: float, 
            dx: float,
            image_shape: tuple[int, int],
            color: str,
            label: str,
        ) -> None:

        # Initialisation of an image
        image = np.zeros(image_shape, dtype='int8')

        _, lines = self.cube_contour(
            rho=rho,
            theta=theta,
            empty_image=image,
            d_theta=d_theta,
            dx=dx,
        )

        # Plot
        if lines is not None:
            line = lines[0]
            plt.plot(line[1], line[0], color=color, label=label, **self.global_data['plot']['contour'])
            for line in lines: plt.plot(line[1], line[0], color=color, **self.global_data['plot']['contour'])

    def cube_contour(
            self,
            rho: np.ndarray,
            theta: np.ndarray,
            empty_image: np.ndarray,
            dx: float,
            d_theta: float,
        ) -> tuple[np.ndarray, list[tuple[list[float], list[float]]]] | tuple[None, None]:

        # Starting from image border
        rho -= min(self.projection_borders['radial distance']) * 1000
        theta -= min(self.projection_borders['polar angle'])

        # Binning
        rho //= dx
        theta //= d_theta

        # Filtering duplicates
        polar = np.stack([rho, theta], axis=0)
        polar_indexes = np.unique(polar, axis=1).astype('int64')

        # Keeping indexes inside the image
        rho_filter = (polar_indexes[0] > 0) & (polar_indexes[0] < empty_image.shape[0])
        theta_filter = (polar_indexes[1] > 0) & (polar_indexes[1] < empty_image.shape[1])
        full_filter = rho_filter & theta_filter

        # Getting final image indexes
        rho = polar_indexes[0][full_filter]
        theta = polar_indexes[1][full_filter]

        if len(rho) == 0: return None, None

        empty_image[rho, theta] = 1
        lines = self.image_contour(empty_image, dx, d_theta)
        return empty_image, lines
    
    def image_contour(
            self,
            image: np.ndarray,
            dx: float,
            d_theta: float,
    ) -> list[tuple[list[float], list[float]]]:
        """ #TODO: docstring
        To get the contours in the final plot coordinates of a mask given the corresponding information.

        Args:
            image (np.ndarray): the mask.
            projection_borders (dict[str, tuple[int, int]]): the borders of the mask.
            dx (float): the vertical pixel length in km.
            d_theta (float): the horizontal pixel length in degrees.

        Returns:
            list[tuple[list[float], list[float]]]: the lines representing the mask contours.
        """

        # print(f'image contour dx is {dx} and d_theta {d_theta}', flush=True)

        # Get contours
        lines = Plot.contours(image)

        # Get corresponding polar coordinates
        nw_lines = [None] * len(lines)
        for i, line in enumerate(lines):
            nw_lines[i] = ((
                [self.projection_borders['radial distance'][0] + (value * dx) / 1e3 for value in line[0]],
                [self.projection_borders['polar angle'][0] + (value * d_theta) for value in line[1]],
            ))
        return nw_lines

    def sdo_image(
            self,
            filepath: str,
        ) -> dict[str | dict[str, float | np.ndarray], float]:
        """ #TODO: update docstring
        To get the sdo image in polar coordinates and delimited by the final plot borders. Furthermore, needed information are also saved in the 
        output, e.g. dx and d_theta for the created sdo image in polar coordinates.

        Args:
            filepath (str): the filepath to the corresponding sdo FITS file.
            projection_borders (dict[str, tuple[int, int]]): the plot borders.

        Returns:
            dict[str, float | np.ndarray]: the polar sdo image with the necessary image information, e.g. dx and d_theta.
        """

        polar_image_info = CartesianToPolar.get_polar_image(
            filepath=filepath,
            borders=self.projection_borders,
            direction='clockwise',
            theta_offset=90,
            channel_axis=None,
        )
        return polar_image_info
    
    def sdo_image_treatment(
            self,
            image: np.ndarray,
        ) -> np.ndarray:
        """ #TODO: dccstring
        Pre-treatment for the sdo image for better visualisation of the regions of interest.

        Args:
            image (np.ndarray): the SDO image to be treated.

        Returns:
            np.ndarray: the treated SDO image.
        """
        
        # Clipping
        lower_cut = np.nanpercentile(image, 2)
        higher_cut = np.nanpercentile(image, 99.99)

        # Saturating
        image[image < lower_cut] = lower_cut
        image[image > higher_cut] = higher_cut

        # Changing to log
        return np.log(image)
    
    def SDO_image_finder(self) -> dict[str, str]:
        """
        To find the SDO image given its header timestamp and a list of corresponding paths to the corresponding fits file.

        Returns:
            dict[str, str]: the timestamps as keys with the item being the SDO image filepath.
        """

        # Setup
        filepath_end = '/S00000/image_lev1.fits'
        with open(os.path.join(self.paths['codes'], 'SDO_timestamps.txt'), 'r') as files:
            strings = files.read().splitlines()
        tuple_list = [s.split(" ; ") for s in strings]
    
        timestamp_to_path = {}
        for s in tuple_list:
            path, timestamp = s
            timestamp = timestamp.replace(':', '-')[:-6]

            # Weird exception...
            if timestamp == '2012-07-24T20-07': timestamp = '2012-07-24T20-06'
            if timestamp == '2012-07-24T20-20': timestamp = '2012-07-24T20-16'

            timestamp_to_path[timestamp] = path + filepath_end
        return timestamp_to_path



if __name__ == '__main__':
    OrthographicalProjection(
        filename='sig1e20_leg20_lim0_03.h5',
        with_feet=True,
        verbose=2,
        processes=2,
        polynomial_order=[4],
        plot_choices=['polar', 'cube', 'interpolations', 'envelope', 'sdo image'],
        flush=True,
    )





