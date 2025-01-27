"""
To get the projection of the polynomial polynomial inside the envelope. 
This is done to get the angle between the projection seen by SDO and the 3D polynomial
representation of the data.
"""

# IMPORTS
import os
import h5py
import typeguard

# IMPORTS alias
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt

# IMPORTS sub
import multiprocessing.queues

# IMPORTS personal
from common import Decorators, Plot, root_path
from Projection.extract_envelope import Envelope, CreateFitEnvelope
from Projection.cartesian_to_polar import CartesianToPolar
from Projection.projection_dataclasses import *
from Data.get_polynomial import GetCartesianProcessedPolynomial

# todo MINOR: need to update the code for when there is only one process used
# (i.e. no multiprocessing)



class OrthographicalProjection:
    """
    Adds the protuberance voxels (with the corresponding polynomial fit) on SDO's image.
    Different choices are possible in what to plot in the final SDO image.
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
            plot_choices: str | list[str] = [
                'polar', 'sdo image', 'no duplicate', 'envelope', 'polynomial',
            ], 
            verbose: int = 1,
            flush: bool = False
        ) -> None:
        """
        Re-projection of the computed 3D volume to recreate a 2D image of what is seen from SDO's
        POV.
        This is done to recreate the analysis of Dr. Auchere's paper; 'The coronal Monsoon'.

        Args:
            processes (int, optional): the number of parallel processes used in the
                multiprocessing. Defaults to 0.
            integration_time_hours (int, optional): the integration time used for the data to be
                reprojected. Defaults to 24.
            filename (str, optional): the filename of the HDF5 containing all the relevant 3D data.
                Defaults to 'sig1e20_leg20_lim0_03_thisone.h5'.
            data_type (str, optional): the data type of the 3D object to be reprojected
                (e.g. All data). Defaults to 'No duplicates new'.
            with_feet (bool, optional): deciding to use the data with or without added feet.
                Defaults to True.
            polynomial_order (int | list[int], optional): the order(s) of the polynomial
                function(s) that represent the fitting of the integrated 3D volume.
                Defaults to [4, 5, 6].
            plot_choices (str | list[str], optional): the main choices that the user wants to be in
                the reprojection. The possible choices are:
                ['polar', 'cartesian', 'integration', 'no duplicate', 'sdo image', 'sdo mask',
                'envelope', 'fit', 'fit envelope', 'test']. 
            verbose (int, optional): gives the verbosity in the outputted prints. The higher the
                value, the more prints. Starts at 0 for no prints. Defaults to 1.
            flush (bool, optional): used in the 'flush' kwarg of the print() class. Decides to
                force the print buffer to be emptied (i.e. forces the prints). Defaults to False.
        """

        # ATTRIBUTES
        feet = ' with feet' if with_feet else ''
        self.integration_time = integration_time_hours
        self.processes = processes if processes > 1 else 1
        self.filename = filename
        self.data_type = data_type + feet
        self.foldername = filename.split('.')[0] + ''.join(feet.split(' ')) + 'testing'
        if isinstance(polynomial_order, list):
            self.polynomial_order = sorted(polynomial_order)
        else:
            self.polynomial_order = [polynomial_order]
        self.plot_choices = self.plot_choices_creation(plot_choices)
        self.in_local = True
        self.verbose = verbose
        self.flush = flush

        # CONSTANTS
        self.solar_r = 6.96e5  # in km
        self.projection_borders = ImageBorders(
            radial_distance=(690, 870),  # in Mm
            polar_angle=(245, 295),  # in degrees
        )

        # SETUP paths
        self.paths = self.path_setup()
        # SERVER connection
        if self.in_local:
            from common.server_connection import SSHMirroredFilesystem
            self.SSHMirroredFilesystem = SSHMirroredFilesystem
        # GLOBAL data
        self.Auchere_envelope, self.plot_kwargs = self.global_information()
        # PATHS sdo image
        self.sdo_timestamps = self.sdo_image_finder()
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

        # PATH setup
        main_path = os.path.join(root_path, '..')

        # PATHs save
        paths = {
            'main': main_path,
            'codes': root_path,
            'data': os.path.join(root_path, 'Data', self.filename),
            'sdo': os.path.join(main_path, 'sdo'),
            'envelope': os.path.join(main_path, 'Work_done', 'Envelope'),
            'save': os.path.join(main_path, 'Work_done', self.foldername),
        }
        os.makedirs(paths['save'], exist_ok=True)
        return paths

    def plot_choices_creation(self, plot_choices: str | list[str]) -> dict[str, bool]:
        """ 
        Creating a dictionary that chooses what to plot.

        Args:
            plot_choices (str | list[str]): choices made for the plotting.

        Raises:
            ValueError: if the plotting choice string is not recognised.

        Returns:
            dict[str, bool]: decides what will be plotted later on.
        """

        plot_choices = plot_choices if isinstance(plot_choices, list) else [plot_choices]

        # CHOICES
        possibilities = [
            'polar', 'cartesian', 'integration', 'no duplicate', 'sdo image', 'sdo mask',
            'envelope', 'fit', 'fit envelope', 'test'
        ]
        plot_choices_kwargs = {
            key: False 
            for key in possibilities
        }

        # DICT reformatting
        for key in plot_choices: 
            if key in possibilities: 
                plot_choices_kwargs[key] = True
            else: 
                raise ValueError(f"Value for the 'plot_choices' argument not recognised.") 
        if 'envelope' in plot_choices: plot_choices_kwargs['polar'] = True
        return plot_choices_kwargs

    def global_information(
            self,
        ) -> tuple[EnvelopeInformation | None, dict[str, dict[str, list | int | float]]]:
        """ 
        Contains the default choices made for the plotting options (e.g. the opacity, linestyle).

        Returns:
            tuple[EnvelopeInformation | None, dict[str, dict[str, list | int | float]]]: Dr.
            Auchere's envelope data and the default plotting information.
        """

        # ENVELOPE get
        envelope_data = Envelope.get(
            polynomial_order=6,  # ? should I find the image shape from the image itself ?
            number_of_points=int(1e5),
            borders=self.projection_borders,
            verbose=self.verbose,
        ) if self.plot_choices['envelope'] else None

        # COLOURS plot
        colour_generator = Plot.different_colours(omit=['white', 'red'])
        colours = [
            next(colour_generator)
            for _ in self.polynomial_order
        ]

        # SETUP plots kwargs
        plot_kwargs = {
            'fit': {
                'cmap': 'seismic',
                'vmin': -55,
                'vmax': 55,
                's': 1.5,
                'zorder': 9,
            },
            'envelope': {
                'linestyle': '--',
                'alpha': 0.8,
                'zorder': 4,
            },
            'fit envelope': {
                'linestyle': '--',
                'color': 'yellow',
                'alpha': 0.8,
                'zorder':4,
            },
            'image': {
                'extent': (
                    min(self.projection_borders.polar_angle),
                    max(self.projection_borders.polar_angle),
                    min(self.projection_borders.radial_distance),
                    max(self.projection_borders.radial_distance),
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
        return envelope_data, plot_kwargs
    
    @Decorators.running_time
    def multiprocessing_all(self) -> None:
        """
        Setups the multiprocessing for the whole code.
        """

        # STATS data
        with h5py.File(self.paths['data'], 'r') as H5PYFile:
            indexes: np.ndarray = H5PYFile['Time indexes'][...]

        # INFO multiprocessing
        data_len = len(indexes)
        nb_processes = min(self.processes, data_len)
        # SETUP multiprocessing
        processes: list[mp.Process] = [None] * nb_processes
        manager = mp.Manager()
        input_queue = manager.Queue()
        for i in range(data_len): input_queue.put(i)
        for _ in range(nb_processes): input_queue.put(None)
        # RUN multiprocessing
        for i in range(nb_processes):
            p = mp.Process(
                target=self.data_setup,
                kwargs={'input_queue': input_queue},
            )
            p.start()
            processes[i] = p
        for p in processes: p.join()

        if self.in_local: self.SSHMirroredFilesystem.cleanup(verbose=self.verbose)
    
    def data_setup(self, input_queue: mp.queues.Queue) -> None:
        """ 
        Open the HDF5 file and does the processing and final plotting for each cube.
        A while loop is used to decide which data section needs to be processed.

        Args:
            input_queue (mp.queues.Queue): contains the data cube indexes that still need
                processing.
        """

        # CONNECTION to server
        if self.in_local: self.connection = self.SSHMirroredFilesystem(verbose=self.verbose)
        
        # DATA open
        with h5py.File(self.paths['data'], 'r') as H5PYFile:

            # GLOBAL constants
            self.constants = self.get_global_constants(H5PYFile)
            # ! sdo pos needs to be here at the end

            # DATA paths
            filtered_path = 'Filtered/' + self.data_type
            time_integrated_path = (
                'Time integrated/' 
                + self.data_type 
                + f'/Time integration of {self.integration_time}.0 hours' 
            )

            # DATA formatting
            cubes_info = self.get_cubes_information(H5PYFile, filtered_path)
            integrated_info = self.get_cubes_information(H5PYFile, time_integrated_path)

            # if self.plot_choices['test']:
            #     # TEST data  # ! need to change this
            #     test_path = 'TEST data/Sun index'
            #     test_data = H5PYFile[test_path + '/coords']

            #     test_info = {
            #         'dx': dx,
            #         'xmin': H5PYFile[test_path + '/xmin'][...].astype('float64'),
            #         'ymin': H5PYFile[test_path + '/ymin'][...].astype('float64'),
            #         'zmin': H5PYFile[test_path + '/zmin'][...].astype('float64'),
            #     }
            
            while True:
                # INFO process 
                process: int = input_queue.get()
                if process is None: break
                
                # DATA formatting
                time_index = self.constants.time_indexes[process]
                process_constants = ProcessConstants(
                    ID=process,
                    time_index=time_index,
                    date=self.constants.dates[time_index].decode('utf8'),
                )
                projection_data = ProjectionData(ID=process)

                # SDO information
                filepath: str = self.sdo_timestamps[process_constants.date[:-3]]
                if self.in_local: filepath = self.connection.mirror(filepath, strip_level=1)
                sdo_image_info = self.sdo_image(filepath) 
                # todo make the code more readable, right now I did too much random patching.

                if self.plot_choices['sdo image']: projection_data.sdo_image = sdo_image_info

                if self.plot_choices['sdo mask']:
                    # SDO mask
                    filepath = os.path.join(
                        self.paths['sdo'],
                        f"AIA_fullhead_{process_constants.time_index:03d}.fits.gz",
                    )
                    polar_mask_info = self.sdo_image(filepath)
                    image = np.zeros(polar_mask_info.image.shape)
                    image[polar_mask_info.image > 0] = 1
                    polar_mask_info.image = image
                    projection_data.sdo_mask = polar_mask_info

                if self.plot_choices['integration']:
    
                    # ! this is wrong, need to get the integration data. as I create the integration
                    #one by one I need to give a different dataclass to the integration data
                    integration = CubeInformation(
                        xt_min=integrated_info.xt_min,
                        yt_min=integrated_info.yt_min,
                        zt_min=integrated_info.zt_min,
                        coords=integrated_info[process],
                    )
                    integration = self.cartesian_pos(integration)

                    # DATA formatting
                    projection_data.integration = self.get_polar_image(self.matrix_rotation(
                        data=integration.coords,
                        sdo_pos=sdo_image_info.sdo_pos,
                    ))

                if self.plot_choices['no duplicate']: # ? should I change the argument names with
                    #?self.data_type

                    cube = CubeInformation(
                        xt_min=cubes_info.xt_min,
                        yt_min=cubes_info.yt_min,
                        zt_min=cubes_info.zt_min,
                        coords=cubes_info[process],
                    )
                    cube = self.cartesian_pos(cube)
                    projection_data.cube = self.get_polar_image(self.matrix_rotation(
                        data=cube.coords,
                        sdo_pos=sdo_image_info.sdo_pos,
                    ))

                if self.plot_choices['fit']:

                    polynomials_info: list[PolynomialInformation] = (
                        [None] * len(self.polynomial_order)
                    )#type: ignore

                    for i, poly_order in enumerate(self.polynomial_order):

                        polynomial_instance = GetCartesianProcessedPolynomial(
                            polynomial_order=poly_order,
                            integration_time=self.integration_time,
                            number_of_points=250,
                            dx=self.constants.dx,
                            data_type=self.data_type,
                        )
                        initial_data = polynomial_instance.reprocessed_polynomial(process)

                        polar_r, polar_theta, angles = self.get_polar_image_angles(
                            self.matrix_rotation(
                                data=self.cartesian_pos(initial_data).coords,
                                sdo_pos=sdo_image_info.sdo_pos,
                            ))

                        # DATA formatting
                        polynomial_information = PolynomialInformation(
                            order=poly_order,
                            xt_min=initial_data.xt_min,
                            yt_min=initial_data.yt_min,
                            zt_min=initial_data.zt_min,
                            polar_r=polar_r,
                            polar_theta=polar_theta,
                            angles=angles,
                        )
                        polynomials_info[i] = polynomial_information

                        # HDF5 close
                        polynomial_instance.close()
                    projection_data.fits = polynomials_info
                
                # if self.plot_choices['test']:
                #     test = self.filter_data(test_data, process)
                #     data['test'] = self.get_polar_image(self.matrix_rotation(
                #         data=self.cartesian_pos(test, test_info),
                #         sdo_pos=sdo_image_info['sdo_pos'],
                #     ))
                # # print(f"d_theta and dx are {data['d_theta']}, {data['dx']}")
                # self.plotting(data)
                self.plotting(process_constants, projection_data)
        if self.in_local: self.connection.close()

    def get_global_constants(self, H5PYFile: h5py.File) -> GlobalConstants:
        """
        To get the global constants of the data.

        Args:
            H5PYFile (h5py.File): the HDF5 file containing the data.

        Returns:
            GlobalConstants: the global constants of the data.
        """

        # DATA open
        dx = float(H5PYFile['dx'][...])
        time_indexes: np.ndarray = H5PYFile['Time indexes'][...]
        dates: np.ndarray = H5PYFile['Dates'][...]

        # FORMAT data
        constants = GlobalConstants(
            dx=dx,
            solar_r=self.solar_r,
            time_indexes=time_indexes,
            dates=dates,
        )
        return constants

    def get_cubes_information(self, H5PYFile: h5py.File, group_path: str) -> CubesInformation:
        """
        To get the information about the data cubes.

        Args:
            H5PYFile (h5py.File): the HDF5 file containing the data.
            group_path (str): the path to the group containing the data.

        Returns:
            CubesInformation: the information about the data cubes.
        """

        # BORDERS
        xt_min = float(H5PYFile[group_path + '/xt_min'][...])
        yt_min = float(H5PYFile[group_path + '/yt_min'][...])
        zt_min = float(H5PYFile[group_path + '/zt_min'][...])

        # FORMAT data
        cube_info = CubesInformation(
            xt_min=xt_min,
            yt_min=yt_min,
            zt_min=zt_min,
            pointer=H5PYFile[group_path + '/coords'],
        )
        return cube_info

    def cartesian_pos(self, data: CubeInformation) -> CubeInformation:
        """ # todo update docstring
        To calculate the heliographic cartesian positions given a ndarray of index positions.

        Args:
            data (np.ndarray): the index positions.
            data_info (dict[str, np.ndarray]): the data information.

        Returns:
            np.ndarray: the corresponding heliographic cartesian positions.
        """

        data.coords[0, :] = data.coords[0, :] * self.constants.dx + data.xt_min
        data.coords[1, :] = data.coords[1, :] * self.constants.dx + data.yt_min
        data.coords[2, :] = data.coords[2, :] * self.constants.dx + data.zt_min
        return data

    def get_polar_image(self, data: tuple[np.ndarray, float]) -> np.ndarray:
        """ 
        Gives the polar coordinates in SDO's image reference frame of the protuberance voxels.

        Args:
            data (tuple[np.ndarray, float]): the heliocentric cartesian positions of the
                protuberance voxels.

        Returns:
            np.ndarray: (r, theta) of the voxels in polar coordinates centred on the disk center as
                seen from SDO and with theta starting from the projected solar north pole.
        """
        
        # DATA open
        coords, z_norm = data
        x, y, z = coords

        # IMAGE polar coordinates
        rho_polar = np.arccos(z / np.sqrt(x**2 + y**2 + z**2))
        theta_polar = (y / np.abs(y)) * np.arccos(x / np.sqrt(x**2 + y**2))
        theta_polar = np.rad2deg((theta_polar + 2 * np.pi) % (2 * np.pi))

        # UNITs to km
        rho_polar = np.tan(rho_polar) / z_norm
        return np.stack([rho_polar, theta_polar], axis=0)
    
    def get_angles(self, coords: np.ndarray) -> np.ndarray:
        """ 
        Gives the angle between the polynomial fit and the SDO image plane. 

        Args:
            coords (np.ndarray): the coordinates of the voxels in heliocentric cartesian
                coordinates.

        Returns:
            np.ndarray: the angles between the coordinates (for b_{n+1} - b_{n}) and SDO's image
                plane. Information needed to correct the velocities seen in 2D in SDO's image.
        """
        
        x, y, z = coords

        # DIRECTIONS a_n = b_{n+1} - b{n}
        x_direction = x[1:] - x[:-1]
        y_direction = y[1:] - y[:-1]
        z_direction = z[1:] - z[:-1]

        # ANGLE rho - image plane
        theta_spherical = np.arccos(
            z_direction / np.sqrt(x_direction**2 + y_direction**2 + z_direction**2)
        )
        theta_spherical -= np.pi / 2
        return theta_spherical
    
    def get_polar_image_angles(self, data: tuple[np.ndarray, float]) -> np.ndarray:
        """ 
        Gives the polar coordinates (r, theta) in the created SDO image (i.e. centred on the disk
        center and with theta starting from the north pole direction). Furthermore, the angle of
        the polynomial fit relative to the SDO image plane is also computed.

        Args:
            data (tuple[np.ndarray, float]): the voxel position in heliocentric cartesian
                coordinates.

        Returns:
            np.ndarray: (r, theta, angle) in the SDO image reference frame.
        """

        #TODO: add an explanation in the equation .md file.

        # DATA open
        coords, _ = data

        # ANGLES
        angles = self.get_angles(coords)

        # POLAR pos
        rho_polar, theta_polar = self.get_polar_image(data)
        return np.stack([rho_polar[:-1], theta_polar[:-1], angles], axis=0)

    def matrix_rotation(self, data: np.ndarray, sdo_pos: np.ndarray) -> tuple[np.ndarray, float]:
        """ 
        Gives the cartesian positions of the voxels in an orthonormal coordinates system centred on
        SDO's position and with the new z-axis pointing to the Sun's center.

        Args:
            data (np.ndarray): the (x, y, z) coordinates of the data voxels in heliocentric
                cartesian coordinates.
            sdo_pos (np.ndarray): the position of the SDO satellite in heliocentric cartesian
                coordinates.

        Returns:
            tuple[np.ndarray, float]: the voxel coordinates in the new reference frame, with the
                normalisation constant of the Z-axis (later needed to calculate the projected polar
                radius from the disk center to each voxel).
        """

        # DATA open
        x, y, z = data
        a, b, c = - sdo_pos.astype('float64')
        sign = a / abs(a)

        # CONSTANTs normalisation
        new_N_x = 1 / np.sqrt(1 + b**2 / a**2 + ((a**2 + b**2) / (a * c))**2)
        new_N_y = a * c / np.sqrt(a**2 + b**2)
        new_N_z = 1 /  np.sqrt(a**2 + b**2 + c**2)

        # COORDS new
        new_x = 1 / new_N_x + sign * new_N_x * (x + y * b / a - z * (a**2 + b**2) / (a * c))
        new_y = 1 / new_N_y + sign * new_N_y * (-x * b / (a * c) + y / c)
        new_z = 1 / new_N_z + sign * new_N_z * (x * a + y * b + z * c)
        
        # DATA return
        coords = np.stack([new_x, new_y, new_z], axis=0)
        return coords, new_N_z

    @Decorators.running_time
    def plotting(
            self,
            process_constants: ProcessConstants,
            projection_data: ProjectionData,
        ) -> None:
        """   # todo update docstring
        To plot the SDO's point of view image.
        What is plotted is dependent on the chosen choices when running the class.

        Args:
            data_info (dict[str, int  |  float  |  list  |  np.ndarray]): contains all the
                processed data initially gotten from the HDF5 file (i.e. the 3D data).
        """

        # IMAGE shape
        image_shape = (
            int(
                (
                    max(self.projection_borders.radial_distance) - 
                    min(self.projection_borders.radial_distance)
                ) * 1e3 / self.constants.dx
            ),
            int(
                (
                    max(self.projection_borders.polar_angle) -
                    min(self.projection_borders.polar_angle)
                ) / self.constants.d_theta
            ),
        )

        # SDO polar projection plotting
        plt.figure(figsize=(18, 5))
        if self.Auchere_envelope is not None: 

            plt.plot(
                self.Auchere_envelope.middle.x_t,
                self.Auchere_envelope.middle.y_t,
                color='black',
                label='Middle path',
                **self.plot_kwargs['envelope'],
            )
            
            plt.plot(
                self.Auchere_envelope.upper.x,
                self.Auchere_envelope.upper.y,
                color='black',
                label='Envelope',
                **self.plot_kwargs['envelope'],
            )
            plt.plot(
                self.Auchere_envelope.lower.x,
                self.Auchere_envelope.lower.y,
                color='black',
                **self.plot_kwargs['envelope'],
            )

        if projection_data.sdo_mask is not None:

            # CONTOURS get
            lines = self.image_contour(
                image=projection_data.sdo_mask.image,
                dx=projection_data.sdo_mask.resolution_km,
                d_theta=projection_data.sdo_mask.resolution_angle,
            )

            # CONTOURS plot
            line = lines[0]
            plt.plot(
                line[1],
                line[0],
                color='green',
                label='SDO mask contours',
                **self.plot_kwargs['contour'],
            )
            for line in lines[1:]:
                plt.plot(
                    line[1],
                    line[0],
                    color='green',
                    **self.plot_kwargs['contour'],
                )

        if projection_data.sdo_image is not None:

            plt.imshow(
                self.sdo_image_treatment(projection_data.sdo_image.image),
                **self.plot_kwargs['image'],
            )

        if projection_data.integration is not None:
            # DATA (r, theta)
            r_cube, theta_cube = projection_data.integration

            # PLOT contours time integrated
            self.plot_contours(
                rho=r_cube,
                theta=theta_cube,
                d_theta=self.constants.d_theta,
                dx=self.constants.dx,
                image_shape=image_shape,
                color='red',
                label='time integrated contours',
            )
        
        if projection_data.cube is not None:
            # DATA (r, theta) no duplicate
            r_no_duplicate, theta_no_duplicate = projection_data.cube

            # PLOT contours no duplicates
            self.plot_contours(
                rho=r_no_duplicate,
                theta=theta_no_duplicate,
                d_theta=self.constants.d_theta,
                dx=self.constants.dx,
                image_shape=image_shape,
                color='orange',
                label='no duplicate contours',
            )
        
        if projection_data.fits is not None:

            for i in range(len(projection_data.fits)):
                
                # POLAR values
                polar_r = projection_data.fits[i].polar_r
                polar_theta = projection_data.fits[i].polar_theta
                ang = projection_data.fits[i].angles

                # PLOT
                sc = plt.scatter(
                    polar_theta,
                    polar_r / 10**3,
                    label=f'{self.polynomial_order[i]}th order polynomial',
                    c=np.rad2deg(ang),
                    **self.plot_kwargs['fit'],
                )
                cbar = plt.colorbar(sc)
                cbar.set_label(r'$\theta$ (degrees)')

                # ENVELOPE polynomial
                if self.plot_choices['fit envelope']:
                    # ENVELOPE get
                    envelopes = CreateFitEnvelope.get(
                        coords=np.stack([polar_r, polar_theta], axis=0),
                        radius = 3e4,
                    )

                    # PLOT envelope
                    for label, new_envelope in enumerate(envelopes):
                        r_env, theta_env = new_envelope
                        plt.plot(
                            theta_env,
                            r_env / 1e3,
                            label='fit envelope' if label==0 else None,
                            **self.plot_kwargs['fit envelope'],
                        )

        # # TEST plot
        # if self.plot_choices['test']:
        #     # DATA (r, theta) fake
        #     r_fake, theta_fake = data_info['test']

        #     self.plot_contours(
        #         rho=r_fake,
        #         theta=theta_fake,
        #         d_theta=data_info['d_theta'],
        #         dx=data_info['dx'],
        #         image_shape=image_shape,
        #         color='white',
        #         label='fake data',
        #     )
        # PLOT settings
        plt.xlim(
            min(self.projection_borders.polar_angle),
            max(self.projection_borders.polar_angle),
        )
        plt.ylim(
            min(self.projection_borders.radial_distance),
            max(self.projection_borders.radial_distance),
        )
        ax = plt.gca()
        ax.minorticks_on()
        ax.set_aspect('auto')
        plt.title(f"SDO polar projection - {process_constants.date}")
        plt.xlabel('Polar angle [degrees]')
        plt.ylabel('Radial distance [Mm]')
        plt.legend(loc='upper right')
        plot_name = f"sdopolarprojection_{process_constants.date}_{self.integration_time}h.png"
        plt.savefig(os.path.join(self.paths['save'], plot_name), dpi=500)
        plt.close()

        if self.verbose > 1: 
            print(f"IMAGE nb {process_constants.time_index}")
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
        """ 
        To plot the contours of the image for the protuberance as seen from SDO's pov.

        Args:
            rho (np.ndarray): the distance positions of the voxels from the disk center seen by
                SDO. The distances are in km. 
            theta (np.ndarray): the theta polar angle position relative to the solar north pole and
                centred on the disk center as seen from SDO.
            image_shape (tuple[int, int]): the image shape needed for the image of the protuberance
                as seen from SDO.
            dx (float): the voxel resolution in km.
            d_theta (float): the theta angle resolution (as a function of the disk's perimeter) in
                degrees.
            color (str): the color used in the plot of the contours of the SDO image.
            label (str): the label in the plot for the contours of the SDO image.
        """

        # CONTOURS cube
        _, lines = self.cube_contour(
            rho=rho,
            theta=theta,
            image_shape=image_shape,
            d_theta=d_theta,
            dx=dx,
        )

        # PLOT
        if lines is not None:
            line = lines[0]
            plt.plot(
                line[1],
                line[0],
                color=color,
                label=label,
                **self.plot_kwargs['contour'],
            )
            for line in lines:
                plt.plot(line[1], line[0], color=color, **self.plot_kwargs['contour'])

    def cube_contour(
            self,
            rho: np.ndarray,
            theta: np.ndarray,
            image_shape: tuple[int, int],
            dx: float,
            d_theta: float,
        ) -> tuple[np.ndarray, list[tuple[list[float], list[float]]]] | tuple[None, None]:
        """ 
        To get the contours of the protuberances voxels if seen as an image from SDO's point of
        view.

        Args:
            rho (np.ndarray): the distance positions of the voxels from the disk center seen by
                SDO. The distances are in km. 
            theta (np.ndarray): the theta polar angle position relative to the solar north pole and
                centred on the disk center as seen from SDO.
            image_shape (tuple[int, int]): the image shape needed for the image of the protuberance
                as seen from SDO.
            dx (float): the voxel resolution in km.
            d_theta (float): the theta angle resolution (as a function of the disk's perimeter) in
                degrees.

        Returns:
            tuple[np.ndarray, list[tuple[list[float], list[float]]]] | tuple[None, None]: the image
                of the protuberance as seen by SDO with the corresponding contour lines positions
                in polar coordinates.
        """

        # BORDER
        rho -= min(self.projection_borders.radial_distance) * 1000
        theta -= min(self.projection_borders.polar_angle)

        # BINNING
        rho //= dx
        theta //= d_theta

        # FILTERING duplicates
        polar = np.stack([rho, theta], axis=0)
        polar_indexes = np.unique(polar, axis=1).astype('int64')

        # KEEP inside the image
        rho_filter = (polar_indexes[0] > 0) & (polar_indexes[0] < image_shape[0])
        theta_filter = (polar_indexes[1] > 0) & (polar_indexes[1] < image_shape[1])
        full_filter = rho_filter & theta_filter

        # INDEXES final image
        rho = polar_indexes[0][full_filter]
        theta = polar_indexes[1][full_filter]

        # CHECK no data
        if len(rho) == 0: return None, None

        # CONTOURS get
        image = np.zeros(image_shape, dtype='float16')
        image[rho, theta] = 1
        lines = self.image_contour(image, dx, d_theta)
        return image, lines
    
    def image_contour(
            self,
            image: np.ndarray,
            dx: float,
            d_theta: float,
        ) -> list[tuple[list[float], list[float]]]:
        """ 
        To get the contours in the final plot coordinates of a mask given the corresponding
        information.

        Args:
            image (np.ndarray): the mask.
            dx (float): the vertical pixel length in km.
            d_theta (float): the horizontal pixel length in degrees.

        Returns:
            list[tuple[list[float], list[float]]]: the lines representing the mask contours.
        """

        # print(f'image contour dx is {dx} and d_theta {d_theta}', flush=True)

        # CONTOURS get
        lines = Plot.contours(image)

        # COORDs polar
        nw_lines = [None] * len(lines)
        for i, line in enumerate(lines):
            nw_lines[i] = ((
                [
                    min(self.projection_borders.radial_distance) + (value * dx) / 1e3
                    for value in line[0]
                ],
                [
                    min(self.projection_borders.polar_angle) + (value * d_theta)
                    for value in line[1]
                ],
            ))
        return nw_lines

    def sdo_image(self, filepath: str) -> PolarImageInfo:
        """ 
        To get the sdo image in polar coordinates and delimited by the final plot borders.
        Furthermore, needed information are also saved in the output, e.g. dx and d_theta for the
        created sdo image in polar coordinates.

        Args:
            filepath (str): the filepath to the corresponding sdo FITS file.

        Returns:
            PolarImageInfo: the polar SDO image information.
        """

        polar_image_info = CartesianToPolar.get_polar_image(
            filepath=filepath,
            borders=self.projection_borders,
            direction='clockwise',
            theta_offset=90,
            channel_axis=None,
        )
        return polar_image_info
    
    def sdo_image_treatment(self, image: np.ndarray) -> np.ndarray:
        """ 
        Pre-treatment for the sdo image for better visualisation of the regions of interest.

        Args:
            image (np.ndarray): the SDO image to be treated.

        Returns:
            np.ndarray: the treated SDO image.
        """
        
        # CLIP
        lower_cut = np.nanpercentile(image, 2)
        higher_cut = np.nanpercentile(image, 99.99)

        # SATURATION
        image[image < lower_cut] = lower_cut
        image[image > higher_cut] = higher_cut
        return np.log(image)
    
    def sdo_image_finder(self) -> dict[str, str]:
        """ 
        To find the SDO image given its header timestamp and a list of corresponding paths to the
        corresponding fits file.

        Returns:
            dict[str, str]: the timestamps as keys with the item being the SDO image filepath.
        """

        # SETUP
        filepath_end = '/S00000/image_lev1.fits'
        with open(os.path.join(self.paths['codes'], 'SDO_timestamps.txt'), 'r') as files:
            strings = files.read().splitlines()
        tuple_list = [s.split(" ; ") for s in strings]
    
        timestamp_to_path = {}
        for s in tuple_list:
            path, timestamp = s
            timestamp = timestamp.replace(':', '-')[:-6]

            # EXCEPTION weird cases...
            if timestamp == '2012-07-24T20-07': timestamp = '2012-07-24T20-06'
            if timestamp == '2012-07-24T20-20': timestamp = '2012-07-24T20-16'

            timestamp_to_path[timestamp] = path + filepath_end
        return timestamp_to_path



if __name__ == '__main__':
    OrthographicalProjection(
        filename='sig1e20_leg20_lim0_03.h5',
        with_feet=False,
        verbose=2,
        processes=4,
        polynomial_order=[4],
        plot_choices=[
            'polar', 'no duplicate', 'sdo mask', 'sdo image', 'envelope', 'test', 'fit',
            'fit envelope', 'integration',
        ],
        flush=True,
    )
