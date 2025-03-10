"""
To get the projection of the polynomial polynomial inside the envelope. 
This is done to get the angle between the projection seen by SDO and the 3D polynomial
representation of the data.
"""

# IMPORTS
import os
import h5py
import yaml

# IMPORTS alias
import numpy as np
import multiprocessing as mp

# IMPORTS sub
from typing import Any
import matplotlib.pyplot as plt

# IMPORTS personal
from common import config, Decorators, Plot
from codes.data.get_polynomial import GetCartesianProcessedPolynomial
from codes.projection.extract_envelope import ExtractEnvelope, CreateFitEnvelope
from codes.projection.cartesian_to_polar import CartesianToPolar
from codes.projection.projection_dataclasses import *

# PLACEHOLDERs type annotation
QueueProxy = Any



class OrthographicalProjection:
    """
    Adds the protuberance voxels (with the corresponding polynomial fit) on SDO's image.
    Different choices are possible in what to plot in the final SDO image.
    Used to recreate recreate the plots made by Dr. Auchere in his 'the coronal Monsoon' paper.
    """

    @Decorators.running_time
    def __init__(
            self,
            processes: int | None = None,
            integration_time_hours: int = 24,
            filepath: str | None = None,
            with_feet: bool = False,
            polynomial_order: int | list[int] = [4],
            plot_choices: str | list[str] = [
                'sdo image', 'no duplicates', 'envelope', 'polynomial', 'test data',
            ], 
            with_fake_data: bool = False,
            verbose: int | None = None,
            flush: bool | None = None,
        ) -> None:
        """
        Re-projection of the computed 3D volume to recreate a 2D image of what is seen from SDO's
        POV.
        This is done to recreate the analysis of Dr. Auchere's paper; 'The coronal Monsoon'.

        Args:
            processes (int | None, optional): the number of parallel processes used in the
                multiprocessing. When None, uses the config file. Defaults to None.
            integration_time_hours (int, optional): the integration time used for the data to be
                reprojected. Defaults to 24.
            filepath (str | None, optional): the filepath of the HDF5 containing all the relevant
                3D data. When None, uses the config file. Defaults to None.
            with_feet (bool, optional): deciding to use the data with or without added feet.
                Defaults to False.
            polynomial_order (int | list[int], optional): the order(s) of the polynomial
                function(s) that represent the fitting of the integrated 3D volume.
                Defaults to [4].
            plot_choices (str | list[str], optional): the main choices that the user wants to be in
                the reprojection. The possible choices are:
                ['sdo image', 'no duplicates', 'envelope', 'polynomial', 'test data'].
                Defaults to [ 'sdo image', 'no duplicates', 'envelope', 'polynomial', 'test data'].
            with_fake_data (bool, optional): if the input data is the fusion HDF5 file.
                Defaults to False.
            verbose (int | None, optional): gives the verbosity in the outputted prints. The higher
                the value, the more prints. Starts at 0 for no prints. When None, uses the config
                file. Defaults to None.
            flush (bool | None, optional): used in the 'flush' kwarg of the print() class.\
                Decides to force the print buffer to be emptied (i.e. forces the prints). When
                None, uses the config file. Defaults to None.
        """

        # CONFIG values
        if processes is None:
            self.processes: int = config.run.processes
        else:
            self.processes = processes if processes > 1 else 1
        self.verbose: int = config.run.verbose if verbose is None else verbose
        self.flush: bool = config.run.flush if flush is None else flush
        self.in_local = True if 'Documents' in config.root_path else False

        # SERVER connection
        if self.in_local:
            from common.server_connection import SSHMirroredFilesystem
            self.SSHMirroredFilesystem = SSHMirroredFilesystem

        # CONSTANTs
        self.solar_r = 6.96e5  # in km
        self.projection_borders = ImageBorders(
            radial_distance=(690, 870),  # in Mm
            polar_angle=(245, 295),  # in degrees
        )

        # ATTRIBUTEs
        self.with_fake_data = with_fake_data
        self.plot_choices = self.plot_choices_creation(plot_choices)        
        self.integration_time = integration_time_hours
        self.multiprocessing = True if self.processes > 1 else False
        if isinstance(polynomial_order, list):
            self.polynomial_order = sorted(polynomial_order)
        else:
            self.polynomial_order = [polynomial_order]

        # PATHs setup
        self.feet = ' with feet' if with_feet else ''
        self.filepath = self.filepath_setup(filepath)
        self.foldername = (
            os.path.basename(self.filepath).split('.')[0] + ''.join(self.feet.split(' '))
        )
        self.paths = self.path_setup()  # path setup
        self.sdo_timestamps = self.sdo_image_finder()  # sdo image timestamps

        # GLOBAL data
        self.Auchere_envelope, self.plot_kwargs = self.global_information()

    def filepath_setup(self, filepath: str | None) -> str:
        """
        To setup the data filepath (using the config.yml file if filepath is None).

        Args:
            filepath (str | None): the filepath to the data.

        Returns:
            str: the real filepath to the data.
        """

        if filepath is None:
            filepath = config.path.data.fusion if self.with_fake_data else config.path.data.real
        return filepath

    def path_setup(self) -> dict[str, str]:
        """
        To get the paths to the needed directories and files.

        Returns:
            dict[str, str]: the needed paths.
        """

        # PATHs save
        paths = {
            'sdo': config.path.dir.data.sdo,
            'sdo times': config.path.data.sdo_timestamp,
            'save': os.path.join(config.path.dir.data.result.projection, self.foldername),
        }

        # PATHs update
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
        possibilities = [  # ? does the cartesian option still work?
            'integration', 'no duplicates', 'sdo image', 'sdo mask', 'test cube', 'fake data',
            'envelope', 'fit', 'fit envelope', 'test data', 'line of sight', 'all data'
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
                raise ValueError(f"plot_choices argument '{key}' not recognised.") 
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
        envelope_data = ExtractEnvelope.get(  # todo look at the function to see if any changes are needed
            polynomial_order=6,
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
    def run(self) -> None:
        """
        Setups the multiprocessing for the whole code.
        """

        # PATH setup
        init_path = 'Real/' if self.with_fake_data else ''

        # STATS data
        with h5py.File(self.filepath, 'r') as H5PYFile:
            indexes: np.ndarray = H5PYFile[init_path + 'Time indexes'][...]

        if self.multiprocessing:
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
        else:
            self.data_setup(index_list=indexes)

        if self.in_local: self.SSHMirroredFilesystem.cleanup(verbose=self.verbose)
    
    def data_setup(
            self,
            input_queue: QueueProxy | None = None,
            index_list: np.ndarray | None = None,
        ) -> None:
        """
        Open the HDF5 file and does the processing and final plotting for each cube.
        A while loop is used to decide which data section needs to be processed.

        Args:
            input_queue (mp.queues.Queue | None): contains the data cube indexes that still need
                processing.
            index_list (np.ndarray | None): the list of indexes of the data cubes that still need
                processing. Only used when not multiprocessing.
        """

        # CONNECTION to server
        if self.in_local: self.connection = self.SSHMirroredFilesystem(verbose=self.verbose)

        # INIT no multiprocessing
        process_id = 0

        # DATA open
        with h5py.File(self.filepath, 'r') as H5PYFile:
            # PATH setup
            init_path = 'Real/' if self.with_fake_data else ''

            # GLOBAL constants
            self.constants = self.get_global_constants(H5PYFile, init_path)
            data_pointers = CubesPointers()
            
            # DATA pointers
            if self.plot_choices['no duplicates']:
                filtered_path = init_path + 'Filtered/No duplicates'
                data_pointers.no_duplicates = self.get_cubes_information(H5PYFile, filtered_path)
            if self.plot_choices['all data']:
                filtered_path = init_path + 'Filtered/All data'
                data_pointers.all_data = self.get_cubes_information(H5PYFile, filtered_path)
            if self.plot_choices['integration']:
                time_integrated_path = (
                    init_path + 'Time integrated/No duplicates' 
                    + f'/Time integration of {self.integration_time}.0 hours' 
                )
                data_pointers.integration = self.get_cubes_information(
                    H5PYFile,
                    time_integrated_path,
                )
            if self.plot_choices['line of sight']:
                line_of_sight_path = init_path + 'Filtered/SDO line of sight'
                data_pointers.line_of_sight = self.get_cubes_information(
                    H5PYFile,
                    line_of_sight_path,
                )
                # ? add the stereo line of sight just to see the intersection ?
            if self.plot_choices['fake data']:
                fake_path = 'Fake/Filtered/All data'
                data_pointers.fake_data = self.get_cubes_information(
                    H5PYFile=H5PYFile,
                    group_path=fake_path,
                    cube_type='fake',
                )

            if self.plot_choices['test cube']:
                test_path = 'Test data/Sun surface'
                data_pointers.test_cube = self.get_cubes_information(
                    H5PYFile=H5PYFile,
                    group_path=test_path,
                    cube_type='test',
                )

            while True:
                # INFO process 
                if input_queue is not None:
                    process: int | None = input_queue.get()
                else:
                    process = index_list[process_id]
                    process_id += 1
                    process = None if process_id > len(index_list) else process
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
                filepath = self.sdo_timestamps[process_constants.date[:-3]]
                if self.in_local: filepath = self.connection.mirror(filepath, strip_level=1)
                sdo_image_info = self.sdo_image(filepath, colour='') 

                if self.plot_choices['sdo image']: projection_data.sdo_image = sdo_image_info

                if self.plot_choices['sdo mask']:
                    # SDO mask
                    filepath = os.path.join(
                        self.paths['sdo'],
                        f"AIA_fullhead_{process_constants.time_index:03d}.fits.gz",
                    )
                    polar_mask_info = self.sdo_image(filepath, colour='green')
                    image = np.zeros(polar_mask_info.image.shape)
                    image[polar_mask_info.image > 0] = 1
                    polar_mask_info.image = image

                    # DATA formatting
                    projection_data.sdo_mask = polar_mask_info

                if data_pointers.no_duplicates is not None:
                    projection_data.no_duplicates = self.format_cube(
                        data=data_pointers.no_duplicates,
                        index=process,
                        name='no duplicates',
                        colour='orange',
                        sdo_pos=sdo_image_info.sdo_pos,
                    )

                if data_pointers.all_data is not None:
                    projection_data.all_data = self.format_cube(
                        data=data_pointers.all_data,
                        index=process,
                        name='all data',
                        colour='blue',
                        sdo_pos=sdo_image_info.sdo_pos,
                    )

                if data_pointers.line_of_sight is not None:
                    projection_data.line_of_sight = self.format_cube(
                        data=data_pointers.line_of_sight,
                        index=process,
                        name='line of sight',
                        colour='purple',
                        sdo_pos=sdo_image_info.sdo_pos,
                    )

                if data_pointers.integration is not None: 
                    projection_data.integration = self.format_cube(
                        data=data_pointers.integration,
                        index=process,
                        name='integration',
                        colour='red',
                        sdo_pos=sdo_image_info.sdo_pos,
                    )
                
                if data_pointers.fake_data is not None:
                    projection_data.fake_data = self.format_cube(
                        data=data_pointers.fake_data,
                        index=process,
                        name='fake data',
                        colour='black',
                        sdo_pos=sdo_image_info.sdo_pos,
                    )
                    rho, theta = projection_data.fake_data

                if data_pointers.test_cube is not None:
                    projection_data.test_cube = self.format_cube(
                        data=data_pointers.test_cube,
                        index=process,
                        name='test cube',
                        colour='yellow',
                        sdo_pos=sdo_image_info.sdo_pos,
                    )
                    rho, theta = projection_data.test_cube

                if self.plot_choices['fit']:

                    polynomials_info: list[PolynomialInformation] = (
                        [None] * len(self.polynomial_order)
                    )#type: ignore

                    for i, poly_order in enumerate(self.polynomial_order):
                        polynomial_instance = GetCartesianProcessedPolynomial(
                            filepath=self.filepath,
                            polynomial_order=poly_order,
                            integration_time=self.integration_time,
                            number_of_points=250,
                            dx=self.constants.dx,
                            with_fake_data=self.with_fake_data,
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
                    
                # CHILD CLASSes functionality
                self.plotting(process_constants, projection_data)
                self.create_fake_fits(process_constants, projection_data)
                
        if self.in_local: self.connection.close()

    def format_cube(
            self,
            data: CubePointer | TestCubePointer | FakeCubePointer,
            index: int,
            name: str,
            colour: str,
            sdo_pos: np.ndarray,
        ) -> ProjectedCube:
        """
        To format the cube data for the projection.

        Args:
            data (CubePointer | TestCubePointer | FakeCubePointer): the data cube to be formatted.
            index (int): the index of the corresponding real data cube.
            colour (str): the colour of the data cube for the plot.
            sdo_pos (np.ndarray): the position of the SDO satellite.

        Returns:
            ProjectedCube: the formatted and reprojected data cube.
        """

        # CUBE formatting
        cube = CubeInformation(
            xt_min=data.xt_min,
            yt_min=data.yt_min,
            zt_min=data.zt_min,
            coords=data[index],
        )
        cube = self.cartesian_pos(cube)
        cube = self.get_polar_image(self.matrix_rotation(data=cube.coords, sdo_pos=sdo_pos))

        # PROJECTION formatting
        return ProjectedCube(data=cube, name=name, colour=colour)

    def get_global_constants(self, H5PYFile: h5py.File, init_path: str) -> GlobalConstants:
        """
        To get the global constants of the data.

        Args:
            H5PYFile (h5py.File): the HDF5 file containing the data.
            init_path (str): the beginning value of the path (depends if the HDF5 file also has
                fake and/or test data).

        Returns:
            GlobalConstants: the global constants of the data.
        """

        # DATA open
        dx = float(H5PYFile['dx'][...])
        time_indexes: np.ndarray = H5PYFile[init_path + 'Time indexes'][...]
        dates: np.ndarray = H5PYFile['Dates'][...]

        # FORMAT data
        constants = GlobalConstants(
            dx=dx,
            solar_r=self.solar_r,
            time_indexes=time_indexes,
            dates=dates,
        )
        return constants
    
    @overload
    def get_cubes_information(
            self,
            H5PYFile: h5py.File,
            group_path: str,
            *,
            cube_type: Literal['real'] = ...,
        ) -> CubePointer: ...
    
    @overload
    def get_cubes_information(
            self,
            H5PYFile: h5py.File,
            group_path: str,
            *,
            cube_type: Literal['test'] = ...,
        ) -> TestCubePointer: ...
    
    @overload
    def get_cubes_information(
            self,
            H5PYFile: h5py.File,
            group_path: str,
            *,
            cube_type: Literal['fake'] = ...,
        ) -> FakeCubePointer: ...
    
    @overload # fallback
    def get_cubes_information(
            self,
            H5PYFile: h5py.File,
            group_path: str,
            cube_type: str = 'real',
        ) -> CubePointer | TestCubePointer | FakeCubePointer: ...

    def get_cubes_information(
            self,
            H5PYFile: h5py.File,
            group_path: str,
            cube_type: str = 'real',
        ) -> CubePointer | TestCubePointer | FakeCubePointer:
        """ # todo update docstring
        To get the information about the data cubes.

        Args:
            H5PYFile (h5py.File): the HDF5 file containing the data.
            group_path (str): the path to the group containing the data.
            test_cube (bool, optional): if the data is test data. Only used for the IDE to infer
                the return type. Defaults to False.

        Returns:
            CubePointer | TestCubePointer: the information about the data cubes.
        """

        # BORDERs
        xt_min = float(H5PYFile[group_path + '/xt_min'][...])
        yt_min = float(H5PYFile[group_path + '/yt_min'][...])
        zt_min = float(H5PYFile[group_path + '/zt_min'][...])

        if cube_type == 'real':
            # FORMAT data
            cube_info = CubePointer(
                xt_min=xt_min,
                yt_min=yt_min,
                zt_min=zt_min,
                pointer=H5PYFile[group_path + '/coords'],
            )
        elif cube_type == 'test':
            # FORMAT data
            cube_info = TestCubePointer(
                xt_min=xt_min,
                yt_min=yt_min,
                zt_min=zt_min,
                pointer=H5PYFile[group_path + '/coords'],
            )
        else:
            # FAKE time indexes
            time_indexes: np.ndarray = H5PYFile['Fake/Time indexes'][...]

            # FORMAT cube
            cube_info = FakeCubePointer(
                xt_min=xt_min,
                yt_min=yt_min,
                zt_min=zt_min,
                pointer=H5PYFile[group_path + '/coords'],
                real_time_indexes=self.constants.time_indexes,
                fake_time_indexes=time_indexes,
            )
        return cube_info
    
    def get_fake_cube_information(self, H5PYFile: h5py.File, group_path: str) -> FakeCubePointer:
        
        # BORDERs
        xt_min = float(H5PYFile[group_path + '/xt_min'][...])
        yt_min = float(H5PYFile[group_path + '/yt_min'][...])
        zt_min = float(H5PYFile[group_path + '/zt_min'][...])

        # FAKE time indexes
        time_indexes: np.ndarray = H5PYFile['Fake/Time indexes'][...]

        # FORMAT data
        cube_info = FakeCubePointer(
            xt_min=xt_min,
            yt_min=yt_min,
            zt_min=zt_min,
            pointer=H5PYFile[group_path + '/coords'],
            real_time_indexes=self.constants.time_indexes,
            fake_time_indexes=time_indexes,
        )
        return cube_info


    def cartesian_pos(self, data: CubeInformation) -> CubeInformation:
        """
        To calculate the heliographic cartesian positions given a ndarray of index positions.

        Args:
            data (CubeInformation): the heliographic cartesian positions of the protuberance.

        Returns:
            CubeInformation: the heliographic cartesian positions.
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
        rho_polar = np.tan(rho_polar) / z_norm  # todo need to re-understand why I put this here
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

        # todo add an explanation in the equation .md file.

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

    def sdo_image(self, filepath: str, colour: str) -> PolarImageInfo:
        """ #todo update docstring
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
            colour=colour,
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
        with open(self.paths['sdo times'], 'r') as files:
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

    def plotting(self, *args, **kwargs) -> None:
        """
        Placeholder for a child class to plot the data.
        """

        pass

    def create_fake_fits(self , *args, **kwargs) -> None:
        """
        Placeholder for a child class to create fake fits files.
        """

        pass

    def cube_contour(
            self,
            rho: np.ndarray,
            theta: np.ndarray,
            image_shape: tuple[int, int],
            dx: float,
            d_theta: float,
        ) -> tuple[np.ndarray, list[tuple[list[float], list[float]]] | None]:
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
            tuple[np.ndarray, list[tuple[list[float], list[float]]] | None]: the image of the
                protuberance as seen from SDO and the contours of the protuberance.
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
        ) -> list[tuple[list[float], list[float]]] | None:
        """ 
        To get the contours in the final plot coordinates of a mask given the corresponding
        information.

        Args:
            image (np.ndarray): the mask.
            dx (float): the vertical pixel length in km.
            d_theta (float): the horizontal pixel length in degrees.

        Returns:
            list[tuple[list[float], list[float]]] | None: the lines representing the mask contours.
                None if no data in the image.
        """

        # CONTOURS get
        lines = Plot.contours(image)

        # CHECK no data
        if lines == []: return None

        # COORDs polar
        nw_lines: list[tuple[list[float], list[float]]] = [None] * len(lines)
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


class Plotting(OrthographicalProjection):
    """
    To plot the SDO's point of view image.
    """

    def __init__(self, *args, **kwargs) -> None:
        """
        To plot the SDO's point of view image.
        """

        # PARENT CLASS initialisation
        super().__init__(*args, **kwargs)

        # RUN code
        self.run()

    @Decorators.running_time
    def plotting(
            self,
            process_constants: ProcessConstants,
            projection_data: ProjectionData,
        ) -> None:
        """
        To plot the SDO's point of view image.
        What is plotted is dependent on the chosen choices when running the class.

        Args:
            process_constants (ProcessConstants): the constants for each cube..
            projection_data (ProjectionData): the data for each cube.
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
            if lines is not None:
                line = lines[0]
                plt.plot(
                    line[1],
                    line[0],
                    color=projection_data.sdo_mask.colour,
                    label='SDO mask contours',
                    **self.plot_kwargs['contour'],
                )
                for line in lines[1:]:
                    plt.plot(
                        line[1],
                        line[0],
                        color=projection_data.sdo_mask.colour,
                        **self.plot_kwargs['contour'],
                    )

        if projection_data.sdo_image is not None:
            plt.imshow(
                self.sdo_image_treatment(projection_data.sdo_image.image),
                **self.plot_kwargs['image'],
            )

        if projection_data.integration is not None:
            # PLOT contours time integrated
            self.plot_contours(
                projection=projection_data.integration,
                d_theta=self.constants.d_theta,
                dx=self.constants.dx,
                image_shape=image_shape,
            )
        
        if projection_data.no_duplicates is not None:
            # PLOT contours no duplicates
            self.plot_contours(
                projection=projection_data.no_duplicates,
                d_theta=self.constants.d_theta,
                dx=self.constants.dx,
                image_shape=image_shape,
            )

        if projection_data.all_data is not None:
            # PLOT contours all data
            self.plot_contours(
                projection=projection_data.all_data,
                d_theta=self.constants.d_theta,
                dx=self.constants.dx,
                image_shape=image_shape,
            )

        if projection_data.line_of_sight is not None:
            # PLOT contours line of sight
            self.plot_contours(
                projection=projection_data.line_of_sight,
                d_theta=self.constants.d_theta,
                dx=self.constants.dx,
                image_shape=image_shape,
            )
        
        # ! haven't checked this if statement yet
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

        # PLOT fake data
        if projection_data.fake_data is not None:
            # PLOT contours fake data
            self.plot_contours(
                projection=projection_data.fake_data,
                d_theta=self.constants.d_theta,
                dx=self.constants.dx,
                image_shape=image_shape,
            )

        # PLOT test cube
        if projection_data.test_cube is not None:
            # PLOT contours test cube
            self.plot_contours(
                projection=projection_data.test_cube,
                d_theta=self.constants.d_theta,
                dx=self.constants.dx,
                image_shape=image_shape,
            )

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
            print(
                f'SAVED - nb {process_constants.time_index:03d} - filename: {plot_name}',
                flush=self.flush,
            )

    def plot_contours(
            self,
            projection: ProjectedCube,
            d_theta: float, 
            dx: float,
            image_shape: tuple[int, int],
        ) -> None:
        """ # todo update docstring
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

        # POLAR coordinates
        rho, theta = projection
        color = projection.colour

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
                label=projection.name + ' contour',
                **self.plot_kwargs['contour'],
            )
            for line in lines:
                plt.plot(line[1], line[0], color=color, **self.plot_kwargs['contour'])



if __name__ == '__main__':
    Plotting(
        polynomial_order=[4],
        plot_choices=[
            'no duplicates', 'sdo image', 'sdo mask', 'integration', 'line of sight', 'fake data',
            'fit', 'test cube',
        ],
        with_fake_data=True,
    )
