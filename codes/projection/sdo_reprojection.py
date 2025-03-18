"""
To get the projection of the polynomial polynomial inside the envelope. 
This is done to get the angle between the projection seen by SDO and the 3D polynomial
representation of the data.
"""

# IMPORTS
import os
import h5py
import time

# IMPORTS alias
import numpy as np
import multiprocessing as mp

# IMPORTS sub
from typing import Any
import matplotlib.pyplot as plt

# IMPORTS personal
from common import config, Decorators, Plot
from codes.projection.base_reprojection import BaseReprojection
from codes.projection.helpers.extract_envelope import ExtractEnvelope
from codes.projection.helpers.cartesian_to_polar import CartesianToPolar
from codes.projection.helpers.projection_dataclasses import *
from codes.data.polynomial_fit.polynomial_reprojection import ReprojectionProcessedPolynomial

# PLACEHOLDERs type annotation
QueueProxy = Any

# ? should I also create the image where there is no initial data when also plotting the integration?



class OrthographicalProjection(BaseReprojection):
    """
    Adds the protuberance voxels (with the corresponding polynomial fit) on SDO's image.
    Different choices are possible in what to plot in the final SDO image.
    Used to recreate recreate the plots made by Dr. Auchere in his 'the coronal Monsoon' paper.
    """

    @Decorators.running_time
    def __init__(
            self,
            processes: int | None = None,
            integration_time: list[int] = [24],
            filepath: str | None = None,
            with_feet: bool = False,
            polynomial_order: list[int] = [4],
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
            integration_time (list[int], optional): the integration time(s) used for the data to be
                reprojected. Defaults to 24.
            filepath (str | None, optional): the filepath of the HDF5 containing all the relevant
                3D data. When None, uses the config file. Defaults to None.
            with_feet (bool, optional): deciding to use the data with or without added feet.
                Defaults to False.
            polynomial_order (list[int], optional): the order(s) of the polynomial function(s) that
                represent the fitting of the integrated 3D volume. Defaults to [4].
            plot_choices (str | list[str], optional): the main choices that the user wants to be in
                the reprojection. The possible choices are:
                ['sdo image', 'no duplicates', 'envelope', 'polynomial', 'test data'].
                Defaults to [ 'sdo image', 'no duplicates', 'envelope', 'polynomial', 'test data'].
            with_fake_data (bool, optional): if the input data is the fusion HDF5 file.
                Defaults to False.
            verbose (int | None, optional): gives the verbosity in the outputted prints. The higher
                the value, the more prints. Starts at 0 for no prints. When None, uses the config
                file. Defaults to None.
            flush (bool | None, optional): used in the 'flush' kwarg of the print() class.
                Decides to force the print buffer to be emptied (i.e. forces the prints). When
                None, uses the config file. Defaults to None.
        """

        # CONFIG values
        if processes is None:
            self.processes: int = config.run.processes  #type:ignore
        else:
            self.processes = processes if processes > 1 else 1
        self.verbose: int = config.run.verbose if verbose is None else verbose  #type:ignore
        self.flush: bool = config.run.flush if flush is None else flush  #type:ignore
        self.in_local = True if 'Documents' in config.root_path else False  #type:ignore

        # PARENT
        super().__init__()

        # SERVER connection
        if self.in_local:
            from common.server_connection import SSHMirroredFilesystem
            self.connection = SSHMirroredFilesystem(verbose=self.verbose)

        # CONSTANTs
        self.solar_r = 6.96e5  # in km
        self.projection_borders = ImageBorders(
            radial_distance=(690, 870),  # in Mm
            polar_angle=(245, 295),  # in degrees
        )

        # ATTRIBUTEs
        self.with_fake_data = with_fake_data
        self.plot_choices = self.plot_choices_creation(plot_choices)        
        self.integration_time = integration_time
        self.multiprocessing = True if self.processes > 1 else False
        self.polynomial_order = sorted(polynomial_order)

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

    def filepath_setup(self, filepath: str | None) -> str:  #type:ignore
        """
        To setup the data filepath (using the config.yml file if filepath is None).

        Args:
            filepath (str | None): the filepath to the data.

        Returns:
            str: the real filepath to the data.
        """

        if filepath is None:
            if self.with_fake_data:
                filepath: str = config.path.data.fusion  #type:ignore
            else:
                filepath: str = config.path.data.real  #type:ignore
        return filepath

    def path_setup(self) -> dict[str, str]:
        """
        To get the paths to the needed directories and files.

        Returns:
            dict[str, str]: the needed paths.
        """

        # PATHs save
        paths = {
            'sdo': config.path.dir.data.sdo, #type:ignore
            'sdo times': config.path.data.sdo_timestamp, #type:ignore
            'save': os.path.join(
                config.path.dir.data.result.projection,  #type:ignore
                self.foldername,
            ),
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
        envelope_data = ExtractEnvelope.get(
            polynomial_order=6,
            number_of_points=int(1e5),
            borders=self.projection_borders,
            verbose=self.verbose,
        ) if self.plot_choices['envelope'] else None

        # COLOURS plot
        colour_generator = Plot.different_colours(omit=['white', 'red'])
        colours = [  # todo change this if I also want different integration times
            next(colour_generator)
            for _ in self.integration_time
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
            indexes: np.ndarray = H5PYFile[init_path + 'Time indexes'][...] #type:ignore

        if self.multiprocessing:
            # INFO multiprocessing
            data_len = len(indexes)
            nb_processes = min(self.processes, data_len)

            # SETUP multiprocessing
            processes: list[mp.Process] = [None] * nb_processes #type:ignore
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

        if self.in_local: self.connection.cleanup(verbose=self.verbose)
    
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
                time_integrated_paths = [
                    init_path + 'Time integrated/No duplicates' 
                    + f'/Time integration of {integration_time}.0 hours'
                    for integration_time in self.integration_time
                ]
                data_pointers.integration = [
                    self.get_cubes_information(
                        H5PYFile,
                        path,
                    )
                    for path in time_integrated_paths
                ]
                    
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
                    date=self.constants.dates[time_index].decode('utf8'), #type:ignore
                )
                projection_data = ProjectionData(ID=process)

                # SDO information
                filepath = self.sdo_timestamps[process_constants.date[:-3]]
                if self.in_local: filepath = self.get_file_from_server(filepath)
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
                        dx=self.constants.dx,
                        index=process,
                        name='no duplicates',
                        colour='orange',
                        sdo_pos=sdo_image_info.sdo_pos,
                    )

                if data_pointers.all_data is not None:
                    projection_data.all_data = self.format_cube(
                        data=data_pointers.all_data,
                        dx=self.constants.dx,
                        index=process,
                        name='all data',
                        colour='blue',
                        sdo_pos=sdo_image_info.sdo_pos,
                    )

                if data_pointers.line_of_sight is not None:
                    projection_data.line_of_sight = self.format_cube(
                        data=data_pointers.line_of_sight,
                        dx=self.constants.dx,
                        index=process,
                        name='line of sight',
                        colour='purple',
                        sdo_pos=sdo_image_info.sdo_pos,
                    )

                if data_pointers.integration is not None: 
                    projection_data.integration = [
                        self.format_cube(
                            data=integration,
                            dx=self.constants.dx,
                            index=process,
                            name=f'{self.integration_time[i]}hours integration',
                            colour=self.plot_kwargs['colours'][i],
                            sdo_pos=sdo_image_info.sdo_pos,
                        )
                        for i, integration in enumerate(data_pointers.integration)
                    ]
                    
                if data_pointers.fake_data is not None:
                    projection_data.fake_data = self.format_cube(
                        data=data_pointers.fake_data,
                        dx=self.constants.dx,
                        index=process,
                        name='fake data',
                        colour='black',
                        sdo_pos=sdo_image_info.sdo_pos,
                    )

                if data_pointers.test_cube is not None:
                    projection_data.test_cube = self.format_cube(
                        data=data_pointers.test_cube,
                        dx=self.constants.dx,
                        index=process,
                        name='test cube',
                        colour='yellow',
                        sdo_pos=sdo_image_info.sdo_pos,
                    )

                if self.plot_choices['fit']:
                    
                    # RESULTs formatting
                    index = -1
                    polynomials_info: list[FitWithEnvelopes] = (
                        [None] * len(self.polynomial_order) * len(self.integration_time)
                    )  #type: ignore

                    for poly_order in self.polynomial_order:

                        for i, integration_time in enumerate(self.integration_time):
                            index += 1
                            polynomial_instance = ReprojectionProcessedPolynomial(
                                colour=self.plot_kwargs['colours'][i],
                                filepath=self.filepath,
                                dx=self.constants.dx,
                                index=process,
                                sdo_pos=sdo_image_info.sdo_pos,
                                polynomial_order=poly_order,
                                integration_time=integration_time,
                                number_of_points=300,  # ? should I add it as an argument ?
                                with_fake_data=self.with_fake_data,
                                create_envelope=self.plot_choices['fit envelope'],
                            )
                            results = polynomial_instance.reprocessed_fit_n_envelopes()

                            # DATA save
                            polynomials_info[index] = results

                            # HDF5 close
                            polynomial_instance.close()

                    projection_data.fits_n_envelopes = polynomials_info
                    
                # CHILD CLASSes functionality
                self.plotting(process_constants, projection_data)
                self.create_fake_fits(process_constants, projection_data)
                
        if self.in_local: self.connection.close()

    def get_file_from_server(self, filepath: str, fail_count: int = 0) -> str:
        """
        To get the file from the server. If the file is not found, the function will try again
        until it reaches a certain number of tries.

        Args:
            filepath (str): the server filepath to the file.
            fail_count (int, optional): the number of times the function has failed to get the
                file from the server. Defaults to 0.

        Raises:
            Exception: if the function has failed to get the file from the server more than a
                certain number of times.

        Returns:
            str: the new local path to the file.
        """

        if fail_count > 10:
            raise Exception(
                f"\033[1;31mFailed to get the file '{filepath}' from the server. "
                "Killing process\033[0m"
            )

        try:
            filepath = self.connection.mirror(filepath, strip_level=1)
        except Exception as e:
            if self.verbose > 1: print(f'\033[1;31m{e}\nTrying again...\033[0m', flush=self.flush)
            fail_count += 1
            time.sleep(0.01)
            self.get_file_from_server(filepath=filepath, fail_count=fail_count)
        finally:
            return filepath

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
        dx = float(H5PYFile['dx'][...])  #type:ignore
        time_indexes: np.ndarray = H5PYFile[init_path + 'Time indexes'][...]  #type:ignore
        dates: np.ndarray = H5PYFile['Dates'][...]  #type:ignore

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
            cube_type: Literal['real'],
        ) -> CubePointer: ...
    
    @overload
    def get_cubes_information(
            self,
            H5PYFile: h5py.File,
            group_path: str,
            *,
            cube_type: Literal['test'],
        ) -> TestCubePointer: ...
    
    @overload
    def get_cubes_information(
            self,
            H5PYFile: h5py.File,
            group_path: str,
            *,
            cube_type: Literal['fake'],
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
        """
        To get the information about the data cubes.

        Args:
            H5PYFile (h5py.File): the HDF5 file containing the data.
            group_path (str): the path to the group containing the data.
            cube_type (str, optional): the type of the dataclass cube used. Defaults to 'real'.

        Returns:
            CubePointer | TestCubePointer | FakeCubePointer: the information about the data cubes.
        """

        # BORDERs
        xt_min = float(H5PYFile[group_path + '/xt_min'][...]) #type:ignore
        yt_min = float(H5PYFile[group_path + '/yt_min'][...]) #type:ignore
        zt_min = float(H5PYFile[group_path + '/zt_min'][...]) #type:ignore

        if cube_type == 'real':
            # FORMAT data
            cube_info = CubePointer(
                xt_min=xt_min,
                yt_min=yt_min,
                zt_min=zt_min,
                pointer=H5PYFile[group_path + '/coords'], #type:ignore
            )
        elif cube_type == 'test':
            # FORMAT data
            cube_info = TestCubePointer(
                xt_min=xt_min,
                yt_min=yt_min,
                zt_min=zt_min,
                pointer=H5PYFile[group_path + '/coords'], #type:ignore
            )
        else:
            # FAKE time indexes
            time_indexes: np.ndarray = H5PYFile['Fake/Time indexes'][...]  #type:ignore

            # FORMAT cube
            cube_info = FakeCubePointer(
                xt_min=xt_min,
                yt_min=yt_min,
                zt_min=zt_min,
                pointer=H5PYFile[group_path + '/coords'],  #type:ignore
                real_time_indexes=self.constants.time_indexes,
                fake_time_indexes=time_indexes,
            )
        return cube_info

    def sdo_image(self, filepath: str, colour: str) -> PolarImageInfo:
        """
        To get the sdo image in polar coordinates and delimited by the final plot borders.
        Furthermore, needed information are also saved in the output, e.g. dx and d_theta for the
        created sdo image in polar coordinates.

        Args:
            filepath (str): the filepath to the corresponding sdo FITS file.
            colour (str): the colour of the image contours in the final plot.

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
        nw_lines: list[tuple[list[float], list[float]]] = [None] * len(lines) #type:ignore
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
            for integration in projection_data.integration:
                # PLOT contours time integrated
                self.plot_contours(
                    projection=integration,
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
        
        if projection_data.fits_n_envelopes is not None:

            for fit_n_envelope in projection_data.fits_n_envelopes:
                
                # PLOT
                sc = plt.scatter(
                    fit_n_envelope.fit_polar_theta,
                    fit_n_envelope.fit_polar_r / 10**3,
                    label=(
                        f"{fit_n_envelope.integration_time}h - "
                        f"{fit_n_envelope.fit_order}th order fit"
                    ),
                    c=np.rad2deg(fit_n_envelope.fit_angles),
                    **self.plot_kwargs['fit'],
                )

                # ENVELOPE fit
                if fit_n_envelope.envelopes is not None:

                    # PLOT envelope
                    for label, new_envelope in enumerate(fit_n_envelope.envelopes):
                        plt.plot(
                            new_envelope.polar_theta,
                            new_envelope.polar_r / 1e3,
                            label=(
                                f"{fit_n_envelope.integration_time}h - "
                                f"{fit_n_envelope.fit_order}th order fit envelope"
                                f'({new_envelope.order}th order polynomial)'
                            ) if label==0 else None,
                            color=fit_n_envelope.colour,
                            **self.plot_kwargs['fit envelope'],
                        )

            cbar = plt.colorbar(sc)
            cbar.set_label(r'$\theta$ (degrees)')

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

        # PLOT save
        plot_name = f"reprojection_{process_constants.date}.png"
        plt.savefig(os.path.join(self.paths['save'], plot_name), dpi=500)
        plt.close()

        if self.verbose > 1: 
            print(
                f'SAVED - nb {process_constants.time_index:03d} - {plot_name}',
                flush=self.flush,
            )

    def plot_contours(
            self,
            projection: ProjectedCube,
            dx: float,
            d_theta: float, 
            image_shape: tuple[int, int],
        ) -> None:
        """
        To plot the contours of the image for the protuberance as seen from SDO's pov.

        Args:
            projection (ProjectedCube): the cube containing the information for the protuberance
                as seen from SDO's pov.
            dx (float): the voxel resolution in km.
            d_theta (float): the theta angle resolution (as a function of the disk's perimeter) in
                degrees.
            image_shape (tuple[int, int]): the image shape needed for the image of the protuberance
                as seen from SDO.
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
        integration_time=[12, 18],
        polynomial_order=[4],
        plot_choices=[
            'integration',
            'fit', 'fit envelope',
            'sdo image',
        ],
        with_fake_data=True,
    )
