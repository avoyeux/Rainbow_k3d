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
import matplotlib.pyplot as plt

# IMPORTS personal
from common import config, Decorators, Plot
from codes.projection.format_data import *
from codes.projection.envelope_distance import EnvelopeDistanceAnnotation
from codes.projection.helpers.warp_sdo_image import WarpSdoImage
from codes.projection.helpers.extract_envelope import ExtractEnvelope
from codes.projection.helpers.base_reprojection import BaseReprojection
from codes.projection.helpers.cartesian_to_polar import CartesianToPolar
from codes.data.polynomial_fit.polynomial_reprojection import ReprojectionProcessedPolynomial

# IMPORTs type annotation
from typing import Any, cast, overload, Literal
from matplotlib.collections import PathCollection

# PLACEHOLDERs type annotation
QueueProxy = Any

# todo add the 'integration' of the warp images
# todo 'integration' of warp images needs to be able to work on multiple datasets at the same time



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
                ['full integration no duplicates', 'full integration all data', 'integration',
                'no duplicates', 'sdo image', 'sdo mask', 'test cube', 'fake data', 'envelope',
                'fit', 'fit envelope', 'test data', 'line of sight', 'all data', 'warp']
                ['sdo image', 'no duplicates', 'envelope', 'polynomial', 'test data'].
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
            self.processes: int = config.run.processes
        else:
            self.processes = processes if processes > 1 else 1
        self.verbose: int = config.run.verbose if verbose is None else verbose
        self.flush: bool = config.run.flush if flush is None else flush
        self.in_local = True if 'Documents' in config.root_path else False

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
            os.path.basename(self.filepath).split('.')[0] + ''.join(self.feet.split(' ')) + '_test'
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
                filepath: str = config.path.data.fusion
            else:
                filepath: str = config.path.data.real
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
            'save': os.path.join(
                config.path.dir.data.result.projection,
                self.foldername,
            ),
        }

        # PATHs update
        paths['save warped'] = paths['save'] + '_warped'
        os.makedirs(paths['save'], exist_ok=True)
        if self.plot_choices['warp']:  os.makedirs(paths['save warped'], exist_ok=True)
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
            'full integration no duplicates', 'full integration all data',
            'integration', 'no duplicates', 'sdo image', 'sdo mask', 'test cube', 'fake data',
            'envelope', 'fit', 'fit envelope', 'test data', 'line of sight', 'all data', 'warp',
            'all sdo images',
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
                raise ValueError(
                    f"\033[1;31mPlot_choices argument '{key}' not recognised. "
                    f"Choices are ['{'\', \''.join(possibilities)}']\033[0m"   
                ) 
        return plot_choices_kwargs

    def global_information(
            self,
        ) -> tuple[
            EnvelopeInformation | None,
            dict[str, list[str] | dict[str, int | float | tuple[int, ...]]],
        ]:
        """
        Contains the default choices made for the plotting options (e.g. the opacity, linestyle).

        Returns:
            tuple[
                EnvelopeInformation | None,
                dict[str, list[str] | dict[str, int | float | tuple[int, ...]]]
            ]: Dr. Auchere's envelope data and the default plotting information.
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
            # DATA index conversion
            self.index_to_cube = self.process_index_to_cube_index(H5PYFile, init_path)

        if self.multiprocessing:
            # INFO multiprocessing
            data_len = len(self.index_to_cube.keys())
            nb_processes = min(self.processes, data_len)

            # SETUP multiprocessing
            processes: list[mp.Process] = cast(list[mp.Process], [None] * nb_processes)
            manager = mp.Manager()
            input_queue = manager.Queue()
            output_queue = manager.Queue()
            for i in range(data_len): input_queue.put(i)  # todo change this to a value proxy
            for _ in range(nb_processes): input_queue.put(None)

            # RUN multiprocessing
            for i in range(nb_processes):
                p = mp.Process(
                    target=self.data_setup,
                    kwargs={'input_queue': input_queue, 'output_queue': output_queue},
                )
                p.start()
                processes[i] = p
            for p in processes: p.join()

            # GET warped data
            warped_data_list: list[np.ndarray] = cast(list[np.ndarray], [None] * data_len)
            while not output_queue.empty():
                identifier, warped = output_queue.get()
                warped_data_list[identifier] = warped
            warped_data = np.stack(warped_data_list, axis=0)
            self.process_warped_data(warped_data)  # ! this only works for one set of warped data
        else:
            self.data_setup(index_list=indexes)  # ! this won't work for now

        if self.in_local: self.connection.cleanup(verbose=self.verbose)

    def process_warped_data(self, warped_data: np.ndarray) -> None:
        """
        To plot the final figure of Dr. Auchere's paper given a set of warped images.

        Args:
            warped_data (np.ndarray): the warped images for which the plot needs to be made.
        """

        # ROW MEDIAN processing
        median_rows = np.median(warped_data.T, axis=1)
        
        # PLOT
        plt.figure(figsize=(18, 5))
        plt.imshow(median_rows, cmap='gray', origin='lower')
        plt.title('Median rows of the warped data')
        plt.xlabel('Time')
        plt.ylabel('Radial distance')
        plt.savefig(os.path.join(config.path.dir.data.temp, 'median_rows.png'), dpi=500)
        plt.close()

    def process_index_to_cube_index(
            self,
            H5PYFile: h5py.File,
            init_path: str,
        ) -> dict[int, int] | dict[int, int | None]:
        """
        To get the corresponding data cube index given the process index.

        Args:
            H5PYFile (h5py.File): the HDF5 file containing the data.
            init_path (str): the initial path of the data (depends if the HDF5 file also has fake
                and/or test data).

        Returns:
            dict[int, int] | dict[int, int | None]: the dictionary containing the process index as
                keys and the corresponding data cube index as values.
        """

        # GET DATA
        dates: np.ndarray = cast(h5py.Dataset, H5PYFile['Dates'])[...]
        time_indexes: np.ndarray = cast(h5py.Dataset, H5PYFile[init_path + 'Time indexes'])[...]

        if self.plot_choices['all sdo images']:

            # ALL INDEXEs
            value = -1
            index_to_cube = {
                index: (value := value + 1) if index in time_indexes else None
                for index in range(dates.shape[0])
            }
        else:
            # TIME INDEXEs
            index_to_cube = {value: index for index, value in enumerate(time_indexes)}
        result: dict[int, int] | dict[int, int | None] = index_to_cube
        return result        

    def data_setup(
            self,
            input_queue: QueueProxy | None = None,
            index_list: np.ndarray | None = None,
            output_queue: QueueProxy | None = None,
        ) -> None:
        """  # todo update docstring
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

        # WARP KWARGS
        warp_kwargs = {
             'extent': cast(tuple[int, ...], self.plot_kwargs['image']['extent']),  #type:ignore
             'image_shape': (1280, 1280),
             'pixel_interpolation_order': 3,
             'nb_of_points': 300,
        }

        # DATA open
        with h5py.File(self.filepath, 'r') as H5PYFile:
            # PATH setup
            init_path = 'Real/' if self.with_fake_data else ''

            # GLOBAL constants
            self.constants = self.get_global_constants(H5PYFile, init_path)
            data_pointers = CubesPointers()
            
            # DATA pointers
            if self.plot_choices['all data']:
                filtered_path = init_path + 'Filtered/All data'
                data_pointers.all_data = self.get_cubes_information(
                    H5PYFile=H5PYFile,
                    group_path=filtered_path,
                    cube_name='All data',
                    cube_type='real',
                    interpolate=False,
                )
            
            if self.plot_choices['no duplicates']:
                filtered_path = init_path + 'Filtered/No duplicates'
                data_pointers.no_duplicates = self.get_cubes_information(
                    H5PYFile=H5PYFile,
                    group_path=filtered_path,
                    cube_name='No duplicates',
                    cube_type='real',
                    interpolate=False,
                )
            
            if self.plot_choices['full integration all data']:
                filtered_path = init_path + 'Time integrated/All data/Full integration'
                data_pointers.full_integration_all_data = self.get_cubes_information(
                    H5PYFile=H5PYFile,
                    group_path=filtered_path,
                    cube_name='All data (full)',
                    cube_type='unique',
                    integration_time='(full)',
                    interpolate=True,
                )
            
            if self.plot_choices['full integration no duplicates']:
                filtered_path = init_path + 'Time integrated/No duplicates/Full integration'
                data_pointers.full_integration_no_duplicates = self.get_cubes_information(
                    H5PYFile=H5PYFile,
                    group_path=filtered_path,
                    cube_name='No duplicates (full)',
                    cube_type='unique',
                    integration_time='(full)',
                    interpolate=True,
                )
            
            if self.plot_choices['integration']:

                integrations: list[DataPointer] = cast(
                    list[DataPointer],
                    [None] * len(self.integration_time),
                )
                for i, integration_time in enumerate(self.integration_time):

                    path = (
                        init_path + 'Time integrated/No duplicates'
                        + f'/Time integration of {integration_time}.0 hours'
                    )
                    integrations[i] = self.get_cubes_information(
                        H5PYFile=H5PYFile,
                        group_path=path,
                        cube_name=f'No duplicates integration ({integration_time} hours)',
                        cube_type='real',
                        integration_time=integration_time,
                        interpolate=True,
                    )
                
                # INTEGRATION save
                data_pointers.integration = integrations
                    
            if self.plot_choices['line of sight']:
                line_of_sight_path = init_path + 'Filtered/SDO line of sight'
                data_pointers.line_of_sight = self.get_cubes_information(
                    H5PYFile=H5PYFile,
                    group_path=line_of_sight_path,
                    cube_name='SDO line of sight',
                    cube_type='real',
                    interpolate=False,
                )
            
            if self.plot_choices['fake data']:
                fake_path = 'Fake/Filtered/All data'
                data_pointers.fake_data = self.get_cubes_information(
                    H5PYFile=H5PYFile,
                    group_path=fake_path,
                    cube_name='Fake data',
                    cube_type='fake',
                    interpolate=False,
                )

            if self.plot_choices['test cube']:
                test_path = 'Test data/Sun surface'
                data_pointers.test_cube = self.get_cubes_information(
                    H5PYFile=H5PYFile,
                    group_path=test_path,
                    cube_name='Sun surface - TEST',
                    cube_type='unique',
                    interpolate=False,
                )

            # MULTIPROCESSING
            while True:
        
                # INFO process 
                if input_queue is not None:
                    process: int | None = input_queue.get()
                else:
                    process = index_list[process_id]  # ! this won't work for now
                    process_id += 1
                    process = None if process_id > len(index_list) else process
                
                if process is None: break
                process: int  # todo change this latter
                
                # DATA formatting
                process_constants = ProcessConstants(
                    ID=process,
                    time_index=(
                        process 
                        if self.plot_choices['all sdo images']
                        else self.constants.time_indexes[process]
                    ),
                    cube_index=self.index_to_cube[process],
                    date=self.constants.dates[process].decode('utf8'),
                )
                projection_data = ProjectionData(ID=process)  # ! what the ID represents changed

                # SDO information
                filepath = self.sdo_timestamps[process_constants.date[:-3]]
                if self.in_local: filepath = self.get_file_from_server(filepath)
                sdo_image_info = self.sdo_image(filepath, colour='')
                sdo_image_info.image = self.sdo_image_treatment(sdo_image_info.image)

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

                if data_pointers.all_data is not None:
                    projection_data.all_data = self.format_cube(
                        data=data_pointers.all_data,
                        cube_index=process_constants.cube_index,
                        colour='blue',
                        sdo_info=sdo_image_info,
                        warp=False,
                    )

                if data_pointers.no_duplicates is not None:
                    projection_data.no_duplicates = self.format_cube(
                        data=data_pointers.no_duplicates,
                        cube_index=process_constants.cube_index,
                        colour='orange',
                        sdo_info=sdo_image_info,
                        warp=False,
                    )
                
                if data_pointers.full_integration_all_data is not None:
                    projection_data.full_integration_all_data = self.format_cube(
                        data=data_pointers.full_integration_all_data,
                        cube_index=process_constants.cube_index,
                        colour='brown',
                        sdo_info=sdo_image_info,
                        warp=self.plot_choices['warp'],
                        warp_kwargs=warp_kwargs,
                    )
                
                if data_pointers.full_integration_no_duplicates is not None:
                    projection_data.full_integration_no_duplicates = self.format_cube(
                        data=data_pointers.full_integration_no_duplicates,
                        cube_index=process_constants.cube_index,
                        colour='pink',
                        sdo_info=sdo_image_info,
                        warp=self.plot_choices['warp'],
                        warp_kwargs=warp_kwargs,
                    )

                if data_pointers.line_of_sight is not None:
                    projection_data.line_of_sight = self.format_cube(
                        data=data_pointers.line_of_sight,
                        cube_index=process_constants.cube_index,
                        colour='purple',
                        sdo_info=sdo_image_info,
                        warp=False,
                    )

                if (
                    data_pointers.integration is not None
                    and process_constants.cube_index is not None
                    ): 
                    projection_data.integration = [
                        self.format_cube(
                            data=integration,
                            cube_index=process_constants.cube_index,
                            colour=cast(list[str], self.plot_kwargs['colours'])[i],
                            sdo_info=sdo_image_info,
                            warp=self.plot_choices['warp'],
                            warp_kwargs=warp_kwargs,
                        )
                        for i, integration in enumerate(data_pointers.integration)
                    ]
                    
                if data_pointers.fake_data is not None:
                    projection_data.fake_data = self.format_cube(
                        data=data_pointers.fake_data,
                        cube_index=process_constants.cube_index,
                        colour='black',
                        sdo_info=sdo_image_info,
                        warp=False,
                    )

                if data_pointers.test_cube is not None:
                    projection_data.test_cube = self.format_cube(
                        data=data_pointers.test_cube,
                        cube_index=process_constants.cube_index,
                        colour='yellow',
                        sdo_info=sdo_image_info,
                        warp=False,
                    )

                # WARP add to Auchere's envelope
                if self.plot_choices['warp'] and self.Auchere_envelope is not None:
                    warped_instance = WarpSdoImage(
                        envelopes=[self.Auchere_envelope.upper, self.Auchere_envelope.lower],
                        sdo_image=sdo_image_info.image,
                        **warp_kwargs,
                    )
                    self.Auchere_envelope.warped_image = warped_instance.warped_image

                # CHILD CLASSes functionality
                self.plotting(process_constants, projection_data)
                self.create_fake_fits(process_constants, projection_data)

                # SAVE warp data
                output_queue.put((process, self.Auchere_envelope.warped_image))
                
                
        if self.in_local: self.connection.close()

    @overload
    def format_cube(
        self,
        data: DataPointer | UniqueDataPointer | FakeDataPointer,
        cube_index: int,
        colour: str,
        sdo_info: PolarImageInfo,
        warp: bool = ...,
        warp_kwargs: dict[str, Any] = ...,
    ) -> ProjectedData: ...

    @overload
    def format_cube(
        self,
        data: DataPointer | UniqueDataPointer | FakeDataPointer,
        cube_index: None,
        colour: str,
        sdo_info: PolarImageInfo,
        warp: bool = ...,
        warp_kwargs: dict[str, Any] = ...,
    ) -> None: ...

    @overload  # fallback
    def format_cube(
        self,
        data: DataPointer | UniqueDataPointer | FakeDataPointer,
        cube_index: int | None,
        colour: str,
        sdo_info: PolarImageInfo,
        warp: bool = ...,
        warp_kwargs: dict[str, Any] = ...,
    ) -> ProjectedData | None: ...

    def format_cube(
            self,
            data: DataPointer | UniqueDataPointer | FakeDataPointer,
            cube_index: int | None,
            colour: str,
            sdo_info: PolarImageInfo,
            warp: bool = False,
            warp_kwargs: dict[str, Any] = {},
        ) -> ProjectedData | None:
            """  # todo update docstring
            To format the cube data for the projection.

            Args:
                data (DataPointer | UniqueDataPointer | FakeDataPointer): the data cube to be
                    formatted.
                index (int): the index of the corresponding real data cube.
                colour (str): the colour of the data cube for the plot.
                sdo_info (PolarImageInfo): the SDO information (e.g. the position, the image).
                warp (bool, optional): if the image section inside the fit envelope should be
                    warped. Defaults to False.
                warp_kwargs (dict[str, Any], optional): the kwargs used for the warping of the SDO
                    image section. Defaults to {} (when there is no warping done).

            Returns:
                ProjectedCube: the formatted and reprojected data cube.
            """

            # NO DATA available
            if cube_index is None:
                return None
            else:
                # CUBE formatting
                cube = CubeInformation(
                    xt_min=data.xt_min,
                    yt_min=data.yt_min,
                    zt_min=data.zt_min,
                    coords=data[cube_index],
                )
                cube = self.cartesian_pos(cube, dx=self.constants.dx)
                coords = self.get_polar_image(
                    self.matrix_rotation(data=cube.coords, sdo_pos=sdo_info.sdo_pos),
                )
                cube.coords = coords

                # FIT formatting
                if data.fit_information is None:
                    fit_n_envelopes = None
                else:
                    fit_n_envelopes: list[FitWithEnvelopes] | None = cast(
                        list[FitWithEnvelopes],
                        [None] * len(data.fit_information),
                    )

                    for i, polynomial_order in enumerate(self.polynomial_order):
                        reprojected_polynomial = ReprojectionProcessedPolynomial(
                            name=f"Fit ({polynomial_order}th) of " + data.name.lower(),
                            colour=cast(list[str], self.plot_kwargs['colours'])[i],  # ! most likely the wrong choice of colours
                            filepath=self.filepath,
                            group_path=data.group_path,
                            dx=self.constants.dx,
                            polynomial_order=polynomial_order,
                            number_of_points=300,  # ? should I add it as an argument ?
                            feet_sigma=0.1,
                            feet_threshold=0.1,
                            envelope_radius=4e4,
                            create_envelope=self.plot_choices['fit envelope'],
                        )
                        
                        fit_n_envelope = (
                            reprojected_polynomial.reprocessed_fit_n_envelopes(
                                index=cube_index,
                                sdo_pos=sdo_info.sdo_pos,
                            )
                        )

                        # WARP IMAGE add
                        if (fit_n_envelope.envelopes is not None) and warp:
                            warped_instance = WarpSdoImage(
                                envelopes=fit_n_envelope.envelopes,
                                sdo_image=sdo_info.image,
                                **warp_kwargs,
                            )
                            fit_n_envelope.warped_image = warped_instance.warped_image
                        # RESULT save
                        fit_n_envelopes[i] = fit_n_envelope

                # PROJECTION formatting
                projection = ProjectedData(
                    name=data.name,
                    colour=colour,
                    cube_index=cube_index,
                    cube=cube,
                    integration_time=data.integration_time,
                    fit_n_envelopes=fit_n_envelopes,
                )
                return projection

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
            time.sleep(0.5)
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
        dx = float(cast(h5py.Dataset, H5PYFile['dx'])[...])
        time_indexes: np.ndarray = cast(h5py.Dataset, H5PYFile[init_path + 'Time indexes'])[...]
        dates: np.ndarray = cast(h5py.Dataset, H5PYFile['Dates'])[...]

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
            cube_name: str,
            cube_type: Literal['real'] = ...,
            integration_time: int | str | None = ...,
            interpolate: bool = ...,
        ) -> DataPointer: ...
    
    @overload
    def get_cubes_information(
            self,
            H5PYFile: h5py.File,
            group_path: str,
            cube_name: str,
            cube_type: Literal['unique'],
            integration_time: int | str | None = ...,
            interpolate: bool = ...,
        ) -> UniqueDataPointer: ...
    
    @overload
    def get_cubes_information(
            self,
            H5PYFile: h5py.File,
            group_path: str,
            cube_name: str,
            cube_type: Literal['fake'],
            integration_time: int | str | None = ...,
            interpolate: bool = ...,
        ) -> FakeDataPointer: ...
    
    @overload # fallback
    def get_cubes_information(
            self,
            H5PYFile: h5py.File,
            group_path: str,
            cube_name: str,
            cube_type: str = ...,
            integration_time: int | str | None = ...,
            interpolate: bool = ...,
        ) -> DataPointer | UniqueDataPointer | FakeDataPointer: ...

    def get_cubes_information(
            self,
            H5PYFile: h5py.File,
            group_path: str,
            cube_name: str,
            cube_type: str = 'real',
            integration_time: int | str | None = None,
            interpolate: bool = False,
        ) -> DataPointer | UniqueDataPointer | FakeDataPointer:
        """
        To get the information about the data cubes.

        Args:
            H5PYFile (h5py.File): the HDF5 file containing the data.
            group_path (str): the path to the group containing the data.
            cube_name (str): the name given to the data cube.
            cube_type (str, optional): the type of the dataclass cube used. Defaults to 'real'.
            integration_time (int | str | None, optional): the integration time of the data. If
                None, the integration time is not used. If a string, it means that the data used is
                one of the full integration one. Defaults to None.
            interpolate (bool, optional): if the interpolation data exists. Defaults to False.

        Returns:
            DataPointer | UniqueDataPointer | FakeDataPointer: the information about the data
                cubes.
        """

        # BORDERs
        xt_min = float(cast(h5py.Dataset, H5PYFile[group_path + '/xt_min'])[...])
        yt_min = float(cast(h5py.Dataset, H5PYFile[group_path + '/yt_min'])[...])
        zt_min = float(cast(h5py.Dataset, H5PYFile[group_path + '/zt_min'])[...])

        if cube_type == 'real':
            # FORMAT data
            cube_info = DataPointer(
                xt_min=xt_min,
                yt_min=yt_min,
                zt_min=zt_min,
                group_path=group_path,
                pointer=cast(h5py.Dataset, H5PYFile[group_path + '/coords']),
                name=cube_name,
                integration_time=integration_time,
                fit_information=[
                    self.get_fit_pointer_information(
                        H5PYFile=H5PYFile,
                        group_path=group_path,
                        cube_type='real',
                        integration_time=integration_time,
                        polynomial_order=polynomial_order,
                    )
                    for polynomial_order in self.polynomial_order
                ] if interpolate and (integration_time is not None) else None,
            )
        elif cube_type == 'unique':
            # FORMAT data
            cube_info = UniqueDataPointer(
                xt_min=xt_min,
                yt_min=yt_min,
                zt_min=zt_min,
                group_path=group_path,
                pointer=cast(h5py.Dataset, H5PYFile[group_path + '/coords']),
                name=cube_name,
                integration_time=integration_time,
                fit_information=[
                    self.get_fit_pointer_information(
                        H5PYFile=H5PYFile,
                        group_path=group_path,
                        cube_type='unique',
                        integration_time=integration_time,
                        polynomial_order=polynomial_order,
                    )
                    for polynomial_order in self.polynomial_order
                ] if interpolate and (integration_time is not None) else None              
            )

        else:
            # FAKE time indexes
            time_indexes: np.ndarray = cast(h5py.Dataset, H5PYFile['Fake/Time indexes'])[...]

            # FORMAT cube
            cube_info = FakeDataPointer(
                xt_min=xt_min,
                yt_min=yt_min,
                zt_min=zt_min,
                group_path=group_path,
                pointer=cast(h5py.Dataset, H5PYFile[group_path + '/coords']),
                name=cube_name,
                integration_time=None,  # ? is this always True for the fake data cubes? 
                real_time_indexes=self.constants.time_indexes,
                fake_time_indexes=time_indexes,
            )
        return cube_info

    @overload
    def get_fit_pointer_information(
            self,
            H5PYFile: h5py.File,
            group_path: str,
            cube_type: Literal['real'],
            integration_time: int | str,
            polynomial_order: int,
        ) -> FitPointer: ...

    @overload
    def get_fit_pointer_information(
            self,
            H5PYFile: h5py.File,
            group_path: str,
            cube_type: Literal['unique'],
            integration_time: int | str,
            polynomial_order: int,
        ) -> UniqueFitPointer: ...
    
    @overload  # fallback
    def get_fit_pointer_information(
            self,
            H5PYFile: h5py.File,
            group_path: str,
            cube_type: str,
            integration_time: int | str,
            polynomial_order: int,
        ) -> FitPointer | UniqueFitPointer: ...
    
    def get_fit_pointer_information(
            self,
            H5PYFile: h5py.File,
            group_path: str,
            cube_type: str,
            integration_time: int | str,
            polynomial_order: int,
        ) -> FitPointer | UniqueFitPointer:
        """
        To get the information about the fit pointer.

        Args:
            H5PYFile (h5py.File): the HDF5 file containing the data.
            group_path (str): the path to the group containing the data.
            cube_type (str): the type of the dataclass cube used.
            integration_time (int | str): the integration time of the data. If the integration time
                is a string, it means that the data used is the full integration type one.
            polynomial_order (int): the order of the polynomial fit.

        Raises:
            ValueError: if the cube type is not recognised.

        Returns:
            FitPointer | UniqueFitPointer: the information about the polynomial fit data.
        """

        # FIT parameters
        params: h5py.Dataset = cast(
            h5py.Dataset,
            H5PYFile[group_path + f'/{polynomial_order}th order polynomial/parameters'],
        )

        # FIT information
        if cube_type == 'real':
            info = FitPointer(
                fit_order=polynomial_order,
                integration_time=integration_time,
                parameters=params,
            )
        elif cube_type == 'unique':
            info = UniqueFitPointer(
                fit_order=polynomial_order,
                integration_time=integration_time,
                parameters=params,
            )
        else:
            raise ValueError("Wrong cube type for function 'get_fit_pointer_information.")
        return info

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
        rho //= self.constants.dx
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
        lines = self.image_contour(image, d_theta)
        return image, lines
    
    def image_contour(
            self,
            image: np.ndarray,
            d_theta: float,
        ) -> list[tuple[list[float], list[float]]] | None:
        """ 
        To get the contours in the final plot coordinates of a mask given the corresponding
        information.

        Args:
            image (np.ndarray): the mask.
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
        nw_lines: list[tuple[list[float], list[float]]] = cast(
            list[tuple[list[float], list[float]]],
            [None] * len(lines),
        )
        for i, line in enumerate(lines):
            nw_lines[i] = ((
                [
                    (
                        min(self.projection_borders.radial_distance) + (value * self.constants.dx)
                        / 1e3
                    )
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
        plt.figure(num=1, figsize=(18, 5))
        if self.Auchere_envelope is not None: 
            # PLOT Auchere's envelope
            self.plt_plot(
                coords=self.Auchere_envelope.middle,
                colour='blue',
                label='Middle path',
                kwargs=cast(dict[str, Any], self.plot_kwargs['envelope']),
            )
            self.plt_plot(
                coords=self.Auchere_envelope.upper,
                colour='black',
                label='Envelope',
                kwargs=cast(dict[str, Any], self.plot_kwargs['envelope']),
            )
            self.plt_plot(
                coords=self.Auchere_envelope.lower,
                colour='black',
                kwargs=cast(dict[str, Any], self.plot_kwargs['envelope']),
            )

            if self.Auchere_envelope.warped_image is not None:
                self.plot_warped_image(
                    warped_image=self.Auchere_envelope.warped_image,
                    integration_time=None,
                    date=process_constants.date,
                    fit_order=None,
                    envelope_order=self.Auchere_envelope.upper.order,
                )

        if projection_data.sdo_mask is not None:
            # CONTOURS get
            lines = self.image_contour(
                image=projection_data.sdo_mask.image,
                d_theta=projection_data.sdo_mask.resolution_angle,
            )

            # CONTOURS plot
            if lines is not None:
                line = lines[0]
                self.plt_plot(
                    coords=(line[1], line[0]),
                    colour=projection_data.sdo_mask.colour,
                    label='SDO mask contours',
                    kwargs=cast(dict[str, Any], self.plot_kwargs['contour']),
                )
                for line in lines[1:]:
                    self.plt_plot(
                        coords=(line[1], line[0]),
                        colour=projection_data.sdo_mask.colour,
                        kwargs=cast(dict[str, Any], self.plot_kwargs['contour']),
                    )

        if projection_data.sdo_image is not None:
            plt.imshow(
                projection_data.sdo_image.image,
                **cast(dict[str, Any], self.plot_kwargs['image']),
            )

        sc = None  # for the colorbar
        if projection_data.integration is not None:
            
            for integration in projection_data.integration:
                # PLOT contours time integrated
                sc = self.plot_projected_data(
                    data=integration,
                    process_constants=process_constants,
                    image_shape=image_shape,
                )
        
        if projection_data.all_data is not None:
            # PLOT contours all data
            self.plot_contours(
                projection=projection_data.all_data,
                d_theta=self.constants.d_theta,
                image_shape=image_shape,
            )

        if projection_data.no_duplicates is not None:
            # PLOT contours no duplicates
            self.plot_contours(
                projection=projection_data.no_duplicates,
                d_theta=self.constants.d_theta,
                image_shape=image_shape,
            )

        if projection_data.full_integration_all_data is not None:
            # PLOT contours full integration all data
            sc = self.plot_projected_data(
                data=projection_data.full_integration_all_data,
                process_constants=process_constants,
                image_shape=image_shape,
            )
        
        if projection_data.full_integration_no_duplicates is not None:
            # PLOT contours full integration no duplicates
            sc = self.plot_projected_data(
                data=projection_data.full_integration_no_duplicates,
                process_constants=process_constants,
                image_shape=image_shape,
            )

        if projection_data.line_of_sight is not None:
            # PLOT contours line of sight
            self.plot_contours(
                projection=projection_data.line_of_sight,
                d_theta=self.constants.d_theta,
                image_shape=image_shape,
            )

        # PLOT fake data
        if projection_data.fake_data is not None:
            # PLOT contours fake data
            self.plot_contours(
                projection=projection_data.fake_data,
                d_theta=self.constants.d_theta,
                image_shape=image_shape,
            )

        # PLOT test cube
        if projection_data.test_cube is not None:
            # PLOT contours test cube
            self.plot_contours(
                projection=projection_data.test_cube,
                d_theta=self.constants.d_theta,
                image_shape=image_shape,
            )

        # # COLORBAR add
        # if sc is not None:
        #     cbar = plt.colorbar(sc)
        #     cbar.set_label(r'$\theta$ (degrees)')

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
        plt.savefig(os.path.join(self.paths['save'], plot_name), dpi=200)
        plt.close()

        if self.verbose > 1: 
            print(
                f'SAVED - nb {process_constants.time_index:03d} - {plot_name}',
                flush=self.flush,
            )

    def plot_projected_data(
            self,
            data: ProjectedData,
            process_constants: ProcessConstants,
            image_shape: tuple[int, int],
        ) -> PathCollection | None:
        """
        To plot the projected data and the corresponding fit and envelope if they exist.

        Args:
            data (ProjectedData): the projected data to be plotted.
            process_constants (ProcessConstants): the constants for each cube.
            image_shape (tuple[int, int]): the final image shape used in the contours plotting.

        Returns:
            PathCollection | None: the scatter plot of the projected polynomial fit. It is set to
                None if there is no polynomial fit to be plotted. This is later used to add a
                colorbar to the final plot.
        """

        # CONTOURS image
        self.plot_contours(
            projection=data,
            d_theta=self.constants.d_theta,
            image_shape=image_shape,
        )

        # FIT plot
        sc = None
        if data.fit_n_envelopes is not None:

            for fit_n_envelope in data.fit_n_envelopes:
                
                # PLOT
                sc = plt.scatter(
                    fit_n_envelope.fit_polar_theta,
                    fit_n_envelope.fit_polar_r / 1e3,
                    label=fit_n_envelope.name,
                    c=np.rad2deg(fit_n_envelope.fit_angles),
                    **cast(dict[str, Any], self.plot_kwargs['fit']),
                )

                # ENVELOPE fit
                if fit_n_envelope.envelopes is not None:
                    # PLOT envelope
                    for label, new_envelope in enumerate(fit_n_envelope.envelopes):
                        self.plt_plot(
                            coords=new_envelope,
                            colour=fit_n_envelope.colour,
                            label=(
                                f'Envelope ({new_envelope.order}th) for '
                                + fit_n_envelope.name.lower()
                            ) if label==0 else None,
                            kwargs=cast(dict[str, Any], self.plot_kwargs['fit envelope']),
                        )

                    # WARP image
                    if fit_n_envelope.warped_image is not None:
                        # PLOT inside a new figure
                        self.plot_warped_image(
                            warped_image=fit_n_envelope.warped_image,
                            integration_time=data.integration_time,
                            date=process_constants.date,
                            fit_order=fit_n_envelope.fit_order,
                            envelope_order=fit_n_envelope.envelopes[0].order,
                        )
        return sc
    
    def plot_contours(
            self,
            projection: ProjectedData,
            d_theta: float, 
            image_shape: tuple[int, int],
        ) -> None:
        """
        To plot the contours of the image for the protuberance as seen from SDO's pov.

        Args:
            projection (ProjectedCube): the cube containing the information for the protuberance
                as seen from SDO's pov.
            d_theta (float): the theta angle resolution (as a function of the disk's perimeter) in
                degrees.
            image_shape (tuple[int, int]): the image shape needed for the image of the protuberance
                as seen from SDO.
        """

        # POLAR coordinates
        rho, theta = projection.cube.coords
        colour = projection.colour

        # CONTOURS cube
        _, lines = self.cube_contour(
            rho=rho,
            theta=theta,
            image_shape=image_shape,
            d_theta=d_theta,
        )

        # PLOT
        if lines is not None:
            line = lines[0]
            self.plt_plot(
                coords=(line[1], line[0]),
                colour=colour,
                label=projection.name + ' contour',
                kwargs=cast(dict[str, Any], self.plot_kwargs['contour']),
            )
            for line in lines:
                self.plt_plot(
                    coords=(line[1], line[0]),
                    colour=colour,
                    kwargs=cast(dict[str, Any], self.plot_kwargs['contour']),
                )

    def plot_warped_image(
            self,
            warped_image: np.ndarray,
            integration_time: int | str | None,
            date: str,
            fit_order: int | None, 
            envelope_order: int,
        ) -> None:
        """
        To plot the warped SDO image inside the fit envelope.

        Args:
            warped_image (np.ndarray): the warped SDO image to be plotted.
            integration_time (int | str | None): the integration time of the data. If the value is
                None, it means that the data doesn't have an integration time. If the value is a
                string, it means that the used data is the full integration one.
            date (str): the date of the data.
            fit_order (int | None): the order of the polynomial fit. If the value is None, it means
                that the fit doesn't have a polynomial order per se (like the middle path of 
                Auchere's envelope).
            envelope_order (int): the polynomial order of the envelope used to warp the SDO image.
        """

        # PLOT
        plot_name = (
            "warped_"
            f"{f'{integration_time}h_' if integration_time is not None else ''}"
            f"{f'{fit_order}fit_' if fit_order is not None else ''}"
            f"{envelope_order}envelope_{date}.png"
        )
        plt.figure(num=2, figsize=(10, 10))
        plt.imshow(
            X=warped_image.T,
            interpolation='none',
            cmap='gray',
            origin='lower',
        )
        plt.title(f'Warped SDO image - {date}')
        plt.savefig(os.path.join(self.paths['save warped'], plot_name), dpi=200)
        plt.close(2)
        plt.figure(1)

        if self.verbose > 0: print(f'SAVED - {plot_name}', flush=self.flush)

    def plt_plot(
            self,
            coords: tuple[np.ndarray, np.ndarray] | tuple[list, list] | FitEnvelopes,
            colour: str,
            label: str | None = None,
            kwargs: dict[str, Any] = {},
        ) -> None:
        """
        As I am using plt.plot a lot in this code, I put the usual plotting parameters in a method.

        Args:
            coords (tuple[np.ndarray, np.ndarray] | FitEnvelopes): the coordinates of the points to
                be plotted.
            colour (str): the colour of the plot lines.
            label (str | None, optional): the label of the plot lines. Defaults to None.
            kwargs (dict[str, Any], optional): additional arguments to be added to the plt.plot().
                Defaults to {}.
        """

        # CHECK
        if isinstance(coords, tuple):
            # PLOT
            plt.plot(coords[0], coords[1], label=label, color=colour, **kwargs)
        else:
            # PLOT
            plt.plot(
                coords.polar_theta,
                coords.polar_r / 1e3,
                label=label,
                color=colour,
                **kwargs,
            )

            # ANNOTATE
            EnvelopeDistanceAnnotation(
                fit_envelope=coords,
                colour=colour,
            )



if __name__ == '__main__':
    Plotting(
        integration_time=[24],
        polynomial_order=[4],
        plot_choices=[
            'no duplicates',
            'full integration no duplicates',
            'fit',
            'sdo image', 'envelope',
            'warp',
            'all sdo images',
        ],
        with_fake_data=False,
    )
