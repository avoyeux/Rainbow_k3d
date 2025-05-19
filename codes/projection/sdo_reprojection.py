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

# IMPORTS personal
from common import config, Decorators, Plot
from codes.projection.format_data import (
    GlobalConstants, ProcessConstants, ImageBorders, PolarImageInfo,
    ProjectionData, ProjectedData, CubeInformation, EnvelopeInformation,
    CubesPointers, DataPointer, UniqueDataPointer, FakeDataPointer, FitPointer, UniqueFitPointer,
    FitWithEnvelopes, WarpedDataGroup, AllWarpedInformation,
)
from codes.projection.helpers.extract_envelope import ExtractEnvelope
from codes.projection.helpers.base_reprojection import BaseReprojection
from codes.projection.helpers.cartesian_to_polar import CartesianToPolar
from codes.projection.helpers.warped_information import AllWarpedTreatment
from codes.data.polynomial_fit.polynomial_reprojection import ReprojectionProcessedPolynomial

# TYPE ANNOTATIONs
import queue
from typing import cast, overload, Literal
type QueueAlias[T] = queue.Queue[T]  # used parent function

# API public
__all__ = ['OrthographicalProjection']

# todo 'integration' of warp images needs to be able to work on multiple datasets at the same time

# ? should I force the creation of a fit and Auchere's envelope as it is necessary for the last 
# ? warped integration plot. As opposed to adding multiple if statements
# ! need to decide how to use different values for the shape of the warped image and the number of
# ! points used in the polynomial fit and the resulting radial distance.



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
            polynomial_order: list[int] = [4],
            plot_choices: list[str] = [
                'sdo image', 'no duplicates', 'envelope', 'polynomial', 'test data',
            ],
            arc_length_points: int = 1280,
            warp_pixel_interpolation_order: int = 3,
            warp_integration_type: Literal['mean', 'median'] = 'mean',
            with_feet: bool = False,
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

            polynomial_order (list[int], optional): the order(s) of the polynomial function(s) that
                represent the fitting of the integrated 3D volume. Defaults to [4].
            plot_choices (list[str], optional): the main choices that the user wants to be in the
                reprojection. The possible choices are:
                ['full integration', 'integration',
                'no duplicates', 'sdo image', 'sdo mask', 'test cube', 'fake data', 'envelope',
                'fit', 'fit envelope', 'test data', 'line of sight', 'all data', 'warp']
                ['sdo image', 'no duplicates', 'envelope', 'polynomial', 'test data'].
            arc_length_points (int, optional): the number of points used to compute the arc length
                for the polynomial fit. Defaults to 1280.
            warp_pixel_interpolation_order (int, optional): the order of the interpolation used to
                warp the data. Defaults to 3.
            warp_integration_type (Literal['mean', 'median'], optional): the type of integration
                used to compute the final warped image. Defaults to 'mean'.
            with_feet (bool, optional): deciding to use the data with or without added feet.
                Defaults to False.
            with_fake_data (bool, optional): if the input data is the fusion HDF5 file.
                Defaults to False.
            verbose (int | None, optional): gives the verbosity in the outputted prints. The higher
                the value, the more prints. Starts at 0 for no prints. When None, uses the config
                file. Defaults to None.
            flush (bool | None, optional): used in the 'flush' kwarg of the print() class.
                Decides to force the print buffer to be emptied (i.e. forces the prints). When
                None, uses the config file. Defaults to None.
        """

        # PLACEHOLDERs
        self._warped_information: AllWarpedInformation | None = None

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
        self.arc_length_points = arc_length_points
        self.warp_integration_type: Literal['mean', 'median'] = warp_integration_type
        self.warp_pixel_interpolation_order = warp_pixel_interpolation_order

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

        # GLOBAL data
        self.Auchere_envelope, self.plot_kwargs = self.global_information()

    @property
    def warped_information(self) -> AllWarpedInformation | None:
        """
        To get the warped information after having initialised the class.

        Returns:
            AllWarpedInformation | None: the totality of the warped information data.
        """

        return self._warped_information

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
        if self.plot_choices['warp']: os.makedirs(paths['save warped'], exist_ok=True)
        return paths

    def plot_choices_creation(self, plot_choices: list[str]) -> dict[str, bool]:
        """ 
        Creating a dictionary that chooses what to plot.

        Args:
            plot_choices (list[str]): choices made for the plotting.

        Raises:
            ValueError: if the plotting choice string is not recognised.

        Returns:
            dict[str, bool]: decides what will be plotted later on.
        """

        # CHOICES
        possibilities = [
            'full integration',
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
            number_of_points=1280,  # ! need to make this value global so that there are no issues in array sizes later on
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

        # STATS data
        with h5py.File(self.filepath, 'r') as H5PYFile:
            dates: np.ndarray = cast(h5py.Dataset, H5PYFile['Dates'])[...]
            data_len = len(dates)

        if self.multiprocessing:
            # INFO multiprocessing
            nb_processes = min(self.processes, data_len)

            # SETUP multiprocessing
            processes: list[mp.Process] = cast(list[mp.Process], [None] * nb_processes)
            manager = mp.Manager()
            input_queue = manager.Queue()
            output_queue: QueueAlias[tuple[int, ProjectionData]] = manager.Queue()
            for i in range(data_len): input_queue.put(i)  # todo change this to a value proxy
            for _ in range(nb_processes): input_queue.put(None)

            # RUN multiprocessing
            for i in range(nb_processes):
                p = mp.Process(
                    target=self.data_setup,
                    kwargs={'inputs': input_queue, 'output_queue': output_queue},
                )
                p.start()
                processes[i] = p
            for p in processes: p.join()

            # GET warped data
            warped_information_list: list[ProjectionData] = cast(
                list[ProjectionData],
                [None] * data_len,
            )
            while not output_queue.empty():
                identifier, warped = output_queue.get()
                warped_information_list[identifier] = warped

        else:
            raise NotImplementedError(
                "Need to decide how to implement the data passing when no multiprocessing"
            )

        if self.in_local: 
            self.connection.close()
            self.connection.cleanup(verbose=self.verbose)

        # SAVE WARPED data
        self._warped_information = self.restructure_warped_information(warped_information_list)  

    @overload
    def data_setup(self, inputs: int, output_queue: QueueAlias | None = ...) -> np.ndarray: ...

    @overload
    def data_setup(self, inputs: QueueAlias, output_queue: QueueAlias | None = ...) -> None: ...

    @overload  #fallback
    def data_setup(
            self,
            inputs: QueueAlias | int,
            output_queue: QueueAlias | None = ...,
        ) -> np.ndarray | None: ...

    def data_setup(
            self,
            inputs: QueueAlias[int] | int,
            output_queue: QueueAlias[tuple[int, ProjectionData]] | None = None,
        ) -> np.ndarray | None:
        """  # todo change docstring and the return type as it cannot be an ndarray any more
        Open the HDF5 file and does the processing and final plotting for each cube.
        A while loop is used to decide which data section needs to be processed.

        Args:
            inputs (QueueAlias | int): the number of processes to be done. If multiprocessing is
                used, it is a QueueAlias object. If not, it is an int.
            output_queue (QueueAlias | None, optional): the queue used to send the warped data
                back to the main process. Defaults to None.
        
        Returns:
            np.ndarray | None: the warped data if no multiprocessing is used. Otherwise, None.
        """

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
            
            if self.plot_choices['full integration']:
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
                        + f'/Time integration of {integration_time} hours'
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

            # NOT MULTIPROCESSING
            outputs: list[np.ndarray] = cast(
                list[np.ndarray],
                [None] * inputs if isinstance(inputs, int) else [],
            )
            # MULTIPROCESSING
            while True:
        
                # INFO process 
                if not isinstance(inputs, int):
                    process: int | None = inputs.get()
                    if process is None: return
                else:
                    inputs -= 1
                    if inputs < 0: break
                    process = inputs

                # DATA formatting
                process_constants = ProcessConstants(
                    ID=process,
                    time_index=process,
                    date=self.constants.dates[process].decode('utf8'),
                )
                projection_data = ProjectionData(ID=process)  # ! what the ID represents changed

                # SDO information
                filepath = self.constants.ias_paths[process].decode('utf8')
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
                        constants=process_constants,
                        colour='blue',
                        sdo_info=sdo_image_info,
                        warp=False,
                    )

                if data_pointers.no_duplicates is not None:
                    projection_data.no_duplicates = self.format_cube(
                        data=data_pointers.no_duplicates,
                        constants=process_constants,
                        colour='orange',
                        sdo_info=sdo_image_info,
                        warp=False,
                    )
                
                if data_pointers.full_integration_no_duplicates is not None:
                    projection_data.full_integration_no_duplicates = self.format_cube(
                        data=data_pointers.full_integration_no_duplicates,
                        constants=process_constants,
                        colour='pink',
                        sdo_info=sdo_image_info,
                        warp=self.plot_choices['warp'],
                    )

                if data_pointers.line_of_sight is not None:
                    projection_data.line_of_sight = self.format_cube(
                        data=data_pointers.line_of_sight,
                        constants=process_constants,
                        colour='purple',
                        sdo_info=sdo_image_info,
                        warp=False,
                    )

                if data_pointers.integration is not None: 
                    projection_data.integration = [
                        self.format_cube(
                            data=integration,
                            constants=process_constants,
                            colour=cast(list[str], self.plot_kwargs['colours'])[i],
                            sdo_info=sdo_image_info,
                            warp=self.plot_choices['warp'],
                        )
                        for i, integration in enumerate(data_pointers.integration)
                    ]
                    
                if data_pointers.fake_data is not None:
                    projection_data.fake_data = self.format_cube(
                        data=data_pointers.fake_data,
                        constants=process_constants,
                        colour='black',
                        sdo_info=sdo_image_info,
                        warp=False,
                    )

                if data_pointers.test_cube is not None:
                    projection_data.test_cube = self.format_cube(
                        data=data_pointers.test_cube,
                        constants=process_constants,
                        colour='yellow',
                        sdo_info=sdo_image_info,
                        warp=False,
                    )

                # CHILD CLASSes functionality
                self.plotting(process_constants, projection_data)
                self.create_fake_fits(process_constants, projection_data)

                # WARP final plot
                if self.plot_choices['warp']:
                    if output_queue is not None:
                        output_queue.put((process, projection_data))
                    else:
                        raise NotImplementedError(
                            'Still need to decide what to do when not multiprocessing'
                        )
        return np.stack(outputs, axis=0) if output_queue is None else None

    def restructure_warped_information(
            self,
            all_projection_data: list[ProjectionData],
        ) -> AllWarpedInformation:
        """
        To restructure all the warped information into a single dataclass for easier use.
        Especially useful for the warp integration plotting.

        Args:
            all_projection_data (list[ProjectionData]): the list of all the projection data after
                the unpickling of the ProjectionData class instances.

        Returns:
            AllWarpedInformation: the totality of the warped information data.
        """

        # EMPTY INIT
        result = AllWarpedInformation()
        result.full_integration_no_duplicates = WarpedDataGroup(
            name='full integration no duplicates',
        )
        result.integration = WarpedDataGroup(name='integration')

        # RESTRUCTURE data
        for projection_data in all_projection_data:

            # DATA full integration
            if (full_integration := projection_data.full_integration_no_duplicates) is not None:
                result.full_integration_no_duplicates = self.restructure_warped_information_sub(
                    data=full_integration,
                    result=result.full_integration_no_duplicates,
                )

            # DATA integrations
            if (integrations := projection_data.integration) is not None:
                for integration in integrations:
                    result.integration = self.restructure_warped_information_sub(
                        data=integration,
                        result=result.integration,
                    )
        return result
    
    def restructure_warped_information_sub(
            self,
            data: ProjectedData,
            result: WarpedDataGroup,
        ) -> WarpedDataGroup:
        """
        To add the warped information to the AllWarpedInformation dataclass.

        Args:
            data (ProjectedData): the warped information to be added.
            result (WarpedDataGroup): the dataclass to populate.

        Returns:
            WarpedDataGroup: the populated dataclass.
        """

        if (fit_n_envelopes := data.fit_n_envelopes) is not None:
            for fit_n_envelope in fit_n_envelopes:
                if (warped_info := fit_n_envelope.warped_information) is not None:
                    result.append(warped_info)
        return result

    def format_cube(
            self,
            data: DataPointer | UniqueDataPointer | FakeDataPointer,
            constants: ProcessConstants,
            colour: str,
            sdo_info: PolarImageInfo,
            warp: bool = False,
        ) -> ProjectedData:
            """
            To format the cube data for the projection.

            Args:
                data (DataPointer | UniqueDataPointer | FakeDataPointer): the data cube to be
                    formatted.
                constants (ProcessConstants): the constants for each process (e.g. the date,
                    time index).
                colour (str): the colour of the data cube for the plot.
                sdo_info (PolarImageInfo): the SDO information (e.g. the position, the image).
                warp (bool, optional): if the image section inside the fit envelope should be
                    warped. Defaults to False.

            Returns:
                ProjectedCube: the formatted and reprojected data cube.
            """

            # CUBE formatting
            cube = CubeInformation(
                xt_min=data.xt_min,
                yt_min=data.yt_min,
                zt_min=data.zt_min,
                coords=data[constants.time_index],
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
                        number_of_points=1280,  # ? should I add it as an argument ?
                        feet_sigma=0.1,
                        feet_threshold=0.1,
                        envelope_radius=4e4,
                        create_envelope=(  # todo need to think about this setup
                            self.plot_choices['fit envelope']
                            and self.Auchere_envelope is None
                        ),
                    )
  
                    fit_n_envelope = (
                        reprojected_polynomial.reprocessed_fit_n_envelopes(
                            index=constants.time_index,
                            sdo_pos=sdo_info.sdo_pos,
                        )
                    )

                    if self.Auchere_envelope is not None:
                        fit_n_envelope.envelopes = self.Auchere_envelope

                    # WARP IMAGE add
                    if (fit_n_envelope.envelopes is not None) and warp:
                        warped_instance = AllWarpedTreatment(
                            sdo_image=sdo_info.image,
                            date=constants.date,
                            integration_time=data.integration_time,
                            fit_n_envelopes=fit_n_envelope,
                            borders=self.projection_borders,
                            nb_of_points=self.arc_length_points,
                            integration_type=self.warp_integration_type,
                            pixel_interpolation_order=self.warp_pixel_interpolation_order,
                        )
                        fit_n_envelope.warped_information = warped_instance.warped_information
                    # RESULT save
                    fit_n_envelopes[i] = fit_n_envelope

            # PROJECTION formatting
            projection = ProjectedData(
                name=data.name,
                colour=colour,
                cube_index=constants.time_index,
                cube=cube,
                integration_time=data.integration_time,
                fit_n_envelopes=fit_n_envelopes,
            )
            return projection

    def get_file_from_server(self, filepath: str, fail_count: int = 0) -> str:
        """  # ! doesn't restart the download as intended... 
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
        ias_paths: np.ndarray = cast(h5py.Dataset, H5PYFile['IAS paths'])[...]

        # FORMAT data
        constants = GlobalConstants(
            dx=dx,
            dates=dates,
            solar_r=self.solar_r,
            ias_paths=ias_paths,
            time_indexes=time_indexes,
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
