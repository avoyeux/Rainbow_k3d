"""
To 3D visualise the rainbow filament data and the corresponding polynomial fitting.
"""

# IMPORTs
import os
import re
import k3d
import time
import h5py
import IPython
import threading
import typeguard
import ipywidgets

# IMPORTs alias
import numpy as np

# IMPORTs sub
from astropy.io import fits
from matplotlib import colors as mcolors
from typing import Any, overload, Literal, cast

# IMPORTs personal
from codes.animation.animation_dataclasses import *
from common import config, Decorators, Plot

# ANNOTATION alias
VoxelType = Any
JsLinkType = Any

# todo need to improve the names of the different datasets in the visualisation



class Setup:
    """
    Manipulates the HDF5 filament data file to setup the necessary data choices for the
    visualisation.
    This class is the parent class to the k3d visualisation class named K3dAnimation.
    """

    @typeguard.typechecked
    def __init__(
            self,
            filepath: str | None = None,
            choices: list[str] = ['sdo pov', 'sun', 'no duplicate', 'fit'],
            time_interval: int = 24,
            polynomial_order: list[int] = [4],
            with_fake_data: bool = False,
            processes: int | None = None,
            verbose: int | None = None,
            flush: int | None = None,
    ) -> None:
        """
        To setup the necessary data for the visualisation.

        Args:
            filepath (str | None, optional): the filepath to the HDF5 file. If None, it will use
                the config file. Defaults to None.
            choices (list[str], optional): the choices for the visualisation.
                Defaults to ['sdo pov', 'sun', 'no duplicate', 'fit'].
            time_interval (int, optional): time interval use for the data integration (in hours).
                Defaults to 24.
            polynomial_order (list[int], optional): the polynomial order(s) used for the fitting.
                Defaults to [4].
            with_fake_data (bool, optional): using a file that contains the real and the fact data.
                Used to find the dataset paths. Defaults to False.
            processes (int | None, optional): the number of processes used for the multiprocessing.
                If None, it will use the config file. Defaults to None.
            verbose (int | None, optional): the verbosity level for the prints. If None, it will
                use the config file. Defaults to None.
            flush (int | None, optional): deciding to flush the buffer each time there is a print.
                If None, it will use the config file. Defaults to None.
        """

        # CONSTANTs
        self.solar_r = 6.96e5

        # CONFIGURATION attributes
        if filepath is None and with_fake_data:
            self.filepath: str = config.path.data.fusion  #type:ignore
        elif filepath is None:
            self.filepath: str = config.path.data.real  #type:ignore
        else:
            self.filepath = filepath
        self.processes = config.run.processes if processes is None else processes  #type:ignore
        self.verbose = config.run.verbose if verbose is None else verbose  #type:ignore
        self.flush = config.run.flush if flush is None else flush  #type:ignore

        # ATTRIBUTEs other
        self.time_interval = time_interval
        self.polynomial_order = polynomial_order
        self.with_fake_data = with_fake_data

        # ATTRIBUTES new
        self.choices = self._choices(choices)
        self.plot_polynomial_colours = [
            next(Plot.random_hexadecimal_int_color_generator())
            for _ in self.polynomial_order
        ]
        self.feet = ' with feet' if self.choices['with feet'] else ''
        
        # PLACEHOLDERs
        self.radius_index: float

        # RUN
        self.paths = self.setup_paths()
        self.cubes: CubesData = self.get_data()

    def setup_paths(self) -> dict[str, str]:
        """
        Creates a dictionary for the filepaths.

        Returns:
            dict[str, str]: the filepath dictionary.
        """

        # PATHs save
        paths = {'sdo': config.path.dir.data.sdo}  #type:ignore

        # PATHs update
        return paths

    def _choices(self, plot_choices: list[str]) -> dict[str, bool]:
        """
        Checks and formats the choices given by the user.

        Args:
            plot_choices (list[str]): the choices given by the user.

        Raises:
            ValueError: if the choice is not recognised.

        Returns:
            dict[str, bool]: the formatted choices.
        """

        # CHOICES
        possibilities = [ 
            'sun', 
            'all data', 'all data integration', 'all data full integration',
            'no duplicate', 'no duplicate integration', 'no duplicate full integration',
            'line of sight sdo', 'line of sight stereo',
            'pov sdo', 'pov stereo',
            'fit', 'with feet',
            'test points', 'test cube', 'fake data',
        ]
        choices_kwargs = {
            key: False 
            for key in possibilities
        }

        # DICT reformatting
        for key in plot_choices: 
            if key in possibilities: 
                choices_kwargs[key] = True
            else: 
                raise ValueError(
                    f"\033[1;0mChoices argument '{key}' not recognised. "
                    f"Valid choices are ['{'\', \''.join(possibilities)}'].\033[0m"
                ) 
        return choices_kwargs

    @Decorators.running_time
    def get_data(self) -> CubesData:
        """
        Opens the HDF5 file to get the necessary data for visualisation.
        """

        # DATA init
        HDF5File = h5py.File(self.filepath, 'r')
        cubes = CubesData(hdf5File=HDF5File)
        init_path = '' if not self.with_fake_data else 'Real/'

        # DATA main
        self.constants = self.get_default_data(HDF5File, init_path)
        self.radius_index = self.solar_r / self.constants.dx

        # CHOICES data
        if self.choices['all data']: 
            path = init_path + 'Filtered/All data' + self.feet
            cubes.all_data = self.get_cube_info(
                HDF5File=HDF5File,
                group_path=path,
                colour='red',
                opacity=0.6,
                interpolate=False,
            )

        if self.choices['no duplicate']: 
            path = init_path + 'Filtered/No duplicates' + self.feet
            cubes.no_duplicate = self.get_cube_info(
                HDF5File=HDF5File,
                group_path=path,
                colour='orange',
                opacity=0.8,
                interpolate=False,
            )

        if self.choices['all data integration']: 
            path = (
                init_path + 'Time integrated/All data' + self.feet +
                f'/Time integration of {round(float(self.time_interval), 1)} hours'
            )
            cubes.integration_all_data = self.get_cube_info(
                HDF5File=HDF5File,
                group_path=path,
                colour='red',
                opacity=0.4,
            )
        
        if self.choices['all data full integration']:
            path = (
                init_path + 'Time integrated/All data' + self.feet +
                '/Full integration'
            )
            cubes.full_integration_all_data = self.get_cube_info(
                HDF5File=HDF5File,
                group_path=path,
                colour='black',
                opacity=0.1,
                cube_type='unique',
            )

        if self.choices['no duplicate integration']:
            path = (
                init_path + 'Time integrated/No duplicates' + self.feet +
                f'/Time integration of {round(float(self.time_interval), 1)} hours'
            )
            cubes.integration_no_duplicate = self.get_cube_info(
                HDF5File=HDF5File,
                group_path=path,
                colour='red',
                opacity=0.5,
            )

        if self.choices['no duplicate full integration']:
            path = (
                init_path + 'Time integrated/No duplicates' + self.feet +
                '/Full integration'
            )
            cubes.full_integration_no_duplicate = self.get_cube_info(
                HDF5File=HDF5File,
                group_path=path,
                colour='grey',
                opacity=0.1,
                cube_type='unique',
            )

        if self.choices['line of sight sdo']:
            path = init_path + 'Filtered/SDO line of sight'
            cubes.los_sdo = self.get_cube_info(
                HDF5File=HDF5File,
                group_path=path,
                colour='blue',
                opacity=0.15,
                interpolate=False,
            )
        
        if self.choices['line of sight stereo']:
            path = init_path + 'Filtered/STEREO line of sight'
            cubes.los_stereo = self.get_cube_info(
                HDF5File=HDF5File,
                group_path=path,
                colour='blue',
                opacity=0.15,
                interpolate=False,
            )
        
        # POVs sdo, stereo
        if self.choices['pov sdo']:
            # SDO positions
            sdo_positions: np.ndarray = HDF5File['SDO positions'][...]  #type:ignore

            # DATA formatting
            cubes.sdo_pos = (
                sdo_positions[self.constants.time_indexes] / self.constants.dx
            ).astype('float32')
        if self.choices['pov stereo']:
            # STEREO positions
            stereo_positions: np.ndarray = HDF5File['STEREO B positions'][...]  #type:ignore

            # DATA formatting
            cubes.stereo_pos = (
                stereo_positions[self.constants.time_indexes] / self.constants.dx
            ).astype('float32')
            # todo will need to add the POV center

        if self.choices['fake data']:
            cubes.fake_cube = self.get_cube_info(
                HDF5File=HDF5File,
                group_path='Fake/Filtered/All data',
                colour='purple',
                opacity=0.6,
                interpolate=False,
                cube_type='fake',
            )

        if self.choices['test cube']:
            cubes.test_cube = self.get_cube_info(
                HDF5File=HDF5File,
                group_path='Test data/Sun surface',
                colour='yellow',
                opacity=0.7,
                interpolate=False,
                cube_type='unique',
            )
        return cubes

    def get_sdo_fov(self) -> float:
        """
        Gets the field of view of SDO.

        Returns:
            float: the field of view of SDO.
        """

        # FOV get
        hdul = fits.open(os.path.join(self.paths['sdo'], 'AIA_fullhead_000.fits.gz'))
        image_shape = hdul[0].data.shape
        fov_degrees = image_shape[0] * hdul[0].header['CDELT1'] / 3600
        hdul.close()
        return fov_degrees # ? in my old code I divide it by 3, no clue why
    
    def get_default_data(self, HDF5File: h5py.File, init_path: str) -> CubesConstants:
        """
        Gives the global information for the data.

        Args:
            HDF5File (h5py.File): the HDF5 file where the data is stored.
            init_path (str): the initial path to the data depending on which datafile is being
                used.

        Returns:
            CubesConstants: the global information of protuberance.
        """
        
        # DATA setup
        time_indexes: np.ndarray = HDF5File[init_path + 'Time indexes'][...]  #type:ignore
        dates_bytes: np.ndarray = HDF5File['Dates'][...]  #type:ignore
        dates_str = [dates_bytes[number].decode('utf-8') for number in time_indexes]  #type:ignore

        constants = CubesConstants(
            dx=float(HDF5File['dx'][...]),  #type:ignore
            time_indexes=time_indexes,
            dates=dates_str,
        )
        return constants
    
    @overload
    def get_cube_info(
            self,
            HDF5File: h5py.File,
            group_path: str,
            opacity: float = ...,
            interpolate: bool = ...,
            colour: str = ...,
            *,
            cube_type: Literal['real'] = ...,
        ) -> CubeInfo: ...

    @overload
    def get_cube_info(
            self,
            HDF5File: h5py.File,
            group_path: str,
            opacity: float = ...,
            interpolate: bool = ...,
            colour: str = ...,
            *,
            cube_type: Literal['fake'],
        ) -> FakeCubeInfo: ...

    @overload
    def get_cube_info(
            self,
            HDF5File: h5py.File,
            group_path: str,
            opacity: float = ...,
            interpolate: bool = ...,
            colour: str = ...,
            *,
            cube_type: Literal['unique'],
        ) -> UniqueCubeInfo: ...

    @overload # fallback
    def get_cube_info(
            self,
            HDF5File: h5py.File,
            group_path: str,
            opacity: float = ...,
            interpolate: bool = ...,
            colour: str = ...,
            cube_type: str = ...,
        ) -> CubeInfo | FakeCubeInfo | UniqueCubeInfo: ...

    def get_cube_info(
            self,
            HDF5File: h5py.File,
            group_path: str,
            opacity: float = 1.,
            interpolate: bool = True,
            colour: str = 'blue',
            cube_type: str = 'real',
        ) -> CubeInfo | FakeCubeInfo | UniqueCubeInfo:
        """
        Gives the protuberance and polynomial fit information for a chosen cube 'type'.

        Args:
            HDF5File (h5py.File): the HDF5 file where the data is stored.
            group_path (str): the group path to the group where the needed datasets are stored.
            opacity (float, optional): the opacity of the voxels in the visualisation.
                Defaults to 1..
            interpolate (bool, optional): if there is interpolation data in the group to visualise.
                Defaults to True.
            colour (str, optional): the colour of the voxels in the visualisation.
                Defaults to 'blue'.
            cube_type (str, optional): the type of cube to visualise. Defaults to 'real'.

        Returns:
            CubeInfo | FakeCubeInfo | UniqueCubeInfo: the protuberance and polynomial fit
                information.
        """

        # BORDERs as index
        xt_min_index = float(HDF5File[group_path + '/xt_min'][...]) / self.constants.dx#type:ignore
        yt_min_index = float(HDF5File[group_path + '/yt_min'][...]) / self.constants.dx#type:ignore
        zt_min_index = float(HDF5File[group_path + '/zt_min'][...]) / self.constants.dx#type:ignore

        # COO data
        data_coords = cast(h5py.Dataset, HDF5File[group_path + '/coords'])
        data_values = cast(h5py.Dataset, HDF5File[group_path + '/values'])

        # FORMATTING data
        if cube_type == 'real':
            # POLYNOMIAL data
            polynomials = self.get_polynomial_data(
                HDF5File=HDF5File,
                group_path=group_path,
                cube_type=cube_type,
                interpolate=interpolate,
            )

            cube_info = CubeInfo(
                group_path=group_path,
                colour=colour,
                opacity=opacity,
                xt_min_index=xt_min_index,
                yt_min_index=yt_min_index,
                zt_min_index=zt_min_index,
                dataset_coords=data_coords,
                dataset_values=data_values,
                polynomials=polynomials,
            )
        elif cube_type == 'unique':
            # POLYNOMIAL data
            polynomials = self.get_polynomial_data(
                HDF5File=HDF5File,
                group_path=group_path,
                cube_type=cube_type,
                interpolate=interpolate,
            )

            cube_info = UniqueCubeInfo(
                group_path=group_path,
                opacity=opacity,
                colour=colour,
                xt_min_index=xt_min_index,
                yt_min_index=yt_min_index,
                zt_min_index=zt_min_index,
                dataset_coords=data_coords,
                dataset_values=data_values,
                polynomials=polynomials,
            )
        else:
            cube_info = FakeCubeInfo(
                group_path=group_path,
                opacity=opacity,
                colour=colour,
                xt_min_index=xt_min_index,
                yt_min_index=yt_min_index,
                zt_min_index=zt_min_index,
                dataset_coords=data_coords,
                dataset_values=data_values,
                time_indexes_real=self.constants.time_indexes,
                time_indexes_fake=cast(h5py.Dataset, HDF5File['Fake/Time indexes'])[...],
            )
        print(f'FETCHED -- {cube_info.name} data.')
        return cube_info
    
    @overload
    def get_polynomial_data(
            self,
            HDF5File: h5py.File,
            group_path: str,
            cube_type: str = ...,
            *,
            interpolate: Literal[False],
        ) -> None: ...

    @overload
    def get_polynomial_data(
            self,
            HDF5File: h5py.File,
            group_path: str,
            cube_type: Literal['real'],
            interpolate: Literal[True] = ...,
        ) -> list[PolynomialData]: ...

    @overload
    def get_polynomial_data(
            self,
            HDF5File: h5py.File,
            group_path: str,
            cube_type: Literal['unique'] = ...,
            interpolate: Literal[True] = ...,
        ) -> list[UniquePolynomialData]: ...
    
    @overload # fallback 1
    def get_polynomial_data(
            self,
            HDF5File: h5py.File,
            group_path: str,
            cube_type: Literal['real'],
            interpolate: bool = ...,
        ) -> list[PolynomialData] | None: ...
    
    @overload # fallback 2
    def get_polynomial_data(
            self,
            HDF5File: h5py.File,
            group_path: str,
            cube_type: Literal['unique'] = ...,
            interpolate: bool = ...,
        ) -> list[UniquePolynomialData] | None: ...

    @overload  # fallback main
    def get_polynomial_data(
            self,
            HDF5File: h5py.File,
            group_path: str,
            cube_type: str = ...,
            interpolate: bool = ...,
        ) -> list[PolynomialData] | list[UniquePolynomialData] | None: ...
    
    def get_polynomial_data(
            self,
            HDF5File: h5py.File,
            group_path: str,
            cube_type: str = 'unique',
            interpolate: bool = True,
        ) -> list[PolynomialData] | list[UniquePolynomialData] | None:
        """
        To get the polynomial data corresponding to the HDF5 group path.

        Args:
            group_path (str): the main group path where all the different corresponding polynomial
                fits are stored.
            cube_type (str, optional): the type of cube to visualise. Defaults to 'unique'.
            interpolate (bool, optional): if the fit data exits for that group path.
                Defaults to True.

        Returns:
            list[PolynomialData] | list[UniquePolynomialData] | None: the corresponding polynomial
                fits information. If there is no polynomial data, it returns None.
        """

        if self.choices['fit'] and interpolate:
            # POLYNOMIALs setup
            polynomials: list[PolynomialData] | list[UniquePolynomialData] = (
                [None] * len(self.polynomial_order)
            )  #type:ignore

            for i, order in enumerate(self.polynomial_order):
                # DATA get
                dataset_path = group_path + f'/{order}th order polynomial/coords'
                interpolation_coords = cast(h5py.Dataset, HDF5File[dataset_path])

                if cube_type != 'unique':
                    # DATA formatting
                    polynomials[i] = PolynomialData(  # ? type check requirements are weird ...
                        dataset=interpolation_coords,
                        order=order,
                        name=self.name_data(dataset_path),
                        color_hex=self.plot_polynomial_colours[i],
                    )
                else:
                    # DATA formatting
                    polynomials[i] = UniquePolynomialData(
                        dataset=interpolation_coords,
                        order=order,
                        name=self.name_data(dataset_path),
                        color_hex=self.plot_polynomial_colours[i],
                    )
                print(f'FETCHED -- {polynomials[i].name}.')
            return polynomials
        else:
            return None

    def name_data(self, group_path: str) -> str:
        """
        To name the data depending on the group path.

        Args:
            group_path (str): the hdf5 group path to the data being named.

        Raises:
            ValueError: if the group path is not recognised.

        Returns:
            str: the name for the corresponding dataset.
        """

        # INIT name
        name = ''

        # FIT check
        if 'polynomial' in group_path:
            name += 'FIT'

            # PATTERN
            pattern_fit = r'(\d+)th'
            result = re.search(pattern_fit, group_path)

            if result is not None:
                name += f'({result.group(1)}th) '
            else:
                raise ValueError(
                    f"\033[1;31m The group path '{group_path}' pattern not recognised. \033[0m"
                )

        # GROUP check
        if 'All data' in group_path:
            name += 'all data'
        elif 'No duplicates' in group_path:
            name += 'no dupli'

        # INTEGRATION check
        if 'integration' in group_path:
            name += ' integration'
            
            if 'Full' in group_path:
                name += '(full) '
            else:
                # PATTERN
                pattern_integration_time = r'(\d+)\.\d*\s*hours'
                result = re.search(pattern_integration_time, group_path)

                if result is not None:
                    name += f'({result.group(1)}h) '
                else:
                    raise ValueError(
                        f"\033[1;31m The group path '{group_path}' pattern not recognised. \033[0m"
                    )
        return name
    
    def color_str_to_hex(self, colour: str) -> int:
        """
        Converts a colour string to hexadecimal int value.

        Args:
            colour (str): the colour name to convert.

        Returns:
            int: the corresponding hexadecimal int value.
        """

        return int(mcolors.to_hex(mcolors.CSS4_COLORS[colour])[1:], 16)

    def close(self) -> None:
        """
        To close the HDF5 file.
        """

        self.cubes.close()


class K3dAnimation(Setup):
    """
    Creates the corresponding k3d animation to then be used in a Jupyter notebook file.
    """

    @typeguard.typechecked
    def __init__(
            self,
            compression_level: int = 9,
            plot_height: int = 1260, 
            sleep_time: int | float = 2, 
            camera_zoom_speed: int | float = 0.7, 
            camera_pos: tuple[int | float, int | float, int | float] | None = None,
            camera_fov: float = 0.23,
            up_vector: tuple[int, int, int] = (0, 0, 1), 
            visible_grid: bool = False, 
            outlines: bool = False,
            texture_resolution: int = 960,  
            **kwargs,
        ) -> None:
        """
        To visualise the data in k3d. The fetching and naming of the data is done in the parent
        class.

        Args:
            compression_level (int, optional): the compression level used in k3d. The higher the
                value, the more compressed it is (max is 9). Defaults to 9.
            plot_height (int, optional): the height in pixels of the jupyter display plot.
                Defaults to 1260.
            sleep_time (int | float, optional): the sleep time used when playing through the cubes.
                Used when trying to save screenshots of the display. Defaults to 2.
            camera_zoom_speed (int | float, optional): the zoom speed when trying to zoom in on the
                display. Defaults to 0.7.
            camera_pos (tuple[int | float, int | float, int | float] | None, optional): the index
                position of the camera. Automatically set if 'pov_sdo' or 'pov_stereo' is chosen.
                Defaults to None.
            camera_fov (float, optional): the field of view, in degrees. Defaults to 0.23.
            up_vector (tuple[int, int, int], optional): the up vector when displaying the
                protuberance. Defaults to (0, 0, 1).
            visible_grid (bool, optional): deciding to make the k3d grid visible or not.
                Defaults to False.
            outlines (bool, optional): deciding to add the voxel outlines when displaying the data.
                Defaults to False.
            texture_resolution (int, optional): the resolution of the sun's texture, i.e. how many
                points need to be displayed phi direction. Defaults to 960.
            **kwargs: the other arguments to pass to the parent class.
        """
        
        # PARENT init
        super().__init__(**kwargs)

        # ATTRIBUTEs
        self.plot_height = plot_height  # the height in pixels of the plot (initially it was 512)
        self.sleep_time = sleep_time  # sets the time between each frames (in seconds)
        self.camera_zoom_speed = camera_zoom_speed  # zoom speed of the camera 
        self.camera_pos = camera_pos  # position of the camera multiplied by 1au
        self.camera_fov = self.get_sdo_fov() if self.choices['pov sdo'] else camera_fov  # in deg
        self.up_vector = up_vector  # up vector for the camera
        self.visible_grid = visible_grid  # setting the grid to be visible or not
        self.texture_resolution = texture_resolution
        self.compression_level = compression_level
        self.outlines = outlines

        # PLACEHOLDERs
        self.plot: k3d.plot.Plot  # plot object
        self.plot_alldata: list[VoxelType] # voxels plot of the all data 
        self.plot_dupli_new: list[VoxelType]  # same for the second method
        self.plot_full_alldata: list[VoxelType]  # voxels plot for the full integration
        self.plot_full_no_duplicate: list[VoxelType]  # same for the no duplicate data
        self.plot_interv_new: list[VoxelType]  # same for the second method
        self.plot_interv_dupli_new: list[VoxelType]  # same for the second method
        self.plot_los_sdo: list[VoxelType]
        self.plot_los_stereo: list[VoxelType]
        self.plot_fake_cube: list[VoxelType]
        self.play_pause_button: ipywidgets.ToggleButton  # Play/Pause widget initialisation
        self.time_slider: ipywidgets.IntSlider # time slider widget
        self.date_dropdown: ipywidgets.Dropdown  # Date dropdown widget to show the date
        self.time_link: JsLinkType  # JavaScript Link between the two widgets

        # RUN
        self.animation()

    def animation(self) -> None:
        """
        Creates the 3D animation using k3d. 
        """

        # * also used [0x0000ff] as color_map
        
        # DISPLAY setup
        self.plot = k3d.plot(grid_visible=self.visible_grid)  # black: background_color=0x000000
        self.plot.height = self.plot_height  
            
        # CAMERA params
        self.camera_params()

        # DEFAULT params
        kwargs = {
            'compression_level': self.compression_level,
            'outlines': self.outlines,
        }

        # SUN add
        if self.choices['sun']:
            self.add_sun()
            points = k3d.points(
                positions=self.sun_points,
                point_size=0.7,
                colors=[0xffff00] * len(self.sun_points),
                shader='flat',
                name='SUN',
                compression_level=self.compression_level,
            )
            self.plot += points

        if self.choices['test points']:
            cube = self.find_first_cube()
            if cube is not None:
                point = k3d.points(
                    positions=np.array([cube.xt_min_index, cube.yt_min_index, cube.zt_min_index]),
                    point_size=5,
                    colors=[0xff0000],
                    shader='3d',
                    name='BORDERs',
                )
                point2 = k3d.points(
                    positions=np.array([
                        cube.xt_min_index + 62,
                        cube.yt_min_index + 65,
                        cube.zt_min_index + 150,
                    ]),
                    point_size=5,
                    colors=[0xff0000],
                    shader='3d',
                    name='BORDERs2',
                )
                self.plot += point
                self.plot += point2

        # TEST CUBE add
        if self.cubes.test_cube is not None:
            # VOXELs create
            self.plot_test_cube = self.create_voxels(self.cubes.test_cube, **kwargs)
            for plot in self.plot_test_cube: self.plot += plot

        # ALL DATA add
        if self.cubes.all_data is not None:
            # VOXELS create
            self.plot_alldata = self.create_voxels(self.cubes.all_data, **kwargs)
            for plot in self.plot_alldata: self.plot += plot     
        
        # NO DUPLICATES add
        if self.cubes.no_duplicate is not None:
            # VOXELs create
            self.plot_dupli_new = self.create_voxels(self.cubes.no_duplicate, **kwargs)
            for plot in self.plot_dupli_new: self.plot += plot

        # TIME INTEGRATION add
        if self.cubes.integration_all_data is not None:
            # VOXELs create
            self.plot_interv_new = self.create_voxels(self.cubes.integration_all_data, **kwargs)
            for plot in self.plot_interv_new: self.plot += plot
    
        # TIME NO DUPLICATES add       
        if self.cubes.integration_no_duplicate is not None:
            # VOXELs create
            self.plot_interv_dupli_new = self.create_voxels(
                self.cubes.integration_no_duplicate,
                **kwargs,
            )
            for plot in self.plot_interv_dupli_new: self.plot += plot  
        
        # FULL INTEGRATION add
        if self.cubes.full_integration_all_data is not None:
            # VOXELs create
            self.plot_full_alldata = self.create_voxels(
                self.cubes.full_integration_all_data,
                **kwargs,
            )
            for plot in self.plot_full_alldata: self.plot += plot

        # FULL NO DUPLICATES add
        if self.cubes.full_integration_no_duplicate is not None:
            # VOXELs create
            self.plot_full_no_duplicate = self.create_voxels(
                self.cubes.full_integration_no_duplicate,
                **kwargs,
            )
            for plot in self.plot_full_no_duplicate: self.plot += plot

        # SDO LINE OF SIGHT add
        if self.cubes.los_sdo is not None:
            # VOXELs create
            self.plot_los_sdo = self.create_voxels(self.cubes.los_sdo, **kwargs)
            for plot in self.plot_los_sdo: self.plot += plot

        # STEREO LINE OF SIGHT add
        if self.cubes.los_stereo is not None:
            # VOXELs create
            self.plot_los_stereo = self.create_voxels(self.cubes.los_stereo, **kwargs)
            for plot in self.plot_los_stereo: self.plot += plot
        
        # CUBE fake
        if self.cubes.fake_cube is not None:
            # VOXELs create
            self.plot_fake_cube = self.create_voxels(self.cubes.fake_cube, **kwargs)
            for plot in self.plot_fake_cube: self.plot += plot

        # BUTTON play/pause
        self.play_pause_button = ipywidgets.ToggleButton(
            value=False,
            description='Play',
            icon='play',
        )

        # SETUP time-slider and play/pause
        self.time_slider = ipywidgets.IntSlider(
            min=0,
            max=len(self.constants.dates)-1,
            description='Frame:',
        )
        self.date_dropdown = ipywidgets.Dropdown(
            options=self.constants.dates,
            description='Date:',
        )
        self.time_slider.observe(self.update_plot, names='value')
        self.time_link = ipywidgets.jslink(
            (self.time_slider,'value'),
            (self.date_dropdown, 'index'),
        )
        self.play_pause_button.observe(self.play_pause_handler, names='value')

        # DISPLAY
        IPython.display.display(
            self.plot,
            self.time_slider,
            self.date_dropdown,
            self.play_pause_button,
        )

    def create_voxels(
            self,
            cube: CubeInfo | FakeCubeInfo | UniqueCubeInfo,
            index: int = 0,
            **kwargs,
        ) -> list[VoxelType]:
        """
        Creates the initial k3d voxels and the corresponding polynomial fit voxels for the
        visualisation.  

        Args:
            cube (CubeInfo | FakeCubeInfo | UniqueCubeInfo): the cube information to visualise.
            index (int, optional): the index of the time value to visualise. Defaults to 0.

        Returns:
            list[VoxelType]: the corresponding voxels for the k3d visualisation.
        """

        # PLACEHOLDER voxels
        plots: list[VoxelType] = [None] * (
            1 + (len(cube.polynomials) if cube.polynomials is not None else 0)
        )

        # INDEX translation
        translation = (cube.xt_min_index, cube.yt_min_index, cube.zt_min_index)

        plots[0] = k3d.voxels(
            voxels=cube[index][0].transpose((2, 1, 0)),
            name=cube.name,
            opacity=cube.opacity,
            color_map=[self.color_str_to_hex(cube.colour)],
            translation=translation,
            **kwargs,
        )
        
        if cube.polynomials is not None:
            # INTERPOLATIONs setup
            polynomials = cube.polynomials

            # INTERPOLATION orders
            for i, polynomial_data in enumerate(polynomials):
                plots[i + 1] = k3d.voxels(
                    voxels=polynomial_data[index].transpose((2, 1, 0)),
                    name=polynomial_data.name,
                    opacity=cube.opacity,  # todo change it to polynomial opacity
                    color_map=[polynomial_data.color_hex],
                    translation=translation,
                    **kwargs,
                )
        return plots

    def get_camera_reference(self) -> np.ndarray:
        """
        Gets the camera reference point.

        Returns:
            np.ndarray: the camera reference point.
        """

        # REFERENCE point
        cube = self.find_first_cube()

        if cube is not None:
            data = cube[0]
            reference = np.array([
                (data[0].shape[0] / 2) + cube.xt_min_index,
                (data[0].shape[1] / 2) + cube.yt_min_index,
                (data[0].shape[2] / 2) + cube.zt_min_index,
            ], dtype='float16')
        else:
            reference = np.array([0, 0, 0], dtype='float32')
        return reference
    
    def find_first_cube(self) -> CubeInfo | None:
        """
        Finds the first cube that has data in it.

        Returns:
            CubeInfo | None: the first cube that has data in it. None if no data found.
        """
        
        # FIND cube
        for attr_name in self.cubes.__slots__:
            if attr_name in ['hdf5File', 'sdo_pos', 'stereo_pos']: continue
            if getattr(self.cubes, attr_name) is not None: return getattr(self.cubes, attr_name)

        # NO DATA
        return None

    def camera_params(self) -> None:
        """
        Camera visualisation parameters.
        """
        
        # PARAMs constant
        self.plot.camera_auto_fit = False
        self.plot.camera_fov = self.camera_fov  # FOV in degrees
        self.plot.camera_zoom_speed = self.camera_zoom_speed  # zooming too quickly (default=1.2)

        self._camera_reference = self.get_camera_reference()
        
        # POV stereo
        if self.cubes.stereo_pos is not None:
            self.plot.camera = [
                self.cubes.stereo_pos[0, 0],
                self.cubes.stereo_pos[0, 1],
                self.cubes.stereo_pos[0, 2],
                self._camera_reference[0],
                self._camera_reference[1],
                self._camera_reference[2],
                self.up_vector[0],
                self.up_vector[1],
                self.up_vector[2],
            ] 
        # POV sdo
        elif self.cubes.sdo_pos is not None:
            self.plot.camera = [
                self.cubes.sdo_pos[0, 0],
                self.cubes.sdo_pos[0, 1],
                self.cubes.sdo_pos[0, 2],
                self._camera_reference[0],
                self._camera_reference[1],
                self._camera_reference[2],
                self.up_vector[0],
                self.up_vector[1],
                self.up_vector[2]
            ]  
        else:
            au_in_solar_r = 215  # 1 au in solar radii
            distance_to_sun = au_in_solar_r * self.radius_index 

            if not self.camera_pos:
                print("No 'camera_pos', setting default values.")
                self.camera_pos = (- distance_to_sun, -0.5 * distance_to_sun, 0)

            self.plot.camera = [
                self._camera_reference[0] + self.camera_pos[0], # ! there is an overflow here
                self._camera_reference[1] + self.camera_pos[1],
                self._camera_reference[2] + self.camera_pos[2],
                self._camera_reference[0],
                self._camera_reference[1],
                self._camera_reference[2],
                self.up_vector[0],
                self.up_vector[1],
                self.up_vector[2]
            ] 

    def add_sun(self):
        """
        Add the Sun in the visualisation.
        The Sun is just made up of small spheres positioned at the Sun's surface.
        """
        # todo re-add the choice where I can decide to add a texture.

        # COORDs spherical
        N = self.texture_resolution  # number of points in the theta direction
        phi = np.linspace(0, np.pi, N)  # latitude of the points
        theta = np.linspace(0, 2 * np.pi, 2 * N)  # longitude of the points
        phi, theta = np.meshgrid(phi, theta)  # the subsequent meshgrid

        # COORDs cartesian
        x = self.radius_index * np.sin(phi) * np.cos(theta)
        y = self.radius_index * np.sin(phi) * np.sin(theta)
        z = self.radius_index * np.cos(phi)

        # SAVE coords
        self.sun_points = np.array([x.ravel(), y.ravel(), z.ravel()], dtype='float32').T

    def update_plot(self, change: dict[str, Any]) -> None:
        """
        Updates the voxels depending on which time value is chosen.

        Args:
            change (dict[str, Any]): the time value, with the 'new' key being the new value and
                'old' key being the old value.
        """

        # POV stereo
        if self.cubes.stereo_pos is not None:
            self.plot.camera = [
                self.cubes.stereo_pos[change['new'], 0],
                self.cubes.stereo_pos[change['new'], 1],
                self.cubes.stereo_pos[change['new'], 2],
                self._camera_reference[0],
                self._camera_reference[1],
                self._camera_reference[2],
                self.up_vector[0],
                self.up_vector[1],
                self.up_vector[2],
            ]
            time.sleep(0.2) 

        # POV sdo
        elif self.cubes.sdo_pos is not None:
            self.plot.camera = [
                self.cubes.sdo_pos[change['new'], 0],
                self.cubes.sdo_pos[change['new'], 1],
                self.cubes.sdo_pos[change['new'], 2],
                self._camera_reference[0],
                self._camera_reference[1],
                self._camera_reference[2],
                self.up_vector[0],
                self.up_vector[1],
                self.up_vector[2],
            ]
            time.sleep(0.2)
        
        # ALL DATA
        if self.cubes.all_data is not None:
            self.update_voxel(self.plot_alldata, self.cubes.all_data, change['new'])
        # NO DUPLICATES
        if self.cubes.no_duplicate is not None:
            self.update_voxel(self.plot_dupli_new, self.cubes.no_duplicate, change['new'])
        # TIME INTEGRATION
        if self.cubes.integration_all_data is not None:
            self.update_voxel(self.plot_interv_new, self.cubes.integration_all_data, change['new'])
        # TIME NO DUPLICATES
        if self.cubes.integration_no_duplicate is not None:     
            self.update_voxel(
                self.plot_interv_dupli_new, self.cubes.integration_no_duplicate, change['new'],
            )        
        # SDO LINE OF SIGHT
        if self.cubes.los_sdo is not None:
            self.update_voxel(self.plot_los_sdo, self.cubes.los_sdo, change['new'])
        # STEREO LINE OF SIGHT
        if self.cubes.los_stereo is not None:
            self.update_voxel(self.plot_los_stereo, self.cubes.los_stereo, change['new'])
        # FAKE CUBE
        if self.cubes.fake_cube is not None:
            self.update_voxel(self.plot_fake_cube, self.cubes.fake_cube, change['new'])

    def update_voxel(
            self,
            plots: list[VoxelType],
            cube_info: CubeInfo | FakeCubeInfo,
            index: int,
        ) -> None:
        """
        Updates the k3d plot voxels for each cube and the corresponding polynomials.

        Args:
            plots (list[VoxelType]): the different voxel plots for each cube type.
            cube_info (CubeInfo | FakeCubeInfo): the data cubes. None if it doesn't exist.
            index (int): the index to plot.
        """

        # DATA get
        cubes = cube_info[index]
        
        # PLOTs add data
        for i, plot in enumerate(plots): plot.voxels = cubes[i].transpose((2, 1, 0))

    def play(self) -> None:
        """
        Params for the play button.
        """
        
        if self.play_pause_button.value and self.time_slider.value < len(self.constants.dates) - 1:
            self.time_slider.value += 1
            threading.Timer(self.sleep_time, self.play).start()
            # where you also set the sleep() time.
                
        else:
            self.play_pause_button.description = 'Play'
            self.play_pause_button.icon = 'play'

    def play_pause_handler(self, change: dict[str, Any]) -> None:
        """
        Changes the play button to pause when it is clicked.

        Args:
            change (dict[str, Any]): the dictionary representing the value.
        """

        if change['new']:  # if clicked play
            self.play()
            self.play_pause_button.description = 'Pause'
            self.play_pause_button.icon = 'pause'
