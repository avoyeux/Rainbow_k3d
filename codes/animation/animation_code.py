"""
To 3D visualise the rainbow filament data and the corresponding polynomial fitting.
"""

# IMPORTs
import os
import k3d
import time
import h5py
import yaml
import IPython
import threading
import typeguard
import ipywidgets

# IMPORTs alias
import numpy as np

# IMPORTs sub
from typing import Any, overload, Literal
from astropy.io import fits

# IMPORTs personal
from animation.animation_dataclasses import *
from common import Decorators, Plot, root_path, DictToObject

# CONFIGURATION load
with open(os.path.join(root_path, 'config.yml'), 'r') as conf:
    config = DictToObject(yaml.safe_load(conf))

# ANNOTATION alias
VoxelType = Any
JsLinkType = Any



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
            sun: bool = False,
            with_feet: bool = True,
            all_data: bool = False,
            no_duplicate: bool = False,
            time_interval: int = 24,
            all_data_integration: bool = False,
            no_duplicate_integration: bool = False,
            line_of_sight_SDO: bool = False,
            line_of_sight_STEREO: bool = False,
            pov_sdo: bool = False,
            pov_stereo: bool = False,
            processes: int | None = None,
            polynomial: bool = False,
            polynomial_order: int | list[int] = [4],
            with_fake_data: bool = False,
            test_points: bool = False, # todo will take it out when Dr. Auchere has already seen it
    ) -> None:
        """ # todo update docstring
        To setup the instance attributes needed for the 3D visualisation of the Rainbow filament
        data.

        Args:
            filename (str, optional): the HDF5 filename containing all the cube data.
                Defaults to 'order0321.h5'.
            sun (bool, optional): choosing to add the sun visualisation. Defaults to False.
            with_feet (bool, optional): choosing the data sets that have the feet manually added.
                Defaults to True.
            all_data (bool, optional): choosing to visualise the data that also contains the
                duplicates. Defaults to False.
            no_duplicate (bool, optional): choosing to visualise the data with no duplicate at
                all. Defaults to False.
            time_interval (int, optional): the time interval for the time integration (in hours).
                Defaults to 24.
            all_data_integration (bool, optional): choosing to visualise the time integrated data
                with duplicates. Defaults to False.
            no_duplicate_integration (bool, optional): choosing to visualise the time integrated
                data without any duplicates. Defaults to False.
            line_of_sight_SDO (bool, optional): deciding to visualise the line of sight data from
                SDO's position. Defaults to False.
            line_of_sight_STEREO (bool, optional): deciding to visualise the line of sight data
                from STEREO's position. Defaults to False.
            pov_sdo (bool, optional): choosing to take SDO's point of view when looking at the
                data. Defaults to False.
            pov_stereo (bool, optional): choosing to take STEREO B's point of view when looking at
                the data. Defaults to False.
            processes (int, optional): the number of processes used in the multiprocessing.
                Defaults to 5.
            polynomial (bool, optional): choosing to visualise the polynomial data fits.
                Defaults to False.
            polynomial_order (int | list[int], optional): the polynomial orders that you want to
                visualise (if polynomial is set to True). Defaults to [5].
        """

        # todo need to make the __init__ cleaner.
        # FEET
        self.feet = ' with feet' if with_feet else ''
        
        # ATTRIBUTES
        self.solar_r = 6.96e5
        self.sun = sun
        self.all_data = all_data
        self.no_duplicate = no_duplicate
        self.time_interval = time_interval
        self.all_data_integration = all_data_integration
        self.no_duplicate_integration = no_duplicate_integration
        self.line_of_sight_SDO = line_of_sight_SDO
        self.line_of_sight_STEREO = line_of_sight_STEREO
        self.pov_sdo = pov_sdo
        self.pov_stereo = pov_stereo
        self.polynomial = polynomial
        if isinstance(polynomial_order, list):
            self.polynomial_order = polynomial_order
        else:
            self.polynomial_order = [polynomial_order]

        self.with_fake_data = with_fake_data

        # DATA filepath
        if filepath is None and with_fake_data:
            self.filepath = os.path.join(root_path, *config.paths.data.fusion.split('/'))
        elif filepath is None:
            self.filepath = os.path.join(root_path, *config.paths.data.real.split('/'))
        else:
            self.filepath = filepath

        self.processes = config.processes if processes is None else processes
        self.test_points = test_points

        # ATTRIBUTES new
        self.plot_polynomial_colours = [
            next(Plot.random_hexadecimal_int_color_generator())
            for _ in self.polynomial_order
        ]

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
        paths = {
            'code': root_path,
            'sdo': os.path.join(root_path, '..', 'sdo'), # ? should I add it to config ?
        }

        # PATHs update
        return paths

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
        if self.all_data: 
            path = init_path + 'Filtered/All data' + self.feet
            cubes.all_data = self.get_cube_info(HDF5File, path, interpolate=False)

        if self.no_duplicate: 
            path = init_path + 'Filtered/No duplicates' + self.feet
            cubes.no_duplicate = self.get_cube_info(HDF5File, path, interpolate=False)

        if self.all_data_integration: 
            path = (
                init_path + 'Time integrated/All data' + self.feet +
                f'/Time integration of {round(float(self.time_interval), 1)} hours'
            )
            cubes.integration_all_data = self.get_cube_info(HDF5File, path, opacity=0.4)

        if self.no_duplicate_integration:
            path = (
                init_path + 'Time integrated/No duplicates' + self.feet +
                f'/Time integration of {round(float(self.time_interval), 1)} hours'
            )
            cubes.integration_no_duplicate = self.get_cube_info(HDF5File, path, opacity=0.6)

        if self.line_of_sight_SDO:
            path = init_path + 'Filtered/SDO line of sight'
            cubes.los_sdo = self.get_cube_info(HDF5File, path, opacity=0.15, interpolate=False)
        
        if self.line_of_sight_STEREO:
            path = init_path + 'Filtered/STEREO line of sight'
            cubes.los_stereo = self.get_cube_info(HDF5File, path, opacity=0.15, interpolate=False)
        
        # POVs sdo, stereo
        if self.pov_sdo:
            # SDO positions
            sdo_positions: np.ndarray = HDF5File['SDO positions'][...]

            # DATA formatting
            cubes.sdo_pos = (
                sdo_positions[self.constants.time_indexes] / self.constants.dx
            ).astype('float32')
        if self.pov_stereo:
            # STEREO positions
            stereo_positions: np.ndarray = HDF5File['STEREO B positions'][...]

            # DATA formatting
            cubes.stereo_pos = (
                stereo_positions[self.constants.time_indexes] / self.constants.dx
            ).astype('float32')
            #TODO: will need to add the POV center

        if self.with_fake_data:
            cubes.fake_cube = self.get_cube_info(
                HDF5File=HDF5File,
                group_path='Fake/Filtered/All data',
                interpolate=False,
                fake_cubes=True,
            )

        # ADD shape
        cubes.add_shape()  # * had to as there seems to be a bug in the k3d module
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
        """ # todo update docstring
        Gives the global information for the data.

        Args:
            HDF5File (h5py.File): the HDF5 file where the data is stored.

        Returns:
            CubesConstants: the global information of protuberance.
        """
        
        # DATA setup
        time_indexes: np.ndarray = HDF5File[init_path + 'Time indexes'][...]
        dates_bytes: np.ndarray = HDF5File['Dates'][...]
        dates_str = [dates_bytes[number].decode('utf-8') for number in time_indexes]

        constants = CubesConstants(
            dx=float(HDF5File['dx'][...]),
            time_indexes=time_indexes,
            dates=dates_str,
        )
        return constants
    
    @overload  # ? add another when fake_cube is not given ? VSCode seems to understand though
    def get_cube_info(
            self,
            HDF5File: h5py.File,
            group_path: str,
            opacity: float = ...,
            interpolate: bool = ...,
            *,
            fake_cubes: Literal[False] = ...,
        ) -> CubeInfo: ...

    @overload
    def get_cube_info(
            self,
            HDF5File: h5py.File,
            group_path: str,
            opacity: float = ...,
            interpolate: bool = ...,
            *,
            fake_cubes: Literal[True] = ...,
        ) -> FakeCubeInfo: ...

    def get_cube_info(
            self,
            HDF5File: h5py.File,
            group_path: str,
            opacity: float = 1.,
            interpolate: bool = True,
            fake_cubes: bool = False,
        ) -> CubeInfo | FakeCubeInfo:
        """ # todo update docstring
        Gives the protuberance and polynomial fit information for a chosen cube 'type'.

        Args:
            HDF5File (h5py.File): the HDF5 file where the data is stored.
            group_path (str): the HDF5 group absolute path (i.e. represents the cube 'type').
            interpolate (bool, optional): If there exists polynomial data for that cube 'type'.
                Defaults to True.

        Returns:
            CubeInfo: the chosen cube 'type' information with the corresponding polynomials'.
        """

        # BORDERs as index
        xt_min_index = float(HDF5File[group_path + '/xt_min'][...]) / self.constants.dx
        yt_min_index = float(HDF5File[group_path + '/yt_min'][...]) / self.constants.dx
        zt_min_index = float(HDF5File[group_path + '/zt_min'][...]) / self.constants.dx

        # COO data
        data_coords: h5py.Dataset = HDF5File[group_path + '/coords']
        data_values: h5py.Dataset = HDF5File[group_path + '/values']

        # INTERPOLATION data
        if self.polynomial and interpolate:
            polynomials: list[PolynomialData] = [None] * len(self.polynomial_order)

            for i, order in enumerate(self.polynomial_order):
                # DATA get
                dataset_path = group_path + f'/{order}th order polynomial/coords'
                interp_coords: h5py.Dataset = HDF5File[dataset_path]

                # DATA formatting
                polynomials[i] = PolynomialData(
                    dataset=interp_coords,
                    order=order,
                    name=f'{order}th ' + group_path.split('/')[1],
                    color_hex=self.plot_polynomial_colours[i],
                )
                print(f'FETCHED -- {polynomials[i].name} data.')
        else:
            polynomials = None

        if not fake_cubes:
            # FORMATTING data
            cube_info = CubeInfo(
                group_path=group_path,
                opacity=opacity,
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
                xt_min_index=xt_min_index,
                yt_min_index=yt_min_index,
                zt_min_index=zt_min_index,
                dataset_coords=data_coords,
                dataset_values=data_values,
                time_indexes_real=self.constants.time_indexes,
                time_indexes_fake=HDF5File['Fake/Time indexes'][...],
            )
        
        print(f'FETCHED -- {cube_info.name} data.')
        return cube_info

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
        """ # todo update docstring
        To visualise the data in k3d. The fetching and naming of the data is done in the parent
        class.

        Args:
            compression_level (int, optional): the compression used in k3d. The higher the value,
                the more compressed it is (max is 9). Defaults to 9.
            plot_height (int, optional): the hight in pixels of the jupyter display plot.
                Defaults to 1260.
            sleep_time (int | float, optional): the sleep time used when playing through the cubes.
                Used when trying to save screenshots of the display. Defaults to 2.
            camera_fov (int | float | str, optional): the field of view, in degrees. Can be also a
                string if the field of view needs to represent 'sdo' or 'stereo'. Defaults to 0.23.
            camera_zoom_speed (int | float, optional): the zoom speed when trying to zoom in on the
                display. Defaults to 0.7.
            camera_pos (tuple[int | float, int | float, int | float] | None, optional): the
                index position of the camera. Automatically set if 'pov_sdo' or 'pov_stereo' is set
                to True. Defaults to None.
            up_vector (tuple[int, int, int], optional): the up vector when displaying the
                protuberance. Defaults to (0, 0, 1).
            visible_grid (bool, optional): deciding to make the k3d grid visible or not.
                Defaults to False.
            outlines (bool, optional): deciding to add the voxel outlines when displaying the data.
                Defaults to False.
            texture_resolution (int, optional): the resolution of the sun's texture, i.e. how many
                points need to be displayed phi direction. Defaults to 960.

        Raises:
            ValueError: if 'camera_fov' string value is not recognised.
        """
        
        # PARENT init
        super().__init__(**kwargs)

        # ATTRIBUTEs
        self.plot_height = plot_height  # the height in pixels of the plot (initially it was 512)
        self.sleep_time = sleep_time  # sets the time between each frames (in seconds)
        self.camera_zoom_speed = camera_zoom_speed  # zoom speed of the camera 
        self.camera_pos = camera_pos  # position of the camera multiplied by 1au
        self.camera_fov = self.get_sdo_fov() if self.pov_sdo else camera_fov  # fov in degrees
        self.up_vector = up_vector  # up vector for the camera
        self.visible_grid = visible_grid  # setting the grid to be visible or not
        self.texture_resolution = texture_resolution
        self.compression_level = compression_level
        self.outlines = outlines

        # PLACEHOLDERs
        self.plot: k3d.plot.Plot  # plot object
        self.plot_alldata: list[VoxelType] # voxels plot of the all data 
        self.plot_dupli_new: list[VoxelType]  # same for the second method
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
        if self.sun:
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

        # BORDERs add

        if self.test_points:
            cube = self.find_first_cube()
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
            cube: CubeInfo | FakeCubeInfo,
            index: int = 0,
            color_map: list[int] = [0x0000ff],
            **kwargs,
        ) -> list[VoxelType]:
        """
        Creates the k3d.plot.voxels() for a given cube 'type'.

        Args:
            cube (CubeInfo | FakeCubeInfo): the cube and polynomial fit information.
            index (int, optional): the chosen index to initially plot. Defaults to 0.
            opacity (float, optional): the voxel opacity in the plot. Defaults to 0.8.
            color_map (list[int], optional): the color chosen for the voxels.
                Defaults to [0x0000ff].

        Returns:
            list[VoxelType]: the k3d.plot.voxels() created.
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
            color_map=color_map,
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
        # TODO: re-add the choice where I can decide to add a texture.

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
