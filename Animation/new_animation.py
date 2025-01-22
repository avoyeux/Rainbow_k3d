"""
To 3D visualise the rainbow filament data and the corresponding polynomial fitting.
"""

# IMPORTs
import os
import k3d
import time
import h5py
import sparse
import IPython
import threading
import typeguard
import ipywidgets

# IMPORTs alias
import numpy as np

# IMPORTs sub
from typing import Any

# IMPORTs personal
from common import Decorators, main_paths, Plot
from Animation.animation_dataclasses import *

# ANNOTATION alias
VoxelType = Any



class Setup:
    """
    Manipulates the HDF5 filament data file to setup the necessary data choices for the
    visualisation.
    This class is the parent class to the k3d visualisation class named K3dAnimation.
    """

    @typeguard.typechecked
    def __init__(
            self,
            filename: str = 'sig1e20_leg20_lim0_03.h5',
            sun: bool = False,
            with_feet: bool = True,
            all_data: bool = False,
            no_duplicates: bool = False,
            time_interval: int = 24,
            time_interval_all_data: bool = False,
            time_interval_no_duplicates: bool = False,
            line_of_sight_SDO: bool = False,
            line_of_sight_STEREO: bool = False,
            sdo_pov: bool = False,
            stereo_pov: bool = False,
            processes: int = 5,
            interpolation: bool = False,
            interpolation_order: int | list[int] = [5],
            test_data: bool = False,
            test_filename: str | None = None,
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
            no_duplicates (bool, optional): choosing to visualise the data with no duplicates at
                all. Defaults to False.
            time_interval (int, optional): the time interval for the time integration (in hours).
                Defaults to 24.
            time_interval_all_data (bool, optional): choosing to visualise the time integrated data
                with duplicates. Defaults to False.
            time_interval_no_duplicates (bool, optional): choosing to visualise the time integrated
                data without any duplicates. Defaults to False.
            line_of_sight_SDO (bool, optional): deciding to visualise the line of sight data from
                SDO's position. Defaults to False.
            line_of_sight_STEREO (bool, optional): deciding to visualise the line of sight data
                from STEREO's position. Defaults to False.
            sdo_pov (bool, optional): choosing to take SDO's point of view when looking at the
                data. Defaults to False.
            stereo_pov (bool, optional): choosing to take STEREO B's point of view when looking at
                the data. Defaults to False.
            processes (int, optional): the number of processes used in the multiprocessing.
                Defaults to 5.
            interpolation (bool, optional): choosing to visualise the polynomial data fits.
                Defaults to False.
            interpolation_order (int | list[int], optional): the polynomial orders that you want to
                visualise (if interpolation is set to True). Defaults to [5].
        """

        # FEET
        self.feet = ' with feet' if with_feet else ''
        
        # ATTRIBUTES
        self.solar_r = 6.96e5
        self.filename = filename
        self.sun = sun
        self.all_data = all_data
        self.no_duplicates = no_duplicates
        self.time_interval = time_interval
        self.time_interval_all_data = time_interval_all_data
        self.time_interval_no_duplicates = time_interval_no_duplicates
        self.line_of_sight_SDO = line_of_sight_SDO
        self.line_of_sight_STEREO = line_of_sight_STEREO
        self.sdo_pov = sdo_pov
        self.stereo_pov = stereo_pov
        self.processes = processes
        self.interpolation = interpolation
        if isinstance(interpolation, list):
            self.interpolation_data = interpolation_order
        else:
            self.interpolation_order = [interpolation_order]  # ? what is the type annotation doing
        self.test_data = test_data
        self.test_filename = test_filename

        # ATTRIBUTES new
        self.plot_interpolation_colours = [
            next(Plot.random_hexadecimal_int_color_generator())
            for _ in self.interpolation_order
        ]

        # PLACEHOLDERs
        self.radius_index: float

        # RUN
        self.paths = self.setup_paths()
        self.cubes: CubesData = self.get_data()  # ? what is happening with the IDE???

    def setup_paths(self) -> dict[str, str]:
        """
        Creates a dictionary for the filepaths.

        Returns:
            dict[str, str]: the filepath dictionary.
        """

        # PATHs setup
        codes_path = main_paths.root_path

        # PATHs save
        paths = {
            'codes': codes_path,
            'data': os.path.join(codes_path, 'Data'),
            'fake data': os.path.join(codes_path, 'Data', 'fake_data'),
            # todo need to add the option where the test data is in the real data file
        }
        return paths

    @Decorators.running_time
    def get_data(self) -> CubesData:
        """
        Opens the HDF5 file to get the necessary data for visualisation.
        """

        # DATA init
        cubes = CubesData()
        with h5py.File(os.path.join(self.paths['data'], self.filename), 'r') as HDF5File:

            # DATA main
            self.constants = self.get_default_data(HDF5File)
            self.radius_index = self.solar_r / self.constants.dx

            # CHOICES data
            if self.all_data: 
                path = 'Filtered/All data' + self.feet
                cubes.all_data = self.get_cube_info(HDF5File, path, interpolate=False)

            if self.no_duplicates: 
                path = 'Filtered/No duplicates new' + self.feet
                cubes.no_duplicate = self.get_cube_info(HDF5File, path, interpolate=False)

            if self.time_interval_all_data: 
                path = (
                    'Time integrated/All data' + self.feet +
                    f'/Time integration of {round(float(self.time_interval), 1)} hours'
                )
                cubes.integration_all_data = self.get_cube_info(HDF5File, path)

            if self.time_interval_no_duplicates:
                path = (
                    'Time integrated/No duplicates new' + self.feet +
                    f'/Time integration of {round(float(self.time_interval), 1)} hours'
                )
                cubes.integration_no_duplicate = self.get_cube_info(HDF5File, path)

            if self.line_of_sight_SDO:
                path = ('Filtered/SDO line of sight')
                cubes.los_sdo = self.get_cube_info(HDF5File, path, interpolate=False)
            
            if self.line_of_sight_STEREO:
                path = ('Filtered/STEREO line of sight')
                cubes.los_stereo = self.get_cube_info(HDF5File, path, interpolate=False)
            
            # POVs sdo, stereo
            if self.sdo_pov:
                cubes.sdo_pos = (
                    HDF5File['SDO positions'][...] / self.constants.dx #type: ignore
                ).astype('float32')
                #TODO: need to add fov for sdo
            elif self.stereo_pov:
                cubes.stereo_pos = (
                    HDF5File['STEREO B positions'][...] / self.constants.dx  #type: ignore
                ).astype('float32')
                #TODO: will need to add the POV center
            
        if self.test_data:

            with h5py.File(self.paths['test data'], 'r') as HDF5File:
                
                # todo could add a 'safe' arg where it checks if the xt_min, yt_min are the same
                cubes.fake_cube = self.get_cube_info(HDF5File, 'Test data/Fake cube')
        return cubes
    
    def get_default_data(self, HDF5File: h5py.File) -> CubesConstants:
        """
        Gives the global information for the data.

        Args:
            HDF5File (h5py.File): the HDF5 file where the data is stored.

        Returns:
            CubesConstants: the global information of protuberance.
        """
        
        # DATA setup
        time_indexes: np.ndarray = HDF5File['Time indexes'][...]
        dates_bytes: np.ndarray = HDF5File['Dates'][...]
        dates_str = [dates_bytes[number].decode('utf-8') for number in time_indexes]

        constants = CubesConstants(
            dx=float(HDF5File['dx'][...]),
            time_indexes=time_indexes,
            dates=dates_str,
        )
        return constants
    
    def get_cube_info(
            self,
            HDF5File: h5py.File,
            group_path: str,
            interpolate: bool = True,
        ) -> CubeInfo:
        """
        Gives the protuberance and polynomial fit information for a chosen cube 'type'.

        Args:
            HDF5File (h5py.File): the HDF5 file where the data is stored.
            group_path (str): the HDF5 group absolute path (i.e. represents the cube 'type').
            interpolate (bool, optional): If there exists interpolation data for that cube 'type'.
                Defaults to True.

        Returns:
            CubeInfo: the chosen cube 'type' information with the corresponding interpolations'.
        """

        # BORDERs as index
        xt_min_index = float(HDF5File[group_path + '/xmin'][...]) / self.constants.dx
        yt_min_index = float(HDF5File[group_path + '/ymin'][...]) / self.constants.dx
        zt_min_index = float(HDF5File[group_path + '/zmin'][...]) / self.constants.dx

        # COO data
        data_coords: np.ndarray = HDF5File[group_path + '/coords'][...]
        data_values: np.ndarray = HDF5File[group_path + '/values'][...] 
        data_shape = np.max(data_coords, axis=1) + 1
        data_coo = sparse.COO(
            coords=data_coords, data=data_values, shape=data_shape
        ).astype('uint8')

        # INTERPOLATION data
        if self.interpolation and interpolate:
            interpolations: list[InterpolationData] = [None] * len(self.interpolation_order)

            for i, order in enumerate(self.interpolation_order):
                dataset_path = group_path + f'/{order}th order interpolation/coords'

                # DATA get
                interp_coords: np.ndarray = HDF5File[dataset_path][...]
                interp_coo = sparse.COO(
                    coords=interp_coords,
                    data=np.ones(interp_coords.ravel()),
                    shape=data_shape,
                ).astype('uint8')

                # DATA formatting
                interpolations[i] = InterpolationData(
                    coo=interp_coo,
                    order=order,
                    name=f'{order}th ' + group_path.split('/')[1],
                    color_hex=self.plot_interpolation_colours[i],
                )
        else:
            interpolations = None

        # FORMATTING data
        cube_info = CubeInfo(
            name=' '.join(group_path.split('/')[:2]),
            xt_min_index=xt_min_index,
            yt_min_index=yt_min_index,
            zt_min_index=zt_min_index,
            coo=data_coo,
            interpolations=interpolations,
        )
        return cube_info


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
            camera_fov: int | float | str = 0.23, 
            camera_zoom_speed: int | float = 0.7, 
            camera_pos: tuple[int | float, int | float, int | float] | None = None,
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
                index position of the camera. Automatically set if 'sdo_pov' or 'stereo_pov' is set
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
        self.up_vector = up_vector  # up vector for the camera
        self.visible_grid = visible_grid  # setting the grid to be visible or not
        self.texture_resolution = texture_resolution
        self.compression_level = compression_level
        self.outlines = outlines
        
        # CHECK fov
        if camera_fov=='stereo':
            self.camera_fov = 0.26
        elif isinstance(camera_fov, (int, float)):
            self.camera_fov = camera_fov
        else:
            raise ValueError('When "camera_fov" a string, needs to be `sdo` or `stereo`.')

        # PLACEHOLDERs
        self.plot: k3d.plot.Plot  # plot object
        # self.plot_alldata: VoxelType # voxels plot of the all data 
        # self.plot_dupli_new: VoxelType  # same for the second method
        # self.plot_interv_new: VoxelType  # same for the second method
        # self.plot_interv_dupli_new: VoxelType  # same for the second method
        # self.plot_fake_cube: VoxelType
        self.play_pause_button: ipywidgets.ToggleButton  # Play/Pause widget initialisation
        self.time_slider: ipywidgets.IntSlider # time slider widget
        self.date_dropdown: ipywidgets.Dropdown  # Date dropdown widget to show the date
        self.time_link: ipywidgets.jslink  # JavaScript Link between the two widgets

        # RUN
        if self.time_interval_all_data or self.time_interval_no_duplicates:
            self.time_interval_string()
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
                point_size=0.7, colors=[0xffff00] * len(self.sun_points),
                shader='flat',
                name='SUN',
                compression_level=self.compression_level,
            )
            self.plot += points

        # ALL DATA add
        if self.cubes.all_data is not None:
            # VOXELS create
            self.plot_alldata = self.create_voxels(self.cubes.all_data)
            for plot in self.plot_alldata: self.plot += plot     
        
        # NO DUPLICATES add
        if self.cubes.no_duplicate is not None:
            # VOXELs create
            self.plot_dupli_new = self.create_voxels(self.cubes.no_duplicate)
            for plot in self.plot_dupli_new: self.plot += plot

        # TIME INTEGRATION add
        if self.cubes.integration_all_data is not None:
            # VOXELs create
            self.plot_interv_new = self.create_voxels(self.cubes.integration_all_data)
            for plot in self.plot_interv_new: self.plot += plot
       
        # TIME NO DUPLICATES add       
        if self.cubes.integration_no_duplicate is not None:
            # VOXELs create
            self.plot_interv_dupli_new = self.create_voxels(self.cubes.integration_no_duplicate)
            for plot in self.plot_interv_dupli_new: self.plot += plot  

        # SDO LINE OF SIGHT add
        if self.cubes.los_sdo is not None:
            # VOXELs create
            self.plot_los_sdo = self.create_voxels(self.cubes.los_sdo)
            for plot in self.plot_los_sdo: self.plot += plot

        # STEREO LINE OF SIGHT add
        if self.cubes.los_stereo is not None:
            # VOXELs create
            self.plot_los_stereo = self.create_voxels(self.cubes.los_stereo)
            for plot in self.plot_los_stereo: self.plot += plot
        
        # CUBE fake # todo will need to change this
        if self.cubes.fake_cube is not None:
            self.plot_fake_cube = k3d.voxels(
                voxels=self.cubes.fake_cube.coo.todense().transpose(2, 1, 0),
                color_map=[0xff6666],
                opacity=1,
                name=f'fake cube',
                **kwargs,
            )
            self.plot += self.plot_fake_cube

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
        self.date_dropdown = ipywidgets.Dropdown(options=self.constants.dates, description='Date:')
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
            cube: CubeInfo,
            index: int = 0,
            opacity: float = 0.8,
            color_map: list[int] = [0x0000ff],
            **kwargs,
        ) -> list[VoxelType]:
        """
        Creates the k3d.plot.voxels() for a given cube 'type'

        Args:
            cube (CubeInfo): the cube and polynomial fit information.
            index (int, optional): the chosen index to initially plot. Defaults to 0.
            opacity (float, optional): the voxel opacity in the plot. Defaults to 0.8.
            color_map (list[int], optional): the color chosen for the voxels.
                Defaults to [0x0000ff].

        Returns:
            list[VoxelType]: the k3d.plot.voxels() created.
        """

        # PLACEHOLDER voxels
        plots: list[VoxelType] = [None] * (
            1 + (len(cube.interpolations) if cube.interpolations is not None else 0)
        )

        # INDEX translation
        translation = (cube.zt_min_index, cube.yt_min_index, cube.xt_min_index)

        plots[0] = k3d.voxels(
            voxels=cube.coo[index].todense().transpose(2, 1, 0), #type: ignore
            name=cube.name,
            opacity=opacity,
            color_map=color_map,
            translation=translation,
            **kwargs,
        )
        
        if cube.interpolations is not None:
            # INTERPOLATIONs setup
            interpolations = cube.interpolations

            # INTERPOLATION orders
            for i, interpolation_data in enumerate(interpolations):
                plots[i + 1] = k3d.voxels(
                    voxels=interpolation_data[index].todense().transpose(2, 1, 0),
                    name=interpolation_data.name,
                    opacity=opacity,
                    color_map=[interpolation_data.color_hex],
                    translation=translation,
                    **kwargs,
                )
        return plots

    def camera_params(self) -> None:
        """
        Camera visualisation parameters.
        """

        # PARAMs constant
        self.plot.camera_auto_fit = False
        self.plot.camera_fov = self.camera_fov  # FOV in degrees
        self.plot.camera_zoom_speed = self.camera_zoom_speed  # zooming too quickly (default=1.2)

        self._camera_reference = np.array([0, 0, 0])
        # self._camera_reference = np.array([self.shape[3], self.shape[2], self.shape[1]]) / 2
        # TODO: this is wrong but will do for now
        
        # POV stereo
        if self.cubes.stereo_pos is not None:
            self.plot.camera = [
                self.cubes.stereo_pos[0, 0],
                self.cubes.stereo_pos[0, 1],
                self.cubes.stereo_pos[0, 2],
                self._camera_reference[0], self._camera_reference[1], self._camera_reference[2],
                self.up_vector[0], self.up_vector[1], self.up_vector[2] # up vector
            ] 
        # POV sdo
        elif self.cubes.sdo_pos is not None:
            self.plot.camera = [
                self.cubes.sdo_pos[0, 0], self.cubes.sdo_pos[0, 1], self.cubes.sdo_pos[0, 2],
                self._camera_reference[0], self._camera_reference[1], self._camera_reference[2],
                self.up_vector[0], self.up_vector[1], self.up_vector[2]
            ]  
        else:
            au_in_solar_r = 215  # 1 au in solar radii
            distance_to_sun = au_in_solar_r * self.radius_index 

            if not self.camera_pos:
                print("No 'camera_pos', setting default values.")
                self.camera_pos = (- distance_to_sun, -0.5 * distance_to_sun, 0)

            self.plot.camera = [
                self._camera_reference[0] + self.camera_pos[0], 
                self._camera_reference[1] + self.camera_pos[1],
                self._camera_reference[2] + self.camera_pos[2],
                self._camera_reference[0], self._camera_reference[1], self._camera_reference[2],
                self.up_vector[0], self.up_vector[1], self.up_vector[2]
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

    def update_plot(self, change: dict[str, any]) -> None:
        """
        Updates the voxels depending on which time value is chosen.

        Args:
            change (dict[str, any]): the time value, with the 'new' key being the new value and
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
        self.update_voxel(self.plot_alldata, self.cubes.all_data, change['new'])
        # NO DUPLICATES
        self.update_voxel(self.plot_dupli_new, self.cubes.no_duplicate, change['new'])
        # TIME INTEGRATION
        self.update_voxel(self.plot_interv_new, self.cubes.integration_all_data, change['new'])
        # TIME NO DUPLICATES     
        self.update_voxel(
            self.plot_interv_dupli_new, self.cubes.integration_no_duplicate, change['new'],
        )        
        # SDO LINE OF SIGHT
        self.update_voxel(self.plot_los_sdo, self.cubes.los_sdo, change['new'])
        # STEREO LINE OF SIGHT
        self.update_voxel(self.plot_los_stereo, self.cubes.los_stereo, change['new'])

    def update_voxel(self, plots: list[VoxelType], cube_info: CubeInfo | None, index: int) -> None:
        """
        Updates the k3d plot voxels for each cube and the corresponding interpolations.

        Args:
            plots (list[VoxelType]): the different voxel plots for each cube type.
            cube_info (CubeInfo | None): each cube type. None if it wasn't created.
            index (int): the index to plot.
        """

        # DATA get
        if cube_info is None: return
        cubes = cube_info[index]
        
        # PLOTs add data
        for i, plot in enumerate(plots): plot.voxels = cubes[i].todense().transpose(2, 1, 0)

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

    def play_pause_handler(self, change: dict[str, any]) -> None:
        """
        Changes the play button to pause when it is clicked.

        Args:
            change (dict[str, any]): the dictionary representing the value.
        """

        if change['new']:  # if clicked play
            self.play()
            self.play_pause_button.description = 'Pause'
            self.play_pause_button.icon = 'pause'

    def time_interval_string(self) -> None:
        """
        To change self.time_interval to a string giving a value in day, hours or minutes.
        """
        # todo code looks ugly, will need to look for another (prettier) option

        self.time_interval *= 3600
        if self.time_interval < 60:
            self.time_interval = f'{self.time_interval}s'
        elif self.time_interval < 3600:
            time_interval = f'{self.time_interval // 60}min'
            if self.time_interval % 60 != 0:
                self.time_interval = time_interval + f'{self.time_interval % 60}s'
            else:
                self.time_interval = time_interval
        elif self.time_interval < 3600 * 24:
            time_interval = f'{self.time_interval // 3600}h'
            if self.time_interval % 3600 != 0:
                self.time_interval = time_interval + f'{(self.time_interval % 3600) // 60}min'
            else: 
                self.time_interval = time_interval
        elif self.time_interval < 3600 * 24 * 3.1:
            time_interval = f'{self.time_interval // (3600 * 24)}days'
            if self.time_interval % (3600 * 24) != 0:
                self.time_interval = time_interval + f'{(self.time_interval % 3600 * 24) // 3600}h'
            else:
                self.time_interval = time_interval
        else:
            raise ValueError("Time interval is way too large")
