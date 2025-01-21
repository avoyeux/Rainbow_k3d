"""
To 3D visualise the rainbow filament data and the corresponding polynomial fitting.
"""

# IMPORTs
import os
import re
import k3d
import time
import h5py
import sparse
import typing
import IPython
import threading
import typeguard
import ipywidgets

# IMPORTs alias
import numpy as np

# IMPORTs personal
from common import Decorators, main_paths
from Animation.animation_dataclasses import CubesData

# todo need to create something for default values in the k3d color_maps



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
            self.interpolation_order = [interpolation_order]
        self.test_data = test_data
        self.test_filename = test_filename

        # PLACEHOLDERs for [...]
        # [...] interpolation
        self.interpolation_names: list[str]
        self.interpolation_data: list[sparse.COO]
        # [...] cubes
        self.cubes: CubesData
        self.shape: tuple[int, int, int, int]  # ? do I really really need this
        self.radius_index: float

        # RUN
        self.paths = self.setup_paths()
        self.get_data()

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
            'data': os.path.join(codes_path, 'Data', self.filename),
            'test data': os.path.join(codes_path, 'Data', 'fake_data', self.test_filename),
            # todo need to add the option where the test data is in the real data file
        }
        return paths

    @Decorators.running_time
    def get_data(self) -> None:
        """
        Opens the HDF5 file to get the necessary data for visualisation.
        """

        with h5py.File(self.paths['data'], 'r') as HDF5File:

            # DATA main
            cubes = self.get_default_data(HDF5File)
            self.radius_index = self.solar_r / cubes.dx

            # CHOICES data
            shape = None
            paths = []
            if self.all_data: 
                paths.append('Filtered/All data' + self.feet)
                cubes.all_data = self.get_COO(HDF5File, paths[-1])
                shape = cubes.all_data.shape

            if self.no_duplicates: 
                paths.append('Filtered/No duplicates new' + self.feet)
                cubes.no_duplicate = self.get_COO(HDF5File, paths[-1])
                if shape is None: shape = cubes.no_duplicate.shape

            if self.time_interval_all_data: 
                paths.append(
                    'Time integrated/All data' + self.feet +
                    f'/Time integration of {round(float(self.time_interval), 1)} hours'
                )
                cubes.integration_all_data = self.get_COO(HDF5File, paths[-1])
                if shape is None: shape = cubes.integration_all_data.shape

            if self.time_interval_no_duplicates:
                paths.append(
                    'Time integrated/No duplicates new' + self.feet +
                    f'/Time integration of {round(float(self.time_interval), 1)} hours'
                )
                cubes.integration_no_duplicate = self.get_COO(HDF5File, paths[-1])
                if shape is None: shape = cubes.integration_no_duplicate.shape

            # SHAPE cubes
            cubes_shape = shape
            print(f'the data cube shape is {shape}')

            if self.line_of_sight_SDO:
                paths.append('Filtered/SDO line of sight')
                cubes.los_sdo = self.get_COO(HDF5File, paths[-1])
                shape = cubes.los_sdo.shape
            
            if self.line_of_sight_STEREO:
                paths.append('Filtered/STEREO line of sight')
                cubes.los_stereo = self.get_COO(HDF5File, paths[-1])
                if shape is None: shape = cubes.los_stereo.shape

            # SHAPE save
            if shape is not None:
                self.shape = shape
            else:
                raise ValueError(f'You need to set at least one of the data attributes to True.')
            
            # INTERPOLATIONs
            if self.interpolation and (cubes_shape is not None):
                # PATH data
                interpolation_paths = [
                    path + f'/{order}th order interpolation/coords' 
                    for path in paths
                    if path.startswith('Time integrated')
                    for order in self.interpolation_order
                ]

                # POLYNOMIAL names
                self.interpolation_names = self.name_polynomial(interpolation_paths)

                # CURVES get
                self.interpolation_data = [
                    sparse.COO(
                        coords=HDF5File[path][...],
                        data=1,
                        shape=cubes_shape
                    ).astype('uint8')
                    for path in interpolation_paths
                ]
            
            # POVs sdo, stereo
            if self.sdo_pov:
                cubes.sdo_pos = (
                    HDF5File['SDO positions'][...] / cubes.dx + cubes.sun_center
                ).astype('float32')  #TODO: need to add fov for sdo
            elif self.stereo_pov:
                cubes.stereo_pos = (
                    HDF5File['STEREO B positions'][...] / cubes.dx + cubes.sun_center
                ).astype('float32')  #TODO: will need to add the POV center
            
        if self.test_data:

            with h5py.File(self.paths['test data'], 'r') as HDF5File:
                
                # ! this won't work if xt_min, yt_min and zt_min are not the same
                # todo could add a 'safe' arg where it checks if the xt_min, yt_min are the same
                # cubes.sun_surface = self.get_COO(HDF5File, 'Test data/Sun surface')
                cubes.fake_cube = self.get_COO(HDF5File, 'Test data/Fake cube')

        # DATA as instance attribute
        self.cubes = cubes
    
    def get_default_data(self, HDF5File: h5py.File) -> CubesData:
        
        print('here')
        # DATA setup
        time_indexes: np.ndarray = HDF5File['Time indexes'][...]
        dates_bytes: np.ndarray = HDF5File['Dates'][...]
        dates_str = [dates_bytes[number].decode('utf-8') for number in time_indexes]

        print('here2')
        # BORDERs find
        border_group = HDF5File['Filtered/All data' + self.feet]
        # border_group = HDF5File['Test data/Fake cube']
        print('here3')

        # DATA formatting
        main_data = CubesData(
            dx=float(HDF5File['dx'][...]),
            time_indexes=time_indexes,
            dates=dates_str,
            xt_min=float(border_group['xmin'][...]),
            yt_min=float(border_group['ymin'][...]),
            zt_min=float(border_group['zmin'][...]),
        )
        print(f'here4 {float(border_group['xmin'][...])}')
        print(f'sun center is {main_data.sun_center}')
        print('here5')
        return main_data

    def get_COO(self, HDF5File: h5py.File, group_path: str) -> sparse.COO:
        """
        To get the sparse.COO object from the corresponding coords and values.

        Args:
            HDF5File (h5py.File): the file object.
            group_path (str): the path to the group where the data is stored.

        Returns:
            sparse.COO: the corresponding sparse data.
        """

        # todo will need to change this when I change the datasets names to be more implicit

        data_coords: np.ndarray = HDF5File[group_path + '/coords'][...]
        data_data: np.ndarray = HDF5File[group_path + '/values'][...]
        data_shape = np.max(data_coords, axis=1) + 1
        return sparse.COO(coords=data_coords, data=data_data, shape=data_shape).astype('uint8')
    
    def name_polynomial(self, group_paths: list[str]) -> list[str]:
        """
        To get the polynomial curves name when visualising them.

        Args:
            group_paths (list[str]): the paths to the data inside the HDF5 file.

        Raises:
            ValueError: if the pattern for the integration time doesn't find a match.
            ValueError: if the pattern for the interpolation order doesn't find a match.

        Returns:
            list[str]: the polynomial curves names.
        """

        # PATTERN setup
        intergration_time_pattern = re.compile(r"(\d+\.?\d+|\d+)")
        interpolation_order_pattern = re.compile(r"(\d+)")

        # PLACEHOLDER
        names = [None] * len(group_paths)

        # NAMEs get
        for i, group_path in enumerate(group_paths):
            # NAMEs group
            segments = group_path.split('/')
            data_type = segments[1]

            # POLYNOMIAL name
            beginning = 'Fit '
            if 'All data' in data_type:
                beginning += 'all'
            elif 'No duplicates new' in data_type:
                beginning += 'dupli'
            if 'feet' in data_type:
                beginning += ' n feet'

            # TIME integration    
            integration = segments[2]
            
            # CHECK pattern
            pattern_match = re.search(intergration_time_pattern, integration)
            if pattern_match is not None:
                time_integration = float(pattern_match.group(0))
                beginning += f' {time_integration}h'
            else:
                raise ValueError('Pattern is wrong boss.')
            
            # ORDER of polynomial
            interpolation_name = segments[-2]
            interp_match = re.search(interpolation_order_pattern, interpolation_name)
            if interp_match is not None:
                beginning += f' {interp_match.group(0)}order'
            else:
                raise ValueError('Interpolation pattern is wrong my lord.')
            names[i] = beginning
        return names
    

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
        self.plot: k3d.plot  # plot object
        self.plot_alldata: k3d.voxels  # voxels plot of the all data 
        self.plot_dupli_new: k3d.voxels  # same for the second method
        self.plot_interv_new: k3d.voxels  # same for the second method
        self.plot_interv_dupli_new: k3d.voxels  # same for the second method
        self.plot_fake_cube: k3d.voxels
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
            self.plot_alldata = k3d.voxels(
                voxels=self.cubes.all_data[0].todense().transpose(2, 1, 0),
                opacity=0.7,
                color_map=[0x0000ff],
                name='allData',
                **kwargs,
            )
            self.plot += self.plot_alldata      
        
        # NO DUPLICATES add
        if self.cubes.no_duplicate is not None:
            self.plot_dupli_new = k3d.voxels(
                voxels=self.cubes.no_duplicate[0].todense().transpose(2, 1, 0),
                color_map=[0x0000ff],
                opacity=0.3,
                name='noDuplicates',
                **kwargs,
            )
            self.plot += self.plot_dupli_new

        # TIME INTEGRATION add
        if self.cubes.integration_all_data is not None:
            self.plot_interv_new = k3d.voxels(
                voxels=self.cubes.integration_all_data[0].todense().transpose(2, 1, 0),
                color_map=[0xff6666],
                opacity=1,
                name=f'allData {self.time_interval}',
                **kwargs,
            )
            self.plot += self.plot_interv_new        
       
        # TIME NO DUPLICATES add       
        if self.cubes.integration_no_duplicate is not None:
            self.plot_interv_dupli_new = k3d.voxels(
                voxels=self.cubes.integration_no_duplicate[0].todense().transpose(2, 1, 0),
                color_map=[0x0000ff],
                opacity=0.35,
                name=f'noDupli {self.time_interval}',
                **kwargs,
            )
            self.plot += self.plot_interv_dupli_new      

        # SDO LINE OF SIGHT add
        if self.cubes.los_sdo is not None:
            self.plot_los_sdo = k3d.voxels(
                voxels=self.cubes.los_sdo[0].todense().transpose(2, 1, 0),
                color_map=[0xff6666],
                opacity=1,
                name=f'lineOfSight SDO',
                **kwargs,
            )
            self.plot += self.plot_los_sdo

        # STEREO LINE OF SIGHT add
        if self.cubes.los_stereo is not None:
            self.plot_los_stereo = k3d.voxels(
                voxels=self.cubes.los_stereo[0].todense().transpose(2, 1, 0),
                color_map=[0xff6666],
                opacity=1,
                name=f'lineOfSight STEREO',
                **kwargs,
            )
            self.plot += self.plot_los_stereo
        
        # FAKE SUN surface
        if self.cubes.fake_cube is not None:
            self.plot_fake_cube = k3d.voxels(
                voxels=self.cubes.fake_cube.todense().transpose(2, 1, 0),
                color_map=[0xff6666],
                opacity=1,
                name=f'fake cube',
                **kwargs,
            )
            self.plot += self.plot_fake_cube

        # POLYNOMIAL FIT add
        if self.interpolation:
            self.plot_interpolation = []
            for (data, name) in zip(self.interpolation_data, self.interpolation_names):
                plot = k3d.voxels(
                    voxels=data[0].todense().transpose(2, 1, 0), 
                    opacity=0.95, 
                    color_map=[next(self.random_hexadecimal_color_generator())],
                    name=name,
                    **kwargs,
                )
                self.plot += plot
                self.plot_interpolation.append(plot)

        # if self.skeleton:
        #     self.plot_skeleton = k3d.voxels(
        #         self.Full_array(self.cubes_skeleton[0]),
        #         color_map=[0xff6e00],
        #         opacity=1,
        #         name='barycenter for the no dupliactes',
        #         **self.kwargs,
        #     )
        #     self.plot += self.plot_skeleton

        # if self.convolution:
        #     self.plot_convolution = k3d.voxels(
        #         self.Full_array(self.cubes_convolution[0]),
        #         color_map=[0xff6e00],
        #         opacity=0.5,
        #         name='conv3d',
        #         **self.kwargs,
        #     )
        #     self.plot += self.plot_convolution

        # BUTTON play/pause
        self.play_pause_button = ipywidgets.ToggleButton(
            value=False,
            description='Play',
            icon='play',
        )

        # SETUP time-slider and play/pause
        self.time_slider = ipywidgets.IntSlider(
            min=0,
            max=len(self.cubes.dates)-1,
            description='Frame:',
        )
        self.date_dropdown = ipywidgets.Dropdown(options=self.cubes.dates, description='Date:')
        self.time_slider.observe(self.update_voxel, names='value')
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
        
        # SCREENSHOTS save
        # if self.make_screenshots:
        #     self.plot.screenshot_scale = self.screenshot_scale
        #     self.Screenshot_making()

    def camera_params(self) -> None:
        """
        Camera visualisation parameters.
        """

        # PARAMs constant
        self.plot.camera_auto_fit = False
        self.plot.camera_fov = self.camera_fov  # FOV in degrees
        self.plot.camera_zoom_speed = self.camera_zoom_speed  # zooming too quickly (default=1.2)

        self._camera_reference = np.array([self.shape[3], self.shape[2], self.shape[1]]) / 2
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
        x = self.radius_index * np.sin(phi) * np.cos(theta) + self.cubes.sun_center[0]
        y = self.radius_index * np.sin(phi) * np.sin(theta) + self.cubes.sun_center[1]
        z = self.radius_index * np.cos(phi) + self.cubes.sun_center[2] 

        # SAVE coords
        self.sun_points = np.array([x.ravel(), y.ravel(), z.ravel()], dtype='float32').T

    def update_voxel(self, change: dict[str, any]) -> None:
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
        if self.cubes.all_data is not None:
            self.plot_alldata.voxels = (
                self.cubes.all_data[change['new']].todense().transpose(2, 1, 0)
            )

        # NO DUPLICATES
        if self.cubes.no_duplicate is not None:
            self.plot_dupli_new.voxels = (
                self.cubes.no_duplicate[change['new']].todense().transpose(2, 1, 0)
            )

        # TIME INTEGRATION
        if self.cubes.integration_all_data is not None:
            self.plot_interv_new.voxels = (
                self.cubes.integration_all_data[change['new']].todense().transpose(2, 1, 0)
            )

        # TIME NO DUPLICATES             
        if self.cubes.integration_no_duplicate is not None:
            self.plot_interv_dupli_new.voxels = (
                self.cubes.integration_no_duplicate[change['new']].todense().transpose(2, 1, 0)
            )

        # SDO LINE OF SIGHT
        if self.cubes.los_sdo is not None:
            self.plot_los_sdo.voxels = (
                self.cubes.los_sdo[change['new']].todense().transpose(2, 1, 0)
            )

        # STEREO LINE OF SIGHT
        if self.cubes.los_stereo is not None:
            self.plot_los_stereo.voxels = (
                self.cubes.los_stereo[change['new']].todense().transpose(2, 1, 0)
            )
       
        # if self.skeleton:
        #     self.plot_skeleton.voxels = self.Full_array(self.cubes_skeleton[change['new']])
        # if self.convolution:
        #     self.plot_convolution.voxels = self.Full_array(self.cubes_convolution[change['new']])

        # POLYNOMIAL FIT
        if self.interpolation:
            for i, plot in enumerate(self.plot_interpolation):
                data = self.interpolation_data[i]
                plot.voxels = data[change['new']].todense().transpose(2, 1, 0)

    def play(self) -> None:
        """
        Params for the play button.
        """
        
        if self.play_pause_button.value and self.time_slider.value < len(self.cubes.dates) - 1:
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
    
    def random_hexadecimal_color_generator(self) -> typing.Generator[int, None, None]:
        """
        Generator that yields a color value in integer hexadecimal code format.
        """

        while True: yield np.random.randint(0, 0xffffff)
