"""
To 3D visualise the rainbow filament data and the corresponding polynomial fitting.
"""

# Imports
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

import numpy as np

# Personal imports
from common import Decorators


#TODO: I have a problem with the north and south pole in the visualisation. Problem most likely comes from the data.h5 itself as the same problem arises in the
#final reprojection.

class Setup:
    """
    Manipulates the HDF5 filament data file to setup the necessary data choices for the visualisation.
    This class is the parent class to the k3d visualisation class named K3dAnimation.
    """

    @typeguard.typechecked
    def __init__(
            self,
            filename: str = 'order0321.h5',
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
    ) -> None:
        """ 
        To setup the instance attributes needed for the 3D visualisation of the Rainbow filament data.
        Args:
            filename (str, optional): the HDF5 filename containing all the cube data. Defaults to 'order0321.h5'.
            sun (bool, optional): choosing to add the sun visualisation. Defaults to False.
            with_feet (bool, optional): choosing the data sets that have the feet manually added. Defaults to True.
            all_data (bool, optional): choosing to visualise the data that also contains the duplicates. Defaults to False.
            no_duplicates (bool, optional): choosing to visualise the data with no duplicates at all. Defaults to False.
            time_interval (int, optional): the time interval for the time integration (in hours). Defaults to 24.
            time_interval_all_data (bool, optional): choosing to visualise the time integrated data with duplicates. Defaults to False.
            time_interval_no_duplicates (bool, optional): choosing to visualise the time integrated data without any duplicates. Defaults to False.
            sdo_pov (bool, optional): choosing to take SDO's point of view when looking at the data. Defaults to False.
            stereo_pov (bool, optional): choosing to take STEREO B's point of view when looking at the data. Defaults to False.
            processes (int, optional): the number of processes used in the multiprocessing. Defaults to 5.
            interpolation (bool, optional): choosing to visualise the polynomial data fits. Defaults to False.
            interpolation_order (int | list[int], optional): the polynomial orders that you want to visualise (if interpolation is set to True).
                Defaults to [5].
        """
        
        self.filename = filename
        self.sun = sun
        self.all_data = all_data
        self.with_feet = with_feet  # TODO: will need to change this so that I can choose which has feet or not
        self.no_duplicates = no_duplicates
        self.time_interval = time_interval
        self.time_interval_all_data = time_interval_all_data
        self.time_interval_no_duplicates = time_interval_no_duplicates
        self.line_of_sight_SDO = line_of_sight_SDO
        self.line_of_sight_STEREO = line_of_sight_STEREO
        self.sdo_pov = sdo_pov
        self.stereo_pov = stereo_pov
        self.processes = processes
        self.interpolation = interpolation  # TODO: probably will keep the interpolation only for the main data
        self.interpolation_order = interpolation_order if isinstance(interpolation_order, list) else [interpolation_order]
        self.solar_r = 6.96e5

        self.paths = self.setup_paths()
        self.get_data()

        # Attributes placeholders for [...]
        # [...] interpolation data
        self.interpolation_names: list[str]
        self.interpolation_data: list[sparse.COO]
        # [...] data cubes
        self.shape: tuple[int, int, int, int]
        self.dates: list[str]
        self.radius_index: float
        self.sun_center: np.ndarray
        self.cubes_all_data: sparse.COO
        self.cubes_no_duplicates: sparse.COO
        self.cubes_integrated_all_data: sparse.COO
        self.cubes_integrated_no_duplicate: sparse.COO
        self.cubes_lost_sdo: sparse.COO
        self.cubes_lost_stereo: sparse.COO

        # [...] satellite positions
        self.sdo_pos: np.ndarray
        self.stereo_pos: np.ndarray

    def setup_paths(self) -> dict[str, str]:
        """
        Creates a dictionary for the filepaths.

        Raises:
            ValueError: if the main filepath doesn't exist.

        Returns:
            dict[str, str]: the filepath dictionary.
        """

        main = '/home/avoyeux/old_project/avoyeux'
        if not os.path.exists(main): 
            main = '/home/avoyeux/Documents/avoyeux'
            if not os.path.exists(main): raise ValueError(f"Main path not found: {main}")
        codes = os.path.join(main, 'python_codes')

        paths = {
            'main': main,
            'codes': codes,
            'data': os.path.join(codes, 'Data', self.filename)
        }
        return paths

    @Decorators.running_time
    def get_data(self) -> None:
        """
        To get the visualisation data.
        """

        with h5py.File(self.paths['data'], 'r') as H5PYFile:

            # Get main data
            dx = H5PYFile['dx'][...]
            dates = H5PYFile['Dates'][...]
            numbers = H5PYFile['Time indexes'][...]
            self.dates = [dates[number].decode('utf-8') for number in numbers]
            self.radius_index = self.solar_r / dx

            # Feet setup
            feet = ' with feet' if self.with_feet else ''

            # Sun center setup
            border_group = H5PYFile['Filtered/All data' + feet]  
            self.sun_center = np.array([- border_group['xmin'][...], - border_group['ymin'][...], - border_group['zmin'][...]]) / dx

            # Get data choices
            shape = None
            paths = []
            if self.all_data: 
                paths.append('Filtered/All data' + feet)
                self.cubes_all_data = self.get_COO(H5PYFile, paths[-1])
                shape = self.cubes_all_data.shape

            if self.no_duplicates: 
                paths.append('Filtered/No duplicates new' + feet)
                self.cubes_no_duplicates = self.get_COO(H5PYFile, paths[-1])
                if shape is None: shape = self.cubes_no_duplicates.shape

            if self.time_interval_all_data: 
                paths.append('Time integrated/All data' + feet + f'/Time integration of {round(float(self.time_interval), 1)} hours')
                self.cubes_integrated_all_data = self.get_COO(H5PYFile, paths[-1])
                if shape is None: shape = self.cubes_integrated_all_data.shape

            if self.time_interval_no_duplicates:
                paths.append('Time integrated/No duplicates new' + feet + f'/Time integration of {round(float(self.time_interval), 1)} hours')
                self.cubes_integrated_no_duplicate = self.get_COO(H5PYFile, paths[-1])
                if shape is None: shape = self.cubes_integrated_no_duplicate.shape
            # Saving cubes shape
            cubes_shape = shape
            print(f'the data cube shape is {shape}')

            if self.line_of_sight_SDO:
                paths.append('Filtered/SDO line of sight')
                self.cubes_los_sdo = self.get_COO(H5PYFile, paths[-1])
                shape = self.cubes_los_sdo.shape
            
            if self.line_of_sight_STEREO:
                paths.append('Filtered/STEREO line of sight')
                self.cubes_los_stereo = self.get_COO(H5PYFile, paths[-1])
                if shape is None: shape = self.cubes_los_stereo.shape

            # Saving the shape
            if shape is not None:
                self.shape = shape
            else:
                raise ValueError(f'You need to set at least one of the data attributes to True.')
            
            if self.interpolation and (cubes_shape is not None):
                # Data path
                interpolation_paths = [
                    path + f'/{order}th order interpolation/coords' 
                    for path in paths
                    if path.startswith('Time integrated')
                    for order in self.interpolation_order
                ]

                # Get the polynomial curve names
                self.interpolation_names = self.name_polynomial(interpolation_paths)

                # Get interpolation curves
                self.interpolation_data = [
                    sparse.COO(coords=H5PYFile[path][...], data=1, shape=cubes_shape).astype('uint8')
                    for path in interpolation_paths
                ]
            
            #TODO: weirdly, the [2,1,0] seems to be right as I am pointing on the right side of the sun. Problem is the cubes themselves
            if self.sdo_pov:
                self.sdo_pos = (H5PYFile['SDO positions'][numbers] / dx + self.sun_center).astype('float32')  #TODO: need to add fov for sdo
            elif self.stereo_pov:
                self.stereo_pos = (H5PYFile['STEREO B positions'][numbers] / dx + self.sun_center).astype('float32')  #TODO: will need to add the POV center

    def get_COO(self, H5PYFile: h5py.File, group_path: str) -> sparse.COO:
        """
        To get the sparse.COO object from the corresponding coords and values.

        Args:
            H5PYFile (h5py.File): the file object.
            group_path (str): the path to the group where the data is stored.

        Returns:
            sparse.COO: the corresponding sparse data.
        """

        data_coords = H5PYFile[group_path + '/coords'][...]
        data_data = H5PYFile[group_path + '/values'][...]
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

        # Pattern setup
        intergration_time_pattern = re.compile(r"(\d+\.?\d+|\d+)")
        interpolation_order_pattern = re.compile(r"(\d+)")

        names = [None] * len(group_paths)

        for i, group_path in enumerate(group_paths):
            # Get group names
            segments = group_path.split('/')
            data_type = segments[1]

            # Naming the polynomials
            beginning = 'Fit '
            if 'All data' in data_type:
                beginning += 'all'
            elif 'No duplicates new' in data_type:
                beginning += 'dupli'
            if 'feet' in data_type:
                beginning += ' n feet'

            # Get integration time        
            integration = segments[2]
            # print(f'end is {end}')
            pattern_match = re.search(intergration_time_pattern, integration)
            if pattern_match is not None:
                time_integration = float(pattern_match.group(0))
                beginning += f' {time_integration}h'
            else:
                raise ValueError('Pattern is wrong boss.')
            
            # Get polynomial order
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
        
        super().__init__(**kwargs)

        # Arguments
        self.plot_height = plot_height  # the height in pixels of the plot (initially it was 512)
        self.sleep_time = sleep_time  # sets the time between each frames (in seconds)
        self.camera_zoom_speed = camera_zoom_speed  # zoom speed of the camera 
        self.camera_pos = camera_pos  # position of the camera multiplied by 1au
        self.up_vector = up_vector  # up vector for the camera
        self.visible_grid = visible_grid  # setting the grid to be visible or not
        self.texture_resolution = texture_resolution
        self.compression_level = compression_level
        self.outlines = outlines
        
        if camera_fov=='sdo':
            self.camera_fov = self.Fov_for_SDO()
        elif camera_fov=='stereo':
            self.camera_fov = 0.26
        elif isinstance(camera_fov, (int, float)):
            self.camera_fov = camera_fov
        else:
            raise ValueError('When "camera_fov" a string, needs to be `sdo` or `stereo`.')

        # Instance attributes set when running the class
        self.plot: k3d.plot  # plot object
        self.plot_alldata: k3d.voxels  # voxels plot of the all data 
        self.plot_dupli_new: k3d.voxels  # same for the second method
        self.plot_interv_new: k3d.voxels  # same for the second method
        self.plot_interv_dupli_new: k3d.voxels  # same for the second method
        self.play_pause_button: ipywidgets.ToggleButton  # Play/Pause widget initialisation
        self.time_slider: ipywidgets.IntSlider # time slider widget
        self.date_dropdown: ipywidgets.Dropdown  # Date dropdown widget to show the date
        self.time_link: ipywidgets.jslink  # JavaScript Link between the two widgets

        # Making the animation
        if self.time_interval_all_data or self.time_interval_no_duplicates: self.time_interval_string()
        self.animation()

    def animation(self) -> None:
        """
        Creates the 3D animation using k3d. 
        """
        
        # Initialisation of the plot
        self.plot = k3d.plot(grid_visible=self.visible_grid)  # plot with no axes. If a dark background is needed then background_color=0x000000
        self.plot.height = self.plot_height  
            
        # Add camera parameters
        self.camera_params()

        # Add default parameters
        kwargs = {
            'compression_level': self.compression_level, # the compression level of the data in the 3D visualisation
            'outlines': self.outlines,
        }

        # Add filaments
        if self.all_data: 
            self.plot_alldata = k3d.voxels(
                voxels=self.cubes_all_data[0].todense().transpose(2, 1, 0),
                opacity=0.7,
                color_map=[0x0000ff],
                name='allData',
                **kwargs,
            )
            self.plot += self.plot_alldata      
               
        if self.no_duplicates:
            self.plot_dupli_new = k3d.voxels(
                voxels=self.cubes_no_duplicates[0].todense().transpose(2, 1, 0),
                color_map=[0x0000ff],
                opacity=0.3,
                name='noDuplicates',
                **kwargs,
            )
            self.plot += self.plot_dupli_new

        if self.time_interval_all_data:
            self.plot_interv_new = k3d.voxels(
                voxels=self.cubes_integrated_all_data[0].todense().transpose(2, 1, 0),
                color_map=[0xff6666],
                opacity=1,
                name=f'allData {self.time_interval}',
                **kwargs,
            )
            self.plot += self.plot_interv_new        
       
        if self.time_interval_no_duplicates:
            self.plot_interv_dupli_new = k3d.voxels(
                voxels=self.cubes_integrated_no_duplicate[0].todense().transpose(2, 1, 0),
                color_map=[0x0000ff],
                opacity=0.35,
                name=f'noDupli {self.time_interval}',
                **kwargs,
            )
            self.plot += self.plot_interv_dupli_new      

        if self.line_of_sight_SDO:
            self.plot_los_sdo = k3d.voxels(
                voxels=self.cubes_los_sdo[0].todense().transpose(2, 1, 0),
                color_map=[0xff6666],
                opacity=1,
                name=f'lineOfSight SDO',
                **kwargs,
            )
            self.plot += self.plot_los_sdo

        if self.line_of_sight_STEREO:
            self.plot_los_stereo = k3d.voxels(
                voxels=self.cubes_los_stereo[0].todense().transpose(2, 1, 0),
                color_map=[0xff6666],
                opacity=1,
                name=f'lineOfSight STEREO',
                **kwargs,
            )
            self.plot += self.plot_los_stereo
    
        # if self.skeleton:
        #     self.plot_skeleton = k3d.voxels(self.Full_array(self.cubes_skeleton[0]), color_map=[0xff6e00], opacity=1, name='barycenter for the no dupliactes',
        #                                     **self.kwargs)
        #     self.plot += self.plot_skeleton

        # if self.convolution:
        #     self.plot_convolution = k3d.voxels(self.Full_array(self.cubes_convolution[0]), color_map=[0xff6e00], opacity=0.5, name='conv3d', **self.kwargs)
        #     self.plot += self.plot_convolution

        # Add curve fitting
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

        # Add Sun
        if self.sun:
            self.add_sun()
            self.plot += k3d.points(
                positions=self.sun_points,
                point_size=0.7, colors=[0xffff00] * len(self.sun_points),
                shader='flat',
                name='SUN',
                compression_level=self.compression_level,
            )

        # Adding a play/pause button
        self.play_pause_button = ipywidgets.ToggleButton(value=False, description='Play', icon='play')

        # Set up the time slider and the play/pause button
        self.time_slider = ipywidgets.IntSlider(min=0, max=len(self.dates)-1, description='Frame:')
        self.date_dropdown = ipywidgets.Dropdown(options=self.dates, description='Date:')
        self.time_slider.observe(self.update_voxel, names='value')
        self.time_link = ipywidgets.jslink((self.time_slider, 'value'), (self.date_dropdown, 'index'))
        self.play_pause_button.observe(self.play_pause_handler, names='value')

        # Display
        IPython.display.display(self.plot, self.time_slider, self.date_dropdown, self.play_pause_button)
        # if self.make_screenshots:
        #     self.plot.screenshot_scale = self.screenshot_scale
        #     self.Screenshot_making()

    def camera_params(self) -> None:
        """
        Camera visualisation parameters.
        """
 
        self.plot.camera_auto_fit = False
        self.plot.camera_fov = self.camera_fov  # FOV in degrees
        self.plot.camera_zoom_speed = self.camera_zoom_speed  # it was zooming too quickly (default=1.2)
        
        # Point to look at, i.e. initial rotational reference

        self._camera_reference = np.array([self.shape[3], self.shape[2], self.shape[1]]) / 2  # TODO: this is wrong but will do for now
        
        if self.stereo_pov:
            self.plot.camera = [
                self.stereo_pos[0, 0], self.stereo_pos[0, 1], self.stereo_pos[0, 2],
                self._camera_reference[0], self._camera_reference[1], self._camera_reference[2],
                self.up_vector[0], self.up_vector[1], self.up_vector[2] # up vector
            ] 
        elif self.sdo_pov:
            self.plot.camera = [
                self.sdo_pos[0, 0], self.sdo_pos[0, 1], self.sdo_pos[0, 2],
                self._camera_reference[0], self._camera_reference[1], self._camera_reference[2],
                self.up_vector[0], self.up_vector[1], self.up_vector[2]
            ]  
        else:
            au_in_solar_r = 215  # 1 au in solar radii
            distance_to_sun = au_in_solar_r * self.radius_index 

            if not self.camera_pos:
                print("No 'camera_pos', setting default values.")
                self.camera_pos = (-1, -0.5, 0)
            self.plot.camera = [
                self._camera_reference[0] + self.camera_pos[0] * distance_to_sun, 
                self._camera_reference[1] + self.camera_pos[1] * distance_to_sun,
                self._camera_reference[2] + self.camera_pos[2] * distance_to_sun,
                self._camera_reference[0], self._camera_reference[1], self._camera_reference[2],
                self.up_vector[0], self.up_vector[1], self.up_vector[2]
            ] 

    def add_sun(self):
        # TODO: to add the sun, will need to change it later to re-add the grid.

        # Initialisation
        N = self.texture_resolution  # number of points in the theta direction
        phi = np.linspace(0, np.pi, N)  # latitude of the points
        theta = np.linspace(0, 2 * np.pi, 2 * N)  # longitude of the points
        phi, theta = np.meshgrid(phi, theta)  # the subsequent meshgrid

        # Conversion to cartesian coordinates
        x = self.radius_index * np.sin(phi) * np.cos(theta) + self.sun_center[0]
        y = self.radius_index * np.sin(phi) * np.sin(theta) + self.sun_center[1]
        z = self.radius_index * np.cos(phi) + self.sun_center[2] 

        print(f"x, y and z shapes for the sun are {x.shape}, {y.shape}, {z.shape}.")

        # Creation of the position of the spherical cloud of points
        self.sun_points = np.array([x.ravel(), y.ravel(), z.ravel()], dtype='float32').T

    def update_voxel(self, change: dict[str, any]) -> None:
        """
        Updates the voxels depending on which time value is chosen.

        Args:
            change (dict[str, any]): the time value, with the 'new' key being the new value and 'old' key being the old value.
        """

        if self.stereo_pov:
            self.plot.camera = [
                self.stereo_pos[change['new'], 0], self.stereo_pos[change['new'], 1], self.stereo_pos[change['new'], 2],
                self._camera_reference[0], self._camera_reference[1], self._camera_reference[2],
                self.up_vector[0], self.up_vector[1], self.up_vector[2],
            ]
            time.sleep(0.2) 
        elif self.sdo_pov:
            self.plot.camera = [
                self.sdo_pos[change['new'], 0], self.sdo_pos[change['new'], 1], self.sdo_pos[change['new'], 2],
                self._camera_reference[0], self._camera_reference[1], self._camera_reference[2],
                self.up_vector[0], self.up_vector[1], self.up_vector[2],
            ]
            time.sleep(0.2)

        if self.all_data: self.plot_alldata.voxels = self.cubes_all_data[change['new']].todense().transpose(2, 1, 0)

        if self.no_duplicates: self.plot_dupli_new.voxels = self.cubes_no_duplicates[change['new']].todense().transpose(2, 1, 0)
 
        if self.time_interval_all_data: self.plot_interv_new.voxels = self.cubes_integrated_all_data[change['new']].todense().transpose(2, 1, 0)
                          
        if self.time_interval_no_duplicates: self.plot_interv_dupli_new.voxels = self.cubes_integrated_no_duplicate[change['new']].todense().transpose(2, 1, 0)

        if self.line_of_sight_SDO: self.plot_los_sdo.voxels = self.cubes_los_sdo[change['new']].todense().transpose(2, 1, 0)

        if self.line_of_sight_STEREO: self.plot_los_stereo.voxels = self.cubes_los_stereo[change['new']].todense().transpose(2, 1, 0)
       
        # if self.skeleton: self.plot_skeleton.voxels = self.Full_array(self.cubes_skeleton[change['new']])
        # if self.convolution: self.plot_convolution.voxels = self.Full_array(self.cubes_convolution[change['new']])

        if self.interpolation:
            for i, plot in enumerate(self.plot_interpolation):
                data = self.interpolation_data[i]
                plot.voxels = data[change['new']].todense().transpose(2, 1, 0)

    def play(self) -> None:
        """
        Params for the play button.
        """
        
        if self.play_pause_button.value and self.time_slider.value < len(self.dates) - 1:
            self.time_slider.value += 1
            threading.Timer(self.sleep_time, self.play).start()  # where you also set the sleep() time.
                
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
