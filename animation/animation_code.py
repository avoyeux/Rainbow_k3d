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
from astropy.io import fits

# IMPORTs personal
from animation.animation_dataclasses import *
from common import Decorators, Plot, root_path

# ANNOTATION alias
VoxelType = Any
JsLinkType = Any

# todo need to understand why the polynomial is at the wrong place. borders for the polynomial seem
# different than for the corresponding data...
# ! the position of the STEREO satellite also seems to be at the wrong position
# ! the no duplicates cube and the corresponding integrated cubes seem to be from the wrong date
# ! while the line of sight data seems to be from the right date
# ! that being said, the all data cubes seem to be from the right date
# * seems like the position of the sdo satellite is the right one for the 3D visualisation and for
# * the re-projection of the data.
# ? can the problem only come from wrong border values?
# * the no duplicate data used in the animation is the right one from the data.h5 file.
# ! the creation of the data cubes must be wrong



class Setup:
    """
    Manipulates the HDF5 filament data file to setup the necessary data choices for the
    visualisation.
    This class is the parent class to the k3d visualisation class named K3dAnimation.
    """

    @typeguard.typechecked
    def __init__(
            self,
            filename: str = 'data.h5',
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
            processes: int = 5,
            polynomial: bool = False,
            polynomial_order: int | list[int] = [4],
            only_fake_data: bool = False,
            test_data: bool = False,
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

        # FEET
        self.feet = ' with feet' if with_feet else ''
        
        # ATTRIBUTES
        self.solar_r = 6.96e5
        self.filename = filename
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
        self.processes = processes
        self.polynomial = polynomial
        if isinstance(polynomial_order, list):
            self.polynomial_order = polynomial_order
        else:
            self.polynomial_order = [polynomial_order]

        self.only_fake_data = only_fake_data
        self.test_data = test_data

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
            'data': os.path.join(root_path, 'data'),
            'sdo': os.path.join(root_path, '..', 'sdo'),
        }

        # PATHs update
        if self.only_fake_data: paths['data'] = os.path.join(paths['data'], 'fake_data')
        return paths

    @Decorators.running_time
    def get_data(self) -> CubesData:
        """
        Opens the HDF5 file to get the necessary data for visualisation.
        """

        # DATA init
        HDF5File = h5py.File(os.path.join(self.paths['data'], self.filename), 'r')
        cubes = CubesData(hdf5File=HDF5File)

        # DATA main
        self.constants = self.get_default_data(HDF5File)
        self.radius_index = self.solar_r / self.constants.dx

        # CHOICES data
        if self.all_data: 
            path = 'Filtered/All data' + self.feet
            cubes.all_data = self.get_cube_info(HDF5File, path, interpolate=False)

        if self.no_duplicate: 
            path = 'Filtered/No duplicates' + self.feet
            cubes.no_duplicate = self.get_cube_info(HDF5File, path, interpolate=False)

        if self.all_data_integration: 
            path = (
                'Time integrated/All data' + self.feet +
                f'/Time integration of {round(float(self.time_interval), 1)} hours'
            )
            cubes.integration_all_data = self.get_cube_info(HDF5File, path)

        if self.no_duplicate_integration:
            path = (
                'Time integrated/No duplicates' + self.feet +
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
        if self.pov_sdo:
            # SDO positions
            time_indexes: np.ndarray = HDF5File['Time indexes'][...]
            sdo_positions: np.ndarray = HDF5File['SDO positions'][...]

            # DATA formatting
            cubes.sdo_pos = (
                sdo_positions[time_indexes] / self.constants.dx  #type: ignore
            ).astype('float32')
        if self.pov_stereo:
            # STEREO positions
            time_indexes: np.ndarray = HDF5File['Time indexes'][...]
            stereo_positions: np.ndarray = HDF5File['STEREO B positions'][...]

            # DATA formatting
            cubes.stereo_pos = (
                stereo_positions[time_indexes] / self.constants.dx  #type: ignore
            ).astype('float32')
            #TODO: will need to add the POV center

        if self.test_data:
            cubes.fake_cube = self.get_cube_info(
                HDF5File,
                'Test data/Fake cube',
                interpolate=False,
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
                dataset_path = group_path + f'/{order}th order polynomial/coords'

                # DATA get
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

        # FORMATTING data
        cube_info = CubeInfo(
            name=' '.join(group_path.split('/')[:2]),
            xt_min_index=xt_min_index,
            yt_min_index=yt_min_index,
            zt_min_index=zt_min_index,
            dataset_coords=data_coords,
            dataset_values=data_values,
            polynomials=polynomials,
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
                point_size=0.7, colors=[0xffff00] * len(self.sun_points),
                shader='flat',
                name='SUN',
                compression_level=self.compression_level,
            )
            self.plot += points

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
            cube: CubeInfo,
            index: int = 0,
            opacity: float = 0.8,
            color_map: list[int] = [0x0000ff],
            **kwargs,
        ) -> list[VoxelType]:
        """
        Creates the k3d.plot.voxels() for a given cube 'type'.

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
            1 + (len(cube.polynomials) if cube.polynomials is not None else 0)
        )

        # INDEX translation
        translation = (cube.xt_min_index, cube.yt_min_index, cube.zt_min_index)
        print(f'the borders are {translation}')

        plots[0] = k3d.voxels(
            voxels=cube[index][0].transpose(2, 1, 0),
            name=cube.name,
            opacity=opacity,
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
                    voxels=polynomial_data[index].transpose(2, 1, 0),
                    name=polynomial_data.name,
                    opacity=opacity,
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
                self._camera_reference[0] + self.camera_pos[0], 
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

    def update_voxel(self, plots: list[VoxelType], cube_info: CubeInfo, index: int) -> None:
        """
        Updates the k3d plot voxels for each cube and the corresponding polynomials.

        Args:
            plots (list[VoxelType]): the different voxel plots for each cube type.
            cube_info (CubeInfo): each cube type. None if it wasn't created.
            index (int): the index to plot.
        """

        # DATA get
        cubes = cube_info[index]
        print(f'number of voxels for name {cube_info.name} is {np.sum(cubes[0])}')
        
        # PLOTs add data
        for i, plot in enumerate(plots): plot.voxels = cubes[i].transpose(2, 1, 0)

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
