"""
To test the visualisation but by also defining the viewer position and stuff.
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
            data: bool = False,
            pov: bool = False,
            center: bool = False,
            sun_pos:tuple[int, int, int] = (10, 10, 10),
            camera_pos: tuple[int, int, int] = (10, 0, 0),
            camera_reference: tuple[int, int, int] | None = None,
            processes: int = 5,
    ) -> None:
        """ 
        #TODO: docstring.
        """
        
        self.filename = filename
        self.sun = sun
        self.data =data
        self.pov = pov
        self.center = center
        self.sun_center = sun_pos
        self.camera_pos = camera_pos
        self.camera_reference = camera_reference
        self.processes = processes
        self.solar_r = 6.96e5

        self.get_data()

    @Decorators.running_time
    def get_data(self) -> None:
        """
        #TODO: docstring.
        """

        data = np.zeros((5, 10, 15), dtype='uint8')
        data[0, :, :] = 1

        self.radius_index = 3
        self.cube_data = self.get_COO(data)
        

    def get_COO(self, data: np.ndarray) -> sparse.COO:
        """
        #TODO:docstring.
        """

        return sparse.COO(data).astype('uint8')
    

class K3dAnimation(Setup):
    """
    Creates the corresponding k3d animation to then be used in a Jupyter notebook file.
    """

    @typeguard.typechecked
    def __init__(
            self,
            compression_level: int = 9,
            sleep_time: int | float = 2, 
            camera_fov: int | float | str = 0.23, 
            camera_zoom_speed: int | float = 0.7, 
            up_vector: tuple[int, int, int] = (0, 0, 1), 
            visible_grid: bool = False, 
            outlines: bool = False,
            texture_resolution: int = 10,
            **kwargs,
    ) -> None:
        
        super().__init__(**kwargs)

        # Arguments
        self.sleep_time = sleep_time  # sets the time between each frames (in seconds)
        self.camera_zoom_speed = camera_zoom_speed  # zoom speed of the camera 
        self.up_vector = up_vector  # up vector for the camera
        self.visible_grid = visible_grid  # setting the grid to be visible or not
        self.compression_level = compression_level
        self.outlines = outlines
        self.camera_fov = camera_fov
        self.texture_resolution = texture_resolution

        self.animation()

    def animation(self) -> None:
        """
        Creates the 3D animation using k3d. 
        """
        
        # Initialisation of the plot
        self.plot = k3d.plot(grid_visible=self.visible_grid)  # plot with no axes. If a dark background is needed then background_color=0x000000
            
        # Add camera parameters
        self.camera_params()

        # Add default parameters
        kwargs = {
            'compression_level': self.compression_level, # the compression level of the data in the 3D visualisation
            'outlines': self.outlines,
        }

        # Add filaments
        if self.data: 
            self.plot_data= k3d.voxels(
                voxels=self.cube_data.todense().transpose(2, 1, 0),
                opacity=0.7,
                color_map=[0x0000ff],
                name='allData',
                **kwargs,
            )
            self.plot += self.plot_data      
            
        # Add Sun
        if self.sun:
            self.add_sun()
            self.plot += k3d.points(
                positions=self.sun_points,
                point_size=1, colors=[0xffff00] * len(self.sun_points),
                shader='flat',
                name='SUN',
                compression_level=self.compression_level,
            )
        
        if self.center:
            self.plot += k3d.points(
                positions=(0, 0, 0),
                point_size=3,
                colors=[0xff00ff],
                shader='3d',
                name='coordinates center',
                compression_level=self.compression_level,
            )

        # Display
        IPython.display.display(self.plot)

    def camera_params(self) -> None:
        """
        Camera visualisation parameters.
        """
 
        self.plot.camera_auto_fit = False
        self.plot.camera_fov = self.camera_fov  # FOV in degrees
        self.plot.camera_zoom_speed = self.camera_zoom_speed  # it was zooming too quickly (default=1.2)
        
        # Point to look at, i.e. initial rotational reference

        if self.camera_reference is None: self.camera_reference = np.array(self.cube_data.shape) / 2  # TODO: this is wrong but will do for now
        
        if self.pov:
            self.plot.camera = [
                self.camera_pos[0], self.camera_pos[1], self.camera_pos[2],
                self.camera_reference[0], self.camera_reference[1], self.camera_reference[2],
                self.up_vector[0], self.up_vector[1], self.up_vector[2] # up vector
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

        # Creation of the position of the spherical cloud of points
        self.sun_points = np.array([x.ravel(), y.ravel(), z.ravel()], dtype='float32').T

