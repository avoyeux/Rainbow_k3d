"""
Code to then put in the jupyter notebook (just because I prefer .py files).
"""

import os
import re
import k3d
import glob
import threading
import numpy as np
import ipywidgets as widgets
from astropy.io import fits
from scipy.io import readsav
from IPython.display import display
import time

class CustomDate:
    """
    To separate the year, month, day, hour, minute, second from a string dateutil.parser.parser
    doesn't work in this case. 
    """

    def __init__(self, year, month, day, hour, minute, second):
        self.year = year
        self.month = month
        self.day = day
        self.hour = hour
        self.minute = minute
        self.second = second

    @classmethod
    def parse_date(cls, date_str):
        """
        Separating a tring in the format YYYY-MM-DDThh-mm-ss to get the different time attributes.
        """

        date_part, time_part = date_str.split("T")
        year, month, day = map(int, date_part.split("-"))
        hour, minute, second = map(int, time_part.split("-"))
        return cls(year, month, day, hour, minute, second)
    
    @classmethod
    def parse_date2(cls, date_str):
        """
        Separating a bytestring in the format YYYY/MM/DD hh:mm:ss to get the different date attributes.
        """

        date_part, time_part = date_str.split(b' ')
        year, month, day = map(int, date_part.split(b"/"))
        hour, minute, second = map(int, time_part.split(b':'))
        return cls(year, month, day, hour, minute, second)


class Data:
    """
    To upload and manipulate the data to then be inputted in the k3d library for 3D animations.
    """

    def __init__(self, everything=False, sun=False, stars=False, all_data=False, duplicates=False, no_duplicate=False, 
                 line_of_sight=False, trace_data=False, trace_no_duplicate=False, day_trace=False, day_trace_no_duplicate=False,
                 time_intervals_all_data=False, time_intervals_no_duplicate = False, time_interval=1, sun_texture_resolution=960,
                 sdo_pov = False, stereo_pov=False):
        
        # Arguments
        self.sun = (sun or everything)  # choosing to plot the Sun
        self.stars = (stars or everything)  # choosing to plot the stars
        self.all_data = (all_data or everything)  # choosing to plot all the data (i.e. data containing the duplicates)
        self.duplicates = (duplicates or everything)  # for the duplicates data (i.e. from SDO and STEREO)
        self.no_duplicate = (no_duplicate or everything) # for the no duplicates data
        self.line_of_sight = (line_of_sight or everything)  # for the line of sight data 
        self.trace_data = (trace_data or everything) # for the trace of all_data
        self.trace_no_duplicate = (trace_no_duplicate or everything)  # same for no_duplicate
        self.day_trace = (day_trace or everything)  # for the trace of all_data for each day
        self.day_trace_no_duplicate = (day_trace_no_duplicate or everything)  # same for no_duplicate
        self.time_intervals_all_data = (time_intervals_all_data or everything)  # for the data integration over time for all_data
        self.time_intervals_no_duplicate = (time_intervals_no_duplicate or everything)  # same for no_duplicate
        self.time_interval = time_interval  # time interval in hours (if 'int' or 'h' in str), days (if 'd' in str), minutes (if 'min' in str)  
        self.sun_texture_resolution = sun_texture_resolution  # choosing the Sun's texture resolution
        self.sdo_pov = sdo_pov
        self.stereo_pov = stereo_pov

        # Instance attributes set when running the class
        self._cube_names = None  # sorted data cubes filenames
        self._cube_numbers = None  # list of the sorted number corresponding to each cube file
        self.dates = None  # list of the date with .year, .month, etc
        self.cubes = None  # all the data with values [1, 3, 5, 7]
        self.cubes_all_data = None  # boolean 4D array of all the data
        self.cubes_lineofsight_STEREO = None  # boolean 4D array of the line of sight data seen from STEREO
        self.cubes_lineofsight_SDO = None  # same for SDO
        self.cubes_no_duplicates_SDO = None  # boolean 4D array of the no duplicates seen from SDO
        self.cubes_no_duplicates_STEREO = None  # same for STEREO
        self.cubes_no_duplicate = None  # same for no duplicates
        self.trace_cubes = None  # boolean 3D array of all the data
        self.trace_cubes_no_duplicate = None  # same for the no_duplicate data
        self.day_cubes_all_data = None  # boolean 4D array of the integration of all_data for each days
        self.day_cubes_no_duplicate = None  # same for no_duplicate
        self.day_index = None  # list with len(axis0) being the number of days and axis1 being the axis0 index in self.cubes (so for most of the cubes)
        self.time_cubes_all_data = None  # boolean 4D array of the integration of all_data over time_interval
        self.time_cubes_no_duplicate = None  # same for no_duplicate
        self._date_min = None  # minimum date in seconds for each time_chunk
        self._date_max = None  # maximum date in seconds for each time_chunk
        self.radius_index = None  # radius of the Sun in grid units
        self.sun_center = None  # position of the Sun's center [x, y, z] in the grid
        self._texture_height = None  # height in pixels of the input texture image
        self._texture_width = None  # width in pixels of the input texture image 
        self.sun_texture = None  # Sun's texture image after some visualisation treatment
        self.sun_points = None  # positions of the pixels for the Sun's texture
        self._sun_texture_x = None  # 1D array with values corresponding to the height texture image indexes and array indexes to the theta direction position
        self._sun_texture_y = None  # same for width and phi direction
        self.hex_colours = None  # 1D array with values being integer hex colours and indexes being the position in the Sun's surface
        self.stars_points = None  # position of the stars

        # Functions
        self.Paths()
        self.Names()
        self.Dates_n_times()
        self.Uploading_data()
        self.Sun_pos()
        self.Choices()

    def Choices(self):
        """
        To choose what is computed and added depending on the arguments choices
        """

        if self.sun:
            self.Sun_texture()
            self.Sun_points()
            self.Colours_1D()
        
        if self.stars:
            self.Stars()

        if self.time_intervals_all_data or self.time_intervals_no_duplicate:
            self.Time_interval()
            self.Time_chunks_choices()

        if self.sdo_pov:
            self.SDO_stats()
        elif self.stereo_pov:
            self.STEREO_stats()

    def Paths(self):
        """
        Input and output paths manager.
        """

        main_path = '/home/avoyeux/old_project/avoyeux/'
        self.paths = {'Main': main_path,
                      'Cubes': os.path.join(main_path, 'Cubes_karine'),
                      'Textures': os.path.join(main_path, 'Textures'),
                      'Intensities': os.path.join(main_path, 'STEREO', 'int'),
                      'SDO': os.path.join(main_path, 'sdo')}

    def Names(self):
        """
        To get the file names of all the cubes.
        """

        # Setting the cube name pattern (only cube{:03d}.save files are kept)
        pattern = re.compile(r'cube(\d{3})\.save')

        # Getting the cube names and sorting it so that they are in the right order
        self._cube_names = [cube_name for cube_name in os.listdir(self.paths['Cubes']) \
                      if pattern.match(cube_name)]
        self._cube_names.sort()

        # Getting the corresponding cube_numbers 
        self._cube_numbers = [int(pattern.match(cube_name).group(1)) for cube_name in self._cube_names]

    def Dates_n_times(self):
        """
        To get the dates and times corresponding to the cube numbers. 
        To do so images where both numbers are in the filename are used.
        """

        pattern = re.compile(r'\d{4}_(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2})\.\d{3}\.png')
        all_filenames = glob.glob(os.path.join(self.paths['Intensities'], '*.png'))

        # Getting the corresponding filenames 
        filenames = []
        for number in self._cube_numbers:
            for filepath in all_filenames:
                filename = os.path.basename(filepath)
                if filename[:4] == f'{number:04d}':
                    filenames.append(filename)
                    break
        self.dates = [CustomDate.parse_date(pattern.match(filename).group(1)) for filename in filenames]

    def Uploading_data(self):
        """
        Uploading and preparing the data.
        """

        # Importing the necessary data
        cubes = [readsav(os.path.join(self.paths['Cubes'], cube_name)).cube for cube_name in self._cube_names]
        self.cubes = np.array(cubes)  # all data

        # Importing line_of_sight_data
        if self.line_of_sight:  
            cubes1 = [readsav(os.path.join(self.paths['Cubes'], cube_name)).cube1 for cube_name in self._cube_names]  
            cubes2 = [readsav(os.path.join(self.paths['Cubes'], cube_name)).cube2 for cube_name in self._cube_names]
            self.cubes_lineofsight_STEREO = np.array(cubes1).astype('uint8')  # line of sight seen from STEREO 
            self.cubes_lineofsight_SDO = np.array(cubes2).astype('uint8')  # line of sight seen from SDO

        # Separating the data
        if self.all_data or self.time_intervals_all_data:
            self.cubes_all_data = self.cubes != 0
            self.cubes_all_data = self.cubes_all_data.astype('uint8')
        if self.duplicates:
            self.cubes_no_duplicates_SDO = (self.cubes == 3) | (self.cubes==7)  # no duplicates seen from SDO
            self.cubes_no_duplicates_STEREO = (self.cubes == 5) | (self.cubes==7)  # no duplicates seen from STEREO
            self.cubes_no_duplicates_SDO = self.cubes_no_duplicates_SDO.astype('uint8')
            self.cubes_no_duplicates_STEREO = self.cubes_no_duplicates_STEREO.astype('uint8')
        if self.no_duplicate or self.trace_no_duplicate or self.day_trace_no_duplicate or self.time_intervals_no_duplicate:
            self.cubes_no_duplicate = (self.cubes == 7)  # no  duplicates
            self.cubes_no_duplicate = self.cubes_no_duplicate.astype('uint8')

        # Other useful data
        if self.trace_data:
            self.trace_cubes = np.any(self.cubes, axis=0)  # the "trace" of all the data
            self.trace_cubes = self.trace_cubes.astype('uint8')
        if self.trace_no_duplicate:
            self.trace_cubes_no_duplicate = np.any(self.cubes_no_duplicate, axis=0)  # the "trace" of the no duplicates data
            self.trace_cubes_no_duplicate = self.trace_cubes_no_duplicate.astype('uint8')

        # Trace by day 
        if self.day_trace:
            self.day_cubes_all_data = self.Day_cubes(self.cubes)
        if self.day_trace_no_duplicate:
            self.day_cubes_no_duplicate = self.Day_cubes(self.cubes_no_duplicate)

    def Day_cubes(self, cubes):
        """
        To integrate the data for each day.
        The input being the used data set and the output being an np.ndarray of axis0 length equal to the number of days.
        """

        days_all = np.array([date.day for date in self.dates])
        days_unique = np.unique(days_all)

        day_cubes = []
        days_indexes = []
        for day in days_unique:
            day_indexes = np.where(days_all==day)[0]
            day_trace = np.any(cubes[day_indexes], axis=0)
            day_cubes.append(day_trace)
            days_indexes.append(day_indexes)
        
        self.day_index = days_indexes
        return np.array(day_cubes).astype('uint8')

    def Time_interval(self):
        """
        Checking the date interval type an assigning a value to be used in another function to get the min and max date.
        """

        if isinstance(self.time_interval, int):
            time_delta = self.time_interval * 3600
        elif isinstance(self.time_interval, str):
            number = re.search(r'\d+', self.time_interval)
            number = int(number.group())
            self.time_interval = self.time_interval.lower()
            if 'min' in self.time_interval: 
                time_delta = number * 60
            elif 'h' in self.time_interval:
                time_delta = number * 3600
            elif 'd' in self.time_interval:
                time_delta = number * 3600 * 24
            else:
                print('Date interval format not supported. Please add min for minutes, h for hours and d for days.')
        else:
            print('Date interval format not supported. Please add min for minutes, h for hours and d for days.')
        
        self.time_interval = time_delta
    
    def Time_chunks_choices(self):
        """
        To integrate the data given a time chunk. 
        """

        self.days_per_month = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        date = self.dates[0]

        if (date.year % 4 == 0 and date.year % 100 !=0) or (date.year % 400 == 0):  # Only works if the year doesn't change
            self.days_per_month[2] = 29  # for leap years

        if self.time_intervals_all_data:
            self.time_cubes_all_data = []
        if self.time_intervals_no_duplicate:
            self.time_cubes_no_duplicate = [] 

        for date in self.dates:
            date_seconds = (((self.days_per_month[date.month] + date.day) * 24 + date.hour) * 60 + date.minute) * 60 \
                + date.second

            self._date_min = date_seconds - self.time_interval / 2
            self._date_max = date_seconds + self.time_interval / 2
        
            if self.time_intervals_all_data:
                self.time_cubes_all_data.append(self.Time_chunks(self.cubes_all_data))
            if self.time_intervals_no_duplicate:
                self.time_cubes_no_duplicate.append(self.Time_chunks(self.cubes_no_duplicate))
        
        if self.time_intervals_all_data:
            self.time_cubes_all_data = np.array([np.any(data, axis=0) for data in self.time_cubes_all_data]).astype('uint8')
        if self.time_intervals_no_duplicate:
            self.time_cubes_no_duplicate = np.array([np.any(data, axis=0) for data in self.time_cubes_no_duplicate]).astype('uint8')

    def Time_chunks(self, cubes):
        """
        To select the data in the time chunk given the data chosen for the integration.
        """

        chunk = []
        for date2, data2 in zip(self.dates, cubes):
            date_seconds2 = (((self.days_per_month[date2.month] + date2.day) * 24 + date2.hour) * 60 + date2.minute) * 60 \
                + date2.second

            if date_seconds2 < self._date_min:
                continue
            elif date_seconds2 <= self._date_max:
                chunk.append(data2)
            else:
                break
        return np.array(chunk)  

    def Sun_pos(self):
        """
        To find the Sun's radius and the center position in the cubes reference frame.
        """

        # Reference data 
        first_cube_name = os.path.join(self.paths['Cubes'], self._cube_names[0])

        # Initial data values
        solar_r = 6.96e5 
        self._length_dx = readsav(first_cube_name).dx
        self._length_dy = readsav(first_cube_name).dy
        self._length_dz = readsav(first_cube_name).dz
        self._x_min = readsav(first_cube_name).xt_min
        self._y_min = readsav(first_cube_name).yt_min
        self._z_min = readsav(first_cube_name).zt_min

        # The Sun's radius
        self.radius_index = solar_r / self._length_dx  # TODO: need to change this if dx!=dy!=dz.

        # The Sun center's position
        x_index = self._x_min / self._length_dx 
        y_index = self._y_min / self._length_dy 
        z_index = self._z_min / self._length_dz 
        self.sun_center = np.array([0 - x_index, 0 - y_index, 0 - z_index])

    def Sun_texture(self):
        """
        To create upload the Sun's texture and do the usual treatment so that the contrasts are more visible.
        A logarithmic intensity treatment is done with a saturation of really high an really low intensities.
        """

        # Importing AIA 33.5nm synoptics map
        hdul = fits.open(os.path.join(self.paths['Textures'], 'syn_AIA_304_2012-07-23T00-00-00_a_V1.fits'))
        image = hdul[0].data  # (960, 1920) monochromatic image

        # Image shape
        self._texture_height, self._texture_width = image.shape

        # Image treatment
        lower_cut = np.nanpercentile(image, 0.5)
        upper_cut = np.nanpercentile(image, 99.99)
        image[image < lower_cut] = lower_cut
        image[image > upper_cut] = upper_cut

        # Replacing nan values to the lower_cut 
        nw_image = np.where(np.isnan(image), lower_cut, image)  # TODO: would need to change the nan values to the interpolation for the pole
        nw_image = np.flip(nw_image, axis=0)
        # Changing values to a logarithmic scale
        self.sun_texture = np.log(nw_image)

    def Sun_points(self):
        """
        Creates a spherical cloud of points that represents the pixels on the Sun's surface.
        """

        # Initialisation
        N = self.sun_texture_resolution  # number of points in the theta direction
        theta = np.linspace(0, np.pi, N)  # latitude of the points
        phi = np.linspace(0, 2 * np.pi, 2 * N)  # longitude of the points
        theta, phi = np.meshgrid(theta, phi)  # the subsequent meshgrid

        # Conversion to cartesian coordinates
        x = self.radius_index * np.sin(theta) * np.cos(phi) + self.sun_center[0]
        y = self.radius_index * np.sin(theta) * np.sin(phi) + self.sun_center[1]
        z = self.radius_index * np.cos(theta) + self.sun_center[2] 

        # Creation of the position of the spherical cloud of points
        self.sun_points = np.array([x, y, z]).T
        self.sun_points = self.sun_points.astype('float32') 

        # The corresponding image indexes to get the colors
        self._sun_texture_x = np.linspace(0, self._texture_height - 1, self.sun_points.shape[0]).astype('int')
        self._sun_texture_y = np.linspace(0, self._texture_width - 1, self.sun_points.shape[1]).astype('int')

    def Colours_1D(self):
        """
        Creates a 1D array of the integer Hex pixel values (0x000000 format) of a 2D sun texture image.
        """

        x_indices = self._sun_texture_x[:, np.newaxis]
        y_indices = self._sun_texture_y[np.newaxis, :]

        colours = self.sun_texture[x_indices, y_indices].flatten()
        normalized_colours = (colours - np.min(colours)) / (np.max(colours) - np.min(colours))
        blue_val = (normalized_colours * 255).astype('int')
        self.hex_colours = (blue_val << 16) + (blue_val << 8) + blue_val
        self.hex_colours = self.hex_colours.astype('uint32')

    def Stars(self):
        """
        Creating stars to then add to the background.
        """

        # Stars spherical positions 
        stars_N = 1500  # total number of stars
        stars_radius = np.random.uniform(self.radius_index * 150, self.radius_index * 200, stars_N)
        stars_theta = np.random.uniform(0, np.pi, stars_N)
        stars_phi = np.random.uniform(0, 2 * np.pi, stars_N)

        # To cartesian
        stars_x = stars_radius * np.sin(stars_theta) * np.cos(stars_phi) + self.sun_center[0]
        stars_y = stars_radius * np.sin(stars_theta) * np.sin(stars_phi) + self.sun_center[1]
        stars_z = stars_radius * np.cos(stars_theta) + self.sun_center[2]

        # Cartesian positions
        self.stars_points = np.array([stars_x, stars_y, stars_z]).T
        self.stars_points = self.stars_points.astype('float32')

    def SDO_stats(self):
        """
        To save the information needed to find the position of SDO.
        """

        from astropy import units as u
        from astropy.coordinates import CartesianRepresentation
        from sunpy.coordinates.frames import  HeliographicCarrington


        SDO_fits_names = [os.path.join(self.paths['SDO'], f'AIA_fullhead_{number:03d}.fits.gz')
                           for number in self._cube_numbers]
        
        SDO_pos = []

        for fits_name in SDO_fits_names:
            hdul = fits.open(fits_name)
            header = hdul[0].header


            hec_coords = HeliographicCarrington(header['CRLN_OBS']* u.deg, header['CRLT_OBS'] * u.deg, 
                                                header['DSUN_OBS'] * u.m, obstime=header['DATE-OBS'], 
                                                observer='self')
            hec_coords = hec_coords.represent_as(CartesianRepresentation)

            Xhec = hec_coords.x.value
            Yhec = hec_coords.y.value
            Zhec = hec_coords.z.value

            xpos_index = Xhec / (1000 * self._length_dx)
            ypos_index = Yhec / (1000 * self._length_dy)
            zpos_index = Zhec / (1000 * self._length_dz)

            SDO_pos.append(self.sun_center + np.array([xpos_index, ypos_index, zpos_index]))
            hdul.close()  
        self.SDO_pos = np.array(SDO_pos)

    def STEREO_stats(self):
        """
        To save the information needed to find the position of STEREO.
        """

        from astropy import units as u
        from astropy.coordinates import CartesianRepresentation
        from sunpy.coordinates.frames import  HeliographicCarrington

        data = readsav(os.path.join(self.paths['Main'], 'rainbow_stereob_304.save')).datainfos
        stereo_pos = []

        for number in self._cube_numbers:
            stereo_lon = data[number].lon
            stereo_lat = data[number].lat
            stereo_dsun = data[number].dist
            stereo_date = data[number].strdate

            stereo_date = CustomDate.parse_date2(stereo_date)
            stereo_date = f'{stereo_date.year}-{stereo_date.month}-{stereo_date.day}T{stereo_date.hour}:{stereo_date.minute}:{stereo_date.second}'

            hec_coords = HeliographicCarrington(stereo_lon * u.deg, stereo_lat * u.deg, stereo_dsun * u.km,
                                                obstime=stereo_date, observer='self')
            hec_coords = hec_coords.represent_as(CartesianRepresentation)

            Xhec = hec_coords.x.value
            Yhec = hec_coords.y.value
            Zhec = hec_coords.z.value

            xpos_index = Xhec / ( self._length_dx)
            ypos_index = Yhec / ( self._length_dy)
            zpos_index = Zhec / ( self._length_dz)

            stereo_pos.append(self.sun_center + np.array([xpos_index, ypos_index, zpos_index])) 
        self.stereo_pos = np.array(stereo_pos)          

class K3dAnimation(Data):
    """
    Creates the corresponding k3d animation to then be used in a Jupyter notebook file.
    """

    def __init__(self, compression_level=9, plot_height=1220, sleep_time=2, camera_fov=1, camera_zoom_speed=0.7, 
                 trace_opacity=0.1, make_screenshots=False, screenshot_scale=2, screenshot_sleep=5, 
                 screenshot_version='v0', **kwargs):
        
        super().__init__(**kwargs)

        # Arguments
        self.compression_level = compression_level  # the compression level of the data in the 3D visualisation
        self.plot_height = plot_height  # the height in pixels of the plot (initially it was 512)
        self.sleep_time = sleep_time  # sets the time between each frames (in seconds)
        self.camera_fov = camera_fov  # the name speaks for itself
        self.camera_zoom_speed = camera_zoom_speed  # zoom speed of the camera 
        self.trace_opacity = trace_opacity  # opacity factor for all the trace voxels
        self.make_screenshots = make_screenshots  # creating screenshots when clicking play
        self.screenshot_scale = screenshot_scale  # the 'resolution' of the screenshot 
        self.screenshot_sleep = screenshot_sleep  # sleep time between each screenshot as synchronisation time is needed
        self.version = screenshot_version  # to save the screenshot with different names if multiple screenshots need to be saved

        # Instance attributes set when running the class
        self.plot = None  # k3d plot object
        self.init_plot = None  # voxels plot of the all data 
        self.init_plot1 = None  # voxels plot of the no duplicates seen from SDO
        self.init_plot2 = None  # voxels plot of the no duplicates seen from STEREO
        self.init_plot3 = None  # voxels plot of the no duplicate 
        self.init_plot4 = None  # voxels plot of the line of sight from STEREO
        self.init_plot5 = None  # voxels plot of the line of sight from SDO 
        self.play_pause_button = None  # Play/Pause widget initialisation
        self.time_slider = None  # time slider widget
        self.date_dropdown = None  # Date dropdown widget to show the date
        self.time_link = None  # JavaScript Link between the two widgets
        self.date_text = None  # list of strings giving the text associated to each time frames 

        # Making the animation
        self.Update_paths()
        if self.time_intervals_all_data or self.time_intervals_no_duplicate:
            self.Time_interval_string()
        self.Date_strings()
        self.Animation()

    def Update_paths(self):
        """
        Updating the paths of the parent class to be able to save screenshots.
        """

        if self.make_screenshots:
            self.paths['Screenshots'] = os.path.join(self.paths['Main'], 'texture_screenshot2')
            os.makedirs(self.paths['Screenshots'], exist_ok=True)

    def Camera_params(self):
        """
        Camera visualisation parameters.
        """
 
        self.plot.camera_auto_fit = False
        self.plot.camera_fov = self.camera_fov  # FOV in degrees
        self.plot.camera_zoom_speed = self.camera_zoom_speed  # it was zooming too quickly (default=1.2)
        
        # Point to look at, i.e. initial rotational reference
        self._camera_reference = np.array([self.cubes.shape[3], self.cubes.shape[2], self.cubes.shape[1]]) / 2  # I got no clue why it is the other way around than for the cubes, but I tested it.

        if self.sdo_pov:
            self.plot.camera = [self.SDO_pos[0, 0], self.SDO_pos[0, 1], self.SDO_pos[0, 2],
                        self._camera_reference[0], self._camera_reference[1], self._camera_reference[2],
                        0, 0, 1]  # up vector
        elif self.stereo_pov:
            self.plot.camera = [self.stereo_pos[0, 0], self.stereo_pos[0, 1], self.stereo_pos[0, 2],
                                self._camera_reference[0], self._camera_reference[1], self._camera_reference[2],
                                0, 0, 1]  # up vector
        else:
            au_in_solar_r = 215  # 1 au in solar radii
            distance_to_sun = au_in_solar_r * self.radius_index 

            self.plot.camera = [self._camera_reference[0] - distance_to_sun, self._camera_reference[1] - distance_to_sun / 2, 0,
                        self._camera_reference[0], self._camera_reference[1], self._camera_reference[2],
                        0, 0, 1]  # up vector

    def Update_voxel(self, change):
        """
        Updates the plots depending on which time frame you want to be shown. 
        Also creates the screenshots if it is set to True.
        """

        if self.sdo_pov:
            self.plot.camera = [self.SDO_pos[change['new'], 0], self.SDO_pos[change['new'], 1], self.SDO_pos[change['new'], 2],
                                self._camera_reference[0], self._camera_reference[1], self._camera_reference[2],
                                0, 0, 1]
            time.sleep(0.3)
        elif self.stereo_pov:
            self.plot.camera = [self.stereo_pos[change['new'], 0], self.stereo_pos[change['new'], 1], self.stereo_pos[change['new'], 2],
                                self._camera_reference[0], self._camera_reference[1], self._camera_reference[2],
                                0, 0, 1]
            time.sleep(0.3)          
        
        if self.all_data:
            self.init_plot.voxels = self.cubes_all_data[change['new']]
        if self.duplicates:
            self.init_plot1.voxels = self.cubes_no_duplicates_SDO[change['new']]
            self.init_plot2.voxels = self.cubes_no_duplicates_STEREO[change['new']]
        if self.no_duplicate:
            self.init_plot3.voxels = self.cubes_no_duplicate[change['new']]
        if self.line_of_sight:
            self.init_plot4.voxels = self.cubes_lineofsight_STEREO[change['new']]
            self.init_plot5.voxels = self.cubes_lineofsight_SDO[change['new']]
        if self.time_intervals_all_data:
            self.init_plot6.voxels = self.time_cubes_all_data[change['new']]
        if self.time_intervals_no_duplicate:
            self.init_plot7.voxels = self.time_cubes_no_duplicate[change['new']]
        if self.day_trace or self.day_trace_no_duplicate:
            for day_nb, day_index in enumerate(self.day_index):
                if (change['new'] in day_index) and (change['old'] not in day_index):
                    if self.day_trace:
                        self.init_plot8.voxels = self.day_cubes_all_data[day_nb]
                    if self.day_trace_no_duplicate:
                        self.init_plot9.voxels = self.day_cubes_no_duplicate[day_nb]
                    break
        if self.make_screenshots:
            self.Screenshot_making()


    def Play(self):
        """
        Params for the play button.
        """
        
        if self.play_pause_button.value and self.time_slider.value < len(self.cubes) - 1:
            self.time_slider.value += 1
            threading.Timer(self.sleep_time, self.Play).start()  # where you also set the sleep() time.
                
        else:
            self.play_pause_button.description = 'Play'
            self.play_pause_button.icon = 'play'

    def Screenshot_making(self):
        """
        To create a screenshot of the plot. A sleep time was added as the screenshot method relies
        on asynchronus traitlets mechanism.
        """

        import base64

        self.plot.fetch_screenshot()

        time.sleep(self.screenshot_sleep)

        screenshot_png = base64.b64decode(self.plot.screenshot)
        if self.time_intervals_all_data or self.time_intervals_no_duplicate:
            screenshot_name = f'interval{self.time_interval}_{self.time_slider.value}_{self.version}.png'
        else:
            screenshot_name = f'Plot_{self.time_slider.value:03d}_{self.date_text[self.time_slider.value]}_{self.version}.png'
        screenshot_namewpath = os.path.join(self.paths['Screenshots'], screenshot_name)
        with open(screenshot_namewpath, 'wb') as f:
            f.write(screenshot_png)

    def Play_pause_handler(self, change):
        """
        Changes the play button to pause when it is clicked.
        """

        if change['new']:  # if clicked play
            self.Play()
            self.play_pause_button.description = 'Pause'
            self.play_pause_button.icon = 'pause'
        else:  
            pass

    def Date_strings(self):
        """
        Uses the dates for the files to create a corresponding string list.
        """

        self.date_text = [f'{date.year}-{date.month:02d}-{date.day:02d}_{date.hour:02d}h{date.minute:02d}min'
                          for date in self.dates]

    def Time_interval_string(self):
        """
        To change self.time_interval to a string giving a value in day, hours or minutes.
        """

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
        elif self.time_interval < 3600 * 24 * 2:
            time_interval = f'{self.time_interval // (3600 * 24)}days'
            if self.time_interval % (3600 * 24) != 0:
                self.time_interval = time_interval + f'{(self.time_interval % 3600 * 24) // 3600}h'
            else:
                self.time_interval = time_interval
        else:
            print("Time interval is way too large")

    def Animation(self):
        """
        Creates the 3D animation using k3d. 
        """
        
        # Initialisation of the plot
        self.plot = k3d.plot(grid_visible=False, background_color=0x000000)  # plot with no axes and a dark background
        # self.plot = k3d.plot(grid_visible=True)
        self.plot.height = self.plot_height  
            
        # Adding the camera specific parameters
        self.Camera_params()
        
        # Adding the SUN!!!
        if self.sun:
            self.plot += k3d.points(positions=self.sun_points, point_size=2.5, colors=self.hex_colours, shader='flat',
                                    name='SUN', compression_level=self.compression_level)
        
        # Adding the stars      
        if self.stars: 
            self.plot += k3d.points(positions=self.stars_points, point_size=100, color=0xffffff, shader='3d', name='Stars', 
                            compression_level=self.compression_level)

        # Adding the different data sets (i.e. with or without duplicates)
        if self.all_data:
            self.init_plot = k3d.voxels(self.cubes_all_data[0], outlines=False, opacity=0.4, 
                                        compression_level=self.compression_level, color_map=[0x90ee90], name='All data')
            self.plot += self.init_plot
      
        if self.duplicates:
            self.init_plot1 = k3d.voxels(self.cubes_no_duplicates_SDO[0], compression_level=self.compression_level, outlines=True, 
                                        color_map=[0xff0000], name='No duplicates from SDO')
            self.init_plot2 = k3d.voxels(self.cubes_no_duplicates_STEREO[0], compression_level=self.compression_level, outlines=True, 
                                        color_map=[0x0000ff], name='No duplicates from STEREO')
            self.plot += self.init_plot1
            self.plot += self.init_plot2
               
        if self.no_duplicate:
            self.init_plot3 = k3d.voxels(self.cubes_no_duplicate[0], compression_level=self.compression_level, outlines=True, 
                                        color_map=[0x8B0000], name='No duplicates')
            self.plot += self.init_plot3

        if self.line_of_sight:
            self.init_plot4 = k3d.voxels(self.cubes_lineofsight_STEREO[0], compression_level=self.compression_level, outlines=True, 
                                        color_map=[0x0000ff], name='Seen from Stereo')
            self.init_plot5 = k3d.voxels(self.cubes_lineofsight_SDO[0], compression_level=self.compression_level, outlines=True, 
                                        color_map=[0xff0000], name='Seen from SDO')
            self.plot += self.init_plot4
            self.plot += self.init_plot5

        if self.time_intervals_all_data:
            self.init_plot6 = k3d.voxels(self.time_cubes_all_data[0], compression_level=self.compression_level, outlines=False, 
                                        color_map=[0xff6666],opacity=1, 
                                        name=f'All data for {self.time_interval}')
            self.plot += self.init_plot6
       
        if self.time_intervals_no_duplicate:
            self.init_plot7 = k3d.voxels(self.time_cubes_no_duplicate[0], compression_level=self.compression_level, outlines=False,
                                        color_map=[0xff6666], opacity=1,
                                        name=f'No duplicate for {self.time_interval}')
            self.plot += self.init_plot7
        
        if self.trace_data:
            self.plot += k3d.voxels(self.trace_cubes, compression_level=self.compression_level, outlines=False, color_map=[0xff6666],
                                    opacity=self.trace_opacity, name='Trace of all the data')
        
        if self.trace_no_duplicate:
            self.plot += k3d.voxels(self.trace_cubes_no_duplicate, compression_level=self.compression_level, outlines=False, 
                                    color_map=[0xff6666], opacity=self.trace_opacity, name='Trace of the no duplicates')
            
        if self.day_trace:
            date = self.dates[0]
            self.init_plot8 = k3d.voxels(self.day_cubes_all_data[0], compression_level=self.compression_level, outlines=False,
                                         color_map=[0xff6666], opacity=self.trace_opacity, 
                                         name=f'All data trace for {date.month:02d}.{date.day:02d}')
            self.plot += self.init_plot8

        if self.day_trace_no_duplicate:
            date = self.dates[0]
            self.init_plot9 = k3d.voxels(self.day_cubes_no_duplicate[0], compression_level=self.compression_level, outlines=False,
                                         color_map=[0xff6666], opacity=self.trace_opacity, 
                                         name=f'No duplicate trace for {date.month:02d}.{date.day:02d}') #TODO: change the name for each day
            self.plot += self.init_plot9

        # Adding a play/pause button
        self.play_pause_button = widgets.ToggleButton(value=False, description='Play', icon='play')

        # Set up the time slider and the play/pause button
        self.time_slider = widgets.IntSlider(min=0, max=len(self.cubes)-1, description='Frame:')
        self.date_dropdown = widgets.Dropdown(options=self.date_text, description='Date:')
        self.time_slider.observe(self.Update_voxel, names='value')
        self.time_link= widgets.jslink((self.time_slider, 'value'), (self.date_dropdown, 'index'))
        self.play_pause_button.observe(self.Play_pause_handler, names='value')

        # Display
        display(self.plot, self.time_slider, self.date_dropdown, self.play_pause_button)
        if self.make_screenshots:
            self.plot.screenshot_scale = self.screenshot_scale
            self.Screenshot_making()
