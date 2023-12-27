"""
Code to then put in the jupyter notebook (just because I prefer .py files).
"""

import os
import re
import k3d
import time
import glob
import threading
import numpy as np
import ipywidgets as widgets

from astropy.io import fits
from scipy.io import readsav
from scipy.sparse import coo_matrix
from typeguard import typechecked
from IPython.display import display


class CustomDate:
    """
    To separate the year, month, day, hour, minute, second from a string dateutil.parser.parser
    doesn't work in this case. 
    """

    def __init__(self, year: int, month: int, day: int, hour: int, minute: int, second: int):
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

    @typechecked
    def __init__(self, everything: bool = False, both_cubes: str | bool = 'Alfred', sun: bool = False, stars: bool = False, 
                 all_data: bool = False, duplicates: bool = False, no_duplicate: bool = False, line_of_sight: bool = False, 
                 trace_data: bool = False, trace_no_duplicate: bool = False, day_trace: bool = False, 
                 day_trace_no_duplicate: bool = False, time_intervals_all_data: bool = False, 
                 time_intervals_no_duplicate: bool = False, time_interval: str | int = 1, sun_texture_resolution: int = 960,
                 sdo_pov: bool = False, stereo_pov: bool = False, memory_saver: bool = True):
        
        # Arguments
        self.first_cube = False
        self.second_cube = False
        if isinstance(both_cubes, bool):
            self.first_cube = (both_cubes or everything)
            self.second_cube = (both_cubes or everything)
        elif isinstance(both_cubes, str):
            if 'alf' in both_cubes.lower():
                self.first_cube = True
            elif 'kar' in both_cubes.lower():
                self.second_cube = True
            elif not everything:
                raise ValueError('Wrong string value for argument both_cubes. It has to contain "alf" or "kar".')
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
        self._sun_texture_resolution = sun_texture_resolution  # choosing the Sun's texture resolution
        self.sdo_pov = sdo_pov
        self.stereo_pov = stereo_pov
        self.memory_saver = memory_saver  # to choose to use sparse 4D np.ndarrays to save memory

        # Instance attributes set when running the class
        self.paths = None  # dictionary containing all the path names and the corresponding path
        self._cube_names_all = None  # all the cube names that will be used
        self._cube_names_1 = None  # cube names for the first set 
        self._cube_names_2 = None  # for the second set
        self.cube_numbers_all = None  # sorted list of the number corresponding to each used cube
        self._cube_numbers_1 = None  # same but only for the first set
        self._cube_numbers_2 = None  # same for the second set
        self.dates_all = None  # list of the date with .year, .month, etc for all the cubes
        self.dates_1 = None  # same for the first set of cubes
        self.dates_2 = None  # same for the second set of cubes 
        self.cubes_shape = None  # the shape of cube 1 or cube 2 if cube 2 is also chosen
        self.cubes_all_data_1 = None  # (sparse or not) boolean 4D array of all the data for the first set
        self.cubes_all_data_2 = None  # same for the second set 
        self.cubes_lineofsight_STEREO_1 = None  # (sparse or not) boolean 4D array of the line of sight data seen from STEREO for the first set
        self.cubes_lineofsight_STEREO_2 = None  # same for the second set
        self.cubes_lineofsight_SDO_1 = None  # same for SDO and the first set
        self.cubes_lineofsight_SDO_2 = None  # same for the second set
        self.cubes_no_duplicate_1 = None  # (sparse or not) boolean 4D array for the no duplicates for the first set
        self.cubes_no_duplicate_2 = None  # same for the second set
        self.cubes_no_duplicates_STEREO_1 = None  # same seen from STEREO for the first set
        self.cubes_no_duplicates_STEREO_2 = None  # same for the second set
        self.cubes_no_duplicates_SDO_1 = None  # same seen from SDO for the first set
        self.cubes_no_duplicates_SDO_2 = None  # same for the second set 
        self.trace_cubes_1 = None  # boolean 3D array of all the data for the first set 
        self.trace_cubes_2 = None  # same for the second set
        self.trace_cubes_no_duplicate_1 = None  # same  for the no duplicate data for the first set
        self.trace_cubes_no_duplicate_2 = None  # same for the second set 
        self.day_cubes_all_data_1 = None  # (sparse or not) boolean 4D array of the integration of all_data for each days for the first set
        self.day_cubes_all_data_2 = None  # same for the second set 
        self.day_cubes_no_duplicate_1 = None  # same for the no duplicate data the first set
        self.day_cubes_no_duplicate_2 = None  # same for the second set 
        self.day_indexes = None  # list of the data cube index for each day. index[n][m]=x is for day n the cube index is x. For all chosen sets
        self.day_indexes_1 = None  # same for the first set
        self.day_indexes_2 = None  # same for the second set
        self.time_cubes_all_data_1 = None  # (sparse or not) boolean 4D array of the integration of all_data over time_interval for the first set
        self.time_cubes_all_data_2 = None  # same for the second set
        self.time_cubes_no_duplicate_1 = None  # same for the no duplicate for the first set
        self.time_cubes_no_duplicate_2 = None  # same for the second set 
        self._date_min = None  # minimum date in seconds for each time_chunk
        self._date_max = None  # maximum date in seconds for each time_chunk
        self.radius_index = None  # radius of the Sun in grid units
        self._sun_center = None  # position of the Sun's center [x, y, z] in the grid
        self._texture_height = None  # height in pixels of the input texture image
        self._texture_width = None  # width in pixels of the input texture image 
        self._sun_texture = None  # Sun's texture image after some visualisation treatment
        self.sun_points = None  # positions of the pixels for the Sun's texture
        self._sun_texture_x = None  # 1D array with values corresponding to the height texture image indexes and array indexes to the theta direction position
        self._sun_texture_y = None  # same for width and phi direction
        self.hex_colours = None  # 1D array with values being integer hex colours and indexes being the position in the Sun's surface
        self.stars_points = None  # position of the stars
        self._pattern_int = None  # re.compile pattern of the 'int' STEREO .png filenames
        self._all_filenames = None  # all the 'int' filenames for both cubes
        self._days_per_month = None  # list of the number of day per each month in a normal year
        self._length_dx = None  # x direction unit voxel size in km
        self._length_dy = None  # same for the y direction
        self._length_dz = None  # same for the z direction
        self.SDO_pos = None  # array giving the position of the SDO satellite for each time step
        self.STEREO_pos = None  # same for STEREO

        # Functions
        self.Paths()
        self.Names()
        self.Sun_pos()
        self.Dates_all_data_sets()            
        self.Choices()

        # Deleting the private class attributes
        self.Attribute_deletion()
    
    def Attribute_deletion(self):
        """
        To delete some of the attributes that are not used in the inherited class. Done to save some RAM.
        """

        # Private attributes 
        del self._sun_texture_resolution, self._cube_names_all, self._cube_names_1, self._cube_names_2, self._cube_numbers_1, self._cube_numbers_2
        del self._date_max, self._date_min, self._sun_center, self._texture_height, self._texture_width, self._sun_texture, self._sun_texture_x, self._sun_texture_y
        del self._pattern_int, self._all_filenames, self._days_per_month, self._length_dx, self._length_dy, self._length_dz

    def Choices(self):
        """
        To choose what is computed and added depending on the arguments chosen.
        """

        if self.time_intervals_all_data or self.time_intervals_no_duplicate:
            self.Time_interval()
        
        if self.first_cube:
            self.dates_1 = self.Dates_n_times(self._cube_numbers_1)
            self.cubes_lineofsight_STEREO_1, self.cubes_lineofsight_SDO_1, self.cubes_all_data_1, \
                self.cubes_no_duplicate_1, self.cubes_no_duplicates_STEREO_1, self.cubes_no_duplicates_SDO_1, \
                self.trace_cubes_1, self.trace_cubes_no_duplicate_1, self.day_cubes_all_data_1, \
                self.day_cubes_no_duplicate_1, self.day_indexes_1 \
                    = self.Uploading_data(self.paths['Cubes'], self._cube_names_1, self.dates_1) 
            if self.time_intervals_all_data or self.time_intervals_no_duplicate:
                self.time_cubes_all_data_1, self.time_cubes_no_duplicate_1 = \
                    self.Time_chunks_choices(self.dates_1, self.cubes_all_data_1, self.cubes_no_duplicate_1)
                self.cubes_all_data_1 = self.Sparse_data(self.cubes_all_data_1)
                self.cubes_no_duplicate_1 = self.Sparse_data(self.cubes_no_duplicate_1)

        if self.second_cube:
            self.dates_2 = self.Dates_n_times(self._cube_numbers_2)
            self.cubes_lineofsight_STEREO_2, self.cubes_lineofsight_SDO_2, self.cubes_all_data_2, \
                self.cubes_no_duplicate_2, self.cubes_no_duplicates_STEREO_2, self.cubes_no_duplicates_SDO_2, \
                self.trace_cubes_2, self.trace_cubes_no_duplicate_2, self.day_cubes_all_data_2, \
                self.day_cubes_no_duplicate_2, self.day_indexes_2 \
                    = self.Uploading_data(self.paths['Cubes_karine'], self._cube_names_2, self.dates_2) 
            if self.time_intervals_all_data or self.time_intervals_no_duplicate:
                self.time_cubes_all_data_2, self.time_cubes_no_duplicate_2 = \
                    self.Time_chunks_choices(self.dates_2, self.cubes_all_data_2, self.cubes_no_duplicate_2)  
                self.cubes_all_data_2 = self.Sparse_data(self.cubes_all_data_2)
                self.cubes_no_duplicate_2 = self.Sparse_data(self.cubes_no_duplicate_2)

        if self.sun:
            self.Sun_texture()
            self.Sun_points()
            self.Colours_1D()
        
        if self.stars:
            self.Stars() 
                         
        if self.sdo_pov:
            self.SDO_stats()
        elif self.stereo_pov:
            self.STEREO_stats()

    def Paths(self):
        """
        Input and output paths manager.
        """

        main_path = '../'
        self.paths = {'Main': main_path,
                      'Cubes': os.path.join(main_path, 'Cubes'),
                      'Cubes_karine': os.path.join(main_path, 'Cubes_karine'),
                      'Textures': os.path.join(main_path, 'Textures'),
                      'Intensities': os.path.join(main_path, 'STEREO', 'int'),
                      'SDO': os.path.join(main_path, 'sdo')}

    def Names(self):
        """
        To get the file names of all the cubes.
        """

        # Setting the cube name pattern (only cube{:03d}.save files are kept)
        pattern = re.compile(r'cube(\d{3})\.save')

        # The cube names
        cube_names = []

        if self.first_cube:
            cube_names.extend([cube_name for cube_name in os.listdir(self.paths['Cubes']) \
                      if pattern.match(cube_name)])
            self._cube_names_1 = sorted(cube_names)  # first data set
            self._cube_numbers_1 = [int(pattern.match(cube_name).group(1)) for cube_name in self._cube_names_1]

        if self.second_cube:
            cube_names_2 = [cube_name for cube_name in os.listdir(self.paths['Cubes_karine']) \
                               if pattern.match(cube_name)] 
            cube_names.extend(cube_names_2)
            self._cube_names_2 = sorted(cube_names_2) # second set
            self._cube_numbers_2 = [int(pattern.match(cube_name).group(1)) for cube_name in self._cube_names_2]

        self._cube_names_all = [cube_name for cube_name in set(cube_names)]
        self._cube_names_all.sort()

        # Getting the corresponding cube_numbers 
        self.cube_numbers_all = [int(pattern.match(cube_name).group(1)) for cube_name in self._cube_names_all]

    def Dates_all_data_sets(self):
        """
        To get the dates and times corresponding to all the used cubes.
        To do so images where both numbers are in the filename are used.
        """

        self._pattern_int = re.compile(r'\d{4}_(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2})\.\d{3}\.png')
        self._all_filenames = glob.glob(os.path.join(self.paths['Intensities'], '*.png'))

        # Getting the corresponding filenames 
        filenames = []
        for number in self.cube_numbers_all: 
            for filepath in self._all_filenames:
                filename = os.path.basename(filepath)
                if filename[:4] == f'{number:04d}':
                    filenames.append(filename)
                    break
        self.dates_all = [CustomDate.parse_date(self._pattern_int.match(filename).group(1)) for filename in filenames]

    def Dates_n_times(self, cube_numbers):
        """
        To get the dates and times corresponding to each cube sets. 
        To do so images where both numbers are in the filename are used.
        """

        # Getting the corresponding filenames 
        filenames = []
        for number in cube_numbers:
            for filepath in self._all_filenames:
                filename = os.path.basename(filepath)
                if filename[:4] == f'{number:04d}':
                    filenames.append(filename)
                    break
        dates = [CustomDate.parse_date(self._pattern_int.match(filename).group(1)) for filename in filenames]
        return dates

    def Uploading_data(self, cubes_path, cube_names, dates):
        """
        Uploading and preparing the data.
        """

        # Importing the necessary data
        cubes = [readsav(os.path.join(cubes_path, cube_name)).cube for cube_name in cube_names]
        cubes = np.array(cubes)  # all data
        self.cubes_shape = cubes.shape

        # Initialisation so that the return of all the arguments works
        cubes_lineofsight_STEREO = None
        cubes_lineofsight_SDO = None
        cubes_all_data = None
        cubes_no_duplicates_STEREO = None
        cubes_no_duplicates_SDO = None
        cubes_no_duplicate = None
        trace_cube = None
        trace_cube_no_duplicate = None
        day_cubes_all_data = None
        day_cubes_no_duplicate = None
        day_indexes = None

        # Importing line_of_sight_data
        if self.line_of_sight:  
            cubes_lineofsight_STEREO = [readsav(os.path.join(cubes_path, cube_name)).cube1 for cube_name in cube_names]  
            cubes_lineofsight_SDO = [readsav(os.path.join(cubes_path, cube_name)).cube2 for cube_name in cube_names]
            cubes_lineofsight_STEREO = np.array(cubes_lineofsight_STEREO, dtype='uint8')  # line of sight seen from STEREO 
            cubes_lineofsight_SDO = np.array(cubes_lineofsight_SDO, dtype='uint8')  # line of sight seen from SDO
            cubes_lineofsight_STEREO = self.Sparse_data(cubes_lineofsight_STEREO)
            cubes_lineofsight_SDO = self.Sparse_data(cubes_lineofsight_SDO)

        # Separating the data
        if self.all_data or self.time_intervals_all_data:
            cubes_all_data = (cubes != 0).astype('uint8')
            if not self.time_intervals_all_data:
                cubes_all_data = self.Sparse_data(cubes_all_data)

        if self.no_duplicate or self.trace_no_duplicate or self.day_trace_no_duplicate or self.time_intervals_no_duplicate:
            cubes_no_duplicate = (cubes == 7).astype('uint8')  # no  duplicates
            if not (self.trace_no_duplicate or self.day_trace_no_duplicate or self.time_intervals_no_duplicate):
                cubes_no_duplicate = self.Sparse_data(cubes_no_duplicate)

        if self.duplicates:
            cubes_no_duplicates_STEREO = (cubes == 5) | (cubes==7)  # no duplicates seen from STEREO
            cubes_no_duplicates_SDO = (cubes == 3) | (cubes==7)  # no duplicates seen from SDO
            cubes_no_duplicates_STEREO = cubes_no_duplicates_STEREO.astype('uint8')
            cubes_no_duplicates_SDO = cubes_no_duplicates_SDO.astype('uint8')
            cubes_no_duplicates_STEREO = self.Sparse_data(cubes_no_duplicates_STEREO)
            cubes_no_duplicates_SDO = self.Sparse_data(cubes_no_duplicates_SDO)

        # Other useful data
        if self.trace_data:
            trace_cube = np.any(cubes, axis=0).astype('uint8')  # the "trace" of all the data
            #trace_cube = self.Sparse_data(trace_cube)

        if self.trace_no_duplicate:
            trace_cube_no_duplicate = np.any(cubes_no_duplicate, axis=0).astype('uint8')  # the "trace" of the no duplicates data
            #trace_cube_no_duplicate = self.Sparse_data(trace_cube_no_duplicate)
            if not (self.day_trace_no_duplicate or self.time_intervals_no_duplicate):
                cubes_no_duplicate = self.Sparse_data(cubes_no_duplicate)

        # Trace by day 
        if self.day_trace:
            day_cubes_all_data, day_indexes = self.Day_cubes(cubes, dates)
            day_cubes_all_data = self.Sparse_data(day_cubes_all_data)

        if self.day_trace_no_duplicate:
            day_cubes_no_duplicate, day_indexes = self.Day_cubes(cubes_no_duplicate, dates)
            day_cubes_no_duplicate = self.Sparse_data(day_cubes_no_duplicate)
            if not self.time_intervals_no_duplicate:
                cubes_no_duplicate = self.Sparse_data(cubes_no_duplicate)

        return  cubes_lineofsight_STEREO, cubes_lineofsight_SDO, cubes_all_data, \
                cubes_no_duplicate, cubes_no_duplicates_STEREO, cubes_no_duplicates_SDO, trace_cube, \
                trace_cube_no_duplicate, day_cubes_all_data, day_cubes_no_duplicate, day_indexes
    
    def Sparse_data(self, cubes):
        """
        To make the voxel positions np.ndarrays less memory heavy by taking into account that they are sparse arrays.
        """

        if self.memory_saver:
            time, depth, height, width = cubes.shape
            sparse_matrices = [coo_matrix(cubes[t].reshape(depth * height, width)) for t in range(time)]
            return sparse_matrices
        else:
            return cubes
        
    def Day_cubes(self, cubes, dates):
        """
        To integrate the data for each day.
        The input being the used data set and the output being an np.ndarray of axis0 length equal to the number of days.
        """

        days_unique = np.array([date.day for date in set(self.dates_all)])
        days = np.array([date.day for date in dates])

        day_cubes = []
        days_indexes = []
        for day in days_unique:
            day_indexes = np.where(days==day)[0]
            day_trace = np.any(cubes[day_indexes], axis=0)
            day_cubes.append(day_trace)
            days_indexes.append(day_indexes)
        return np.array(day_cubes, dtype='uint8'), days_indexes

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
                raise ValueError("Time_interval string value not supported. Add 'min' for minutes, 'h' for hours or 'd' for days.")
        else:
            raise TypeError('Problem for time interval. Typeguard should have already raised an error.')
        
        self.time_interval = time_delta
    
    def Time_chunks_choices(self, dates, cubes_all_data, cubes_no_duplicate):
        """
        To integrate the data given a time chunk. 
        """

        self._days_per_month = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        date = self.dates_all[0]

        if (date.year % 4 == 0 and date.year % 100 !=0) or (date.year % 400 == 0):  # Only works if the year doesn't change
            self._days_per_month[2] = 29  # for leap years

        time_cubes_all_data = []
        time_cubes_no_duplicate = [] 

        for date in self.dates_all:
            date_seconds = (((self._days_per_month[date.month] + date.day) * 24 + date.hour) * 60 + date.minute) * 60 \
                + date.second

            self._date_min = date_seconds - self.time_interval / 2
            self._date_max = date_seconds + self.time_interval / 2
        
            if self.time_intervals_all_data:
                time_cubes_all_data.append(self.Time_chunks(dates, cubes_all_data)) 
            if self.time_intervals_no_duplicate:
                time_cubes_no_duplicate.append(self.Time_chunks(dates, cubes_no_duplicate))
        
        if self.time_intervals_all_data:
            time_cubes_all_data = np.array(time_cubes_all_data, dtype='uint8')
            time_cubes_all_data = self.Sparse_data(time_cubes_all_data)
        if self.time_intervals_no_duplicate:
            time_cubes_no_duplicate = np.array(time_cubes_no_duplicate, dtype='uint8')
            time_cubes_no_duplicate = self.Sparse_data(time_cubes_no_duplicate)
        return time_cubes_all_data, time_cubes_no_duplicate

    def Time_chunks(self, dates, cubes):
        """
        To select the data in the time chunk given the data chosen for the integration.
        """

        chunk = []
        for date2, data2 in zip(dates, cubes):
            date_seconds2 = (((self._days_per_month[date2.month] + date2.day) * 24 + date2.hour) * 60 + date2.minute) * 60 \
                + date2.second

            if date_seconds2 < self._date_min:
                continue
            elif date_seconds2 <= self._date_max:
                chunk.append(data2)
            else:
                break
        if len(chunk) == 0:  # i.e. if nothing was found
            return np.zeros((cubes.shape[1], cubes.shape[2], cubes.shape[3]))
        elif len(chunk) == 1:
            return data2
        else:
            return np.any(chunk, axis=0)
    
    def Complete_sparse_arrays(self):
        """
        To reformat the sparse arrays so that the number of values is equal to the total number of cube numbers used.
        """

        if self.first_cube:
            cubes_lineofsight_STEREO_1 = []
            cubes_lineofsight_SDO_1 = []
            cubes_all_data_1 = []
            cubes_no_duplicate_1 = []
            cubes_no_duplicates_STEREO_1 = []
            cubes_no_duplicates_SDO_1 = []
            time_cubes_all_data_1 = []
            time_cubes_no_duplicate_1 = []
            index = -1
            for number in self.cube_numbers_all:
                if number in self._cube_numbers_1:
                    index += 1
                    cubes_lineofsight_STEREO_1.append(self.cubes_lineofsight_STEREO_1[index])
                    cubes_lineofsight_SDO_1.append(self.cubes_lineofsight_SDO_1[index])
                    cubes_all_data_1.append(self.cubes_all_data_1[index])
                    cubes_no_duplicate_1.append(self.cubes_no_duplicate_1[index])
                    cubes_no_duplicates_STEREO_1.append(self.cubes_no_duplicates_STEREO_1[index])
                    cubes_no_duplicates_SDO_1.append(self.cubes_no_duplicates_SDO_1[index])
                    time_cubes_all_data_1.append(self.time_cubes_all_data_1[index])
                    time_cubes_no_duplicate_1.append(self.time_cubes_no_duplicate_1[index])
                else:
                    cubes_lineofsight_STEREO_1.append(None)
                    cubes_lineofsight_SDO_1.append(None)
                    cubes_all_data_1.append(None)
                    cubes_no_duplicate_1.append(None)
                    cubes_no_duplicates_STEREO_1.append(None)
                    cubes_no_duplicates_SDO_1.append(None)
                    time_cubes_all_data_1.append(None)
                    time_cubes_no_duplicate_1.append(None)
            self.cubes_lineofsight_STEREO_1 = cubes_lineofsight_STEREO_1
            self.cubes_lineofsight_SDO_1 = cubes_lineofsight_SDO_1
            self.cubes_all_data_1 = cubes_all_data_1
            self.cubes_no_duplicate_1 = cubes_no_duplicate_1
            self.cubes_no_duplicates_STEREO_1 = cubes_no_duplicates_STEREO_1
            self.cubes_no_duplicates_SDO_1 = cubes_no_duplicates_SDO_1
            self.time_cubes_all_data_1 = time_cubes_all_data_1
            self.time_cubes_no_duplicate_1 = time_cubes_no_duplicate_1

        if self.second_cube:
            cubes_lineofsight_STEREO_2 = []
            cubes_lineofsight_SDO_2 = []
            cubes_all_data_2 = []
            cubes_no_duplicate_2 = []
            cubes_no_duplicates_STEREO_2 = []
            cubes_no_duplicates_SDO_2 = []
            time_cubes_all_data_2 = []
            time_cubes_no_duplicate_2 = []
            index = -1
            for number in self.cube_numbers_all:
                if number in self._cube_numbers_2:
                    index += 1
                    cubes_lineofsight_STEREO_2.append(self.cubes_lineofsight_STEREO_2[index])
                    cubes_lineofsight_SDO_2.append(self.cubes_lineofsight_SDO_2[index])
                    cubes_all_data_2.append(self.cubes_all_data_2[index])
                    cubes_no_duplicate_2.append(self.cubes_no_duplicate_2[index])
                    cubes_no_duplicates_STEREO_2.append(self.cubes_no_duplicates_STEREO_2[index])
                    cubes_no_duplicates_SDO_2.append(self.cubes_no_duplicates_SDO_2[index])
                    time_cubes_all_data_2.append(self.time_cubes_all_data_2[index])
                    time_cubes_no_duplicate_2.append(self.time_cubes_no_duplicate_2[index])
                else:
                    cubes_lineofsight_STEREO_2.append(None)
                    cubes_lineofsight_SDO_2.append(None)
                    cubes_all_data_2.append(None)
                    cubes_no_duplicate_2.append(None)
                    cubes_no_duplicates_STEREO_2.append(None)
                    cubes_no_duplicates_SDO_2.append(None)
                    time_cubes_all_data_2.append(None)
                    time_cubes_no_duplicate_2.append(None)
            self.cubes_lineofsight_STEREO_2 = cubes_lineofsight_STEREO_2
            self.cubes_lineofsight_SDO_2 = cubes_lineofsight_SDO_2
            self.cubes_all_data_2 = cubes_all_data_2
            self.cubes_no_duplicate_2 = cubes_no_duplicate_2
            self.cubes_no_duplicates_STEREO_2 = cubes_no_duplicates_STEREO_2
            self.cubes_no_duplicates_SDO_2 = cubes_no_duplicates_SDO_2
            self.time_cubes_all_data_2 = time_cubes_all_data_2
            self.time_cubes_no_duplicate_2 = time_cubes_no_duplicate_2

    def Sun_pos(self):
        """
        To find the Sun's radius and the center position in the cubes reference frame.
        """

        # Reference data 
        first_cube_name = os.path.join(self.paths['Cubes'], self._cube_names_1[0])

        # Initial data values
        solar_r = 6.96e5 
        self._length_dx = readsav(first_cube_name).dx
        self._length_dy = readsav(first_cube_name).dy
        self._length_dz = readsav(first_cube_name).dz
        x_min = readsav(first_cube_name).xt_min
        y_min = readsav(first_cube_name).yt_min
        z_min = readsav(first_cube_name).zt_min

        # The Sun's radius
        self.radius_index = solar_r / self._length_dx  # TODO: need to change this if dx!=dy!=dz.

        # The Sun center's position
        x_index = x_min / self._length_dx 
        y_index = y_min / self._length_dy 
        z_index = z_min / self._length_dz 
        self._sun_center = np.array([0 - x_index, 0 - y_index, 0 - z_index])

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
        self._sun_texture = np.log(nw_image)

    def Sun_points(self):
        """
        Creates a spherical cloud of points that represents the pixels on the Sun's surface.
        """

        # Initialisation
        N = self._sun_texture_resolution  # number of points in the theta direction
        theta = np.linspace(0, np.pi, N)  # latitude of the points
        phi = np.linspace(0, 2 * np.pi, 2 * N)  # longitude of the points
        theta, phi = np.meshgrid(theta, phi)  # the subsequent meshgrid

        # Conversion to cartesian coordinates
        x = self.radius_index * np.sin(theta) * np.cos(phi) + self._sun_center[0]
        y = self.radius_index * np.sin(theta) * np.sin(phi) + self._sun_center[1]
        z = self.radius_index * np.cos(theta) + self._sun_center[2] 

        # Creation of the position of the spherical cloud of points
        self.sun_points = np.array([x, y, z], dtype='float32').T

        # The corresponding image indexes to get the colors
        self._sun_texture_x = np.linspace(0, self._texture_height - 1, self.sun_points.shape[0], dtype='uint16')
        self._sun_texture_y = np.linspace(0, self._texture_width - 1, self.sun_points.shape[1], dtype='uint16')

    def Colours_1D(self):
        """
        Creates a 1D array of the integer Hex pixel values (0x000000 format) of a 2D sun texture image.
        """

        x_indices = self._sun_texture_x[:, np.newaxis]
        y_indices = self._sun_texture_y[np.newaxis, :]

        colours = self._sun_texture[x_indices, y_indices].flatten()
        normalized_colours = (colours - np.min(colours)) / (np.max(colours) - np.min(colours))
        blue_val = (normalized_colours * 255).astype('uint8')
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
        stars_x = stars_radius * np.sin(stars_theta) * np.cos(stars_phi) + self._sun_center[0]
        stars_y = stars_radius * np.sin(stars_theta) * np.sin(stars_phi) + self._sun_center[1]
        stars_z = stars_radius * np.cos(stars_theta) + self._sun_center[2]

        # Cartesian positions
        self.stars_points = np.array([stars_x, stars_y, stars_z], dtype='float32').T

    def SDO_stats(self):
        """
        To save the information needed to find the position of SDO.
        """

        from astropy import units as u
        from astropy.coordinates import CartesianRepresentation
        from sunpy.coordinates.frames import  HeliographicCarrington


        SDO_fits_names = [os.path.join(self.paths['SDO'], f'AIA_fullhead_{number:03d}.fits.gz')
                           for number in self.cube_numbers_all]
        
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

            SDO_pos.append(self._sun_center + np.array([xpos_index, ypos_index, zpos_index]))
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

        for number in self.cube_numbers_all:
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

            xpos_index = Xhec / self._length_dx
            ypos_index = Yhec / self._length_dy
            zpos_index = Zhec / self._length_dz

            stereo_pos.append(self._sun_center + np.array([xpos_index, ypos_index, zpos_index])) 
        self.STEREO_pos = np.array(stereo_pos)          


class K3dAnimation(Data):
    """
    Creates the corresponding k3d animation to then be used in a Jupyter notebook file.
    """

    @typechecked
    def __init__(self, compression_level: int = 9, plot_height: int = 1220, sleep_time: int | float = 2, 
                 camera_fov: int | float = 1, camera_zoom_speed: int | float = 0.7, trace_opacity: int | float = 0.1, 
                 make_screenshots: bool = False, screenshot_scale: int | float = 2, screenshot_sleep: int | float = 5, 
                 screenshot_version: str = 'v0', **kwargs):
        
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
            self.paths['Screenshots'] = os.path.join(self.paths['Main'], 'Screenshots')
            os.makedirs(self.paths['Screenshots'], exist_ok=True)
   
    def Full_array(self, sparse_cube):
        """
        To recreate a full 3D np.array from a sparse np.ndarray representing a 3D volume.
        """

        if self.memory_saver:
            if not (sparse_cube==None):
                cube = sparse_cube.toarray().reshape(self.cubes_shape[1], self.cubes_shape[2], self.cubes_shape[3])
                return cube
            else:
                return np.zeros((self.cubes_shape[1], self.cubes_shape[2], self.cubes_shape[3]), dtype='uint8')
        else:
            return sparse_cube

    def Camera_params(self):
        """
        Camera visualisation parameters.
        """
 
        self.plot.camera_auto_fit = False
        self.plot.camera_fov = self.camera_fov  # FOV in degrees
        self.plot.camera_zoom_speed = self.camera_zoom_speed  # it was zooming too quickly (default=1.2)
        
        # Point to look at, i.e. initial rotational reference
        self._camera_reference = np.array([self.cubes_shape[3], self.cubes_shape[2], self.cubes_shape[1]]) / 2
        #self._camera_reference = np.array([self.cubes_shape[3], self.cubes_shape[2], self.cubes_shape[1]]) / 2  # I got no clue why it is the other way around than for the cubes, but I tested it.
        
        if self.sdo_pov:
            self.plot.camera = [self.SDO_pos[0, 0], self.SDO_pos[0, 1], self.SDO_pos[0, 2],
                        self._camera_reference[0], self._camera_reference[1], self._camera_reference[2],
                        0, 0, 1]  # up vector
        elif self.stereo_pov:
            self.plot.camera = [self.STEREO_pos[0, 0], self.STEREO_pos[0, 1], self.STEREO_pos[0, 2],
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
            time.sleep(0.2)
        elif self.stereo_pov:
            self.plot.camera = [self.STEREO_pos[change['new'], 0], self.STEREO_pos[change['new'], 1], self.STEREO_pos[change['new'], 2],
                                self._camera_reference[0], self._camera_reference[1], self._camera_reference[2],
                                0, 0, 1]
            time.sleep(0.2)          
        
        if self.all_data:
            if self.first_cube:
                data = self.Full_array(self.cubes_all_data_1[change['new']])
                self.plot_alldata_set1.voxels = data
            if self.second_cube:
                data = self.Full_array(self.cubes_all_data_2[change['new']])
                self.plot_alldata_set2.voxels = data
        if self.duplicates:
            if self.first_cube:
                data = self.Full_array(self.cubes_no_duplicates_STEREO_1[change['new']])
                self.plot_dupli_STEREO_set1.voxels = data
                data = self.Full_array(self.cubes_no_duplicates_SDO_1[change['new']])
                self.plot_dupli_SDO_set1.voxels = data
            if self.second_cube:
                data = self.Full_array(self.cubes_no_duplicates_STEREO_2[change['new']])
                self.plot_dupli_STEREO_set2.voxels = data
                data = self.Full_array(self.cubes_no_duplicates_SDO_2[change['new']])
                self.plot_dupli_SDO_set2.voxels = data
        if self.no_duplicate:
            if self.first_cube:
                data = self.Full_array(self.cubes_no_duplicate_1[change['new']])
                self.plot_dupli_set1.voxels = data
            if self.second_cube:
                data = self.Full_array(self.cubes_no_duplicate_2[change['new']])
                self.plot_dupli_set2.voxels = data            
        if self.line_of_sight:
            if self.first_cube:
                data = self.Full_array(self.cubes_lineofsight_STEREO_1[change['new']])
                self.plot_los_STEREO_set1.voxels = data
                data = self.Full_array(self.cubes_lineofsight_SDO_1[change['new']])
                self.plot_los_SDO_set1.voxels = data
            if self.second_cube:
                data = self.Full_array(self.cubes_lineofsight_STEREO_2[change['new']])
                self.plot_los_STEREO_set2.voxels = data
                data = self.Full_array(self.cubes_lineofsight_SDO_2[change['new']])
                self.plot_los_SDO_set2.voxels = data
        if self.time_intervals_all_data:
            if self.first_cube:
                data = self.Full_array(self.time_cubes_all_data_1[change['new']])
                self.plot_interv_set1.voxels = data
            if self.second_cube:
                data = self.Full_array(self.time_cubes_all_data_2[change['new']])
                self.plot_interv_set2.voxels = data               
        if self.time_intervals_no_duplicate:
            if self.first_cube:
                data = self.Full_array(self.time_cubes_no_duplicate_1[change['new']])
                self.plot_interv_dupli_set1.voxels = data
            if self.second_cube:
                data = self.Full_array(self.time_cubes_no_duplicate_2[change['new']])
                self.plot_interv_dupli_set2.voxels = data
        if self.day_trace or self.day_trace_no_duplicate:
            if self.first_cube:
                for day_nb, day_index in enumerate(self.day_indexes_1):
                    if (change['new'] in day_index) and (change['old'] not in day_index):
                        if self.day_trace:
                            data = self.Full_array(self.day_cubes_all_data_1[day_nb])
                            self.plot_day_set1.voxels = data
                        if self.day_trace_no_duplicate:
                            data = self.Full_array(self.day_cubes_no_duplicate_1[day_nb])
                            self.plot_day_dupli_set1.voxels = data
                        break
            if self.second_cube:
                for day_nb, day_index in enumerate(self.day_indexes_2):
                    if (change['new'] in day_index) and (change['old'] not in day_index):
                        if self.day_trace:
                            data = self.Full_array(self.day_cubes_all_data_2[day_nb])
                            self.plot_day_set2.voxels = data
                        if self.day_trace_no_duplicate:
                            data = self.Full_array(self.day_cubes_no_duplicate_2[day_nb])
                            self.plot_day_dupli_set2.voxels = data
                        break           
        if self.make_screenshots:
            self.Screenshot_making()

    def Play(self):
        """
        Params for the play button.
        """
        
        if self.play_pause_button.value and self.time_slider.value < len(self.cube_numbers_all) - 1:
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
                          for date in self.dates_all]

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
            if self.first_cube:
                data = self.Full_array(self.cubes_all_data_1[0])
                self.plot_alldata_set1 = k3d.voxels(data, outlines=False, opacity=0.1, compression_level=self.compression_level,
                                                     color_map=[0x90ee90], name='Set1: all data')
                self.plot += self.plot_alldata_set1
            if self.second_cube:
                data = self.Full_array(self.cubes_all_data_2[0])
                self.plot_alldata_set2 = k3d.voxels(data, outlines=False, opacity=0.1, compression_level=self.compression_level,
                                                     color_map=[0x90ee90], name='Set2: all data')
                self.plot += self.plot_alldata_set2           
       
        if self.duplicates:
            if self.first_cube:
                data = self.Full_array(self.cubes_no_duplicates_STEREO_1[0])
                self.plot_dupli_STEREO_set1 = k3d.voxels(data, compression_level=self.compression_level, outlines=True, 
                                        color_map=[0xff0000], name='Set1: no duplicates from SDO')
                data = self.Full_array(self.cubes_no_duplicates_SDO_1[0])
                self.plot_dupli_SDO_set1 = k3d.voxels(data, compression_level=self.compression_level, outlines=True, 
                                        color_map=[0x0000ff], name='Set1: no duplicates from STEREO')
                self.plot += self.plot_dupli_STEREO_set1
                self.plot += self.plot_dupli_SDO_set1
            if self.second_cube:
                data = self.Full_array(self.cubes_no_duplicates_STEREO_2[0])
                self.plot_dupli_STEREO_set2 = k3d.voxels(data, compression_level=self.compression_level, outlines=True, 
                                        color_map=[0xff0000], name='Set2: no duplicates from SDO')
                data = self.Full_array(self.cubes_no_duplicates_SDO_2[0])
                self.plot_dupli_SDO_set2 = k3d.voxels(data, compression_level=self.compression_level, outlines=True, 
                                        color_map=[0x0000ff], name='Set2: no duplicates from STEREO')
                self.plot += self.plot_dupli_STEREO_set2
                self.plot += self.plot_dupli_SDO_set2
               
        if self.no_duplicate:
            if self.first_cube:
                data = self.Full_array(self.cubes_no_duplicate_1[0])
                self.plot_dupli_set1 = k3d.voxels(data, compression_level=self.compression_level, outlines=True, 
                                        color_map=[0x8B0000], name='Set1: no duplicates')
                self.plot += self.plot_dupli_set1
            if self.second_cube:
                data = self.Full_array(self.cubes_no_duplicate_2[0])
                self.plot_dupli_set2 = k3d.voxels(data, compression_level=self.compression_level, outlines=True, 
                                        color_map=[0x8B0000], name='Set2: no duplicates')
                self.plot += self.plot_dupli_set2

        if self.line_of_sight:
            if self.first_cube:
                data = self.Full_array(self.cubes_lineofsight_STEREO_1[0])
                self.plot_los_STEREO_set1 = k3d.voxels(data, compression_level=self.compression_level, outlines=True, 
                                        color_map=[0x0000ff], name='Set1: seen from Stereo')
                data = self.Full_array(self.cubes_lineofsight_SDO_1[0])
                self.plot_los_SDO_set1 = k3d.voxels(data, compression_level=self.compression_level, outlines=True, 
                                        color_map=[0xff0000], name='Set1: seen from SDO')
                self.plot += self.plot_los_STEREO_set1
                self.plot += self.plot_los_SDO_set1
            if self.second_cube:
                data = self.Full_array(self.cubes_lineofsight_STEREO_2[0])
                self.plot_los_STEREO_set2 = k3d.voxels(data, compression_level=self.compression_level, outlines=True, 
                                        color_map=[0x0000ff], name='Set2: seen from Stereo')
                data = self.Full_array(self.cubes_lineofsight_SDO_2[0])
                self.plot_los_SDO_set2 = k3d.voxels(data, compression_level=self.compression_level, outlines=True, 
                                        color_map=[0xff0000], name='Set2: seen from SDO')
                self.plot += self.plot_los_STEREO_set2
                self.plot += self.plot_los_SDO_set2

        if self.time_intervals_all_data:
            if self.first_cube:
                data = self.Full_array(self.time_cubes_all_data_1[0])
                self.plot_interv_set1 = k3d.voxels(data, compression_level=self.compression_level, outlines=False, 
                                        color_map=[0xff6666],opacity=1, name=f'Set1: all data for {self.time_interval}')
                self.plot += self.plot_interv_set1
            if self.second_cube:
                data = self.Full_array(self.time_cubes_all_data_2[0])
                self.plot_interv_set2 = k3d.voxels(data, compression_level=self.compression_level, outlines=False, 
                                        color_map=[0xff6666],opacity=1, name=f'Set2: all data for {self.time_interval}')
                self.plot += self.plot_interv_set2           
       
        if self.time_intervals_no_duplicate:
            if self.first_cube:
                data = self.Full_array(self.time_cubes_no_duplicate_1[0])
                self.plot_interv_dupli_set1 = k3d.voxels(data, compression_level=self.compression_level, outlines=False,
                                        color_map=[0xff6666], opacity=1, name=f'Set1: no duplicate for {self.time_interval}')
                self.plot = self.plot_interv_dupli_set1
            if self.second_cube:
                data = self.Full_array(self.time_cubes_no_duplicate_2[0])
                self.plot_interv_dupli_set2 = k3d.voxels(data, compression_level=self.compression_level, outlines=False,
                                        color_map=[0xff6666], opacity=1, name=f'Set2: no duplicate for {self.time_interval}')
                self.plot = self.plot_interv_dupli_set2           
        
        if self.trace_data:
            if self.first_cube:
                self.plot += k3d.voxels(self.trace_cubes_1, compression_level=self.compression_level, outlines=False, color_map=[0xff6666],
                                    opacity=self.trace_opacity, name='Set1: total trace')
            if self.second_cube:
                self.plot += k3d.voxels(self.trace_cubes_2, compression_level=self.compression_level, outlines=False, color_map=[0xff6666],
                                    opacity=self.trace_opacity, name='Set2: total trace')
        
        if self.trace_no_duplicate:
            if self.first_cube:
                self.plot += k3d.voxels(self.trace_cubes_no_duplicate_1, compression_level=self.compression_level, outlines=False, 
                                    color_map=[0xff6666], opacity=self.trace_opacity, name='Set1: no duplicates trace')
            if self.second_cube:
                self.plot += k3d.voxels(self.trace_cubes_no_duplicate_2, compression_level=self.compression_level, outlines=False, 
                                    color_map=[0xff6666], opacity=self.trace_opacity, name='Set2: no duplicates trace')
            
        if self.day_trace:
            if self.first_cube:
                date = self.dates_1[0]
                data = self.Full_array(self.day_cubes_all_data_1[0])
                self.plot_day_set1 = k3d.voxels(data, compression_level=self.compression_level, outlines=False,
                                         color_map=[0xff6666], opacity=self.trace_opacity, 
                                         name=f'Set1: total trace for {date.month:02d}.{date.day:02d}')
                self.plot = self.plot_day_set1
            if self.second_cube:
                date = self.dates_2[0]
                data = self.Full_array(self.day_cubes_all_data_2[0])
                self.plot_day_set2 = k3d.voxels(data, compression_level=self.compression_level, outlines=False,
                                         color_map=[0xff6666], opacity=self.trace_opacity, 
                                         name=f'Set2: total trace for {date.month:02d}.{date.day:02d}')
                self.plot = self.plot_day_set2
        
        if self.day_trace_no_duplicate:
            if self.first_cube:
                date = self.dates_1[0]
                data = self.Full_array(self.day_cubes_no_duplicate_1[0])
                self.plot_day_dupli_set1 = k3d.voxels(data, compression_level=self.compression_level, outlines=False,
                                         color_map=[0xff6666], opacity=self.trace_opacity, 
                                         name=f'Set1: no duplicate trace for {date.month:02d}.{date.day:02d}')
                self.plot += self.plot_day_dupli_set1
            if self.second_cube:
                date = self.dates_2[0]
                data = self.Full_array(self.day_cubes_no_duplicate_2[0])
                self.plot_day_dupli_set2 = k3d.voxels(data, compression_level=self.compression_level, outlines=False,
                                         color_map=[0xff6666], opacity=self.trace_opacity, 
                                         name=f'Set2: no duplicate trace for {date.month:02d}.{date.day:02d}')
                self.plot += self.plot_day_dupli_set2

        # Adding a play/pause button
        self.play_pause_button = widgets.ToggleButton(value=False, description='Play', icon='play')

        # Set up the time slider and the play/pause button
        self.time_slider = widgets.IntSlider(min=0, max=len(self.cube_numbers_all)-1, description='Frame:')
        self.date_dropdown = widgets.Dropdown(options=self.date_text, description='Date:')
        self.time_slider.observe(self.Update_voxel, names='value')
        self.time_link= widgets.jslink((self.time_slider, 'value'), (self.date_dropdown, 'index'))
        self.play_pause_button.observe(self.Play_pause_handler, names='value')

        # Display
        display(self.plot, self.time_slider, self.date_dropdown, self.play_pause_button)
        if self.make_screenshots:
            self.plot.screenshot_scale = self.screenshot_scale
            self.Screenshot_making()
