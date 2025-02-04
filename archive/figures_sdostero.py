"""
Used 3 k3d images (sdo pov, stereo pov, and random pov) with 2 direct acquisition images (sdo pov and stereo pov) to create a final 
plt.savefig plot. Also creates the corresponding GIF object. 
"""

# Imports 
import os
import re
import warnings

import numpy as np
import imageio.v3 as iio3
import astropy.units as u
import multiprocessing as mp
import matplotlib.pyplot as plt

from glob import glob
from PIL import Image
from typing import Match
from astropy.io import fits
from typeguard import typechecked
from sunpy.map import Map, GenericMap

from Common import Decorators, SSHMirroredFilesystem



class ImageFinder:

    @typechecked
    def __init__(self, interval: str | None = None, processes: int = 4):
        
        # Arguments
        self.interval = interval
        self.multiprocessing = True if processes > 1 else False
        self.processes = processes

        # Functions
        self.Paths()
        self.SDO_image_finder()
        self.Images()
        self.Patterns()
        self.Main_loop()
        self.Multiprocessing()

    def Paths(self) -> None:
        """
        Creating the paths to the files.
        """

        main_path = os.path.join(os.getcwd(), '..')
        self.paths = {
            'Main': main_path,
            'STEREO': os.path.join(main_path, 'STEREO', 'int'),
            'Screenshots': os.path.join(main_path, 'texture_screenshots'),
            'MP4': os.path.join(main_path, 'MP4_saves'),
            'Save': os.path.join(main_path, 'k3d_final_plots'),
            }      
        os.makedirs(self.paths['Save'], exist_ok=True)

    def Images(self) -> None:
        """
        Getting the path to the images as lists.
        """

        self.stereo_image = sorted(glob(os.path.join(self.paths['STEREO'], '*.png')))
        self.screenshot = sorted(glob(os.path.join(self.paths['Screenshots'], '*.png')))
        self.MP4_filepaths = sorted(glob(os.path.join(self.paths['MP4'], 'Frame*.png')))
        print(f'len of MP4 filepaths is {len(self.MP4_filepaths)}')

    def Patterns(self) -> None:
        """
        Setting up the patterns for the filenames so that I can choose the right images.
        """

        self.stereo_pattern = re.compile(r'''(?P<number>\d{4})
                                         _\d{4}-\d{2}-
                                         (?P<day>\d{2})T
                                         (?P<hour>\d{2})-
                                         (?P<minute>\d{2})-
                                         \d{2}\.000\.png''', re.VERBOSE)
        self.MP4_pattern = re.compile(r'''Frame_\d{2}m
                                      (?P<day>\d{2})d_
                                      (?P<hour>\d{2})h
                                      (?P<minute>\d{2})\.png''', re.VERBOSE)
        self.sdo_date_pattern = re.compile(r'''\d{4}-\d{2}-
                                           (?P<day>\d{2})T
                                           (?P<hour>\d{2}):
                                           (?P<minute>\d{2})
                                           :\d{2}\.\d{2}''', re.VERBOSE)
        if 'nodupli' in self.interval:
            self.screenshot_pattern = re.compile(r'''(?P<interval>nodupli)_
                                                 \d{3}_\d{4}-\d{2}-
                                                 (?P<day>\d{2})_
                                                 (?P<hour>\d{2})h
                                                 (?P<minute>\d{2})min_
                                                 v(?P<version>\d{1})\.png''', re.VERBOSE)
        else:
            self.screenshot_pattern = re.compile(r'''nodupli_interval(?P<interval>\d+h|\d+min|\d+days|nodupli)_
                                                \d{4}-\d{2}-
                                                (?P<day>\d{2})_
                                                (?P<hour>\d{2})h
                                                (?P<minute>\d{2})min_
                                                v(?P<version>\d{1})\.png''', re.VERBOSE)

    def SDO_image_finder(self) -> None:
        """
        To find the SDO image given its header timestamp and a list of corresponding paths to the corresponding fits file.
        """

        # Setup
        filepath_end = '/S00000/image_lev1.fits'
        with open('SDO_timestamps.txt', 'r') as files:
            strings = files.read().splitlines()
        tuple_list = [s.split(" ; ") for s in strings]
        
        # Looking for the data
        first_path = os.path.join(tuple_list[0][0], filepath_end)
        if os.path.exists(first_path):
            timestamp_to_path = {}
            for s in tuple_list:
                path, timestamp = s
                timestamp_to_path[timestamp[:-3]] = path + filepath_end
        else:
            server = SSHMirroredFilesystem(verbose=3)
            timestamp_to_path = {}
            for s in tuple_list:
                path, timestamp = s
                timestamp_to_path[timestamp[:-3]] = server.mirror(path + filepath_end, strip_level=1)
            server.close()
        self.sdo_timestamp = timestamp_to_path

    @Decorators.running_time
    def Main_loop(self) -> None:
        """
        Loop for the STEREO images.
        """

        self.groups = []
        for path_stereo in self.stereo_image[1:]:  # because the first image in k3d is always wrong
            stereo_group = self.stereo_pattern.match(os.path.basename(path_stereo))

            if stereo_group:
                self.Second_loop(stereo_group)
            else:
                raise ValueError(f"Stereo filename {os.path.basename(path_stereo)} doesn't match.")
        self.groups = np.array(self.groups)

    def Second_loop(self, stereo_group: Match[str]) -> None:
        """
        Loop for the screenshot images (i.e. the ones gotten with k3d).
        """

        for path_screenshot in self.screenshot:
            screenshot_group = self.screenshot_pattern.match(os.path.basename(path_screenshot))

            if screenshot_group and (screenshot_group.group('interval') == self.interval):
                day = int(screenshot_group.group('day'))
                hour = int(screenshot_group.group('hour'))
                minute = round(int(screenshot_group.group('minute'))) 
                stereo_day = int(stereo_group.group('day'))
                stereo_hour = int(stereo_group.group('hour'))
                stereo_minute = int(stereo_group.group('minute'))

                if (day==stereo_day) and (hour==stereo_hour) and (minute==stereo_minute):
                    self.Third_loop(stereo_group, screenshot_group)
                    return

    def Third_loop(self, stereo_group: Match[str], screenshot_group: Match[str]) -> None:
        """
        Loop on the SDO images.
        """

        for date in self.sdo_timestamp.keys():
            sdo_group = self.sdo_date_pattern.match(date)

            if sdo_group:
                day = int(sdo_group.group('day'))
                hour = int(sdo_group.group('hour'))
                minute = int(sdo_group.group('minute'))
                stereo_day = int(stereo_group.group('day'))
                stereo_hour = int(stereo_group.group('hour'))
                stereo_minute = int(stereo_group.group('minute'))

                if (day==stereo_day) and (hour==stereo_hour) and (minute in [stereo_minute, stereo_minute + 1]):
                    self.Fourth_loop(stereo_group, screenshot_group, sdo_group)
                    return
            else:
                raise ValueError(f"The date {date} doesn't match the usual pattern.")

    def Fourth_loop(self, stereo_group: Match[str], screenshot_group: Match[str], sdo_group: Match[str]) -> None:
        """
        For the loop on the images gotten from the MP4 powerpoint video.
        """

        for filepath in self.MP4_filepaths:
            MP4_group = self.MP4_pattern.match(os.path.basename(filepath))

            if MP4_group:
                day = int(MP4_group.group('day'))
                hour = int(MP4_group.group('hour'))
                minute = int(MP4_group.group('minute')) 
                stereo_day = int(stereo_group.group('day'))
                stereo_hour = int(stereo_group.group('hour'))
                stereo_minute = round(int(stereo_group.group('minute')) / 10) * 10

                # print(f"minute is {minute} and stereo_minute is {stereo_minute}")

                if stereo_minute==60:
                    stereo_hour += 1
                    stereo_minute = 0
                    if stereo_hour==24:
                        stereo_day += 1
                        stereo_hour = 0

                if (day==stereo_day) and (hour==stereo_hour) and (minute in [stereo_minute, stereo_minute + 1]):
                    self.groups.append([sdo_group.group(), stereo_group.group(), screenshot_group.group(), MP4_group.group()])
                    return
            else:
                raise ValueError(f"The filename {os.path.basename(filepath)} doesn't match the MP4 pattern.")

    @Decorators.running_time
    def Multiprocessing(self) -> None:
        """
        For the multiprocessing.
        Some class attributes are set to None as multiprocessing doesn't like pattern objects.
        """

        # Multiprocesses hates patterns
        self.stereo_pattern = None
        self.sdo_date_pattern = None
        self.MP4_pattern = None
        self.screenshot_pattern = None

        if self.multiprocessing:
            pool = mp.Pool(processes=self.processes)
            args = [(group_str,) for group_str in self.groups]
            pool.starmap(self.Plotting, args)
            pool.close()
            pool.join()
        else:
            for arg in self.groups: self.Plotting(arg)
        SSHMirroredFilesystem.cleanup()

    def SDO_prepocessing(self, image: np.ndarray) -> Image:
        """
        To open and do the corresponding preprocessing for the SDO image.
        """

        index = round(image.shape[0] / 3)
        image = image[index: index * 2 + 1, :]
        index = round(image.shape[1] / 3)
        image = image[:, :index + 1]

        image = np.where(np.isnan(image), np.nanmin(image), image)
        image -= np.min(image) 

        lower_cut = np.percentile(image, 1)
        upper_cut = np.percentile(image, 99.99)
        image[image <= lower_cut] = lower_cut
        image[image >= upper_cut] = upper_cut
        image = np.flip(image, axis=0) # TODO: why is there a flip??
        image = np.log(image)
        image = Image.fromarray(image)
        image = image.resize((512, 512), Image.Resampling.LANCZOS)
        return np.array(image)

    def STEREO_preprocessing(self, filename: str) -> Image:
        """
        To get and do the corresponding preprocessing for the STEREO image.
        """

        full_image = Image.open(os.path.join(self.paths['MP4'], filename))
        stereo_image = np.split(np.array(full_image), 2, axis=1)[0]
        stereo_image = Image.fromarray(stereo_image)
        return np.array(stereo_image.resize((512, 512), Image.Resampling.LANCZOS))

    def map_section(self, sunpy_map: GenericMap) -> GenericMap:
        
        dimensions = sunpy_map.dimensions

        # Setting up the sub_map borders so that you get the middle left section of a ninth of the original image
        x1 = 0  # left edge
        x2 = dimensions.x.value / 3
        y1 = dimensions.y.value / 3
        y2 = 2 * dimensions.y.value / 3  # middle section

        # Corresponding corner position in the coordinate frame
        bottom_left = sunpy_map.pixel_to_world(x1 * u.pix, y1 * u.pix)
        top_right = sunpy_map.pixel_to_world(x2 * u.pix, y2 * u.pix)
        return sunpy_map.submap(bottom_left=bottom_left, top_right=top_right)
    
    def Plotting(self, group_str: list[str]) -> None:
        """
        Created to test the possibilities for the plotting.
        """

        # Separation of the different image string IDs
        sdo_str, stereo_str, screen_str, MP4_str = group_str

        # Not printing the VerifyWarning from astropy.is.fits
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', fits.verify.VerifyWarning)
            # Preparation of the SDO carrington map
            hdul = fits.open(self.sdo_timestamp[sdo_str])
            hdul[1].verify('fix')
            decompressed_sdo = hdul[1].data
            aia_map = Map(decompressed_sdo, hdul[1].header)

        # Creation of the re.Match items
        self.Patterns()
        stereo_groups = self.stereo_pattern.match(stereo_str)
        sdo_groups = self.sdo_date_pattern.match(sdo_str)
        MP4_groups = self.MP4_pattern.match(MP4_str)

        # Opening and preprocessing of the SDO and STEREO images
        stereo_image = self.STEREO_preprocessing(MP4_str)
        # sdo_image = self.SDO_prepocessing(np.array(decompressed_sdo))
        hdul.close()

        # Opening the images
        sdo_screenshot = Image.open(os.path.join(self.paths['Screenshots'], screen_str))
        stereo_screenshot = Image.open(os.path.join(self.paths['Screenshots'], screen_str[:-5] + '1.png'))
        screenshot2 = Image.open(os.path.join(self.paths['Screenshots'], screen_str[:-5] + '2.png'))
        screenshot3 = Image.open(os.path.join(self.paths['Screenshots'], screen_str[:-5] + '3.png'))

        # Resizing
        sdo_screenshot = sdo_screenshot.resize((512, 512), Image.Resampling.LANCZOS)
        stereo_screenshot = stereo_screenshot.resize((512, 512), Image.Resampling.LANCZOS)
        screenshot2 = screenshot2.resize((512, 512), Image.Resampling.LANCZOS)
        screenshot3 = screenshot3.resize((512, 512), Image.Resampling.LANCZOS)

        # Plot setup
        aia_sub_map = self.map_section(aia_map)
        fig, axs = plt.subplots(2, 3, figsize=(6, 4))

        # Plotting the carringtion map for STEREO
        ax_with_wcs = plt.subplot(2, 3, 1, projection = aia_sub_map.wcs)  # top left subplot
        aia_sub_map.plot(axes=ax_with_wcs, clip_interval=(1, 99.99) * u.percent, annotate=False, title=False, interpolation='none',
                         cmap='gray')
        # Taking out the ticks and labels
        ax_with_wcs.grid(False)
        ax_with_wcs.coords[0].set_axislabel('')
        ax_with_wcs.coords[1].set_axislabel('')
        ax_with_wcs.coords[0].set_ticks_visible(False)
        ax_with_wcs.coords[1].set_ticks_visible(False)
        ax_with_wcs.coords[0].set_ticklabel_visible(False)
        ax_with_wcs.coords[1].set_ticklabel_visible(False)
        aia_sub_map.draw_grid(axes=ax_with_wcs, grid_spacing=15*u.deg, annotate=False, system='carrington')

        # Choosing the images
        axs[0, 1].imshow(sdo_screenshot, interpolation='none')
        axs[0, 2].imshow(screenshot2, interpolation='none')
        axs[1, 0].imshow(stereo_image, interpolation='none')
        axs[1, 1].imshow(stereo_screenshot, interpolation='none')
        axs[1, 2].imshow(screenshot3, interpolation='none')

        # # Taking out the axes
        # for ax in axs.flat: ax.axis('off')
        
        # Adding the date text
        axs[0, 0].text(260, 500, f"2012-07-{sdo_groups.group('day')} {sdo_groups.group('hour')}:{sdo_groups.group('minute')}",
                      fontsize=6, color='black', alpha=1, weight='bold')
        axs[1, 0].text(260, 500, f"2012-07-{MP4_groups.group('day')} {MP4_groups.group('hour')}:{MP4_groups.group('minute')}",
                       fontsize=6, color='black', alpha=1, weight='bold')
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)

        # Saving the plot
        if 'nodu' in self.interval:
            figname = f"new_nodupli_{stereo_groups.group('number')}.png"
        else:
            figname = f"new_Fig_{self.interval}_{stereo_groups.group('number')}.png"
        plt.savefig(os.path.join(self.paths['Save'], figname), dpi=300)
        plt.close()
        print('one plot done.')


class MP4_making:
    """
    To create the corresponding GIF.
    """

    def __init__(self, interval='1h', fps=5):

        self.fps = fps
        self.interval = interval
        self.Paths()
        self.MP4()

    def Paths(self):
        """
        Paths creator.
        """

        main_path = '../'

        self.paths = {'Main': main_path,
                      'Figures': os.path.join(main_path, f'k3d_final_plots'),
                      'MP4': os.path.join(main_path, 'MP4_creations')}
        os.makedirs(self.paths['MP4'], exist_ok=True)

    def MP4(self):
        """
        Making a corresponding mp4 file.
        """

        image_paths = sorted(glob(os.path.join(self.paths['Figures'], f'*{self.interval}*.png')))
        images = [iio3.imread(image_path) for image_path in image_paths]

        iio3.imwrite(os.path.join(self.paths['MP4'], f'{self.interval}_fps{self.fps}.mp4'), images, fps=self.fps)

if __name__=='__main__':
    ImageFinder(interval='1h', processes=20)
    # MP4_making(interval='1h', fps=10)


