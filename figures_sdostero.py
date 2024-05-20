"""
Used 3 k3d images (sdo pov, stereo pov, and random pov) with 2 direct acquisition images (sdo pov and stereo pov) to create a final 
plt.savefig plot. Also creates the corresponding GIF object. 
"""

# Imports 
import os
import re

import numpy as np
import imageio.v3 as iio3
import multiprocessing as mp
import matplotlib.pyplot as plt

from glob import glob
from PIL import Image
from typing import Match
from astropy.io import fits
from typeguard import typechecked
from matplotlib.gridspec import GridSpec



class ImageFinder:

    @typechecked
    def __init__(self, interval: str | None = None):
        
        # Arguments
        self.interval = interval

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
        self.screenshot = sorted(glob(self.paths['Screenshots'], '*.png'))

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
        self.sdo_date_pattern = re.compile(r'''\d{4}-\d{2}-
                                           (?P<day>\d{2})T
                                           (?P<hour>\d{2}):
                                           (?P<minute>\{2})
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

        with open('SDO_timestamps.txt', 'r') as files:
            strings = files.read().splitlines()
        tuple_list = [s.split(" ; ") for s in strings]
        
        timestamp_to_path = {}
        for s in tuple_list:
            path, timestamp = s
            timestamp_to_path[timestamp] = path + '/S00000/image_lev1.fits'
        self.sdo_timestamp = timestamp_to_path
    
    def Main_loop(self) -> None:
        """
        Loop for the STEREO images.
        """

        self.groups = []
        for path_stereo in self.stereo_image:
            stereo_group = self.stereo_pattern.match(path_stereo)

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
                minute = int(sdo_group.match('minute'))
                stereo_day = int(stereo_group.group('day'))
                stereo_hour = int(stereo_group.group('hour'))
                stereo_minute = int(stereo_group.group('minute'))

                if (day==stereo_day) and (hour==stereo_hour) and (minute in [stereo_minute, stereo_minute + 1]):
                    print(f"time in minutes is {minute} while the stereo time is {stereo_minute}")

                    self.groups.append([sdo_group.group(), stereo_group.group(), screenshot_group.group()])
                    return
            else:
                raise ValueError(f"The date {date} doesn't match the usual pattern.")

    def Multiprocessing(self) -> None:
        """
        For the multiprocessing.
        Some class attributes are set to None as multiprocessing doesn't like pattern objects.
        """

        # Multiprocesses hates patterns
        self.stereo_pattern = None
        self.sdo_date_pattern = None
        self.screenshot_pattern = None

        pool = mp.Pool(processes=14)
        args = [(group_str,) for group_str in self.groups]
        pool.starmap(self.Plotting, args)
        pool.close()
        pool.join()

    def SDO_prepocessing(self, date: str) -> Image:
        """
        To open and do the corresponding preprocessing for the SDO image.
        """

        image = np.array(fits.getdata(self.sdo_timestamp[date], 1))
        index = round(image.shape[0] / 3)
        image = image[index: index * 2 + 1, :]
        index = round(image.shape[1] / 3)
        image = image[:, :index + 1]

        lower_cut = np.nanpercentile(image, 1)
        upper_cut = np.nanpercentile(image, 99.99)
        image[image < lower_cut] = lower_cut
        image[image > upper_cut] = upper_cut
        image = np.where(np.isnan(image), lower_cut, image)
        image = np.flip(image, axis=0) # TODO: why is there a flip??
        image = np.log(image)
        image = Image.fromarray(image)
        return image.resize((512, 512), Image.Resampling.LANCZOS)

    def STEREO_preprocessing(self, filename: str) -> Image:
        """
        To get and do the corresponding preprocessing for the STEREO image.
        """

        full_image = Image.open(os.path.join(self.paths['MP4'], filename))
        full_image = np.split(np.array(full_image), 2, axis=1)
        stereo_image = Image.fromarray(full_image[0])
        return stereo_image.resize((512, 512), Image.Resampling.LANCZOS)
    
    def Plotting(self, group_str: list[str]) -> None:
        """
        Created to test the possibilities for the plotting.
        """

        sdo_str, stereo_str, screen_str = group_str
        self.Patterns()
        stereo_groups = self.stereo_pattern.match(stereo_str)

        # Opening and preprocessing of the SDO and STEREO images
        stereo_image = self.STEREO_preprocessing(stereo_str)
        sdo_image = self.SDO_prepocessing(sdo_str)

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

        # Plotting
        fig = plt.figure(figsize=(6, 4))
        gs = GridSpec(2, 3)
        
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(sdo_image, interpolation='none')
        ax1.axis('off')
        ax1.text(260, 500, f"2012-07-{sdo_groups.group('day')} {sdo_groups.group('hour')}:{sdo_groups.group('minute')}",
                        fontsize=6, color='black', alpha=1, weight='bold')
        
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(sdo_screenshot, interpolation='none')
        ax2.axis('off')

        ax3 = fig.add_subplot(gs[0, 2])
        ax3.imshow(screenshot2, interpolation='none')
        ax3.axis('off')

        ax4 = fig.add_subplot(gs[1, 0])
        ax4.imshow(stereo_image, interpolation='none')
        ax4.axis('off')
        ax4.text(260, 500, f"2012-07-{stereo_groups.group('day')} {stereo_groups.group('hour')}:{stereo_groups.group('minute')}",
                        fontsize=6, color='black', alpha=1, weight='bold')

        ax5 = fig.add_subplot(gs[1, 1])
        ax5.imshow(stereo_screenshot, interpolation='none')
        ax5.axis('off')

        ax6 = fig.add_subplot(gs[1, 2])
        ax6.imshow(screenshot3, interpolation='none')
        ax6.axis('off')
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0.13, hspace=0.08)

        if 'nodu' in self.interval:
            figname = f"new_nodupli_{stereo_groups.group('number')}.png"
        else:
            figname = f"new_Fig_{self.interval}_{stereo_groups.group('number')}.png"
        plt.savefig(os.path.join(self.paths['Save'], figname), dpi=300)
        plt.close()

class MP4_making:
    """
    To create the corresponding GIF.
    """

    def __init__(self, interval='1h', fps=5, stereo='int'):

        self.fps = fps
        self.interval = interval
        self.stereo = stereo
        self.Paths()
        self.MP4()

    def Paths(self):
        """
        Paths creator.
        """

        main_path = '../'

        self.paths = {'Main': main_path,
                      'Figures': os.path.join(main_path, f'texture_plots2'),
                      'GIF': os.path.join(main_path, 'texture_mp4')}
        os.makedirs(self.paths['MP4'], exist_ok=True)

    def MP4(self):
        """
        Making a corresponding mp4 file.
        """

        # writer = iio3.get_writer(os.path.join(self.paths['GIF'], f'MP4_test.mp4'), fps=self.fps)

        image_paths = sorted(Path(self.paths['Figures']).glob(f'*{self.interval}*.png'))
        images = [iio3.imread(image_path) for image_path in image_paths]

        iio3.imwrite(os.path.join(self.paths['MP4'], f'{self.interval}_fps{self.fps}.mp4'), images, fps=self.fps)

if __name__=='__main__':
    ImageFinder(interval='1h')
    # MP4_making(interval='nodupli', stereo='int', fps=10)


