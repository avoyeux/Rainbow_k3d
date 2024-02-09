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
import matplotlib.image as mpimg
from PIL import Image

from pathlib import Path
from typeguard import typechecked


class ImageFinder:

    @typechecked
    def __init__(self, ints: bool = True, avg: bool = False, interval: str | None = None):
        
        # Arguments
        self.ints = ints
        self.avgs = avg
        if avg:
            self.ints = False
        self.interval = interval

        # Functions
        self.Paths()
        self.Images()
        self.Patterns()
        self.Main_loop()
        self.Multiprocessing()

    def Paths(self):
        """
        Creating the paths to the files.
        """

        main_path = '../'
        if self.ints:
            self.paths = {'Main': main_path,
                        'SDO': os.path.join(main_path, 'MP4_saves'),
                        'STEREO': os.path.join(main_path, 'STEREO', 'int'),
                        'Screenshots': os.path.join(main_path, 'texture_screenshots'),
                        'Save': os.path.join(main_path, 'texture_plots')}
        else:
            self.paths = {'Main': main_path,
                        'SDO': os.path.join(main_path, 'MP4_saves'),
                        'STEREO': os.path.join(main_path, 'STEREO', 'avg'),
                        'Screenshots': os.path.join(main_path, 'Screenshots_both'),
                        'Save': os.path.join(main_path, 'texture_both_avg')}
        os.makedirs(self.paths['Save'], exist_ok=True)

    def Images(self):
        """
        Getting the path to the images as lists.
        """

        self.sdo_image = sorted(Path(self.paths['SDO']).glob('*.png'))
        self.stereo_image = sorted(Path(self.paths['STEREO']).glob('*.png'))
        self.screenshot = sorted(Path(self.paths['Screenshots']).glob('*v0.png'))

    def Patterns(self):
        """
        Setting up the patterns for the filenames so that I can choose the right images.
        """

        self.sdo_pattern = re.compile(r'''Frame_\d{2}m
                                      (?P<day>\d{2})d_
                                      (?P<hour>\d{2})h
                                      (?P<minute>\d{2})\.png''', re.VERBOSE)
        self.stereo_pattern = re.compile(r'''(?P<number>\d{4})
                                         _\d{4}-\d{2}-
                                         (?P<day>\d{2})T
                                         (?P<hour>\d{2})-
                                         (?P<minute>\d{2})-
                                         \d{2}\.000\.png''', re.VERBOSE)
        if 'nodupli' in self.interval:
            self.screenshot_pattern = re.compile(r'''(?P<interval>nodupli)_
                                                 \d{3}_\d{4}-\d{2}-
                                                 (?P<day>\d{2})_
                                                 (?P<hour>\d{2})h
                                                 (?P<minute>\d{2})min_
                                                 v(?P<version>\d{1})\.png''', re.VERBOSE)
        else:
            self.screenshot_pattern = re.compile(r'''interval(?P<interval>\d+h|\d+min|\d+days|nodupli)_
                                                \d{4}-\d{2}-
                                                (?P<day>\d{2})_
                                                (?P<hour>\d{2})h
                                                (?P<minute>\d{2})min_
                                                v(?P<version>\d{1})\.png''', re.VERBOSE)

    def Main_loop(self):
        """
        Loop for the SDO images.
        """

        self.groups = []
        for path_sdo in self.sdo_image[2:]:
            sdo_groups = self.sdo_pattern.match(os.path.basename(path_sdo))

            if sdo_groups:
                self.Second_loop(sdo_groups)
            else:
                raise ValueError(f"The sdo filename {os.path.basename(path_sdo)} doesn't match")
        self.groups = np.array(self.groups)

    def Second_loop(self, sdo_groups):
        """
        Loop for the screenshot images (i.e. the ones gotten with k3d).
        """
        
        for path_screenshot in self.screenshot:
            screenshot_groups = self.screenshot_pattern.match(os.path.basename(path_screenshot))

            if screenshot_groups:
                if screenshot_groups.group('interval') == self.interval:
                    day = int(screenshot_groups.group('day'))
                    hour = int(screenshot_groups.group('hour'))
                    minute = round(int(screenshot_groups.group('minute')) / 10) * 10 # to match the sdo one
                    sdo_day = int(sdo_groups.group('day'))
                    sdo_hour = int(sdo_groups.group('hour'))
                    sdo_minute = int(sdo_groups.group('minute'))

                    if minute==60:
                        hour += 1
                        minute = 0
                        if hour==24:
                            day += 1
                            hour = 0
                    if (day==sdo_day) and (hour==sdo_hour) and (minute==sdo_minute):
                        self.Third_loop(sdo_groups, screenshot_groups)
                        return
            
    def Third_loop(self, sdo_groups, screenshot_groups):
        """
        Loop for the STEREO images.
        """

        for path_stereo in self.stereo_image:
            stereo_groups = self.stereo_pattern.match(os.path.basename(path_stereo))

            if stereo_groups:
                day = stereo_groups.group('day')
                hour = stereo_groups.group('hour')
                minute = stereo_groups.group('minute')
                screen_day = screenshot_groups.group('day')
                screen_hour = screenshot_groups.group('hour')
                screen_minute =screenshot_groups.group('minute')

                if (day==screen_day) and (hour==screen_hour) and (minute==screen_minute):
                    self.groups.append([sdo_groups.group(), stereo_groups.group(), screenshot_groups.group()])
                    return
            else:
                raise ValueError(f"Stereo filename {os.path.basename(path_stereo)} doesn't match")

    def Multiprocessing(self):
        """
        For the multiprocessing.
        Some class attributes are set to None as multiprocessing doesn't like pattern objects.
        """

        # Multiprocesses hates patterns
        self.last_screenshot = None
        self.sdo_pattern = None
        self.stereo_pattern = None
        self.screenshot_pattern = None
        self.stereo_ratio_pattern = None

        pool = mp.Pool(processes=14)
        args = [(group_str,) for group_str in self.groups]
        pool.starmap(self.Plotting, args)
        pool.close()
        pool.join()
        
    def Plotting2(self, group_str):
        """
        Plots the corresponding images together.
        """

        sdo_str, stereo_str, screen_str = group_str

        self.Patterns()
        stereo_groups = self.stereo_pattern.match(stereo_str)
        sdo_groups = self.sdo_pattern.match(sdo_str)

        print(f'screen str is {screen_str}')
        sdo_screenshot = Image.open(os.path.join(self.paths['Screenshots'], screen_str))
        stereo_screenshot = Image.open(os.path.join(self.paths['Screenshots'], screen_str[:-5] + '1.png'))
        screenshot2 = Image.open(os.path.join(self.paths['Screenshots'], screen_str[:-5] + '2.png'))
        screenshot3 = Image.open(os.path.join(self.paths['Screenshots'], screen_str[:-5] + '3.png'))

        full_image = Image.open(os.path.join(self.paths['SDO'], sdo_str))
        stereo_image = Image.open(os.path.join(self.paths['STEREO'], stereo_str))

        full_image = np.split(np.array(full_image), 2, axis=1)

        sdo_image = Image.fromarray(full_image[1])

        x1, sdo_screenshot, x3 = np.split(np.array(sdo_screenshot), 3, axis=0)
        sdo_screenshot, x2, x3 = np.split(sdo_screenshot, 3, axis=1)
        sdo_screenshot = Image.fromarray(sdo_screenshot)

        sdo_screenshot = sdo_screenshot.resize((512, 512), Image.Resampling.LANCZOS)
        stereo_screenshot = stereo_screenshot.resize((512, 512), Image.Resampling.LANCZOS)
        screenshot2 = screenshot2.resize((512, 512), Image.Resampling.LANCZOS)
        sdo_image = sdo_image.resize((512, 512), Image.Resampling.LANCZOS)
        stereo_image = stereo_image.resize((512, 512), Image.Resampling.LANCZOS)
        screenshot3 = screenshot3.resize((512, 512), Image.Resampling.LANCZOS)

        fig, axs = plt.subplots(2, 3, figsize=(8, 4), constrained_layout=True)
        
        axs[0, 0].imshow(sdo_screenshot, interpolation='none')
        axs[0, 0].axis('off')
        axs[0, 0].set_title('SDO', fontsize=7)
        axs[0, 0].text(240, 500, f"2012-07-{sdo_groups.group('day')} {sdo_groups.group('hour')}:{sdo_groups.group('minute')}",
                        fontsize=5, color='white', alpha=0.9)

        axs[0, 1].imshow(stereo_screenshot, interpolation='none')
        axs[0, 1].axis('off')
        axs[0, 1].set_title('STEREO', fontsize=7)
        axs[0, 1].text(240, 500, f"2012-07-{stereo_groups.group('day')} {stereo_groups.group('hour')}:{stereo_groups.group('minute')}",
                        fontsize=5, color='white', alpha=0.9)

        axs[0, 2].imshow(screenshot2, interpolation='none')
        axs[0, 2].axis('off')

        axs[1, 0].imshow(sdo_image, interpolation='none')
        axs[1, 0].axis('off')

        axs[1, 1].imshow(stereo_image, interpolation='none')
        axs[1, 1].axis('off')

        axs[1, 2].imshow(screenshot3, interpolation='none')
        axs[1, 2].axis('off')
        # plt.tight_layout(constrained_layout=True)
        # plt.tight_layout(pad=0.01, h_pad=0.01, w_pad=0.01)

        if 'nodu' in self.interval:
            figname = f"nodupli_{stereo_groups.group('number')}.png"
        else:
            figname = f"Fig_{self.interval}_{stereo_groups.group('number')}.png"
        plt.savefig(os.path.join(self.paths['Save'], figname), dpi=250)
        plt.close()

    def Plotting(self, group_str):
        """
        Plots the corresponding images together.
        """
        from matplotlib.gridspec import GridSpec

        sdo_str, stereo_str, screen_str = group_str

        self.Patterns()
        stereo_groups = self.stereo_pattern.match(stereo_str)
        sdo_groups = self.sdo_pattern.match(sdo_str)

        print(f'screen str is {screen_str}')
        sdo_screenshot = Image.open(os.path.join(self.paths['Screenshots'], screen_str))
        stereo_screenshot = Image.open(os.path.join(self.paths['Screenshots'], screen_str[:-5] + '1.png'))
        screenshot2 = Image.open(os.path.join(self.paths['Screenshots'], screen_str[:-5] + '2.png'))
        screenshot3 = Image.open(os.path.join(self.paths['Screenshots'], screen_str[:-5] + '3.png'))

        full_image = Image.open(os.path.join(self.paths['SDO'], sdo_str))
        stereo_image = Image.open(os.path.join(self.paths['STEREO'], stereo_str))

        full_image = np.split(np.array(full_image), 2, axis=1)

        sdo_image = Image.fromarray(full_image[1])

        x1, sdo_screenshot, x3 = np.split(np.array(sdo_screenshot), 3, axis=0)
        sdo_screenshot, x2, x3 = np.split(sdo_screenshot, 3, axis=1)
        sdo_screenshot = Image.fromarray(sdo_screenshot)

        sdo_screenshot = sdo_screenshot.resize((512, 512), Image.Resampling.LANCZOS)
        stereo_screenshot = stereo_screenshot.resize((512, 512), Image.Resampling.LANCZOS)
        screenshot2 = screenshot2.resize((512, 512), Image.Resampling.LANCZOS)
        sdo_image = sdo_image.resize((512, 512), Image.Resampling.LANCZOS)
        stereo_image = stereo_image.resize((512, 512), Image.Resampling.LANCZOS)
        screenshot3 = screenshot3.resize((512, 512), Image.Resampling.LANCZOS)

        fig = plt.figure(figsize=(6, 4))
        gs = GridSpec(2, 3)
        
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(sdo_screenshot, interpolation='none')
        ax1.axis('off')
        ax1.set_title('SDO', fontsize=7)
        ax1.text(260, 500, f"2012-07-{sdo_groups.group('day')} {sdo_groups.group('hour')}:{sdo_groups.group('minute')}",
                        fontsize=6, color='white', alpha=1)

        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(stereo_screenshot, interpolation='none')
        ax2.axis('off')
        ax2.set_title('STEREO', fontsize=7)
        ax2.text(260, 500, f"2012-07-{stereo_groups.group('day')} {stereo_groups.group('hour')}:{stereo_groups.group('minute')}",
                        fontsize=6, color='white', alpha=1)

        ax3 = fig.add_subplot(gs[0, 2])
        ax3.imshow(screenshot2, interpolation='none')
        ax3.axis('off')

        ax4 = fig.add_subplot(gs[1, 0])
        ax4.imshow(sdo_image, interpolation='none')
        ax4.axis('off')

        ax5 = fig.add_subplot(gs[1, 1])
        ax5.imshow(stereo_image, interpolation='none')
        ax5.axis('off')

        ax6 = fig.add_subplot(gs[1, 2])
        ax6.imshow(screenshot3, interpolation='none')
        ax6.axis('off')
        # plt.tight_layout()
        plt.tight_layout(pad=0.01, h_pad=0.01, w_pad=0.01)

        if 'nodu' in self.interval:
            figname = f"nodupli_{stereo_groups.group('number')}.png"
        else:
            figname = f"Fig_{self.interval}_{stereo_groups.group('number')}.png"
        plt.savefig(os.path.join(self.paths['Save'], figname), dpi=300)
        plt.close()


class GIF_making:
    """
    To create the corresponding GIF.
    """

    def __init__(self, interval='1h', fps=5, stereo='int'):

        self.fps = fps
        self.interval = interval
        self.stereo = stereo
        self.Paths()
        self.GIF()

    def Paths(self):
        """
        Paths creator.
        """

        main_path = '../'

        self.paths = {'Main': main_path,
                      'Figures': os.path.join(main_path, f'texture_both_{self.stereo}'),
                      'GIF': os.path.join(main_path, 'GIF_both')}
        os.makedirs(self.paths['GIF'], exist_ok=True)

    def GIF(self):
        """
        Making the GIF.
        """

        images_path = sorted(Path(self.paths['Figures']).glob(f'*_{self.interval}_*.png'))

        images = [iio3.imread(image_path) for image_path in images_path]
        print(f'the type of an image is {images[0].dtype}')
        print(f'The total size of the images is {round(np.array(images).nbytes / 2**20, 1)}MB')

        iio3.imwrite(os.path.join(self.paths['GIF'], f'GIF_both_{self.stereo}_{self.interval}_fps{self.fps}.gif'),
                      images, format='GIF', fps=self.fps)


if __name__=='__main__':
    ImageFinder(interval='nodupli', ints=True)
    # GIF_making(interval='nodupli', stereo='int', fps=10)


