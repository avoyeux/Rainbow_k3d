"""
Creates figures, and the corresponding GIF, for the STEREO images with the contours of the mask.
Also adds the gridlines showing the latitude and longitude.
It's quite an old code so needs a lot of improvements. Will change it when I use it again.
"""

# Imports
import os
import re

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from PIL import Image
from glob import glob
from astropy.io import fits

from common_alf import PlotFunctions


class ForPlotting:
    """
    Has static method functions that I usually use when plotting stuff.
    """

    @staticmethod
    def Contours(mask):
        """
        To plot the contours given a mask
        Source: https://stackoverflow.com/questions/40892203/can-matplotlib-contours-match-pixel-edges
        """

        pad = np.pad(mask, [(1, 1), (1, 1)])  # zero padding
        im0 = np.abs(np.diff(pad, n=1, axis=0))[:, 1:]
        im1 = np.abs(np.diff(pad, n=1, axis=1))[1:, :]
        lines = []
        for ii, jj in np.ndindex(im0.shape):
            if im0[ii, jj] == 1:
                lines += [([ii - .5, ii - .5], [jj - .5, jj + .5])]
            if im1[ii, jj] == 1:
                lines += [([ii - .5, ii + .5], [jj - .5, jj - .5])]
        return lines
    
    @staticmethod
    def Grid_line_positions(cen, width, image_shape, axis, dx=0.075, deg_grid_width=15):
        """
        To get the positions and text for the carrington coordinates.
        """

        # Image border
        border = cen - width / 2

        # Getting the first grid line position
        first_pos = border
        for loop in range(image_shape[axis]):
            if round(first_pos, 2) % deg_grid_width == 0:
                img_index = loop
                break
            first_pos += dx

        # Getting the rest of the gridline positions
        positions = np.arange(img_index, image_shape[axis] + 0.1, deg_grid_width/dx, dtype='uint16')
        values = np.arange(round(first_pos, 2), \
                        first_pos + (len(positions) - 1) * deg_grid_width + 1e-4, deg_grid_width)

        text_val = []
        for value in values:
            if axis == 1:
                if value > 180:
                    value -= 360
                    text = f'{abs(value)}° W'
                else:
                    text = f'{value}° E'
                text_val.append(text)
            else:
                if value > 0:
                    text = f'{value}° N'
                else:
                    text = f'{abs(value)}° S'
                text_val.append(text)
        return positions, text_val
    
    @staticmethod
    def Grid_linesntext(ax, image_shape, loncen, latcen, lonwidth, latwidth, color='white', 
                        linestyle='--', linewidth=0.4, alpha=0.4, textsize=3):
        """
        Function to add the text to the grid lines.
        """

        lon_lines, lon_text = ForPlotting.Grid_line_positions(loncen, lonwidth, image_shape, 1)
        lat_lines, lat_text = ForPlotting.Grid_line_positions(latcen, latwidth, image_shape, 0)

        for pos, lat_line in enumerate(lat_lines):
            ax.axhline(lat_line, color=color, linestyle=linestyle, linewidth=linewidth, alpha=alpha)
            ax.text(3, lat_line - 2, lat_text[-(1 + pos)], color=color, alpha=alpha, size=textsize)
        for pos, lon_line in enumerate(lon_lines):
            ax.axvline(lon_line, color=color, linestyle=linestyle, linewidth=linewidth, alpha=alpha)
            ax.text(lon_line, 10, lon_text[pos], color=color, alpha=alpha, size=textsize)


class FirstFigure:
    """
    To plot the first figure. It is still in the preparation phase.
    """

    def __init__(self, loncen=195, latcen=0, lonwidth=45, latwidth=45, dlon=0.075, dlat=0.075):
        # Image stats
        self.loncen = loncen
        self.latcen = latcen
        self.lonwidth = lonwidth
        self.latwidth = latwidth
        self.dlon = dlon
        self.dlat = dlat

        # Initialisation 
        self.Patterns()
        self.Paths()
        self.SDO_image_finder()
        self.Data_fullnames()

    def Paths(self):
        """
        To create the paths to the files and to where we want to save the results 
        """

        main_path = os.path.join(os.getcwd(), '..')

        self.paths = {
            # Initial paths for loading data
            'Main': main_path,
            'Stereo init image': os.path.join(main_path, 'STEREO', 'int'),
            'Stereo avg image': os.path.join(main_path, 'STEREO', 'avg'),
            'Stereo mask': os.path.join(main_path, 'STEREO', 'masque_karine'),
            'Sdo_mask': os.path.join(main_path, 'sdo'),

            # Path to upload results
            'Plots': os.path.join(main_path, 'contours_3images_new')}
        
        # Creating the upload path
        os.makedirs(self.paths['Plots'], exist_ok=True)

    def Patterns(self):
        """
        Setting up the patterns so that I can match the MP4 images with the masks.
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
    
    def SDO_image_finder(self):
        """
        To find the SDO image given its header timestamp and a list of corresponding paths to the corresponding fits file.
        """

        with open('SDO_timestamps.txt', 'r') as files:
            strings = files.read().splitlines()
        tuple_list = [s.split(" ; ") for s in strings]
        
        timestamp_to_path = {}
        for s in tuple_list:
            path, timestamp = s
            timestamp_to_path[timestamp[:-3]] = path + '/S00000/image_lev1.fits'
        self.sdo_timestamp = timestamp_to_path

    def Data_fullnames(self):
        """
        To get the path to each unique data element (e.g. image or masks) and the corresponding image
        number to then be able to correctly match them together. 
        """

        # Each data paths
        self.image_names = sorted(glob(os.path.join(self.paths['Stereo init image'], '*.png')))
        self.avg_names = sorted(glob(os.path.join(self.paths['Stereo avg image'], '*.png')))
        self.mask_names = sorted(glob(os.path.join(self.paths['Stereo mask'], '*.png')))
        self.sdo_mask_names = sorted(glob(os.path.join(self.paths['Sdo_mask'], '*.fits.gz')))

        # Getting the corresponding image and mask numbers 
        mask_pattern = re.compile(r'frame(\d{4})\.png')

        self.mask_numbers = [int(mask_pattern.match(os.path.basename(mask_name)).group(1)) for mask_name in self.mask_names]

    def Main_structure(self):
        """
        From the mask numbers, it adds the class' functions together so that the plots are created.
        Basically, it is the main for loop for this class. 
        """

        for loop, number in enumerate(self.mask_numbers):
            # Uploads and initialisation
            stereo_name = self.image_names[number]
            stereo_group = self.stereo_pattern.match(os.path.basename(stereo_name))

            if stereo_group:
                stereo_image = mpimg.imread(stereo_name)
                avg_image = mpimg.imread(self.avg_names[number])
                rgb_mask = mpimg.imread(self.mask_names[loop])
                sdo_mask, sdo_image = self.SDO_mask(number)

                # Changing to gray_scale and getting the mask contours
                gray_mask = np.mean(rgb_mask, axis=2)
                normalised_mask = 1 - gray_mask 
                maximum = np.max(normalised_mask)
                normalised_mask /= maximum if maximum > 0 else 1 
                filters = (normalised_mask == 0)
                normalised_mask[filters] = np.nan # to be used with the 'Reds' cmap

                # Creating a bool array to get the contours 
                bool_array = ~np.isnan(normalised_mask)
                lines = PlotFunctions.Contours(bool_array)
                lines_sdo = PlotFunctions.Contours(sdo_mask)

                self.Plotting_func_new(number, stereo_image, avg_image, sdo_image, lines, lines_sdo)
                print(f'Plotting for image nb {number} done.', flush=True)

            else:
                raise ValueError(f'The string {os.path.basename(stereo_name)} has the wrong format.')

    def SDO_mask(self, number):
        """
        To treat the MP4 images so that you get the right SDO mask section.
        """

        sdo_hdul = fits.open(self.sdo_mask_names[number])
        sdo_header = sdo_hdul[0].header
        sdo_mask = np.array(sdo_hdul[0].data)

        index = round(sdo_mask.shape[0] / 3)
        sdo_mask = sdo_mask[index:index * 2 + 1, :]
        index = round(sdo_mask.shape[1] / 3)
        sdo_mask = sdo_mask[:, :index + 1]        
        sdo_mask[sdo_mask < 0.5] = 0
        sdo_mask[sdo_mask > 0] = 1
        sdo_mask = np.flip(sdo_mask, axis=0)

        for date in self.sdo_timestamp.keys():
            if date == sdo_header['DATE-OBS'][:-3]:
                print(f"sdo file found for date {date}", flush=True)
                hdul_image = fits.open(self.sdo_timestamp[date])
                image = np.array(hdul_image[1].data)
                index = round(image.shape[0] / 3)
                image = image[index: index * 2 + 1, :]
                index = round(image.shape[1] / 3)
                image = image[:, :index + 1]

                lower_cut = np.nanpercentile(image, 1)
                upper_cut = np.nanpercentile(image, 99.99)
                image[image < lower_cut] = lower_cut
                image[image > upper_cut] = upper_cut
                image = np.where(np.isnan(image), lower_cut, image)
                image = np.flip(image, axis=0)
                image = np.log(image)
                break

        sdo_hdul.close()
        hdul_image.close()

        sdo_mask = self.Resizing(sdo_mask)
        image = self.Resizing(image)
        return sdo_mask, image

    def Resizing(self, image: np.ndarray, size: tuple[int, int] = (1200, 1200)) -> np.ndarray:
        """
        Function to resize a given 2D np.ndarray.
        """

        image = Image.fromarray(image)
        image.resize(size, Image.Resampling.LANCZOS)
        return np.array(image)

    def Plotting_func(self, number, stereo_image, avg_image,  lines, loop):
        """
        Plotting the data.
        """

        fig, axs = plt.subplots(1, 3, figsize=(8, 8))

        # For the first image
        axs[0].imshow(stereo_image, interpolation='none')
        ForPlotting.Grid_linesntext(axs[0], avg_image.shape, self.loncen, self.latcen, self.lonwidth, self.latwidth)
        axs[0].axis('off')
        axs[0].set_title(f'img{int(os.path.basename(self.image_names[number]).rstrip(".png"))}')

        # The contrast image 
        axs[1].imshow(avg_image, interpolation='none')
        ForPlotting.Grid_linesntext(axs[1], avg_image.shape, self.loncen, self.latcen, self.lonwidth, self.latwidth)
        axs[1].axis('off')
        axs[1].set_title(f'avg{int(os.path.basename(self.avg_names[number]).rstrip(".png"))}')
        plt.tight_layout()

        # For the contrast with the mask lines
        axs[2].imshow(avg_image, interpolation='none')
        ForPlotting.Grid_linesntext(axs[2], avg_image.shape, self.loncen, self.latcen, self.lonwidth, self.latwidth)
        for line in lines:
            axs[2].plot(line[1], line[0], color='r', linewidth=0.5, alpha=0.3)
        axs[2].axis('off')
        axs[2].set_title(f'avg{int(os.path.basename(self.avg_names[number]).rstrip(".png"))}, '
                         f'mask{int(os.path.basename(self.mask_names[loop]).lstrip("frame").rstrip(".png"))}')
        plt.tight_layout()

        # Saving the plot
        fig_name = f'Plot_{number:04d}.png'
        plt.savefig(os.path.join(self.paths['Plots'], fig_name), bbox_inches='tight', pad_inches=0.05, dpi=800)
        plt.close()

    def Plotting_func_new(self, number, stereo_image, avg_image, sdo_image, lines, lines_sdo):
        """
        Plotting the data.
        """

        # Figure setup
        fig, axs = plt.subplots(2, 3, figsize=(7, 4.5))
        tick_params_kwargs_top = {
            'direction': 'out',
            'length': 2,
            'width': 0.7,
            'colors': 'black',
            'labelcolor': 'black',
            'axis': 'both',
            'which': 'major',
            'labelsize': 3,
            'top': False,
            'bottom': True,
            'labeltop': False,
            'labelbottom': True,
        }
        tick_params_kwargs_bottom = tick_params_kwargs_top.copy()
        tick_params_kwargs_bottom['top'] = True
        tick_params_kwargs_bottom['bottom'] = False
        tick_params_kwargs_bottom['labelbottom'] = False

        print('the shapes are:')
        print(stereo_image.shape, avg_image.shape, sdo_image.shape)

        lon_positions, lon_text = ForPlotting.Grid_line_positions(self.loncen, self.lonwidth, stereo_image.shape, 1)
        lat_positions, lat_text = ForPlotting.Grid_line_positions(self.latcen, self.latwidth, stereo_image.shape, 0)

        # Plotting the images
        axs[0, 0].imshow(stereo_image, interpolation='none')
        axs[0, 1].imshow(avg_image, interpolation='none')
        axs[0, 2].imshow(sdo_image, interpolation='none')
        axs[1, 0].imshow(stereo_image, interpolation='none')
        axs[1, 1].imshow(avg_image, interpolation='none')
        axs[1, 2].imshow(sdo_image, interpolation='none')

        # Labels and stuff for [...]
        ## [...] the first image
        axs[0, 0] = self.Subplot_params(axs[0, 0], lon_positions, lat_positions, lon_text, lat_text, tick_params_kwargs_top)
        axs[0, 0].yaxis.tick_right()

        ## [...] the contrast image 
        axs[0, 1] = self.Subplot_params(axs[0, 1], lon_positions, lat_positions, lon_text, lat_text, tick_params_kwargs_top)

        ## [...] the sdo image
        axs[0, 2].axis('off')


        ## [...] the first image with the mask lines
        for line in lines:
            axs[1, 0].plot(line[1], line[0], color='r', linewidth=0.5, alpha=0.2)
        axs[1, 0] = self.Subplot_params(axs[1, 0], lon_positions, lat_positions, lon_text, lat_text, tick_params_kwargs_bottom)
        axs[1, 0].yaxis.tick_right()

        ## [...] the contrast with the mask lines
        for line in lines:
            axs[1, 1].plot(line[1], line[0], color='r', linewidth=0.5, alpha=0.2)
        axs[1, 1] = self.Subplot_params(axs[1, 1], lon_positions, lat_positions, lon_text, lat_text, tick_params_kwargs_bottom)

        ## [...] the sdo image with the mask lines
        axs[1, 2].axis('off')
        for line in lines_sdo:
            axs[1, 2].plot(line[1], line[0], color='r', linewidth=0.5, alpha=0.2)

        # Taking out the black spine around each plot
        a = 10
        for ax in axs.flat:
            for spine in ax.spines.values():
                spine.set_visible(False)
            for t in ax.yaxis.get_ticklabels():
                a -= 1
                t.set_zorder(a)
                t.set_bbox({
                    'facecolor': 'white',
                    'alpha': 0.4,
                    'edgecolor': 'none',
                    'pad': 1,
                })
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0.13, hspace=0.08)
        # Saving the plot
        fig_name = f'final_plot_{number:04d}.png'
        plt.savefig(os.path.join(self.paths['Plots'], fig_name), bbox_inches='tight', pad_inches=0.0, dpi=800)
        plt.close()

    def Subplot_params(self, ax, lon_pos, lat_pos, lon_text, lat_text, tick_params):
        """
        Function that set ups the params and text for the labels/ticks of a subplot.
        """

        ax.set_xticks(lon_pos)
        ax.set_yticks(lat_pos)
        ax.set_xticklabels(lon_text)
        ax.set_yticklabels(lat_text[::-1])
        ax.tick_params(**tick_params)

        ax.set_facecolor('none')
        return ax



class GifMaker(FirstFigure):
    """
    Making the Gif using the created plots
    """

    def __init__(self, fps=0.8):
        super().__init__()
        self.fps = fps
        self.Updating_paths()

    def Updating_paths(self):
        """
        Updating the paths created in the main class.
        """

        self.paths['GIF'] = os.path.join(self.paths['Main'], 'GIF')
        os.makedirs(self.paths['GIF'], exist_ok=True)

    def Creating_gif(self):
        """
        To create the gif using the plots from the main class
        """

        import imageio

        img_paths = [os.path.join(self.paths['Plots'], f'Plot_{number:04d}.png') for number in self.mask_numbers]

        with imageio.get_writer(os.path.join(self.paths['GIF'], 'the_gif.gif'), mode='I', duration=self.fps*1000) as writer:
            for img_path in img_paths:
                image = imageio.imread(img_path)
                writer.append_data(image)
        print('Gif is done')


if __name__ == '__main__':
    # gif = GifMaker()
    # gif.Main_structure()
    # gif.Creating_gif()

    plot = FirstFigure()
    plot.Main_structure()