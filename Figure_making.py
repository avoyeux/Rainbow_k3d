"""
Creates figures, and the corresponding GIF, for the STEREO images with the contours of the mask.
Also adds the gridlines showing the latitude and longitude.
It's quite an old code so needs a lot of improvements. Will change it when I use it again.
"""

# Imports
import os
import re
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from pathlib import Path


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
        self.Paths()
        self.Data_fullnames()

    def Paths(self):
        """
        To create the paths to the files and to where we want to save the results 
        """

        main_path = '/home/avoyeux/old_project/avoyeux'

        self.paths = {
            # Initial paths for loading data
            'Main': main_path,
            'Stereo init image': os.path.join(main_path, '..', 'backup', 'int'),
            'Stereo avg image': os.path.join(main_path, 'ratio'),
            'Stereo mask': os.path.join(main_path, 'prog', 'masque'),

            # Path to upload results
            'Plots': os.path.join(main_path, 'Plots')
        }

        # Creating the upload path
        os.makedirs(self.paths['Plots'], exist_ok=True)

    def Data_fullnames(self):
        """
        To get the path to each unique data element (e.g. image or masks) and the corresponding image
        number to then be able to correctly match them together. 
        """

        # Each data paths
        self.image_names = sorted(Path(self.paths['Stereo init image']).glob('*.png'))
        self.avg_names = sorted(Path(self.paths['Stereo avg image']).glob('*.png'))
        self.mask_names = sorted(Path(self.paths['Stereo mask']).glob('*.png'))

        # Getting the corresponding image and mask numbers 
        mask_pattern = re.compile(r'frame(\d{4})\.png')

        # Getting the corresponding image number for each mask
        self.mask_numbers = [int(mask_pattern.match(os.path.basename(mask_name)).group(1)) for mask_name in self.mask_names]

    def Main_structure(self):
        """
        From the mask numbers, it adds the class' functions together so that the plots are created.
        Basically, it is the main for loop for this class. 
        """

        for loop, number in enumerate(self.mask_numbers):
            # Uploads and initialisation
            stereo_image = mpimg.imread(self.image_names[number])
            avg_image = mpimg.imread(self.avg_names[number])
            rgb_mask = mpimg.imread(self.mask_names[loop])

            # Changing to gray_scale and getting the mask contours
            gray_mask = np.mean(rgb_mask, axis=2)
            normalised_mask = 1 - gray_mask 
            normalised_mask /= np.max(normalised_mask) 
            filters = (normalised_mask == 0)
            normalised_mask[filters] = np.nan # to be used with the 'Reds' cmap

            # Creating a bool array to get the contours 
            bool_array = ~np.isnan(normalised_mask)
            lines = ForPlotting.Contours(bool_array)

            self.Plotting_func(number, stereo_image, avg_image, lines, loop)
            print(f'Plotting for image nb {number} done.')

    def Plotting_func(self, number, stereo_image, avg_image, lines, loop):
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
    gif = GifMaker()
    gif.Main_structure()
    gif.Creating_gif()