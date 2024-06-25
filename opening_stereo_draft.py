"""Just to test some stuff.
right now I am testing StereoUtils from the Common repository
"""

# Imports
import os

import astropy.units as u
import matplotlib.pyplot as plt

from astropy.io import fits
from sunpy.map import Map, GenericMap, sources

from Common import StereoUtils, Decorators



class RainbowStereoImages:
    """To get the Rainbow even relevent stereo images.

    Returns:
        _type_: _description_
    """

    def __init__(self, date_interval: tuple[str, str] = ('2012/07/23 00:06:00', '2012/07/25 11:57:00')):

        self.date_interval = date_interval
        
        self.path = os.path.join('..', 'opening_stereo_tests')
        os.makedirs(self.path, exist_ok=True)
        self.image_creation()

    @Decorators.running_time
    def catalogue_filtering(self):

        catalogue_df = StereoUtils.read_catalogue()
        catalogue_df = catalogue_df[catalogue_df['dateobs'] > self.date_interval[0]]
        catalogue_df = catalogue_df[catalogue_df['dateobs'] < self.date_interval[1]]
        catalogue_df = catalogue_df[catalogue_df['polar'] == 171].reset_index(drop=True)

        return catalogue_df['filename']
    
    def sunpy_map_section(self, aia_map: GenericMap) -> GenericMap:

        dimensions = aia_map.dimensions

        x1 = dimensions.x.value / 3  #left edge
        x2 = 2 *dimensions.x.value / 3
        y1 = dimensions.y.value / 3
        y2 = 2 * dimensions.y.value / 3

        bottom_left = aia_map.pixel_to_world(x1 * u.pix, y1 * u.pix)
        top_right = aia_map.pixel_to_world(x2 * u.pix, y2 * u.pix)
        return aia_map.submap(bottom_left=bottom_left, top_right=top_right)
    
    @Decorators.running_time
    def image_creation(self):

        filenames = self.catalogue_filtering()
        print(f'filenames length is {len(filenames)} as filenames = {filenames}')

    
        for filename in filenames:
            aia_map = Map(StereoUtils.fullpath(filename))
            sub_aia_map = self.sunpy_map_section(aia_map)

            fig = plt.figure()
            ax = fig.add_subplot(projection=sub_aia_map)
            sub_aia_map.plot(axes=ax, clip_interval=(1, 99.99) * u.percent)
            sub_aia_map.draw_grid(axes=ax, grid_spacing=15*u.deg)
            plt.savefig(os.path.join(self.path, f'{os.path.splitext(filename)[0]}.png'), dpi=900)
            plt.close()
            print(f'plot done for file {filename}')


if __name__=='__main__':
    RainbowStereoImages()