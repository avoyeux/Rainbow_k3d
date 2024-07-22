"""Just to test some stuff.
right now I am testing StereoUtils from the Common repository
"""

# Imports
import os

import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.coordinates import SkyCoord
from sunpy.coordinates import frames
from sunpy.map import Map, GenericMap, sources

from .Common import StereoUtils, Decorators



class RainbowStereoImages:
    """To get the Rainbow even relevant stereo images.

    Returns:
        _type_: _description_
    """

    def __init__(self, date_interval: tuple[str, str] = ('2012/07/23 00:06:00', '2012/07/25 11:57:00'), 
                 roi_width: int | float = 2):

        self.date_interval = date_interval
        self.roi_width = roi_width
        
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
    
    def sunpy_map_centering(self, aia_map: GenericMap, carrington_center_lonLat: tuple[float | int] = (-165, 0), width_deg: int = 45) -> GenericMap:
        if len(carrington_center_lonLat) == 2:
            x, y = carrington_center_lonLat* u.deg
            rsun_arcsec = aia_map.rsun_obs
            distance_to_sun = aia_map.dsun.to(u.km)

            z = (np.sin(rsun_arcsec.to(u.rad).value) * distance_to_sun.value) * u.km
        else:
            x, y = carrington_center_lonLat[:2] * u.deg
            z = carrington_center_lonLat[-1] * u.km

        width = (width_deg * u.deg).to(u.arcsec)

        # Setting up the center position
        observer = aia_map.observer_coordinate
        coords = SkyCoord(x.to(u.arcsec), y.to(u.arcsec), z, frame=frames.HeliographicCarrington(observer=observer))

        # Setting up the corners
        bottom_left = SkyCoord(coords.lon - width / 2, coords.lat - width / 2, frame=coords.frame)
        top_right = SkyCoord(coords.lon + width / 2, coords.lat + width / 2, frame=coords.frame)
        return aia_map.submap(bottom_left=bottom_left, top_right=top_right)
    
    def rainbow_feet_visualisation(self, ax, sunpy_map: GenericMap, feet_pos: list[tuple[float | int]] = [(-177, 15), (-163, -16)]):

        left_foot = (feet_pos[0] * u.deg).to(u.arcsec)
        right_foot = (feet_pos[1] * u.deg).to(u.arcsec)
        width = (self.roi_width * u.deg).to(u.arcsec)
        
        rsun_arcsec = sunpy_map.rsun_obs
        distance_to_sun = sunpy_map.dsun.to(u.km).value
        z = (np.sin(rsun_arcsec.to(u.rad).value) * distance_to_sun) * u.km

        # Setting up the wcs frames
        observer = sunpy_map.observer_coordinate
        left_coords = SkyCoord(left_foot[0], left_foot[1], z, frame=frames.HeliographicCarrington(observer=observer))
        right_coords = SkyCoord(right_foot[0], right_foot[1], z, frame=frames.HeliographicCarrington(observer=observer))

        # Setting up the bottom_left corners
        left_bottom_left = SkyCoord(left_coords.lon - width / 2, left_coords.lat - width / 2, frame=left_coords.frame)
        right_bottom_left = SkyCoord(right_coords.lon - width / 2, right_coords.lat - width / 2, frame=right_coords.frame)

        #plot
        sunpy_map.draw_quadrangle(left_bottom_left, axes=ax, width=width, height=width, edgecolor='red', linewidth=1)
        sunpy_map.draw_quadrangle(right_bottom_left, axes=ax, width=width, height=width, edgecolor='red', linewidth=1)

    @Decorators.running_time
    def image_creation(self):

        filenames = self.catalogue_filtering()
        print(f'filenames length is {len(filenames)}')

        for filename in filenames:
            aia_map = Map(StereoUtils.fullpath(filename))
            # sub_aia_map = self.sunpy_map_section(aia_map)
            sub_aia_map = self.sunpy_map_centering(aia_map)

            fig = plt.figure(figsize=(5,5))
            ax = fig.add_subplot(projection=sub_aia_map)
            sub_aia_map.plot(axes=ax, clip_interval=(1, 99.99) * u.percent, title=False, interpolation='none', cmap='gray')
            ax.grid(False)
            ax.coords[0].set_axislabel('')
            ax.coords[1].set_axislabel('')
            ax.coords[0].set_ticks_visible(False)
            ax.coords[1].set_ticks_visible(False)
            ax.coords[0].set_ticklabel_visible(False)
            ax.coords[1].set_ticklabel_visible(False)
            sub_aia_map.draw_grid(axes=ax, grid_spacing=15*u.deg, system='carrington')
            self.rainbow_feet_visualisation(ax, sub_aia_map)
            plt.savefig(os.path.join(self.path, f'{os.path.splitext(filename)[0]}.png'), dpi=500)
            plt.close()
            print(f'plot done for file {filename}')


if __name__=='__main__':
    RainbowStereoImages()