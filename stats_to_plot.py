"""
Code to use the .csv stats file to create corresponding plots 
"""

# Imports
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typeguard import typechecked


class Plotting:
    """
    PLOTTING!!!!!!!!!!!!!!!!!!!!!!!!!
    """

    @typechecked
    def __init__(self, interval: int):
        
        self.interval = interval
        self.Paths()
        self.Main()
    
    def Paths(self):
        """
        To create the saving path
        """

        main_path = '../'
        self.paths = {'Main': main_path,
                      'Stats': os.path.join(main_path, 'STATS'),
                      'Saving': os.path.join(main_path, 'STATS', 'STATS_plots')}
        os.makedirs(self.paths['Saving'], exist_ok=True)
        
    def Main(self):
        """
        Main structure of the class.
        """

        # Getting the data
        df = pd.read_csv(os.path.join(self.paths['Stats'], 'k3d_volumes.csv'))
        self.dates = df['dates (seconds)']

        all_data_alf = df['all_data_alf']
        all_data_kar = df['all_data_kar']
        no_duplicates_stereo_alf = df['no_duplicates_stereo_alf']
        no_duplicates_stereo_kar = df['no_duplicates_stereo_kar']
        no_duplicates_sdo_alf = df['no_duplicates_sdo_alf']
        no_duplicates_sdo_kar = df['no_duplicates_sdo_kar']
        no_duplicates_alf = df['no_duplicates_alf']
        no_duplicates_kar = df['no_duplicates_kar']
        interval_all_data_alf = df[f'interval_{self.interval}_all_data_alf']
        interval_all_data_kar = df[f'interval_{self.interval}_all_data_kar']
        interval_no_duplicates_alf = df[f'interval_{self.interval}_no_duplicates_alf']
        interval_no_duplicates_kar = df[f'interval_{self.interval}_no_duplicates_kar']

        # Plotting
        self.Plots(all_data_alf, all_data_kar, ('all_data_alf', 'all_data_kar'))
        self.Plots(no_duplicates_alf, no_duplicates_kar, ('no_duplicates_alf', 'no_duplicates_kar'))
        self.Plots(no_duplicates_stereo_alf, no_duplicates_stereo_kar, 
                   ('no_duplicates_stereo_alf', 'no_duplicates_stereo_kar'))
        self.Plots(no_duplicates_sdo_alf, no_duplicates_sdo_kar, 
                   ('no_duplicates_sdo_alf', 'no_duplicates_sdo_kar'))
        self.Plots(interval_all_data_alf, interval_all_data_kar,
                   (f'interval_{self.interval}_all_data_alf', f'interval_{self.interval}_all_data_kar'))
        self.Plots(interval_no_duplicates_alf, interval_no_duplicates_kar,
                   (f'interval_{self.interval}_no_duplicates_alf', f'interval_{self.interval}_no_duplicates_kar'))
        
    def Plots(self, cubes_1, cubes_2, name):
        """
        To plot the values.
        """

        # Time evolution
        plt.figure(figsize=(16, 8))
        plt.plot(self.dates/3600, cubes_1, label=name[0], color='blue')
        plt.plot(self.dates/3600, cubes_2, label=name[1], color='orange')

        plt.title(f'Time evolution for {name[0][:-3]}')
        plt.xlabel(f'Time in hours')
        plt.ylabel(f'Voxel count')
        plt.legend()
        plt.savefig(os.path.join(self.paths['Saving'],
                                 f'Line_time_evo_{name[0][:-3]}.png'), dpi=100)

        # Normalised time evolution
        norm_1 = cubes_1 / cubes_1.max()
        norm_2 = cubes_2 / cubes_2.max()
        plt.figure(figsize=(16, 8))
        plt.plot(self.dates/3600, norm_1, label=name[0], color='blue')
        plt.plot(self.dates/3600, norm_2, label=name[1], color='orange')

        plt.title(f'Normalised time evolution for {name[0][:-3]}')
        plt.xlabel(f'Time in hours')
        plt.ylabel(f'Voxel count')
        plt.legend()
        plt.savefig(os.path.join(self.paths['Saving'], 
                                 f'Line_norm_evo_{name[0][:-3]}.png'), dpi=100)


class Plotting_v2:
    """
    For the voxel volume and mask surface as a function of the date.
    """

    def __init__(self):

        # Constants
        self.dx = 0.003 * 6.96e5 * 0.7068
        
        # Functions
        self.Paths()
        self.Download()
        self.Plotting()

    def Paths(self):
        """
        For the filepaths.
        """

        main_path = '../'
        self.paths = {'Main': main_path,
                      'STATS': os.path.join(main_path, 'STATS'),
                      'Plots': os.path.join(main_path, 'STATS', 'STATS_plots')}
        os.makedirs(self.paths['Plots'], exist_ok=True)

    def Download(self):
        """
        Getting the data from the corresponding csv files.
        """

        df_volumes = pd.read_csv(os.path.join(self.paths['STATS'], 'k3d_volumes.csv'))
        df_area = pd.read_csv(os.path.join(self.paths['STATS'], 'k3d_mask_area.csv'))


        self.dates = df_volumes['dates (seconds)'] / 3600
        self.volumes_min_old = df_volumes['no_duplicates_kar_old'] * self.dx**3
        self.volumes_max = df_volumes['all_data_kar'] * self.dx**3
        self.volumes_min_new = df_volumes['no_duplicates_kar_new'] * self.dx**3
        self.area_sdo = df_area['SDO mask']

    def Preprocessing(self, array):
        """
        To preprocess the data.
        For now it only changes the 0 values to np.nan.
        """

        array[array==0] = np.nan
        return array

    def Plotting(self):
        """
        Creating the plot.
        """

        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax1.set_xlabel('Time (hours since 23-Jul-2012 00:00)')
        ax1.set_xlim(-0.5, 60.5)

        ax1kwargs = {'color': 'blue', 'marker': 'None', 'alpha': 0.5}
        # ax1.plot(self.dates, self.Preprocessing(self.volumes_min), linestyle='--', label='Minimum total volume', **ax1kwargs)
        # ax1.plot(self.dates, self.Preprocessing(self.volumes_max), linestyle='-', label='Maximum total volume', **ax1kwargs)
        # ax1.set_ylim(0, 1.8e5)
        ax1.set_ylabel(r'Volume ($Mm^3$)', color=ax1kwargs['color'])
        ax1.tick_params(axis='y', labelcolor=ax1kwargs['color'])
        # Volume fill
        plt.fill_between(self.dates, self.Preprocessing(self.volumes_max) * 1e-9, self.Preprocessing(self.volumes_min_old) * 1e-9, color='blue', alpha=0.3, label='Volume range old')
        plt.fill_between(self.dates, self.Preprocessing(self.volumes_max) * 1e-9, self.Preprocessing(self.volumes_min_new) * 1e-9, color='red', alpha=0.3, label='Volume range new')

        ax2 = ax1.twinx()
        ax2kwargs = {'color': 'red', 'marker': 'None', 'alpha': 0.8, 'linewidth': 1}
        ax2.plot(self.dates, self.Preprocessing(self.area_sdo), dashes=(3, 3, 7, 1), label='Total SDO mask fov', **ax2kwargs) 
        ax2.set_ylim(0, 1.3e4)
        ax2.tick_params(axis='y', labelcolor=ax2kwargs['color'])
        ax2.set_ylabel(r'Total fov ($arsec^2$)', color=ax2kwargs['color'])

        # Setting up the shared legend
        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        handles = handles1 + handles2
        labels = labels1 + labels2
        fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(0.097, 0.97))

        plt.tight_layout()
        plt.savefig('new_test_normalplot.png', dpi=200)
        plt.close()


if __name__=='__main__':
    Plotting_v2()





