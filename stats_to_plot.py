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
                      'Stats': main_path,
                      'Saving': os.path.join(main_path, 'Stats_plots')}
        os.makedirs(self.paths['Saving'], exist_ok=True)
        
    def Main(self):
        """
        Main structure of the class.
        """

        # Getting the data
        df = pd.read_csv(os.path.join(self.paths['Stats'], 'k3d_stats.csv'))
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
        plt.plot(self.dates, cubes_1, label=name[0], color='blue')
        plt.plot(self.dates, cubes_2, label=name[1], color='orange')

        plt.title(f'Time evolution for {name[0][:-3]}')
        plt.xlabel(f'Time in seconds')
        plt.ylabel(f'Voxel count')
        plt.legend()
        plt.savefig(os.path.join(self.paths['Saving'],
                                 f'Line_time_evo_{name[0][:-3]}.png'), dpi=100)

        # Normalised time evolution
        norm_1 = cubes_1 / cubes_1.max()
        norm_2 = cubes_2 / cubes_2.max()
        plt.figure(figsize=(16, 8))
        plt.plot(self.dates, norm_1, label=name[0], color='blue')
        plt.plot(self.dates, norm_2, label=name[1], color='orange')

        plt.title(f'Normalised time evolution for {name[0][:-3]}')
        plt.xlabel(f'Time in seconds')
        plt.ylabel(f'Voxel count')
        plt.legend()
        plt.savefig(os.path.join(self.paths['Saving'], 
                                 f'Line_norm_evo_{name[0][:-3]}.png'), dpi=100)


if __name__=='__main__':
    Plotting(interval=1200)





