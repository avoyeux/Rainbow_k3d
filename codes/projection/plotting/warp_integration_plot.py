"""
To store the plotting code for the final warped integration figure.
"""

# IMPORTs
import os

# IMPORTs alias
import numpy as np

# IMPORTs sub
import matplotlib.pyplot as plt

# IMPORTs personal
from common import config
from ..format_data import ProjectionData, FitWithEnvelopes, WarpedInformation
from ..sdo_reprojection import OrthographicalProjection

# TYPE ANNOTATIONs
from typing import TypeGuard


class WarpIntegrationPlot:

    def __init__(
            self,
            plot_choices: list[str],
            integration_time: list[int] = [24],
            with_feet: bool = False,
            polynomial_order: list[int] = [4],
            with_fake_data: bool = False,
            ) -> None:
        
        # WARPED DATA
        processing = OrthographicalProjection(
            plot_choices=plot_choices,
            integration_time=integration_time,
            with_feet=with_feet,
            polynomial_order=polynomial_order,
            with_fake_data=with_fake_data,
        )
        warped_information = processing.warped_information
        
        # DATA CHECK
        if not self.information_check(warped_information): # todo take this away after corresponding code update
            raise ValueError('No warped information given')
        
        self.warped_information = warped_information

    def information_check(  # todo most likely not needed if the main code is updated
            self,
            data: list[ProjectionData] | None,
        ) -> TypeGuard[list[ProjectionData]]:
        
        return data is not None

    def warped_integration(self) -> None: 
        
        # ! will need to change the return to add info to the warped integration plot

        # todo integrate the different warped images.
        for warped_information in self.warped_information:
            
            if warped_information




    def process_warped_data(self, warped_information: WarpedInformation) -> None:
        """  # todo update docstring
        To plot the final figure of Dr. Auchere's paper given a set of warped images.

        Args:
            warped_data (np.ndarray): the warped images for which the plot needs to be made.
        """

        plot_name = warped_information.name + '_' + warped_information.integration_type + '.png'

        # PLOT
        plt.figure(figsize=(18, 5))
        plt.imshow(mean_rows, cmap='gray', origin='lower', aspect='auto')
        plt.title('Mean rows of the warped data')
        plt.xlabel('Time')
        plt.ylabel('Radial distance')
        plt.savefig(os.path.join(config.path.dir.data.temp, 'mean_rows.png'), dpi=500)
        plt.close()    
        
        # PLOT
        plt.figure(figsize=(18, 5))
        plt.imshow(median_rows, cmap='gray', origin='lower', aspect='auto')
        plt.title('Median rows of the warped data')
        plt.xlabel('Time')
        plt.ylabel('Radial distance')
        plt.savefig(os.path.join(config.path.dir.data.temp, 'median_rows.png'), dpi=500)
        plt.close()   
