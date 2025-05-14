"""
To store the plotting code for the final warped integration figure.
"""

# IMPORTs
import os

# IMPORTs alias
import numpy as np

# IMPORTs sub
from datetime import datetime
import matplotlib.pyplot as plt

# IMPORTs personal
from common import config, Decorators
from codes.projection.format_data import WarpedIntegration, AllWarpedInformation
from codes.projection.sdo_reprojection import OrthographicalProjection

# TYPE ANNOTATIONs
from typing import TypeGuard

# todo add a plot with the max arc length for each date



class WarpIntegrationPlot:
    """
    For the final warped integration plot.
    Uses the OrthographicalProjection class to get the warped data.
    """

    @Decorators.running_time
    def __init__(
            self,
            plot_choices: list[str],
            integration_time: list[int] = [24],
            with_feet: bool = False,
            polynomial_order: list[int] = [4],
            with_fake_data: bool = False,
            ) -> None:
        """
        To get the final plots of the warped integration data. The initialisation creates the
        corresponding plots without having to call an instance method.

        Args:
            plot_choices (list[str]): the choices for the plots. These choices are used in the
                OrthographicalProjection class to get the warped data.
            integration_time (list[int], optional): the integration time(s) (in hours) to use in
                the datasets. Defaults to [24].
            with_feet (bool, optional): deciding to use the datasets that contains added feet.
                Defaults to False.
            polynomial_order (list[int], optional): the polynomial order(s) to use for the 3D
                fitting. Defaults to [4].
            with_fake_data (bool, optional): deciding to use the HDF5 file that also contains fake
                data (important as the data group paths are different). Defaults to False.

        Raises:
            ValueError: if no warped information is given.
        """

        # WARPED DATA
        processing = OrthographicalProjection(
            plot_choices=plot_choices,
            integration_time=integration_time,
            with_feet=with_feet,
            polynomial_order=polynomial_order,
            with_fake_data=with_fake_data,
        )
        processing.run()
        warped_information = processing.warped_information
        
        # DATA CHECK
        if not self.information_check(warped_information): # todo take this away after corresponding code update
            raise ValueError('\033[1;31mNo warped information given.\033[0m')
        self.warped_information = warped_information

        # RUN
        self.plot_all()

    def information_check(  # todo most likely not needed if the main code is updated
            self,
            data: AllWarpedInformation | None,
        ) -> TypeGuard[AllWarpedInformation]:
        
        return data is not None
    
    @Decorators.running_time
    def plot_all(self) -> None:
        """
        To plot all the warped integration data.
        """

        for data in [
            self.warped_information.full_integration_no_duplicates,
            self.warped_information.integration,
            ]:

            # PLOT data
            if data is not None:
                for warped_integration in data.warped_integrations:
                    self.warped_integration_plot(warped_integration)

    def warped_integration_plot(self, data: WarpedIntegration) -> None:
        """
        To plot the warped integration data for a given dataset.

        Args:
            data (WarpedIntegration): the warped integration data to plot.
        """

        # todo add the arc_lengths somewhere or on the plots.
        
        # SORT data on dates
        data.sort()

        # WARPED INTEGRATION image
        image = np.stack([
            warped_information.warped_values
            for warped_information in data.warped_informations
        ], axis=0)
        angles = np.stack([
            warped_information.angles
            for warped_information in data.warped_informations
        ], axis=0)

        # NAMING
        name_end = (
            f"fit{data.fit_order}_" +
            (
                f"{data.integration_time}hours"
                if data.integration_time is not None
                else "full"
            )
        )

        # PLOT
        self.plot_data(
            name=f"warp_{data.integration_type}_" + name_end,
            data=image,
        )
        self.plot_data(
            name=f"warp_angles_" + name_end,
            data=angles,
        )

        # ARC LENGTHs plot
        name = "arc_lengths_" + name_end
        max_lengths = np.array([lengths[-1] for lengths in data.arc_lengths])
        plt.figure(figsize=(18, 5))
        plt.plot(max_lengths)
        plt.xlabel('Date')
        plt.ylabel('Max arc length')
        plt.title('Max arc length for each date')
        plt.savefig(os.path.join(config.path.dir.data.temp, name + '.png'), dpi=500)
        plt.close()
    
    def plot_data(self, name: str, data: np.ndarray) -> None:
        """
        Simple matplotlib.pyplot plotting function to plot a given ndarray.

        Args:
            name (str): the name to give to the saved plot.
            data (np.ndarray): the data to plot.
        """

        # PLOT
        plt.figure(figsize=(18, 5))
        plt.imshow(data.T, cmap='gray', origin='lower', aspect='auto', interpolation='none')
        plt.title('Mean rows of the warped data')
        plt.xlabel('Time')
        plt.ylabel('Radial distance')
        plt.colorbar(label='angles' if 'angles' in name else 'intensity')
        plt.savefig(os.path.join(config.path.dir.data.temp, name + '.png'), dpi=500)
        plt.close()



if __name__ == '__main__':

    WarpIntegrationPlot(
        integration_time=[24],
        polynomial_order=[4],
        plot_choices=[
            'no duplicates',
            'integration',
            'full integration',
            'fit',
            'sdo image', 'envelope',
            'warp',
            'all sdo images',
        ],
        with_fake_data=False,
    )
