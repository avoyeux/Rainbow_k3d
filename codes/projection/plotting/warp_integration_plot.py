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
import matplotlib.dates as mdates

# IMPORTs personal
from common import config, Decorators
from codes.projection.format_data import WarpedIntegration, AllWarpedInformation
from codes.projection.sdo_reprojection import OrthographicalProjection

# TYPE ANNOTATIONs
from typing import TypeGuard

# todo need a constant time step for the x axis of the warp integration plot



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
        if not self.information_check(warped_information):
            raise ValueError('\033[1;31mNo warped information given.\033[0m')
        self.warped_information = warped_information

        # PATHs
        self.paths = self.paths_setup()

        # RUN
        self.plot_all()

    def information_check(  # todo most likely not needed if the main code is updated
            self,
            data: AllWarpedInformation | None,
        ) -> TypeGuard[AllWarpedInformation]:
        
        return data is not None
    
    def paths_setup(self) -> dict[str, str]:
        """
        To setup and format the paths needed for the plotting.

        Returns:
            dict[str, str]: the path(s) to the directory(s).
        """

        # PATHs formatting
        paths = {
            'warped': os.path.join(config.path.dir.data.result.projection, 'warped_integration'),
        }

        # PATHs creation
        os.makedirs(paths['warped'], exist_ok=True)
        return paths

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
        
        # SORT data on dates
        data.sort()

        # WARPED INTEGRATION image
        image = np.stack([
            warped_information.warped_values
            for warped_information in data.warped_informations
        ], axis=0)
        angles = np.stack([
            np.rad2deg(warped_information.angles)
            for warped_information in data.warped_informations
        ], axis=0)

        # NAMING saved plots suffix
        name_end = (
            f"fit{data.fit_order}_" +
            (
                f"{data.integration_time}hours"
                if data.integration_time is not None
                else "full"
            )
        )

        # DATEs setup
        matplotlib_dates = self.date_treatment(data.dates)

        # PLOT
        self.plot_data(
            name=f"warp_{data.integration_type}_" + name_end,
            data=image,
            arc_length=data.arc_lengths[0],
            dates=matplotlib_dates,
        )
        self.plot_data(
            name=f"warp_angles_" + name_end,
            data=angles,
            arc_length=data.arc_lengths[0],
            dates=matplotlib_dates,
        )

        # ARC LENGTHs plot
        name = "arc_lengths_" + name_end
        max_lengths = np.array([lengths[-1] for lengths in data.arc_lengths])
        plt.figure(figsize=(18, 5))
        plt.plot(max_lengths)
        plt.xlabel('Date')
        plt.ylabel('Max arc length')
        plt.title('Max arc length for each date')
        plt.savefig(os.path.join(self.paths['warped'], name + '.png'), dpi=500)
        plt.close()
    
    def date_treatment(self, dates: list[str]) -> np.ndarray:
        """
        To treat the dates so that they can be used in one of the axes of the plot.

        Args:
            dates (list[str]): the dates to treat.

        Returns:
            np.ndarray: the corresponding matplotlib dates.
        """

        datetimes = [datetime.strptime(date,'%Y-%m-%dT%H-%M-%S') for date in dates]
        matplotlib_dates = mdates.date2num(datetimes)
        return matplotlib_dates

    def plot_data(
            self,
            name: str,
            data: np.ndarray,
            arc_length: np.ndarray,  # ! only works when arc_length is the same for all dates.
            dates: np.ndarray,
        ) -> None:
        """
        To plot the warped integration data for a given dataset.

        Args:
            name (str): the filename of the PNG figure to save.
            data (np.ndarray): the data to plot.
            arc_length (np.ndarray): the arc length to use for the y axis.
            dates (np.ndarray): the dates to use for the x axis.
        """

        print(f"arc_length shape is {arc_length.shape}")
        print(f"data shape is {data.shape}")
        print(f"dates shape is {dates.shape}", flush=True)

        # AXEs setup
        X, Y = np.meshgrid(arc_length, dates)

        # PLOT init
        fig, ax = plt.subplots(figsize=(18, 5))

        # IMAGE
        im = ax.pcolormesh(Y.T, X.T, data.T, cmap='gray', shading='nearest')

        # TIME axis
        ax.xaxis_date()
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=4))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%dT%H-%M-%S'))
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

        # PLOT
        ax.set_title('Mean rows of the warped data')
        ax.set_xlabel('Time [hours]')
        ax.set_ylabel('Distance [Mm]')
        plt.colorbar(im, label='angles [degrees]' if 'angles' in name else 'intensity')
        plt.tight_layout()
        plt.savefig(os.path.join(self.paths['warped'], name + '.png'), dpi=500)
        plt.close()

        print(f"SAVED - {name}.png")



if __name__ == '__main__':

    WarpIntegrationPlot(
        integration_time=[],
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
