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
from common import config, Decorators
from codes.projection.format_data import WarpedIntegration, AllWarpedInformation
from codes.projection.sdo_reprojection import OrthographicalProjection

# TYPE ANNOTATIONs
from typing import TypeGuard



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
        # todo add docstring

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
        # todo add docstring

        for data in [
            self.warped_information.full_integration_no_duplicates,
            self.warped_information.integration,
            ]:

            # PLOT data
            if data is not None:
                for warped_integration in data.warped_integrations:
                    self.warped_integration_plot(warped_integration)

    def warped_integration_plot(self, data: WarpedIntegration) -> None:
        # todo add docstring

        # todo add the arc_lengths somewhere or on the plots.
        
        # SORT dates
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

        # PLOT
        self.plot_data(
            name=(
                f"warp_{data.integration_type}integration_fit{data.fit_order}" +
                (
                    f"_{data.integration_time}hours"
                    if data.integration_time is not None
                    else "_full"
                )
            ),
            data=image,
        )
        self.plot_data(
            name=(
                f"warp_angles_fit{data.fit_order}_" +
                f"{data.integration_time}hours" if data.integration_time is not None else "_full"
            ),
            data=angles,
        )
    
    def plot_data(self, name: str, data: np.ndarray) -> None:
        # todo add docstring

        # PLOT
        plt.figure(figsize=(18, 5))
        plt.imshow(data, cmap='gray', origin='lower', aspect='auto')
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
